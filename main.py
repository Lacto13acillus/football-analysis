# main.py
# Pipeline Penalty Kick Detection
#
# - Mendeteksi tendangan penalty dari 2 pemain (Merah & Abu-Abu)
# - Counting gol vs miss menggunakan deteksi gawang dari YOLO
# - Identifikasi pemain berdasarkan warna baju
# - Deteksi kick menggunakan velocity spike bola

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers                          import Tracker
from trackers.penalty_detector import PenaltyDetector
from team_assigner.player_identifier   import PlayerIdentifier
from trackers.player_ball_assigner     import PlayerBallAssigner
from trackers.penalty_detector         import PenaltyDetector
from draw_gate import (
    draw_gawang_on_frame,
    draw_kick_result,
    draw_penalty_stats_panel,
    draw_rounded_rect,
)
from utils.bbox_utils import get_center_of_bbox_bottom


# ============================================================
# KONFIGURASI
# ============================================================

CONFIG = {
    "input_video" : "input_videos/penalty_kick.mp4",
    "output_video": "output_videos/penalty_kick_result.avi",
    "model_path"  : "/home/dika/football-analysis/models/best.pt",
    "stub_path"   : "stubs/tracks_cache.pkl",
    "use_stub"    : False,
    "fps"         : 30,

    # Tidak perlu jersey_mapping manual - identifikasi via warna baju
    "reassign_distance_threshold": 150.0,
    "lost_timeout_frames"        : 60,

    # Possession (masih digunakan untuk visualisasi siapa pegang bola)
    "max_possession_distance": 200,

    # Penalty detection — NILAI YANG BENAR
    "kick_velocity_threshold": 15.0,
    "pre_kick_search"        : 50,      # WAS 10 — 50 frames = 0.8s@60fps
    "max_kicker_distance"    : 700,     # WAS 200 — portrait video, player besar
    "goal_check_window"      : 60,
    "cooldown_frames"        : 120,     # WAS 60 — menghasilkan tepat 6 kicks
    "gawang_shrink_ratio"    : 0.10,

    # Visualisasi
    "show_gawang"           : True,
    "show_keeper"           : True,
    "show_kick_result"      : True,
    "show_stats_panel"      : True,
    "debug_trajectory"      : False,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Penalty Kick Detection & Counting"
    )
    parser.add_argument("--input",  type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--stub",   action="store_true")
    parser.add_argument("--debug",  action="store_true")
    return parser.parse_args()


# ============================================================
# PROGRESSIVE STATS
# ============================================================

def compute_progressive_penalty_stats(
    detected_penalties: List[Dict],
    up_to_frame       : int
) -> Dict:
    kicks_so_far = [
        p for p in detected_penalties
        if p['frame_kick'] <= up_to_frame
    ]

    total = len(kicks_so_far)
    gol   = [p for p in kicks_so_far if p['is_goal']]
    miss  = [p for p in kicks_so_far if not p['is_goal']]

    per_player: Dict[str, Dict] = {}
    for p in kicks_so_far:
        jersey = p['kicker_jersey']
        if jersey not in per_player:
            per_player[jersey] = {
                'total': 0, 'goals': 0, 'misses': 0, 'goal_pct': 0.0
            }
        per_player[jersey]['total'] += 1
        if p['is_goal']:
            per_player[jersey]['goals'] += 1
        else:
            per_player[jersey]['misses'] += 1

    for jersey, stat in per_player.items():
        stat['goal_pct'] = round(
            stat['goals'] / stat['total'] * 100, 1
        ) if stat['total'] > 0 else 0.0

    return {
        'total_kicks'  : total,
        'total_goals'  : len(gol),
        'total_misses' : len(miss),
        'goal_pct'     : round(len(gol) / total * 100, 1) if total > 0 else 0.0,
        'per_player'   : per_player
    }


def print_penalty_details(penalties: List[Dict], stats: Dict) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("   STATISTIK PENALTY KICK")
    print(sep)
    print(f"  Total Tendangan  : {stats['total_kicks']}")
    print(f"  GOL              : {stats['total_goals']}")
    print(f"  MISS             : {stats['total_misses']}")
    print(f"  Konversi         : {stats['goal_pct']}%")
    print("-" * 62)
    print("  Per Pemain:")
    for jersey, pstat in stats.get('per_player', {}).items():
        filled = int(pstat['goal_pct'] / 10)
        bar    = "█" * filled + "░" * (10 - filled)
        print(f"    {jersey:<12}: {pstat['goals']:>2}/{pstat['total']:>2} gol "
              f"| {bar} {pstat['goal_pct']:>5.1f}%")
    print(sep)

    if penalties:
        print("\n  Detail Tendangan:")
        print(f"  {'No':<4} {'Penendang':<12} {'Frame':>8} "
              f"{'Velocity':>10} {'Hasil':<8} Keterangan")
        print("  " + "-" * 68)
        for i, p in enumerate(penalties):
            status = "GOL!" if p['is_goal'] else "MISS"
            vel = p.get('ball_velocity', 0.0)
            reason = p.get('reason', '-')
            print(f"  {i+1:<4} {p['kicker_jersey']:<12} "
                  f"{p['frame_kick']:>8} "
                  f"{vel:>9.1f} "
                  f"{status:<8} "
                  f"{reason}")
    print()


# ============================================================
# RENDERING
# ============================================================

def render_frames(
    frames, tracks, ball_possessions, detected_penalties,
    player_identifier, config
) -> List[np.ndarray]:
    output_frames = []
    total_frames  = len(frames)

    # Map frame -> penalty event untuk display
    kick_display_map: Dict[int, Dict] = {}
    kick_display_duration = config.get("kick_display_duration", 50)

    for p in detected_penalties:
        for f in range(p['frame_kick'],
                       min(p['frame_kick'] + kick_display_duration, total_frames)):
            kick_display_map[f] = p

    print(f"\n[RENDER] Mulai merender {total_frames} frames...")

    for frame_num in range(total_frames):
        if frame_num % 100 == 0:
            pct = frame_num / total_frames * 100
            print(f"[RENDER] Progress: {frame_num}/{total_frames} ({pct:.1f}%)...")

        frame = frames[frame_num]
        annotated = frame.copy()

        # 1. Gambar gawang
        if config.get("show_gawang", True):
            gawang_data = tracks["gawang"][frame_num].get(1)
            if gawang_data and 'bbox' in gawang_data:
                is_goal_now = (frame_num in kick_display_map and
                               kick_display_map[frame_num]['is_goal'])
                annotated = draw_gawang_on_frame(
                    annotated, gawang_data['bbox'], is_goal=is_goal_now
                )

        # 2. Gambar keeper
        if config.get("show_keeper", True):
            keeper_data = tracks["keeper"][frame_num].get(1)
            if keeper_data and 'bbox' in keeper_data:
                kx1, ky1, kx2, ky2 = map(int, keeper_data['bbox'])
                cv2.rectangle(annotated, (kx1, ky1), (kx2, ky2), (0, 165, 255), 2)
                label = "KEEPER"
                (lw, lh), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(annotated,
                              (kx1, ky1 - lh - 8), (kx1 + lw + 8, ky1),
                              (0, 165, 255), -1)
                cv2.putText(annotated, label, (kx1 + 4, ky1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 3. Bounding box pemain + label warna baju
        for player_id, player_data in tracks["players"][frame_num].items():
            bbox = player_data.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            has_ball = (
                frame_num < len(ball_possessions) and
                ball_possessions[frame_num] == player_id
            )

            jersey = player_identifier.get_jersey_number_for_player(player_id)

            # Warna bounding box berdasarkan identitas pemain
            if jersey == "Merah":
                box_color = (0, 0, 220) if not has_ball else (0, 255, 0)
            elif jersey == "Abu-Abu":
                box_color = (160, 160, 160) if not has_ball else (0, 255, 0)
            else:
                box_color = (200, 80, 50) if not has_ball else (0, 200, 50)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            label = jersey
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(annotated,
                          (x1, y1 - lh - 8), (x1 + lw + 8, y1),
                          box_color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if has_ball:
                foot_x = (x1 + x2) // 2
                foot_y = y2
                cv2.circle(annotated, (foot_x, foot_y), 8,  (0, 230, 255), -1)
                cv2.circle(annotated, (foot_x, foot_y), 11, (255, 255, 255), 2)

        # 4. Bola
        ball_data = tracks["ball"][frame_num].get(1)
        if ball_data:
            bx1, by1, bx2, by2 = map(int, ball_data["bbox"])
            bcx  = (bx1 + bx2) // 2
            bcy  = (by1 + by2) // 2
            brad = max(8, (bx2 - bx1) // 2)
            cv2.circle(annotated, (bcx, bcy), brad,     (0, 230, 255), 2)
            cv2.circle(annotated, (bcx, bcy), brad - 2, (0, 170, 200), 1)

        # 5. Hasil tendangan (GOL/MISS overlay)
        if config.get("show_kick_result", True) and frame_num in kick_display_map:
            event = kick_display_map[frame_num]
            annotated = draw_kick_result(
                annotated,
                kicker_pos=event.get('kicker_pos'),
                is_goal=event['is_goal'],
                kicker_jersey=event['kicker_jersey']
            )

        # 6. Panel statistik realtime
        if config.get("show_stats_panel", True):
            realtime_stats = compute_progressive_penalty_stats(
                detected_penalties, frame_num
            )
            annotated = draw_penalty_stats_panel(
                annotated, stats=realtime_stats,
                position=(20, 20), panel_width=320
            )

        # 7. Label frame
        h_frame, w_frame = annotated.shape[:2]
        cv2.putText(annotated, f"Frame: {frame_num}",
                    (w_frame - 140, h_frame - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1)

        output_frames.append(annotated)

    print(f"[RENDER] Selesai: {len(output_frames)}/{total_frames} frames dirender.")
    return output_frames


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    args = parse_args()

    if args.input:
        CONFIG["input_video"] = args.input
    if args.output:
        CONFIG["output_video"] = args.output
    if args.stub:
        CONFIG["use_stub"] = True
    if args.debug:
        CONFIG["debug_trajectory"] = True

    print("\n" + "=" * 62)
    print("   PENALTY KICK DETECTION & COUNTING")
    print("   Player Merah vs Player Abu-Abu")
    print("=" * 62)
    print(f"  Input        : {CONFIG['input_video']}")
    print(f"  Output       : {CONFIG['output_video']}")
    print(f"  Model        : {CONFIG['model_path']}")
    print(f"  Gunakan cache: {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Vel threshold: {CONFIG['velocity_threshold']} px/frame")
    print(f"  Possession d : {CONFIG['max_possession_distance']}px")
    print("=" * 62)

    # TAHAP 1: Baca video
    print("\n[MAIN] TAHAP 1: Membaca video input...")
    if not os.path.exists(CONFIG["input_video"]):
        print(f"[MAIN] ERROR: File tidak ditemukan: {CONFIG['input_video']}")
        return

    frames = Tracker.read_video(CONFIG["input_video"])
    if not frames:
        print("[MAIN] ERROR: Video kosong!")
        return

    cap = cv2.VideoCapture(CONFIG["input_video"])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        fps = CONFIG["fps"]
    print(f"[MAIN] FPS: {fps}, Total frames: {len(frames)}, "
          f"Durasi: {len(frames)/fps:.1f}s")

    # TAHAP 2: Deteksi & Tracking
    print("\n[MAIN] TAHAP 2: Deteksi & Tracking objek...")
    if not os.path.exists(CONFIG["model_path"]):
        print(f"[MAIN] ERROR: Model tidak ditemukan: {CONFIG['model_path']}")
        return

    tracker = Tracker(model_path=CONFIG["model_path"])
    tracks  = tracker.get_object_tracks(
        frames, read_from_stub=CONFIG["use_stub"], stub_path=CONFIG["stub_path"]
    )

    n_player_frames = len(tracks.get('players', []))
    n_ball_frames   = len(tracks.get('ball',    []))
    print(f"[MAIN] Tracks 'players': {n_player_frames} frames")
    print(f"[MAIN] Tracks 'ball'   : {n_ball_frames} frames")
    print(f"[MAIN] Tracks 'gawang' : {len(tracks.get('gawang', []))} frames")
    print(f"[MAIN] Tracks 'keeper' : {len(tracks.get('keeper', []))} frames")

    if n_player_frames == 0 or n_ball_frames == 0:
        print("[MAIN] ERROR: Tracking gagal!")
        return

    # TAHAP 3: Identifikasi pemain berdasarkan warna baju
    print("\n[MAIN] TAHAP 3: Identifikasi pemain berdasarkan warna baju...")
    player_identifier = PlayerIdentifier(
        track_id_to_jersey          = None,
        reassign_distance_threshold = CONFIG.get("reassign_distance_threshold", 150.0),
        lost_timeout_frames         = CONFIG.get("lost_timeout_frames", 60)
    )

    print("[MAIN] Mengidentifikasi warna baju per frame...")
    for frame_num in range(n_player_frames):
        player_identifier.update_frame(
            frame_num=frame_num,
            player_tracks=tracks['players'][frame_num],
            debug=False
        )
        player_identifier.identify_players_by_color(
            frame=frames[frame_num],
            frame_num=frame_num,
            player_tracks=tracks['players'][frame_num],
            debug=(frame_num % 200 == 0)
        )

    player_identifier.print_mappings()

    # TAHAP 4: Tentukan possession bola (untuk visualisasi)
    print("\n[MAIN] TAHAP 4: Menentukan possession bola per frame...")
    assigner = PlayerBallAssigner(
        max_possession_distance=CONFIG.get("max_possession_distance", 200.0)
    )
    assigner.set_player_identifier(player_identifier)
    ball_possessions = assigner.assign_ball_to_players_bulk(tracks)

    possession_count: Dict[str, int] = {}
    for pid in ball_possessions:
        jersey = ("Tidak ada" if pid == -1
                  else player_identifier.get_jersey_number_for_player(pid))
        possession_count[jersey] = possession_count.get(jersey, 0) + 1
    print(f"\n[MAIN] Distribusi possession:")
    for jersey, count in sorted(possession_count.items(), key=lambda x: -x[1]):
        pct = count / len(ball_possessions) * 100
        print(f"[MAIN]   {jersey:<12}: {count:>5} frames ({pct:.1f}%)")

    # TAHAP 5: Deteksi penalty kick
    print("\n[MAIN] TAHAP 5: Deteksi penalty kick (velocity-based)...")
    penalty_detector = PenaltyDetector(fps=fps)
    penalty_detector.set_jersey_map(player_identifier)
    penalty_detector.kick_velocity_threshold = CONFIG.get("kick_velocity_threshold", 15.0)
    penalty_detector.pre_kick_search         = CONFIG.get("pre_kick_search", 50)
    penalty_detector.max_kicker_distance     = CONFIG.get("max_kicker_distance", 700)
    penalty_detector.goal_check_window       = CONFIG.get("goal_check_window", 60)
    penalty_detector.cooldown_frames         = CONFIG.get("cooldown_frames", 120)
    penalty_detector.gawang_shrink_ratio     = CONFIG.get("gawang_shrink_ratio", 0.10)

    detected_penalties = penalty_detector.detect_penalties(
        tracks, ball_possessions,
        frames=frames,                          # BARU: kirim frames untuk re-detect warna
        player_identifier=player_identifier, debug=True
    )

    # TAHAP 6: Statistik
    print("\n[MAIN] TAHAP 6: Menghitung statistik...")
    stats = penalty_detector.get_penalty_statistics(detected_penalties)
    print_penalty_details(detected_penalties, stats)

    # TAHAP 7: Render video
    print("\n[MAIN] TAHAP 7: Merender video output...")
    output_frames = render_frames(
        frames, tracks, ball_possessions, detected_penalties,
        player_identifier, CONFIG
    )

    # TAHAP 8: Simpan video
    print(f"\n[MAIN] TAHAP 8: Menyimpan video ke: {CONFIG['output_video']}...")
    output_dir = os.path.dirname(CONFIG["output_video"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    Tracker.save_video(output_frames, CONFIG["output_video"], fps=fps)

    print("\n" + "=" * 62)
    print("   PIPELINE SELESAI!")
    print("=" * 62)
    print(f"  Video output     : {CONFIG['output_video']}")
    print(f"  Total frames     : {len(output_frames)}")
    print(f"  Durasi output    : {len(output_frames) / fps:.1f} detik")
    print(f"  Total tendangan  : {stats['total_kicks']}")
    print(f"  GOL              : {stats['total_goals']}")
    print(f"  MISS             : {stats['total_misses']}")
    print(f"  Konversi         : {stats['goal_pct']}%")
    for jersey, pstat in stats.get('per_player', {}).items():
        print(f"    {jersey}: {pstat['goals']}/{pstat['total']} gol ({pstat['goal_pct']}%)")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
