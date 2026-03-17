# main.py
# Pipeline Penalty Kick: On Target + Gol/Saved Detection
# v6 — Keeper-Centric + Reaction Window Save Detection

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers                          import Tracker
from trackers.penalty_detector         import PenaltyDetector
from team_assigner.player_identifier   import PlayerIdentifier
from trackers.player_ball_assigner     import PlayerBallAssigner
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
    "output_video": "output_videos/penalty_kick2.avi",
    "model_path"  : "/home/dika/football-analysis/models/best.pt",
    "stub_path"   : "stubs/tracks_cache.pkl",
    "use_stub"    : False,
    "fps"         : 30,

    # Player Identifier
    "reassign_distance_threshold": 150.0,
    "lost_timeout_frames"        : 60,
    "color_vote_window"          : 15,
    "lock_after_votes"           : 8,

    # Manual jersey mapping
    "manual_jersey_mapping": {
        4:  "Merah",
        5:  "Merah",
        10: "Merah",
        14: "Merah",
        16: "Merah",
        17: "Merah",
        18: "Abu-Abu",
    },

    # Manual kick mapping (handle ID switching track 6 & 9)
    "manual_kick_mapping": {
        86:   "Merah",
        266:  "Abu-Abu",
        476:  "Merah",
        668:  "Abu-Abu",
        894:  "Merah",
        1030: "Abu-Abu",
    },

    # Possession
    "max_possession_distance": 200,

    # Penalty detection — kick
    "kick_velocity_threshold"  : 15.0,
    "pre_kick_search"          : 50,
    "max_kicker_distance"      : 700,
    "on_target_check_window"   : 60,
    "cooldown_frames"          : 120,
    "gawang_shrink_ratio"      : 0.05,
    "on_target_min_frames"     : 1,

    # Penalty detection — save (v6: keeper-centric + reaction window)
    "save_check_window"            : 60,
    "save_reaction_time"           : 0.25,   # detik — window setelah bola masuk gawang
    "penetration_goal_threshold"   : 0.35,
    "save_score_threshold"         : 3,
    "save_keeper_max_dist"         : 120,
    "save_ball_max_vel"            : 5.0,
    "overlap_min_frames"           : 2,
    "overlap_bbox_expand"          : 20,
    "bounce_back_frames_thr"       : 8,
    "bounce_back_margin"           : 30,
    "bounce_keeper_max_dist"       : 200,

    # Visualisasi
    "show_gawang"           : True,
    "show_keeper"           : True,
    "show_kick_result"      : True,
    "show_stats_panel"      : True,
    "kick_display_duration" : 50,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Penalty Kick: On Target + Gol/Saved Detection v6"
    )
    parser.add_argument("--input",  type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--stub",   action="store_true")
    return parser.parse_args()


# ============================================================
# PROGRESSIVE STATS
# ============================================================

def compute_progressive_stats(
    detected_penalties: List[Dict],
    up_to_frame       : int
) -> Dict:
    kicks_so_far = [
        p for p in detected_penalties
        if p['frame_kick'] <= up_to_frame
    ]

    total      = len(kicks_so_far)
    on_list    = [p for p in kicks_so_far if p['is_on_target']]
    off_list   = [p for p in kicks_so_far if not p['is_on_target']]
    goal_list  = [p for p in kicks_so_far if p['is_goal']]
    saved_list = [p for p in kicks_so_far if p['is_saved']]

    per_player: Dict[str, Dict] = {}
    for p in kicks_so_far:
        jersey = p['kicker_jersey']
        if jersey not in per_player:
            per_player[jersey] = {
                'total': 0, 'on_target': 0, 'off_target': 0,
                'goals': 0, 'saved': 0,
                'on_target_pct': 0.0, 'goal_pct': 0.0
            }
        per_player[jersey]['total'] += 1
        if p['is_on_target']:
            per_player[jersey]['on_target'] += 1
        else:
            per_player[jersey]['off_target'] += 1
        if p['is_goal']:
            per_player[jersey]['goals'] += 1
        if p['is_saved']:
            per_player[jersey]['saved'] += 1

    for jersey, stat in per_player.items():
        t = stat['total']
        stat['on_target_pct'] = round(
            stat['on_target'] / t * 100, 1) if t > 0 else 0.0
        stat['goal_pct'] = round(
            stat['goals'] / t * 100, 1) if t > 0 else 0.0

    return {
        'total_kicks'     : total,
        'total_on_target' : len(on_list),
        'total_off_target': len(off_list),
        'total_goals'     : len(goal_list),
        'total_saved'     : len(saved_list),
        'on_target_pct'   : round(
            len(on_list) / total * 100, 1) if total > 0 else 0.0,
        'goal_pct'        : round(
            len(goal_list) / total * 100, 1) if total > 0 else 0.0,
        'per_player'      : per_player
    }


def print_shoot_details(penalties: List[Dict], stats: Dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("   STATISTIK PENALTY KICK")
    print(sep)
    print(f"  Total Tendangan  : {stats['total_kicks']}")
    print(f"  ON TARGET        : {stats['total_on_target']}")
    print(f"  OFF TARGET       : {stats['total_off_target']}")
    print(f"  GOL              : {stats['total_goals']}")
    print(f"  SAVED            : {stats['total_saved']}")
    print(f"  On Target %      : {stats['on_target_pct']}%")
    print(f"  Gol %            : {stats['goal_pct']}%")
    print("-" * 72)
    print("  Per Pemain:")
    for jersey, pstat in stats.get('per_player', {}).items():
        g_filled = int(pstat['goal_pct'] / 10)
        g_bar    = "█" * g_filled + "░" * (10 - g_filled)
        print(f"    {jersey:<12}: {pstat['goals']:>1} gol / "
              f"{pstat['saved']:>1} saved / "
              f"{pstat['off_target']:>1} off  "
              f"| {g_bar} gol {pstat['goal_pct']:>5.1f}% "
              f"| on target {pstat['on_target']}/{pstat['total']}")
    print(sep)

    if penalties:
        print("\n  Detail Tendangan:")
        print(f"  {'No':<4} {'Penendang':<12} {'Trk':>4} {'Frame':>6} "
              f"{'Vel':>6} {'Target':<10} {'Hasil':<8} Keterangan")
        print("  " + "-" * 90)
        for i, p in enumerate(penalties):
            target = "ON" if p['is_on_target'] else "OFF"
            if p['is_goal']:
                result = "GOL!"
            elif p['is_saved']:
                result = "SAVED"
            else:
                result = "MISS"
            vel = p.get('ball_velocity', 0.0)
            reason = p.get('result_reason', '-')
            print(f"  {i+1:<4} {p['kicker_jersey']:<12} "
                  f"{p['kicker_id']:>4} "
                  f"{p['frame_kick']:>6} "
                  f"{vel:>6.1f} "
                  f"{target:<10} "
                  f"{result:<8} "
                  f"{reason}")
    print()


# ============================================================
# RENDERING
# ============================================================

def get_jersey_for_render(
    player_id, frame_num, player_identifier,
    manual_kick_mapping, detected_penalties
):
    jersey = player_identifier.get_jersey_number_for_player(player_id)
    if jersey in ("Merah", "Abu-Abu"):
        for p in detected_penalties:
            if p['kicker_id'] == player_id:
                kick_f = p['frame_kick']
                if abs(frame_num - kick_f) < 80:
                    return p['kicker_jersey']
    return jersey


def render_frames(
    frames, tracks, ball_possessions, detected_penalties,
    player_identifier, config
) -> List[np.ndarray]:
    output_frames = []
    total_frames  = len(frames)

    manual_kick_mapping = config.get("manual_kick_mapping", {})

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
                is_goal_now = (
                    frame_num in kick_display_map and
                    kick_display_map[frame_num]['is_goal']
                )
                annotated = draw_gawang_on_frame(
                    annotated, gawang_data['bbox'],
                    is_goal=is_goal_now
                )

        # 2. Gambar keeper
        if config.get("show_keeper", True):
            keeper_data = tracks["keeper"][frame_num].get(1)
            if keeper_data and 'bbox' in keeper_data:
                kx1, ky1, kx2, ky2 = map(int, keeper_data['bbox'])

                keeper_color = (0, 165, 255)
                label = "KEEPER"
                if frame_num in kick_display_map:
                    event = kick_display_map[frame_num]
                    if event['is_saved']:
                        keeper_color = (0, 255, 150)
                        label = "KEEPER SAVE!"

                cv2.rectangle(annotated, (kx1, ky1), (kx2, ky2),
                              keeper_color, 2)
                (lw, lh), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(annotated,
                              (kx1, ky1 - lh - 8), (kx1 + lw + 8, ky1),
                              keeper_color, -1)
                cv2.putText(annotated, label, (kx1 + 4, ky1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 3. Bounding box pemain
        for player_id, player_data in tracks["players"][frame_num].items():
            bbox = player_data.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            has_ball = (
                frame_num < len(ball_possessions) and
                ball_possessions[frame_num] == player_id
            )

            jersey = get_jersey_for_render(
                player_id, frame_num, player_identifier,
                manual_kick_mapping, detected_penalties
            )

            if jersey == "Merah":
                box_color = (0, 0, 220) if not has_ball else (0, 255, 0)
            elif jersey == "Abu-Abu":
                box_color = (160, 160, 160) if not has_ball else (0, 255, 0)
            else:
                box_color = (200, 80, 50) if not has_ball else (0, 200, 50)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            label = f"{jersey} [{player_id}]"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(annotated,
                          (x1, y1 - lh - 8), (x1 + lw + 8, y1),
                          box_color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

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

        # 5. Hasil tendangan
        if config.get("show_kick_result", True) and frame_num in kick_display_map:
            event = kick_display_map[frame_num]
            annotated = draw_kick_result(
                annotated,
                kicker_pos=event.get('kicker_pos'),
                is_on_target=event['is_on_target'],
                is_goal=event['is_goal'],
                is_saved=event['is_saved'],
                kicker_jersey=event['kicker_jersey']
            )

        # 6. Panel statistik realtime
        if config.get("show_stats_panel", True):
            realtime_stats = compute_progressive_stats(
                detected_penalties, frame_num
            )
            annotated = draw_penalty_stats_panel(
                annotated, stats=realtime_stats,
                position=(20, 20), panel_width=340
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

    print("\n" + "=" * 62)
    print("   PENALTY KICK: ON TARGET + GOL/SAVED DETECTION v6")
    print("   Player Merah vs Player Abu-Abu")
    print("=" * 62)
    print(f"  Input        : {CONFIG['input_video']}")
    print(f"  Output       : {CONFIG['output_video']}")
    print(f"  Model        : {CONFIG['model_path']}")
    print(f"  Gunakan cache: {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Manual kick  : {len(CONFIG.get('manual_kick_mapping', {}))} entries")
    print(f"  Manual goal  : {len(CONFIG.get('manual_goal_mapping', {}))} entries")
    print(f"  Save method  : KEEPER-CENTRIC + REACTION WINDOW "
          f"(reaction={CONFIG.get('save_reaction_time', 0.25)}s, "
          f"score_thr={CONFIG.get('save_score_threshold', 3)})")
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
        frames,
        read_from_stub=CONFIG["use_stub"],
        stub_path=CONFIG["stub_path"]
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

    # TAHAP 3: Identifikasi pemain
    print("\n[MAIN] TAHAP 3: Identifikasi pemain...")
    manual_map = CONFIG.get("manual_jersey_mapping", None)

    player_identifier = PlayerIdentifier(
        track_id_to_jersey          = manual_map,
        reassign_distance_threshold = CONFIG.get("reassign_distance_threshold", 150.0),
        lost_timeout_frames         = CONFIG.get("lost_timeout_frames", 60),
        color_vote_window           = CONFIG.get("color_vote_window", 15),
        lock_after_votes            = CONFIG.get("lock_after_votes", 8),
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

    # TAHAP 4: Possession bola
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

    # TAHAP 5: Deteksi Penalty
    print("\n[MAIN] TAHAP 5: Deteksi Penalty (On Target + Gol/Saved)...")
    penalty_detector = PenaltyDetector(fps=fps)
    penalty_detector.set_jersey_map(player_identifier)

    # --- Set kick detection parameters ---
    penalty_detector.kick_velocity_threshold = CONFIG.get("kick_velocity_threshold", 15.0)
    penalty_detector.pre_kick_search         = CONFIG.get("pre_kick_search", 50)
    penalty_detector.max_kicker_distance     = CONFIG.get("max_kicker_distance", 700)
    penalty_detector.on_target_check_window  = CONFIG.get("on_target_check_window", 60)
    penalty_detector.cooldown_frames         = CONFIG.get("cooldown_frames", 120)
    penalty_detector.gawang_shrink_ratio     = CONFIG.get("gawang_shrink_ratio", 0.05)
    penalty_detector.on_target_min_frames    = CONFIG.get("on_target_min_frames", 1)

    # --- Set save detection parameters (v6) ---
    penalty_detector.save_check_window          = CONFIG.get("save_check_window", 60)
    penalty_detector.save_reaction_time         = CONFIG.get("save_reaction_time", 0.25)
    penalty_detector.penetration_goal_threshold = CONFIG.get("penetration_goal_threshold", 0.35)
    penalty_detector.save_score_threshold       = CONFIG.get("save_score_threshold", 3)
    penalty_detector.save_keeper_max_dist       = CONFIG.get("save_keeper_max_dist", 120)
    penalty_detector.save_ball_max_vel          = CONFIG.get("save_ball_max_vel", 5.0)
    penalty_detector.overlap_min_frames         = CONFIG.get("overlap_min_frames", 2)
    penalty_detector.overlap_bbox_expand        = CONFIG.get("overlap_bbox_expand", 20)
    penalty_detector.bounce_back_frames_thr     = CONFIG.get("bounce_back_frames_thr", 8)
    penalty_detector.bounce_back_margin         = CONFIG.get("bounce_back_margin", 30)
    penalty_detector.bounce_keeper_max_dist     = CONFIG.get("bounce_keeper_max_dist", 200)

    # --- Manual mappings ---
    manual_kick_map = CONFIG.get("manual_kick_mapping", {})
    if manual_kick_map:
        penalty_detector.set_manual_kick_mapping(manual_kick_map)

    manual_goal_map = CONFIG.get("manual_goal_mapping", {})
    if manual_goal_map:
        penalty_detector.set_manual_goal_mapping(manual_goal_map)

    detected_penalties = penalty_detector.detect_penalties(
        tracks, ball_possessions,
        frames=frames,
        player_identifier=player_identifier,
        debug=True
    )

    # TAHAP 6: Statistik
    print("\n[MAIN] TAHAP 6: Menghitung statistik...")
    stats = penalty_detector.get_penalty_statistics(detected_penalties)
    print_shoot_details(detected_penalties, stats)

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
    print(f"  Video output      : {CONFIG['output_video']}")
    print(f"  Total frames      : {len(output_frames)}")
    print(f"  Durasi output     : {len(output_frames) / fps:.1f} detik")
    print(f"  Total tendangan   : {stats['total_kicks']}")
    print(f"  ON TARGET         : {stats['total_on_target']}")
    print(f"  OFF TARGET        : {stats['total_off_target']}")
    print(f"  GOL               : {stats['total_goals']}")
    print(f"  SAVED             : {stats['total_saved']}")
    print(f"  On Target %       : {stats['on_target_pct']}%")
    print(f"  Gol %             : {stats['goal_pct']}%")
    print("-" * 62)
    for jersey, pstat in stats.get('per_player', {}).items():
        print(f"    {jersey}: {pstat['goals']} gol / "
              f"{pstat['saved']} saved / "
              f"{pstat['off_target']} off target "
              f"(gol {pstat['goal_pct']}%)")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
