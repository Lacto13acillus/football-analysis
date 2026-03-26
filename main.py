# main.py
# Pipeline utama: baca video -> deteksi -> tracking -> analisis dribbling -> render -> simpan

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers                          import Tracker
from team_assigner.player_identifier   import PlayerBallAssigner
from trackers.player_ball_assigner     import PlayerBallAssigner
from trackers.dribble_detector         import DribbleDetector
from draw_dribble import (
    draw_cone_zones_on_frame,
    draw_ball_trajectory_on_frame,
    draw_dribble_status,
    draw_dribble_stats_panel,
    draw_result_flash,
    draw_entry_exit_zones,      # opsional, untuk debug
    draw_attempt_summary,       # opsional, untuk frame terakhir
)

from utils.bbox_utils import get_center_of_bbox_bottom


# ============================================================
# KONFIGURASI UTAMA
# ============================================================

CONFIG = {
    "input_video" : "input_videos/dribbling_count.mp4",
    "output_video": "output_videos/dribbling_count.avi",
    "model_path"  : "/home/dika/football-analysis/models/best.pt",
    "stub_path"   : "stubs/tracks_cache_dribble.pkl",
    "use_stub"    : False,
    "fps"         : 30,

    # ============================================================
    # JERSEY MAPPING (SEED AWAL)
    # ============================================================
    "jersey_mapping": {
        1: "#3",
        2: "#19",
        3: "Unknown",
        4: "#3",
        5: "#19",
        6: "Unknown",
        7: "Unknown"
    },

    # Parameter re-identification
    "reassign_distance_threshold": 150.0,
    "lost_timeout_frames"        : 60,

    # ============================================================
    # KONFIGURASI DRIBBLE
    # ============================================================
    # Radius cone
    "cone_radius_multiplier"  : 1.5,
    "default_cone_radius"     : 40.0,
    "min_cone_radius"         : 20.0,
    "max_cone_radius"         : 80.0,

    # Dribble attempt detection
    "entry_exit_zone_radius"  : 150.0,
    "min_attempt_frames"      : 15,
    "cooldown_frames"         : 30,
    "min_cones_passed"        : 2,

    # Arah urutan cone: 'auto', 'top_to_bottom', 'left_to_right', dll
    "cone_order_direction"    : "auto",

    # ============================================================
    # KONFIGURASI POSSESSION
    # ============================================================
    "max_possession_distance" : 130,

    # ============================================================
    # VISUALISASI
    # ============================================================
    "show_cone_zones"    : True,
    "show_stats_panel"   : True,
    "debug_trajectory"   : True,
    "show_dribble_status": True,
}


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Football Dribbling Analytics - Cone Touch Detection"
    )
    parser.add_argument("--input", type=str, help="Path video input")
    parser.add_argument("--output", type=str, help="Path video output")
    parser.add_argument("--stub", action="store_true", help="Gunakan cache tracking")
    parser.add_argument("--debug", action="store_true", help="Aktifkan debug trajectory")
    return parser.parse_args()


# ============================================================
# STATISTIK PROGRESSIVE (REALTIME)
# ============================================================

def compute_progressive_stats(
    detected_attempts: List[Dict],
    up_to_frame      : int
) -> Dict:
    """Hitung statistik kumulatif dari attempt yang sudah ditampilkan."""
    attempts_so_far = [
        a for a in detected_attempts
        if a['frame_end'] <= up_to_frame
    ]

    total = len(attempts_so_far)
    sukses = [a for a in attempts_so_far if a['success']]
    gagal = [a for a in attempts_so_far if not a['success']]

    per_player: Dict[str, Dict] = {}
    cone_hit_count: Dict[int, int] = {}

    for a in attempts_so_far:
        jersey = a['jersey']
        if jersey not in per_player:
            per_player[jersey] = {
                'total': 0, 'success': 0, 'failed': 0,
                'accuracy_pct': 0.0, 'avg_duration': 0.0,
                'total_cones_hit': 0, '_dur_sum': 0.0,
            }
        per_player[jersey]['total'] += 1
        per_player[jersey]['_dur_sum'] += a['duration_seconds']
        per_player[jersey]['total_cones_hit'] += len(a['touched_cones'])
        if a['success']:
            per_player[jersey]['success'] += 1
        else:
            per_player[jersey]['failed'] += 1

        for tc in a['touched_cones']:
            cone_hit_count[tc] = cone_hit_count.get(tc, 0) + 1

    for jersey, stat in per_player.items():
        stat['accuracy_pct'] = round(
            stat['success'] / stat['total'] * 100, 1
        ) if stat['total'] > 0 else 0.0
        stat['avg_duration'] = round(
            stat['_dur_sum'] / stat['total'], 2
        ) if stat['total'] > 0 else 0.0
        del stat['_dur_sum']

    return {
        'total_attempts'     : total,
        'successful_attempts': len(sukses),
        'failed_attempts'    : len(gagal),
        'accuracy_pct'       : round(len(sukses) / total * 100, 1) if total > 0 else 0.0,
        'avg_duration'       : round(float(np.mean(
                                   [a['duration_seconds'] for a in attempts_so_far])), 2)
                               if attempts_so_far else 0.0,
        'cone_hit_frequency' : cone_hit_count,
        'per_player'         : per_player,
    }


# ============================================================
# CETAK DETAIL KE CONSOLE
# ============================================================

def print_dribble_details(attempts: List[Dict], stats: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("   STATISTIK HASIL ANALISIS DRIBBLING")
    print(sep)
    print(f"  Total Attempt        : {stats['total_attempts']}")
    print(f"  Sukses               : {stats['successful_attempts']}")
    print(f"  Gagal                : {stats['failed_attempts']}")
    print(f"  Akurasi              : {stats['accuracy_pct']}%")
    print(f"  Rata2 Durasi         : {stats['avg_duration']}s")
    print("-" * 70)

    if stats.get('cone_hit_frequency'):
        print("  Cone Paling Sering Disentuh:")
        for cid, cnt in sorted(stats['cone_hit_frequency'].items(), key=lambda x: -x[1]):
            print(f"    Cone {cid}: {cnt}x disentuh")
        print("-" * 70)

    print("  Statistik Per Pemain:")
    for jersey, pstat in stats.get('per_player', {}).items():
        filled = int(pstat['accuracy_pct'] / 10)
        bar = "█" * filled + "░" * (10 - filled)
        print(f"    {jersey:<10}: {pstat['success']:>2}/{pstat['total']:>2} "
              f"| {bar} {pstat['accuracy_pct']:>5.1f}% "
              f"| avg={pstat['avg_duration']:.1f}s "
              f"| cone_hit={pstat.get('total_cones_hit', 0)}")
    print(sep)

    if not attempts:
        print("  Tidak ada dribble attempt terdeteksi.\n")
        return

    print("\n  Detail Semua Dribble Attempts:")
    print(f"  {'No':<4} {'Pemain':<10} {'Frame':<14} "
          f"{'Durasi':>8} {'Cone Hit':>10} {'Status':<10}")
    print("  " + "-" * 60)
    for i, a in enumerate(attempts):
        status = "SUKSES" if a['success'] else "GAGAL"
        touched = len(a['touched_cones'])
        total_c = a['total_cones']
        print(f"  {i+1:<4} "
              f"{a['jersey']:<10} "
              f"{a['frame_start']:>4}-{a['frame_end']:<7} "
              f"{a['duration_seconds']:>6.1f}s "
              f"{touched:>3}/{total_c:<5} "
              f"{status:<10}")

        if a['touched_cones']:
            for tc in a['touched_cones']:
                d = a['cone_details'][tc]['min_distance']
                r = a['cone_details'][tc]['radius']
                print(f"         └─ Cone {tc}: min_dist={d:.1f}px (radius={r:.1f}px)")
    print()


# ============================================================
# RENDERING SEMUA FRAME
# ============================================================

def render_frames(
    frames            : List[np.ndarray],
    tracks            : Dict,
    ball_possessions  : List[int],
    detected_attempts : List[Dict],
    player_identifier : PlayerBallAssigner,
    dribble_detector  : DribbleDetector,
    config            : Dict
) -> List[np.ndarray]:
    output_frames = []
    total_frames = len(frames)

    stabilized_cones = dribble_detector.get_all_cones() or {}
    cone_radii = dribble_detector.get_cone_radii()

    rolling_trajectory: List = []

    # Pre-compute: frame ranges per attempt untuk highlight
    attempt_frame_map: Dict[int, Dict] = {}
    for a in detected_attempts:
        for f in range(a['frame_start'], a['frame_end'] + 1):
            attempt_frame_map[f] = a

    print(f"\n[RENDER] Mulai merender {total_frames} frames...")

    for frame_num, frame in enumerate(frames):
        if frame_num % 100 == 0:
            pct = frame_num / total_frames * 100
            print(f"[RENDER] Progress: {frame_num}/{total_frames} ({pct:.1f}%)...")

        annotated = frame.copy()

        # Cek apakah frame ini termasuk dalam dribble attempt
        active_attempt = attempt_frame_map.get(frame_num)
        touched_cones_now = []
        if active_attempt:
            touched_cones_now = active_attempt.get('touched_cones', [])

        # 1. Zona cone
        if config.get("show_cone_zones", True) and stabilized_cones:
            annotated = draw_cone_zones_on_frame(
                annotated,
                stabilized_cones = stabilized_cones,
                cone_radii       = cone_radii,
                touched_cones    = touched_cones_now if active_attempt else [],
                active_attempt   = active_attempt is not None,
            )

        # 2. Bounding box pemain + label jersey
        for player_id, player_data in tracks["players"][frame_num].items():
            bbox = player_data.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            has_ball = (
                frame_num < len(ball_possessions) and
                ball_possessions[frame_num] == player_id
            )
            box_color = (0, 200, 50) if has_ball else (200, 80, 50)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            jersey = player_identifier.get_jersey_number_for_player(player_id)
            label = f"#{jersey}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 8, y1), box_color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if has_ball:
                foot_x = (x1 + x2) // 2
                foot_y = y2
                cv2.circle(annotated, (foot_x, foot_y), 8, (0, 230, 255), -1)
                cv2.circle(annotated, (foot_x, foot_y), 11, (255, 255, 255), 2)

        # 3. Bola
        ball_data = tracks["ball"][frame_num].get(1)
        if ball_data:
            bx1, by1, bx2, by2 = map(int, ball_data["bbox"])
            bcx = (bx1 + bx2) // 2
            bcy = (by1 + by2) // 2
            brad = max(8, (bx2 - bx1) // 2)
            cv2.circle(annotated, (bcx, bcy), brad, (0, 230, 255), 2)
            cv2.circle(annotated, (bcx, bcy), brad - 2, (0, 170, 200), 1)

            if config.get("debug_trajectory", False):
                rolling_trajectory.append((bcx, bcy))
                if len(rolling_trajectory) > 60:
                    rolling_trajectory.pop(0)

        # 4. Trajectory bola (debug)
        if config.get("debug_trajectory", False) and len(rolling_trajectory) > 1:
            annotated = draw_ball_trajectory_on_frame(
                annotated,
                trajectory=rolling_trajectory,
                max_points=40
            )

        # 5. Dribble status indicator
        if config.get("show_dribble_status", True) and active_attempt:
            annotated = draw_dribble_status(
                annotated,
                is_active   = True,
                jersey      = active_attempt['jersey'],
                touched_cnt = len(touched_cones_now),
                total_cones = active_attempt['total_cones'],
            )

        # 6. Panel statistik realtime
        if config.get("show_stats_panel", True):
            realtime_stats = compute_progressive_stats(detected_attempts, frame_num)
            annotated = draw_dribble_stats_panel(
                annotated,
                stats       = realtime_stats,
                position    = (20, 20),
                panel_width = 295,
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

    print("\n" + "=" * 70)
    print("   FOOTBALL DRIBBLING ANALYTICS v1.0")
    print("   Cone Touch Detection via Radius Logic")
    print("=" * 70)
    print(f"  Input        : {CONFIG['input_video']}")
    print(f"  Output       : {CONFIG['output_video']}")
    print(f"  Model        : {CONFIG['model_path']}")
    print(f"  Gunakan cache: {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Cone radius  : multiplier={CONFIG['cone_radius_multiplier']}, "
          f"default={CONFIG['default_cone_radius']}px")
    print(f"  Possession d : {CONFIG['max_possession_distance']}px")
    print("=" * 70)

    # ==========================================================
    # TAHAP 1: Baca Video
    # ==========================================================
    print("\n[MAIN] TAHAP 1: Membaca video input...")

    if not os.path.exists(CONFIG["input_video"]):
        print(f"[MAIN] ERROR: File tidak ditemukan: {CONFIG['input_video']}")
        return

    frames = Tracker.read_video(CONFIG["input_video"])
    if not frames:
        print("[MAIN] ERROR: Video tidak bisa dibaca atau kosong!")
        return

    cap = cv2.VideoCapture(CONFIG["input_video"])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        fps = CONFIG["fps"]
    print(f"[MAIN] FPS: {fps}, Total frames: {len(frames)}")

    # ==========================================================
    # TAHAP 2: Deteksi & Tracking Objek
    # ==========================================================
    print("\n[MAIN] TAHAP 2: Deteksi & Tracking objek...")

    tracker = Tracker(model_path=CONFIG["model_path"])
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=CONFIG["use_stub"],
        stub_path=CONFIG["stub_path"]
    )

    # ==========================================================
    # TAHAP 3: Identifikasi Jersey Pemain
    # ==========================================================
    print("\n[MAIN] TAHAP 3: Mapping jersey pemain...")

    player_identifier = PlayerBallAssigner(
        track_id_to_jersey          = CONFIG["jersey_mapping"],
        reassign_distance_threshold = CONFIG["reassign_distance_threshold"],
        lost_timeout_frames         = CONFIG["lost_timeout_frames"]
    )

    for frame_num in range(len(tracks['players'])):
        player_identifier.update_frame(
            frame_num     = frame_num,
            player_tracks = tracks['players'][frame_num],
            debug         = (frame_num % 100 == 0)
        )
    player_identifier.print_mappings()

    # ==========================================================
    # TAHAP 4: Inisialisasi Dribble Detector & Cone
    # ==========================================================
    print("\n[MAIN] TAHAP 4: Inisialisasi Dribble Detector & Cone...")

    dribble_detector = DribbleDetector(fps=fps)
    dribble_detector.set_jersey_map(player_identifier)

    # Transfer config ke detector
    dribble_detector.cone_radius_multiplier = CONFIG["cone_radius_multiplier"]
    dribble_detector.default_cone_radius    = CONFIG["default_cone_radius"]
    dribble_detector.min_cone_radius        = CONFIG["min_cone_radius"]
    dribble_detector.max_cone_radius        = CONFIG["max_cone_radius"]
    dribble_detector.entry_exit_zone_radius = CONFIG["entry_exit_zone_radius"]
    dribble_detector.min_attempt_frames     = CONFIG["min_attempt_frames"]
    dribble_detector.cooldown_frames        = CONFIG["cooldown_frames"]
    dribble_detector.min_cones_passed       = CONFIG["min_cones_passed"]
    dribble_detector.cone_order_direction   = CONFIG["cone_order_direction"]

    cone_ok = dribble_detector.initialize_cones(
        tracks, cone_key='cones', sample_frames=30, debug=True
    )

    if not cone_ok:
        print("[MAIN] WARNING: Cone tidak teridentifikasi!")
        return

    # ==========================================================
    # TAHAP 5: Assignment Possession Bola Per Frame
    # ==========================================================
    print("\n[MAIN] TAHAP 5: Menentukan possession bola per frame...")

    assigner = PlayerBallAssigner(
        max_possession_distance=CONFIG["max_possession_distance"]
    )
    assigner.set_player_identifier(player_identifier)
    ball_possessions = assigner.assign_ball_to_players_bulk(tracks)

    # ==========================================================
    # TAHAP 6: Deteksi Dribble Attempts
    # ==========================================================
    print("\n[MAIN] TAHAP 6: Deteksi dribble attempts...")

    detected_attempts = dribble_detector.detect_dribble_attempts(
        tracks,
        ball_possessions,
        player_identifier=player_identifier,
        debug=True
    )

    # ==========================================================
    # TAHAP 7: Statistik
    # ==========================================================
    print("\n[MAIN] TAHAP 7: Menghitung statistik...")

    stats = dribble_detector.get_dribble_statistics(detected_attempts)
    print_dribble_details(detected_attempts, stats)

    # ==========================================================
    # TAHAP 8: Render Video Output
    # ==========================================================
    print("\n[MAIN] TAHAP 8: Merender video output...")

    output_frames = render_frames(
        frames            = frames,
        tracks            = tracks,
        ball_possessions  = ball_possessions,
        detected_attempts = detected_attempts,
        player_identifier = player_identifier,
        dribble_detector  = dribble_detector,
        config            = CONFIG,
    )

    # ==========================================================
    # TAHAP 9: Simpan Video Output
    # ==========================================================
    print(f"\n[MAIN] TAHAP 9: Menyimpan video ke: {CONFIG['output_video']}...")

    output_dir = os.path.dirname(CONFIG["output_video"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    Tracker.save_video(output_frames, CONFIG["output_video"], fps=fps)

    # ==========================================================
    # SELESAI
    # ==========================================================
    print("\n" + "=" * 70)
    print("   PIPELINE SELESAI!")
    print("=" * 70)
    print(f"  Video output      : {CONFIG['output_video']}")
    print(f"  Total frames      : {len(output_frames)}")
    print(f"  Durasi output     : {len(output_frames) / fps:.1f} detik")
    print(f"  Total dribble     : {stats['total_attempts']}")
    print(f"  Akurasi           : {stats['accuracy_pct']}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
