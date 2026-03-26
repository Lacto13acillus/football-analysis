# main.py
# Pipeline utama: baca video -> deteksi -> tracking -> analisis dribbling -> render -> simpan
# Versi simplified: 1 pemain, tanpa jersey mapping

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers                          import Tracker
from trackers.player_ball_assigner     import PlayerBallAssigner
from trackers.dribble_detector         import DribbleDetector
from draw_dribble import (
    draw_cone_zones_on_frame,
    draw_ball_trajectory_on_frame,
    draw_dribble_status,
    draw_dribble_stats_panel,
    draw_result_flash,
    draw_entry_exit_zones,
    draw_attempt_summary,
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
    # KONFIGURASI DRIBBLE
    # ============================================================
    "cone_radius_multiplier"  : 1.5,
    "default_cone_radius"     : 40.0,
    "min_cone_radius"         : 20.0,
    "max_cone_radius"         : 80.0,

    "entry_exit_zone_radius"  : 150.0,
    "min_attempt_frames"      : 15,
    "cooldown_frames"         : 30,

    "cone_order_direction"    : "auto",

    # ============================================================
    # KONFIGURASI POSSESSION
    # ============================================================
    "max_possession_distance" : 130,

    # ============================================================
    # KONFIGURASI DETEKSI SENTUHAN (BARU)
    # ============================================================
    # Minimum consecutive frames bola harus di dalam radius cone
    # agar dianggap menyentuh cone. Nilai 1 = tanpa filter.
    "min_consecutive_touch_frames": 2,

    # Gunakan edge bola (bukan hanya center) untuk hitung jarak
    "use_ball_edge_distance": True,

    # Jumlah sub-step interpolasi antar frame
    "interpolation_substeps": 3,

    # ============================================================
    # VISUALISASI
    # ============================================================
    "show_cone_zones"    : True,
    "show_stats_panel"   : True,
    "debug_trajectory"   : True,
    "show_dribble_status": True,
    "result_flash_frames": 45,   # durasi flash SUKSES/GAGAL dalam frame
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
    """Hitung statistik kumulatif dari attempt yang sudah selesai."""
    attempts_so_far = [
        a for a in detected_attempts
        if a['frame_end'] <= up_to_frame
    ]

    total = len(attempts_so_far)
    sukses = [a for a in attempts_so_far if a['success']]
    gagal = [a for a in attempts_so_far if not a['success']]

    cone_hit_count: Dict[int, int] = {}
    for a in attempts_so_far:
        for tc in a['touched_cones']:
            cone_hit_count[tc] = cone_hit_count.get(tc, 0) + 1

    return {
        'total_attempts'     : total,
        'successful_attempts': len(sukses),
        'failed_attempts'    : len(gagal),
        'accuracy_pct'       : round(len(sukses) / total * 100, 1) if total > 0 else 0.0,
        'avg_duration'       : round(float(np.mean(
                                   [a['duration_seconds'] for a in attempts_so_far])), 2)
                               if attempts_so_far else 0.0,
        'cone_hit_frequency' : cone_hit_count,
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

    if not attempts:
        print("  Tidak ada dribble attempt terdeteksi.\n")
        return

    print("\n  Detail Semua Dribble Attempts:")
    print(f"  {'No':<4} {'Player':<10} {'Frame':<14} "
          f"{'Durasi':>8} {'Cone Hit':>10} {'Status':<10}")
    print("  " + "-" * 60)
    for i, a in enumerate(attempts):
        status = "SUKSES" if a['success'] else "GAGAL"
        touched = len(a['touched_cones'])
        total_c = a['total_cones']
        print(f"  {i+1:<4} "
              f"{'P' + str(a['player_id']):<10} "
              f"{a['frame_start']:>4}-{a['frame_end']:<7} "
              f"{a['duration_seconds']:>6.1f}s "
              f"{touched:>3}/{total_c:<5} "
              f"{status:<10}")

        if a['touched_cones']:
            for tc in a['touched_cones']:
                d = a['cone_details'][tc]['min_distance']
                r = a['cone_details'][tc]['radius']
                consec = a['cone_details'][tc].get('consecutive_max', 0)
                print(f"         └─ Cone {tc}: min_dist={d:.1f}px "
                      f"(radius={r:.1f}px, consecutive={consec})")
    print()


# ============================================================
# RENDERING SEMUA FRAME
# ============================================================

def render_frames(
    frames            : List[np.ndarray],
    tracks            : Dict,
    ball_possessions  : List[int],
    detected_attempts : List[Dict],
    dribble_detector  : DribbleDetector,
    config            : Dict
) -> List[np.ndarray]:
    output_frames = []
    total_frames = len(frames)

    stabilized_cones = dribble_detector.get_all_cones() or {}
    cone_radii = dribble_detector.get_cone_radii()
    ordered_ids = dribble_detector.get_ordered_cone_ids()

    rolling_trajectory: List = []

    # Pre-compute: frame ranges per attempt untuk highlight
    attempt_frame_map: Dict[int, Dict] = {}
    for a in detected_attempts:
        for f in range(a['frame_start'], a['frame_end'] + 1):
            attempt_frame_map[f] = a

    # Pre-compute: result flash frames
    flash_frames: Dict[int, Dict] = {}
    flash_duration = config.get("result_flash_frames", 45)
    for a in detected_attempts:
        for f in range(a['frame_end'], min(a['frame_end'] + flash_duration, total_frames)):
            flash_frames[f] = a

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
            ball_pos_for_viz = None
            ball_data_viz = tracks["ball"][frame_num].get(1)
            if ball_data_viz:
                ball_pos_for_viz = get_center_of_bbox_bottom(ball_data_viz["bbox"])

            annotated = draw_cone_zones_on_frame(
                annotated,
                stabilized_cones = stabilized_cones,
                cone_radii       = cone_radii,
                ordered_ids      = ordered_ids,
                touched_cones    = touched_cones_now if active_attempt else [],
                is_dribbling     = active_attempt is not None,
                ball_pos         = ball_pos_for_viz,
            )

        # 2. Bounding box pemain
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

            label = f"Player {player_id}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 8, y1), box_color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

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
                touched_cnt = len(touched_cones_now),
                total_cones = active_attempt['total_cones'],
                duration_sec = (frame_num - active_attempt['frame_start']) / config.get('fps', 30),
            )

        # 6. Result flash (SUKSES/GAGAL) setelah attempt selesai
        flash_attempt = flash_frames.get(frame_num)
        if flash_attempt and frame_num >= flash_attempt['frame_end']:
            annotated = draw_result_flash(
                annotated,
                success       = flash_attempt['success'],
                touched_cones = flash_attempt['touched_cones'],
                total_cones   = flash_attempt['total_cones'],
                attempt_number= flash_attempt['attempt_id'],
            )

        # 7. Panel statistik realtime
        if config.get("show_stats_panel", True):
            realtime_stats = compute_progressive_stats(detected_attempts, frame_num)
            annotated = draw_dribble_stats_panel(
                annotated,
                stats       = realtime_stats,
                position    = (20, 20),
                panel_width = 295,
            )

        # 8. Label frame
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
    print("   FOOTBALL DRIBBLING ANALYTICS v2.0")
    print("   Cone Touch Detection — Enhanced Logic")
    print("=" * 70)
    print(f"  Input             : {CONFIG['input_video']}")
    print(f"  Output            : {CONFIG['output_video']}")
    print(f"  Model             : {CONFIG['model_path']}")
    print(f"  Gunakan cache     : {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Cone radius       : multiplier={CONFIG['cone_radius_multiplier']}, "
          f"default={CONFIG['default_cone_radius']}px")
    print(f"  Possession dist   : {CONFIG['max_possession_distance']}px")
    print(f"  Temporal filter   : {CONFIG['min_consecutive_touch_frames']} frames")
    print(f"  Ball edge dist    : {'Ya' if CONFIG['use_ball_edge_distance'] else 'Tidak'}")
    print(f"  Interpolation     : {CONFIG['interpolation_substeps']} substeps")
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
    CONFIG["fps"] = fps
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
    # TAHAP 3: Inisialisasi Dribble Detector & Cone
    # ==========================================================
    print("\n[MAIN] TAHAP 3: Inisialisasi Dribble Detector & Cone...")

    dribble_detector = DribbleDetector(fps=fps)

    # Transfer config ke detector
    dribble_detector.cone_radius_multiplier       = CONFIG["cone_radius_multiplier"]
    dribble_detector.default_cone_radius          = CONFIG["default_cone_radius"]
    dribble_detector.min_cone_radius              = CONFIG["min_cone_radius"]
    dribble_detector.max_cone_radius              = CONFIG["max_cone_radius"]
    dribble_detector.entry_exit_zone_radius       = CONFIG["entry_exit_zone_radius"]
    dribble_detector.min_attempt_frames           = CONFIG["min_attempt_frames"]
    dribble_detector.cooldown_frames              = CONFIG["cooldown_frames"]
    dribble_detector.cone_order_direction         = CONFIG["cone_order_direction"]
    dribble_detector.min_consecutive_touch_frames = CONFIG["min_consecutive_touch_frames"]
    dribble_detector.use_ball_edge_distance       = CONFIG["use_ball_edge_distance"]
    dribble_detector.interpolation_substeps       = CONFIG["interpolation_substeps"]

    cone_ok = dribble_detector.initialize_cones(
        tracks, cone_key='cones', sample_frames=30, debug=True
    )

    if not cone_ok:
        print("[MAIN] WARNING: Cone tidak teridentifikasi!")
        return

    # ==========================================================
    # TAHAP 4: Assignment Possession Bola Per Frame
    # ==========================================================
    print("\n[MAIN] TAHAP 4: Menentukan possession bola per frame...")

    assigner = PlayerBallAssigner(
        max_possession_distance=CONFIG["max_possession_distance"]
    )
    ball_possessions = assigner.assign_ball_to_players_bulk(tracks)

    # ==========================================================
    # TAHAP 5: Deteksi Dribble Attempts
    # ==========================================================
    print("\n[MAIN] TAHAP 5: Deteksi dribble attempts...")

    detected_attempts = dribble_detector.detect_dribble_attempts(
        tracks,
        ball_possessions,
        debug=True
    )

    # ==========================================================
    # TAHAP 6: Statistik
    # ==========================================================
    print("\n[MAIN] TAHAP 6: Menghitung statistik...")

    stats = dribble_detector.get_dribble_statistics(detected_attempts)
    print_dribble_details(detected_attempts, stats)

    # ==========================================================
    # TAHAP 7: Render Video Output
    # ==========================================================
    print("\n[MAIN] TAHAP 7: Merender video output...")

    output_frames = render_frames(
        frames            = frames,
        tracks            = tracks,
        ball_possessions  = ball_possessions,
        detected_attempts = detected_attempts,
        dribble_detector  = dribble_detector,
        config            = CONFIG,
    )

    # ==========================================================
    # TAHAP 8: Simpan Video Output
    # ==========================================================
    print(f"\n[MAIN] TAHAP 8: Menyimpan video ke: {CONFIG['output_video']}...")

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
