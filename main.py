# main.py

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
    # CONE DETECTION
    # ============================================================
    "expected_num_cones"       : 7,       # ← BARU: jumlah cone sebenarnya
    "cone_dedup_distance"      : 80.0,    # ← DIUBAH: 50 → 80
    "min_cone_appearance_ratio": 0.03,    # ← DIUBAH: 0.01 → 0.03

    # ============================================================
    # CONE RADIUS (untuk visualisasi)
    # ============================================================
    "cone_radius_multiplier"  : 0.8,
    "default_cone_radius"     : 25.0,
    "min_cone_radius"         : 15.0,
    "max_cone_radius"         : 40.0,

    # ============================================================
    # DRIBBLE ATTEMPT
    # ============================================================
    "entry_exit_zone_radius"  : 150.0,
    "min_attempt_frames"      : 15,
    "cooldown_frames"         : 30,
    "cone_order_direction"    : "auto",
    "detection_mode"          : "auto",
    "auto_mode_max_duration_sec": 10.0,

    # ============================================================
    # POSSESSION
    # ============================================================
    "max_possession_distance" : 130,

    # ============================================================
    # === BARU: ZIG-ZAG TOUCH DETECTION ===
    # ============================================================
    # Metode 1: Cone displacement (cone bergeser = tertabrak)
    "use_cone_displacement"          : True,
    "cone_displacement_threshold"    : 12.0,   # px pergeseran minimum
    "cone_displacement_window"       : 3,      # frame window
    "cone_displacement_ball_proximity": 150.0, # bola harus dekat

    # Metode 2: BBox overlap (bbox bola tumpang tindih bbox cone)
    "use_bbox_overlap"               : True,
    "bbox_overlap_shrink"            : 3.0,    # shrink cone bbox (px)
    "min_overlap_consecutive_frames" : 2,      # frame berturut-turut

    # Metode 3: Ball speed anomaly (bola melambat dekat cone)
    "use_speed_anomaly"              : True,
    "speed_drop_ratio"               : 0.3,    # drop ke < 30% avg
    "speed_anomaly_proximity"        : 60.0,   # px proximity
    "speed_avg_window"               : 10,     # window rata-rata

    # Kombinasi: min berapa metode harus setuju
    "min_methods_agree"              : 2,      # 1 = salah satu cukup

    # ============================================================
    # VISUALISASI
    # ============================================================
    "show_cone_zones"    : True,
    "show_stats_panel"   : True,
    "debug_trajectory"   : True,
    "show_dribble_status": True,
    "result_flash_frames": 45,
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
        for cid, cnt in sorted(
            stats['cone_hit_frequency'].items(), key=lambda x: -x[1]
        ):
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
                detail = a['cone_details'].get(tc, {})
                votes = detail.get('votes', 0)
                methods = detail.get('methods_triggered', [])
                methods_str = ", ".join(methods) if methods else "-"

                # Info per metode
                extra_info = []
                disp = detail.get('displacement', {})
                if disp.get('displaced'):
                    extra_info.append(
                        f"shift={disp.get('max_displacement', 0):.0f}px"
                    )
                overlap = detail.get('bbox_overlap', {})
                if overlap.get('overlapped'):
                    extra_info.append(
                        f"overlap={overlap.get('overlap_count', 0)}f"
                    )
                speed = detail.get('speed_anomaly', {})
                if speed.get('speed_anomaly'):
                    extra_info.append(
                        f"speed_min={speed.get('min_speed_near_cone', 0):.1f}"
                    )

                extra_str = (
                    " (" + ", ".join(extra_info) + ")" if extra_info else ""
                )
                print(f"         └─ Cone {tc}: votes={votes} "
                      f"[{methods_str}]{extra_str}")
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

    attempt_frame_map: Dict[int, Dict] = {}
    for a in detected_attempts:
        for f in range(a['frame_start'], a['frame_end'] + 1):
            attempt_frame_map[f] = a

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

        # 4. Trajectory bola
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

        # 6. Result flash
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
    print("   FOOTBALL DRIBBLING ANALYTICS v2.1")
    print("   Cone Touch Detection — Enhanced Logic")
    print("=" * 70)
    print(f"  Input             : {CONFIG['input_video']}")
    print(f"  Output            : {CONFIG['output_video']}")
    print(f"  Model             : {CONFIG['model_path']}")
    print(f"  Gunakan cache     : {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Cone radius       : multiplier={CONFIG['cone_radius_multiplier']}, "
          f"default={CONFIG['default_cone_radius']}px, "
          f"range=[{CONFIG['min_cone_radius']}, {CONFIG['max_cone_radius']}]")
    print(f"  Cone dedup dist   : {CONFIG['cone_dedup_distance']}px")
    print(f"  Possession dist   : {CONFIG['max_possession_distance']}px")
    print(f"  Detection mode    : {CONFIG['detection_mode']}")
    # print(f"  Temporal filter   : {CONFIG['min_consecutive_touch_frames']} frames")
    # print(f"  Ball edge dist    : {'Ya' if CONFIG['use_ball_edge_distance'] else 'Tidak'}")
    # print(f"  Interpolation     : {CONFIG['interpolation_substeps']} substeps")
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

        # Transfer parameter BARU ke detector
    dribble_detector.expected_num_cones              = CONFIG.get("expected_num_cones", None)
    dribble_detector.use_cone_displacement           = CONFIG["use_cone_displacement"]
    dribble_detector.cone_displacement_threshold     = CONFIG["cone_displacement_threshold"]
    dribble_detector.cone_displacement_window        = CONFIG["cone_displacement_window"]
    dribble_detector.cone_displacement_ball_proximity= CONFIG["cone_displacement_ball_proximity"]
    dribble_detector.use_bbox_overlap                = CONFIG["use_bbox_overlap"]
    dribble_detector.bbox_overlap_shrink             = CONFIG["bbox_overlap_shrink"]
    dribble_detector.min_overlap_consecutive_frames  = CONFIG["min_overlap_consecutive_frames"]
    dribble_detector.use_speed_anomaly               = CONFIG["use_speed_anomaly"]
    dribble_detector.speed_drop_ratio                = CONFIG["speed_drop_ratio"]
    dribble_detector.speed_anomaly_proximity         = CONFIG["speed_anomaly_proximity"]
    dribble_detector.speed_avg_window                = CONFIG["speed_avg_window"]
    dribble_detector.min_methods_agree               = CONFIG["min_methods_agree"]


    # sample_frames=-1 → scan SEMUA frame
    cone_ok = dribble_detector.initialize_cones(
        tracks, cone_key='cones', sample_frames=-1, debug=True
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
