# main.py — Heading Detection Pipeline

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers import Tracker
from trackers.heading_detector import HeadingDetector
from utils.draw_heading import (
    draw_head_bbox_on_frame,
    draw_heading_status,
    draw_heading_result_flash,
    draw_heading_stats_panel,
    draw_ball_trajectory_on_frame,
)

from utils.bbox_utils import get_center_of_bbox


# ============================================================
# KONFIGURASI UTAMA
# ============================================================

CONFIG = {
    "input_video" : "input_videos/heading.mp4",
    "output_video": "output_videos/heading_count.avi",
    "model_path"  : "/home/dika/football-analysis/models/best.pt",
    "stub_path"   : "stubs/tracks_cache_heading.pkl",
    "use_stub"    : False,
    "fps"         : 30,

    # ============================================================
    # PARAMETER HEADING DETECTION
    # ============================================================
    "head_bbox_margin"              : 15.0,   # Margin toleransi overlap
    "max_head_ball_center_distance" : 60.0,   # Jarak center fallback
    "max_approach_distance"         : 180.0,  # Jarak maks bola → pemain
    "min_approach_frames"           : 3,
    "cooldown_frames"               : 25,
    "max_approach_duration_sec"     : 3.0,
    "min_away_frames"               : 8,

    # ============================================================
    # VISUALISASI
    # ============================================================
    "show_head_bbox"       : True,
    "show_stats_panel"     : True,
    "debug_trajectory"     : True,
    "show_heading_status"  : True,
    "result_flash_frames"  : 45,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Football Heading Analytics"
    )
    parser.add_argument("--input", type=str, help="Path video input")
    parser.add_argument("--output", type=str, help="Path video output")
    parser.add_argument("--stub", action="store_true", help="Gunakan cache tracking")
    parser.add_argument("--debug", action="store_true", help="Debug trajectory")
    return parser.parse_args()


def compute_progressive_stats(
    heading_events: List[Dict],
    up_to_frame: int
) -> Dict:
    """Statistik heading realtime sampai frame tertentu."""
    events_so_far = [
        e for e in heading_events if e['frame_end'] <= up_to_frame
    ]
    total = len(events_so_far)
    sukses = sum(1 for e in events_so_far if e['success'])
    gagal = total - sukses

    return {
        'total_headings': total,
        'successful_headings': sukses,
        'failed_headings': gagal,
        'accuracy_pct': round(sukses / total * 100, 1) if total > 0 else 0.0,
    }


def print_heading_details(events: List[Dict], stats: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("   STATISTIK HASIL ANALISIS HEADING")
    print(sep)
    print(f"  Total Heading       : {stats['total_headings']}")
    print(f"  Sukses              : {stats['successful_headings']}")
    print(f"  Gagal               : {stats['failed_headings']}")
    print(f"  Akurasi             : {stats['accuracy_pct']}%")
    print(f"  Avg Dist (Sukses)   : {stats.get('avg_dist_success', 0)}px")
    print(f"  Avg IoU (Sukses)    : {stats.get('avg_iou_success', 0)}")
    print("-" * 70)

    if stats.get('player_stats'):
        print("  Statistik Per Pemain:")
        for pid, ps in stats['player_stats'].items():
            print(f"    Player {pid}: Total={ps['total']}, "
                  f"Sukses={ps['sukses']}, Gagal={ps['gagal']}")
        print("-" * 70)

    if not events:
        print("  Tidak ada heading terdeteksi.\n")
        return

    print(f"\n  {'No':<4} {'Player':<10} {'Frame':<14} "
          f"{'Dist':>8} {'IoU':>8} {'Status':<10}")
    print("  " + "-" * 60)
    for i, e in enumerate(events):
        status = "SUKSES" if e['success'] else "GAGAL"
        print(f"  {i+1:<4} "
              f"{'P' + str(e['player_id']):<10} "
              f"{e['frame_start']:>4}-{e['frame_end']:<7} "
              f"{e['head_ball_distance']:>6.1f}px "
              f"{e['iou']:>6.3f} "
              f"{status:<10}")
    print()


def render_frames(
    frames: List[np.ndarray],
    tracks: Dict,
    heading_events: List[Dict],
    heading_detector: HeadingDetector,
    config: Dict
) -> List[np.ndarray]:
    output_frames = []
    total_frames = len(frames)

    rolling_trajectory: List = []

    # Pre-compute mapping frame → event aktif
    event_frame_map: Dict[int, Dict] = {}
    for e in heading_events:
        for f in range(e['frame_start'], e['frame_end'] + 1):
            event_frame_map[f] = e

    # Flash frames
    flash_frames: Dict[int, Dict] = {}
    flash_duration = config.get("result_flash_frames", 45)
    for e in heading_events:
        for f in range(e['frame_end'],
                       min(e['frame_end'] + flash_duration, total_frames)):
            flash_frames[f] = {
                'event': e,
                'progress': 1.0 - (f - e['frame_end']) / flash_duration,
            }

    print(f"\n[RENDER] Mulai merender {total_frames} frames...")

    for frame_num, frame in enumerate(frames):
        if frame_num % 100 == 0:
            pct = frame_num / total_frames * 100
            print(f"[RENDER] Progress: {frame_num}/{total_frames} ({pct:.1f}%)...")

        annotated = frame.copy()
        active_event = event_frame_map.get(frame_num)
        flash_info = flash_frames.get(frame_num)

        # ------ 1. Bounding box pemain ------
        for player_id, player_data in tracks["players"][frame_num].items():
            bbox = player_data.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            is_active = (
                active_event is not None and
                active_event['player_id'] == player_id
            )
            box_color = (0, 200, 50) if is_active else (200, 80, 50)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            label = f"Player {player_id}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
            )
            cv2.rectangle(annotated, (x1, y1 - lh - 8),
                          (x1 + lw + 8, y1), box_color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        # ------ 2. Bbox kepala (class Heading) ------
        if config.get("show_head_bbox", True):
            heads_in_frame = tracks.get('heading', [{}] * total_frames)[frame_num]
            for head_id, head_data in heads_in_frame.items():
                head_bbox = head_data.get('bbox')
                if head_bbox is None:
                    continue

                is_contact = False
                is_success = False
                is_fail = False

                if flash_info:
                    evt = flash_info['event']
                    if evt.get('head_bbox') and head_bbox:
                        # Cek apakah head bbox ini dekat dengan event head bbox
                        if evt['success']:
                            is_success = True
                        else:
                            is_fail = True
                elif active_event:
                    is_contact = True

                annotated = draw_head_bbox_on_frame(
                    annotated,
                    head_bbox=head_bbox,
                    is_contact=is_contact,
                    is_success=is_success,
                    is_fail=is_fail,
                )

        # ------ 3. Bola ------
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

        # ------ 4. Trajectory bola ------
        if config.get("debug_trajectory", False) and len(rolling_trajectory) > 1:
            annotated = draw_ball_trajectory_on_frame(
                annotated,
                trajectory=rolling_trajectory,
                max_points=40
            )

        # ------ 5. Heading status indicator ------
        if config.get("show_heading_status", True) and active_event:
            annotated = draw_heading_status(
                annotated,
                is_approaching=True,
                is_contact=(active_event['frame_contact'] == frame_num),
                player_id=active_event['player_id'],
                head_dist=active_event.get('head_ball_distance', 0),
                iou=active_event.get('iou', 0),
            )

        # ------ 6. Result flash ------
        if flash_info:
            e = flash_info['event']
            annotated = draw_heading_result_flash(
                annotated,
                success=e['success'],
                event_number=e['event_id'],
                head_distance=e['head_ball_distance'],
                iou=e.get('iou', 0),
                flash_progress=flash_info['progress'],
            )

        # ------ 7. Panel stats realtime ------
        if config.get("show_stats_panel", True):
            rt_stats = compute_progressive_stats(heading_events, frame_num)
            annotated = draw_heading_stats_panel(
                annotated,
                stats=rt_stats,
                position=(20, 20),
                panel_width=280,
            )

        # ------ 8. Frame label ------
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
    print("   FOOTBALL HEADING ANALYTICS v1.0")
    print("   Ball-Head Contact Detection (YOLO Class Based)")
    print("=" * 70)
    print(f"  Input                : {CONFIG['input_video']}")
    print(f"  Output               : {CONFIG['output_video']}")
    print(f"  Model                : {CONFIG['model_path']}")
    print(f"  Head bbox margin     : {CONFIG['head_bbox_margin']}px")
    print(f"  Head-ball center dist: {CONFIG['max_head_ball_center_distance']}px")
    print(f"  Approach distance    : {CONFIG['max_approach_distance']}px")
    print(f"  Cooldown frames      : {CONFIG['cooldown_frames']}")
    print("=" * 70)

    # TAHAP 1: Baca Video
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
    CONFIG["fps"] = fps
    print(f"[MAIN] FPS: {fps}, Total frames: {len(frames)}")

    # TAHAP 2: Deteksi & Tracking
    print("\n[MAIN] TAHAP 2: Deteksi & Tracking objek...")
    tracker = Tracker(model_path=CONFIG["model_path"])
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=CONFIG["use_stub"],
        stub_path=CONFIG["stub_path"]
    )

    # Pastikan key 'heading' ada
    if 'heading' not in tracks:
        print("[MAIN] WARNING: 'heading' tidak ada di tracks.")
        print("[MAIN] Pastikan Tracker menangani class_id=0 sebagai 'heading'.")
        tracks['heading'] = [{} for _ in range(len(frames))]

    # TAHAP 3: Inisialisasi Heading Detector
    print("\n[MAIN] TAHAP 3: Inisialisasi Heading Detector...")
    heading_detector = HeadingDetector(fps=fps)
    heading_detector.head_bbox_margin              = CONFIG["head_bbox_margin"]
    heading_detector.max_head_ball_center_distance = CONFIG["max_head_ball_center_distance"]
    heading_detector.max_approach_distance          = CONFIG["max_approach_distance"]
    heading_detector.min_approach_frames            = CONFIG["min_approach_frames"]
    heading_detector.cooldown_frames                = CONFIG["cooldown_frames"]
    heading_detector.max_approach_duration_sec      = CONFIG["max_approach_duration_sec"]
    heading_detector.min_away_frames                = CONFIG["min_away_frames"]

    # TAHAP 4: Deteksi Heading
    print("\n[MAIN] TAHAP 4: Deteksi heading events...")
    heading_events = heading_detector.detect_headings(tracks, debug=True)

    # TAHAP 5: Statistik
    print("\n[MAIN] TAHAP 5: Menghitung statistik...")
    stats = heading_detector.get_heading_statistics(heading_events)
    print_heading_details(heading_events, stats)

    # TAHAP 6: Render
    print("\n[MAIN] TAHAP 6: Merender video output...")
    output_frames = render_frames(
        frames=frames,
        tracks=tracks,
        heading_events=heading_events,
        heading_detector=heading_detector,
        config=CONFIG,
    )

    # TAHAP 7: Simpan
    print(f"\n[MAIN] TAHAP 7: Menyimpan video...")
    output_dir = os.path.dirname(CONFIG["output_video"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    Tracker.save_video(output_frames, CONFIG["output_video"], fps=fps)

    # SELESAI
    print("\n" + "=" * 70)
    print("   PIPELINE SELESAI!")
    print("=" * 70)
    print(f"  Video output  : {CONFIG['output_video']}")
    print(f"  Total frames  : {len(output_frames)}")
    print(f"  Durasi        : {len(output_frames) / fps:.1f} detik")
    print(f"  Total heading : {stats['total_headings']}")
    print(f"  Sukses        : {stats['successful_headings']}")
    print(f"  Gagal         : {stats['failed_headings']}")
    print(f"  Akurasi       : {stats['accuracy_pct']}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
