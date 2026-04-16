# longpass_main.py — Long Pass Counting Pipeline
# ============================================================
# Pipeline untuk mendeteksi dan menghitung long pass antara 2 pemain.
#
# Model YOLO: 2 class (ball=0, player=1)
# Logic: SUKSES = bola dari Player A diterima oleh KAKI Player B
# ============================================================

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers import Tracker
from trackers.longpass_detector import LongPassDetector
from utils.draw_longpass import (
    draw_longpass_status,
    draw_longpass_result_flash,
    draw_longpass_stats_panel,
    draw_longpass_trajectory,
    draw_player_label,
)
from utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance


# ============================================================
# KONFIGURASI
# ============================================================

CONFIG = {
    "input_video" : "input_videos/longpass2.mp4",
    "output_video": "output_videos/longpass_count2.avi",
    "model_path"  : "/home/dika/football-analysis/models/longpass.pt",
    "stub_path"   : "stubs/tracks_cache_longpass.pkl",
    "use_stub"    : False,
    "fps"         : 30,

    # CLASS MAPPING (longpass model: 2 class)
    "class_mapping": {
        'ball': 0,
        'player': 1,
    },

    # PARAMETER LONGPASS
    "ball_possession_distance" : 150.0,
    "kick_away_distance"       : 150.0,
    "receive_distance"         : 200.0,
    "min_possession_frames"    : 3,
    "min_receive_frames"       : 2,
    "max_flight_frames"        : 180,
    "cooldown_frames"          : 30,
    "min_away_frames"          : 5,

    # VISUALISASI
    "show_stats_panel"     : True,
    "debug_trajectory"     : True,
    "show_longpass_status" : True,
    "result_flash_frames"  : 45,

    # DEBUG
    "debug_distances"      : True,
    "debug_sample_every"   : 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Football Long Pass Counting"
    )
    parser.add_argument("--input", type=str, help="Path video input")
    parser.add_argument("--output", type=str, help="Path video output")
    parser.add_argument("--stub", action="store_true", help="Gunakan cache")
    parser.add_argument("--no-stub", action="store_true", help="Jangan pakai cache")
    parser.add_argument("--debug", action="store_true", help="Debug trajectory")
    return parser.parse_args()


def compute_progressive_stats(
    longpass_events: List[Dict],
    up_to_frame: int
) -> Dict:
    """Hitung statistik secara progresif sampai frame tertentu."""
    events_so_far = [
        e for e in longpass_events if e['frame_end'] <= up_to_frame
    ]
    total = len(events_so_far)
    sukses = sum(1 for e in events_so_far if e['success'])
    gagal = total - sukses
    return {
        'total_longpass': total,
        'successful_longpass': sukses,
        'failed_longpass': gagal,
        'accuracy_pct': round(sukses / total * 100, 1) if total > 0 else 0.0,
    }


def print_longpass_details(events: List[Dict], stats: Dict) -> None:
    """Print detail hasil longpass ke console."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("   STATISTIK HASIL ANALISIS LONG PASS")
    print(sep)
    print(f"  Total Long Pass     : {stats['total_longpass']}")
    print(f"  Sukses              : {stats['successful_longpass']}")
    print(f"  Gagal               : {stats['failed_longpass']}")
    print(f"  Akurasi             : {stats['accuracy_pct']}%")
    print(f"  Avg Flight (Sukses) : {stats.get('avg_flight_time_success', 0)}s")
    print(f"  Avg Closest Dist    : {stats.get('avg_closest_distance', 0)}px")
    print("-" * 70)

    if stats.get('player_stats'):
        print("  Statistik Per Pemain (Sender):")
        for pid, ps in stats['player_stats'].items():
            print(f"    Player {pid}: Total={ps['total']}, "
                  f"Sukses={ps['sukses']}, Gagal={ps['gagal']}")
        print("-" * 70)

    if not events:
        print("  Tidak ada longpass terdeteksi.\n")
        return

    print(f"\n  {'No':<4} {'Sender':<10} {'Receiver':<10} {'Frame':<16} "
          f"{'Flight':<10} {'Dist':<10} {'Status':<10}")
    print("  " + "-" * 72)
    for i, e in enumerate(events):
        status = "SUKSES" if e['success'] else "GAGAL"
        print(f"  {i+1:<4} "
              f"{'P' + str(e['sender_id']):<10} "
              f"{'P' + str(e['receiver_id']):<10} "
              f"{e['frame_kick']:>4}-{e['frame_receive']:<7} "
              f"{e['flight_seconds']:<8.1f}s "
              f"{e['closest_distance']:<8.1f}px "
              f"{status:<10}")
    print()


def render_frames(
    frames: List[np.ndarray],
    tracks: Dict,
    longpass_events: List[Dict],
    longpass_detector: LongPassDetector,
    config: Dict,
) -> List[np.ndarray]:
    """Render semua frame dengan visualisasi longpass."""
    output_frames = []
    total_frames = len(frames)
    rolling_trajectory: List = []
    fps = config.get("fps", 30)

    # Identifikasi 2 pemain
    player_a, player_b = longpass_detector._identify_two_players(tracks)

    # Pre-compute: frame → event aktif (saat ball_in_air)
    flight_frame_map: Dict[int, Dict] = {}
    for e in longpass_events:
        for f in range(e['frame_kick'], e['frame_end'] + 1):
            flight_frame_map[f] = e

    # Flash frames
    flash_frames: Dict[int, Dict] = {}
    flash_duration = config.get("result_flash_frames", 45)
    for e in longpass_events:
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
        active_event = flight_frame_map.get(frame_num)
        flash_info = flash_frames.get(frame_num)

        # Determine current state for visualization
        current_sender = active_event['sender_id'] if active_event else -1
        current_receiver = active_event['receiver_id'] if active_event else -1

        # ------ 1. Bounding box pemain ------
        for player_id, player_data in tracks["players"][frame_num].items():
            bbox = player_data.get("bbox")
            if bbox is None:
                continue

            is_sender = (player_id == current_sender)
            is_receiver = (player_id == current_receiver)

            # Cek apakah pemain menerima bola (saat flash sukses)
            has_ball = False
            if flash_info and flash_info['event']['success']:
                if player_id == flash_info['event']['receiver_id']:
                    has_ball = True

            annotated = draw_player_label(
                annotated,
                bbox=bbox,
                player_id=player_id,
                is_sender=is_sender,
                is_receiver=is_receiver,
                has_ball=has_ball,
            )

        # ------ 2. Bola ------
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
                if len(rolling_trajectory) > 80:
                    rolling_trajectory.pop(0)

        # ------ 3. Trajectory bola ------
        if config.get("debug_trajectory", False) and len(rolling_trajectory) > 1:
            annotated = draw_longpass_trajectory(
                annotated,
                trajectory=rolling_trajectory,
                max_points=60
            )

        # ------ 4. Longpass status ------
        if config.get("show_longpass_status", True) and active_event:
            # Hitung jarak bola ke penerima saat ini
            ball_dist = 0.0
            if ball_data:
                ball_pos = get_center_of_bbox(ball_data["bbox"])
                receiver_data = tracks["players"][frame_num].get(
                    active_event['receiver_id']
                )
                if receiver_data:
                    foot_pos = get_foot_position(receiver_data['bbox'])
                    ball_dist = measure_distance(ball_pos, foot_pos)

            flight_frames_current = frame_num - active_event['frame_kick']

            annotated = draw_longpass_status(
                annotated,
                state='ball_in_air',
                sender_id=active_event['sender_id'],
                receiver_id=active_event['receiver_id'],
                ball_dist=ball_dist,
                flight_frames=flight_frames_current,
                fps=fps,
            )

        # ------ 5. Result flash ------
        if flash_info:
            e = flash_info['event']
            annotated = draw_longpass_result_flash(
                annotated,
                success=e['success'],
                event_number=e['event_id'],
                flight_seconds=e.get('flight_seconds', 0),
                receive_distance=e.get('closest_distance', 0),
                flash_progress=flash_info['progress'],
            )

        # ------ 6. Panel stats realtime ------
        if config.get("show_stats_panel", True):
            rt_stats = compute_progressive_stats(longpass_events, frame_num)
            annotated = draw_longpass_stats_panel(
                annotated,
                stats=rt_stats,
                position=(20, 20),
                panel_width=280,
            )

        # ------ 7. Frame label ------
        h_frame, w_frame = annotated.shape[:2]
        cv2.putText(annotated, f"Frame: {frame_num}",
                    (w_frame - 140, h_frame - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1)

        output_frames.append(annotated)

    print(f"[RENDER] Selesai: {len(output_frames)}/{total_frames} frames dirender.")
    return output_frames


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.input:
        CONFIG["input_video"] = args.input
    if args.output:
        CONFIG["output_video"] = args.output
    if args.stub:
        CONFIG["use_stub"] = True
    if args.no_stub:
        CONFIG["use_stub"] = False
    if args.debug:
        CONFIG["debug_trajectory"] = True

    print("\n" + "=" * 70)
    print("   FOOTBALL LONG PASS COUNTING v1.0")
    print("   Ball → Foot Receive Detection (YOLO 2-Class)")
    print("=" * 70)
    print(f"  Input                  : {CONFIG['input_video']}")
    print(f"  Output                 : {CONFIG['output_video']}")
    print(f"  Model                  : {CONFIG['model_path']}")
    print(f"  Cache                  : {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Possession distance    : {CONFIG['ball_possession_distance']}px")
    print(f"  Kick away distance     : {CONFIG['kick_away_distance']}px")
    print(f"  Receive distance       : {CONFIG['receive_distance']}px")
    print(f"  Max flight frames      : {CONFIG['max_flight_frames']}")
    print(f"  Cooldown frames        : {CONFIG['cooldown_frames']}")
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
    tracker = Tracker(
        model_path=CONFIG["model_path"],
        class_mapping=CONFIG["class_mapping"],
    )
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=CONFIG["use_stub"],
        stub_path=CONFIG["stub_path"]
    )

    # TAHAP 3: Inisialisasi Long Pass Detector
    print("\n[MAIN] TAHAP 3: Inisialisasi Long Pass Detector...")
    longpass_detector = LongPassDetector(fps=fps)
    longpass_detector.ball_possession_distance = CONFIG["ball_possession_distance"]
    longpass_detector.kick_away_distance       = CONFIG["kick_away_distance"]
    longpass_detector.receive_distance         = CONFIG["receive_distance"]
    longpass_detector.min_possession_frames    = CONFIG["min_possession_frames"]
    longpass_detector.min_receive_frames       = CONFIG["min_receive_frames"]
    longpass_detector.max_flight_frames        = CONFIG["max_flight_frames"]
    longpass_detector.cooldown_frames          = CONFIG["cooldown_frames"]
    longpass_detector.min_away_frames          = CONFIG["min_away_frames"]

    # TAHAP 3.5: DEBUG — Print jarak bola-kaki
    if CONFIG.get("debug_distances", False):
        longpass_detector.debug_distances(
            tracks,
            sample_every=CONFIG.get("debug_sample_every", 5)
        )

    # TAHAP 4: Deteksi Long Pass
    print("\n[MAIN] TAHAP 4: Deteksi longpass events...")
    longpass_events = longpass_detector.detect_longpasses(tracks, debug=True)

    # TAHAP 5: Statistik
    print("\n[MAIN] TAHAP 5: Menghitung statistik...")
    stats = longpass_detector.get_longpass_statistics(longpass_events)
    print_longpass_details(longpass_events, stats)

    # TAHAP 6: Render
    print("\n[MAIN] TAHAP 6: Merender video output...")
    output_frames = render_frames(
        frames=frames,
        tracks=tracks,
        longpass_events=longpass_events,
        longpass_detector=longpass_detector,
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
    print(f"  Video output   : {CONFIG['output_video']}")
    print(f"  Total frames   : {len(output_frames)}")
    print(f"  Durasi         : {len(output_frames) / fps:.1f} detik")
    print(f"  Total longpass : {stats['total_longpass']}")
    print(f"  Sukses         : {stats['successful_longpass']}")
    print(f"  Gagal          : {stats['failed_longpass']}")
    print(f"  Akurasi        : {stats['accuracy_pct']}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
