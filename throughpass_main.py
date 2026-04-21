# throughpass_main.py — Through Pass Counting Pipeline
# ============================================================
# Pipeline untuk mendeteksi dan menghitung through pass antara 2 pemain.
#
# Model YOLO: 3 class (ball=0, cone=1, player=2)
# Logic: SUKSES = bola melewati celah 2 cone LAWAN
# Setup: 2 pemain + 4 cone (2 cone di depan setiap pemain)
# ============================================================

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers import Tracker
from trackers.throughpass_detector import ThroughPassDetector
from utils.draw_throughpass import (
    draw_throughpass_status,
    draw_throughpass_result_flash,
    draw_throughpass_stats_panel,
    draw_throughpass_trajectory,
    draw_player_label_tp,
    draw_cone_markers,
)
from utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance


# ============================================================
# KONFIGURASI
# ============================================================

CONFIG = {
    "input_video" : "input_videos/through_pass.mp4",
    "output_video": "output_videos/through_pass.avi",
    "model_path"  : "/home/dika/football-analysis/models/through_pass.pt",
    "stub_path"   : "stubs/tracks_cache_throughpass.pkl",
    "use_stub"    : False,
    "fps"         : 30,

    # CLASS MAPPING (through_pass model: 3 class)
    "class_mapping": {
        'ball': 0,
        'cone': 1,
        'player': 2,
    },

    # PARAMETER THROUGH PASS
    "ball_possession_distance" : 150.0,
    "kick_away_distance"       : 150.0,
    "min_possession_frames"    : 3,
    "max_flight_frames"        : 120,
    "gate_proximity_threshold" : 50.0,
    "min_trajectory_points"    : 3,
    "cooldown_frames"          : 30,
    "min_away_frames"          : 5,
    "cone_stabilize_frames"    : 60,

    # VISUALISASI
    "show_stats_panel"      : True,
    "debug_trajectory"      : True,
    "show_throughpass_status": True,
    "show_cone_markers"     : True,
    "result_flash_frames"   : 45,

    # DEBUG
    "debug_distances"       : True,
    "debug_sample_every"    : 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Football Through Pass Counting"
    )
    parser.add_argument("--input", type=str, help="Path video input")
    parser.add_argument("--output", type=str, help="Path video output")
    parser.add_argument("--stub", action="store_true", help="Gunakan cache")
    parser.add_argument("--no-stub", action="store_true", help="Jangan pakai cache")
    parser.add_argument("--debug", action="store_true", help="Debug trajectory")
    return parser.parse_args()


def compute_progressive_stats(
    events: List[Dict],
    up_to_frame: int
) -> Dict:
    """Hitung statistik secara progresif sampai frame tertentu."""
    events_so_far = [
        e for e in events if e['frame_end'] <= up_to_frame
    ]
    total = len(events_so_far)
    sukses = sum(1 for e in events_so_far if e['success'])
    gagal = total - sukses

    # Per-player stats (real-time)
    player_stats: Dict[int, Dict] = {}
    for e in events_so_far:
        sid = e['sender_id']
        if sid not in player_stats:
            player_stats[sid] = {'total': 0, 'sukses': 0, 'gagal': 0}
        player_stats[sid]['total'] += 1
        if e['success']:
            player_stats[sid]['sukses'] += 1
        else:
            player_stats[sid]['gagal'] += 1

    return {
        'total_throughpass': total,
        'successful_throughpass': sukses,
        'failed_throughpass': gagal,
        'accuracy_pct': round(sukses / total * 100, 1) if total > 0 else 0.0,
        'player_stats': player_stats,
    }


def print_throughpass_details(events: List[Dict], stats: Dict) -> None:
    """Print detail hasil through pass ke console."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("   STATISTIK HASIL ANALISIS THROUGH PASS")
    print(sep)
    print(f"  Total Through Pass  : {stats['total_throughpass']}")
    print(f"  Sukses              : {stats['successful_throughpass']}")
    print(f"  Gagal               : {stats['failed_throughpass']}")
    print(f"  Akurasi             : {stats['accuracy_pct']}%")
    print(f"  Avg Flight (Sukses) : {stats.get('avg_flight_time_success', 0)}s")
    print("-" * 70)

    if stats.get('player_stats'):
        print("  Statistik Per Pemain (Sender):")
        for pid, ps in stats['player_stats'].items():
            print(f"    Player {pid}: Total={ps['total']}, "
                  f"Sukses={ps['sukses']}, Gagal={ps['gagal']}")
        print("-" * 70)

    if not events:
        print("  Tidak ada through pass terdeteksi.\n")
        return

    print(f"\n  {'No':<4} {'Sender':<10} {'Frame':<16} "
          f"{'Flight':<10} {'Status':<10} {'Reason'}")
    print("  " + "-" * 80)
    for i, e in enumerate(events):
        status = "SUKSES" if e['success'] else "GAGAL"
        reason = e.get('reason', '')[:30]
        print(f"  {i+1:<4} "
              f"{'P' + str(e['sender_id']):<10} "
              f"{e['frame_kick']:>4}-{e['frame_end']:<7} "
              f"{e['flight_seconds']:<8.1f}s "
              f"{status:<10} "
              f"{reason}")
    print()


def render_frames(
    frames: List[np.ndarray],
    tracks: Dict,
    throughpass_events: List[Dict],
    throughpass_detector: ThroughPassDetector,
    config: Dict,
) -> List[np.ndarray]:
    """Render semua frame dengan visualisasi through pass."""
    output_frames = []
    total_frames = len(frames)
    rolling_trajectory: List = []
    fps = config.get("fps", 30)

    # Dapatkan info gate dan player untuk visualisasi
    player_a, player_b, gate_a, gate_b = throughpass_detector.get_gates(tracks)

    # Pre-compute: frame → event aktif (saat ball_in_flight)
    flight_frame_map: Dict[int, Dict] = {}
    for e in throughpass_events:
        for f in range(e['frame_kick'], e['frame_end'] + 1):
            flight_frame_map[f] = e

    # Flash frames
    flash_frames: Dict[int, Dict] = {}
    flash_duration = config.get("result_flash_frames", 45)
    for e in throughpass_events:
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

        current_sender = active_event['sender_id'] if active_event else -1

        # ------ 1. Cone markers & gate lines ------
        if config.get("show_cone_markers", True):
            # Ambil cone bboxes di frame ini
            cone_bboxes_frame = {}
            if 'cones' in tracks and frame_num < len(tracks['cones']):
                for cid, cdata in tracks['cones'][frame_num].items():
                    if 'bbox' in cdata:
                        cone_bboxes_frame[cid] = cdata['bbox']

            # Tentukan gate yang aktif
            active_gate_name = None
            highlight_success = False
            highlight_fail = False

            if active_event:
                if current_sender == player_a:
                    active_gate_name = 'B'  # Player A → target Gate B
                elif current_sender == player_b:
                    active_gate_name = 'A'
            if flash_info:
                evt = flash_info['event']
                if evt['sender_id'] == player_a:
                    active_gate_name = 'B'
                elif evt['sender_id'] == player_b:
                    active_gate_name = 'A'
                highlight_success = evt['success']
                highlight_fail = not evt['success']

            annotated = draw_cone_markers(
                annotated,
                gate_a=gate_a,
                gate_b=gate_b,
                cone_bboxes=cone_bboxes_frame,
                active_gate=active_gate_name,
                highlight_success=highlight_success,
                highlight_fail=highlight_fail,
            )

        # ------ 2. Bounding box pemain ------
        for player_id, player_data in tracks["players"][frame_num].items():
            bbox = player_data.get("bbox")
            if bbox is None:
                continue

            is_sender = (player_id == current_sender)
            has_ball = False

            # Cek label pemain (A/B)
            player_label = ""
            if player_id == player_a:
                player_label = "A"
            elif player_id == player_b:
                player_label = "B"

            annotated = draw_player_label_tp(
                annotated,
                bbox=bbox,
                player_id=player_id,
                is_sender=is_sender,
                has_ball=has_ball,
                player_label=player_label,
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
                if len(rolling_trajectory) > 80:
                    rolling_trajectory.pop(0)

        # ------ 4. Trajectory bola ------
        if config.get("debug_trajectory", False) and len(rolling_trajectory) > 1:
            annotated = draw_throughpass_trajectory(
                annotated,
                trajectory=rolling_trajectory,
                max_points=60
            )

        # ------ 5. Through pass status ------
        if config.get("show_throughpass_status", True) and active_event:
            flight_frames_current = frame_num - active_event['frame_kick']

            annotated = draw_throughpass_status(
                annotated,
                state='ball_in_flight',
                sender_id=active_event['sender_id'],
                flight_frames=flight_frames_current,
                fps=fps,
            )

        # ------ 6. Result flash ------
        if flash_info:
            e = flash_info['event']
            annotated = draw_throughpass_result_flash(
                annotated,
                success=e['success'],
                event_number=e['event_id'],
                sender_id=e['sender_id'],
                flight_seconds=e.get('flight_seconds', 0),
                flash_progress=flash_info['progress'],
            )

        # ------ 7. Panel stats realtime ------
        if config.get("show_stats_panel", True):
            rt_stats = compute_progressive_stats(throughpass_events, frame_num)
            annotated = draw_throughpass_stats_panel(
                annotated,
                stats=rt_stats,
                position=(20, 20),
                panel_width=300,
                player_a_id=player_a,
                player_b_id=player_b,
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
    print("   FOOTBALL THROUGH PASS COUNTING v1.0")
    print("   Ball Through Cone Gate Detection (YOLO 3-Class)")
    print("=" * 70)
    print(f"  Input                  : {CONFIG['input_video']}")
    print(f"  Output                 : {CONFIG['output_video']}")
    print(f"  Model                  : {CONFIG['model_path']}")
    print(f"  Cache                  : {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Possession distance    : {CONFIG['ball_possession_distance']}px")
    print(f"  Kick away distance     : {CONFIG['kick_away_distance']}px")
    print(f"  Gate proximity thresh  : {CONFIG['gate_proximity_threshold']}px")
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
    print("\n[MAIN] TAHAP 2: Deteksi & Tracking objek (ball, cone, player)...")
    tracker = Tracker(
        model_path=CONFIG["model_path"],
        class_mapping=CONFIG["class_mapping"],
    )
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=CONFIG["use_stub"],
        stub_path=CONFIG["stub_path"]
    )

    # Validasi cone tracks
    if 'cones' not in tracks:
        print("[MAIN] WARNING: 'cones' tidak ada di tracks!")
        tracks['cones'] = [{} for _ in range(len(frames))]

    # TAHAP 3: Inisialisasi Through Pass Detector
    print("\n[MAIN] TAHAP 3: Inisialisasi Through Pass Detector...")
    tp_detector = ThroughPassDetector(fps=fps)
    tp_detector.ball_possession_distance = CONFIG["ball_possession_distance"]
    tp_detector.kick_away_distance       = CONFIG["kick_away_distance"]
    tp_detector.min_possession_frames    = CONFIG["min_possession_frames"]
    tp_detector.max_flight_frames        = CONFIG["max_flight_frames"]
    tp_detector.gate_proximity_threshold = CONFIG["gate_proximity_threshold"]
    tp_detector.min_trajectory_points    = CONFIG["min_trajectory_points"]
    tp_detector.cooldown_frames          = CONFIG["cooldown_frames"]
    tp_detector.min_away_frames          = CONFIG["min_away_frames"]
    tp_detector.cone_stabilize_frames    = CONFIG["cone_stabilize_frames"]

    # TAHAP 3.5: DEBUG — Print jarak bola-pemain
    if CONFIG.get("debug_distances", False):
        tp_detector.debug_distances(
            tracks,
            sample_every=CONFIG.get("debug_sample_every", 5)
        )

    # TAHAP 4: Deteksi Through Pass
    print("\n[MAIN] TAHAP 4: Deteksi through pass events...")
    throughpass_events = tp_detector.detect_throughpasses(tracks, debug=True)

    # TAHAP 5: Statistik
    print("\n[MAIN] TAHAP 5: Menghitung statistik...")
    stats = tp_detector.get_throughpass_statistics(throughpass_events)
    print_throughpass_details(throughpass_events, stats)

    # TAHAP 6: Render
    print("\n[MAIN] TAHAP 6: Merender video output...")
    output_frames = render_frames(
        frames=frames,
        tracks=tracks,
        throughpass_events=throughpass_events,
        throughpass_detector=tp_detector,
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
    print(f"  Video output      : {CONFIG['output_video']}")
    print(f"  Total frames      : {len(output_frames)}")
    print(f"  Durasi            : {len(output_frames) / fps:.1f} detik")
    print(f"  Total through pass: {stats['total_throughpass']}")
    print(f"  Sukses            : {stats['successful_throughpass']}")
    print(f"  Gagal             : {stats['failed_throughpass']}")
    print(f"  Akurasi           : {stats['accuracy_pct']}%")
    if stats.get('player_stats'):
        for pid, ps in stats['player_stats'].items():
            print(f"  Player {pid}         : "
                  f"{ps['sukses']}/{ps['total']} sukses")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
