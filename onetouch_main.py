# onetouch_main.py — One-Touch Pass Counting Pipeline
# ============================================================
# Pipeline untuk mendeteksi dan menghitung one-touch pass antara 2 pemain.
#
# Model YOLO: 3 class (ball=0, cone=1, player=2)
# Logic: SUKSES = bola dari Player A diterima oleh Player B
#                 dengan sentuhan < 2 detik (one-touch)
#        GAGAL  = bola ditahan > 2 detik (bukan one-touch)
#               = bola tidak sampai ke pemain lain
# ============================================================

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers import Tracker
from trackers.onetouch_detector import OneTouchDetector
from utils.draw_onetouch import (
    draw_onetouch_status,
    draw_onetouch_result_flash,
    draw_onetouch_stats_panel,
    draw_onetouch_trajectory,
    draw_player_label_otp,
)
from utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance


# ============================================================
# KONFIGURASI
# ============================================================

CONFIG = {
    "input_video" : "input_videos/one_touch_pass.mp4",
    "output_video": "output_videos/one_touch_pass.avi",
    "model_path"  : "/home/dika/football-analysis/models/one_touch_pass.pt",
    "stub_path"   : "stubs/tracks_cache_onetouch.pkl",
    "use_stub"    : False,
    "fps"         : 30,

    # CLASS MAPPING (one_touch_pass model: 3 class)
    "class_mapping": {
        'ball': 0,
        'cone': 1,
        'player': 2,
    },

    # PARAMETER ONE-TOUCH PASS
    "ball_possession_distance" : 150.0,
    "kick_away_distance"       : 150.0,
    "receive_distance"         : 200.0,
    "min_possession_frames"    : 2,
    "min_receive_frames"       : 2,
    "max_touch_seconds"        : 2.0,
    "max_transit_frames"       : 90,
    "cooldown_frames"          : 15,
    "min_away_frames"          : 3,
    "player_separation_distance": 150.0,

    # VISUALISASI
    "show_stats_panel"      : True,
    "debug_trajectory"      : True,
    "show_onetouch_status"  : True,
    "result_flash_frames"   : 45,

    # DEBUG
    "debug_distances"       : True,
    "debug_sample_every"    : 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Football One-Touch Pass Counting"
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

    # Avg touch time (sukses saja)
    sukses_events = [e for e in events_so_far if e['success']]
    avg_touch = (
        round(float(np.mean([e['touch_seconds'] for e in sukses_events])), 2)
        if sukses_events else 0.0
    )

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
        'total_onetouch': total,
        'successful_onetouch': sukses,
        'failed_onetouch': gagal,
        'accuracy_pct': round(sukses / total * 100, 1) if total > 0 else 0.0,
        'avg_touch_time_success': avg_touch,
        'player_stats': player_stats,
    }


def print_onetouch_details(events: List[Dict], stats: Dict) -> None:
    """Print detail hasil one-touch pass ke console."""
    sep = "=" * 75
    print(f"\n{sep}")
    print("   STATISTIK HASIL ANALISIS ONE-TOUCH PASS")
    print(sep)
    print(f"  Total One-Touch Pass : {stats['total_onetouch']}")
    print(f"  Sukses               : {stats['successful_onetouch']}")
    print(f"  Gagal                : {stats['failed_onetouch']}")
    print(f"  Akurasi              : {stats['accuracy_pct']}%")
    print(f"  Avg Touch (Sukses)   : {stats.get('avg_touch_time_success', 0)}s")
    print(f"  Avg Transit (Sukses) : {stats.get('avg_transit_time_success', 0)}s")
    print("-" * 75)

    # Breakdown gagal
    if stats['failed_onetouch'] > 0:
        print(f"  Gagal Breakdown:")
        print(f"    Ditahan terlalu lama : {stats.get('gagal_held_too_long', 0)}")
        print(f"    Timeout transit      : {stats.get('gagal_timeout', 0)}")
        print(f"    Bola kembali         : {stats.get('gagal_ball_return', 0)}")
        print(f"    Lainnya              : {stats.get('gagal_other', 0)}")
        print("-" * 75)

    if stats.get('player_stats'):
        print("  Statistik Per Pemain (Sender):")
        for pid, ps in stats['player_stats'].items():
            print(f"    Player {pid}: Total={ps['total']}, "
                  f"Sukses={ps['sukses']}, Gagal={ps['gagal']}")
        print("-" * 75)

    if not events:
        print("  Tidak ada one-touch pass terdeteksi.\n")
        return

    print(f"\n  {'No':<4} {'Sender':<8} {'Recv':<8} {'Frame':<14} "
          f"{'Touch':<8} {'Transit':<9} {'Status':<8} {'Reason'}")
    print("  " + "-" * 80)
    for i, e in enumerate(events):
        status = "SUKSES" if e['success'] else "GAGAL"
        recv_str = f"P{e['receiver_id']}" if e['receiver_id'] != -1 else "-"
        reason = e.get('reason', '')[:25]
        print(f"  {i+1:<4} "
              f"{'P' + str(e['sender_id']):<8} "
              f"{recv_str:<8} "
              f"{e.get('frame_kick', e['frame_start']):>4}-{e['frame_end']:<6} "
              f"{e['touch_seconds']:<6.2f}s "
              f"{e['flight_seconds']:<7.1f}s "
              f"{status:<8} "
              f"{reason}")
    print()


def render_frames(
    frames: List[np.ndarray],
    tracks: Dict,
    onetouch_events: List[Dict],
    onetouch_detector: OneTouchDetector,
    config: Dict,
) -> List[np.ndarray]:
    """Render semua frame dengan visualisasi one-touch pass."""
    output_frames = []
    total_frames = len(frames)
    rolling_trajectory: List = []
    fps = config.get("fps", 30)
    max_touch_seconds = config.get("max_touch_seconds", 2.0)

    # Identifikasi 2 pemain
    player_a, player_b = onetouch_detector._identify_two_players(tracks)

    # ============================================================
    # Pre-compute: maps frame → event info untuk rendering
    # ============================================================

    # Frame during possession (sebelum kick)
    possession_frame_map: Dict[int, Dict] = {}
    for e in onetouch_events:
        kick_f = e.get('frame_kick', e['frame_start'])
        for f in range(e['frame_start'], kick_f + 1):
            possession_frame_map[f] = e

    # Frame during transit (setelah kick, sebelum end)
    transit_frame_map: Dict[int, Dict] = {}
    for e in onetouch_events:
        kick_f = e.get('frame_kick', e['frame_start'])
        for f in range(kick_f + 1, e['frame_end'] + 1):
            transit_frame_map[f] = e

    # Flash frames
    flash_frames: Dict[int, Dict] = {}
    flash_duration = config.get("result_flash_frames", 45)
    for e in onetouch_events:
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
        possession_event = possession_frame_map.get(frame_num)
        transit_event = transit_frame_map.get(frame_num)
        flash_info = flash_frames.get(frame_num)

        # Active event (possession or transit)
        active_event = transit_event or possession_event
        current_possessor = active_event['sender_id'] if active_event else -1

        # ------ 1. Bounding box pemain ------
        for player_id, player_data in tracks["players"][frame_num].items():
            bbox = player_data.get("bbox")
            if bbox is None:
                continue

            is_possessor = (player_id == current_possessor)
            has_ball = False
            touch_sec = 0.0

            # Cek apakah pemain sedang possession
            if possession_event and player_id == possession_event['sender_id']:
                has_ball = True
                # Hitung touch frames sampai frame ini
                frames_since_start = frame_num - possession_event['frame_start']
                touch_sec = frames_since_start / fps

            # Label pemain (A/B)
            player_label = ""
            if player_id == player_a:
                player_label = "A"
            elif player_id == player_b:
                player_label = "B"

            annotated = draw_player_label_otp(
                annotated,
                bbox=bbox,
                player_id=player_id,
                is_possessor=is_possessor,
                has_ball=has_ball,
                touch_seconds=touch_sec,
                max_touch_seconds=max_touch_seconds,
                player_label=player_label,
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
            annotated = draw_onetouch_trajectory(
                annotated,
                trajectory=rolling_trajectory,
                max_points=60
            )

        # ------ 4. One-touch status ------
        if config.get("show_onetouch_status", True):
            if possession_event and not transit_event:
                # Sedang possession — tampilkan touch timer
                frames_since_start = frame_num - possession_event['frame_start']
                touch_sec = frames_since_start / fps

                annotated = draw_onetouch_status(
                    annotated,
                    state='possession',
                    sender_id=possession_event['sender_id'],
                    touch_seconds=touch_sec,
                    max_touch_seconds=max_touch_seconds,
                )

            elif transit_event:
                # Bola sedang transit
                kick_f = transit_event.get('frame_kick', transit_event['frame_start'])
                transit_f = frame_num - kick_f

                # Hitung jarak bola ke semua pemain
                ball_dist = 0.0
                if ball_data:
                    ball_pos = get_center_of_bbox(ball_data["bbox"])
                    for pid, pdata in tracks["players"][frame_num].items():
                        if pid == transit_event['sender_id']:
                            continue
                        p_bbox = pdata.get('bbox')
                        if p_bbox:
                            foot = get_foot_position(p_bbox)
                            d = measure_distance(ball_pos, foot)
                            if ball_dist == 0.0 or d < ball_dist:
                                ball_dist = d

                annotated = draw_onetouch_status(
                    annotated,
                    state='ball_in_transit',
                    sender_id=transit_event['sender_id'],
                    transit_frames=transit_f,
                    ball_dist=ball_dist,
                    fps=fps,
                )

        # ------ 5. Result flash ------
        if flash_info:
            e = flash_info['event']
            annotated = draw_onetouch_result_flash(
                annotated,
                success=e['success'],
                event_number=e['event_id'],
                sender_id=e['sender_id'],
                receiver_id=e.get('receiver_id', -1),
                touch_seconds=e.get('touch_seconds', 0),
                flight_seconds=e.get('flight_seconds', 0),
                reason=e.get('reason', ''),
                flash_progress=flash_info['progress'],
            )

        # ------ 6. Panel stats realtime ------
        if config.get("show_stats_panel", True):
            rt_stats = compute_progressive_stats(onetouch_events, frame_num)
            annotated = draw_onetouch_stats_panel(
                annotated,
                stats=rt_stats,
                position=(20, 20),
                panel_width=300,
                player_a_id=player_a,
                player_b_id=player_b,
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

    print("\n" + "=" * 75)
    print("   FOOTBALL ONE-TOUCH PASS COUNTING v1.0")
    print("   Quick Pass Detection — Touch Duration < 2s (YOLO 3-Class)")
    print("=" * 75)
    print(f"  Input                  : {CONFIG['input_video']}")
    print(f"  Output                 : {CONFIG['output_video']}")
    print(f"  Model                  : {CONFIG['model_path']}")
    print(f"  Cache                  : {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Possession distance    : {CONFIG['ball_possession_distance']}px")
    print(f"  Kick away distance     : {CONFIG['kick_away_distance']}px")
    print(f"  Receive distance       : {CONFIG['receive_distance']}px")
    print(f"  Max touch seconds      : {CONFIG['max_touch_seconds']}s")
    print(f"  Max transit frames     : {CONFIG['max_transit_frames']}")
    print(f"  Cooldown frames        : {CONFIG['cooldown_frames']}")
    print("=" * 75)

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

    # Validasi cone tracks (tidak digunakan dalam logic, tapi ada di model)
    if 'cones' not in tracks:
        print("[MAIN] INFO: 'cones' tidak ada di tracks (tidak diperlukan untuk one-touch).")
        tracks['cones'] = [{} for _ in range(len(frames))]

    # TAHAP 3: Inisialisasi One-Touch Detector
    print("\n[MAIN] TAHAP 3: Inisialisasi One-Touch Pass Detector...")
    otp_detector = OneTouchDetector(fps=fps)
    otp_detector.ball_possession_distance = CONFIG["ball_possession_distance"]
    otp_detector.kick_away_distance       = CONFIG["kick_away_distance"]
    otp_detector.receive_distance         = CONFIG["receive_distance"]
    otp_detector.min_possession_frames    = CONFIG["min_possession_frames"]
    otp_detector.min_receive_frames       = CONFIG["min_receive_frames"]
    otp_detector.max_touch_seconds        = CONFIG["max_touch_seconds"]
    otp_detector.max_transit_frames       = CONFIG["max_transit_frames"]
    otp_detector.cooldown_frames          = CONFIG["cooldown_frames"]
    otp_detector.min_away_frames          = CONFIG["min_away_frames"]
    otp_detector.player_separation_distance = CONFIG.get("player_separation_distance", 150.0)

    # TAHAP 3.5: DEBUG — Print jarak bola-pemain
    if CONFIG.get("debug_distances", False):
        otp_detector.debug_distances(
            tracks,
            sample_every=CONFIG.get("debug_sample_every", 5)
        )

    # TAHAP 4: Deteksi One-Touch Pass
    print("\n[MAIN] TAHAP 4: Deteksi one-touch pass events...")
    onetouch_events = otp_detector.detect_onetouch_passes(tracks, debug=True)

    # TAHAP 5: Statistik
    print("\n[MAIN] TAHAP 5: Menghitung statistik...")
    stats = otp_detector.get_onetouch_statistics(onetouch_events)
    print_onetouch_details(onetouch_events, stats)

    # TAHAP 6: Render
    print("\n[MAIN] TAHAP 6: Merender video output...")
    output_frames = render_frames(
        frames=frames,
        tracks=tracks,
        onetouch_events=onetouch_events,
        onetouch_detector=otp_detector,
        config=CONFIG,
    )

    # TAHAP 7: Simpan
    print(f"\n[MAIN] TAHAP 7: Menyimpan video...")
    output_dir = os.path.dirname(CONFIG["output_video"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    Tracker.save_video(output_frames, CONFIG["output_video"], fps=fps)

    # SELESAI
    print("\n" + "=" * 75)
    print("   PIPELINE SELESAI!")
    print("=" * 75)
    print(f"  Video output      : {CONFIG['output_video']}")
    print(f"  Total frames      : {len(output_frames)}")
    print(f"  Durasi            : {len(output_frames) / fps:.1f} detik")
    print(f"  Total one-touch   : {stats['total_onetouch']}")
    print(f"  Sukses            : {stats['successful_onetouch']}")
    print(f"  Gagal             : {stats['failed_onetouch']}")
    print(f"  Akurasi           : {stats['accuracy_pct']}%")
    if stats.get('avg_touch_time_success', 0) > 0:
        print(f"  Avg Touch (Sukses): {stats['avg_touch_time_success']}s")
    if stats.get('player_stats'):
        for pid, ps in stats['player_stats'].items():
            print(f"  Player {pid}         : "
                  f"{ps['sukses']}/{ps['total']} sukses")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
