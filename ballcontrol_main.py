# ballcontrol_main.py — Ball Control Counting Pipeline
# ============================================================
# Pipeline untuk mendeteksi dan menghitung ball control antara 2 pemain.
#
# Model YOLO: 3 class (ball=0, cone=1, player=2)
# Logic: SUKSES = bola diterima dan tetap di dalam area 4 cone
#        GAGAL  = bola keluar dari area 4 cone setelah diterima
# Setup: 2 pemain + 8 cone (4 cone membentuk persegi per pemain)
# ============================================================

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers import Tracker
from trackers.ballcontrol_detector import BallControlDetector
from utils.draw_ballcontrol import (
    draw_ballcontrol_zone,
    draw_cone_markers_bc,
    draw_ballcontrol_status,
    draw_ballcontrol_result_flash,
    draw_ballcontrol_stats_panel,
    draw_ballcontrol_trajectory,
    draw_player_label_bc,
)
from utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance


# ============================================================
# KONFIGURASI
# ============================================================

CONFIG = {
    "input_video" : "input_videos/aerial_touch.mp4",
    "output_video": "output_videos/aerial_touch.avi",
    "model_path"  : "models/ball_control.pt",
    "stub_path"   : "stubs/tracks_cache_ballcontrol.pkl",
    "use_stub"    : False,
    "fps"         : 30,

    "class_mapping": {
        'ball': 0,
        'cone': 1,
        'player': 2,
    },

    # PARAMETER BALL CONTROL
    "ball_possession_distance" : 150.0,
    "kick_away_distance"       : 150.0,
    "receive_distance"         : 200.0,
    "min_possession_frames"    : 3,
    "min_receive_frames"       : 2,
    "control_check_frames"     : 30,
    "zone_margin"              : 30.0,
    "max_transit_frames"       : 120,
    "cooldown_frames"          : 20,
    "min_away_frames"          : 5,
    "cone_stabilize_frames"    : 60,
    "player_separation_distance": 150.0,

    # VISUALISASI
    "show_stats_panel"      : True,
    "debug_trajectory"      : True,
    "show_ballcontrol_status": True,
    "show_zones"            : True,
    "show_cone_markers"     : True,
    "result_flash_frames"   : 45,

    # DEBUG
    "debug_distances"       : True,
    "debug_sample_every"    : 5,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Football Ball Control Counting")
    parser.add_argument("--input", type=str, help="Path video input")
    parser.add_argument("--output", type=str, help="Path video output")
    parser.add_argument("--stub", action="store_true", help="Gunakan cache")
    parser.add_argument("--no-stub", action="store_true", help="Jangan pakai cache")
    parser.add_argument("--debug", action="store_true", help="Debug trajectory")
    return parser.parse_args()


def compute_progressive_stats(events: List[Dict], up_to_frame: int) -> Dict:
    """Hitung statistik secara progresif sampai frame tertentu."""
    evts = [e for e in events if e['frame_end'] <= up_to_frame]
    total = len(evts)
    sukses = sum(1 for e in evts if e['success'])
    gagal = total - sukses

    player_stats: Dict[int, Dict] = {}
    for e in evts:
        rid = e.get('receiver_id', -1)
        if rid not in player_stats:
            player_stats[rid] = {'total': 0, 'sukses': 0, 'gagal': 0}
        player_stats[rid]['total'] += 1
        if e['success']:
            player_stats[rid]['sukses'] += 1
        else:
            player_stats[rid]['gagal'] += 1

    return {
        'total_ballcontrol': total,
        'successful_ballcontrol': sukses,
        'failed_ballcontrol': gagal,
        'accuracy_pct': round(sukses / total * 100, 1) if total > 0 else 0.0,
        'player_stats': player_stats,
    }


def print_ballcontrol_details(events: List[Dict], stats: Dict) -> None:
    """Print detail hasil ball control ke console."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("   STATISTIK HASIL ANALISIS BALL CONTROL")
    print(sep)
    print(f"  Total Ball Control : {stats['total_ballcontrol']}")
    print(f"  Sukses             : {stats['successful_ballcontrol']}")
    print(f"  Gagal              : {stats['failed_ballcontrol']}")
    print(f"  Akurasi            : {stats['accuracy_pct']}%")
    print("-" * 70)

    if stats.get('player_stats'):
        print("  Statistik Per Pemain (Receiver):")
        for pid, ps in stats['player_stats'].items():
            print(f"    Player {pid}: Total={ps['total']}, "
                  f"Sukses={ps['sukses']}, Gagal={ps['gagal']}")
        print("-" * 70)

    if not events:
        print("  Tidak ada ball control terdeteksi.\n")
        return

    print(f"\n  {'No':<4} {'Sender':<8} {'Recv':<8} {'Frame':<14} "
          f"{'Flight':<9} {'Status':<8} {'Reason'}")
    print("  " + "-" * 75)
    for i, e in enumerate(events):
        status = "SUKSES" if e['success'] else "GAGAL"
        recv = f"P{e['receiver_id']}" if e['receiver_id'] != -1 else "-"
        reason = e.get('reason', '')[:30]
        print(f"  {i+1:<4} "
              f"{'P'+str(e['sender_id']):<8} "
              f"{recv:<8} "
              f"{e['frame_kick']:>4}-{e['frame_end']:<6} "
              f"{e.get('flight_seconds',0):<7.1f}s "
              f"{status:<8} "
              f"{reason}")
    print()


def render_frames(frames, tracks, bc_events, bc_detector, config):
    """Render semua frame dengan visualisasi ball control."""
    output_frames = []
    total_frames = len(frames)
    rolling_traj = []
    fps = config.get("fps", 30)

    # Get zone & player info
    player_a = getattr(bc_detector, '_player_a', -1)
    player_b = getattr(bc_detector, '_player_b', -1)
    zone_a = getattr(bc_detector, '_zone_a', None)
    zone_b = getattr(bc_detector, '_zone_b', None)

    # Pre-compute transit frames
    transit_map = {}
    for e in bc_events:
        ctrl_start = e.get('frame_end', e['frame_kick']) - e.get('control_frames', 0)
        for f in range(e['frame_kick'], ctrl_start):
            transit_map[f] = e

    # Pre-compute checking frames
    checking_map = {}
    for e in bc_events:
        ctrl_frames = e.get('control_frames', 0)
        if ctrl_frames > 0:
            ctrl_start = e['frame_end'] - ctrl_frames + 1
            for f in range(ctrl_start, e['frame_end'] + 1):
                checking_map[f] = e

    # Flash frames
    flash_frames = {}
    flash_dur = config.get("result_flash_frames", 45)
    for e in bc_events:
        for f in range(e['frame_end'], min(e['frame_end']+flash_dur, total_frames)):
            flash_frames[f] = {
                'event': e,
                'progress': 1.0 - (f - e['frame_end']) / flash_dur,
            }

    print(f"\n[RENDER] Mulai merender {total_frames} frames...")

    for frame_num, frame in enumerate(frames):
        if frame_num % 100 == 0:
            print(f"[RENDER] Progress: {frame_num}/{total_frames} "
                  f"({frame_num/total_frames*100:.1f}%)...")

        annotated = frame.copy()
        transit_evt = transit_map.get(frame_num)
        checking_evt = checking_map.get(frame_num)
        flash_info = flash_frames.get(frame_num)
        active_evt = checking_evt or transit_evt

        # Determine zone status for visualization
        zone_a_status = {'active': False, 'checking': False, 'success': False, 'fail': False}
        zone_b_status = {'active': False, 'checking': False, 'success': False, 'fail': False}

        if checking_evt:
            recv = checking_evt.get('receiver_id', -1)
            pos_a = bc_detector._get_player_avg_position(tracks, player_a)
            pos_b = bc_detector._get_player_avg_position(tracks, player_b)
            recv_center = None
            for pid, pdata in tracks['players'][frame_num].items():
                if pid == recv:
                    bbox = pdata.get('bbox')
                    if bbox:
                        recv_center = get_center_of_bbox(bbox)
            if recv_center and pos_a and pos_b:
                da = measure_distance(recv_center, pos_a)
                db = measure_distance(recv_center, pos_b)
                if da < db:
                    zone_a_status['checking'] = True
                else:
                    zone_b_status['checking'] = True

        if flash_info:
            evt = flash_info['event']
            recv = evt.get('receiver_id', -1)
            pos_a2 = bc_detector._get_player_avg_position(tracks, player_a)
            pos_b2 = bc_detector._get_player_avg_position(tracks, player_b)
            recv_c2 = None
            for pid, pdata in tracks['players'][frame_num].items():
                if pid == recv:
                    bbox = pdata.get('bbox')
                    if bbox:
                        recv_c2 = get_center_of_bbox(bbox)
            if recv_c2 and pos_a2 and pos_b2:
                da2 = measure_distance(recv_c2, pos_a2)
                db2 = measure_distance(recv_c2, pos_b2)
                target = zone_a_status if da2 < db2 else zone_b_status
                target['success'] = evt['success']
                target['fail'] = not evt['success']

        # 1. Draw zones
        if config.get("show_zones", True):
            annotated = draw_ballcontrol_zone(
                annotated, zone_a, "A",
                is_active=zone_a_status['active'],
                is_checking=zone_a_status['checking'],
                is_success=zone_a_status['success'],
                is_fail=zone_a_status['fail'])
            annotated = draw_ballcontrol_zone(
                annotated, zone_b, "B",
                is_active=zone_b_status['active'],
                is_checking=zone_b_status['checking'],
                is_success=zone_b_status['success'],
                is_fail=zone_b_status['fail'])

        # 2. Cone markers
        if config.get("show_cone_markers", True):
            cone_bboxes = {}
            if 'cones' in tracks and frame_num < len(tracks['cones']):
                for cid, cdata in tracks['cones'][frame_num].items():
                    if 'bbox' in cdata:
                        cone_bboxes[cid] = cdata['bbox']
            annotated = draw_cone_markers_bc(annotated, cone_bboxes)

        # 3. Player labels
        current_sender = active_evt['sender_id'] if active_evt else -1
        current_recv = -1
        is_ctrl = False
        ball_in_z = True
        if checking_evt:
            current_recv = checking_evt.get('receiver_id', -1)
            is_ctrl = True
            # Check ball in zone
            ball_pos = bc_detector._get_ball_position(tracks, frame_num)
            if ball_pos and checking_evt:
                recv_zone = None
                pos_a3 = bc_detector._get_player_avg_position(tracks, player_a)
                pos_b3 = bc_detector._get_player_avg_position(tracks, player_b)
                for pid, pdata in tracks['players'][frame_num].items():
                    if pid == current_recv:
                        bbox = pdata.get('bbox')
                        if bbox and pos_a3 and pos_b3:
                            rc = get_center_of_bbox(bbox)
                            da3 = measure_distance(rc, pos_a3)
                            db3 = measure_distance(rc, pos_b3)
                            recv_zone = zone_a if da3 < db3 else zone_b
                if recv_zone:
                    ball_in_z = bc_detector._point_in_polygon(ball_pos, recv_zone)

        for pid, pdata in tracks["players"][frame_num].items():
            bbox = pdata.get("bbox")
            if bbox is None:
                continue
            plabel = ""
            if pid == player_a:
                plabel = "A"
            elif pid == player_b:
                plabel = "B"
            annotated = draw_player_label_bc(
                annotated, bbox, pid,
                is_sender=(pid == current_sender),
                is_receiver=(pid == current_recv and not is_ctrl),
                is_controlling=(pid == current_recv and is_ctrl),
                ball_in_zone=ball_in_z,
                player_label=plabel)

        # 4. Ball
        ball_data = tracks["ball"][frame_num].get(1)
        if ball_data:
            bx1, by1, bx2, by2 = map(int, ball_data["bbox"])
            bcx, bcy = (bx1+bx2)//2, (by1+by2)//2
            brad = max(8, (bx2-bx1)//2)
            cv2.circle(annotated, (bcx, bcy), brad, (0, 230, 255), 2)
            cv2.circle(annotated, (bcx, bcy), brad-2, (0, 170, 200), 1)
            if config.get("debug_trajectory", False):
                rolling_traj.append((bcx, bcy))
                if len(rolling_traj) > 80:
                    rolling_traj.pop(0)

        # 5. Trajectory
        if config.get("debug_trajectory", False) and len(rolling_traj) > 1:
            annotated = draw_ballcontrol_trajectory(annotated, rolling_traj)

        # 6. Status
        if config.get("show_ballcontrol_status", True):
            if transit_evt and not checking_evt:
                tf = frame_num - transit_evt['frame_kick']
                annotated = draw_ballcontrol_status(
                    annotated, 'ball_transit',
                    sender_id=transit_evt['sender_id'],
                    transit_frames=tf, fps=fps)
            elif checking_evt:
                ctrl_total = checking_evt.get('control_frames', config['control_check_frames'])
                ctrl_start = checking_evt['frame_end'] - ctrl_total + 1
                cf = frame_num - ctrl_start
                annotated = draw_ballcontrol_status(
                    annotated, 'checking_control',
                    receiver_id=checking_evt.get('receiver_id', -1),
                    control_frames=cf,
                    control_check_frames=config['control_check_frames'],
                    ball_in_zone=ball_in_z, fps=fps)

        # 7. Flash
        if flash_info:
            e = flash_info['event']
            annotated = draw_ballcontrol_result_flash(
                annotated, e['success'], e['event_id'],
                e['sender_id'], e.get('receiver_id', -1),
                e.get('flight_seconds', 0), e.get('reason', ''),
                flash_info['progress'])

        # 8. Stats panel
        if config.get("show_stats_panel", True):
            rt = compute_progressive_stats(bc_events, frame_num)
            annotated = draw_ballcontrol_stats_panel(
                annotated, rt, (20, 20), 300, player_a, player_b)

        # 9. Frame label
        h_f, w_f = annotated.shape[:2]
        cv2.putText(annotated, f"Frame: {frame_num}",
                    (w_f-140, h_f-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110,110,110), 1)

        output_frames.append(annotated)

    print(f"[RENDER] Selesai: {len(output_frames)}/{total_frames} frames.")
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
    print("   FOOTBALL BALL CONTROL COUNTING v1.0")
    print("   Ball Control Detection (YOLO 3-Class)")
    print("=" * 70)
    print(f"  Input              : {CONFIG['input_video']}")
    print(f"  Output             : {CONFIG['output_video']}")
    print(f"  Model              : {CONFIG['model_path']}")
    print(f"  Cache              : {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Possession dist    : {CONFIG['ball_possession_distance']}px")
    print(f"  Control check      : {CONFIG['control_check_frames']}f")
    print(f"  Zone margin        : {CONFIG['zone_margin']}px")
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
    print("\n[MAIN] TAHAP 2: Deteksi & Tracking (ball, cone, player)...")
    tracker = Tracker(
        model_path=CONFIG["model_path"],
        class_mapping=CONFIG["class_mapping"],
    )
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=CONFIG["use_stub"],
        stub_path=CONFIG["stub_path"]
    )

    if 'cones' not in tracks:
        print("[MAIN] WARNING: 'cones' tidak ada di tracks!")
        tracks['cones'] = [{} for _ in range(len(frames))]

    # TAHAP 3: Inisialisasi Ball Control Detector
    print("\n[MAIN] TAHAP 3: Inisialisasi Ball Control Detector...")
    bc = BallControlDetector(fps=fps)
    bc.ball_possession_distance  = CONFIG["ball_possession_distance"]
    bc.kick_away_distance        = CONFIG["kick_away_distance"]
    bc.receive_distance          = CONFIG["receive_distance"]
    bc.min_possession_frames     = CONFIG["min_possession_frames"]
    bc.min_receive_frames        = CONFIG["min_receive_frames"]
    bc.control_check_frames      = CONFIG["control_check_frames"]
    bc.zone_margin               = CONFIG["zone_margin"]
    bc.max_transit_frames        = CONFIG["max_transit_frames"]
    bc.cooldown_frames           = CONFIG["cooldown_frames"]
    bc.min_away_frames           = CONFIG["min_away_frames"]
    bc.cone_stabilize_frames     = CONFIG["cone_stabilize_frames"]
    bc.player_separation_distance = CONFIG["player_separation_distance"]

    # TAHAP 3.5: DEBUG
    if CONFIG.get("debug_distances", False):
        bc.debug_distances(tracks, sample_every=CONFIG.get("debug_sample_every", 5))

    # TAHAP 4: Deteksi Ball Control
    print("\n[MAIN] TAHAP 4: Deteksi ball control events...")
    bc_events = bc.detect_ball_controls(tracks, debug=True)

    # TAHAP 5: Statistik
    print("\n[MAIN] TAHAP 5: Menghitung statistik...")
    stats = bc.get_ballcontrol_statistics(bc_events)
    print_ballcontrol_details(bc_events, stats)

    # TAHAP 6: Render
    print("\n[MAIN] TAHAP 6: Merender video output...")
    output_frames = render_frames(frames, tracks, bc_events, bc, CONFIG)

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
    print(f"  Durasi            : {len(output_frames)/fps:.1f} detik")
    print(f"  Total ball control: {stats['total_ballcontrol']}")
    print(f"  Sukses            : {stats['successful_ballcontrol']}")
    print(f"  Gagal             : {stats['failed_ballcontrol']}")
    print(f"  Akurasi           : {stats['accuracy_pct']}%")
    if stats.get('player_stats'):
        for pid, ps in stats['player_stats'].items():
            print(f"  Player {pid}         : "
                  f"{ps['sukses']}/{ps['total']} sukses")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
