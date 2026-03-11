# main.py
# Pipeline utama v3.0
#
# PERUBAHAN v3.0:
# - Hanya #3 dan #19 yang dihitung akurasi passing
# - Sukses = bola sampai ke kaki Unknown (penerima)
# - Unknown hanya penerima, tidak dihitung akurasi
# - Visualisasi receiver radius di kaki Unknown (efisien)

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers                          import Tracker
from team_assigner.player_identifier   import PlayerIdentifier
from trackers.player_ball_assigner     import PlayerBallAssigner
from trackers.pass_detector            import PassDetector
from draw_gate import (
    draw_gate_on_frame,
    draw_ball_trajectory_on_frame,
    draw_pass_arrow,
    draw_stats_panel,
    draw_target_cone_on_frame,
    draw_front_cones_on_frame
)
from utils.bbox_utils import extract_ball_trajectory, get_center_of_bbox_bottom


# ============================================================
# KONFIGURASI UTAMA
# ============================================================

CONFIG = {
    "input_video" : "input_videos/passing_number.mp4",
    "output_video": "output_videos/passing_number.avi",
    "model_path"  : "/home/dika/football-analysis/models/best.pt",
    "stub_path"   : "stubs/tracks_cache.pkl",
    "use_stub"    : True,
    "fps"         : 30,

    "jersey_mapping": {
        1: "#3",
        2: "#19",
        3: "Unknown",
        4: "#3",
        5: "#19",
        6: "Unknown",
        7: "Unknown"
    },

    "reassign_distance_threshold": 150.0,
    "lost_timeout_frames"        : 60,

    # --- Target cone (HANYA visualisasi opsional) ---
    "manual_target_cone_id"  : 3,
    "target_selection_mode"  : "highest",
    "target_proximity_radius": 100,

    # --- Front cones (HANYA visualisasi opsional) ---
    "front_cone_ids"         : [0, 1, 2],
    "front_cone_radius"      : 125,

    # --- v3.0: Evaluasi ke kaki penerima ---
    "receiver_proximity_radius": 30,

    # --- Possession ---
    "max_possession_distance": 130,

    # --- Visualisasi ---
    "show_gate"              : False,
    "show_target_cone"       : False,
    "show_front_cones"       : False,
    "show_receiver_radius"   : True,
    "debug_trajectory"       : False,
    "show_pass_arrows"       : True,
    "show_stats_panel"       : True,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Football Passing Analytics v3.0"
    )
    parser.add_argument("--input",    type=str)
    parser.add_argument("--output",   type=str)
    parser.add_argument("--stub",     action="store_true")
    parser.add_argument("--debug",    action="store_true")
    parser.add_argument("--no-gate",  action="store_true")
    return parser.parse_args()


# ============================================================
# STATISTIK PROGRESSIVE (REALTIME)
# ============================================================

def compute_progressive_stats(
    detected_passes: List[Dict],
    up_to_frame    : int
) -> Dict:
    passes_so_far = [
        p for p in detected_passes
        if p['frame_display'] <= up_to_frame
    ]

    total  = len(passes_so_far)
    sukses = [p for p in passes_so_far if p['success']]
    gagal  = [p for p in passes_so_far if not p['success']]

    per_player: Dict[str, Dict] = {}
    for p in passes_so_far:
        jersey = p['from_jersey']
        if jersey not in per_player:
            per_player[jersey] = {
                'total': 0, 'success': 0, 'failed': 0,
                'accuracy_pct': 0.0, 'avg_closest': 0.0, '_closest_sum': 0.0
            }
        per_player[jersey]['total'] += 1
        per_player[jersey]['_closest_sum'] += p.get('closest_dist', 0.0)
        if p['success']:
            per_player[jersey]['success'] += 1
        else:
            per_player[jersey]['failed'] += 1

    for jersey, stat in per_player.items():
        stat['accuracy_pct'] = round(
            stat['success'] / stat['total'] * 100, 1
        ) if stat['total'] > 0 else 0.0
        stat['avg_closest'] = round(
            stat['_closest_sum'] / stat['total'], 1
        ) if stat['total'] > 0 else 0.0
        del stat['_closest_sum']

    return {
        'total_passes'           : total,
        'successful_passes'      : len(sukses),
        'failed_passes'          : len(gagal),
        'accuracy_pct'           : round(len(sukses) / total * 100, 1)
                                   if total > 0 else 0.0,
        'avg_distance'           : round(float(np.mean(
                                       [p['distance'] for p in passes_so_far])), 1)
                                   if passes_so_far else 0.0,
        'avg_distance_successful': round(float(np.mean(
                                       [p['distance'] for p in sukses])), 1)
                                   if sukses else 0.0,
        'avg_closest_dist'       : round(float(np.mean(
                                       [p.get('closest_dist', 0)
                                        for p in passes_so_far])), 1)
                                   if passes_so_far else 0.0,
        'per_player'             : per_player
    }


def print_pass_details(passes: List[Dict], stats: Dict) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("   STATISTIK HASIL ANALISIS PASSING")
    print(sep)
    print(f"  Total Pass           : {stats['total_passes']}")
    print(f"  Sukses (ke kaki)     : {stats['successful_passes']}")
    print(f"  Gagal  (tidak sampai): {stats['failed_passes']}")
    print(f"  Akurasi              : {stats['accuracy_pct']}%")
    print(f"  Rata2 Jarak Semua    : {stats['avg_distance']} px")
    print(f"  Rata2 Jarak Sukses   : {stats['avg_distance_successful']} px")
    print(f"  Rata2 Closest Dist   : {stats.get('avg_closest_dist', 0.0)} px")
    print("-" * 62)
    print("  Statistik Per Pemain (sebagai Pengirim):")
    for jersey, pstat in stats.get('per_player', {}).items():
        filled = int(pstat['accuracy_pct'] / 10)
        bar    = "█" * filled + "░" * (10 - filled)
        avg_cl = pstat.get('avg_closest', 0.0)
        print(f"    #{jersey:<10}: {pstat['success']:>2}/{pstat['total']:>2} pass "
              f"| {bar} {pstat['accuracy_pct']:>5.1f}% "
              f"| avg_closest={avg_cl:.0f}px")
    print(sep)

    if not passes:
        print("  Tidak ada pass event terdeteksi.\n")
        return

    print("\n  Detail Semua Pass Events:")
    print(f"  {'No':<4} {'Dari':<11} {'Ke':<11} "
          f"{'Jarak':>8} {'Bola':>8} {'Closest':>9} {'Status':<10} Alasan")
    print("  " + "-" * 78)
    for i, p in enumerate(passes):
        status  = "SUKSES" if p['success'] else "GAGAL"
        reason  = p.get('target_reason', '-')
        closest = p.get('closest_dist', 0.0)
        print(f"  {i+1:<4} "
              f"{p['from_jersey']:<10} "
              f"{p['to_jersey']:<10} "
              f"{p['distance']:>7.0f}px "
              f"{p['ball_movement']:>7.0f}px "
              f"{closest:>8.0f}px "
              f"{status:<10} "
              f"{reason}")
    print()


# ============================================================
# RENDERING — DIPERBAIKI v3.0
# ============================================================

def render_frames(
    frames, tracks, ball_possessions, detected_passes,
    player_identifier, pass_detector, config
) -> List[np.ndarray]:
    output_frames = []
    total_frames  = len(frames)

    pass_display_map: Dict[int, Dict] = {}
    for p in detected_passes:
        pass_display_map[p['frame_display']] = p

    rolling_trajectory: List = []

    # Pre-compute receiver radius config
    show_recv   = config.get("show_receiver_radius", False)
    recv_radius = int(config.get("receiver_proximity_radius", 30))

    print(f"\n[RENDER] Mulai merender {total_frames} frames...")

    for frame_num in range(total_frames):
        if frame_num % 100 == 0:
            pct = frame_num / total_frames * 100
            print(f"[RENDER] Progress: {frame_num}/{total_frames} ({pct:.1f}%)...")

        frame = frames[frame_num]
        annotated = frame.copy()

        # 1. Target cone (opsional)
        target_info = pass_detector.get_target_cone()
        if target_info and config.get("show_target_cone", False):
            _, target_pos = target_info
            radius = config.get("target_proximity_radius", 100.0)
            target_active = any(
                p['frame_start'] <= frame_num <= p['frame_end'] and p['success']
                for p in detected_passes
            )
            annotated = draw_target_cone_on_frame(
                annotated, target_pos=target_pos,
                proximity_radius=radius, is_active=target_active
            )

        # 2. Front cones (opsional)
        if config.get("show_front_cones", False):
            front_cones  = pass_detector.get_front_cones()
            front_radius = pass_detector.get_front_cone_radius()
            if front_cones:
                annotated = draw_front_cones_on_frame(
                    annotated, front_cones=front_cones,
                    proximity_radius=front_radius, is_active=False
                )

        # 3. Receiver radius di kaki Unknown (v3.0)
        #    PERBAIKAN: satu overlay untuk SEMUA Unknown, bukan per-player
        if show_recv:
            overlay_recv = annotated.copy()
            drew_any = False

            for player_id, player_data in tracks["players"][frame_num].items():
                jersey = player_identifier.get_jersey_number_for_player(player_id)
                if jersey != "Unknown":
                    continue
                bbox = player_data.get("bbox")
                if bbox is None:
                    continue

                foot_x = int((bbox[0] + bbox[2]) / 2)
                foot_y = int(bbox[3])

                # Warna: hijau jika baru saja menerima pass sukses
                recv_active = False
                for p in detected_passes:
                    if (p['success']
                        and p['to_player'] == player_id
                        and p['frame_end'] <= frame_num <= p['frame_end'] + 20):
                        recv_active = True
                        break

                color = (0, 255, 80) if recv_active else (0, 200, 255)

                # Gambar lingkaran radius di overlay
                cv2.circle(overlay_recv, (foot_x, foot_y), recv_radius, color, -1)
                drew_any = True

                # Border dan label langsung di annotated (tipis, cepat)
                cv2.circle(annotated, (foot_x, foot_y), recv_radius, color, 1)

                label_r = "RECV"
                (lw_r, lh_r), _ = cv2.getTextSize(
                    label_r, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                lx_r = foot_x - lw_r // 2
                ly_r = foot_y - recv_radius - 8
                cv2.rectangle(annotated,
                              (lx_r - 2, ly_r - lh_r - 2),
                              (lx_r + lw_r + 2, ly_r + 2),
                              color, -1)
                cv2.putText(annotated, label_r, (lx_r, ly_r),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Blend SATU KALI untuk semua Unknown
            if drew_any:
                cv2.addWeighted(overlay_recv, 0.20, annotated, 0.80, 0, annotated)

        # 4. Bounding box pemain + label jersey
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
            label  = f"#{jersey}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(annotated,
                          (x1, y1 - lh - 8), (x1 + lw + 8, y1),
                          box_color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if has_ball:
                foot_x = (x1 + x2) // 2
                foot_y = y2
                cv2.circle(annotated, (foot_x, foot_y), 8,  (0, 230, 255), -1)
                cv2.circle(annotated, (foot_x, foot_y), 11, (255, 255, 255), 2)

        # 5. Bola
        ball_data = tracks["ball"][frame_num].get(1)
        if ball_data:
            bx1, by1, bx2, by2 = map(int, ball_data["bbox"])
            bcx  = (bx1 + bx2) // 2
            bcy  = (by1 + by2) // 2
            brad = max(8, (bx2 - bx1) // 2)

            cv2.circle(annotated, (bcx, bcy), brad,     (0, 230, 255), 2)
            cv2.circle(annotated, (bcx, bcy), brad - 2, (0, 170, 200), 1)

            if config.get("debug_trajectory", False):
                rolling_trajectory.append((bcx, bcy))
                if len(rolling_trajectory) > 45:
                    rolling_trajectory.pop(0)

        # 6. Trajectory bola rolling (debug)
        if config.get("debug_trajectory", False) and len(rolling_trajectory) > 1:
            annotated = draw_ball_trajectory_on_frame(
                annotated, trajectory=rolling_trajectory, max_points=30
            )

        # 7. Panah passing
        if config.get("show_pass_arrows", True) and frame_num in pass_display_map:
            pass_event = pass_display_map[frame_num]
            annotated  = draw_pass_arrow(
                annotated,
                from_pos=pass_event['from_pos'], to_pos=pass_event['to_pos'],
                success=pass_event['success'],
                from_jersey=pass_event['from_jersey'],
                to_jersey=pass_event['to_jersey'],
                distance=pass_event['distance']
            )

        # 8. Panel statistik realtime
        if config.get("show_stats_panel", True):
            realtime_stats = compute_progressive_stats(detected_passes, frame_num)
            annotated = draw_stats_panel(
                annotated, stats=realtime_stats,
                position=(20, 20), panel_width=320
            )

        # 9. Label frame
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
    if args.no_gate:
        CONFIG["manual_gate_cone_ids"]      = None
        CONFIG["gate_position_hint"]        = None
        CONFIG["expected_gate_width_range"] = (9999.0, 99999.0)

    print("\n" + "=" * 62)
    print("   FOOTBALL PASSING ANALYTICS v3.0")
    print("   #3 & #19 -> Unknown (kaki penerima)")
    print("=" * 62)
    print(f"  Input        : {CONFIG['input_video']}")
    print(f"  Output       : {CONFIG['output_video']}")
    print(f"  Model        : {CONFIG['model_path']}")
    print(f"  Gunakan cache: {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Debug traj   : {'Ya' if CONFIG['debug_trajectory'] else 'Tidak'}")
    print(f"  Possession d : {CONFIG['max_possession_distance']}px")
    print(f"  Re-ID thresh : {CONFIG['reassign_distance_threshold']}px")
    print(f"  Recv radius  : {CONFIG['receiver_proximity_radius']}px (ke kaki Unknown)")
    print("=" * 62)

    # TAHAP 1
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
    print(f"[MAIN] FPS video         : {fps}")
    print(f"[MAIN] Total frame dibaca: {len(frames)}")
    print(f"[MAIN] Durasi video      : {len(frames) / fps:.1f} detik")

    # TAHAP 2
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
    print(f"[MAIN] Tracks 'cones'  : {len(tracks.get('cones', []))} frames")

    if n_player_frames == 0 or n_ball_frames == 0:
        print("[MAIN] ERROR: Tracking gagal!")
        return

    # TAHAP 3
    print("\n[MAIN] TAHAP 3: Mapping jersey pemain (Dynamic Re-ID)...")
    player_identifier = PlayerIdentifier(
        track_id_to_jersey          = CONFIG["jersey_mapping"],
        reassign_distance_threshold = CONFIG.get("reassign_distance_threshold", 150.0),
        lost_timeout_frames         = CONFIG.get("lost_timeout_frames", 60)
    )
    print("[MAIN] Menjalankan dynamic re-identification per frame...")
    for frame_num in range(n_player_frames):
        player_identifier.update_frame(
            frame_num=frame_num,
            player_tracks=tracks['players'][frame_num],
            debug=(frame_num % 100 == 0)
        )
    player_identifier.print_mappings()
    for jersey in ["#3", "#19"]:
        all_tids = player_identifier.get_all_track_ids_for_jersey(jersey)
        print(f"[MAIN] Jersey {jersey} pernah di-map ke track IDs: {all_tids}")

    # TAHAP 4
    print("\n[MAIN] TAHAP 4: Inisialisasi Pass Detector...")
    pass_detector = PassDetector(fps=fps)
    pass_detector.set_jersey_map(player_identifier)

    # Cone settings (opsional, untuk visualisasi)
    pass_detector.manual_target_cone_id   = CONFIG.get("manual_target_cone_id")
    pass_detector.target_selection_mode   = CONFIG.get("target_selection_mode", "highest")
    pass_detector.target_proximity_radius = CONFIG.get("target_proximity_radius", 100.0)
    pass_detector.front_cone_ids          = CONFIG.get("front_cone_ids", [0, 1, 2])
    pass_detector.front_cone_radius       = CONFIG.get("front_cone_radius", 125.0)

    # v3.0: Evaluasi ke kaki penerima
    pass_detector.receiver_proximity_radius = CONFIG.get("receiver_proximity_radius", 30)

    target_ok = pass_detector.initialize_target_cone(
        tracks, cone_key='cones', sample_frames=30, debug=True
    )
    if not target_ok:
        print("[MAIN] WARNING: Cone tidak teridentifikasi (opsional untuk v3.0)")

    # TAHAP 5
    print("\n[MAIN] TAHAP 5: Menentukan possession bola per frame...")
    assigner = PlayerBallAssigner(
        max_possession_distance=CONFIG.get("max_possession_distance", 70.0)
    )
    assigner.set_player_identifier(player_identifier)
    ball_possessions = assigner.assign_ball_to_players_bulk(tracks)

    possession_count: Dict[str, int] = {}
    for pid in ball_possessions:
        jersey = "Tidak ada" if pid == -1 else player_identifier.get_jersey_number_for_player(pid)
        possession_count[jersey] = possession_count.get(jersey, 0) + 1
    print(f"\n[MAIN] Distribusi possession:")
    for jersey, count in sorted(possession_count.items(), key=lambda x: -x[1]):
        pct = count / len(ball_possessions) * 100
        print(f"[MAIN]   #{jersey:<12}: {count:>5} frames ({pct:.1f}%)")

    # TAHAP 6
    print("\n[MAIN] TAHAP 6: Deteksi passing & evaluasi ke kaki penerima...")
    detected_passes = pass_detector.detect_passes(
        tracks, ball_possessions,
        player_identifier=player_identifier, debug=True
    )

    # TAHAP 7
    print("\n[MAIN] TAHAP 7: Menghitung statistik...")
    stats = pass_detector.get_pass_statistics(detected_passes)
    print_pass_details(detected_passes, stats)

    # TAHAP 8
    print("\n[MAIN] TAHAP 8: Merender video output...")
    output_frames = render_frames(
        frames, tracks, ball_possessions, detected_passes,
        player_identifier, pass_detector, CONFIG
    )

    # TAHAP 9
    print(f"\n[MAIN] TAHAP 9: Menyimpan video ke: {CONFIG['output_video']}...")
    output_dir = os.path.dirname(CONFIG["output_video"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    Tracker.save_video(output_frames, CONFIG["output_video"], fps=fps)

    print("\n" + "=" * 62)
    print("   PIPELINE SELESAI!")
    print("=" * 62)
    print(f"  Video output  : {CONFIG['output_video']}")
    print(f"  Total frames  : {len(output_frames)}")
    print(f"  Durasi output : {len(output_frames) / fps:.1f} detik")
    print(f"  Total pass    : {stats['total_passes']}")
    print(f"  Akurasi       : {stats['accuracy_pct']}%")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
