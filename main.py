# main.py
# Pipeline utama: baca video -> deteksi -> tracking -> analisis -> render -> simpan

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append('../')

from trackers              import Tracker
from team_assigner.player_identifier    import PlayerIdentifier
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.pass_detector        import PassDetector
from draw_gate            import (
    draw_gate_on_frame,
    draw_ball_trajectory_on_frame,
    draw_pass_arrow,
    draw_stats_panel,
    draw_target_cone_on_frame
)
from utils.bbox_utils import extract_ball_trajectory


# ============================================================
# KONFIGURASI UTAMA
# Sesuaikan seluruh nilai di bawah ini sebelum menjalankan program
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
        4: "Unknown",
        5: "#3",
        6: "#19",
        7: "Unknown",
        8: "#3",
        9: "#19",
    },
    # ============================================================
    # KONFIGURASI TARGET CONE
    # Dari log: Cone ID 3 di (932.4, 276.4) = cone paling atas
    # ============================================================
    "manual_target_cone_id"  : 3,       # ID cone target (cone paling atas)
    "target_selection_mode"  : "highest", # Fallback jika manual None
    "target_proximity_radius": 120.0,   # Radius sukses dalam pixel
                                        # Naikkan jika terlalu banyak GAGAL
                                        # Turunkan jika terlalu banyak SUKSES palsu
    "max_possession_distance": 120.0,
    "show_gate"              : False,   # Gate tidak dipakai lagi
    "show_target_cone"       : True,    # Tampilkan target cone di video
    "debug_trajectory"       : False,
    "show_pass_arrows"       : True,
    "show_stats_panel"       : True,
}


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    """Parse argumen command line (semua opsional, override CONFIG)."""
    parser = argparse.ArgumentParser(
        description="Football Passing Analytics - Gate Detection"
    )
    parser.add_argument(
        "--input",    type=str,
        help="Path video input (override CONFIG['input_video'])"
    )
    parser.add_argument(
        "--output",   type=str,
        help="Path video output (override CONFIG['output_video'])"
    )
    parser.add_argument(
        "--stub",     action="store_true",
        help="Gunakan cache tracking dari stub_path"
    )
    parser.add_argument(
        "--debug",    action="store_true",
        help="Aktifkan visualisasi trajectory bola di output video"
    )
    parser.add_argument(
        "--no-gate",  action="store_true",
        help="Nonaktifkan evaluasi gate (semua pass dianggap SUKSES)"
    )
    return parser.parse_args()


# ============================================================
# CETAK DETAIL PASS EVENTS KE CONSOLE
# ============================================================

def print_pass_details(passes: List[Dict], stats: Dict) -> None:
    """
    Cetak ringkasan statistik dan detail setiap pass event ke console.

    Args:
        passes: list pass event dari detect_passes()
        stats : dict statistik dari get_pass_statistics()
    """
    sep = "=" * 62

    print(f"\n{sep}")
    print("   STATISTIK HASIL ANALISIS PASSING")
    print(sep)
    print(f"  Total Pass           : {stats['total_passes']}")
    print(f"  Sukses (lewat gate)  : {stats['successful_passes']}")
    print(f"  Gagal  (tidak gate)  : {stats['failed_passes']}")
    print(f"  Akurasi              : {stats['accuracy_pct']}%")
    print(f"  Rata2 Jarak Semua    : {stats['avg_distance']} px")
    print(f"  Rata2 Jarak Sukses   : {stats['avg_distance_successful']} px")
    print("-" * 62)

    # Statistik per pemain dengan progress bar sederhana
    print("  Statistik Per Pemain (sebagai Pengirim):")
    for jersey, pstat in stats.get('per_player', {}).items():
        filled = int(pstat['accuracy_pct'] / 10)
        bar    = "█" * filled + "░" * (10 - filled)
        print(f"    #{jersey:<10}: {pstat['success']:>2}/{pstat['total']:>2} pass "
              f"| {bar} {pstat['accuracy_pct']:>5.1f}%")
    print(sep)

    if not passes:
        print("  Tidak ada pass event terdeteksi.\n")
        return

    # Tabel detail setiap pass
    print("\n  Detail Semua Pass Events:")
    print(f"  {'No':<4} {'Dari':<11} {'Ke':<11} "
          f"{'Jarak':>8} {'Bola':>8} {'Status':<10} Alasan Gate")
    print("  " + "-" * 68)

    for i, p in enumerate(passes):
        status = "SUKSES" if p['success'] else "GAGAL"
        print(f"  {i+1:<4} "
              f"#{p['from_jersey']:<10} "
              f"#{p['to_jersey']:<10} "
              f"{p['distance']:>7.0f}px "
              f"{p['ball_movement']:>7.0f}px "
              f"{status:<10} "
              f"{p['gate_reason']}")

    print()


# ============================================================
# RENDERING SEMUA FRAME
# ============================================================

def render_frames(
    frames           : List[np.ndarray],
    tracks           : Dict,
    ball_possessions : List[int],
    detected_passes  : List[Dict],
    player_identifier: PlayerIdentifier,
    pass_detector    : PassDetector,
    stats            : Dict,
    config           : Dict
) -> List[np.ndarray]:
    """
    Render semua frame dengan annotasi lengkap.

    Annotasi yang digambar per frame:
    1. Gate cone (garis + area semi-transparan)
    2. Bounding box pemain + label jersey
    3. Indikator possession (lingkaran di kaki pemain)
    4. Bounding box bola
    5. Trajectory bola rolling (mode debug)
    6. Panah passing + label SUKSES/GAGAL (di frame_display)
    7. Panel statistik di pojok kiri atas

    Args:
        frames           : list frame video asli
        tracks           : dict hasil tracker
        ball_possessions : list possession per frame
        detected_passes  : list pass events dari PassDetector
        player_identifier: objek PlayerIdentifier
        pass_detector    : objek PassDetector (untuk akses gate)
        stats            : dict statistik dari get_pass_statistics()
        config           : dict konfigurasi

    Returns:
        List frame numpy array yang sudah diannotasi
    """
    output_frames = []
    total_frames  = len(frames)

    # Buat map frame_display -> pass_event untuk akses O(1)
    pass_display_map: Dict[int, Dict] = {}
    for p in detected_passes:
        pass_display_map[p['frame_display']] = p

    # Buffer trajectory bola untuk visualisasi rolling
    rolling_trajectory: List = []

    print(f"\n[RENDER] Mulai merender {total_frames} frames...")

    for frame_num, frame in enumerate(frames):
        if frame_num % 100 == 0:
            pct = frame_num / total_frames * 100
            print(f"[RENDER] Progress: {frame_num}/{total_frames} ({pct:.1f}%)...")

        annotated = frame.copy()

        # ------------------------------------------------------
        # 1. Gambar gate (jika tersedia dan diaktifkan)
        # ------------------------------------------------------
        target_info = pass_detector.get_target_cone()
        if target_info and config.get("show_target_cone", True):
            _, target_pos = target_info
            radius        = config.get("target_proximity_radius", 120.0)
            # Target aktif jika ada pass sukses yang sedang berlangsung
            target_active = any(
                p['frame_start'] <= frame_num <= p['frame_end'] and p['success']
                for p in detected_passes
            )
            annotated = draw_target_cone_on_frame(
                annotated,
                target_pos       = target_pos,
                proximity_radius = radius,
                is_active        = target_active
            )

        # ------------------------------------------------------
        # 2. Gambar bounding box pemain + label jersey
        # ------------------------------------------------------
        for player_id, player_data in tracks["players"][frame_num].items():
            bbox = player_data.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            # Cek apakah pemain ini sedang memegang bola
            has_ball = (
                frame_num < len(ball_possessions) and
                ball_possessions[frame_num] == player_id
            )

            # Warna kotak: hijau jika punya bola, biru tua jika tidak
            box_color = (0, 200, 50) if has_ball else (200, 80, 50)

            # Kotak bounding box pemain
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Label jersey di atas kotak
            jersey = player_identifier.get_jersey_number_for_player(player_id)
            label  = f"#{jersey}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            # Background label
            cv2.rectangle(
                annotated,
                (x1, y1 - lh - 8),
                (x1 + lw + 8, y1),
                box_color, -1
            )
            cv2.putText(
                annotated, label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )

            # Indikator possession: lingkaran kuning di posisi kaki
            if has_ball:
                foot_x = (x1 + x2) // 2
                foot_y = y2
                cv2.circle(annotated, (foot_x, foot_y), 8,  (0, 230, 255), -1)
                cv2.circle(annotated, (foot_x, foot_y), 11, (255, 255, 255), 2)

        # ------------------------------------------------------
        # 3. Gambar bola
        # ------------------------------------------------------
        ball_data = tracks["ball"][frame_num].get(1)
        if ball_data:
            bx1, by1, bx2, by2 = map(int, ball_data["bbox"])
            bcx  = (bx1 + bx2) // 2
            bcy  = (by1 + by2) // 2
            brad = max(8, (bx2 - bx1) // 2)

            # Lingkaran bola: kuning-cyan
            cv2.circle(annotated, (bcx, bcy), brad,     (0, 230, 255), 2)
            cv2.circle(annotated, (bcx, bcy), brad - 2, (0, 170, 200), 1)

            # Update buffer trajectory rolling
            if config.get("debug_trajectory", False):
                rolling_trajectory.append((bcx, bcy))
                # Batasi panjang buffer agar tidak memenuhi layar
                if len(rolling_trajectory) > 45:
                    rolling_trajectory.pop(0)

        # ------------------------------------------------------
        # 4. Gambar trajectory bola rolling (mode debug)
        # ------------------------------------------------------
        if config.get("debug_trajectory", False) and len(rolling_trajectory) > 1:
            annotated = draw_ball_trajectory_on_frame(
                annotated,
                trajectory = rolling_trajectory,
                max_points = 30
            )

        # ------------------------------------------------------
        # 5. Gambar panah passing (hanya di frame_display)
        # ------------------------------------------------------
        if config.get("show_pass_arrows", True) and frame_num in pass_display_map:
            pass_event = pass_display_map[frame_num]
            annotated  = draw_pass_arrow(
                annotated,
                from_pos    = pass_event['from_pos'],
                to_pos      = pass_event['to_pos'],
                success     = pass_event['success'],
                from_jersey = pass_event['from_jersey'],
                to_jersey   = pass_event['to_jersey'],
                distance    = pass_event['distance']
            )

        # ------------------------------------------------------
        # 6. Gambar panel statistik di pojok kiri atas
        # ------------------------------------------------------
        if config.get("show_stats_panel", True):
            annotated = draw_stats_panel(
                annotated,
                stats       = stats,
                position    = (20, 20),
                panel_width = 295
            )

        # ------------------------------------------------------
        # 7. Label nomor frame di pojok kanan bawah (untuk debug)
        # ------------------------------------------------------
        h_frame, w_frame = annotated.shape[:2]
        cv2.putText(
            annotated,
            f"Frame: {frame_num}",
            (w_frame - 140, h_frame - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1
        )

        output_frames.append(annotated)

    print(f"[RENDER] Selesai: {len(output_frames)}/{total_frames} frames dirender.")
    return output_frames


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """
    Fungsi utama yang menjalankan seluruh pipeline analitik passing.

    Urutan tahap:
    1.  Baca video input
    2.  Deteksi & tracking objek (YOLOv8 + ByteTrack)
    3.  Identifikasi jersey pemain
    4.  Inisialisasi PassDetector & Gate
    5.  Assignment possession bola per frame
    6.  Deteksi pass events & evaluasi gate
    7.  Hitung statistik
    8.  Render video output dengan annotasi
    9.  Simpan video output
    """

    # --- Parse argumen command line ---
    args = parse_args()

    # Override CONFIG dari argumen jika diberikan
    if args.input:
        CONFIG["input_video"] = args.input
    if args.output:
        CONFIG["output_video"] = args.output
    if args.stub:
        CONFIG["use_stub"] = True
    if args.debug:
        CONFIG["debug_trajectory"] = True
    if args.no_gate:
        # Nonaktifkan semua opsi gate
        CONFIG["manual_gate_cone_ids"]      = None
        CONFIG["gate_position_hint"]        = None
        CONFIG["expected_gate_width_range"] = (9999.0, 99999.0)  # Tidak ada yang valid

    # Cetak header
    print("\n" + "=" * 62)
    print("   FOOTBALL PASSING ANALYTICS v2.0")
    print("   Gate-Based Pass Accuracy Detection")
    print("=" * 62)
    print(f"  Input        : {CONFIG['input_video']}")
    print(f"  Output       : {CONFIG['output_video']}")
    print(f"  Model        : {CONFIG['model_path']}")
    print(f"  Gunakan cache: {'Ya' if CONFIG['use_stub'] else 'Tidak'}")
    print(f"  Debug traj   : {'Ya' if CONFIG['debug_trajectory'] else 'Tidak'}")
    print("=" * 62)

    # ==========================================================
    # TAHAP 1: Baca Video
    # ==========================================================
    print("\n[MAIN] TAHAP 1: Membaca video input...")

    if not os.path.exists(CONFIG["input_video"]):
        print(f"[MAIN] ERROR: File tidak ditemukan: {CONFIG['input_video']}")
        print(f"[MAIN] Pastikan path video sudah benar di CONFIG['input_video']")
        return

    frames = Tracker.read_video(CONFIG["input_video"])

    if not frames:
        print("[MAIN] ERROR: Video tidak bisa dibaca atau kosong!")
        return

    # Baca FPS asli dari metadata video
    cap = cv2.VideoCapture(CONFIG["input_video"])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if fps <= 0:
        fps = CONFIG["fps"]
        print(f"[MAIN] WARNING: FPS tidak bisa dibaca dari video. "
              f"Menggunakan fallback: {fps} fps")
    else:
        print(f"[MAIN] FPS video         : {fps}")

    print(f"[MAIN] Total frame dibaca: {len(frames)}")
    print(f"[MAIN] Durasi video      : {len(frames) / fps:.1f} detik")

    # ==========================================================
    # TAHAP 2: Deteksi & Tracking Objek
    # ==========================================================
    print("\n[MAIN] TAHAP 2: Deteksi & Tracking objek...")
    print(f"[MAIN] Model YOLOv8: {CONFIG['model_path']}")

    if not os.path.exists(CONFIG["model_path"]):
        print(f"[MAIN] ERROR: Model tidak ditemukan: {CONFIG['model_path']}")
        return

    tracker = Tracker(model_path=CONFIG["model_path"])
    tracks  = tracker.get_object_tracks(
        frames,
        read_from_stub = CONFIG["use_stub"],
        stub_path      = CONFIG["stub_path"]
    )

    # Verifikasi data tracks
    n_player_frames = len(tracks.get('players', []))
    n_ball_frames   = len(tracks.get('ball',    []))
    n_cone_frames   = len(tracks.get('cones',   []))

    print(f"[MAIN] Tracks 'players': {n_player_frames} frames")
    print(f"[MAIN] Tracks 'ball'   : {n_ball_frames} frames")
    print(f"[MAIN] Tracks 'cones'  : {n_cone_frames} frames")

    if n_player_frames == 0 or n_ball_frames == 0:
        print("[MAIN] ERROR: Tracking gagal, tidak ada data player atau bola!")
        return

    # ==========================================================
    # TAHAP 3: Identifikasi Jersey Pemain
    # ==========================================================
    print("\n[MAIN] TAHAP 3: Mapping jersey pemain...")

    player_identifier = PlayerIdentifier(
        track_id_to_jersey = CONFIG["jersey_mapping"]
    )
    player_identifier.print_mappings()

    # ==========================================================
    # TAHAP 4: Inisialisasi Pass Detector & Gate
    # ==========================================================
    print("\n[MAIN] TAHAP 4: Inisialisasi Pass Detector & Target Cone...")
    pass_detector = PassDetector(fps=fps)
    pass_detector.set_jersey_map(player_identifier)
    # Konfigurasi target cone
    pass_detector.manual_target_cone_id  = CONFIG.get("manual_target_cone_id")
    pass_detector.target_selection_mode  = CONFIG.get("target_selection_mode", "highest")
    pass_detector.target_proximity_radius = CONFIG.get("target_proximity_radius", 120.0)
    target_ok = pass_detector.initialize_target_cone(
        tracks,
        cone_key      = 'cones',
        sample_frames = 30,
        debug         = True
    )
    if not target_ok:
        print("[MAIN] WARNING: Target cone tidak teridentifikasi!")
        print("[MAIN]          Semua pass akan dianggap SUKSES.")

    # ==========================================================
    # TAHAP 5: Assignment Possession Bola Per Frame
    # ==========================================================
    print("\n[MAIN] TAHAP 5: Menentukan possession bola per frame...")

    assigner = PlayerBallAssigner(
        max_possession_distance = CONFIG.get("max_possession_distance", 70.0)
    )
    ball_possessions = assigner.assign_ball_to_players_bulk(tracks)

    # Hitung distribusi possession per jersey untuk info
    possession_count: Dict[str, int] = {}
    for pid in ball_possessions:
        if pid == -1:
            jersey = "Tidak ada"
        else:
            jersey = player_identifier.get_jersey_number_for_player(pid)
        possession_count[jersey] = possession_count.get(jersey, 0) + 1

    print(f"\n[MAIN] Distribusi possession:")
    for jersey, count in sorted(possession_count.items(), key=lambda x: -x[1]):
        pct = count / len(ball_possessions) * 100
        print(f"[MAIN]   #{jersey:<12}: {count:>5} frames ({pct:.1f}%)")

    # ==========================================================
    # TAHAP 6: Deteksi Pass Events & Evaluasi Gate
    # ==========================================================
    print("\n[MAIN] TAHAP 6: Deteksi passing & evaluasi gate...")

    detected_passes = pass_detector.detect_passes(
        tracks,
        ball_possessions,
        player_identifier = player_identifier,
        debug             = True
    )

    # ==========================================================
    # TAHAP 7: Hitung Statistik
    # ==========================================================
    print("\n[MAIN] TAHAP 7: Menghitung statistik...")

    stats = pass_detector.get_pass_statistics(detected_passes)

    # Cetak statistik lengkap ke console
    print_pass_details(detected_passes, stats)

    # ==========================================================
    # TAHAP 8: Render Video Output
    # ==========================================================
    print("\n[MAIN] TAHAP 8: Merender video output dengan annotasi...")

    output_frames = render_frames(
        frames            = frames,
        tracks            = tracks,
        ball_possessions  = ball_possessions,
        detected_passes   = detected_passes,
        player_identifier = player_identifier,
        pass_detector     = pass_detector,
        stats             = stats,
        config            = CONFIG
    )

    # ==========================================================
    # TAHAP 9: Simpan Video Output
    # ==========================================================
    print(f"\n[MAIN] TAHAP 9: Menyimpan video ke: {CONFIG['output_video']}...")

    # Buat folder output jika belum ada
    output_dir = os.path.dirname(CONFIG["output_video"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[MAIN] Folder output dibuat: {output_dir}")

    Tracker.save_video(output_frames, CONFIG["output_video"], fps=fps)

    # ==========================================================
    # SELESAI
    # ==========================================================
    print("\n" + "=" * 62)
    print("   PIPELINE SELESAI!")
    print("=" * 62)
    print(f"  Video output  : {CONFIG['output_video']}")
    print(f"  Total frames  : {len(output_frames)}")
    print(f"  Durasi output : {len(output_frames) / fps:.1f} detik")
    print(f"  Total pass    : {stats['total_passes']}")
    print(f"  Akurasi       : {stats['accuracy_pct']}%")
    print("=" * 62 + "\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()