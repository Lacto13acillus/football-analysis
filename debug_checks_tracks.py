# debug_track_ids.py
# Menampilkan semua ByteTrack ID beserta frame range dan posisi rata-rata.
# Render video output dengan RAW track ID agar bisa di-mapping manual.
#
# Cara pakai:
#   python3 debug_track_ids.py
#
# Output:
#   1. Tabel track ID di console
#   2. Video debug: output_videos/debug_track_ids.avi

import os
import sys
import cv2
import pickle
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.append('../')

# ============================================================
# KONFIGURASI — sesuaikan dengan setup Anda
# ============================================================

STUB_PATH    = "stubs/tracks_cache.pkl"
VIDEO_PATH   = "input_videos/passing_number.mp4"
OUTPUT_PATH  = "output_videos/debug_track_ids.avi"

# Warna unik per track ID (hingga 12 ID, bisa ditambah)
COLORS = [
    (0, 255, 0),     # hijau
    (255, 0, 0),     # biru
    (0, 0, 255),     # merah
    (255, 255, 0),   # cyan
    (0, 255, 255),   # kuning
    (255, 0, 255),   # magenta
    (128, 255, 0),   # lime
    (0, 128, 255),   # oranye
    (255, 128, 0),   # biru muda
    (128, 0, 255),   # ungu
    (0, 255, 128),   # hijau muda
    (255, 128, 128), # biru pastel
]


def get_color(track_id: int) -> Tuple[int, int, int]:
    """Warna unik per track ID."""
    return COLORS[track_id % len(COLORS)]


def main():
    # ==========================================================
    # 1. Baca tracks dari cache
    # ==========================================================
    if not os.path.exists(STUB_PATH):
        print(f"ERROR: Cache tidak ditemukan: {STUB_PATH}")
        print("Jalankan main.py tanpa --stub terlebih dahulu.")
        return

    print(f"Membaca tracks dari: {STUB_PATH}")
    with open(STUB_PATH, 'rb') as f:
        tracks = pickle.load(f)

    total_frames = len(tracks['players'])
    print(f"Total frames: {total_frames}")

    # ==========================================================
    # 2. Analisis semua track ID
    # ==========================================================
    track_info: Dict[int, Dict] = {}

    for frame_num in range(total_frames):
        for track_id, data in tracks['players'][frame_num].items():
            if track_id not in track_info:
                track_info[track_id] = {
                    'first_frame': frame_num,
                    'last_frame' : frame_num,
                    'frame_count': 0,
                    'positions_x': [],
                    'positions_y': [],
                }
            info = track_info[track_id]
            info['last_frame']  = frame_num
            info['frame_count'] += 1

            bbox = data.get('bbox')
            if bbox:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                info['positions_x'].append(cx)
                info['positions_y'].append(cy)

    # ==========================================================
    # 3. Cetak tabel ringkasan
    # ==========================================================
    print("\n" + "=" * 80)
    print("  SEMUA BYTETRACK ID — ANALISIS POSISI & FRAME RANGE")
    print("=" * 80)
    print(f"  {'ID':>4}  {'Frame Range':>16}  {'#Frames':>8}  "
          f"{'Avg X':>8}  {'Avg Y':>8}  {'Y Range':>16}  {'Mobilitas':>10}")
    print("  " + "-" * 76)

    for tid in sorted(track_info.keys()):
        info = track_info[tid]
        avg_x = np.mean(info['positions_x']) if info['positions_x'] else 0
        avg_y = np.mean(info['positions_y']) if info['positions_y'] else 0
        min_y = np.min(info['positions_y'])  if info['positions_y'] else 0
        max_y = np.max(info['positions_y'])  if info['positions_y'] else 0
        y_range = max_y - min_y

        # Heuristik mobilitas: range Y besar = mobile, kecil = stasioner
        if info['frame_count'] < 10:
            mobility = "NOISE"
        elif y_range < 100:
            mobility = "STASIONER"
        else:
            mobility = "MOBILE"

        print(f"  {tid:>4}  "
              f"{info['first_frame']:>5}-{info['last_frame']:<5}  "
              f"{info['frame_count']:>8}  "
              f"{avg_x:>8.1f}  {avg_y:>8.1f}  "
              f"{min_y:>6.0f}-{max_y:<6.0f}  "
              f"{mobility:>10}")

    print("=" * 80)
    print("\nPetunjuk:")
    print("  - STASIONER + dekat cone target (Y rendah) = kemungkinan #3")
    print("  - MOBILE + Y range besar = kemungkinan #19")
    print("  - NOISE (< 10 frame) = bisa diabaikan")
    print("  - Lihat video debug untuk konfirmasi visual\n")

    # ==========================================================
    # 4. Cek apakah ada track ID yang muncul tepat setelah
    #    track ID lain hilang (indikasi ID switching)
    # ==========================================================
    print("=" * 80)
    print("  POTENSI ID SWITCHING (track hilang → track baru muncul)")
    print("=" * 80)

    sorted_tids = sorted(track_info.keys(), key=lambda t: track_info[t]['first_frame'])
    for i in range(len(sorted_tids)):
        for j in range(i + 1, len(sorted_tids)):
            tid_old = sorted_tids[i]
            tid_new = sorted_tids[j]

            old_last  = track_info[tid_old]['last_frame']
            new_first = track_info[tid_new]['first_frame']

            gap = new_first - old_last

            if 0 <= gap <= 10:  # Muncul dalam 10 frame setelah hilang
                # Cek kedekatan posisi
                old_x = np.mean(track_info[tid_old]['positions_x'][-10:])
                old_y = np.mean(track_info[tid_old]['positions_y'][-10:])
                new_x = np.mean(track_info[tid_new]['positions_x'][:10])
                new_y = np.mean(track_info[tid_new]['positions_y'][:10])

                dist = np.sqrt((old_x - new_x)**2 + (old_y - new_y)**2)

                if dist < 200:
                    print(f"  Track {tid_old} (hilang frame {old_last}) → "
                          f"Track {tid_new} (muncul frame {new_first}) | "
                          f"gap={gap} frames | jarak={dist:.0f}px ← KEMUNGKINAN SAMA!")
                else:
                    print(f"  Track {tid_old} (hilang frame {old_last}) → "
                          f"Track {tid_new} (muncul frame {new_first}) | "
                          f"gap={gap} frames | jarak={dist:.0f}px")

    print("=" * 80)

    # ==========================================================
    # 5. Render video debug dengan RAW track ID
    # ==========================================================
    print(f"\nMerender video debug ke: {OUTPUT_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Tidak bisa membaca video: {VIDEO_PATH}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num < total_frames:
            # Gambar semua player dengan RAW track ID
            for track_id, data in tracks['players'][frame_num].items():
                bbox = data.get('bbox')
                if bbox is None:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                color = get_color(track_id)

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label besar dengan track ID
                label = f"ID:{track_id}"
                font_scale = 0.8
                thickness  = 2
                (lw, lh), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                # Background label
                cv2.rectangle(frame,
                              (x1, y1 - lh - 12),
                              (x1 + lw + 8, y1),
                              color, -1)
                cv2.putText(frame, label,
                            (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), thickness)

                # Titik di kaki (bottom center)
                foot_x = (x1 + x2) // 2
                foot_y = y2
                cv2.circle(frame, (foot_x, foot_y), 5, color, -1)

            # Gambar bola
            ball_data = tracks['ball'][frame_num].get(1)
            if ball_data:
                bx1, by1, bx2, by2 = map(int, ball_data['bbox'])
                bcx = (bx1 + bx2) // 2
                bcy = (by1 + by2) // 2
                cv2.circle(frame, (bcx, bcy), 10, (0, 255, 255), 2)

            # Label frame di pojok
            cv2.putText(frame, f"Frame: {frame_num}",
                        (w - 200, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Daftar track ID aktif di pojok kiri atas
            active_ids = sorted(tracks['players'][frame_num].keys())
            y_offset = 30
            cv2.putText(frame, "Active Track IDs:",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            for tid in active_ids:
                color = get_color(tid)
                cv2.putText(frame, f"  ID:{tid}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 22

        out.write(frame)
        frame_num += 1

        if frame_num % 100 == 0:
            print(f"  Rendering: {frame_num}/{total_frames}...")

    cap.release()
    out.release()

    print(f"\nVideo debug disimpan: {OUTPUT_PATH}")
    print(f"Total frame: {frame_num}")
    print("\n>>> Tonton video debug, lalu update CONFIG['jersey_mapping']")
    print("    di main.py dengan mapping yang benar.\n")


if __name__ == "__main__":
    main()
