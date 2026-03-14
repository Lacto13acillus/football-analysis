# debug_check_tracks.py
# Debug track IDs untuk penalty kick video
# 
# Fitur:
#   1. Tabel semua track ID + frame range + posisi
#   2. Detail track ID aktif di setiap KICK FRAME
#   3. Analisis warna baju (HSV) per track di kick frames
#   4. Render video debug dengan track ID + warna terdeteksi
#   5. Export crop baju untuk verifikasi visual
#
# Cara pakai:
#   python3 debug_check_tracks.py

import os
import sys
import cv2
import pickle
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

sys.path.append('../')

# ============================================================
# KONFIGURASI
# ============================================================

STUB_PATH    = "stubs/tracks_cache.pkl"
VIDEO_PATH   = "input_videos/penalty_kick.mp4"
OUTPUT_PATH  = "output_videos/debug_track_ids_penalty.avi"
CROP_DIR     = "output_videos/shirt_crops"   # folder untuk crop baju

# Kick frames dari hasil deteksi
KICK_FRAMES  = [86, 266, 476, 668, 894, 1030]

# Warna unik per track ID
COLORS = [
    (0, 255, 0),     # 0  hijau
    (255, 0, 0),     # 1  biru
    (0, 0, 255),     # 2  merah
    (255, 255, 0),   # 3  cyan
    (0, 255, 255),   # 4  kuning
    (255, 0, 255),   # 5  magenta
    (128, 255, 0),   # 6  lime
    (0, 128, 255),   # 7  oranye
    (255, 128, 0),   # 8  biru muda
    (128, 0, 255),   # 9  ungu
    (0, 255, 128),   # 10 hijau muda
    (255, 128, 128), # 11 biru pastel
    (128, 128, 0),   # 12 teal
    (0, 128, 128),   # 13 olive
    (128, 0, 128),   # 14 purple
    (200, 200, 0),   # 15
    (0, 200, 200),   # 16
    (200, 0, 200),   # 17
    (255, 200, 100), # 18
    (100, 200, 255), # 19
]


def get_color(track_id: int) -> Tuple[int, int, int]:
    return COLORS[track_id % len(COLORS)]


def detect_shirt_color_detailed(frame, bbox):
    """
    Deteksi warna baju dengan detail HSV values.
    Returns: (label, mean_h, mean_s, mean_v, red_ratio, gray_ratio)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_box = y2 - y1
    shirt_y1 = y1 + int(h_box * 0.10)
    shirt_y2 = y1 + int(h_box * 0.45)
    margin_x = int((x2 - x1) * 0.15)
    shirt_x1 = max(0, x1 + margin_x)
    shirt_x2 = min(frame.shape[1], x2 - margin_x)

    if shirt_x2 <= shirt_x1 or shirt_y2 <= shirt_y1:
        return "Unknown", 0, 0, 0, 0, 0

    shirt_region = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    if shirt_region.size == 0:
        return "Unknown", 0, 0, 0, 0, 0

    hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_h = float(np.mean(h))
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))

    # Merah
    mask_red1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([12, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 60, 50]), np.array([180, 255, 255]))
    mask_red = mask_red1 | mask_red2
    red_ratio = np.count_nonzero(mask_red) / max(mask_red.size, 1)

    # Abu-abu
    mask_gray = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 60, 180]))
    gray_ratio = np.count_nonzero(mask_gray) / max(mask_gray.size, 1)

    if red_ratio > 0.25:
        label = "Merah"
    elif gray_ratio > 0.35:
        label = "Abu-Abu"
    else:
        label = "Unknown"

    return label, mean_h, mean_s, mean_v, red_ratio, gray_ratio


def save_shirt_crop(frame, bbox, track_id, frame_num, crop_dir):
    """Simpan crop area baju untuk verifikasi visual."""
    x1, y1, x2, y2 = map(int, bbox)
    h_box = y2 - y1
    shirt_y1 = y1 + int(h_box * 0.10)
    shirt_y2 = y1 + int(h_box * 0.45)
    margin_x = int((x2 - x1) * 0.15)
    shirt_x1 = max(0, x1 + margin_x)
    shirt_x2 = min(frame.shape[1], x2 - margin_x)

    if shirt_x2 <= shirt_x1 or shirt_y2 <= shirt_y1:
        return

    shirt = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    if shirt.size == 0:
        return

    # Perbesar agar mudah dilihat
    scale = 4
    shirt_big = cv2.resize(shirt, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_NEAREST)

    fname = f"track{track_id:02d}_frame{frame_num:05d}.png"
    cv2.imwrite(os.path.join(crop_dir, fname), shirt_big)


def main():
    # ==========================================================
    # 1. Baca tracks dari cache
    # ==========================================================
    if not os.path.exists(STUB_PATH):
        print(f"ERROR: Cache tidak ditemukan: {STUB_PATH}")
        print("Jalankan main.py terlebih dahulu untuk generate cache.")
        return

    print(f"Membaca tracks dari: {STUB_PATH}")
    with open(STUB_PATH, 'rb') as f:
        tracks = pickle.load(f)

    total_frames = len(tracks['players'])
    print(f"Total frames: {total_frames}")

    # Baca video untuk analisis warna
    print(f"Membaca video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Tidak bisa membaca video: {VIDEO_PATH}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Baca semua frame
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Frame dibaca: {len(frames)}")

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
                    'bboxes_by_frame': {},
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
                info['bboxes_by_frame'][frame_num] = bbox

    # ==========================================================
    # 3. Cetak tabel ringkasan
    # ==========================================================
    print("\n" + "=" * 90)
    print("  SEMUA BYTETRACK ID — ANALISIS POSISI & FRAME RANGE")
    print("=" * 90)
    print(f"  {'ID':>4}  {'Frame Range':>16}  {'#Frames':>8}  "
          f"{'Avg X':>8}  {'Avg Y':>8}  {'Y Range':>16}  {'Mobilitas':>10}")
    print("  " + "-" * 84)

    for tid in sorted(track_info.keys()):
        info = track_info[tid]
        avg_x = np.mean(info['positions_x']) if info['positions_x'] else 0
        avg_y = np.mean(info['positions_y']) if info['positions_y'] else 0
        min_y = np.min(info['positions_y'])  if info['positions_y'] else 0
        max_y = np.max(info['positions_y'])  if info['positions_y'] else 0
        y_range = max_y - min_y

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

    print("=" * 90)

    # ==========================================================
    # 4. DETAIL TRACK IDS DI SETIAP KICK FRAME
    #    Ini yang paling penting untuk mapping
    # ==========================================================
    print("\n" + "=" * 110)
    print("  DETAIL TRACK IDs DI SETIAP KICK FRAME")
    print("  (Menampilkan semua track aktif + analisis warna baju)")
    print("=" * 110)

    os.makedirs(CROP_DIR, exist_ok=True)

    for kick_idx, kick_frame in enumerate(KICK_FRAMES):
        print(f"\n  ┌─── KICK #{kick_idx + 1} @ Frame {kick_frame} "
              f"(waktu: {kick_frame/fps:.2f}s) ───┐")
        print(f"  │")
        print(f"  │  {'Track':>6}  {'BBox':>30}  {'Warna':>10}  "
              f"{'H':>5} {'S':>5} {'V':>5}  "
              f"{'Red%':>6} {'Gray%':>6}  {'Jarak ke Bola':>14}")

        # Posisi bola di kick frame
        ball_data = tracks['ball'][kick_frame].get(1)
        ball_pos = None
        if ball_data and 'bbox' in ball_data:
            bx1, by1, bx2, by2 = ball_data['bbox']
            ball_pos = ((bx1 + bx2) / 2, (by1 + by2) / 2)
            print(f"  │  Bola: ({ball_pos[0]:.0f}, {ball_pos[1]:.0f})")

        print(f"  │  " + "-" * 100)

        # Cek semua track aktif di window [kick_frame-30, kick_frame+5]
        check_start = max(0, kick_frame - 30)
        check_end   = min(total_frames - 1, kick_frame + 5)

        # Kumpulkan track IDs yang aktif di sekitar kick
        active_tracks_near_kick = set()
        for f in range(check_start, check_end + 1):
            for tid in tracks['players'][f].keys():
                active_tracks_near_kick.add(tid)

        # Analisis warna per track di kick frame dan sekitarnya
        for tid in sorted(active_tracks_near_kick):
            # Cari frame terdekat dengan kick_frame yang punya data
            best_f = None
            for f in [kick_frame] + list(range(kick_frame - 5, kick_frame + 5)):
                if 0 <= f < total_frames and tid in tracks['players'][f]:
                    best_f = f
                    break

            if best_f is None:
                continue

            data = tracks['players'][best_f][tid]
            bbox = data.get('bbox')
            if bbox is None:
                continue

            # Deteksi warna detail
            if best_f < len(frames):
                label, mh, ms, mv, rr, gr = detect_shirt_color_detailed(
                    frames[best_f], bbox
                )
                # Simpan crop baju
                save_shirt_crop(frames[best_f], bbox, tid, best_f, CROP_DIR)
            else:
                label, mh, ms, mv, rr, gr = "N/A", 0, 0, 0, 0, 0

            # Jarak ke bola
            dist_str = "N/A"
            if ball_pos:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                foot_y = bbox[3]
                dist_center = np.sqrt(
                    (cx - ball_pos[0])**2 + (cy - ball_pos[1])**2
                )
                dist_foot = np.sqrt(
                    (cx - ball_pos[0])**2 + (foot_y - ball_pos[1])**2
                )
                dist = min(dist_center, dist_foot)
                dist_str = f"{dist:.0f}px"

            bbox_str = (f"({bbox[0]:.0f},{bbox[1]:.0f})-"
                        f"({bbox[2]:.0f},{bbox[3]:.0f})")

            # Tandai track terdekat ke bola
            marker = ""
            if ball_pos:
                if dist_str != "N/A" and float(dist_str.replace("px", "")) < 250:
                    marker = " ← DEKAT BOLA"

            at_kick = "(at kick)" if best_f == kick_frame else f"(at f{best_f})"

            print(f"  │  {tid:>6}  {bbox_str:>30}  {label:>10}  "
                  f"{mh:>5.1f} {ms:>5.1f} {mv:>5.1f}  "
                  f"{rr*100:>5.1f}% {gr*100:>5.1f}%  "
                  f"{dist_str:>14}{marker}  {at_kick}")

        print(f"  │")
        print(f"  └{'─' * 107}┘")

    # ==========================================================
    # 5. Analisis warna baju per track ID (multi-frame voting)
    # ==========================================================
    print("\n" + "=" * 90)
    print("  ANALISIS WARNA BAJU PER TRACK ID (sampling 20 frame)")
    print("=" * 90)

    for tid in sorted(track_info.keys()):
        info = track_info[tid]
        if info['frame_count'] < 5:
            continue

        # Sample frame dari track ini
        all_frames_for_track = sorted(info['bboxes_by_frame'].keys())
        step = max(1, len(all_frames_for_track) // 20)
        sample_frames = all_frames_for_track[::step][:20]

        color_votes = []
        hsv_samples = []

        for f in sample_frames:
            if f >= len(frames):
                continue
            bbox = info['bboxes_by_frame'][f]
            label, mh, ms, mv, rr, gr = detect_shirt_color_detailed(
                frames[f], bbox
            )
            color_votes.append(label)
            hsv_samples.append((mh, ms, mv, rr, gr))

        # Voting
        vote_counts = Counter(color_votes)
        total_votes = len(color_votes)
        non_unknown = [v for v in color_votes if v != "Unknown"]
        best_color = Counter(non_unknown).most_common(1)[0][0] if non_unknown else "Unknown"

        avg_h = np.mean([s[0] for s in hsv_samples]) if hsv_samples else 0
        avg_s = np.mean([s[1] for s in hsv_samples]) if hsv_samples else 0
        avg_v = np.mean([s[2] for s in hsv_samples]) if hsv_samples else 0
        avg_rr = np.mean([s[3] for s in hsv_samples]) if hsv_samples else 0
        avg_gr = np.mean([s[4] for s in hsv_samples]) if hsv_samples else 0

        vote_str = ", ".join(f"{k}:{v}" for k, v in vote_counts.most_common())

        print(f"  Track {tid:>3} | {best_color:<10} | "
              f"Votes: {vote_str:<35} | "
              f"avgH={avg_h:>5.1f} avgS={avg_s:>5.1f} avgV={avg_v:>5.1f} | "
              f"Red={avg_rr*100:>4.1f}% Gray={avg_gr*100:>4.1f}%")

    print("=" * 90)

    # ==========================================================
    # 6. Potensi ID switching
    # ==========================================================
    print("\n" + "=" * 90)
    print("  POTENSI ID SWITCHING (track hilang → track baru muncul)")
    print("=" * 90)

    sorted_tids = sorted(track_info.keys(),
                         key=lambda t: track_info[t]['first_frame'])
    switch_found = False
    for i in range(len(sorted_tids)):
        for j in range(i + 1, len(sorted_tids)):
            tid_old = sorted_tids[i]
            tid_new = sorted_tids[j]

            old_last  = track_info[tid_old]['last_frame']
            new_first = track_info[tid_new]['first_frame']

            gap = new_first - old_last

            if 0 <= gap <= 15:
                old_x = np.mean(track_info[tid_old]['positions_x'][-10:])
                old_y = np.mean(track_info[tid_old]['positions_y'][-10:])
                new_x = np.mean(track_info[tid_new]['positions_x'][:10])
                new_y = np.mean(track_info[tid_new]['positions_y'][:10])

                dist = np.sqrt((old_x - new_x)**2 + (old_y - new_y)**2)

                if dist < 300:
                    switch_found = True
                    print(f"  Track {tid_old:>3} (hilang frame {old_last:>5}) → "
                          f"Track {tid_new:>3} (muncul frame {new_first:>5}) | "
                          f"gap={gap:>2} frames | jarak={dist:>5.0f}px "
                          f"← KEMUNGKINAN SAMA!")

    if not switch_found:
        print("  Tidak ada potensi ID switching terdeteksi.")
    print("=" * 90)

    # ==========================================================
    # 7. Ringkasan: Track ID mana yang aktif di setiap kick
    # ==========================================================
    print("\n" + "=" * 70)
    print("  RINGKASAN: TRACK ID TERDEKAT BOLA DI SETIAP KICK")
    print("  (Gunakan ini untuk mapping manual)")
    print("=" * 70)

    for kick_idx, kick_frame in enumerate(KICK_FRAMES):
        ball_data = tracks['ball'][kick_frame].get(1)
        if not ball_data:
            print(f"  Kick #{kick_idx+1} @ frame {kick_frame}: Bola tidak terdeteksi")
            continue

        ball_pos = (
            (ball_data['bbox'][0] + ball_data['bbox'][2]) / 2,
            (ball_data['bbox'][1] + ball_data['bbox'][3]) / 2
        )

        # Cari player terdekat di window sebelum kick
        candidates = []
        for f in range(max(0, kick_frame - 30), kick_frame + 1):
            for tid, pdata in tracks['players'][f].items():
                bbox = pdata.get('bbox')
                if bbox is None:
                    continue
                cx = (bbox[0] + bbox[2]) / 2
                foot_y = bbox[3]
                dist = np.sqrt(
                    (cx - ball_pos[0])**2 + (foot_y - ball_pos[1])**2
                )
                candidates.append((tid, f, dist, bbox))

        # Sort by distance, ambil top 3
        candidates.sort(key=lambda x: x[2])
        seen_tids = set()
        top_candidates = []
        for tid, f, dist, bbox in candidates:
            if tid not in seen_tids and len(top_candidates) < 3:
                seen_tids.add(tid)

                if f < len(frames):
                    label, _, _, _, _, _ = detect_shirt_color_detailed(
                        frames[f], bbox
                    )
                else:
                    label = "N/A"

                top_candidates.append((tid, f, dist, label))

        print(f"\n  Kick #{kick_idx+1} @ frame {kick_frame}:")
        for rank, (tid, f, dist, label) in enumerate(top_candidates):
            marker = " ← PENENDANG" if rank == 0 else ""
            print(f"    #{rank+1} Track {tid:>3} (frame {f:>5}) | "
                  f"dist={dist:>6.0f}px | warna={label:<10}{marker}")

    print("\n" + "=" * 70)
    print("\n  PETUNJUK MAPPING MANUAL:")
    print("  Setelah melihat hasil di atas + video debug,")
    print("  tambahkan di CONFIG main.py:")
    print()
    print('    "manual_jersey_mapping": {')
    print('        5: "Merah",    # contoh')
    print('        6: "Abu-Abu",  # contoh')
    print('        9: "Merah",    # contoh')
    print('        ...dst')
    print('    }')
    print()

    # ==========================================================
    # 8. Render video debug
    # ==========================================================
    print(f"\nMerender video debug ke: {OUTPUT_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    for frame_num in range(len(frames)):
        frame = frames[frame_num].copy()

        if frame_num < total_frames:
            # Gambar semua player dengan RAW track ID + warna terdeteksi
            for track_id, data in tracks['players'][frame_num].items():
                bbox = data.get('bbox')
                if bbox is None:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                color = get_color(track_id)

                # Deteksi warna baju
                label, mh, ms, mv, rr, gr = detect_shirt_color_detailed(
                    frame, bbox
                )

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label: ID + warna terdeteksi
                id_label = f"ID:{track_id} [{label}]"
                font_scale = 0.6
                thickness  = 2
                (lw, lh_text), _ = cv2.getTextSize(
                    id_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                cv2.rectangle(frame,
                              (x1, y1 - lh_text - 12),
                              (x1 + lw + 8, y1),
                              color, -1)
                cv2.putText(frame, id_label,
                            (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), thickness)

                # Info HSV kecil di bawah
                hsv_label = f"R:{rr*100:.0f}% G:{gr*100:.0f}%"
                cv2.putText(frame, hsv_label,
                            (x1 + 2, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (200, 200, 200), 1)

                # Titik di kaki
                foot_x = (x1 + x2) // 2
                foot_y = y2
                cv2.circle(frame, (foot_x, foot_y), 5, color, -1)

            # Gambar bola
            ball_data = tracks['ball'][frame_num].get(1)
            if ball_data:
                bx1, by1, bx2, by2 = map(int, ball_data['bbox'])
                bcx = (bx1 + bx2) // 2
                bcy = (by1 + by2) // 2
                cv2.circle(frame, (bcx, bcy), 12, (0, 255, 255), 2)
                cv2.putText(frame, "BALL",
                            (bcx - 15, bcy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255), 1)

            # Gambar gawang
            gawang_data = tracks['gawang'][frame_num].get(1)
            if gawang_data and 'bbox' in gawang_data:
                gx1, gy1, gx2, gy2 = map(int, gawang_data['bbox'])
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2),
                              (0, 200, 255), 2)
                cv2.putText(frame, "GAWANG",
                            (gx1 + 5, gy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 200, 255), 1)

            # Gambar keeper
            keeper_data = tracks['keeper'][frame_num].get(1)
            if keeper_data and 'bbox' in keeper_data:
                kx1, ky1, kx2, ky2 = map(int, keeper_data['bbox'])
                cv2.rectangle(frame, (kx1, ky1), (kx2, ky2),
                              (0, 165, 255), 2)
                cv2.putText(frame, "KEEPER",
                            (kx1 + 2, ky1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 165, 255), 1)

            # Indikator KICK FRAME
            if frame_num in KICK_FRAMES:
                kick_idx = KICK_FRAMES.index(frame_num)
                cv2.putText(frame,
                            f">>> KICK #{kick_idx + 1} <<<",
                            (w // 2 - 100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 3)

            # Indikator di sekitar kick frame
            for kf in KICK_FRAMES:
                if abs(frame_num - kf) <= 30 and frame_num != kf:
                    kick_idx = KICK_FRAMES.index(kf)
                    cv2.putText(frame,
                                f"Near Kick #{kick_idx + 1} (f{kf})",
                                (w // 2 - 80, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 200, 255), 1)

            # Frame label
            cv2.putText(frame, f"Frame: {frame_num}  |  Time: {frame_num/fps:.2f}s",
                        (w - 300, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Active track IDs panel
            active_ids = sorted(tracks['players'][frame_num].keys())
            y_offset = 30
            cv2.putText(frame, "Active Tracks:",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += 20
            for tid in active_ids:
                tc = get_color(tid)
                cv2.putText(frame, f"  ID:{tid}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, tc, 1)
                y_offset += 18

        out.write(frame)

        if frame_num % 200 == 0:
            print(f"  Rendering: {frame_num}/{len(frames)}...")

    out.release()

    print(f"\nVideo debug disimpan: {OUTPUT_PATH}")
    print(f"Crop baju disimpan di: {CROP_DIR}/")
    print(f"Total frame: {len(frames)}")
    print("\n>>> LANGKAH SELANJUTNYA:")
    print("    1. Tonton video debug untuk identifikasi pemain")
    print("    2. Lihat crop baju di folder shirt_crops/")
    print("    3. Update manual_jersey_mapping di main.py")
    print("    4. Jalankan main.py lagi\n")


if __name__ == "__main__":
    main()
