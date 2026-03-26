# draw_dribble.py
# Fungsi visualisasi lengkap untuk dribbling analytics:
#   - Zona radius cone (idle / aktif / tersentuh)
#   - Trajectory bola (fading trail)
#   - Status indicator dribble (pojok kanan atas)
#   - Result flash SUKSES/GAGAL (tengah layar)
#   - Stats panel modern (pojok kiri atas)
#   - Entry/Exit zone indicator
#   - Dribble attempt arrow path

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any


# ============================================================
# PALET WARNA GLOBAL
# ============================================================

COLOR_IDLE       = (0, 220, 255)     # Kuning-cyan (cone idle)
COLOR_SAFE       = (0, 220, 100)     # Hijau (cone saat dribbling, belum disentuh)
COLOR_TOUCHED    = (0, 0, 230)       # Merah (cone disentuh bola)
COLOR_ENTRY      = (100, 255, 100)   # Hijau terang (label entry)
COLOR_EXIT       = (100, 100, 255)   # Biru terang (label exit)
COLOR_SUCCESS    = (0, 180, 0)       # Hijau (flash sukses)
COLOR_FAIL       = (0, 0, 200)       # Merah (flash gagal)
WHITE            = (255, 255, 255)
BLACK            = (0, 0, 0)
GRAY_LIGHT       = (200, 200, 200)
GRAY_MED         = (140, 140, 140)
GRAY_DIM         = (70, 70, 70)

# Panel colors
BG_DARK          = (25, 25, 30)
BG_HEADER        = (50, 42, 35)
ACCENT           = (230, 200, 0)
GREEN            = (80, 220, 60)
RED              = (70, 70, 230)
YELLOW           = (50, 210, 255)
BAR_BG           = (50, 50, 55)


# ============================================================
# HELPER: ROUNDED RECTANGLE
# ============================================================

def draw_rounded_rect(
    img      : np.ndarray,
    pt1      : Tuple[int, int],
    pt2      : Tuple[int, int],
    color    : Tuple[int, int, int],
    radius   : int,
    thickness: int = -1
) -> None:
    """
    Gambar persegi panjang dengan sudut membulat.

    Args:
        img      : frame target
        pt1      : sudut kiri atas (x1, y1)
        pt2      : sudut kanan bawah (x2, y2)
        color    : warna BGR
        radius   : radius sudut bulat (pixel)
        thickness: -1 = filled, > 0 = border saja
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Pastikan dimensi valid
    if x2 <= x1 or y2 <= y1:
        return

    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return

    if thickness == -1:
        # Filled rounded rectangle
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        # Border only
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


# ============================================================
# HELPER: TEXT DENGAN BACKGROUND
# ============================================================

def draw_text_with_bg(
    img       : np.ndarray,
    text      : str,
    position  : Tuple[int, int],
    font_scale: float = 0.5,
    thickness : int = 1,
    text_color: Tuple[int, int, int] = WHITE,
    bg_color  : Tuple[int, int, int] = BLACK,
    padding   : int = 4,
    radius    : int = 4
) -> None:
    """Gambar teks dengan background rounded rectangle."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = position
    bg_pt1 = (x - padding, y - th - padding)
    bg_pt2 = (x + tw + padding, y + baseline + padding)

    draw_rounded_rect(img, bg_pt1, bg_pt2, bg_color, radius=radius)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)


# ============================================================
# DRAW CONE ZONES
# ============================================================

def draw_cone_zones_on_frame(
    frame           : np.ndarray,
    stabilized_cones: Dict[int, Tuple[float, float]],
    cone_radii      : Dict[int, float],
    ordered_ids     : List[int],
    touched_cones   : Optional[List[int]] = None,
    is_dribbling    : bool = False,
    alpha           : float = 0.20,
    show_entry_exit : bool = True,
    show_radius_label: bool = True,
    show_distance_lines: bool = False,
    ball_pos        : Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Gambar zona radius di sekitar setiap cone.

    Warna zona:
    - Kuning/cyan : cone idle (tidak sedang dribbling)
    - Hijau       : cone saat dribbling aktif, bola belum sentuh
    - Merah       : bola menyentuh cone ini (masuk radius)

    Args:
        frame            : frame video BGR
        stabilized_cones : posisi cone {cone_id: (x, y)}
        cone_radii       : radius per cone {cone_id: radius_px}
        ordered_ids      : urutan cone dari entry → exit
        touched_cones    : list cone_id yang bola sudah sentuh
        is_dribbling     : True jika sedang ada dribble attempt aktif
        alpha            : transparansi zona fill
        show_entry_exit  : tampilkan label ENTRY/EXIT
        show_radius_label: tampilkan label radius (r=XXpx)
        show_distance_lines: gambar garis putus-putus antar cone berurutan
        ball_pos         : posisi bola saat ini (untuk garis jarak ke cone terdekat)

    Returns:
        Frame dengan zona cone tergambar
    """
    output = frame.copy()
    overlay = output.copy()

    if touched_cones is None:
        touched_cones = []

    # ---- Gambar garis penghubung antar cone (opsional) ----
    if show_distance_lines and len(ordered_ids) >= 2:
        for i in range(len(ordered_ids) - 1):
            cid_a = ordered_ids[i]
            cid_b = ordered_ids[i + 1]
            if cid_a in stabilized_cones and cid_b in stabilized_cones:
                pos_a = stabilized_cones[cid_a]
                pos_b = stabilized_cones[cid_b]
                pt_a = (int(pos_a[0]), int(pos_a[1]))
                pt_b = (int(pos_b[0]), int(pos_b[1]))
                # Garis putus-putus (simulasi dengan titik-titik)
                _draw_dashed_line(output, pt_a, pt_b, GRAY_DIM, thickness=1, gap=8)

    # ---- Gambar setiap cone ----
    for idx, cone_id in enumerate(ordered_ids):
        if cone_id not in stabilized_cones:
            continue

        pos = stabilized_cones[cone_id]
        cx, cy = int(pos[0]), int(pos[1])
        radius = int(cone_radii.get(cone_id, 40))

        # Tentukan warna berdasarkan status
        if cone_id in touched_cones:
            color = COLOR_TOUCHED
        elif is_dribbling:
            color = COLOR_SAFE
        else:
            color = COLOR_IDLE

        # --- Zona semi-transparan (di overlay) ---
        cv2.circle(overlay, (cx, cy), radius, color, -1)

        # --- Border zona ---
        cv2.circle(output, (cx, cy), radius, color, 2)

        # --- Ring dalam (memberikan efek depth) ---
        inner_r = max(radius - 8, 5)
        cv2.circle(output, (cx, cy), inner_r, color, 1)

        # --- Crosshair di pusat ---
        ll = 12
        cv2.line(output, (cx - ll, cy), (cx + ll, cy), color, 1)
        cv2.line(output, (cx, cy - ll), (cx, cy + ll), color, 1)

        # --- Titik pusat cone ---
        cv2.circle(output, (cx, cy), 5, color, -1)
        cv2.circle(output, (cx, cy), 7, WHITE, 1)

        # --- Label nomor urut cone ---
        label = f"C{idx + 1}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_x = cx - lw // 2
        label_y = cy - radius - 10

        # Background untuk label
        cv2.rectangle(output,
                      (label_x - 3, label_y - lh - 2),
                      (label_x + lw + 3, label_y + 3),
                      (30, 30, 30), -1)
        cv2.putText(output, label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # --- Label radius ---
        if show_radius_label:
            r_label = f"r={radius}px"
            (rw, _), _ = cv2.getTextSize(r_label, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            cv2.putText(output, r_label,
                        (cx - rw // 2, cy + radius + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, GRAY_MED, 1)

        # --- X merah besar jika disentuh ---
        if cone_id in touched_cones:
            x_size = min(radius // 2, 18)
            cv2.line(output,
                     (cx - x_size, cy - x_size),
                     (cx + x_size, cy + x_size),
                     COLOR_TOUCHED, 3)
            cv2.line(output,
                     (cx + x_size, cy - x_size),
                     (cx - x_size, cy + x_size),
                     COLOR_TOUCHED, 3)

            # Label "HIT!" di bawah X
            hit_label = "HIT!"
            (hw, _), _ = cv2.getTextSize(hit_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.putText(output, hit_label,
                        (cx - hw // 2, cy + x_size + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TOUCHED, 1)

    # ---- Label Entry & Exit ----
    if show_entry_exit and len(ordered_ids) >= 2:
        # Entry cone
        entry_id = ordered_ids[0]
        if entry_id in stabilized_cones:
            epos = stabilized_cones[entry_id]
            er = int(cone_radii.get(entry_id, 40))
            ecx, ecy = int(epos[0]), int(epos[1])
            draw_text_with_bg(output, "ENTRY",
                              (ecx - 22, ecy - er - 26),
                              font_scale=0.40, thickness=1,
                              text_color=BLACK, bg_color=COLOR_ENTRY,
                              padding=4, radius=4)

        # Exit cone
        exit_id = ordered_ids[-1]
        if exit_id in stabilized_cones:
            xpos = stabilized_cones[exit_id]
            xr = int(cone_radii.get(exit_id, 40))
            xcx, xcy = int(xpos[0]), int(xpos[1])
            draw_text_with_bg(output, "EXIT",
                              (xcx - 18, xcy - xr - 26),
                              font_scale=0.40, thickness=1,
                              text_color=BLACK, bg_color=COLOR_EXIT,
                              padding=4, radius=4)

    # ---- Garis dari bola ke cone terdekat (opsional, untuk debug) ----
    if ball_pos is not None and stabilized_cones:
        bx, by = int(ball_pos[0]), int(ball_pos[1])
        min_dist = float('inf')
        closest_cone_pos = None
        closest_cone_id = None

        for cid, cpos in stabilized_cones.items():
            d = np.sqrt((bx - cpos[0])**2 + (by - cpos[1])**2)
            if d < min_dist:
                min_dist = d
                closest_cone_pos = cpos
                closest_cone_id = cid

        if closest_cone_pos and min_dist < 200:
            ccx, ccy = int(closest_cone_pos[0]), int(closest_cone_pos[1])
            r = cone_radii.get(closest_cone_id, 40)
            dist_color = COLOR_TOUCHED if min_dist <= r else (200, 200, 200)
            _draw_dashed_line(output, (bx, by), (ccx, ccy), dist_color, 1, 6)

            # Label jarak
            mid_x = (bx + ccx) // 2
            mid_y = (by + ccy) // 2
            dist_text = f"{min_dist:.0f}px"
            cv2.putText(output, dist_text,
                        (mid_x + 5, mid_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, dist_color, 1)

    # Apply overlay dengan alpha blending
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def _draw_dashed_line(
    img      : np.ndarray,
    pt1      : Tuple[int, int],
    pt2      : Tuple[int, int],
    color    : Tuple[int, int, int],
    thickness: int = 1,
    gap      : int = 8
) -> None:
    """Gambar garis putus-putus antara dua titik."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    if length == 0:
        return

    segments = int(length / gap)
    if segments == 0:
        segments = 1

    for i in range(0, segments, 2):
        t1 = i / segments
        t2 = min((i + 1) / segments, 1.0)
        sx = int(x1 + dx * t1)
        sy = int(y1 + dy * t1)
        ex = int(x1 + dx * t2)
        ey = int(y1 + dy * t2)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


# ============================================================
# DRAW ENTRY/EXIT ZONE CIRCLES (opsional, untuk debug)
# ============================================================

def draw_entry_exit_zones(
    frame              : np.ndarray,
    stabilized_cones   : Dict[int, Tuple[float, float]],
    ordered_ids        : List[int],
    entry_exit_radius  : float = 150.0,
    alpha              : float = 0.10
) -> np.ndarray:
    """
    Gambar lingkaran besar semi-transparan di sekitar cone entry dan exit
    untuk menunjukkan zona trigger dribble attempt.
    """
    if len(ordered_ids) < 2:
        return frame

    output = frame.copy()
    overlay = output.copy()

    # Entry zone
    entry_id = ordered_ids[0]
    if entry_id in stabilized_cones:
        epos = stabilized_cones[entry_id]
        ecx, ecy = int(epos[0]), int(epos[1])
        r = int(entry_exit_radius)
        cv2.circle(overlay, (ecx, ecy), r, COLOR_ENTRY, -1)
        cv2.circle(output, (ecx, ecy), r, COLOR_ENTRY, 1)

    # Exit zone
    exit_id = ordered_ids[-1]
    if exit_id in stabilized_cones:
        xpos = stabilized_cones[exit_id]
        xcx, xcy = int(xpos[0]), int(xpos[1])
        r = int(entry_exit_radius)
        cv2.circle(overlay, (xcx, xcy), r, COLOR_EXIT, -1)
        cv2.circle(output, (xcx, xcy), r, COLOR_EXIT, 1)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


# ============================================================
# DRAW DRIBBLE STATUS INDICATOR (pojok kanan atas)
# ============================================================

def draw_dribble_status(
    frame       : np.ndarray,
    is_active   : bool,
    touched_cnt : int = 0,
    total_cones : int = 0,
    duration_sec: float = 0.0,
) -> np.ndarray:
    """
    Tampilkan indikator status dribble real-time di pojok kanan atas.

    Menunjukkan:
    - Status: DRIBBLING... / CONE HIT
    - Jumlah cone yang disentuh
    - Durasi dribble saat ini
    """
    output = frame.copy()
    if not is_active:
        return output

    h, w = frame.shape[:2]
    box_w = 250
    box_h = 70
    px = w - box_w - 20
    py = 20

    # Background semi-transparan
    overlay = output.copy()
    draw_rounded_rect(overlay, (px, py), (px + box_w, py + box_h),
                      BG_DARK, radius=10)
    cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)

    # Border
    draw_rounded_rect(output, (px, py), (px + box_w, py + box_h),
                      GRAY_DIM, radius=10, thickness=1)

    # Status text
    if touched_cnt == 0:
        status_color = GREEN
        status_text = "DRIBBLING..."
        # Animasi: titik-titik berkedip berdasarkan frame
    else:
        status_color = RED
        status_text = f"CONE HIT: {touched_cnt}/{total_cones}"

    # Indicator dot (bulatan kecil berkedip)
    dot_x = px + 16
    dot_y = py + 22
    cv2.circle(output, (dot_x, dot_y), 6, status_color, -1)
    cv2.circle(output, (dot_x, dot_y), 8, status_color, 1)

    # Status text
    cv2.putText(output, status_text,
                (dot_x + 14, py + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2)

    # Durasi
    dur_text = f"Durasi: {duration_sec:.1f}s"
    cv2.putText(output, dur_text,
                (px + 16, py + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY_LIGHT, 1)

    # Cone status mini (titik-titik hijau/merah untuk setiap cone)
    cone_dot_start_x = px + 120
    cone_dot_y = py + 52
    for i in range(total_cones):
        dot_cx = cone_dot_start_x + i * 16
        if dot_cx > px + box_w - 10:
            break
        dot_color = RED if i < touched_cnt else GREEN
        cv2.circle(output, (dot_cx, cone_dot_y), 5, dot_color, -1)
        cv2.circle(output, (dot_cx, cone_dot_y), 6, WHITE, 1)

    return output


# ============================================================
# DRAW BALL TRAJECTORY
# ============================================================

def draw_ball_trajectory_on_frame(
    frame     : np.ndarray,
    trajectory: List[Tuple[int, int]],
    color     : Tuple[int, int, int] = (255, 120, 0),
    max_points: int = 50,
    fade      : bool = True
) -> np.ndarray:
    """
    Gambar trail trajectory bola dengan efek fading.

    Titik-titik lama lebih redup dan kecil, titik terbaru paling terang.
    Garis penghubung antar titik juga digambar.

    Args:
        frame     : frame video
        trajectory: list posisi bola [(x,y), ...]
        color     : warna dasar trail
        max_points: maksimum titik yang ditampilkan
        fade      : True = efek fading, False = warna seragam
    """
    output = frame.copy()
    recent = trajectory[-max_points:] if len(trajectory) > max_points else trajectory
    n = len(recent)
    if n == 0:
        return output

    for i, point in enumerate(recent):
        px, py = int(point[0]), int(point[1])

        if fade and n > 1:
            ratio = (i + 1) / n
            faded = tuple(int(c * ratio) for c in color)
            r = max(2, int(5 * ratio))
        else:
            faded = color
            r = 4

        cv2.circle(output, (px, py), r, faded, -1)

        if i > 0:
            prev = (int(recent[i - 1][0]), int(recent[i - 1][1]))
            line_thick = max(1, int(2 * ((i + 1) / n))) if fade else 1
            cv2.line(output, prev, (px, py), faded, line_thick)

    # Highlight titik terakhir (posisi bola saat ini)
    last = (int(recent[-1][0]), int(recent[-1][1]))
    cv2.circle(output, last, 8, color, 2)
    cv2.circle(output, last, 3, WHITE, -1)

    return output


# ============================================================
# DRAW RESULT FLASH (SUKSES / GAGAL)
# ============================================================

def draw_result_flash(
    frame         : np.ndarray,
    success       : bool,
    touched_cones : List[int],
    total_cones   : int,
    alpha         : float = 0.55,
    attempt_number: Optional[int] = None
) -> np.ndarray:
    """
    Tampilkan flash overlay besar di tengah layar setelah attempt selesai.

    - SUKSES: background hijau, teks "SUKSES"
    - GAGAL : background merah, teks "GAGAL" + jumlah cone tersentuh

    Args:
        frame          : frame video
        success        : True = sukses, False = gagal
        touched_cones  : list cone_id yang tersentuh
        total_cones    : total cone di course
        alpha          : transparansi background
        attempt_number : nomor attempt (opsional)
    """
    output = frame.copy()
    h, w = output.shape[:2]
    overlay = output.copy()

    if success:
        bg_color = COLOR_SUCCESS
        main_text = "SUKSES"
        sub_text = "Tidak ada cone tersentuh!"
    else:
        bg_color = COLOR_FAIL
        main_text = "GAGAL"
        n_touched = len(touched_cones)
        sub_text = f"{n_touched} dari {total_cones} cone tersentuh"

    # Box di tengah layar
    box_w = 380
    box_h = 120
    bx1 = (w - box_w) // 2
    by1 = (h - box_h) // 2
    bx2 = bx1 + box_w
    by2 = by1 + box_h

    # Background box
    draw_rounded_rect(overlay, (bx1, by1), (bx2, by2), bg_color, radius=18)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Border putih
    draw_rounded_rect(output, (bx1, by1), (bx2, by2), WHITE, radius=18, thickness=2)

    # Attempt number (jika ada)
    if attempt_number is not None:
        attempt_text = f"Attempt #{attempt_number}"
        (aw, _), _ = cv2.getTextSize(attempt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(output, attempt_text,
                    ((w - aw) // 2, by1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)

    # Main text (besar)
    (tw, th), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
    text_y = by1 + 65 if attempt_number else by1 + 55
    cv2.putText(output, main_text,
                ((w - tw) // 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 3)

    # Sub text
    (sw, _), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(output, sub_text,
                ((w - sw) // 2, text_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    # Ikon check/cross di samping teks
    icon_x = (w - tw) // 2 - 40
    icon_y = text_y - 15
    if success:
        # Checkmark
        cv2.line(output, (icon_x, icon_y), (icon_x + 10, icon_y + 12), WHITE, 3)
        cv2.line(output, (icon_x + 10, icon_y + 12), (icon_x + 28, icon_y - 8), WHITE, 3)
    else:
        # Cross
        cv2.line(output, (icon_x, icon_y - 8), (icon_x + 20, icon_y + 12), WHITE, 3)
        cv2.line(output, (icon_x + 20, icon_y - 8), (icon_x, icon_y + 12), WHITE, 3)

    return output


# ============================================================
# DRAW DRIBBLE STATS PANEL (pojok kiri atas)
# ============================================================

def draw_dribble_stats_panel(
    frame        : np.ndarray,
    stats        : dict,
    cone_analysis: Optional[Dict] = None,
    ordered_ids  : Optional[List[int]] = None,
    position     : Tuple[int, int] = (20, 20),
    panel_width  : int = 300
) -> np.ndarray:
    """
    Panel statistik dribbling dengan desain modern.

    Menampilkan:
    - Header: DRIBBLE ANALYTICS
    - Summary: Total / Sukses / Gagal (angka besar, 3 kolom)
    - Progress bar akurasi
    - Rata-rata durasi
    - Detail per cone (jika cone_analysis diberikan)

    Args:
        frame        : frame video BGR
        stats        : dict statistik dari compute_progressive_stats()
        cone_analysis: dict analisis per cone (opsional, dari get_dribble_statistics)
        ordered_ids  : urutan cone IDs (untuk label C1, C2, ...)
        position     : posisi kiri atas panel (x, y)
        panel_width  : lebar panel dalam pixel

    Returns:
        Frame dengan panel statistik
    """
    output = frame.copy()
    px, py = position
    pad = 14

    # ---- Hitung dimensi panel ----
    cone_rows = cone_analysis or {}
    n_cone_rows = min(len(cone_rows), 8)  # Max 8 cone ditampilkan

    header_h = 44
    summary_h = 70
    accuracy_h = 42
    duration_h = 28
    divider_pad = 10
    cone_row_h = 22
    cone_header_h = 18
    cone_sec_h = (divider_pad + cone_header_h + cone_row_h * n_cone_rows) if n_cone_rows > 0 else 0
    bottom_pad = 10
    total_h = header_h + summary_h + accuracy_h + duration_h + cone_sec_h + bottom_pad

    x1, y1 = px, py
    x2, y2 = px + panel_width, py + total_h

    # Clamp agar tidak keluar frame
    fh, fw = frame.shape[:2]
    if x2 > fw:
        x2 = fw - 5
    if y2 > fh:
        y2 = fh - 5

    # ============== BACKGROUND ==============
    overlay = output.copy()
    draw_rounded_rect(overlay, (x1, y1), (x2, y2), BG_DARK, radius=14)
    cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
    draw_rounded_rect(output, (x1, y1), (x2, y2), GRAY_DIM, radius=14, thickness=1)

    # ============== HEADER ==============
    overlay_h = output.copy()
    draw_rounded_rect(overlay_h,
                      (x1 + 1, y1 + 1),
                      (x2 - 1, y1 + header_h),
                      BG_HEADER, radius=13)
    cv2.rectangle(overlay_h,
                  (x1 + 1, y1 + header_h - 13),
                  (x2 - 1, y1 + header_h),
                  BG_HEADER, -1)
    cv2.addWeighted(overlay_h, 0.95, output, 0.05, 0, output)

    # Garis aksen di bawah header
    cv2.line(output, (x1 + pad, y1 + header_h),
             (x2 - pad, y1 + header_h), ACCENT, 2)

    # Ikon bola
    icon_cx = x1 + pad + 11
    icon_cy = y1 + header_h // 2
    cv2.circle(output, (icon_cx, icon_cy), 11, YELLOW, -1)
    cv2.circle(output, (icon_cx, icon_cy), 11, WHITE, 1)
    cv2.circle(output, (icon_cx, icon_cy), 4, (30, 30, 30), -1)

    # Judul
    cv2.putText(output, "DRIBBLE ANALYTICS",
                (icon_cx + 18, y1 + header_h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 2)

    # ============== SUMMARY: 3 KOLOM ==============
    row_y = y1 + header_h + 8
    col_w = (panel_width - 2 * pad) // 3

    total_a = stats.get('total_attempts', 0)
    succ_a = stats.get('successful_attempts', 0)
    fail_a = stats.get('failed_attempts', 0)

    items = [
        (total_a, "Total",  ACCENT),
        (succ_a,  "Sukses", GREEN),
        (fail_a,  "Gagal",  RED),
    ]

    for idx, (val, label, clr) in enumerate(items):
        col_cx = x1 + pad + idx * col_w + col_w // 2

        # Angka besar
        val_str = str(val)
        (vw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(output, val_str,
                    (col_cx - vw // 2, row_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, clr, 2)

        # Sub-label
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        cv2.putText(output, label,
                    (col_cx - lw // 2, row_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, GRAY_MED, 1)

    # Garis vertikal pemisah kolom
    for idx in range(1, 3):
        div_x = x1 + pad + idx * col_w
        cv2.line(output, (div_x, row_y + 8), (div_x, row_y + 48), GRAY_DIM, 1)

    # ============== PROGRESS BAR AKURASI ==============
    acc_y = row_y + summary_h
    acc_pct = stats.get('accuracy_pct', 0.0)

    # Label "Akurasi"
    cv2.putText(output, "Akurasi",
                (x1 + pad, acc_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, GRAY_LIGHT, 1)

    # Warna adaptif berdasarkan persentase
    if acc_pct >= 70:
        pct_color = GREEN
    elif acc_pct >= 40:
        pct_color = YELLOW
    else:
        pct_color = RED

    # Persentase (rata kanan)
    pct_str = f"{acc_pct:.0f}%"
    (pw, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
    cv2.putText(output, pct_str,
                (x2 - pad - pw, acc_y + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, pct_color, 2)

    # Bar background
    bar_x1 = x1 + pad
    bar_x2 = x2 - pad
    bar_y1 = acc_y + 12
    bar_y2 = bar_y1 + 14
    bar_w = bar_x2 - bar_x1

    draw_rounded_rect(output, (bar_x1, bar_y1), (bar_x2, bar_y2), BAR_BG, radius=7)

    # Bar fill
    fill_w = int(bar_w * min(acc_pct / 100.0, 1.0))
    if fill_w > 5:
        draw_rounded_rect(output,
                          (bar_x1, bar_y1),
                          (bar_x1 + fill_w, bar_y2),
                          pct_color, radius=7)

    # ============== DURASI RATA-RATA ==============
    dur_y = bar_y2 + 8
    avg_dur = stats.get('avg_duration', 0.0)
    cv2.putText(output, f"Avg durasi: {avg_dur:.1f}s",
                (x1 + pad, dur_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY_MED, 1)

    # ============== PER-CONE ANALYSIS ==============
    current_y = dur_y + duration_h

    if cone_rows and ordered_ids:
        # Garis pemisah
        cv2.line(output, (x1 + pad, current_y),
                 (x2 - pad, current_y), GRAY_DIM, 1)
        current_y += 4

        # Sub-judul
        cv2.putText(output, "Per Cone",
                    (x1 + pad, current_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, GRAY_MED, 1)
        current_y += cone_header_h

        for idx_c, cid in enumerate(ordered_ids):
            if idx_c >= 8:  # Max 8 cone
                remaining = len(ordered_ids) - 8
                cv2.putText(output, f"  +{remaining} more...",
                            (x1 + pad + 6, current_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, GRAY_DIM, 1)
                break

            if cid not in cone_rows:
                continue

            ca = cone_rows[cid]
            hits = ca.get('times_touched', 0)
            avg_d = ca.get('avg_min_distance', 0.0)
            r = ca.get('radius', 0.0)

            # Warna berdasarkan hit count
            if hits > 0:
                hit_color = RED
                indicator = "X"
            else:
                hit_color = GREEN
                indicator = "O"

            # Indicator
            cv2.putText(output, indicator,
                        (x1 + pad + 4, current_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, hit_color, 1)

            # Cone info
            line = f"C{idx_c+1}: hit={hits}x  min={avg_d:.0f}px  r={r:.0f}"
            cv2.putText(output, line,
                        (x1 + pad + 18, current_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, hit_color, 1)

            current_y += cone_row_h

    return output


# ============================================================
# DRAW ATTEMPT SUMMARY TABLE (untuk frame akhir / screenshot)
# ============================================================

def draw_attempt_summary(
    frame   : np.ndarray,
    attempts: List[Dict],
    position: Tuple[int, int] = (20, 20)
) -> np.ndarray:
    """
    Gambar tabel ringkasan semua attempt di frame.
    Cocok untuk frame terakhir video atau screenshot.
    """
    if not attempts:
        return frame

    output = frame.copy()
    px, py = position

    row_h = 28
    header_h = 40
    col_widths = [40, 100, 80, 80, 80]  # No, Frame, Durasi, Cone Hit, Status
    total_w = sum(col_widths) + 20
    total_h = header_h + row_h * len(attempts) + 10

    # Background
    overlay = output.copy()
    draw_rounded_rect(overlay, (px, py), (px + total_w, py + total_h),
                      BG_DARK, radius=10)
    cv2.addWeighted(overlay, 0.90, output, 0.10, 0, output)

    # Header
    draw_rounded_rect(output, (px + 1, py + 1),
                      (px + total_w - 1, py + header_h),
                      BG_HEADER, radius=9)

    headers = ["No", "Frame", "Durasi", "Cone Hit", "Status"]
    cx = px + 10
    for i, (header, cw) in enumerate(zip(headers, col_widths)):
        cv2.putText(output, header,
                    (cx, py + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ACCENT, 1)
        cx += cw

    # Rows
    for i, a in enumerate(attempts):
        ry = py + header_h + i * row_h
        cx = px + 10

        # Alternating row background
        if i % 2 == 0:
            cv2.rectangle(output, (px + 2, ry), (px + total_w - 2, ry + row_h),
                          (35, 35, 40), -1)

        row_data = [
            str(i + 1),
            f"{a['frame_start']}-{a['frame_end']}",
            f"{a['duration_seconds']:.1f}s",
            f"{len(a['touched_cones'])}/{a['total_cones']}",
            "SUKSES" if a['success'] else "GAGAL"
        ]

        for j, (text, cw) in enumerate(zip(row_data, col_widths)):
            clr = GRAY_LIGHT
            if j == 4:  # Status column
                clr = GREEN if a['success'] else RED
            cv2.putText(output, text,
                        (cx, ry + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, clr, 1)
            cx += cw

    return output
