# draw_gate.py
# Fungsi visualisasi untuk gate cone, trajectory bola, dan panel statistik
#
# PERUBAHAN v2.5:
#   - draw_stats_panel() didesain ulang — modern, rounded corners,
#     progress bar, badge jersey, 3-column layout

import cv2
import numpy as np
from typing import Tuple, List, Optional


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
# DRAW GATE
# ============================================================

def draw_gate_on_frame(
    frame: np.ndarray,
    gate_cone_left: Tuple[float, float],
    gate_cone_right: Tuple[float, float],
    is_active: bool = False,
    gate_color_normal: Tuple[int, int, int] = (0, 220, 255),
    gate_color_active: Tuple[int, int, int] = (0, 255, 80),
    alpha: float = 0.30,
    line_thickness: int = 2,
    show_width_label: bool = True
) -> np.ndarray:
    """Gambar visualisasi gate di atas frame video."""
    output = frame.copy()
    color  = gate_color_active if is_active else gate_color_normal

    gl = (int(gate_cone_left[0]),  int(gate_cone_left[1]))
    gr = (int(gate_cone_right[0]), int(gate_cone_right[1]))

    gate_height = 70
    poly_pts = np.array([
        gl, gr,
        (gr[0], gr[1] - gate_height),
        (gl[0], gl[1] - gate_height)
    ], dtype=np.int32)

    overlay = output.copy()
    cv2.fillPoly(overlay, [poly_pts], color)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    cv2.line(output, gl, gr, color, line_thickness + 1)

    cv2.circle(output, gl, 10, color,          -1)
    cv2.circle(output, gr, 10, color,          -1)
    cv2.circle(output, gl, 12, (255, 255, 255), 2)
    cv2.circle(output, gr, 12, (255, 255, 255), 2)

    pillar_height = 35
    cv2.line(output, gl, (gl[0], gl[1] - pillar_height), color, 2)
    cv2.line(output, gr, (gr[0], gr[1] - pillar_height), color, 2)

    mid_x = (gl[0] + gr[0]) // 2
    mid_y = min(gl[1], gr[1]) - gate_height - 12

    label_text = "GATE"
    (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(output,
                  (mid_x - lw // 2 - 4, mid_y - lh - 2),
                  (mid_x + lw // 2 + 4, mid_y + 2),
                  color, -1)
    cv2.putText(output, label_text,
                (mid_x - lw // 2, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if show_width_label:
        gate_width = int(np.linalg.norm(
            np.array(gate_cone_left) - np.array(gate_cone_right)
        ))
        width_text = f"{gate_width}px"
        cv2.putText(output, width_text,
                    (mid_x - 18, mid_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    return output


# ============================================================
# DRAW BALL TRAJECTORY (DEBUG)
# ============================================================

def draw_ball_trajectory_on_frame(
    frame: np.ndarray,
    trajectory: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 120, 0),
    max_points: int = 30,
    fade: bool = True
) -> np.ndarray:
    """Gambar titik-titik trajectory bola untuk debugging."""
    output = frame.copy()

    recent = trajectory[-max_points:] if len(trajectory) > max_points else trajectory
    n = len(recent)

    if n == 0:
        return output

    for i, point in enumerate(recent):
        px, py = int(point[0]), int(point[1])

        if fade and n > 1:
            ratio       = (i + 1) / n
            faded_color = tuple(int(c * ratio) for c in color)
            pt_radius   = max(2, int(5 * ratio))
        else:
            faded_color = color
            pt_radius   = 4

        cv2.circle(output, (px, py), pt_radius, faded_color, -1)

        if i > 0:
            prev = (int(recent[i-1][0]), int(recent[i-1][1]))
            cv2.line(output, prev, (px, py), faded_color, 1)

    last = (int(recent[-1][0]), int(recent[-1][1]))
    cv2.circle(output, last, 7, color, 2)

    return output


# ============================================================
# DRAW PASS ARROW
# ============================================================

def draw_pass_arrow(
    frame: np.ndarray,
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    success: bool,
    from_jersey: str,
    to_jersey: str,
    distance: float
) -> np.ndarray:
    """Gambar panah passing di frame dengan warna dan label status."""
    output = frame.copy()
    color  = (0, 220, 0) if success else (0, 0, 220)
    label  = "SUKSES" if success else "GAGAL"

    fp = (int(from_pos[0]), int(from_pos[1]))
    tp = (int(to_pos[0]),   int(to_pos[1]))

    cv2.arrowedLine(output, fp, tp, color, 3, tipLength=0.12)

    mid_x = (fp[0] + tp[0]) // 2
    mid_y = (fp[1] + tp[1]) // 2

    pass_label = f"#{from_jersey}->#  {to_jersey} | {label}"
    dist_label = f"{distance:.0f}px"

    (lw, lh), _ = cv2.getTextSize(pass_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(output,
                  (mid_x - lw // 2 - 3, mid_y - lh - 16),
                  (mid_x + lw // 2 + 3, mid_y - 6),
                  (30, 30, 30), -1)
    cv2.putText(output, pass_label,
                (mid_x - lw // 2, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(output, dist_label,
                (mid_x - 15, mid_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return output


# ============================================================
# DRAW STATS PANEL v2 — MODERN DESIGN
# ============================================================

def draw_stats_panel(
    frame      : np.ndarray,
    stats      : dict,
    position   : Tuple[int, int] = (20, 20),
    panel_width: int = 320
) -> np.ndarray:
    """
    Panel statistik passing dengan desain modern.

    Fitur:
    - Rounded corners dengan semi-transparansi
    - Header bergaya dengan ikon bola
    - 3-column layout (Total / Sukses / Gagal) angka besar
    - Progress bar akurasi dengan warna adaptif
    - Per-pemain: badge jersey, mini progress bar, avg closest

    Args:
        frame      : frame video BGR
        stats      : dict statistik dari get_pass_statistics()
        position   : posisi kiri atas panel (x, y)
        panel_width: lebar panel dalam pixel

    Returns:
        Frame dengan panel statistik
    """
    output = frame.copy()
    px, py = position
    pad = 14

    # ---- Palet warna ----
    BG_DARK      = (25, 25, 30)
    BG_HEADER    = (50, 42, 35)
    ACCENT       = (230, 200, 0)       # Kuning-cyan (BGR)
    GREEN        = (80, 220, 60)
    RED          = (70, 70, 230)
    YELLOW       = (50, 210, 255)
    WHITE        = (255, 255, 255)
    GRAY_LIGHT   = (200, 200, 200)
    GRAY_MED     = (140, 140, 140)
    GRAY_DIM     = (70, 70, 70)
    BAR_BG       = (50, 50, 55)

    per_player = stats.get('per_player', {})
    n_players  = len(per_player)

    # ---- Hitung dimensi panel ----
    header_h       = 44
    summary_h      = 70
    accuracy_h     = 42
    divider_pad    = 12
    player_row_h   = 50
    player_sec_h   = (divider_pad + 22 + player_row_h * n_players) if n_players > 0 else 0
    bottom_pad     = 12
    total_h        = header_h + summary_h + accuracy_h + player_sec_h + bottom_pad

    x1, y1 = px, py
    x2, y2 = px + panel_width, py + total_h

    # ===========================================================
    # BACKGROUND (semi-transparent, rounded)
    # ===========================================================
    overlay = output.copy()
    draw_rounded_rect(overlay, (x1, y1), (x2, y2), BG_DARK, radius=14)
    cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)

    # Border halus
    draw_rounded_rect(output, (x1, y1), (x2, y2), GRAY_DIM, radius=14, thickness=1)

    # ===========================================================
    # HEADER
    # ===========================================================
    overlay_h = output.copy()
    draw_rounded_rect(overlay_h, (x1 + 1, y1 + 1), (x2 - 1, y1 + header_h), BG_HEADER, radius=13)
    # Ratakan sudut bawah header
    cv2.rectangle(overlay_h, (x1 + 1, y1 + header_h - 13), (x2 - 1, y1 + header_h), BG_HEADER, -1)
    cv2.addWeighted(overlay_h, 0.95, output, 0.05, 0, output)

    # Garis aksen di bawah header
    cv2.line(output, (x1 + pad, y1 + header_h), (x2 - pad, y1 + header_h), ACCENT, 2)

    # Ikon bola
    icon_cx = x1 + pad + 11
    icon_cy = y1 + header_h // 2
    cv2.circle(output, (icon_cx, icon_cy), 11, YELLOW, -1)
    cv2.circle(output, (icon_cx, icon_cy), 11, WHITE, 1)
    cv2.circle(output, (icon_cx, icon_cy), 4, (30, 30, 30), -1)

    # Judul
    cv2.putText(output, "PASSING ANALYTICS",
                (icon_cx + 18, y1 + header_h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, WHITE, 2)

    # ===========================================================
    # SUMMARY: 3 KOLOM (Total / Sukses / Gagal)
    # ===========================================================
    row_y = y1 + header_h + 8
    col_w = (panel_width - 2 * pad) // 3

    total_p = stats.get('total_passes', 0)
    succ_p  = stats.get('successful_passes', 0)
    fail_p  = stats.get('failed_passes', 0)

    items = [
        (total_p, "Total",  ACCENT),
        (succ_p,  "Sukses", GREEN),
        (fail_p,  "Gagal",  RED),
    ]

    for idx, (val, label, color) in enumerate(items):
        col_cx = x1 + pad + idx * col_w + col_w // 2

        # Angka besar
        val_str = str(val)
        (vw, vh), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)
        cv2.putText(output, val_str,
                    (col_cx - vw // 2, row_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2)

        # Sub-label
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(output, label,
                    (col_cx - lw // 2, row_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, GRAY_MED, 1)

    # Garis vertikal pemisah kolom
    for idx in range(1, 3):
        div_x = x1 + pad + idx * col_w
        cv2.line(output, (div_x, row_y + 8), (div_x, row_y + 50), GRAY_DIM, 1)

    # ===========================================================
    # PROGRESS BAR AKURASI
    # ===========================================================
    acc_y   = row_y + summary_h
    acc_pct = stats.get('accuracy_pct', 0.0)

    # Label "Akurasi"
    cv2.putText(output, "Akurasi",
                (x1 + pad, acc_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY_LIGHT, 1)

    # Warna adaptif
    pct_color = GREEN if acc_pct >= 70 else (YELLOW if acc_pct >= 40 else RED)

    # Persentase (rata kanan)
    pct_str = f"{acc_pct:.0f}%"
    (pw, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.putText(output, pct_str,
                (x2 - pad - pw, acc_y + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, pct_color, 2)

    # Bar background
    bar_x1 = x1 + pad
    bar_x2 = x2 - pad
    bar_y1 = acc_y + 12
    bar_y2 = bar_y1 + 16
    bar_total_w = bar_x2 - bar_x1

    draw_rounded_rect(output, (bar_x1, bar_y1), (bar_x2, bar_y2), BAR_BG, radius=8)

    # Bar fill
    fill_w = int(bar_total_w * min(acc_pct / 100.0, 1.0))
    if fill_w > 6:
        draw_rounded_rect(output, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), pct_color, radius=8)

    # ===========================================================
    # PER-PLAYER STATS
    # ===========================================================
    if per_player:
        sec_y = bar_y2 + divider_pad

        # Garis pemisah
        cv2.line(output, (x1 + pad, sec_y), (x2 - pad, sec_y), GRAY_DIM, 1)
        sec_y += 6

        # Sub-judul
        cv2.putText(output, "Per Pemain",
                    (x1 + pad, sec_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, GRAY_MED, 1)
        sec_y += 22

        for jersey, pstats in per_player.items():
            p_total = pstats.get('total', 0)
            p_succ  = pstats.get('success', 0)
            p_acc   = pstats.get('accuracy_pct', 0.0)
            avg_cl  = pstats.get('avg_closest', 0.0)

            # --- Badge jersey (warna berdasarkan nomor) ---
            badge_x = x1 + pad
            badge_y = sec_y + 2
            badge_w = 50
            badge_h = 22
            badge_color = (180, 120, 0) if '19' in jersey else (0, 120, 180)

            overlay_b = output.copy()
            draw_rounded_rect(overlay_b,
                              (badge_x, badge_y),
                              (badge_x + badge_w, badge_y + badge_h),
                              badge_color, radius=6)
            cv2.addWeighted(overlay_b, 0.85, output, 0.15, 0, output)

            jersey_label = f"#{jersey}"
            (jw, _), _ = cv2.getTextSize(jersey_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(output, jersey_label,
                        (badge_x + (badge_w - jw) // 2, badge_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)

            # --- Ratio text (misal "3/4") ---
            ratio_str = f"{p_succ}/{p_total}"
            cv2.putText(output, ratio_str,
                        (badge_x + badge_w + 10, sec_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, GRAY_LIGHT, 1)

            # --- Mini progress bar ---
            mini_bar_x1 = badge_x + badge_w + 55
            mini_bar_x2 = x2 - pad - 48
            mini_bar_y1 = sec_y + 6
            mini_bar_y2 = mini_bar_y1 + 12
            mini_bar_w  = mini_bar_x2 - mini_bar_x1

            draw_rounded_rect(output,
                              (mini_bar_x1, mini_bar_y1),
                              (mini_bar_x2, mini_bar_y2),
                              BAR_BG, radius=6)

            mini_fill  = int(mini_bar_w * min(p_acc / 100.0, 1.0))
            mini_color = GREEN if p_acc >= 70 else (YELLOW if p_acc >= 40 else RED)
            if mini_fill > 4:
                draw_rounded_rect(output,
                                  (mini_bar_x1, mini_bar_y1),
                                  (mini_bar_x1 + mini_fill, mini_bar_y2),
                                  mini_color, radius=6)

            # --- Percentage ---
            p_pct_str = f"{p_acc:.0f}%"
            cv2.putText(output, p_pct_str,
                        (x2 - pad - 40, sec_y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, mini_color, 1)

            # --- Sublabel: avg closest ---
            cl_str = f"avg closest: {avg_cl:.0f}px"
            cv2.putText(output, cl_str,
                        (badge_x + badge_w + 10, sec_y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY_DIM, 1)

            sec_y += player_row_h

    return output


# ============================================================
# DRAW TARGET CONE
# ============================================================

def draw_target_cone_on_frame(
    frame           : np.ndarray,
    target_pos      : Tuple[float, float],
    proximity_radius: float = 120.0,
    is_active       : bool  = False,
    color_normal    : Tuple[int, int, int] = (0, 165, 255),
    color_active    : Tuple[int, int, int] = (0, 255, 80),
    alpha           : float = 0.25
) -> np.ndarray:
    """Gambar visualisasi target cone dan radius keberhasilan."""
    output = frame.copy()
    color  = color_active if is_active else color_normal
    cx, cy = int(target_pos[0]), int(target_pos[1])
    radius = int(proximity_radius)

    overlay = output.copy()
    cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    cv2.circle(output, (cx, cy), radius, color, 2)

    line_len = 18
    cv2.line(output, (cx - line_len, cy), (cx + line_len, cy), color, 2)
    cv2.line(output, (cx, cy - line_len), (cx, cy + line_len), color, 2)

    cv2.circle(output, (cx, cy), 10, color,          -1)
    cv2.circle(output, (cx, cy), 12, (255, 255, 255),  2)

    label = "TARGET"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    lx = cx - lw // 2
    ly = cy - radius - 14
    cv2.rectangle(output,
                  (lx - 4, ly - lh - 2),
                  (lx + lw + 4, ly + 4),
                  color, -1)
    cv2.putText(output, label,
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    radius_text = f"r={radius}px"
    cv2.putText(output, radius_text,
                (cx - 28, cy + radius + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    return output

# Tambahkan di AKHIR draw_gate.py (setelah draw_target_cone_on_frame)

def draw_front_cones_on_frame(
    frame           : np.ndarray,
    front_cones     : dict,
    proximity_radius: float = 125.0,
    is_active       : bool  = False,
    color_normal    : Tuple[int, int, int] = (255, 180, 0),   # Biru-cyan
    color_active    : Tuple[int, int, int] = (0, 255, 80),    # Hijau
    alpha           : float = 0.18
) -> np.ndarray:
    """
    Gambar visualisasi 3 front cones dan radius keberhasilan.

    Args:
        frame           : frame video BGR
        front_cones     : dict {cone_id: (x, y)} posisi cone
        proximity_radius: radius keberhasilan dalam pixel
        is_active       : True jika bola sedang mendekati front cone
        color_normal    : warna normal
        color_active    : warna aktif
        alpha           : transparansi

    Returns:
        Frame dengan visualisasi front cones
    """
    output = frame.copy()
    color  = color_active if is_active else color_normal

    for cid, pos in front_cones.items():
        cx, cy = int(pos[0]), int(pos[1])
        radius = int(proximity_radius)

        # Lingkaran radius (semi-transparan)
        overlay = output.copy()
        cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Border
        cv2.circle(output, (cx, cy), radius, color, 1)

        # Crosshair kecil
        line_len = 10
        cv2.line(output, (cx - line_len, cy), (cx + line_len, cy), color, 1)
        cv2.line(output, (cx, cy - line_len), (cx, cy + line_len), color, 1)

        # Titik pusat
        cv2.circle(output, (cx, cy), 6, color,          -1)
        cv2.circle(output, (cx, cy), 8, (255, 255, 255),  1)

        # Label cone ID
        label = f"C{cid}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        lx = cx - lw // 2
        ly = cy - radius - 8
        cv2.rectangle(output,
                      (lx - 2, ly - lh - 2),
                      (lx + lw + 2, ly + 2),
                      color, -1)
        cv2.putText(output, label,
                    (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return output


# ============================================================
# DRAW GAWANG BBOX (BARU)
# ============================================================

def draw_gawang_on_frame(
    frame      : np.ndarray,
    gawang_bbox: List[float],
    is_goal    : bool = False,
    alpha      : float = 0.20
) -> np.ndarray:
    """Gambar visualisasi area gawang di frame."""
    output = frame.copy()
    x1, y1, x2, y2 = map(int, gawang_bbox)

    color = (0, 255, 80) if is_goal else (0, 200, 255)

    overlay = output.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    label = "GAWANG"
    mid_x = (x1 + x2) // 2
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(output,
                  (mid_x - lw // 2 - 4, y1 - lh - 10),
                  (mid_x + lw // 2 + 4, y1 - 2),
                  color, -1)
    cv2.putText(output, label,
                (mid_x - lw // 2, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return output


# ============================================================
# DRAW KICK RESULT INDICATOR (BARU)
# ============================================================

def draw_kick_result(
    frame       : np.ndarray,
    kicker_pos  : tuple,
    is_on_target: bool,
    kicker_jersey: str
) -> np.ndarray:
    """Gambar indikator hasil tendangan (ON TARGET / OFF TARGET)."""
    output = frame.copy()
    color  = (0, 255, 80) if is_on_target else (0, 0, 230)
    status = "ON TARGET!" if is_on_target else "OFF TARGET"

    if kicker_pos:
        px, py = int(kicker_pos[0]), int(kicker_pos[1])
        label = f"{kicker_jersey}: {status}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(output,
                      (px - lw // 2 - 6, py - 60 - lh),
                      (px + lw // 2 + 6, py - 50),
                      color, -1)
        cv2.putText(output, label,
                    (px - lw // 2, py - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return output


# ============================================================
# DRAW PENALTY STATS PANEL — SHOOT ON TARGET COUNTING
# ============================================================

def draw_penalty_stats_panel(
    frame      : np.ndarray,
    stats      : dict,
    position   : Tuple[int, int] = (20, 20),
    panel_width: int = 340
) -> np.ndarray:
    """
    Panel statistik shoot on target dengan desain modern.
    Menampilkan: Total / On Target / Off Target per pemain.
    """
    output = frame.copy()
    px, py = position
    pad = 14

    BG_DARK      = (25, 25, 30)
    BG_HEADER    = (50, 42, 35)
    ACCENT       = (230, 200, 0)
    GREEN        = (80, 220, 60)
    RED          = (70, 70, 230)
    YELLOW       = (50, 210, 255)
    WHITE        = (255, 255, 255)
    GRAY_LIGHT   = (200, 200, 200)
    GRAY_MED     = (140, 140, 140)
    GRAY_DIM     = (70, 70, 70)
    BAR_BG       = (50, 50, 55)

    per_player = stats.get('per_player', {})
    n_players  = len(per_player)

    header_h       = 44
    summary_h      = 70
    accuracy_h     = 42
    divider_pad    = 12
    player_row_h   = 56
    player_sec_h   = (divider_pad + 22 + player_row_h * n_players) if n_players > 0 else 0
    bottom_pad     = 12
    total_h        = header_h + summary_h + accuracy_h + player_sec_h + bottom_pad

    x1, y1 = px, py
    x2, y2 = px + panel_width, py + total_h

    # BACKGROUND
    overlay = output.copy()
    draw_rounded_rect(overlay, (x1, y1), (x2, y2), BG_DARK, radius=14)
    cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
    draw_rounded_rect(output, (x1, y1), (x2, y2), GRAY_DIM, radius=14, thickness=1)

    # HEADER
    overlay_h = output.copy()
    draw_rounded_rect(overlay_h, (x1 + 1, y1 + 1), (x2 - 1, y1 + header_h),
                      BG_HEADER, radius=13)
    cv2.rectangle(overlay_h, (x1 + 1, y1 + header_h - 13),
                  (x2 - 1, y1 + header_h), BG_HEADER, -1)
    cv2.addWeighted(overlay_h, 0.95, output, 0.05, 0, output)
    cv2.line(output, (x1 + pad, y1 + header_h), (x2 - pad, y1 + header_h),
             ACCENT, 2)

    icon_cx = x1 + pad + 11
    icon_cy = y1 + header_h // 2
    cv2.circle(output, (icon_cx, icon_cy), 11, YELLOW, -1)
    cv2.circle(output, (icon_cx, icon_cy), 11, WHITE, 1)
    cv2.circle(output, (icon_cx, icon_cy), 4, (30, 30, 30), -1)

    cv2.putText(output, "SHOOT ON TARGET",
                (icon_cx + 18, y1 + header_h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, WHITE, 2)

    # SUMMARY: 3 KOLOM (Total / On Target / Off Target)
    row_y = y1 + header_h + 8
    col_w = (panel_width - 2 * pad) // 3

    total_k   = stats.get('total_kicks', 0)
    total_on  = stats.get('total_on_target', 0)
    total_off = stats.get('total_off_target', 0)

    items = [
        (total_k,   "Total",      ACCENT),
        (total_on,  "On Target",  GREEN),
        (total_off, "Off Target", RED),
    ]

    for idx, (val, label, color) in enumerate(items):
        col_cx = x1 + pad + idx * col_w + col_w // 2
        val_str = str(val)
        (vw, vh), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)
        cv2.putText(output, val_str,
                    (col_cx - vw // 2, row_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2)
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.putText(output, label,
                    (col_cx - lw // 2, row_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY_MED, 1)

    for idx in range(1, 3):
        div_x = x1 + pad + idx * col_w
        cv2.line(output, (div_x, row_y + 8), (div_x, row_y + 50), GRAY_DIM, 1)

    # PROGRESS BAR ON TARGET %
    acc_y = row_y + summary_h
    on_pct = stats.get('on_target_pct', 0.0)

    cv2.putText(output, "Akurasi On Target",
                (x1 + pad, acc_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY_LIGHT, 1)

    pct_color = GREEN if on_pct >= 60 else (YELLOW if on_pct >= 30 else RED)

    pct_str = f"{on_pct:.0f}%"
    (pw, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.putText(output, pct_str,
                (x2 - pad - pw, acc_y + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, pct_color, 2)

    bar_x1 = x1 + pad
    bar_x2 = x2 - pad
    bar_y1 = acc_y + 12
    bar_y2 = bar_y1 + 16
    bar_total_w = bar_x2 - bar_x1

    draw_rounded_rect(output, (bar_x1, bar_y1), (bar_x2, bar_y2), BAR_BG, radius=8)

    fill_w = int(bar_total_w * min(on_pct / 100.0, 1.0))
    if fill_w > 6:
        draw_rounded_rect(output, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2),
                          pct_color, radius=8)

    # PER-PLAYER
    if per_player:
        sec_y = bar_y2 + divider_pad
        cv2.line(output, (x1 + pad, sec_y), (x2 - pad, sec_y), GRAY_DIM, 1)
        sec_y += 6

        cv2.putText(output, "Per Pemain",
                    (x1 + pad, sec_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, GRAY_MED, 1)
        sec_y += 22

        for jersey, pstats in per_player.items():
            p_total  = pstats.get('total', 0)
            p_on     = pstats.get('on_target', 0)
            p_off    = pstats.get('off_target', 0)
            p_pct    = pstats.get('on_target_pct', 0.0)

            # Badge jersey (warna berdasarkan nama)
            badge_x = x1 + pad
            badge_y = sec_y + 2
            badge_w = 70
            badge_h = 22

            if 'Merah' in jersey:
                badge_color = (0, 0, 180)
            elif 'Abu' in jersey:
                badge_color = (130, 130, 130)
            else:
                badge_color = (100, 100, 0)

            overlay_b = output.copy()
            draw_rounded_rect(overlay_b,
                              (badge_x, badge_y),
                              (badge_x + badge_w, badge_y + badge_h),
                              badge_color, radius=6)
            cv2.addWeighted(overlay_b, 0.85, output, 0.15, 0, output)

            (jw, _), _ = cv2.getTextSize(jersey, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(output, jersey,
                        (badge_x + (badge_w - jw) // 2, badge_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)

            # Ratio text: "2/3 on target"
            ratio_str = f"{p_on}/{p_total} on target"
            cv2.putText(output, ratio_str,
                        (badge_x + badge_w + 10, sec_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, GRAY_LIGHT, 1)

            # Mini progress bar
            mini_bar_x1 = badge_x + badge_w + 10
            mini_bar_x2 = x2 - pad - 48
            mini_bar_y1 = sec_y + 24
            mini_bar_y2 = mini_bar_y1 + 10
            mini_bar_w  = mini_bar_x2 - mini_bar_x1

            if mini_bar_w > 10:
                draw_rounded_rect(output,
                                  (mini_bar_x1, mini_bar_y1),
                                  (mini_bar_x2, mini_bar_y2),
                                  BAR_BG, radius=5)

                mini_fill  = int(mini_bar_w * min(p_pct / 100.0, 1.0))
                mini_color = GREEN if p_pct >= 60 else (YELLOW if p_pct >= 30 else RED)
                if mini_fill > 4:
                    draw_rounded_rect(output,
                                      (mini_bar_x1, mini_bar_y1),
                                      (mini_bar_x1 + mini_fill, mini_bar_y2),
                                      mini_color, radius=5)

            # Percentage
            p_pct_str = f"{p_pct:.0f}%"
            cv2.putText(output, p_pct_str,
                        (x2 - pad - 40, sec_y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        GREEN if p_pct >= 60 else (YELLOW if p_pct >= 30 else RED),
                        1)

            # Sublabel: off target count
            off_str = f"off target: {p_off}"
            cv2.putText(output, off_str,
                        (badge_x + badge_w + 10, sec_y + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY_DIM, 1)

            sec_y += player_row_h

    return output

