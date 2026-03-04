# draw_gate.py
# Fungsi visualisasi untuk gate cone dan trajectory bola di frame video.

import cv2
import numpy as np
from typing import Tuple, List, Optional


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
    """
    Gambar visualisasi gate di atas frame video.

    Menampilkan:
    - Titik lingkaran di posisi kedua cone gate
    - Garis imajiner antara dua cone
    - Area semi-transparan zona gate
    - Label 'GATE' dan lebar dalam pixel

    Args:
        frame             : frame video (numpy array BGR)
        gate_cone_left    : koordinat (x, y) cone kiri
        gate_cone_right   : koordinat (x, y) cone kanan
        is_active         : True jika bola sedang melewati gate
        gate_color_normal : warna default gate (BGR)
        gate_color_active : warna saat bola lewat gate (BGR)
        alpha             : transparansi area gate (0.0 - 1.0)
        line_thickness    : ketebalan garis gate
        show_width_label  : tampilkan label lebar gate

    Returns:
        Frame dengan visualisasi gate
    """
    output = frame.copy()
    color  = gate_color_active if is_active else gate_color_normal

    gl = (int(gate_cone_left[0]),  int(gate_cone_left[1]))
    gr = (int(gate_cone_right[0]), int(gate_cone_right[1]))

    # --- Area semi-transparan zona gate ---
    gate_height = 70
    poly_pts = np.array([
        gl,
        gr,
        (gr[0], gr[1] - gate_height),
        (gl[0], gl[1] - gate_height)
    ], dtype=np.int32)

    overlay = output.copy()
    cv2.fillPoly(overlay, [poly_pts], color)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # --- Garis gate ---
    cv2.line(output, gl, gr, color, line_thickness + 1)

    # --- Lingkaran di posisi cone ---
    cv2.circle(output, gl, 10, color,          -1)
    cv2.circle(output, gr, 10, color,          -1)
    cv2.circle(output, gl, 12, (255, 255, 255), 2)  # Border putih
    cv2.circle(output, gr, 12, (255, 255, 255), 2)

    # --- Garis vertikal di sisi cone (tiang gawang) ---
    pillar_height = 35
    cv2.line(output, gl, (gl[0], gl[1] - pillar_height), color, 2)
    cv2.line(output, gr, (gr[0], gr[1] - pillar_height), color, 2)

    # --- Label GATE ---
    mid_x = (gl[0] + gr[0]) // 2
    mid_y = min(gl[1], gr[1]) - gate_height - 12

    # Background label
    label_text = "GATE"
    (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(output,
                  (mid_x - lw // 2 - 4, mid_y - lh - 2),
                  (mid_x + lw // 2 + 4, mid_y + 2),
                  color, -1)
    cv2.putText(output, label_text,
                (mid_x - lw // 2, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # --- Label lebar gate ---
    if show_width_label:
        gate_width = int(np.linalg.norm(
            np.array(gate_cone_left) - np.array(gate_cone_right)
        ))
        width_text = f"{gate_width}px"
        cv2.putText(output, width_text,
                    (mid_x - 18, mid_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    return output


def draw_ball_trajectory_on_frame(
    frame: np.ndarray,
    trajectory: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 120, 0),
    max_points: int = 30,
    fade: bool = True
) -> np.ndarray:
    """
    Gambar titik-titik trajectory bola untuk keperluan debugging.

    Args:
        frame      : frame video
        trajectory : list of (x, y) dari extract_ball_trajectory()
        color      : warna titik dan garis
        max_points : maksimal titik yang ditampilkan (titik terbaru)
        fade       : jika True, titik lama lebih redup

    Returns:
        Frame dengan trajectory digambar
    """
    output = frame.copy()

    # Ambil N titik terakhir saja agar tidak terlalu penuh
    recent = trajectory[-max_points:] if len(trajectory) > max_points else trajectory
    n = len(recent)

    if n == 0:
        return output

    for i, point in enumerate(recent):
        px, py = int(point[0]), int(point[1])

        if fade and n > 1:
            ratio        = (i + 1) / n
            faded_color  = tuple(int(c * ratio) for c in color)
            pt_radius    = max(2, int(5 * ratio))
        else:
            faded_color = color
            pt_radius   = 4

        cv2.circle(output, (px, py), pt_radius, faded_color, -1)

        if i > 0:
            prev = (int(recent[i-1][0]), int(recent[i-1][1]))
            cv2.line(output, prev, (px, py), faded_color, 1)

    # Tandai titik terbaru dengan lingkaran lebih besar
    last = (int(recent[-1][0]), int(recent[-1][1]))
    cv2.circle(output, last, 7, color, 2)

    return output


def draw_pass_arrow(
    frame: np.ndarray,
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    success: bool,
    from_jersey: str,
    to_jersey: str,
    distance: float
) -> np.ndarray:
    """
    Gambar panah passing di frame dengan warna dan label status.

    Args:
        frame       : frame video
        from_pos    : posisi pengirim (x, y)
        to_pos      : posisi penerima (x, y)
        success     : True = SUKSES (hijau), False = GAGAL (merah)
        from_jersey : label jersey pengirim
        to_jersey   : label jersey penerima
        distance    : jarak passing dalam pixel

    Returns:
        Frame dengan panah passing
    """
    output = frame.copy()
    color  = (0, 220, 0) if success else (0, 0, 220)
    label  = "SUKSES" if success else "GAGAL"

    fp = (int(from_pos[0]), int(from_pos[1]))
    tp = (int(to_pos[0]),   int(to_pos[1]))

    # Gambar panah tebal
    cv2.arrowedLine(output, fp, tp, color, 3, tipLength=0.12)

    # Label di tengah panah
    mid_x = (fp[0] + tp[0]) // 2
    mid_y = (fp[1] + tp[1]) // 2

    pass_label = f"#{from_jersey}->#  {to_jersey} | {label}"
    dist_label = f"{distance:.0f}px"

    # Background label status
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


def draw_stats_panel(
    frame      : np.ndarray,
    stats      : dict,
    position   : Tuple[int, int] = (20, 20),
    panel_width: int = 280
) -> np.ndarray:
    """Gambar panel statistik passing di sudut frame."""
    output  = frame.copy()
    px, py  = position
    line_h  = 26
    padding = 10

    per_player = stats.get('per_player', {})
    n_rows     = 6 + len(per_player)   # +1 untuk avg_closest
    panel_h    = n_rows * line_h + padding * 2

    # Background panel semi-transparan
    overlay = output.copy()
    cv2.rectangle(overlay,
                  (px, py),
                  (px + panel_width, py + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

    # Border panel
    cv2.rectangle(output,
                  (px, py),
                  (px + panel_width, py + panel_h),
                  (100, 100, 100), 1)

    # Header
    cv2.putText(output, "STATISTIK PASSING",
                (px + padding, py + padding + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Garis pemisah header
    cv2.line(output,
             (px + padding, py + padding + 24),
             (px + panel_width - padding, py + padding + 24),
             (100, 100, 100), 1)

    # Baris statistik - label diupdate ke "target"
    rows = [
        ("Total Pass",     str(stats['total_passes']),               (200, 200, 200)),
        ("Sukses",         str(stats['successful_passes']),          (0, 220, 0)),
        ("Gagal",          str(stats['failed_passes']),              (0, 80, 220)),
        ("Akurasi",        f"{stats['accuracy_pct']}%",              (0, 200, 255)),
        ("Rata2 Jarak",    f"{stats['avg_distance']}px",             (180, 180, 180)),
        ("Avg Closest",    f"{stats.get('avg_closest_dist', 0.0)}px",(150, 150, 255)),
    ]

    y_offset = py + padding + 24 + line_h
    for label, value, color in rows:
        cv2.putText(output, f"{label}:",
                    (px + padding, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.putText(output, value,
                    (px + panel_width - padding - 65, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        y_offset += line_h

    # Statistik per pemain
    if per_player:
        cv2.line(output,
                 (px + padding, y_offset - 8),
                 (px + panel_width - padding, y_offset - 8),
                 (80, 80, 80), 1)
        cv2.putText(output, "Per Pemain:",
                    (px + padding, y_offset + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        y_offset += line_h

        for jersey, pstats in per_player.items():
            acc   = pstats['accuracy_pct']
            text  = (f"#{jersey}: {pstats['success']}/{pstats['total']} "
                     f"({acc:.0f}%)")
            color = ((0, 220, 0)   if acc >= 70 else
                     (0, 180, 255) if acc >= 40 else
                     (0, 80, 220))
            cv2.putText(output, text,
                        (px + padding + 8, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y_offset += line_h

    return output

def draw_target_cone_on_frame(
    frame           : np.ndarray,
    target_pos      : Tuple[float, float],
    proximity_radius: float = 120.0,
    is_active       : bool  = False,
    color_normal    : Tuple[int, int, int] = (0, 165, 255),  # Oranye
    color_active    : Tuple[int, int, int] = (0, 255, 80),   # Hijau
    alpha           : float = 0.25
) -> np.ndarray:
    """
    Gambar visualisasi target cone dan radius keberhasilan.

    Menampilkan:
    - Lingkaran di posisi target cone
    - Lingkaran radius sukses (transparan)
    - Label 'TARGET'

    Args:
        frame           : frame video BGR
        target_pos      : posisi (x, y) target cone
        proximity_radius: radius keberhasilan dalam pixel
        is_active       : True jika bola sedang mendekati target
        color_normal    : warna normal (oranye)
        color_active    : warna aktif (hijau)
        alpha           : transparansi area radius

    Returns:
        Frame dengan visualisasi target cone
    """
    output = frame.copy()
    color  = color_active if is_active else color_normal
    cx, cy = int(target_pos[0]), int(target_pos[1])
    radius = int(proximity_radius)

    # --- Lingkaran radius sukses (semi-transparan) ---
    overlay = output.copy()
    cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # --- Border lingkaran radius ---
    cv2.circle(output, (cx, cy), radius, color, 2)

    # --- Garis crosshair di target ---
    line_len = 18
    cv2.line(output, (cx - line_len, cy), (cx + line_len, cy), color, 2)
    cv2.line(output, (cx, cy - line_len), (cx, cy + line_len), color, 2)

    # --- Titik pusat ---
    cv2.circle(output, (cx, cy), 10, color,          -1)
    cv2.circle(output, (cx, cy), 12, (255, 255, 255),  2)

    # --- Label TARGET ---
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

    # --- Label radius ---
    radius_text = f"r={radius}px"
    cv2.putText(output, radius_text,
                (cx - 28, cy + radius + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    return output