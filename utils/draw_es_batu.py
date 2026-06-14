# utils/draw_es_batu.py
# Fungsi-fungsi visualisasi untuk proyek ES BATU.
# Menampilkan bounding box, counter panel, dan animasi flash saat es batu masuk kendaraan.

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================
# WARNA TEMA
# ============================================================

COLOR_ES_BATU   = (200, 230, 255)   # Biru muda (es)
COLOR_ORANG     = (50,  200, 80)    # Hijau (orang)
COLOR_MOTOR     = (0,   180, 255)   # Oranye kekuningan (motor)
COLOR_TRUK      = (0,   100, 255)   # Oranye (truk)
COLOR_CARRIED   = (255, 200, 50)    # Kuning (es dibawa orang)
COLOR_ENTERING  = (0,   255, 180)   # Cyan (es masuk kendaraan)
COLOR_COUNTED   = (100, 100, 100)   # Abu-abu (sudah terhitung)
COLOR_WHITE     = (255, 255, 255)
COLOR_BLACK     = (0,   0,   0)
COLOR_BG_PANEL  = (20,  20,  30)    # Latar panel gelap


# ============================================================
# HELPER UMUM
# ============================================================

def _draw_label(
    frame : np.ndarray,
    text  : str,
    x1    : int,
    y1    : int,
    color : Tuple[int, int, int],
    font_scale: float = 0.45,
    thickness : int   = 1,
) -> None:
    """Gambar label dengan background berwarna di atas bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    bg_y1 = max(y1 - th - pad * 2, 0)
    bg_y2 = y1
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + pad * 2, bg_y2), color, -1)
    cv2.putText(frame, text, (x1 + pad, y1 - pad),
                font, font_scale, COLOR_WHITE, thickness, cv2.LINE_AA)


def _draw_rounded_rect(
    frame : np.ndarray,
    x1    : int,
    y1    : int,
    x2    : int,
    y2    : int,
    color : Tuple[int, int, int],
    radius: int = 8,
    thickness: int = 2,
) -> None:
    """Gambar persegi panjang dengan sudut membulat (approx)."""
    if thickness < 0:
        # Fill
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


# ============================================================
# DRAW BOUNDING BOX PER CLASS
# ============================================================

def draw_es_batu_bbox(
    frame     : np.ndarray,
    bbox      : List[float],
    es_id     : int,
    state     : str,
    conf      : float = 0.0,
) -> np.ndarray:
    """Gambar bbox es batu dengan warna sesuai state."""
    x1, y1, x2, y2 = map(int, bbox)

    if 'COOLDOWN' in str(state) or state == 'COUNTED':
        color = COLOR_COUNTED
    elif state == 'CARRIED':
        color = COLOR_CARRIED
    elif state == 'ENTERING':
        color = COLOR_ENTERING
    else:
        color = COLOR_ES_BATU

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"Es#{es_id}"
    if conf > 0:
        label += f" {conf:.2f}"

    _draw_label(frame, label, x1, y1, color, font_scale=0.42)

    # Titik tengah
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame


def draw_orang_bbox(
    frame  : np.ndarray,
    bbox   : List[float],
    orang_id: int,
    is_carrying: bool = False,
) -> np.ndarray:
    """Gambar bbox orang, warna berbeda jika sedang membawa es batu."""
    x1, y1, x2, y2 = map(int, bbox)
    color = COLOR_CARRIED if is_carrying else COLOR_ORANG

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"Orang#{orang_id}" + (" [bawa es]" if is_carrying else "")
    _draw_label(frame, label, x1, y1, color, font_scale=0.44)

    return frame


def draw_motor_bbox(
    frame   : np.ndarray,
    bbox    : List[float],
    motor_id: int,
    highlight: bool = False,
) -> np.ndarray:
    """Gambar bbox motor."""
    x1, y1, x2, y2 = map(int, bbox)
    color = COLOR_ENTERING if highlight else COLOR_MOTOR
    thickness = 3 if highlight else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    _draw_label(frame, f"Motor#{motor_id}", x1, y1, color, font_scale=0.44)
    return frame


def draw_truk_bbox(
    frame  : np.ndarray,
    bbox   : List[float],
    truk_id: int,
    highlight: bool = False,
) -> np.ndarray:
    """Gambar bbox truk."""
    x1, y1, x2, y2 = map(int, bbox)
    color = COLOR_ENTERING if highlight else COLOR_TRUK
    thickness = 3 if highlight else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    _draw_label(frame, f"Truk#{truk_id}", x1, y1, color, font_scale=0.44)
    return frame


# ============================================================
# PANEL COUNTER UTAMA
# ============================================================

def draw_counter_panel(
    frame       : np.ndarray,
    count_motor : int,
    count_truk  : int,
    position    : Tuple[int, int] = (20, 20),
    panel_width : int = 300,
) -> np.ndarray:
    """
    Gambar panel statistik counter es batu di pojok kiri atas.
    """
    px, py = position
    line_h  = 32
    pad     = 14
    n_lines = 5   # judul + separator + motor + truk + total
    panel_h = n_lines * line_h + pad * 2

    # Background panel (semi-transparan)
    overlay = frame.copy()
    _draw_rounded_rect(overlay, px, py, px + panel_width, py + panel_h,
                       COLOR_BG_PANEL, radius=10, thickness=-1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Border panel
    _draw_rounded_rect(frame, px, py, px + panel_width, py + panel_h,
                       (80, 80, 120), radius=10, thickness=1)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    small  = 0.48
    medium = 0.58
    bold   = 1

    # Judul
    ty = py + pad + line_h - 8
    cv2.putText(frame, "ES BATU COUNTER", (px + pad, ty),
                font, medium, COLOR_ES_BATU, bold, cv2.LINE_AA)

    # Separator
    ty += int(line_h * 0.6)
    cv2.line(frame, (px + pad, ty), (px + panel_width - pad, ty), (80, 80, 120), 1)
    ty += int(line_h * 0.6)

    # Motor
    motor_color = COLOR_MOTOR if count_motor == 0 else (50, 220, 255)
    cv2.putText(frame, "Motor :", (px + pad, ty),
                font, small, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, str(count_motor),
                (px + panel_width - pad - 50, ty),
                font, medium, motor_color, bold, cv2.LINE_AA)
    ty += line_h

    # Truk
    truk_color = COLOR_TRUK if count_truk == 0 else (50, 150, 255)
    cv2.putText(frame, "Truk  :", (px + pad, ty),
                font, small, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, str(count_truk),
                (px + panel_width - pad - 50, ty),
                font, medium, truk_color, bold, cv2.LINE_AA)
    ty += line_h

    # Separator
    cv2.line(frame, (px + pad, ty - 4), (px + panel_width - pad, ty - 4),
             (80, 80, 120), 1)

    # Total
    total = count_motor + count_truk
    total_color = (100, 255, 180) if total > 0 else (150, 150, 150)
    cv2.putText(frame, "Total :", (px + pad, ty + line_h - 8),
                font, small, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, str(total),
                (px + panel_width - pad - 50, ty + line_h - 8),
                font, 0.70, total_color, 2, cv2.LINE_AA)

    return frame


# ============================================================
# FLASH ANIMASI SAAT ES BATU MASUK KENDARAAN
# ============================================================

def draw_entry_flash(
    frame     : np.ndarray,
    vehicle   : str,          # 'motor' atau 'truk'
    count     : int,
    progress  : float,        # 1.0 → 0.0 (1.0 = baru masuk)
) -> np.ndarray:
    """
    Tampilkan flash notifikasi di tengah layar saat es batu masuk kendaraan.
    progress: 1.0 = awal flash, 0.0 = flash hampir selesai.
    """
    if progress <= 0:
        return frame

    h, w = frame.shape[:2]
    alpha = min(1.0, progress * 2.0)   # fade in cepat, fade out lambat

    vehicle_upper = vehicle.upper()
    if vehicle == 'motor':
        flash_color = COLOR_MOTOR
        icon = "MOTOR"
    else:
        flash_color = COLOR_TRUK
        icon = "TRUK"

    text1 = f"ES BATU MASUK {icon}!"
    text2 = f"Total {vehicle_upper}: {count}"

    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale1 = 1.2
    scale2 = 0.85

    (tw1, th1), _ = cv2.getTextSize(text1, font, scale1, 2)
    (tw2, th2), _ = cv2.getTextSize(text2, font, scale2, 2)

    cx = w // 2
    cy = h // 2

    # Background flash
    box_w = max(tw1, tw2) + 60
    box_h = th1 + th2 + 60
    bx1 = cx - box_w // 2
    by1 = cy - box_h // 2
    bx2 = cx + box_w // 2
    by2 = cy + box_h // 2

    overlay = frame.copy()
    _draw_rounded_rect(overlay, bx1, by1, bx2, by2, COLOR_BG_PANEL, radius=12, thickness=-1)
    _draw_rounded_rect(overlay, bx1, by1, bx2, by2, flash_color, radius=12, thickness=3)
    cv2.addWeighted(overlay, alpha * 0.85, frame, 1 - alpha * 0.85, 0, frame)

    # Teks 1
    tx1 = cx - tw1 // 2
    ty1 = cy - th2 // 2 - 10
    cv2.putText(frame, text1, (tx1, ty1), font, scale1, flash_color, 2, cv2.LINE_AA)

    # Teks 2
    tx2 = cx - tw2 // 2
    ty2 = ty1 + th1 + 20
    cv2.putText(frame, text2, (tx2, ty2), font, scale2, COLOR_WHITE, 1, cv2.LINE_AA)

    return frame


# ============================================================
# ANNOTASI LENGKAP SATU FRAME
# ============================================================

def annotate_frame(
    frame       : np.ndarray,
    frame_num   : int,
    tracks      : Dict,
    frame_result: Dict,
    flash_map   : Dict[int, Dict],
    show_ids    : bool = True,
) -> np.ndarray:
    """
    Render anotasi lengkap pada satu frame.

    Args:
        frame        : Frame asli (numpy array)
        frame_num    : Nomor frame
        tracks       : Output EsBatuTracker
        frame_result : Output EsBatuCounter.process_frame()
        flash_map    : {frame_num: {'vehicle': str, 'count': int, 'progress': float}}
        show_ids     : Tampilkan ID tracker

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()

    es_batu_dict = tracks['es_batu'][frame_num]
    orang_dict   = tracks['orang'][frame_num]
    motor_dict   = tracks['motor'][frame_num]
    truk_dict    = tracks['truk'][frame_num]

    es_states    = frame_result.get('es_states', {})
    new_motor    = set(frame_result.get('new_motor', []))
    new_truk     = set(frame_result.get('new_truk', []))

    # Set orang yang sedang membawa es batu di frame ini
    carrying_persons: set = set()
    for es_id, st in es_states.items():
        if st == 'CARRIED':
            eb_data = es_batu_dict.get(es_id)
            if eb_data:
                eb_center = (
                    (eb_data['bbox'][0] + eb_data['bbox'][2]) / 2,
                    (eb_data['bbox'][1] + eb_data['bbox'][3]) / 2,
                )
                for pid, pdata in orang_dict.items():
                    px1, py1, px2, py2 = pdata['bbox']
                    if px1 <= eb_center[0] <= px2 and py1 <= eb_center[1] <= py2:
                        carrying_persons.add(pid)

    # Set kendaraan yang sedang menerima es batu
    motor_highlight: set = set()
    truk_highlight : set = set()
    for es_id in new_motor:
        eb = es_batu_dict.get(es_id)
        if eb:
            for mid, mdata in motor_dict.items():
                if _intersection_ratio(eb['bbox'], mdata['bbox']) > 0.05:
                    motor_highlight.add(mid)
    for es_id in new_truk:
        eb = es_batu_dict.get(es_id)
        if eb:
            for tid, tdata in truk_dict.items():
                if _intersection_ratio(eb['bbox'], tdata['bbox']) > 0.05:
                    truk_highlight.add(tid)

    # ------ Gambar TRUK ------
    for tid, tdata in truk_dict.items():
        annotated = draw_truk_bbox(annotated, tdata['bbox'], tid,
                                   highlight=(tid in truk_highlight))

    # ------ Gambar MOTOR ------
    for mid, mdata in motor_dict.items():
        annotated = draw_motor_bbox(annotated, mdata['bbox'], mid,
                                    highlight=(mid in motor_highlight))

    # ------ Gambar ORANG ------
    for pid, pdata in orang_dict.items():
        annotated = draw_orang_bbox(annotated, pdata['bbox'], pid,
                                    is_carrying=(pid in carrying_persons))

    # ------ Gambar ES BATU ------
    for es_id, eb_data in es_batu_dict.items():
        state = es_states.get(es_id, 'IDLE')
        annotated = draw_es_batu_bbox(
            annotated, eb_data['bbox'], es_id, state, eb_data.get('conf', 0)
        )

    # ------ Panel counter ------
    annotated = draw_counter_panel(
        annotated,
        count_motor = frame_result['count_motor'],
        count_truk  = frame_result['count_truk'],
        position    = (20, 20),
    )

    # ------ Flash animasi ------
    if frame_num in flash_map:
        fi = flash_map[frame_num]
        annotated = draw_entry_flash(
            annotated,
            vehicle  = fi['vehicle'],
            count    = fi['count'],
            progress = fi['progress'],
        )

    # ------ Frame label ------
    h, w = annotated.shape[:2]
    cv2.putText(annotated, f"Frame: {frame_num}",
                (w - 140, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1)

    return annotated
