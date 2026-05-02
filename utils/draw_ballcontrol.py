# utils/draw_ballcontrol.py
# Fungsi visualisasi untuk Ball Control counting

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


def draw_ballcontrol_zone(
    frame: np.ndarray,
    zone: Optional[List[Tuple[float, float]]],
    zone_label: str = "A",
    is_active: bool = False,
    is_success: bool = False,
    is_fail: bool = False,
    is_checking: bool = False,
) -> np.ndarray:
    """Gambar area zone persegi 4 cone sebagai polygon semi-transparan."""
    if zone is None or len(zone) < 3:
        return frame

    pts = np.array([(int(p[0]), int(p[1])) for p in zone], np.int32)

    # Tentukan warna berdasarkan status
    if is_success:
        fill_color = (0, 200, 0)
        border_color = (0, 255, 0)
        alpha = 0.25
    elif is_fail:
        fill_color = (0, 0, 200)
        border_color = (0, 0, 255)
        alpha = 0.25
    elif is_checking:
        fill_color = (0, 200, 255)
        border_color = (0, 255, 255)
        alpha = 0.15
    elif is_active:
        fill_color = (200, 180, 0)
        border_color = (255, 220, 0)
        alpha = 0.12
    else:
        fill_color = (100, 100, 100)
        border_color = (150, 150, 150)
        alpha = 0.08

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], fill_color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], True, border_color, 2)

    # Label zone
    cx = int(np.mean([p[0] for p in zone]))
    cy = int(np.mean([p[1] for p in zone])) - 20
    cv2.putText(frame, f"Zone {zone_label}", (cx - 25, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1)

    return frame


def draw_cone_markers_bc(
    frame: np.ndarray,
    cone_bboxes: Optional[Dict[int, List[float]]] = None,
) -> np.ndarray:
    """Gambar marker cone individual."""
    if not cone_bboxes:
        return frame
    for cid, bbox in cone_bboxes.items():
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        r = max(8, (x2-x1)//2)
        pts = np.array([[cx, cy-r], [cx-r, cy+r], [cx+r, cy+r]], np.int32)
        cv2.polylines(frame, [pts], True, (0, 200, 255), 2)
    return frame


def draw_ballcontrol_status(
    frame: np.ndarray,
    state: str,
    sender_id: int = -1,
    receiver_id: int = -1,
    transit_frames: int = 0,
    control_frames: int = 0,
    control_check_frames: int = 30,
    ball_in_zone: bool = True,
    fps: int = 30,
) -> np.ndarray:
    """Status ball control realtime di pojok kanan atas."""
    h, w = frame.shape[:2]
    x_start = w - 380
    y_start = 80
    overlay = frame.copy()

    if state == 'ball_transit':
        bg_color = (0, 150, 220)
        text = "BALL IN TRANSIT"
        sec = transit_frames / fps if fps > 0 else 0
        info = f"P{sender_id} -> ? | {sec:.1f}s"
    elif state == 'checking_control':
        ratio = min(1.0, control_frames / control_check_frames)
        if ball_in_zone:
            bg_color = (0, int(180 * (1 - ratio * 0.3)), 0)
            text = "CHECKING CONTROL..."
        else:
            bg_color = (0, 0, 200)
            text = "BALL OUT OF ZONE!"
        info = f"P{receiver_id} | {control_frames}/{control_check_frames}f"

        # Progress bar
        cv2.rectangle(overlay, (x_start, y_start),
                      (x_start+360, y_start+72), bg_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, text, (x_start+10, y_start+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.putText(frame, info, (x_start+10, y_start+45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
        # Bar
        bar_x1, bar_x2 = x_start+10, x_start+350
        bar_y1 = y_start+55
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1+8), (60,60,60), -1)
        fill = int((bar_x2-bar_x1) * ratio)
        if fill > 0:
            bc = (0,200,0) if ball_in_zone else (0,0,255)
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1+fill, bar_y1+8), bc, -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1+8), (150,150,150), 1)
        return frame
    elif state == 'possession':
        bg_color = (180, 140, 0)
        text = "POSSESSION"
        info = f"Player {sender_id}"
    else:
        return frame

    cv2.rectangle(overlay, (x_start, y_start),
                  (x_start+360, y_start+60), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (x_start+10, y_start+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
    cv2.putText(frame, info, (x_start+10, y_start+48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
    return frame


def draw_ballcontrol_result_flash(
    frame: np.ndarray,
    success: bool,
    event_number: int,
    sender_id: int = -1,
    receiver_id: int = -1,
    flight_seconds: float = 0.0,
    reason: str = '',
    flash_progress: float = 1.0,
) -> np.ndarray:
    """Flash hasil ball control di tengah layar."""
    h, w = frame.shape[:2]
    alpha = min(0.7, flash_progress * 0.7)

    if success:
        color = (0, 200, 0)
        text = f"BALL CONTROL #{event_number} - SUKSES!"
        sub = f"P{sender_id}->P{receiver_id} | Bola tetap di zone"
    else:
        color = (0, 0, 220)
        text = f"BALL CONTROL #{event_number} - GAGAL!"
        sub = reason[:50] if reason else f"Bola keluar zone P{receiver_id}"

    overlay = frame.copy()
    bw, bh = 560, 90
    bx1 = (w-bw)//2
    by1 = (h-bh)//2 - 30
    cv2.rectangle(overlay, (bx1, by1), (bx1+bw, by1+bh), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx1+bw, by1+bh), (255,255,255), 2)

    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.putText(frame, text, ((w-tw)//2, by1+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, sub, ((w-sw)//2, by1+65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return frame


def draw_ballcontrol_stats_panel(
    frame: np.ndarray,
    stats: Dict,
    position: Tuple[int,int] = (20,20),
    panel_width: int = 300,
    player_a_id: int = -1,
    player_b_id: int = -1,
) -> np.ndarray:
    """Panel statistik ball control di pojok kiri atas."""
    x, y = position
    total = stats.get('total_ballcontrol', 0)
    sukses = stats.get('successful_ballcontrol', 0)
    gagal = stats.get('failed_ballcontrol', 0)
    akurasi = stats.get('accuracy_pct', 0.0)
    ps = stats.get('player_stats', {})

    ph = 150 + (len(ps) * 22 if ps else 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x,y), (x+panel_width, y+ph), (30,30,30), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x,y), (x+panel_width, y+ph), (80,80,80), 1)

    cv2.putText(frame, "BALL CONTROL", (x+10, y+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,200), 2)
    cv2.line(frame, (x+10, y+30), (x+panel_width-10, y+30), (80,80,80), 1)

    ly = y + 50
    lh = 22
    cv2.putText(frame, f"Total   : {total}", (x+10, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1)
    cv2.putText(frame, f"Sukses  : {sukses}", (x+10, ly+lh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,230,0), 1)
    cv2.putText(frame, f"Gagal   : {gagal}", (x+10, ly+lh*2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,230), 1)

    # Accuracy bar
    by = ly + lh * 3
    cv2.putText(frame, f"Akurasi: {akurasi:.1f}%", (x+10, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
    bx1, bx2 = x+140, x+panel_width-15
    bw = bx2 - bx1
    by1 = by - 10
    cv2.rectangle(frame, (bx1, by1), (bx2, by1+12), (60,60,60), -1)
    fw = int(bw * akurasi / 100.0) if akurasi > 0 else 0
    if fw > 0:
        bc = (0,200,0) if akurasi >= 60 else (0,180,255) if akurasi >= 30 else (0,0,200)
        cv2.rectangle(frame, (bx1, by1), (bx1+fw, by1+12), bc, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by1+12), (150,150,150), 1)

    if ps:
        py2 = by + lh + 5
        cv2.line(frame, (x+10, py2-10), (x+panel_width-10, py2-10), (80,80,80), 1)
        for pid, s in sorted(ps.items()):
            label = f"P{pid}: {s['sukses']}/{s['total']} sukses"
            cv2.putText(frame, label, (x+10, py2+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
            py2 += lh

    return frame


def draw_ballcontrol_trajectory(
    frame: np.ndarray,
    trajectory: List[Tuple[int,int]],
    max_points: int = 60,
) -> np.ndarray:
    """Trajectory bola."""
    if len(trajectory) < 2:
        return frame
    pts = trajectory[-max_points:]
    for i in range(1, len(pts)):
        a = i / len(pts)
        t = max(1, int(a * 3))
        cv2.line(frame, pts[i-1], pts[i], (0, int(a*255), 255), t)
    return frame


def draw_player_label_bc(
    frame: np.ndarray,
    bbox: List[float],
    player_id: int,
    is_sender: bool = False,
    is_receiver: bool = False,
    is_controlling: bool = False,
    ball_in_zone: bool = True,
    player_label: str = "",
) -> np.ndarray:
    """Label pemain dengan role."""
    x1, y1, x2, y2 = map(int, bbox)

    if is_controlling:
        box_color = (0, 200, 0) if ball_in_zone else (0, 0, 255)
        suffix = " [CTRL]" if ball_in_zone else " [OUT!]"
    elif is_receiver:
        box_color = (0, 180, 255)
        suffix = " [Recv]"
    elif is_sender:
        box_color = (200, 150, 50)
        suffix = " [Send]"
    else:
        box_color = (200, 80, 50)
        suffix = ""

    label = f"P{player_id}"
    if player_label:
        label += f" ({player_label})"
    label += suffix

    cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+8, y1), box_color, -1)
    cv2.putText(frame, label, (x1+4, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1)
    return frame
