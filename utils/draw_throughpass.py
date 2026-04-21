# utils/draw_throughpass.py
# ============================================================
# Fungsi visualisasi untuk Through Pass counting
# ============================================================

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


def draw_throughpass_status(
    frame: np.ndarray,
    state: str,
    sender_id: int = -1,
    flight_frames: int = 0,
    fps: int = 30,
    ball_dist_to_gate: float = 0.0,
) -> np.ndarray:
    """Status through pass realtime di pojok kanan atas."""
    h, w = frame.shape[:2]
    x_start = w - 380
    y_start = 80

    overlay = frame.copy()

    if state == 'ball_in_flight':
        bg_color = (0, 150, 220)
        status_text = "BALL IN FLIGHT"
        flight_sec = flight_frames / fps if fps > 0 else 0
        info = (f"P{sender_id} passing | "
                f"{flight_sec:.1f}s | gate_dist:{ball_dist_to_gate:.0f}px")
    elif state == 'possession':
        bg_color = (180, 140, 0)
        status_text = "POSSESSION"
        info = f"Player {sender_id} memiliki bola"
    elif state == 'through':
        bg_color = (0, 180, 0)
        status_text = "THROUGH PASS!"
        info = f"P{sender_id} melewati gate lawan!"
    elif state == 'missed':
        bg_color = (0, 0, 200)
        status_text = "MISSED!"
        info = f"Bola tidak melewati gate"
    else:
        return frame

    cv2.rectangle(overlay, (x_start, y_start),
                  (x_start + 360, y_start + 60), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, status_text,
                (x_start + 10, y_start + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.putText(frame, info,
                (x_start + 10, y_start + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    return frame


def draw_throughpass_result_flash(
    frame: np.ndarray,
    success: bool,
    event_number: int,
    sender_id: int = -1,
    flight_seconds: float = 0.0,
    flash_progress: float = 1.0,
) -> np.ndarray:
    """Flash hasil through pass di tengah layar."""
    h, w = frame.shape[:2]
    alpha = min(0.7, flash_progress * 0.7)

    if success:
        color = (0, 200, 0)
        text = f"THROUGH PASS #{event_number} - SUKSES!"
        sub_text = (f"P{sender_id} melewati gate! "
                    f"(flight:{flight_seconds:.1f}s)")
    else:
        color = (0, 0, 220)
        text = f"THROUGH PASS #{event_number} - GAGAL!"
        sub_text = f"Bola P{sender_id} tidak melewati gate"

    overlay = frame.copy()
    box_w, box_h = 560, 90
    bx1 = (w - box_w) // 2
    by1 = (h - box_h) // 2 - 30
    bx2 = bx1 + box_w
    by2 = by1 + box_h

    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    tx = (w - tw) // 2
    cv2.putText(frame, text, (tx, by1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    (sw, sh), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    sx = (w - sw) // 2
    cv2.putText(frame, sub_text, (sx, by1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    return frame


def draw_throughpass_stats_panel(
    frame: np.ndarray,
    stats: Dict,
    position: Tuple[int, int] = (20, 20),
    panel_width: int = 300,
    player_a_id: int = -1,
    player_b_id: int = -1,
) -> np.ndarray:
    """Panel statistik through pass realtime di pojok kiri atas."""
    x, y = position
    total = stats.get('total_throughpass', 0)
    sukses = stats.get('successful_throughpass', 0)
    gagal = stats.get('failed_throughpass', 0)
    akurasi = stats.get('accuracy_pct', 0.0)
    player_stats = stats.get('player_stats', {})

    # Hitung tinggi panel berdasarkan jumlah pemain
    panel_height = 150
    if player_stats:
        panel_height += len(player_stats) * 22

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height),
                  (80, 80, 80), 1)

    # Title
    cv2.putText(frame, "THROUGH PASS COUNTER",
                (x + 10, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.line(frame, (x + 10, y + 30), (x + panel_width - 10, y + 30),
             (80, 80, 80), 1)

    line_y = y + 50
    line_h = 22

    cv2.putText(frame, f"Total Through Pass : {total}",
                (x + 10, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(frame, f"Sukses             : {sukses}",
                (x + 10, line_y + line_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 0), 1)
    cv2.putText(frame, f"Gagal              : {gagal}",
                (x + 10, line_y + line_h * 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 230), 1)

    # Accuracy bar
    bar_y = line_y + line_h * 3
    cv2.putText(frame, f"Akurasi: {akurasi:.1f}%",
                (x + 10, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    bar_x1 = x + 140
    bar_x2 = x + panel_width - 15
    bar_w = bar_x2 - bar_x1
    bar_h = 12
    bar_y1 = bar_y - 10

    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1 + bar_h),
                  (60, 60, 60), -1)

    fill_w = int(bar_w * akurasi / 100.0) if akurasi > 0 else 0
    if fill_w > 0:
        bar_color = (
            (0, 200, 0) if akurasi >= 60 else
            (0, 180, 255) if akurasi >= 30 else
            (0, 0, 200)
        )
        cv2.rectangle(frame, (bar_x1, bar_y1),
                      (bar_x1 + fill_w, bar_y1 + bar_h), bar_color, -1)

    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1 + bar_h),
                  (150, 150, 150), 1)

    # Per-player stats
    if player_stats:
        player_y = bar_y + line_h + 5
        cv2.line(frame, (x + 10, player_y - 10),
                 (x + panel_width - 10, player_y - 10), (80, 80, 80), 1)
        for pid, ps in sorted(player_stats.items()):
            label = f"P{pid}: {ps['sukses']}/{ps['total']} sukses"
            cv2.putText(frame, label,
                        (x + 10, player_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            player_y += line_h

    return frame


def draw_throughpass_trajectory(
    frame: np.ndarray,
    trajectory: List[Tuple[int, int]],
    max_points: int = 60,
) -> np.ndarray:
    """Jejak trajectory bola selama through pass."""
    if len(trajectory) < 2:
        return frame

    points = trajectory[-max_points:]
    for i in range(1, len(points)):
        alpha = i / len(points)
        thickness = max(1, int(alpha * 3))
        color_val = int(alpha * 255)
        cv2.line(frame, points[i - 1], points[i],
                 (0, color_val, 255), thickness)

    return frame


def draw_cone_markers(
    frame: np.ndarray,
    gate_a: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    gate_b: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    cone_bboxes: Optional[Dict[int, List[float]]] = None,
    active_gate: Optional[str] = None,
    highlight_success: bool = False,
    highlight_fail: bool = False,
) -> np.ndarray:
    """
    Visualisasi cone dan garis gate antara cone pairs.

    gate_a: 2 cone di depan Player A (target untuk Player B)
    gate_b: 2 cone di depan Player B (target untuk Player A)
    """
    # Draw individual cone bboxes
    if cone_bboxes:
        for cid, bbox in cone_bboxes.items():
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # Triangle marker for cone
            r = max(8, (x2 - x1) // 2)
            pts = np.array([
                [cx, cy - r],
                [cx - r, cy + r],
                [cx + r, cy + r],
            ], np.int32)
            cv2.polylines(frame, [pts], True, (0, 200, 255), 2)

    # Draw gate lines
    if gate_a:
        a1 = tuple(map(int, gate_a[0]))
        a2 = tuple(map(int, gate_a[1]))
        gate_a_color = (100, 255, 100)  # Hijau muda default
        thickness = 2

        if active_gate == 'A':
            gate_a_color = (0, 255, 255)  # Kuning — gate aktif
            thickness = 3
            if highlight_success:
                gate_a_color = (0, 255, 0)  # Hijau terang
                thickness = 4
            elif highlight_fail:
                gate_a_color = (0, 0, 255)  # Merah
                thickness = 4

        cv2.line(frame, a1, a2, gate_a_color, thickness)
        mid_a = ((a1[0] + a2[0]) // 2, (a1[1] + a2[1]) // 2 - 15)
        cv2.putText(frame, "Gate A", mid_a,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, gate_a_color, 1)

    if gate_b:
        b1 = tuple(map(int, gate_b[0]))
        b2 = tuple(map(int, gate_b[1]))
        gate_b_color = (255, 150, 100)  # Biru muda default
        thickness = 2

        if active_gate == 'B':
            gate_b_color = (0, 255, 255)  # Kuning — gate aktif
            thickness = 3
            if highlight_success:
                gate_b_color = (0, 255, 0)
                thickness = 4
            elif highlight_fail:
                gate_b_color = (0, 0, 255)
                thickness = 4

        cv2.line(frame, b1, b2, gate_b_color, thickness)
        mid_b = ((b1[0] + b2[0]) // 2, (b1[1] + b2[1]) // 2 - 15)
        cv2.putText(frame, "Gate B", mid_b,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, gate_b_color, 1)

    return frame


def draw_player_label_tp(
    frame: np.ndarray,
    bbox: List[float],
    player_id: int,
    is_sender: bool = False,
    has_ball: bool = False,
    player_label: str = "",
) -> np.ndarray:
    """Gambar label pemain dengan warna sesuai role."""
    x1, y1, x2, y2 = map(int, bbox)

    if is_sender and has_ball:
        box_color = (0, 180, 255)  # Orange — pengirim dengan bola
        label = f"P{player_id} [PASSING]"
    elif is_sender:
        box_color = (200, 150, 50)  # Biru muda — pengirim
        label = f"P{player_id} [Sender]"
    elif has_ball:
        box_color = (0, 255, 0)  # Hijau — memiliki bola
        label = f"P{player_id} [BALL]"
    else:
        box_color = (200, 80, 50)  # Default
        label = f"Player {player_id}"

    if player_label:
        label = f"{label} ({player_label})"

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    (lw, lh), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
    )
    cv2.rectangle(frame, (x1, y1 - lh - 8),
                  (x1 + lw + 8, y1), box_color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    return frame
