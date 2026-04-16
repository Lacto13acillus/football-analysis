# utils/draw_longpass.py
# ============================================================
# Fungsi visualisasi untuk Long Pass counting
# ============================================================

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


def draw_longpass_status(
    frame: np.ndarray,
    state: str,
    sender_id: int = -1,
    receiver_id: int = -1,
    ball_dist: float = 0.0,
    flight_frames: int = 0,
    fps: int = 30,
) -> np.ndarray:
    """Status longpass realtime di pojok kanan atas."""
    h, w = frame.shape[:2]
    x_start = w - 350
    y_start = 80

    overlay = frame.copy()

    if state == 'ball_in_air':
        bg_color = (0, 150, 220)
        status_text = "BALL IN AIR"
        flight_sec = flight_frames / fps if fps > 0 else 0
        info = (f"P{sender_id} -> P{receiver_id} | "
                f"{flight_sec:.1f}s | dist:{ball_dist:.0f}px")
    elif state == 'waiting_kick':
        bg_color = (180, 140, 0)
        status_text = "POSSESSION"
        info = f"Player {sender_id} memiliki bola"
    elif state == 'received':
        bg_color = (0, 180, 0)
        status_text = "RECEIVED!"
        info = f"P{receiver_id} menerima bola (dist:{ball_dist:.0f}px)"
    elif state == 'missed':
        bg_color = (0, 0, 200)
        status_text = "MISSED!"
        info = f"Bola tidak diterima (dist:{ball_dist:.0f}px)"
    else:
        return frame

    cv2.rectangle(overlay, (x_start, y_start),
                  (x_start + 330, y_start + 60), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, status_text,
                (x_start + 10, y_start + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.putText(frame, info,
                (x_start + 10, y_start + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    return frame


def draw_longpass_result_flash(
    frame: np.ndarray,
    success: bool,
    event_number: int,
    flight_seconds: float = 0.0,
    receive_distance: float = 0.0,
    flash_progress: float = 1.0,
) -> np.ndarray:
    """Flash hasil longpass di tengah layar."""
    h, w = frame.shape[:2]
    alpha = min(0.7, flash_progress * 0.7)

    if success:
        color = (0, 200, 0)
        text = f"LONGPASS #{event_number} - SUKSES!"
        sub_text = (f"Bola diterima! (flight:{flight_seconds:.1f}s, "
                    f"dist:{receive_distance:.0f}px)")
    else:
        color = (0, 0, 220)
        text = f"LONGPASS #{event_number} - GAGAL!"
        sub_text = f"Bola tidak diterima (dist:{receive_distance:.0f}px)"

    overlay = frame.copy()
    box_w, box_h = 540, 90
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


def draw_longpass_stats_panel(
    frame: np.ndarray,
    stats: Dict,
    position: Tuple[int, int] = (20, 20),
    panel_width: int = 280,
) -> np.ndarray:
    """Panel statistik longpass realtime di pojok kiri atas."""
    x, y = position
    total = stats.get('total_longpass', 0)
    sukses = stats.get('successful_longpass', 0)
    gagal = stats.get('failed_longpass', 0)
    akurasi = stats.get('accuracy_pct', 0.0)

    panel_height = 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height),
                  (80, 80, 80), 1)

    # Title
    cv2.putText(frame, "LONGPASS COUNTER",
                (x + 10, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.line(frame, (x + 10, y + 30), (x + panel_width - 10, y + 30),
             (80, 80, 80), 1)

    line_y = y + 50
    line_h = 22

    cv2.putText(frame, f"Total Longpass : {total}",
                (x + 10, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(frame, f"Sukses         : {sukses}",
                (x + 10, line_y + line_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 0), 1)
    cv2.putText(frame, f"Gagal          : {gagal}",
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

    return frame


def draw_longpass_trajectory(
    frame: np.ndarray,
    trajectory: List[Tuple[int, int]],
    max_points: int = 60,
) -> np.ndarray:
    """Jejak trajectory bola selama longpass."""
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


def draw_player_label(
    frame: np.ndarray,
    bbox: List[float],
    player_id: int,
    is_sender: bool = False,
    is_receiver: bool = False,
    has_ball: bool = False,
) -> np.ndarray:
    """Gambar label pemain dengan warna sesuai role."""
    x1, y1, x2, y2 = map(int, bbox)

    if is_sender and has_ball:
        box_color = (0, 180, 255)  # Orange — pengirim dengan bola
        label = f"P{player_id} [SENDER]"
    elif is_receiver and has_ball:
        box_color = (0, 255, 0)  # Hijau — penerima menerima bola
        label = f"P{player_id} [RECEIVED]"
    elif is_sender:
        box_color = (200, 150, 50)  # Biru muda — pengirim
        label = f"P{player_id} [Sender]"
    elif is_receiver:
        box_color = (150, 200, 50)  # Cyan — penerima
        label = f"P{player_id} [Receiver]"
    else:
        box_color = (200, 80, 50)  # Default
        label = f"Player {player_id}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    (lw, lh), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
    )
    cv2.rectangle(frame, (x1, y1 - lh - 8),
                  (x1 + lw + 8, y1), box_color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    return frame
