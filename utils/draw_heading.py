# draw_heading.py
# Visualisasi heading detection — overlay bbox kepala, status, flash, panel stats.

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


def draw_head_bbox_on_frame(
    frame: np.ndarray,
    head_bbox: List[float],
    is_contact: bool = False,
    is_success: bool = False,
    is_fail: bool = False,
) -> np.ndarray:
    """
    Gambar bbox kepala (class Heading dari YOLO) dengan warna sesuai state.
    """
    x1, y1, x2, y2 = map(int, head_bbox)

    if is_success:
        color = (0, 255, 0)
        thickness = 3
        label = "HEAD - HIT!"
    elif is_fail:
        color = (0, 0, 255)
        thickness = 3
        label = "HEAD - MISS"
    elif is_contact:
        color = (0, 255, 255)
        thickness = 2
        label = "HEAD"
    else:
        color = (255, 180, 0)
        thickness = 2
        label = "HEAD"

    # Semi-transparent fill
    overlay = frame.copy()
    alpha = 0.25 if is_success or is_fail else 0.10
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    return frame


def draw_heading_status(
    frame: np.ndarray,
    is_approaching: bool = False,
    is_contact: bool = False,
    player_id: int = -1,
    head_dist: float = 0.0,
    iou: float = 0.0,
) -> np.ndarray:
    """Status heading realtime di pojok kanan atas."""
    h, w = frame.shape[:2]
    x_start = w - 330
    y_start = 80

    overlay = frame.copy()

    if is_contact:
        bg_color = (0, 180, 0)
        status_text = "BALL HITS HEAD!"
    elif is_approaching:
        bg_color = (0, 150, 220)
        status_text = "BALL APPROACHING"
    else:
        return frame

    cv2.rectangle(overlay, (x_start, y_start),
                  (x_start + 310, y_start + 60), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, status_text,
                (x_start + 10, y_start + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    info = f"Player {player_id} | dist:{head_dist:.0f}px IoU:{iou:.2f}"
    cv2.putText(frame, info,
                (x_start + 10, y_start + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    return frame


def draw_heading_result_flash(
    frame: np.ndarray,
    success: bool,
    event_number: int,
    head_distance: float,
    iou: float = 0.0,
    flash_progress: float = 1.0,
) -> np.ndarray:
    """Flash hasil heading di tengah layar."""
    h, w = frame.shape[:2]
    alpha = min(0.7, flash_progress * 0.7)

    if success:
        color = (0, 200, 0)
        text = f"HEADING #{event_number} - SUKSES!"
        sub_text = f"Bola mengenai kepala (dist:{head_distance:.0f}px, IoU:{iou:.2f})"
    else:
        color = (0, 0, 220)
        text = f"HEADING #{event_number} - GAGAL!"
        sub_text = f"Bola TIDAK mengenai kepala (dist:{head_distance:.0f}px)"

    overlay = frame.copy()
    box_w, box_h = 520, 90
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


def draw_heading_stats_panel(
    frame: np.ndarray,
    stats: Dict,
    position: Tuple[int, int] = (20, 20),
    panel_width: int = 280,
) -> np.ndarray:
    """Panel statistik heading realtime di pojok kiri atas."""
    x, y = position
    total = stats.get('total_headings', 0)
    sukses = stats.get('successful_headings', 0)
    gagal = stats.get('failed_headings', 0)
    akurasi = stats.get('accuracy_pct', 0.0)

    panel_height = 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height),
                  (80, 80, 80), 1)

    # Title
    cv2.putText(frame, "HEADING COUNTER",
                (x + 10, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.line(frame, (x + 10, y + 30), (x + panel_width - 10, y + 30),
             (80, 80, 80), 1)

    line_y = y + 50
    line_h = 22

    cv2.putText(frame, f"Total Heading  : {total}",
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


def draw_ball_trajectory_on_frame(
    frame: np.ndarray,
    trajectory: List[Tuple[int, int]],
    max_points: int = 40,
) -> np.ndarray:
    """Jejak trajectory bola."""
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
