from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def get_center_of_bbox(bbox: List[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_center_of_bbox_bottom(bbox: List[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def get_bbox_width(bbox: List[float]) -> int:
    return int(max(0.0, bbox[2] - bbox[0]))


def get_foot_position(bbox: List[float]) -> Tuple[int, int]:
    return get_center_of_bbox_bottom(bbox)


def measure_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return float(np.linalg.norm(
        np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
    ))


def bbox_area(bbox: List[float]) -> float:
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def interpolate_ball_positions(
    ball_positions: List[Dict[int, Dict[str, Any]]]
) -> List[Dict[int, Dict[str, Any]]]:
    if not ball_positions:
        return ball_positions

    n = len(ball_positions)
    xs1, ys1, xs2, ys2 = [], [], [], []
    for f in range(n):
        bb = ball_positions[f].get(1, {}).get("bbox", None)
        if bb is None:
            xs1.append(np.nan); ys1.append(np.nan)
            xs2.append(np.nan); ys2.append(np.nan)
        else:
            xs1.append(float(bb[0])); ys1.append(float(bb[1]))
            xs2.append(float(bb[2])); ys2.append(float(bb[3]))

    df = pd.DataFrame({"x1": xs1, "y1": ys1, "x2": xs2, "y2": ys2})
    df = df.interpolate(limit_direction="both")

    out: List[Dict[int, Dict[str, Any]]] = []
    for f in range(n):
        x1, y1, x2, y2 = df.loc[f, ["x1", "y1", "x2", "y2"]].tolist()
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            out.append({})
            continue
        out.append({1: {"bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": 1.0}})
    return out


# === BARU: FUNGSI MATEMATIKA UNTUK DETEKSI SUCCESS/FAILED ===

def ccw(A, B, C):
    """Mengecek perputaran (counter-clockwise) dari 3 titik"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def check_line_intersection(A, B, C, D):
    """
    Cek apakah garis lintasan bola (A ke B) 
    memotong garis imajiner antar cone (C ke D).
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)