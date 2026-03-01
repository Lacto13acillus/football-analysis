from typing import Any, Dict, List, Optional, Tuple
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.bbox_utils import get_center_of_bbox, get_center_of_bbox_bottom, measure_distance


class PlayerBallAssigner:
    """
    Assigns ball possession to the nearest player within max distance.

    PERUBAHAN:
    - max_player_ball_distance: 200 -> 150
      Mengurangi false possession ketika bola hanya MELINTAS dekat
      pemain tapi belum benar-benar di kaki pemain tersebut.
    - Ditambahkan _ball_inside_bbox() untuk prioritaskan overlap langsung.
    - Multi-point distance check (kedua kaki + center bottom + center).
    """

    def __init__(self, max_player_ball_distance: float = 150.0):
        self.max_player_ball_distance = float(max_player_ball_distance)

    def _ball_inside_bbox(self, ball_center: Tuple[int, int],
                          bbox: List[float]) -> bool:
        bx, by = ball_center
        x1, y1, x2, y2 = bbox
        return x1 <= bx <= x2 and y1 <= by <= y2

    def assign_ball_to_player(
        self,
        player_track: Dict[int, Dict[str, Any]],
        ball_bbox: List[float],
    ) -> Tuple[int, Optional[float]]:
        if not ball_bbox or len(ball_bbox) != 4:
            return -1, None

        ball_center = get_center_of_bbox(ball_bbox)
        ball_bottom = get_center_of_bbox_bottom(ball_bbox)

        best_player = -1
        best_dist: Optional[float] = None

        for tid_raw, pdata in player_track.items():
            tid = int(tid_raw)
            bbox = pdata.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue

            # Prioritas 1: bola di dalam bounding box pemain
            if self._ball_inside_bbox(ball_center, bbox):
                d = 0.0
            else:
                # Hitung jarak dari berbagai titik referensi
                foot_left = (bbox[0], bbox[3])
                foot_right = (bbox[2], bbox[3])
                foot_center = get_center_of_bbox_bottom(bbox)
                center = get_center_of_bbox(bbox)

                d_foot_l = measure_distance(ball_center, foot_left)
                d_foot_r = measure_distance(ball_center, foot_right)
                d_foot_c = measure_distance(ball_center, foot_center)
                d_center = measure_distance(ball_center, center)
                d_bottom = measure_distance(ball_bottom, foot_center)

                # Ambil jarak minimum dari semua titik referensi
                d = min(d_foot_l, d_foot_r, d_foot_c, d_center, d_bottom)

            if best_dist is None or d < best_dist:
                best_dist = d
                best_player = tid

        if best_dist is None or best_dist > self.max_player_ball_distance:
            return -1, best_dist

        return best_player, best_dist
