# player_ball_assigner.py
# Menentukan apakah pemain sedang menguasai bola (possession)
# Versi sederhana untuk skenario 1 pemain.

import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class PlayerBallAssigner:
    def __init__(self, max_possession_distance: float = 130.0):
        self.max_possession_distance = max_possession_distance

    def _get_foot_position(self, bbox: List[float]) -> Tuple[int, int]:
        """Posisi kaki pemain (bottom-center bounding box)."""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    def assign_ball_to_player(
        self,
        players  : Dict[int, Dict[str, Any]],
        ball_bbox: Optional[List[float]]
    ) -> int:
        if ball_bbox is None or not players:
            return -1

        ball_center = get_center_of_bbox(ball_bbox)
        ball_pos = np.array(ball_center, dtype=np.float32)

        best_player = -1
        best_distance = float('inf')

        for player_id, player_data in players.items():
            player_bbox = player_data.get('bbox')
            if player_bbox is None:
                continue

            foot_pos = np.array(self._get_foot_position(player_bbox), dtype=np.float32)
            distance = float(np.linalg.norm(ball_pos - foot_pos))

            if distance < best_distance:
                best_distance = distance
                best_player = player_id

        if best_distance <= self.max_possession_distance:
            return best_player

        return -1

    def assign_ball_to_players_bulk(self, tracks: Dict) -> List[int]:
        """Proses semua frame dan tentukan possession per frame."""
        total_frames = len(tracks['players'])
        ball_possessions = []
        possessed_count = 0

        for frame_num in range(total_frames):
            if frame_num % 200 == 0:
                print(f"[ASSIGNER] Frame {frame_num}/{total_frames}...")

            players = tracks['players'][frame_num]
            ball_frame = tracks['ball'][frame_num].get(1)
            ball_bbox = ball_frame['bbox'] if ball_frame else None

            assigned = self.assign_ball_to_player(players, ball_bbox)
            ball_possessions.append(assigned)

            if assigned != -1:
                possessed_count += 1

        pct = possessed_count / total_frames * 100 if total_frames > 0 else 0
        print(f"[ASSIGNER] Selesai: {possessed_count}/{total_frames} frame "
              f"dengan possession ({pct:.1f}%)")

        return ball_possessions
