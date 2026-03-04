# player_ball_assigner.py
# Menentukan pemain mana yang sedang menguasai bola (possession)
# berdasarkan kedekatan bounding box bola dengan area kaki pemain.

import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox, get_foot_position
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class PlayerBallAssigner:
    def __init__(
        self,
        max_possession_distance: float = 70.0,
        foot_region_ratio: float = 0.3
    ):
        """
        Inisialisasi parameter possession assignment.

        Args:
            max_possession_distance: jarak maksimal (px) antara bola dan kaki
                                     pemain agar dianggap menguasai bola
            foot_region_ratio      : proporsi bounding box pemain yang dihitung
                                     sebagai area kaki (dari bawah)
        """
        self.max_possession_distance = max_possession_distance
        self.foot_region_ratio = foot_region_ratio

    def _get_foot_region_center(self, bbox: List[float]) -> Tuple[int, int]:
        """
        Hitung pusat area kaki pemain (bagian bawah bounding box).
        Lebih akurat dari center box karena bola di tanah.
        """
        x1, y1, x2, y2 = bbox
        box_height = y2 - y1
        foot_y = y2 - (box_height * self.foot_region_ratio * 0.5)
        center_x = (x1 + x2) / 2
        return int(center_x), int(foot_y)

    def assign_ball_to_player(
        self,
        players: Dict[int, Dict[str, Any]],
        ball_bbox: Optional[List[float]]
    ) -> int:
        """
        Tentukan pemain yang menguasai bola pada satu frame.

        Args:
            players  : dict {track_id: {'bbox': [...], ...}} dari tracks['players'][frame]
            ball_bbox: bounding box bola [x1, y1, x2, y2] atau None

        Returns:
            track_id pemain yang menguasai bola, atau -1 jika tidak ada
        """
        if ball_bbox is None or not players:
            return -1

        ball_center = get_center_of_bbox(ball_bbox)
        ball_pos = np.array(ball_center, dtype=np.float32)

        min_distance = float('inf')
        assigned_player = -1

        for player_id, player_data in players.items():
            player_bbox = player_data.get('bbox')
            if player_bbox is None:
                continue

            # Gunakan pusat area kaki untuk akurasi lebih baik
            foot_center = self._get_foot_region_center(player_bbox)
            foot_pos = np.array(foot_center, dtype=np.float32)

            # Hitung jarak bola ke kaki
            distance = float(np.linalg.norm(ball_pos - foot_pos))

            if distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        # Hanya assign jika dalam radius maksimal
        if min_distance <= self.max_possession_distance:
            return assigned_player

        return -1

    def assign_ball_to_players_bulk(
        self,
        tracks: Dict,
        debug_interval: int = 100
    ) -> List[int]:
        """
        Proses semua frame dan tentukan possession per frame.

        Args:
            tracks        : dict hasil tracker
            debug_interval: interval frame untuk print progress

        Returns:
            List[int] - possession[frame] = player_id atau -1
        """
        total_frames = len(tracks['players'])
        ball_possessions = []

        possessed_count = 0

        for frame_num in range(total_frames):
            if frame_num % debug_interval == 0:
                print(f"[ASSIGNER] Frame {frame_num}/{total_frames}...")

            players = tracks['players'][frame_num]
            ball_frame = tracks['ball'][frame_num].get(1)
            ball_bbox = ball_frame['bbox'] if ball_frame else None

            assigned = self.assign_ball_to_player(players, ball_bbox)
            ball_possessions.append(assigned)

            if assigned != -1:
                possessed_count += 1

        possession_pct = possessed_count / total_frames * 100 if total_frames > 0 else 0
        print(f"\n[ASSIGNER] Selesai: {possessed_count}/{total_frames} frame dengan possession "
              f"({possession_pct:.1f}%)")

        return ball_possessions