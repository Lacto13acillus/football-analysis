# player_ball_assigner.py
# Menentukan pemain mana yang sedang menguasai bola (possession)
# berdasarkan kedekatan bounding box bola dengan area kaki pemain.
#
# PERUBAHAN v2.4:
#   - Jersey Priority Bonus: pemain dengan jersey dikenal (#3, #19)
#     mendapat bonus jarak efektif agar tidak kalah dari Unknown
#     saat jaraknya mirip ke bola.
#   - set_player_identifier() untuk menghubungkan ke PlayerIdentifier.

import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox, get_foot_position
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class PlayerBallAssigner:
    def __init__(
        self,
        max_possession_distance: float = 70.0,
        foot_region_ratio: float = 0.3,
        player_identifier=None,
        jersey_priority_bonus: float = 0.70
    ):
        """
        Inisialisasi parameter possession assignment.

        Args:
            max_possession_distance: jarak maksimal (px) antara bola dan kaki
                                     pemain agar dianggap menguasai bola
            foot_region_ratio      : proporsi bounding box pemain yang dihitung
                                     sebagai area kaki (dari bawah)
            player_identifier      : referensi ke PlayerIdentifier (opsional,
                                     bisa di-set nanti via set_player_identifier)
            jersey_priority_bonus  : faktor pengali jarak untuk pemain dengan
                                     jersey dikenal. Nilai < 1.0 membuat pemain
                                     jersey dikenal "lebih dekat" secara efektif.
                                     Contoh: 0.70 = jarak efektif 70% dari asli.
        """
        self.max_possession_distance = max_possession_distance
        self.foot_region_ratio = foot_region_ratio
        self._player_identifier = player_identifier
        self.jersey_priority_bonus = jersey_priority_bonus

        # Jersey yang diprioritaskan (pemain target analisis)
        self.priority_jerseys = {"#3", "#19"}

    def set_player_identifier(self, player_identifier) -> None:
        """
        Set referensi ke PlayerIdentifier untuk jersey lookup.
        Harus dipanggil SEBELUM assign_ball_to_players_bulk() agar
        jersey priority bonus aktif.
        """
        self._player_identifier = player_identifier

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

        PERBAIKAN v2.4 — Jersey Priority Bonus:
        Saat dua pemain jaraknya mirip ke bola, pemain dengan jersey
        dikenal (#3, #19) mendapat bonus sehingga jarak efektifnya
        lebih kecil. Ini mencegah Unknown "mencuri" possession.

        Contoh dengan bonus=0.70:
          Unknown=75px, #19=95px
          Tanpa bonus : Unknown menang (75 < 95)
          Dengan bonus: #19 efektif = 95*0.70 = 66.5px → #19 menang

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

        candidates = []

        for player_id, player_data in players.items():
            player_bbox = player_data.get('bbox')
            if player_bbox is None:
                continue

            # Gunakan pusat area kaki untuk akurasi lebih baik
            foot_center = self._get_foot_region_center(player_bbox)
            foot_pos = np.array(foot_center, dtype=np.float32)

            # Hitung jarak bola ke kaki (jarak asli)
            distance = float(np.linalg.norm(ball_pos - foot_pos))

            # Hitung jarak efektif berdasarkan jersey priority
            effective_distance = distance
            if self._player_identifier:
                jersey = self._player_identifier.get_jersey_number_for_player(player_id)
                if jersey in self.priority_jerseys:
                    effective_distance = distance * self.jersey_priority_bonus

            candidates.append((player_id, distance, effective_distance))

        if not candidates:
            return -1

        # Sort berdasarkan effective_distance (terkecil menang)
        candidates.sort(key=lambda x: x[2])
        best_player_id, best_actual_distance, best_effective_distance = candidates[0]

        # Threshold tetap menggunakan jarak ASLI (bukan efektif)
        # agar tidak assign pada pemain yang benar-benar jauh
        if best_actual_distance <= self.max_possession_distance:
            return best_player_id

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