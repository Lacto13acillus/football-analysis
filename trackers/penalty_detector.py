# penalty_detector.py
# Mendeteksi event tendangan penalty dan mengevaluasi gol/tidak gol.
#
# Pendekatan: VELOCITY-BASED (bukan possession-based)
#   1. Deteksi lonjakan kecepatan bola = kick event
#   2. Cari pemain terdekat ke bola SEBELUM kick = penendang
#   3. Setelah bola ditendang, cek apakah bola masuk gawang
#   4. GOAL = bola masuk area bounding box gawang
#   5. MISS = bola tidak masuk gawang

import sys
sys.path.append('../')

from utils.bbox_utils import (
    measure_distance,
    get_center_of_bbox,
    get_center_of_bbox_bottom,
)
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class PenaltyDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps

        # --- Parameter deteksi kick (velocity-based) ---
        self.velocity_window       = 3     # frame window untuk hitung velocity
        self.velocity_threshold    = 15.0  # px/frame — lonjakan = tendangan
        self.pre_kick_search       = 10    # frame sebelum kick untuk cari penendang
        self.max_kicker_distance   = 200   # jarak max pemain ke bola agar dianggap penendang

        # --- Parameter cooldown ---
        self.cooldown_frames       = 60    # cooldown antara 2 tendangan

        # --- Parameter deteksi gol ---
        self.goal_check_window     = 60    # frame setelah kick untuk cek gol
        self.gawang_overlap_margin = 25    # margin pixel untuk overlap

        # --- Filter pemain (hanya penendang) ---
        self.kicker_jerseys = {"Merah", "Abu-Abu"}

        # --- Parameter display ---
        self.kick_display_duration = 50

        self._player_identifier = None

    def set_jersey_map(self, player_identifier) -> None:
        self._player_identifier = player_identifier

    def _get_jersey(self, player_id: int) -> str:
        if self._player_identifier:
            return self._player_identifier.get_jersey_number_for_player(player_id)
        return str(player_id)

    # ============================================================
    # STABILISASI POSISI GAWANG
    # ============================================================

    def get_stable_gawang_bbox(
        self,
        tracks: Dict,
        sample_frames: int = 0
    ) -> Optional[List[float]]:
        """
        Hitung posisi rata-rata gawang dari semua frame.
        """
        gawang_bboxes = []
        total = len(tracks.get('gawang', []))
        if sample_frames > 0:
            total = min(sample_frames, total)

        for f in range(total):
            gawang_data = tracks['gawang'][f].get(1)
            if gawang_data and 'bbox' in gawang_data:
                gawang_bboxes.append(gawang_data['bbox'])

        if not gawang_bboxes:
            return None

        avg_bbox = [
            float(np.mean([b[0] for b in gawang_bboxes])),
            float(np.mean([b[1] for b in gawang_bboxes])),
            float(np.mean([b[2] for b in gawang_bboxes])),
            float(np.mean([b[3] for b in gawang_bboxes])),
        ]

        print(f"[PENALTY] Gawang stabil bbox: "
              f"({avg_bbox[0]:.0f}, {avg_bbox[1]:.0f}) - "
              f"({avg_bbox[2]:.0f}, {avg_bbox[3]:.0f})")
        return avg_bbox

    # ============================================================
    # HITUNG KECEPATAN BOLA
    # ============================================================

    def compute_ball_velocities(
        self,
        tracks: Dict
    ) -> List[float]:
        """
        Hitung kecepatan bola (px/frame) untuk setiap frame.
        Menggunakan sliding window untuk smoothing.
        """
        total_frames = len(tracks['ball'])
        velocities = [0.0] * total_frames

        positions = []
        for f in range(total_frames):
            ball_data = tracks['ball'][f].get(1)
            if ball_data and 'bbox' in ball_data:
                pos = get_center_of_bbox(ball_data['bbox'])
                positions.append(pos)
            else:
                positions.append(None)

        for f in range(self.velocity_window, total_frames):
            pos_now  = positions[f]
            pos_prev = positions[f - self.velocity_window]

            if pos_now is not None and pos_prev is not None:
                dist = measure_distance(pos_now, pos_prev)
                velocities[f] = dist / self.velocity_window

        return velocities

    # ============================================================
    # CEK BOLA MASUK GAWANG
    # ============================================================

    def check_ball_in_gawang(
        self,
        tracks: Dict,
        frame_start: int,
        frame_end: int,
        stable_gawang_bbox: Optional[List[float]] = None
    ) -> Tuple[bool, int, str]:
        """
        Cek apakah bola masuk ke area gawang antara frame_start dan frame_end.

        Returns:
            (is_goal, goal_frame, reason)
        """
        total_frames = len(tracks['ball'])
        check_end = min(frame_end, total_frames - 1)
        margin = self.gawang_overlap_margin

        for f in range(frame_start, check_end + 1):
            ball_data = tracks['ball'][f].get(1)
            if not ball_data or 'bbox' not in ball_data:
                continue

            ball_bbox = ball_data['bbox']
            ball_cx, ball_cy = get_center_of_bbox(ball_bbox)

            # Gunakan gawang stabil atau per-frame
            if stable_gawang_bbox:
                gx1, gy1, gx2, gy2 = stable_gawang_bbox
            else:
                gawang_data = tracks['gawang'][f].get(1)
                if not gawang_data or 'bbox' not in gawang_data:
                    continue
                gx1, gy1, gx2, gy2 = gawang_data['bbox']

            # Cek apakah pusat bola berada di dalam area gawang (+ margin)
            if (gx1 - margin <= ball_cx <= gx2 + margin and
                gy1 - margin <= ball_cy <= gy2 + margin):
                return True, f, f"Bola masuk gawang di frame {f}"

        return False, -1, "Bola tidak masuk gawang"

    # ============================================================
    # CARI PENENDANG TERDEKAT
    # ============================================================

    def find_kicker(
        self,
        tracks: Dict,
        kick_frame: int
    ) -> Tuple[Optional[int], Optional[str], Optional[Tuple[int, int]]]:
        """
        Cari pemain (Merah/Abu-Abu) yang paling dekat ke bola
        pada beberapa frame SEBELUM kick.

        Returns:
            (kicker_track_id, kicker_jersey, kicker_position) atau (None, None, None)
        """
        total_frames = len(tracks['players'])
        search_start = max(0, kick_frame - self.pre_kick_search)
        search_end   = min(kick_frame + 1, total_frames)

        best_player_id = None
        best_jersey    = None
        best_distance  = float('inf')
        best_position  = None

        for f in range(search_start, search_end):
            ball_data = tracks['ball'][f].get(1)
            if not ball_data or 'bbox' not in ball_data:
                continue
            ball_pos = get_center_of_bbox(ball_data['bbox'])

            for player_id, player_data in tracks['players'][f].items():
                bbox = player_data.get('bbox')
                if bbox is None:
                    continue

                jersey = self._get_jersey(player_id)

                # Hanya pertimbangkan pemain target (Merah/Abu-Abu)
                if jersey not in self.kicker_jerseys:
                    continue

                # Gunakan posisi kaki (bottom center)
                foot_pos = get_center_of_bbox_bottom(bbox)
                dist = measure_distance(foot_pos, ball_pos)

                if dist < best_distance:
                    best_distance  = dist
                    best_player_id = player_id
                    best_jersey    = jersey
                    best_position  = foot_pos

        if best_distance <= self.max_kicker_distance and best_player_id is not None:
            return best_player_id, best_jersey, best_position
        return None, None, None

    # ============================================================
    # DETEKSI PENALTY UTAMA — VELOCITY-BASED
    # ============================================================

    def detect_penalties(
        self,
        tracks           : Dict,
        ball_possessions : List[int],  # Tetap ada untuk kompatibilitas
        player_identifier = None,
        debug            : bool = True
    ) -> List[Dict]:
        """
        Deteksi semua event tendangan penalty menggunakan velocity spike.

        Logika:
        1. Hitung kecepatan bola per frame
        2. Deteksi frame di mana velocity melonjak = kick event
        3. Cari pemain Merah/Abu-Abu terdekat ke bola sebelum kick
        4. Setelah kick, cek apakah bola masuk gawang
        """
        if player_identifier:
            self._player_identifier = player_identifier

        total_frames = len(tracks['ball'])

        if debug:
            print(f"\n[PENALTY] === PENALTY DETECTION (VELOCITY-BASED) ===")
            print(f"[PENALTY] Total frames         : {total_frames}")
            print(f"[PENALTY] Velocity threshold    : {self.velocity_threshold} px/frame")
            print(f"[PENALTY] Pre-kick search       : {self.pre_kick_search} frames")
            print(f"[PENALTY] Max kicker distance   : {self.max_kicker_distance}px")
            print(f"[PENALTY] Goal check window     : {self.goal_check_window} frames")
            print(f"[PENALTY] Cooldown              : {self.cooldown_frames} frames")

        # Stabilisasi gawang
        stable_gawang = self.get_stable_gawang_bbox(tracks)
        if not stable_gawang:
            print("[PENALTY] WARNING: Gawang tidak terdeteksi!")
            return []

        # Hitung velocity bola
        velocities = self.compute_ball_velocities(tracks)

        if debug:
            max_vel = max(velocities) if velocities else 0
            avg_vel = np.mean(velocities) if velocities else 0
            print(f"[PENALTY] Max velocity          : {max_vel:.1f} px/frame")
            print(f"[PENALTY] Avg velocity          : {avg_vel:.1f} px/frame")

        # Deteksi kick events (velocity spike)
        penalties = []
        last_kick_frame = -999

        for f in range(total_frames):
            velocity = velocities[f]

            # Cek apakah ini velocity spike
            if velocity < self.velocity_threshold:
                continue

            # Cek cooldown
            if (f - last_kick_frame) < self.cooldown_frames:
                continue

            # Pastikan ini AWAL dari spike (frame sebelumnya velocity rendah)
            if f > 0 and velocities[f - 1] >= self.velocity_threshold * 0.7:
                continue

            if debug:
                print(f"[PENALTY] Velocity spike di frame {f}: {velocity:.1f} px/frame")

            # Cari penendang
            kicker_id, kicker_jersey, kicker_pos = self.find_kicker(tracks, f)

            if kicker_id is None:
                if debug:
                    print(f"[PENALTY]   -> Skip: tidak ada penendang Merah/Abu-Abu terdekat")
                continue

            if debug:
                print(f"[PENALTY]   -> Penendang: {kicker_jersey} (track {kicker_id})")

            # Cek gol
            check_end = min(f + self.goal_check_window, total_frames - 1)
            is_goal, goal_frame, reason = self.check_ball_in_gawang(
                tracks, f, check_end, stable_gawang_bbox=stable_gawang
            )

            # Posisi bola saat kick
            ball_pos_kick = None
            ball_data = tracks['ball'][f].get(1)
            if ball_data:
                ball_pos_kick = get_center_of_bbox(ball_data['bbox'])

            penalty_event = {
                'frame_kick'     : f,
                'frame_goal'     : goal_frame if is_goal else -1,
                'frame_display'  : f + 5,
                'kicker_id'      : kicker_id,
                'kicker_jersey'  : kicker_jersey,
                'is_goal'        : is_goal,
                'reason'         : reason,
                'ball_velocity'  : velocity,
                'kicker_pos'     : kicker_pos,
                'ball_pos_kick'  : ball_pos_kick,
                'gawang_bbox'    : stable_gawang,
            }
            penalties.append(penalty_event)
            last_kick_frame = f

            if debug:
                status = "GOL!" if is_goal else "MISS"
                print(f"[PENALTY]   -> {status} | velocity={velocity:.1f} | {reason}")

        if debug:
            gol  = sum(1 for p in penalties if p['is_goal'])
            miss = sum(1 for p in penalties if not p['is_goal'])
            print(f"\n[PENALTY] === HASIL AKHIR ===")
            print(f"[PENALTY] Total tendangan : {len(penalties)}")
            print(f"[PENALTY] GOL             : {gol}")
            print(f"[PENALTY] MISS            : {miss}")
            print(f"[PENALTY] =========================\n")

        return penalties

    # ============================================================
    # STATISTIK
    # ============================================================

    def get_penalty_statistics(self, penalties: List[Dict]) -> Dict:
        total = len(penalties)
        gol   = [p for p in penalties if p['is_goal']]
        miss  = [p for p in penalties if not p['is_goal']]

        per_player: Dict[str, Dict] = {}
        for p in penalties:
            jersey = p['kicker_jersey']
            if jersey not in per_player:
                per_player[jersey] = {
                    'total'    : 0,
                    'goals'    : 0,
                    'misses'   : 0,
                    'goal_pct' : 0.0,
                }
            per_player[jersey]['total'] += 1
            if p['is_goal']:
                per_player[jersey]['goals'] += 1
            else:
                per_player[jersey]['misses'] += 1

        for jersey, stat in per_player.items():
            stat['goal_pct'] = round(
                stat['goals'] / stat['total'] * 100, 1
            ) if stat['total'] > 0 else 0.0

        return {
            'total_kicks'  : total,
            'total_goals'  : len(gol),
            'total_misses' : len(miss),
            'goal_pct'     : round(len(gol) / total * 100, 1) if total > 0 else 0.0,
            'per_player'   : per_player
        }
