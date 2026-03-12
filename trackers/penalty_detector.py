# penalty_detector.py
# Mendeteksi event tendangan penalty dan mengevaluasi gol/tidak gol.
#
# Pendekatan: STATIONARY-TO-MOVING detection
#   1. Smooth posisi bola untuk hilangkan noise
#   2. Deteksi saat bola berubah dari DIAM ke BERGERAK CEPAT = kick
#   3. Cari pemain Merah/Abu-Abu terdekat SEBELUM kick = penendang
#   4. Setelah kick, cek apakah bola masuk gawang dan TETAP di sana
#   5. GOAL = bola masuk area gawang dan tidak memantul kembali
#   6. MISS = bola tidak masuk / memantul kembali (saved)

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

        # --- Parameter smoothing bola ---
        self.ball_smooth_window     = 5     # rolling average window

        # --- Parameter deteksi kick (stationary → moving) ---
        self.velocity_window        = 3     # frame window untuk hitung velocity
        self.stationary_threshold   = 5.0   # px/frame — bola dianggap diam
        self.kick_velocity_threshold = 15.0 # px/frame — lonjakan = tendangan
        self.stationary_min_frames  = 5     # minimal frame bola diam sebelum kick

        # --- Parameter pencarian penendang ---
        self.pre_kick_search        = 30    # frame sebelum kick untuk cari penendang
        self.max_kicker_distance    = 500   # jarak max pemain ke bola

        # --- Parameter cooldown ---
        self.cooldown_frames        = 90    # cooldown antara 2 tendangan (~1.5s@60fps)

        # --- Parameter deteksi gol ---
        self.goal_check_window      = 50    # frame setelah kick untuk cek gol
        self.gawang_shrink_ratio    = 0.10  # shrink gawang bbox 10% dari setiap sisi
        self.goal_min_frames_inside = 5     # minimal frame bola di dalam gawang
        self.bounce_back_check      = 25    # frame setelah masuk gawang untuk cek bounce

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
    # SMOOTH POSISI BOLA
    # ============================================================

    def _get_ball_positions(self, tracks: Dict) -> List[Optional[Tuple[float, float]]]:
        """Ambil posisi pusat bola per frame."""
        total = len(tracks['ball'])
        positions = []
        for f in range(total):
            ball_data = tracks['ball'][f].get(1)
            if ball_data and 'bbox' in ball_data:
                pos = get_center_of_bbox(ball_data['bbox'])
                positions.append((float(pos[0]), float(pos[1])))
            else:
                positions.append(None)
        return positions

    def _smooth_ball_positions(
        self,
        positions: List[Optional[Tuple[float, float]]]
    ) -> List[Optional[Tuple[float, float]]]:
        """Smooth posisi bola menggunakan rolling average."""
        n = len(positions)
        smoothed = list(positions)
        half_w = self.ball_smooth_window // 2

        for i in range(half_w, n - half_w):
            window_x = []
            window_y = []
            for j in range(i - half_w, i + half_w + 1):
                if positions[j] is not None:
                    window_x.append(positions[j][0])
                    window_y.append(positions[j][1])
            if window_x:
                smoothed[i] = (float(np.mean(window_x)), float(np.mean(window_y)))

        return smoothed

    # ============================================================
    # HITUNG KECEPATAN BOLA
    # ============================================================

    def compute_ball_velocities(
        self,
        positions: List[Optional[Tuple[float, float]]]
    ) -> List[float]:
        total = len(positions)
        velocities = [0.0] * total

        for f in range(self.velocity_window, total):
            pos_now  = positions[f]
            pos_prev = positions[f - self.velocity_window]

            if pos_now is not None and pos_prev is not None:
                dist = measure_distance(pos_now, pos_prev)
                velocities[f] = dist / self.velocity_window

        return velocities

    # ============================================================
    # DETEKSI KICK: STATIONARY → MOVING TRANSITION
    # ============================================================

    def detect_kick_frames(
        self,
        velocities: List[float],
        debug: bool = True
    ) -> List[int]:
        """
        Deteksi frame di mana bola berubah dari DIAM ke BERGERAK CEPAT.

        Logika:
        - Bola dianggap DIAM jika velocity < stationary_threshold
          selama minimal stationary_min_frames frame berturut-turut
        - Bola dianggap DITENDANG jika velocity melompat > kick_velocity_threshold
          setelah periode diam tersebut
        """
        kick_frames = []
        last_kick = -999
        n = len(velocities)

        for f in range(self.stationary_min_frames, n):
            # Cek apakah velocity saat ini tinggi (=kick)
            if velocities[f] < self.kick_velocity_threshold:
                continue

            # Cooldown
            if (f - last_kick) < self.cooldown_frames:
                continue

            # Cek apakah beberapa frame SEBELUMNYA bola diam
            stationary_count = 0
            for prev_f in range(f - 1, max(-1, f - self.stationary_min_frames - 5), -1):
                if velocities[prev_f] < self.stationary_threshold:
                    stationary_count += 1
                else:
                    break

            if stationary_count >= self.stationary_min_frames:
                kick_frames.append(f)
                last_kick = f
                if debug:
                    print(f"[PENALTY] Kick detected at frame {f}: "
                          f"velocity={velocities[f]:.1f} px/frame, "
                          f"stationary {stationary_count} frames before")

        return kick_frames

    # ============================================================
    # CEK BOLA MASUK GAWANG (IMPROVED)
    # ============================================================

    def check_ball_in_gawang(
        self,
        positions: List[Optional[Tuple[float, float]]],
        kick_frame: int,
        stable_gawang_bbox: List[float],
        debug: bool = False
    ) -> Tuple[bool, int, str]:
        """
        Cek apakah bola masuk gawang setelah tendangan.

        Logika improved:
        1. Shrink gawang bbox agar tidak terlalu besar
        2. Cek apakah bola masuk area gawang yang di-shrink
        3. Cek apakah bola TETAP di area gawang (tidak memantul kembali)
           - Jika bola masuk lalu kembali ke bawah = SAVED (MISS)
           - Jika bola masuk dan tetap/hilang = GOL
        """
        gx1, gy1, gx2, gy2 = stable_gawang_bbox
        gw = gx2 - gx1
        gh = gy2 - gy1

        # Shrink gawang bbox
        shrink_x = gw * self.gawang_shrink_ratio
        shrink_y = gh * self.gawang_shrink_ratio
        inner_gx1 = gx1 + shrink_x
        inner_gy1 = gy1 + shrink_y
        inner_gx2 = gx2 - shrink_x
        inner_gy2 = gy2 - shrink_y

        total = len(positions)
        check_start = kick_frame
        check_end = min(kick_frame + self.goal_check_window, total - 1)

        # Track kapan bola masuk gawang
        entered_gawang = False
        enter_frame = -1
        frames_inside = 0

        # Posisi bola saat kick (sebagai referensi arah)
        kick_pos = positions[kick_frame] if kick_frame < total else None

        for f in range(check_start, check_end + 1):
            pos = positions[f]
            if pos is None:
                continue

            bx, by = pos

            inside = (inner_gx1 <= bx <= inner_gx2 and
                      inner_gy1 <= by <= inner_gy2)

            if inside:
                frames_inside += 1
                if not entered_gawang:
                    entered_gawang = True
                    enter_frame = f
                    if debug:
                        print(f"[PENALTY]   Bola masuk gawang di frame {f} "
                              f"({bx:.0f}, {by:.0f})")

        if not entered_gawang or frames_inside < self.goal_min_frames_inside:
            return False, -1, (f"Bola tidak masuk gawang "
                               f"(inside={frames_inside} frames, "
                               f"min={self.goal_min_frames_inside})")

        # Cek apakah bola BOUNCE BACK setelah masuk gawang
        # Jika bola kembali ke posisi Y yang lebih rendah (lebih dekat penendang)
        # = keeper menangkap/memantulkan = MISS
        bounce_check_start = enter_frame
        bounce_check_end = min(enter_frame + self.bounce_back_check, total - 1)

        # Posisi Y saat masuk gawang
        enter_pos = positions[enter_frame]
        if enter_pos is None:
            return True, enter_frame, f"Bola masuk gawang di frame {enter_frame}"

        enter_y = enter_pos[1]
        gawang_center_y = (gy1 + gy2) / 2

        # Hitung berapa banyak frame bola di BAWAH gawang setelah masuk
        # (= bounce back ke arah penendang)
        frames_below_gawang = 0
        for f in range(bounce_check_start + 5, bounce_check_end + 1):
            pos = positions[f]
            if pos is None:
                continue
            bx, by = pos
            # Jika bola kembali ke bawah gawang (y > gy2 = di bawah gawang)
            if by > inner_gy2 + 30:
                frames_below_gawang += 1

        if frames_below_gawang >= 5:
            return False, enter_frame, (f"Bola masuk gawang frame {enter_frame} "
                                        f"tapi memantul kembali "
                                        f"({frames_below_gawang} frames di bawah)")

        return True, enter_frame, f"Bola masuk gawang di frame {enter_frame}"

    # ============================================================
    # CARI PENENDANG TERDEKAT (IMPROVED)
    # ============================================================

    def find_kicker(
        self,
        tracks: Dict,
        positions: List[Optional[Tuple[float, float]]],
        kick_frame: int,
        debug: bool = False
    ) -> Tuple[Optional[int], Optional[str], Optional[Tuple[int, int]]]:
        """
        Cari pemain (Merah/Abu-Abu) yang paling dekat ke bola
        pada frame-frame SEBELUM kick (saat bola masih diam).

        Improved:
        - Cari di range yang lebih luas (pre_kick_search = 30 frames)
        - Prioritaskan frame di mana bola masih diam (pemain sedang ancang-ancang)
        - Max distance diperbesar (500px)
        """
        total_frames = len(tracks['players'])
        search_start = max(0, kick_frame - self.pre_kick_search)
        search_end   = min(kick_frame + 1, total_frames)

        best_player_id = None
        best_jersey    = None
        best_distance  = float('inf')
        best_position  = None
        best_frame     = -1

        for f in range(search_start, search_end):
            ball_pos = positions[f] if f < len(positions) else None
            if ball_pos is None:
                continue

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

                # Juga cek center bbox (kadang foot position kurang akurat)
                center_pos = get_center_of_bbox(bbox)
                dist_center = measure_distance(center_pos, ball_pos)
                dist = min(dist, dist_center)

                if dist < best_distance:
                    best_distance  = dist
                    best_player_id = player_id
                    best_jersey    = jersey
                    best_position  = foot_pos
                    best_frame     = f

        if best_distance <= self.max_kicker_distance and best_player_id is not None:
            if debug:
                print(f"[PENALTY]   Kicker found: {best_jersey} (track {best_player_id}) "
                      f"at frame {best_frame}, distance={best_distance:.0f}px")
            return best_player_id, best_jersey, best_position
        else:
            if debug:
                print(f"[PENALTY]   No kicker found within {self.max_kicker_distance}px "
                      f"(best={best_distance:.0f}px, jersey={best_jersey})")
        return None, None, None

    # ============================================================
    # DETEKSI PENALTY UTAMA
    # ============================================================

    def detect_penalties(
        self,
        tracks           : Dict,
        ball_possessions : List[int],
        player_identifier = None,
        debug            : bool = True
    ) -> List[Dict]:
        if player_identifier:
            self._player_identifier = player_identifier

        total_frames = len(tracks['ball'])

        if debug:
            print(f"\n[PENALTY] === PENALTY DETECTION (STATIONARY→MOVING) ===")
            print(f"[PENALTY] Total frames              : {total_frames}")
            print(f"[PENALTY] Stationary threshold       : {self.stationary_threshold} px/frame")
            print(f"[PENALTY] Kick velocity threshold    : {self.kick_velocity_threshold} px/frame")
            print(f"[PENALTY] Stationary min frames      : {self.stationary_min_frames}")
            print(f"[PENALTY] Pre-kick search            : {self.pre_kick_search} frames")
            print(f"[PENALTY] Max kicker distance         : {self.max_kicker_distance}px")
            print(f"[PENALTY] Goal check window           : {self.goal_check_window} frames")
            print(f"[PENALTY] Gawang shrink ratio         : {self.gawang_shrink_ratio}")
            print(f"[PENALTY] Cooldown                    : {self.cooldown_frames} frames")

        # Stabilisasi gawang
        stable_gawang = self.get_stable_gawang_bbox(tracks)
        if not stable_gawang:
            print("[PENALTY] WARNING: Gawang tidak terdeteksi!")
            return []

        # Ambil dan smooth posisi bola
        raw_positions = self._get_ball_positions(tracks)
        positions = self._smooth_ball_positions(raw_positions)

        # Hitung velocity pada posisi yang sudah di-smooth
        velocities = self.compute_ball_velocities(positions)

        if debug:
            max_vel = max(velocities) if velocities else 0
            avg_vel = np.mean(velocities) if velocities else 0
            print(f"[PENALTY] Max velocity (smoothed)     : {max_vel:.1f} px/frame")
            print(f"[PENALTY] Avg velocity (smoothed)     : {avg_vel:.1f} px/frame")

        # Deteksi kick frames (stationary → moving)
        kick_frames = self.detect_kick_frames(velocities, debug=debug)

        if debug:
            print(f"[PENALTY] Total kick frames detected  : {len(kick_frames)}")

        # Untuk setiap kick, cari penendang dan cek gol
        penalties = []

        for kick_frame in kick_frames:
            if debug:
                print(f"\n[PENALTY] --- Processing kick at frame {kick_frame} ---")

            # Cari penendang
            kicker_id, kicker_jersey, kicker_pos = self.find_kicker(
                tracks, positions, kick_frame, debug=debug
            )

            if kicker_id is None:
                if debug:
                    print(f"[PENALTY]   -> Skip: tidak ada penendang terdekat")
                continue

            # Cek gol
            is_goal, goal_frame, reason = self.check_ball_in_gawang(
                positions, kick_frame, stable_gawang, debug=debug
            )

            # Posisi bola saat kick
            ball_pos_kick = positions[kick_frame]

            penalty_event = {
                'frame_kick'     : kick_frame,
                'frame_goal'     : goal_frame if is_goal else -1,
                'frame_display'  : kick_frame + 5,
                'kicker_id'      : kicker_id,
                'kicker_jersey'  : kicker_jersey,
                'is_goal'        : is_goal,
                'reason'         : reason,
                'ball_velocity'  : velocities[kick_frame],
                'kicker_pos'     : kicker_pos,
                'ball_pos_kick'  : ball_pos_kick,
                'gawang_bbox'    : stable_gawang,
            }
            penalties.append(penalty_event)

            if debug:
                status = "GOL!" if is_goal else "MISS"
                print(f"[PENALTY]   -> {kicker_jersey}: {status} "
                      f"| velocity={velocities[kick_frame]:.1f} | {reason}")

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
