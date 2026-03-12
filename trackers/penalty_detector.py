# penalty_detector.py
# Mendeteksi event tendangan penalty dan mengevaluasi gol/tidak gol.
#
# Pendekatan: VELOCITY SPIKE + COOLDOWN
#   1. Hitung velocity bola per frame
#   2. Deteksi frame velocity > threshold, filter dengan cooldown
#   3. Cari player terdekat ke bola SEBELUM kick (TANPA filter jersey)
#   4. Re-detect warna baju penendang langsung dari frame
#   5. Cek apakah bola masuk gawang (shrink bbox + bounce-back check)

import sys
sys.path.append('../')

from utils.bbox_utils import (
    measure_distance,
    get_center_of_bbox,
    get_center_of_bbox_bottom,
)
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class PenaltyDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps

        # --- Parameter deteksi kick ---
        self.velocity_window         = 3     # frame window untuk hitung velocity
        self.kick_velocity_threshold = 15.0  # px/frame — spike = tendangan

        # --- Parameter pencarian penendang ---
        self.pre_kick_search         = 50    # frame sebelum kick untuk cari penendang
        self.max_kicker_distance     = 700   # jarak max player ke bola (portrait video besar)

        # --- Parameter cooldown ---
        self.cooldown_frames         = 120   # ~2 detik pada 60fps

        # --- Parameter deteksi gol ---
        self.goal_check_window       = 60    # frame setelah kick
        self.gawang_shrink_ratio     = 0.10  # shrink gawang bbox 10% tiap sisi
        self.goal_min_frames_inside  = 3     # min frame bola di dalam gawang
        self.bounce_back_check       = 30    # frame cek bola memantul

        # --- Display ---
        self.kick_display_duration   = 50

        self._player_identifier = None

    def set_jersey_map(self, player_identifier) -> None:
        self._player_identifier = player_identifier

    def _get_jersey(self, player_id: int) -> str:
        if self._player_identifier:
            return self._player_identifier.get_jersey_number_for_player(player_id)
        return str(player_id)

    # ============================================================
    # DETEKSI WARNA BAJU (langsung dari frame)
    # ============================================================

    @staticmethod
    def detect_shirt_color_from_frame(
        frame: np.ndarray,
        bbox: List[float]
    ) -> str:
        """Deteksi warna baju dari bbox pemain di frame tertentu."""
        x1, y1, x2, y2 = map(int, bbox)
        h_box = y2 - y1
        shirt_y1 = y1 + int(h_box * 0.10)
        shirt_y2 = y1 + int(h_box * 0.45)
        margin_x = int((x2 - x1) * 0.15)
        shirt_x1 = max(0, x1 + margin_x)
        shirt_x2 = min(frame.shape[1], x2 - margin_x)

        if shirt_x2 <= shirt_x1 or shirt_y2 <= shirt_y1:
            return "Unknown"

        shirt_region = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
        if shirt_region.size == 0:
            return "Unknown"

        hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)

        mask_red1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([12, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 60, 50]), np.array([180, 255, 255]))
        mask_red = mask_red1 | mask_red2
        red_ratio = np.count_nonzero(mask_red) / max(mask_red.size, 1)

        mask_gray = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 60, 180]))
        gray_ratio = np.count_nonzero(mask_gray) / max(mask_gray.size, 1)

        if red_ratio > 0.25:
            return "Merah"
        elif gray_ratio > 0.35:
            return "Abu-Abu"
        return "Unknown"

    # ============================================================
    # STABILISASI POSISI GAWANG
    # ============================================================

    def get_stable_gawang_bbox(self, tracks: Dict) -> Optional[List[float]]:
        gawang_bboxes = []
        total = len(tracks.get('gawang', []))
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
    # HITUNG VELOCITY BOLA
    # ============================================================

    def compute_ball_velocities(self, tracks: Dict) -> List[float]:
        total_frames = len(tracks['ball'])
        velocities = [0.0] * total_frames

        positions = []
        for f in range(total_frames):
            ball_data = tracks['ball'][f].get(1)
            if ball_data and 'bbox' in ball_data:
                positions.append(get_center_of_bbox(ball_data['bbox']))
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
    # DETEKSI KICK FRAMES (velocity spike + cooldown)
    # ============================================================

    def detect_kick_frames(
        self,
        velocities: List[float],
        debug: bool = True
    ) -> List[int]:
        """
        Deteksi frame tendangan menggunakan velocity spike.
        Filter: frame ini > threshold DAN frame sebelumnya < threshold*0.7
        (= awal dari spike, bukan tengah/akhir spike)
        Lalu apply cooldown.
        """
        candidates = []
        n = len(velocities)

        for f in range(1, n):
            if velocities[f] < self.kick_velocity_threshold:
                continue
            # Harus awal dari spike (sebelumnya velocity lebih rendah)
            if velocities[f - 1] >= self.kick_velocity_threshold * 0.7:
                continue
            candidates.append(f)

        # Apply cooldown
        kick_frames = []
        last_kick = -9999
        for f in candidates:
            if f - last_kick >= self.cooldown_frames:
                kick_frames.append(f)
                last_kick = f

        if debug:
            print(f"[PENALTY] Candidates sebelum cooldown: {len(candidates)}")
            print(f"[PENALTY] Kick frames setelah cooldown={self.cooldown_frames}: "
                  f"{len(kick_frames)} -> {kick_frames}")

        return kick_frames

    # ============================================================
    # CEK BOLA MASUK GAWANG
    # ============================================================

    def check_ball_in_gawang(
        self,
        tracks: Dict,
        kick_frame: int,
        stable_gawang_bbox: List[float],
        debug: bool = False
    ) -> Tuple[bool, int, str]:
        gx1, gy1, gx2, gy2 = stable_gawang_bbox
        gw = gx2 - gx1
        gh = gy2 - gy1

        # Shrink gawang
        shrink_x = gw * self.gawang_shrink_ratio
        shrink_y = gh * self.gawang_shrink_ratio
        inner_gx1 = gx1 + shrink_x
        inner_gy1 = gy1 + shrink_y
        inner_gx2 = gx2 - shrink_x
        inner_gy2 = gy2 - shrink_y

        total = len(tracks['ball'])
        check_end = min(kick_frame + self.goal_check_window, total - 1)

        entered_gawang = False
        enter_frame = -1
        frames_inside = 0

        for f in range(kick_frame, check_end + 1):
            ball_data = tracks['ball'][f].get(1)
            if not ball_data or 'bbox' not in ball_data:
                continue

            bx, by = get_center_of_bbox(ball_data['bbox'])

            if (inner_gx1 <= bx <= inner_gx2 and
                inner_gy1 <= by <= inner_gy2):
                frames_inside += 1
                if not entered_gawang:
                    entered_gawang = True
                    enter_frame = f

        if not entered_gawang or frames_inside < self.goal_min_frames_inside:
            return False, -1, (f"Bola tidak masuk gawang "
                               f"(inside={frames_inside} frames)")

        # Cek bounce-back (bola memantul kembali = SAVED)
        bounce_end = min(enter_frame + self.bounce_back_check, total - 1)
        frames_below = 0
        for f in range(enter_frame + 5, bounce_end + 1):
            ball_data = tracks['ball'][f].get(1)
            if not ball_data or 'bbox' not in ball_data:
                continue
            _, by = get_center_of_bbox(ball_data['bbox'])
            if by > inner_gy2 + 30:
                frames_below += 1

        if frames_below >= 5:
            return False, enter_frame, (f"Bola masuk frame {enter_frame} "
                                        f"tapi memantul kembali (saved)")

        return True, enter_frame, f"Bola masuk gawang di frame {enter_frame}"

    # ============================================================
    # CARI PENENDANG — TANPA FILTER JERSEY
    # ============================================================

    def find_kicker(
        self,
        tracks: Dict,
        frames: List[np.ndarray],
        kick_frame: int,
        debug: bool = False
    ) -> Tuple[Optional[int], Optional[str], Optional[Tuple[int, int]]]:
        """
        Cari player MANAPUN yang paling dekat ke bola sebelum kick.
        Lalu re-detect warna baju untuk menentukan Merah/Abu-Abu.
        
        Keeper di-track terpisah (tracks['keeper']), jadi semua yang ada
        di tracks['players'] adalah player yang bisa jadi penendang.
        """
        total_frames = len(tracks['players'])
        search_start = max(0, kick_frame - self.pre_kick_search)
        search_end   = min(kick_frame + 1, total_frames)

        best_player_id = None
        best_distance  = float('inf')
        best_position  = None
        best_frame     = -1
        best_bbox      = None

        for f in range(search_start, search_end):
            ball_data = tracks['ball'][f].get(1)
            if not ball_data or 'bbox' not in ball_data:
                continue
            ball_pos = get_center_of_bbox(ball_data['bbox'])

            for player_id, player_data in tracks['players'][f].items():
                bbox = player_data.get('bbox')
                if bbox is None:
                    continue

                foot_pos = get_center_of_bbox_bottom(bbox)
                center_pos = get_center_of_bbox(bbox)

                dist = min(
                    measure_distance(foot_pos, ball_pos),
                    measure_distance(center_pos, ball_pos)
                )

                if dist < best_distance:
                    best_distance  = dist
                    best_player_id = player_id
                    best_position  = foot_pos
                    best_frame     = f
                    best_bbox      = bbox

        if best_distance > self.max_kicker_distance or best_player_id is None:
            if debug:
                print(f"[PENALTY]   No player within {self.max_kicker_distance}px "
                      f"(best={best_distance:.0f}px)")
            return None, None, None

        # Re-detect warna baju langsung dari frame
        jersey = self._get_jersey(best_player_id)
        if jersey == "Unknown" or jersey.startswith("ID:"):
            # Re-detect dari frame langsung
            if best_frame < len(frames) and best_bbox is not None:
                jersey = self.detect_shirt_color_from_frame(
                    frames[best_frame], best_bbox
                )
                if debug:
                    print(f"[PENALTY]   Re-detected color for track {best_player_id}: {jersey}")

        # Jika masih Unknown, coba beberapa frame di sekitar
        if jersey == "Unknown" or jersey.startswith("ID:"):
            for f in range(max(0, best_frame - 10), min(len(frames), best_frame + 5)):
                pdata = tracks['players'][f].get(best_player_id)
                if pdata and 'bbox' in pdata:
                    color = self.detect_shirt_color_from_frame(frames[f], pdata['bbox'])
                    if color != "Unknown":
                        jersey = color
                        break

        if debug:
            print(f"[PENALTY]   Kicker: {jersey} (track {best_player_id}) "
                  f"at frame {best_frame}, dist={best_distance:.0f}px")

        return best_player_id, jersey, best_position

    # ============================================================
    # DETEKSI PENALTY UTAMA
    # ============================================================

    def detect_penalties(
        self,
        tracks           : Dict,
        ball_possessions : List[int],
        frames           : List[np.ndarray] = None,
        player_identifier = None,
        debug            : bool = True
    ) -> List[Dict]:
        if player_identifier:
            self._player_identifier = player_identifier

        total_frames = len(tracks['ball'])

        if debug:
            print(f"\n[PENALTY] === PENALTY DETECTION (VELOCITY SPIKE) ===")
            print(f"[PENALTY] Total frames              : {total_frames}")
            print(f"[PENALTY] Kick velocity threshold    : {self.kick_velocity_threshold} px/frame")
            print(f"[PENALTY] Pre-kick search            : {self.pre_kick_search} frames")
            print(f"[PENALTY] Max kicker distance        : {self.max_kicker_distance}px")
            print(f"[PENALTY] Goal check window          : {self.goal_check_window} frames")
            print(f"[PENALTY] Gawang shrink ratio        : {self.gawang_shrink_ratio}")
            print(f"[PENALTY] Cooldown                   : {self.cooldown_frames} frames")

        stable_gawang = self.get_stable_gawang_bbox(tracks)
        if not stable_gawang:
            print("[PENALTY] WARNING: Gawang tidak terdeteksi!")
            return []

        velocities = self.compute_ball_velocities(tracks)
        if debug:
            max_vel = max(velocities) if velocities else 0
            avg_vel = np.mean(velocities) if velocities else 0
            print(f"[PENALTY] Max velocity               : {max_vel:.1f} px/frame")
            print(f"[PENALTY] Avg velocity               : {avg_vel:.1f} px/frame")

        kick_frames = self.detect_kick_frames(velocities, debug=debug)

        penalties = []
        for kick_frame in kick_frames:
            if debug:
                print(f"\n[PENALTY] --- Kick at frame {kick_frame} "
                      f"(vel={velocities[kick_frame]:.1f}) ---")

            # Cari penendang (tanpa filter jersey)
            kicker_id, kicker_jersey, kicker_pos = self.find_kicker(
                tracks, frames if frames else [], kick_frame, debug=debug
            )

            if kicker_id is None:
                if debug:
                    print(f"[PENALTY]   -> Skip: tidak ada player terdekat")
                continue

            # Cek gol
            is_goal, goal_frame, reason = self.check_ball_in_gawang(
                tracks, kick_frame, stable_gawang, debug=debug
            )

            ball_pos_kick = None
            ball_data = tracks['ball'][kick_frame].get(1)
            if ball_data:
                ball_pos_kick = get_center_of_bbox(ball_data['bbox'])

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
                print(f"[PENALTY]   -> {kicker_jersey}: {status} | {reason}")

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
                    'total': 0, 'goals': 0, 'misses': 0, 'goal_pct': 0.0,
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
