# penalty_detector.py
# Mendeteksi event tendangan penalty:
#   1. SHOOT ON TARGET / OFF TARGET
#   2. GOL / SAVED BY KEEPER
#
# Logic:
#   - ON TARGET + bola TIDAK dekat keeper → GOL
#   - ON TARGET + bola dekat keeper → SAVED (tidak gol)
#   - OFF TARGET → tidak gol (melebar/melambung)

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
        self.velocity_window         = 3
        self.kick_velocity_threshold = 15.0

        # --- Parameter pencarian penendang ---
        self.pre_kick_search         = 50
        self.max_kicker_distance     = 700

        # --- Parameter cooldown ---
        self.cooldown_frames         = 120

        # --- Parameter deteksi ON TARGET ---
        self.on_target_check_window  = 60
        self.gawang_shrink_ratio     = 0.05
        self.on_target_min_frames    = 1

        # --- Parameter deteksi KEEPER SAVE ---
        self.keeper_save_check_window = 45    # frame setelah kick untuk cek keeper
        self.keeper_ball_distance_thr = 150   # jarak bola ke keeper bbox = dianggap save
        self.keeper_ball_inside_thr   = 0.3   # ratio overlap bola masuk keeper bbox
        self.keeper_velocity_drop_thr = 3.0   # velocity bola turun drastis = tertangkap
        self.keeper_save_min_frames   = 2     # minimal frame bola dekat keeper

        # --- Display ---
        self.kick_display_duration   = 50

        # --- Manual kick mapping ---
        self._manual_kick_mapping: Dict[int, str] = {}

        self._player_identifier = None

    def set_jersey_map(self, player_identifier) -> None:
        self._player_identifier = player_identifier

    def set_manual_kick_mapping(self, mapping: Dict[int, str]) -> None:
        self._manual_kick_mapping = mapping
        print(f"[PENALTY] Manual kick mapping diset: {self._manual_kick_mapping}")

    def _get_jersey(self, player_id: int) -> str:
        if self._player_identifier:
            return self._player_identifier.get_jersey_number_for_player(player_id)
        return str(player_id)

    # ============================================================
    # DETEKSI WARNA BAJU
    # ============================================================

    @staticmethod
    def detect_shirt_color_from_frame(
        frame: np.ndarray,
        bbox: List[float]
    ) -> str:
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
    # DETEKSI KICK FRAMES
    # ============================================================

    def detect_kick_frames(
        self,
        velocities: List[float],
        debug: bool = True
    ) -> List[int]:
        candidates = []
        n = len(velocities)

        for f in range(1, n):
            if velocities[f] < self.kick_velocity_threshold:
                continue
            if velocities[f - 1] >= self.kick_velocity_threshold * 0.7:
                continue
            candidates.append(f)

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
    # CEK SHOOT ON TARGET
    # ============================================================

    def check_shoot_on_target(
        self,
        tracks: Dict,
        kick_frame: int,
        stable_gawang_bbox: List[float],
        debug: bool = False
    ) -> Tuple[bool, int, str]:
        gx1, gy1, gx2, gy2 = stable_gawang_bbox
        gw = gx2 - gx1
        gh = gy2 - gy1

        shrink_x = gw * self.gawang_shrink_ratio
        shrink_y = gh * self.gawang_shrink_ratio
        inner_gx1 = gx1 + shrink_x
        inner_gy1 = gy1 + shrink_y
        inner_gx2 = gx2 - shrink_x
        inner_gy2 = gy2 - shrink_y

        total = len(tracks['ball'])
        check_end = min(kick_frame + self.on_target_check_window, total - 1)

        frames_inside = 0
        first_hit_frame = -1

        for f in range(kick_frame, check_end + 1):
            ball_data = tracks['ball'][f].get(1)
            if not ball_data or 'bbox' not in ball_data:
                continue

            bx, by = get_center_of_bbox(ball_data['bbox'])

            if (inner_gx1 <= bx <= inner_gx2 and
                inner_gy1 <= by <= inner_gy2):
                frames_inside += 1
                if first_hit_frame == -1:
                    first_hit_frame = f

        if frames_inside >= self.on_target_min_frames:
            return True, first_hit_frame, (
                f"ON TARGET - Bola mengenai gawang di frame {first_hit_frame} "
                f"({frames_inside} frames di area gawang)"
            )
        else:
            return False, -1, (
                f"OFF TARGET - Bola tidak mengenai area gawang "
                f"(hanya {frames_inside} frames di area)"
            )

    # ============================================================
    # CEK KEEPER SAVE — BOLA DITANGKAP/DIBLOK KIPER
    # ============================================================

    def check_keeper_save(
        self,
        tracks: Dict,
        velocities: List[float],
        kick_frame: int,
        debug: bool = False
    ) -> Tuple[bool, int, str]:
        """
        Cek apakah bola ditangkap/diblok oleh keeper setelah tendangan.

        Logic deteksi save:
        1. Bola mendekati bbox keeper (jarak center bola ke center keeper < threshold)
        2. ATAU bola masuk ke dalam bbox keeper (overlap)
        3. DAN velocity bola turun drastis setelah dekat keeper (tertangkap/diblok)

        Returns:
            (is_saved, save_frame, reason)
        """
        total = len(tracks['ball'])
        check_end = min(kick_frame + self.keeper_save_check_window, total - 1)

        frames_near_keeper = 0
        first_near_frame = -1
        min_distance = float('inf')
        velocity_dropped = False

        for f in range(kick_frame, check_end + 1):
            # Data bola
            ball_data = tracks['ball'][f].get(1)
            if not ball_data or 'bbox' not in ball_data:
                continue
            ball_bbox = ball_data['bbox']
            ball_cx, ball_cy = get_center_of_bbox(ball_bbox)

            # Data keeper
            keeper_data = tracks['keeper'][f].get(1)
            if not keeper_data or 'bbox' not in keeper_data:
                continue
            keeper_bbox = keeper_data['bbox']
            kx1, ky1, kx2, ky2 = keeper_bbox
            keeper_cx, keeper_cy = get_center_of_bbox(keeper_bbox)

            # --- Metode 1: Jarak center bola ke center keeper ---
            dist = measure_distance(
                (ball_cx, ball_cy), (keeper_cx, keeper_cy)
            )
            if dist < min_distance:
                min_distance = dist

            # --- Metode 2: Bola masuk ke dalam expanded keeper bbox ---
            # Expand keeper bbox sedikit untuk toleransi
            expand_x = (kx2 - kx1) * 0.2
            expand_y = (ky2 - ky1) * 0.2
            exp_kx1 = kx1 - expand_x
            exp_ky1 = ky1 - expand_y
            exp_kx2 = kx2 + expand_x
            exp_ky2 = ky2 + expand_y

            ball_near = (dist < self.keeper_ball_distance_thr)
            ball_inside = (exp_kx1 <= ball_cx <= exp_kx2 and
                           exp_ky1 <= ball_cy <= exp_ky2)

            if ball_near or ball_inside:
                frames_near_keeper += 1
                if first_near_frame == -1:
                    first_near_frame = f

                # --- Metode 3: Cek velocity drop setelah dekat keeper ---
                # Bola yang ditangkap → velocity turun drastis
                if f + 5 < total:
                    vel_now = velocities[f] if f < len(velocities) else 0
                    # Cek velocity di beberapa frame ke depan
                    future_vels = []
                    for ff in range(f + 2, min(f + 10, len(velocities))):
                        future_vels.append(velocities[ff])
                    if future_vels:
                        avg_future_vel = np.mean(future_vels)
                        if avg_future_vel < self.keeper_velocity_drop_thr:
                            velocity_dropped = True

        # --- Keputusan save ---
        if frames_near_keeper >= self.keeper_save_min_frames:
            if velocity_dropped:
                reason = (f"SAVED - Bola ditangkap keeper di frame {first_near_frame} "
                          f"({frames_near_keeper} frames dekat keeper, "
                          f"velocity drop terdeteksi)")
            else:
                reason = (f"SAVED - Bola diblok keeper di frame {first_near_frame} "
                          f"({frames_near_keeper} frames dekat keeper, "
                          f"jarak min={min_distance:.0f}px)")

            if debug:
                print(f"[PENALTY]   Keeper: {reason}")

            return True, first_near_frame, reason

        # Tidak di-save
        if debug:
            print(f"[PENALTY]   Keeper: Tidak menangkap "
                  f"(dekat={frames_near_keeper} frames, "
                  f"jarak min={min_distance:.0f}px)")

        return False, -1, (
            f"Keeper tidak menangkap "
            f"(dekat={frames_near_keeper} frames, "
            f"jarak min={min_distance:.0f}px)"
        )

    # ============================================================
    # CARI PENENDANG
    # ============================================================

    def find_kicker(
        self,
        tracks: Dict,
        frames: List[np.ndarray],
        kick_frame: int,
        debug: bool = False
    ) -> Tuple[Optional[int], Optional[str], Optional[Tuple[int, int]]]:
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

        # Penentuan jersey
        if kick_frame in self._manual_kick_mapping:
            jersey = self._manual_kick_mapping[kick_frame]
            if debug:
                print(f"[PENALTY]   Jersey dari MANUAL KICK MAPPING: "
                      f"frame {kick_frame} -> {jersey}")
        else:
            jersey = self._get_jersey(best_player_id)

            if jersey == "Unknown" or jersey.startswith("ID:"):
                if best_frame < len(frames) and best_bbox is not None:
                    jersey = self.detect_shirt_color_from_frame(
                        frames[best_frame], best_bbox
                    )
                    if debug:
                        print(f"[PENALTY]   Re-detected color for track "
                              f"{best_player_id}: {jersey}")

            if jersey == "Unknown" or jersey.startswith("ID:"):
                for f in range(max(0, best_frame - 10),
                               min(len(frames), best_frame + 5)):
                    pdata = tracks['players'][f].get(best_player_id)
                    if pdata and 'bbox' in pdata:
                        color = self.detect_shirt_color_from_frame(
                            frames[f], pdata['bbox']
                        )
                        if color != "Unknown":
                            jersey = color
                            break

        if debug:
            manual_tag = " [MANUAL]" if kick_frame in self._manual_kick_mapping else ""
            print(f"[PENALTY]   Kicker: {jersey}{manual_tag} "
                  f"(track {best_player_id}) "
                  f"at frame {best_frame}, dist={best_distance:.0f}px")

        return best_player_id, jersey, best_position

    # ============================================================
    # DETEKSI PENALTY UTAMA — ON TARGET + GOL/SAVED
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
            print(f"\n[PENALTY] === PENALTY DETECTION (ON TARGET + GOL/SAVED) ===")
            print(f"[PENALTY] Total frames              : {total_frames}")
            print(f"[PENALTY] Kick velocity threshold    : "
                  f"{self.kick_velocity_threshold} px/frame")
            print(f"[PENALTY] Pre-kick search            : "
                  f"{self.pre_kick_search} frames")
            print(f"[PENALTY] Max kicker distance        : "
                  f"{self.max_kicker_distance}px")
            print(f"[PENALTY] On-target check window     : "
                  f"{self.on_target_check_window} frames")
            print(f"[PENALTY] Gawang shrink ratio        : "
                  f"{self.gawang_shrink_ratio}")
            print(f"[PENALTY] Keeper save check window   : "
                  f"{self.keeper_save_check_window} frames")
            print(f"[PENALTY] Keeper ball distance thr   : "
                  f"{self.keeper_ball_distance_thr}px")
            print(f"[PENALTY] Keeper velocity drop thr   : "
                  f"{self.keeper_velocity_drop_thr} px/frame")
            print(f"[PENALTY] Cooldown                   : "
                  f"{self.cooldown_frames} frames")
            if self._manual_kick_mapping:
                print(f"[PENALTY] Manual kick mapping        : "
                      f"{self._manual_kick_mapping}")

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

            # Cari penendang
            kicker_id, kicker_jersey, kicker_pos = self.find_kicker(
                tracks, frames if frames else [], kick_frame, debug=debug
            )

            if kicker_id is None:
                if debug:
                    print(f"[PENALTY]   -> Skip: tidak ada player terdekat")
                continue

            # 1. Cek SHOOT ON TARGET
            is_on_target, hit_frame, on_target_reason = self.check_shoot_on_target(
                tracks, kick_frame, stable_gawang, debug=debug
            )

            # 2. Cek KEEPER SAVE (hanya jika on target)
            is_saved = False
            save_frame = -1
            save_reason = ""

            if is_on_target:
                is_saved, save_frame, save_reason = self.check_keeper_save(
                    tracks, velocities, kick_frame, debug=debug
                )

            # 3. Tentukan GOL
            # GOL = on target DAN TIDAK di-save keeper
            # TIDAK GOL = off target ATAU di-save keeper
            is_goal = is_on_target and not is_saved

            # Buat reason gabungan
            if is_goal:
                result_reason = f"GOL! {on_target_reason}"
            elif is_on_target and is_saved:
                result_reason = f"SAVED! {save_reason}"
            else:
                result_reason = f"MISS! {on_target_reason}"

            ball_pos_kick = None
            ball_data = tracks['ball'][kick_frame].get(1)
            if ball_data:
                ball_pos_kick = get_center_of_bbox(ball_data['bbox'])

            penalty_event = {
                'frame_kick'      : kick_frame,
                'frame_hit'       : hit_frame,
                'frame_save'      : save_frame,
                'frame_display'   : kick_frame + 5,
                'kicker_id'       : kicker_id,
                'kicker_jersey'   : kicker_jersey,
                'is_on_target'    : is_on_target,
                'is_saved'        : is_saved,
                'is_goal'         : is_goal,
                'on_target_reason': on_target_reason,
                'save_reason'     : save_reason,
                'result_reason'   : result_reason,
                'ball_velocity'   : velocities[kick_frame],
                'kicker_pos'      : kicker_pos,
                'ball_pos_kick'   : ball_pos_kick,
                'gawang_bbox'     : stable_gawang,
            }
            penalties.append(penalty_event)

            if debug:
                target_str = "ON TARGET" if is_on_target else "OFF TARGET"
                if is_goal:
                    result_str = "GOL!"
                elif is_saved:
                    result_str = "SAVED"
                else:
                    result_str = "MISS"
                print(f"[PENALTY]   -> {kicker_jersey}: {target_str} | "
                      f"{result_str} | {result_reason}")

        if debug:
            on   = sum(1 for p in penalties if p['is_on_target'])
            off  = sum(1 for p in penalties if not p['is_on_target'])
            gol  = sum(1 for p in penalties if p['is_goal'])
            save = sum(1 for p in penalties if p['is_saved'])
            print(f"\n[PENALTY] === HASIL AKHIR ===")
            print(f"[PENALTY] Total tendangan : {len(penalties)}")
            print(f"[PENALTY] ON TARGET       : {on}")
            print(f"[PENALTY] OFF TARGET      : {off}")
            print(f"[PENALTY] GOL             : {gol}")
            print(f"[PENALTY] SAVED           : {save}")
            print(f"[PENALTY] =========================\n")

        return penalties

    # ============================================================
    # STATISTIK — ON TARGET + GOL/SAVED
    # ============================================================

    def get_penalty_statistics(self, penalties: List[Dict]) -> Dict:
        total      = len(penalties)
        on_list    = [p for p in penalties if p['is_on_target']]
        off_list   = [p for p in penalties if not p['is_on_target']]
        goal_list  = [p for p in penalties if p['is_goal']]
        saved_list = [p for p in penalties if p['is_saved']]

        per_player: Dict[str, Dict] = {}
        for p in penalties:
            jersey = p['kicker_jersey']
            if jersey not in per_player:
                per_player[jersey] = {
                    'total': 0,
                    'on_target': 0,
                    'off_target': 0,
                    'goals': 0,
                    'saved': 0,
                    'on_target_pct': 0.0,
                    'goal_pct': 0.0,
                }
            per_player[jersey]['total'] += 1
            if p['is_on_target']:
                per_player[jersey]['on_target'] += 1
            else:
                per_player[jersey]['off_target'] += 1
            if p['is_goal']:
                per_player[jersey]['goals'] += 1
            if p['is_saved']:
                per_player[jersey]['saved'] += 1

        for jersey, stat in per_player.items():
            t = stat['total']
            stat['on_target_pct'] = round(
                stat['on_target'] / t * 100, 1
            ) if t > 0 else 0.0
            stat['goal_pct'] = round(
                stat['goals'] / t * 100, 1
            ) if t > 0 else 0.0

        return {
            'total_kicks'     : total,
            'total_on_target' : len(on_list),
            'total_off_target': len(off_list),
            'total_goals'     : len(goal_list),
            'total_saved'     : len(saved_list),
            'on_target_pct'   : round(
                len(on_list) / total * 100, 1
            ) if total > 0 else 0.0,
            'goal_pct'        : round(
                len(goal_list) / total * 100, 1
            ) if total > 0 else 0.0,
            'per_player'      : per_player
        }
