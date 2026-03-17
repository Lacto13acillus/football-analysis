# penalty_detector.py
# Mendeteksi event tendangan penalty:
#   1. SHOOT ON TARGET / OFF TARGET
#   2. GOL / SAVED BY KEEPER
#
# === v6: KEEPER-CENTRIC + REACTION WINDOW ===
#
# Prinsip:
#   - ON TARGET = default GOL, kecuali ada bukti kuat KETERLIBATAN KEEPER
#   - SEMUA sinyal save HARUS melibatkan keeper
#   - CRITICAL: Sinyal save hanya valid dalam REACTION WINDOW
#     (waktu singkat setelah bola masuk gawang, ~0.25 detik)
#   - Setelah reaction window, keeper proximity = pengambilan bola dari jaring
#   - Bounce-back tanpa keeper = bounce dari jaring = tetap GOL
#
# Perbaikan dari v5:
#   - Semua sinyal keeper dicek HANYA dalam reaction window
#   - FPS-adaptive reaction window (0.25 detik)
#   - Keeper proximity setelah reaction window = diabaikan (bukan save)

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

        # --- Parameter deteksi KEEPER SAVE (v6) ---
        self.save_check_window       = 60

        # REACTION WINDOW — KUNCI UTAMA v6
        # Save hanya bisa terjadi dalam waktu singkat setelah bola masuk gawang.
        # Setelah itu, keeper dekat bola = mengambil bola dari jaring.
        # Nilainya dalam DETIK, akan dikonversi ke frame berdasarkan FPS.
        self.save_reaction_time      = 0.25  # detik (0.25s = cukup untuk tangkap)

        # Penetration depth — sebagai MODIFIER
        self.penetration_goal_threshold = 0.35

        # Keeper proximity
        self.save_keeper_max_dist    = 120
        self.save_ball_max_vel       = 5.0

        # Overlap — bola di dalam bbox keeper
        self.overlap_bbox_expand     = 20
        self.overlap_min_frames      = 2

        # Bounce-back (HARUS dengan keeper proximity dalam reaction window)
        self.bounce_back_frames_thr  = 8
        self.bounce_back_margin      = 30
        self.bounce_keeper_max_dist  = 200

        # Scoring threshold untuk keputusan SAVED
        self.save_score_threshold    = 3

        # --- Display ---
        self.kick_display_duration   = 50

        # --- Manual mappings ---
        self._manual_kick_mapping: Dict[int, str] = {}
        self._manual_goal_mapping: Dict[int, bool] = {}

        self._player_identifier = None

    @property
    def save_reaction_frames(self) -> int:
        """Reaction window dalam jumlah frame, berdasarkan FPS."""
        return max(8, int(self.fps * self.save_reaction_time))

    def set_jersey_map(self, player_identifier) -> None:
        self._player_identifier = player_identifier

    def set_manual_kick_mapping(self, mapping: Dict[int, str]) -> None:
        self._manual_kick_mapping = mapping
        print(f"[PENALTY] Manual kick mapping diset: {self._manual_kick_mapping}")

    def set_manual_goal_mapping(self, mapping: Dict[int, bool]) -> None:
        self._manual_goal_mapping = mapping
        print(f"[PENALTY] Manual goal mapping diset: {self._manual_goal_mapping}")

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
    # HELPER: Ambil posisi bola dan keeper
    # ============================================================

    def _get_ball_pos(self, tracks: Dict, frame: int) -> Optional[Tuple[float, float]]:
        if frame < 0 or frame >= len(tracks['ball']):
            return None
        ball_data = tracks['ball'][frame].get(1)
        if ball_data and 'bbox' in ball_data:
            return get_center_of_bbox(ball_data['bbox'])
        return None

    def _get_ball_bbox(self, tracks: Dict, frame: int) -> Optional[List[float]]:
        if frame < 0 or frame >= len(tracks['ball']):
            return None
        ball_data = tracks['ball'][frame].get(1)
        if ball_data and 'bbox' in ball_data:
            return ball_data['bbox']
        return None

    def _get_keeper_bbox(self, tracks: Dict, frame: int) -> Optional[List[float]]:
        if frame < 0 or frame >= len(tracks['keeper']):
            return None
        keeper_data = tracks['keeper'][frame].get(1)
        if keeper_data and 'bbox' in keeper_data:
            return keeper_data['bbox']
        return None

    def _get_keeper_center(self, tracks: Dict, frame: int) -> Optional[Tuple[float, float]]:
        bbox = self._get_keeper_bbox(tracks, frame)
        if bbox:
            return get_center_of_bbox(bbox)
        return None

    def _ball_inside_keeper_bbox(
        self, tracks: Dict, frame: int, expand: int = 15
    ) -> bool:
        """Cek apakah bola berada di dalam bounding box keeper (dengan expand)."""
        ball_pos = self._get_ball_pos(tracks, frame)
        keeper_bbox = self._get_keeper_bbox(tracks, frame)
        if ball_pos is None or keeper_bbox is None:
            return False
        bx, by = ball_pos
        kx1, ky1, kx2, ky2 = keeper_bbox
        return (kx1 - expand <= bx <= kx2 + expand and
                ky1 - expand <= by <= ky2 + expand)

    def _keeper_near_ball(
        self, tracks: Dict, frame: int, max_dist: float = 120
    ) -> Tuple[bool, float]:
        """Cek apakah keeper dekat dengan bola. Return (is_near, distance)."""
        ball_pos = self._get_ball_pos(tracks, frame)
        keeper_center = self._get_keeper_center(tracks, frame)
        if ball_pos is None or keeper_center is None:
            return False, float('inf')
        dist = measure_distance(ball_pos, keeper_center)
        return dist < max_dist, dist

    # ============================================================
    # CEK KEEPER SAVE — v6: KEEPER-CENTRIC + REACTION WINDOW
    # ============================================================

    def check_keeper_save(
        self,
        tracks: Dict,
        velocities: List[float],
        kick_frame: int,
        stable_gawang_bbox: List[float],
        debug: bool = False
    ) -> Tuple[bool, int, str]:
        """
        Deteksi GOL vs SAVED — v6: KEEPER-CENTRIC + REACTION WINDOW

        Prinsip utama:
        1. SAVED memerlukan bukti KETERLIBATAN KEEPER
        2. Keterlibatan keeper hanya valid dalam REACTION WINDOW
           (waktu singkat setelah bola masuk area gawang)
        3. Setelah reaction window, keeper proximity = mengambil bola
           dari jaring, BUKAN save
        4. Penetration depth sebagai modifier, bukan sole discriminator
        5. Bounce-back tanpa keeper dalam reaction window = jaring bounce = GOL

        Reaction window: ~0.25 detik setelah bola masuk area gawang
        - Di 59fps = 15 frame
        - Di 30fps = 8 frame

        Sinyal dan score (hanya dalam reaction window):
        - KEEPER_CATCH: overlap + velocity drop → +4
        - KEEPER_BLOCK: proximity + velocity drop → +3
        - KEEPER_OVERLAP: bola di bbox keeper → +2
        - KEEPER_DEFLECT: direction reversal dekat keeper → +2
        - KEEPER_BOUNCE: bounce-back + keeper dekat → +2
        - NET_BOUNCE: bounce-back tanpa keeper → +0

        Depth modifier:
        - Dangkal: +1
        - Dalam: -1

        Keputusan:
        - total_score >= save_score_threshold → SAVED
        - total_score < save_score_threshold → GOL
        """
        gx1, gy1, gx2, gy2 = stable_gawang_bbox
        gawang_top = gy1
        gawang_bottom = gy2
        gawang_height = gy2 - gy1

        if gawang_height <= 0:
            return False, -1, "Gawang height invalid"

        total = len(tracks['ball'])
        check_end = min(kick_frame + self.save_check_window, total - 1)

        # Reaction window dalam frame
        reaction_frames = self.save_reaction_frames

        # --- Kumpulkan data bola setelah kick ---
        ball_positions = []
        for f in range(kick_frame, check_end + 1):
            ball_pos = self._get_ball_pos(tracks, f)
            if ball_pos:
                ball_positions.append((f, ball_pos[0], ball_pos[1]))

        if not ball_positions:
            return False, -1, "Tidak ada data bola"

        # --- Cari frame dimana bola MASUK area gawang ---
        entered_gawang = False
        enter_frame = -1
        enter_idx = -1

        for idx, (f, bx, by) in enumerate(ball_positions):
            if gx1 <= bx <= gx2 and gy1 <= by <= gy2:
                entered_gawang = True
                enter_frame = f
                enter_idx = idx
                break

        if not entered_gawang:
            if debug:
                print(f"[PENALTY]   Bola tidak masuk area gawang -> NOT SAVED")
            return False, -1, "Bola tidak masuk area gawang"

        # Batas akhir reaction window
        reaction_end_frame = enter_frame + reaction_frames

        if debug:
            print(f"[PENALTY]   [REACTION] Bola masuk gawang di frame {enter_frame}, "
                  f"reaction window = {reaction_frames} frame "
                  f"({self.save_reaction_time:.2f}s @ {self.fps}fps), "
                  f"batas = frame {reaction_end_frame}")

        # ====================================================
        # ANALISIS 1: PENETRATION DEPTH (sebagai MODIFIER)
        # ====================================================
        ball_y_in_gawang = []
        for idx in range(enter_idx, len(ball_positions)):
            f, bx, by = ball_positions[idx]
            if gx1 - 30 <= bx <= gx2 + 30:
                ball_y_in_gawang.append(by)

        if not ball_y_in_gawang:
            if debug:
                print(f"[PENALTY]   Tidak ada data Y bola di gawang -> GOL (default)")
            return False, -1, "Tidak ada data bola di gawang"

        min_ball_y = min(ball_y_in_gawang)
        avg_ball_y = np.mean(ball_y_in_gawang)
        penetration_ratio = (gawang_bottom - min_ball_y) / gawang_height
        avg_penetration = (gawang_bottom - avg_ball_y) / gawang_height

        if debug:
            print(f"[PENALTY]   [DEPTH] Gawang Y range: {gawang_top:.0f}(dalam) - "
                  f"{gawang_bottom:.0f}(depan), height={gawang_height:.0f}")
            print(f"[PENALTY]   [DEPTH] Ball min_Y={min_ball_y:.0f}, "
                  f"avg_Y={avg_ball_y:.0f}")
            print(f"[PENALTY]   [DEPTH] penetration_ratio={penetration_ratio:.2f} "
                  f"(max depth), avg_penetration={avg_penetration:.2f}")

        # ====================================================
        # ANALISIS 2: SINYAL KEEPER (DALAM REACTION WINDOW SAJA)
        # ====================================================
        save_score = 0
        save_signals = []
        save_frame = -1

        # --- Signal A: KEEPER_CATCH ---
        # Bola overlap dengan keeper bbox DAN velocity turun drastis
        # HANYA dalam reaction window
        catch_detected = False
        for idx in range(enter_idx, len(ball_positions)):
            f, bx, by = ball_positions[idx]

            # REACTION WINDOW CHECK
            if f > reaction_end_frame:
                break

            if f >= len(velocities):
                continue

            if self._ball_inside_keeper_bbox(tracks, f, expand=self.overlap_bbox_expand):
                vel = velocities[f]
                if vel < self.save_ball_max_vel:
                    sustained = 0
                    for ff in range(f, min(f + 10, len(velocities))):
                        if velocities[ff] < self.save_ball_max_vel:
                            sustained += 1
                        else:
                            break

                    if sustained >= 3:
                        save_score += 4
                        save_signals.append(
                            f"KEEPER_CATCH: Bola di dalam bbox keeper + velocity drop "
                            f"(frame={f}, vel={vel:.1f}, sustained={sustained}f) "
                            f"[{f - enter_frame}f setelah masuk]"
                        )
                        if save_frame == -1:
                            save_frame = f
                        catch_detected = True
                        if debug:
                            print(f"[PENALTY]   [SIGNAL-A] KEEPER_CATCH: frame={f} "
                                  f"({f - enter_frame}f setelah masuk), "
                                  f"vel={vel:.1f}, sustained={sustained} → +4")
                        break

        # --- Signal B: KEEPER_BLOCK ---
        # Bola dekat keeper DAN velocity turun tajam
        # HANYA dalam reaction window
        if not catch_detected:
            for idx in range(enter_idx, len(ball_positions)):
                f, bx, by = ball_positions[idx]

                # REACTION WINDOW CHECK
                if f > reaction_end_frame:
                    break

                if f >= len(velocities):
                    continue

                is_near, dist = self._keeper_near_ball(
                    tracks, f, max_dist=self.save_keeper_max_dist
                )
                if not is_near:
                    continue

                vel = velocities[f]
                if vel < self.save_ball_max_vel:
                    sustained = 0
                    for ff in range(f, min(f + 8, len(velocities))):
                        if velocities[ff] < self.save_ball_max_vel:
                            sustained += 1
                        else:
                            break

                    if sustained >= 3:
                        save_score += 3
                        save_signals.append(
                            f"KEEPER_BLOCK: Bola dekat keeper + velocity drop "
                            f"(frame={f}, dist={dist:.0f}px, vel={vel:.1f}, "
                            f"sustained={sustained}f) "
                            f"[{f - enter_frame}f setelah masuk]"
                        )
                        if save_frame == -1:
                            save_frame = f
                        if debug:
                            print(f"[PENALTY]   [SIGNAL-B] KEEPER_BLOCK: frame={f} "
                                  f"({f - enter_frame}f setelah masuk), "
                                  f"dist={dist:.0f}px, vel={vel:.1f}, "
                                  f"sustained={sustained} → +3")
                        break

        # --- Signal C: KEEPER_OVERLAP (tanpa velocity drop) ---
        # HANYA hitung overlap dalam reaction window
        overlap_count = 0
        first_overlap_frame = -1

        for idx in range(enter_idx, len(ball_positions)):
            f, bx, by = ball_positions[idx]

            # REACTION WINDOW CHECK
            if f > reaction_end_frame:
                break

            if self._ball_inside_keeper_bbox(tracks, f, expand=self.overlap_bbox_expand):
                overlap_count += 1
                if first_overlap_frame == -1:
                    first_overlap_frame = f

        if overlap_count >= self.overlap_min_frames and not catch_detected:
            save_score += 2
            save_signals.append(
                f"KEEPER_OVERLAP: Bola di dalam bbox keeper "
                f"selama {overlap_count} frame (dalam reaction window)"
            )
            if save_frame == -1:
                save_frame = first_overlap_frame
            if debug:
                print(f"[PENALTY]   [SIGNAL-C] KEEPER_OVERLAP: {overlap_count} frame "
                      f"dalam reaction window → +2")
        elif overlap_count > 0 and debug:
            print(f"[PENALTY]   [SIGNAL-C] KEEPER_OVERLAP: hanya {overlap_count} frame "
                  f"dalam reaction window (min={self.overlap_min_frames}) → +0")

        # --- Cek juga overlap SETELAH reaction window (untuk log) ---
        if debug:
            late_overlap = 0
            for idx in range(enter_idx, len(ball_positions)):
                f, bx, by = ball_positions[idx]
                if f <= reaction_end_frame:
                    continue
                if self._ball_inside_keeper_bbox(tracks, f, expand=self.overlap_bbox_expand):
                    late_overlap += 1
            if late_overlap > 0:
                print(f"[PENALTY]   [INFO] {late_overlap} frame overlap SETELAH "
                      f"reaction window (diabaikan — kemungkinan pengambilan bola)")

        # --- Signal D: KEEPER_DEFLECT ---
        # Direction reversal DAN keeper dekat saat reversal
        # Reversal biasanya terjadi di sekitar enter_frame
        if enter_idx >= 0 and enter_idx + 8 < len(ball_positions):
            y_before = []
            y_after = []

            for idx in range(max(0, enter_idx - 5), enter_idx):
                y_before.append(ball_positions[idx][2])

            for idx in range(enter_idx + 3, min(enter_idx + 8, len(ball_positions))):
                y_after.append(ball_positions[idx][2])

            if len(y_before) >= 2 and len(y_after) >= 2:
                dy_before = y_before[-1] - y_before[0]
                dy_after = y_after[-1] - y_after[0]

                if dy_before < -5 and dy_after > 10:
                    rev_frame = ball_positions[
                        min(enter_idx + 3, len(ball_positions) - 1)
                    ][0]

                    # REACTION WINDOW CHECK
                    if rev_frame <= reaction_end_frame:
                        rev_ball = self._get_ball_pos(tracks, rev_frame)
                        rev_keeper = self._get_keeper_center(tracks, rev_frame)

                        if rev_ball and rev_keeper:
                            rev_dist = measure_distance(rev_ball, rev_keeper)
                            if rev_dist < 150:
                                save_score += 2
                                save_signals.append(
                                    f"KEEPER_DEFLECT: Bola berubah arah dekat keeper "
                                    f"(frame={rev_frame}, dy_before={dy_before:.0f}, "
                                    f"dy_after={dy_after:.0f}, "
                                    f"dist={rev_dist:.0f}px) "
                                    f"[{rev_frame - enter_frame}f setelah masuk]"
                                )
                                if save_frame == -1:
                                    save_frame = rev_frame
                                if debug:
                                    print(f"[PENALTY]   [SIGNAL-D] KEEPER_DEFLECT: "
                                          f"frame={rev_frame} "
                                          f"({rev_frame - enter_frame}f setelah masuk), "
                                          f"dy_before={dy_before:.0f}, "
                                          f"dy_after={dy_after:.0f}, "
                                          f"dist={rev_dist:.0f}px → +2")
                    elif debug:
                        print(f"[PENALTY]   [SIGNAL-D] Direction reversal di frame "
                              f"{rev_frame} tapi DI LUAR reaction window → +0")

        # --- Signal E: BOUNCE-BACK (DENGAN validasi keeper) ---
        # Bola keluar gawang — tapi cek keeper proximity DALAM reaction window
        frames_outside = 0
        first_exit_frame = -1

        for idx in range(enter_idx + 1, len(ball_positions)):
            f, bx, by = ball_positions[idx]

            is_outside = False
            if by > gawang_bottom + self.bounce_back_margin:
                is_outside = True
            if (bx < gx1 - 50 or bx > gx2 + 50) and by > gy1:
                is_outside = True

            if is_outside:
                frames_outside += 1
                if first_exit_frame == -1:
                    first_exit_frame = f

        if frames_outside >= self.bounce_back_frames_thr:
            # Cek keeper proximity DALAM REACTION WINDOW
            keeper_near_in_reaction = False
            min_keeper_dist_reaction = float('inf')

            for f in range(enter_frame, min(reaction_end_frame + 1, total)):
                is_near, dist = self._keeper_near_ball(
                    tracks, f, max_dist=self.bounce_keeper_max_dist
                )
                if dist < min_keeper_dist_reaction:
                    min_keeper_dist_reaction = dist
                if is_near:
                    keeper_near_in_reaction = True

            if keeper_near_in_reaction:
                save_score += 2
                save_signals.append(
                    f"KEEPER_BOUNCE: Bola memantul keluar + keeper dekat dalam "
                    f"reaction window ({frames_outside}f di luar, "
                    f"min_keeper_dist={min_keeper_dist_reaction:.0f}px)"
                )
                if save_frame == -1:
                    save_frame = first_exit_frame
                if debug:
                    print(f"[PENALTY]   [SIGNAL-E] KEEPER_BOUNCE: "
                          f"{frames_outside}f di luar, "
                          f"keeper_dist_in_reaction={min_keeper_dist_reaction:.0f}px → +2")
            else:
                if debug:
                    print(f"[PENALTY]   [SIGNAL-E] NET_BOUNCE: "
                          f"{frames_outside}f di luar, "
                          f"keeper tidak dekat dalam reaction window "
                          f"(min_dist={min_keeper_dist_reaction:.0f}px "
                          f"> {self.bounce_keeper_max_dist}px) → +0 (jaring)")

        # ====================================================
        # DEPTH MODIFIER
        # ====================================================
        if penetration_ratio < self.penetration_goal_threshold:
            depth_modifier = 1
            if debug:
                print(f"[PENALTY]   [DEPTH-MOD] Penetrasi dangkal "
                      f"({penetration_ratio:.2f} < {self.penetration_goal_threshold}) → +1")
        else:
            depth_modifier = -1
            if debug:
                print(f"[PENALTY]   [DEPTH-MOD] Penetrasi dalam "
                      f"({penetration_ratio:.2f} >= {self.penetration_goal_threshold}) → -1")

        total_score = save_score + depth_modifier

        if debug:
            print(f"[PENALTY]   [SCORE] save_signals={save_score}, "
                  f"depth_mod={depth_modifier:+d}, "
                  f"total={total_score} "
                  f"(threshold={self.save_score_threshold})")

        # ====================================================
        # KEPUTUSAN AKHIR
        # ====================================================
        if total_score >= self.save_score_threshold:
            all_reasons = " | ".join(save_signals)
            reason = (f"SAVED (score={total_score}, depth={penetration_ratio:.2f}) "
                      f"- {all_reasons}")
            if debug:
                print(f"[PENALTY]   >>> SAVED (score={total_score} "
                      f">= {self.save_score_threshold})")
                for s in save_signals:
                    print(f"[PENALTY]       - {s}")
            return True, save_frame, reason
        else:
            if save_signals:
                all_reasons = " | ".join(save_signals)
                reason = (f"GOL - Sinyal save tidak cukup kuat "
                          f"(score={total_score} < {self.save_score_threshold}, "
                          f"depth={penetration_ratio:.2f}) - {all_reasons}")
            else:
                reason = (f"GOL - Tidak ada sinyal keeper involvement dalam "
                          f"reaction window "
                          f"(score={total_score}, depth={penetration_ratio:.2f})")
            if debug:
                print(f"[PENALTY]   >>> GOL (score={total_score} "
                      f"< {self.save_score_threshold})")
            return False, -1, reason

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
            print(f"\n[PENALTY] === PENALTY DETECTION v6 (KEEPER-CENTRIC + REACTION WINDOW) ===")
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
            print(f"[PENALTY] Save check window          : "
                  f"{self.save_check_window} frames")
            print(f"[PENALTY] Cooldown                   : "
                  f"{self.cooldown_frames} frames")
            print(f"[PENALTY] --- Save Detection (v6: Reaction Window) ---")
            print(f"[PENALTY]   Reaction time            : "
                  f"{self.save_reaction_time}s "
                  f"({self.save_reaction_frames} frames @ {self.fps}fps)")
            print(f"[PENALTY]   Depth threshold (modifier): "
                  f"{self.penetration_goal_threshold}")
            print(f"[PENALTY]   Save score threshold     : "
                  f"{self.save_score_threshold}")
            print(f"[PENALTY]   Keeper max dist (block)  : "
                  f"{self.save_keeper_max_dist}px")
            print(f"[PENALTY]   Ball max vel (stop)      : "
                  f"{self.save_ball_max_vel} px/f")
            print(f"[PENALTY]   Overlap min frames       : "
                  f"{self.overlap_min_frames}")
            print(f"[PENALTY]   Overlap bbox expand      : "
                  f"{self.overlap_bbox_expand}px")
            print(f"[PENALTY]   Bounce-back frames thr   : "
                  f"{self.bounce_back_frames_thr}")
            print(f"[PENALTY]   Bounce keeper max dist   : "
                  f"{self.bounce_keeper_max_dist}px")
            if self._manual_kick_mapping:
                print(f"[PENALTY] Manual kick mapping        : "
                      f"{self._manual_kick_mapping}")
            if self._manual_goal_mapping:
                print(f"[PENALTY] Manual goal mapping        : "
                      f"{self._manual_goal_mapping}")

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

            # 2. Tentukan GOL / SAVED
            is_saved = False
            is_goal = False
            save_frame = -1
            save_reason = ""

            # Cek manual goal mapping DULU (highest priority)
            if kick_frame in self._manual_goal_mapping:
                manual_goal = self._manual_goal_mapping[kick_frame]
                is_goal = manual_goal
                is_saved = is_on_target and not manual_goal
                save_reason = ("MANUAL: " +
                               ("GOL (override)" if manual_goal
                                else "SAVED (override)"))
                if debug:
                    print(f"[PENALTY]   Goal dari MANUAL GOAL MAPPING: "
                          f"frame {kick_frame} -> "
                          f"{'GOL' if manual_goal else 'SAVED'}")
            elif is_on_target:
                # Deteksi otomatis v6: keeper-centric + reaction window
                is_saved, save_frame, save_reason = self.check_keeper_save(
                    tracks, velocities, kick_frame, stable_gawang, debug=debug
                )
                is_goal = not is_saved
            else:
                is_goal = False
                is_saved = False

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
                manual_tag = " [MANUAL]" if kick_frame in self._manual_goal_mapping else ""
                print(f"[PENALTY]   -> {kicker_jersey}: {target_str} | "
                      f"{result_str}{manual_tag} | {result_reason}")

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
    # STATISTIK
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
