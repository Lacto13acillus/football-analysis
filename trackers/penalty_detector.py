# penalty_detector.py
# Mendeteksi event tendangan penalty:
#   1. SHOOT ON TARGET / OFF TARGET
#   2. GOL / SAVED BY KEEPER
#
# === PERUBAHAN UTAMA v3 ===
# - check_keeper_save() di-rewrite total dengan 5 metode deteksi:
#   1. Keeper-Ball OVERLAP (bola di dalam bbox keeper)
#   2. Velocity DROP + proximity keeper
#   3. Ball-Keeper CONVERGENCE (bola bergerak mendekati keeper)
#   4. Bounce-back (bola memantul keluar gawang)
#   5. Direction reversal + velocity analysis
# - Setiap metode independen, cukup 1 terpenuhi = SAVED
# - Parameter bisa di-tune via CONFIG
# - Tidak ada manual_goal_mapping — sepenuhnya otomatis

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

        # --- Parameter deteksi KEEPER SAVE (BARU — multi-metode) ---
        self.save_check_window       = 60    # frame setelah kick untuk analisis

        # Metode 1: Keeper-Ball Overlap
        self.overlap_enabled         = True
        self.overlap_min_frames      = 3     # minimal N frame bola di dalam bbox keeper
        self.overlap_bbox_expand     = 15    # expand keeper bbox (px) untuk toleransi

        # Metode 2: Velocity Drop + Keeper Proximity
        self.vdrop_enabled           = True
        self.vdrop_proximity         = 120   # max jarak bola-keeper (px)
        self.vdrop_ratio             = 0.25  # velocity harus turun ke <25% dari kick velocity
        self.vdrop_abs_threshold     = 5.0   # ATAU velocity absolut < 5.0 px/frame
        self.vdrop_sustained_frames  = 4     # velocity rendah harus bertahan N frame

        # Metode 3: Ball-Keeper Convergence
        self.converge_enabled        = True
        self.converge_min_dist       = 50    # jarak akhir bola-keeper harus < N px
        self.converge_dist_decrease  = 0.5   # rasio penurunan jarak (jarak akhir / jarak awal < 0.5)

        # Metode 4: Bounce-back (dari versi lama, tetap dipertahankan)
        self.bounce_enabled          = True
        self.bounce_back_frames_thr  = 5
        self.bounce_back_margin      = 30
        self.bounce_strong_multiplier = 2    # bounce kuat = thr * multiplier

        # Metode 5: Direction Reversal
        self.reversal_enabled        = True
        self.ball_direction_window   = 5
        self.reversal_dy_before      = -5    # bola bergerak ke atas (Y menurun)
        self.reversal_dy_after       = 10    # bola berubah arah (Y meningkat)

        # --- Display ---
        self.kick_display_duration   = 50

        # --- Manual mappings ---
        self._manual_kick_mapping: Dict[int, str] = {}
        self._manual_goal_mapping: Dict[int, bool] = {}

        self._player_identifier = None

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
    # HELPER: Ambil posisi bola dan keeper per frame
    # ============================================================

    def _get_ball_pos(self, tracks: Dict, frame: int) -> Optional[Tuple[float, float]]:
        """Ambil posisi center bola di frame tertentu."""
        if frame < 0 or frame >= len(tracks['ball']):
            return None
        ball_data = tracks['ball'][frame].get(1)
        if ball_data and 'bbox' in ball_data:
            return get_center_of_bbox(ball_data['bbox'])
        return None

    def _get_ball_bbox(self, tracks: Dict, frame: int) -> Optional[List[float]]:
        """Ambil bbox bola di frame tertentu."""
        if frame < 0 or frame >= len(tracks['ball']):
            return None
        ball_data = tracks['ball'][frame].get(1)
        if ball_data and 'bbox' in ball_data:
            return ball_data['bbox']
        return None

    def _get_keeper_bbox(self, tracks: Dict, frame: int) -> Optional[List[float]]:
        """Ambil bbox keeper di frame tertentu."""
        if frame < 0 or frame >= len(tracks['keeper']):
            return None
        keeper_data = tracks['keeper'][frame].get(1)
        if keeper_data and 'bbox' in keeper_data:
            return keeper_data['bbox']
        return None

    def _get_keeper_center(self, tracks: Dict, frame: int) -> Optional[Tuple[float, float]]:
        """Ambil posisi center keeper di frame tertentu."""
        bbox = self._get_keeper_bbox(tracks, frame)
        if bbox:
            return get_center_of_bbox(bbox)
        return None

    def _is_ball_inside_keeper_bbox(
        self, tracks: Dict, frame: int, expand: int = 0
    ) -> bool:
        """Cek apakah center bola berada di dalam bbox keeper (dengan expand)."""
        ball_pos = self._get_ball_pos(tracks, frame)
        keeper_bbox = self._get_keeper_bbox(tracks, frame)
        if ball_pos is None or keeper_bbox is None:
            return False

        kx1, ky1, kx2, ky2 = keeper_bbox
        bx, by = ball_pos

        return (kx1 - expand <= bx <= kx2 + expand and
                ky1 - expand <= by <= ky2 + expand)

    # ============================================================
    # CEK KEEPER SAVE — MULTI-METHOD DETECTION (v3)
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
        Deteksi apakah bola di-save keeper menggunakan MULTI-METHOD detection.

        5 Metode independen — cukup 1 terpenuhi = SAVED:

        1. OVERLAP: Bola berada di dalam bbox keeper selama beberapa frame
        2. VELOCITY DROP: Velocity bola turun drastis saat dekat keeper
        3. CONVERGENCE: Bola bergerak mendekati keeper dan berhenti
        4. BOUNCE-BACK: Bola memantul keluar area gawang
        5. DIRECTION REVERSAL: Arah Y bola berubah setelah masuk area gawang

        Returns:
            (is_saved, save_frame, reason)
        """
        gx1, gy1, gx2, gy2 = stable_gawang_bbox
        gawang_bottom = gy2

        total = len(tracks['ball'])
        check_start = kick_frame
        check_end = min(kick_frame + self.save_check_window, total - 1)

        kick_velocity = velocities[kick_frame] if kick_frame < len(velocities) else 0.0

        # Kumpulkan data bola setelah kick
        ball_positions = []  # list of (frame, x, y) atau None
        for f in range(check_start, check_end + 1):
            ball_pos = self._get_ball_pos(tracks, f)
            if ball_pos:
                ball_positions.append((f, ball_pos[0], ball_pos[1]))
            else:
                ball_positions.append(None)

        if not ball_positions:
            return False, -1, "Tidak ada data bola"

        # Cari frame dimana bola MASUK area gawang
        entered_gawang = False
        enter_frame = -1
        enter_idx = -1

        for idx, pos in enumerate(ball_positions):
            if pos is None:
                continue
            f, bx, by = pos
            if gx1 <= bx <= gx2 and gy1 <= by <= gy2:
                entered_gawang = True
                enter_frame = f
                enter_idx = idx
                break

        if not entered_gawang:
            if debug:
                print(f"[PENALTY]   Keeper: Bola tidak masuk area gawang")
            return False, -1, "Bola tidak masuk area gawang"

        # Hasil dari setiap metode
        methods_triggered = []

        # ====================================================
        # METODE 1: KEEPER-BALL OVERLAP
        # ====================================================
        if self.overlap_enabled:
            overlap_count = 0
            first_overlap_frame = -1

            for idx in range(enter_idx, len(ball_positions)):
                pos = ball_positions[idx]
                if pos is None:
                    continue
                f = pos[0]

                if self._is_ball_inside_keeper_bbox(
                    tracks, f, expand=self.overlap_bbox_expand
                ):
                    overlap_count += 1
                    if first_overlap_frame == -1:
                        first_overlap_frame = f

            if overlap_count >= self.overlap_min_frames:
                methods_triggered.append(
                    (first_overlap_frame,
                     f"OVERLAP: Bola di dalam bbox keeper selama "
                     f"{overlap_count} frame (threshold={self.overlap_min_frames})")
                )

            if debug:
                print(f"[PENALTY]   [M1-OVERLAP] overlap_frames={overlap_count}, "
                      f"threshold={self.overlap_min_frames} "
                      f"-> {'TRIGGERED' if overlap_count >= self.overlap_min_frames else 'no'}")

        # ====================================================
        # METODE 2: VELOCITY DROP + KEEPER PROXIMITY
        # ====================================================
        if self.vdrop_enabled:
            vdrop_detected = False
            vdrop_frame = -1

            for idx in range(enter_idx, len(ball_positions)):
                pos = ball_positions[idx]
                if pos is None:
                    continue
                f, bx, by = pos

                # Cek proximity ke keeper
                keeper_center = self._get_keeper_center(tracks, f)
                if keeper_center is None:
                    continue

                dist = measure_distance((bx, by), keeper_center)
                if dist > self.vdrop_proximity:
                    continue

                # Cek velocity drop
                if f >= len(velocities):
                    continue

                # Cek beberapa frame ke depan untuk velocity sustained rendah
                future_vels = []
                for ff in range(f, min(f + self.vdrop_sustained_frames + 3, len(velocities))):
                    future_vels.append(velocities[ff])

                if not future_vels:
                    continue

                avg_vel = np.mean(future_vels)
                vel_ratio = avg_vel / max(kick_velocity, 0.1)

                # Velocity turun ke bawah threshold
                if (vel_ratio < self.vdrop_ratio or
                    avg_vel < self.vdrop_abs_threshold):

                    # Cek sustained: minimal N frame berturut-turut velocity rendah
                    sustained = 0
                    for fv in future_vels:
                        if fv < self.vdrop_abs_threshold or fv / max(kick_velocity, 0.1) < self.vdrop_ratio:
                            sustained += 1
                        else:
                            break

                    if sustained >= self.vdrop_sustained_frames:
                        vdrop_detected = True
                        vdrop_frame = f
                        methods_triggered.append(
                            (f, f"VDROP: Velocity turun ke {avg_vel:.1f} px/f "
                                f"(ratio={vel_ratio:.2f}) dekat keeper "
                                f"(dist={dist:.0f}px) selama {sustained} frame")
                        )
                        break

            if debug:
                print(f"[PENALTY]   [M2-VDROP] detected={vdrop_detected}"
                      + (f", frame={vdrop_frame}" if vdrop_detected else ""))

        # ====================================================
        # METODE 3: BALL-KEEPER CONVERGENCE
        # ====================================================
        if self.converge_enabled:
            converge_detected = False

            # Ambil jarak bola-keeper saat bola baru masuk gawang
            initial_dist = None
            if enter_idx < len(ball_positions) and ball_positions[enter_idx]:
                f0 = ball_positions[enter_idx][0]
                ball_enter_pos = (ball_positions[enter_idx][1], ball_positions[enter_idx][2])
                keeper_enter = self._get_keeper_center(tracks, f0)
                if keeper_enter:
                    initial_dist = measure_distance(ball_enter_pos, keeper_enter)

            # Ambil jarak bola-keeper di beberapa frame terakhir window
            final_dists = []
            last_frames = ball_positions[-10:]  # 10 frame terakhir
            for pos in last_frames:
                if pos is None:
                    continue
                f, bx, by = pos
                keeper_c = self._get_keeper_center(tracks, f)
                if keeper_c:
                    final_dists.append(measure_distance((bx, by), keeper_c))

            if final_dists and initial_dist and initial_dist > 0:
                avg_final_dist = np.mean(final_dists)
                min_final_dist = min(final_dists)
                dist_ratio = min_final_dist / initial_dist

                if (min_final_dist < self.converge_min_dist and
                    dist_ratio < self.converge_dist_decrease):
                    converge_detected = True
                    converge_frame = enter_frame + len(ball_positions) // 2
                    methods_triggered.append(
                        (converge_frame,
                         f"CONVERGE: Bola mendekati keeper "
                         f"(dist: {initial_dist:.0f}px -> {min_final_dist:.0f}px, "
                         f"ratio={dist_ratio:.2f})")
                    )

            if debug:
                init_str = f"{initial_dist:.0f}" if initial_dist else "N/A"
                final_str = f"{min(final_dists):.0f}" if final_dists else "N/A"
                print(f"[PENALTY]   [M3-CONVERGE] initial_dist={init_str}, "
                      f"min_final_dist={final_str} "
                      f"-> {'TRIGGERED' if converge_detected else 'no'}")

        # ====================================================
        # METODE 4: BOUNCE-BACK (bola keluar gawang setelah masuk)
        # ====================================================
        if self.bounce_enabled:
            frames_outside_after_enter = 0
            first_exit_frame = -1

            for idx in range(enter_idx + 1, len(ball_positions)):
                pos = ball_positions[idx]
                if pos is None:
                    continue
                f, bx, by = pos

                # Bola keluar area gawang ke bawah
                if by > gawang_bottom + self.bounce_back_margin:
                    frames_outside_after_enter += 1
                    if first_exit_frame == -1:
                        first_exit_frame = f

                # Bola keluar area gawang ke samping
                if bx < gx1 - 50 or bx > gx2 + 50:
                    if by > gy1:
                        frames_outside_after_enter += 1
                        if first_exit_frame == -1:
                            first_exit_frame = f

            # Bounce kuat
            if frames_outside_after_enter >= self.bounce_back_frames_thr * self.bounce_strong_multiplier:
                methods_triggered.append(
                    (first_exit_frame,
                     f"BOUNCE-STRONG: Bola memantul kuat keluar gawang "
                     f"({frames_outside_after_enter} frame di luar)")
                )
            # Bounce normal + dikombinasi metode lain (cek di bawah)
            elif frames_outside_after_enter >= self.bounce_back_frames_thr:
                # Bounce biasa saja belum cukup — perlu dikombinasi
                # Tapi simpan info untuk combined check
                pass

            if debug:
                print(f"[PENALTY]   [M4-BOUNCE] frames_outside={frames_outside_after_enter}, "
                      f"threshold={self.bounce_back_frames_thr} "
                      f"-> {'TRIGGERED' if frames_outside_after_enter >= self.bounce_back_frames_thr * self.bounce_strong_multiplier else 'no (strong)'}")

        # ====================================================
        # METODE 5: DIRECTION REVERSAL (arah Y bola berubah)
        # ====================================================
        if self.reversal_enabled:
            direction_reversed = False

            if enter_idx >= 0:
                y_before = []
                y_after = []

                for idx in range(max(0, enter_idx - self.ball_direction_window), enter_idx):
                    pos = ball_positions[idx]
                    if pos:
                        y_before.append(pos[2])

                for idx in range(enter_idx + 3,
                                 min(enter_idx + 3 + self.ball_direction_window,
                                     len(ball_positions))):
                    pos = ball_positions[idx]
                    if pos:
                        y_after.append(pos[2])

                if len(y_before) >= 2 and len(y_after) >= 2:
                    dy_before = y_before[-1] - y_before[0]
                    dy_after = y_after[-1] - y_after[0]

                    if dy_before < self.reversal_dy_before and dy_after > self.reversal_dy_after:
                        direction_reversed = True

                        # Direction reversal saja = indikasi kuat, tapi
                        # kombinasi dengan bounce/proximity lebih yakin
                        # Cek proximity keeper saat reversal
                        reversal_frame = ball_positions[enter_idx + 3][0] if enter_idx + 3 < len(ball_positions) and ball_positions[enter_idx + 3] else enter_frame
                        keeper_c = self._get_keeper_center(tracks, reversal_frame)
                        ball_c = self._get_ball_pos(tracks, reversal_frame)

                        if keeper_c and ball_c:
                            rev_dist = measure_distance(ball_c, keeper_c)
                            if rev_dist < 150:  # dekat keeper saat reversal
                                methods_triggered.append(
                                    (reversal_frame,
                                     f"REVERSAL: Arah bola berubah dekat keeper "
                                     f"(dy_before={dy_before:.0f}, dy_after={dy_after:.0f}, "
                                     f"dist_keeper={rev_dist:.0f}px)")
                                )

            if debug:
                print(f"[PENALTY]   [M5-REVERSAL] reversed={direction_reversed} "
                      f"-> {'TRIGGERED' if direction_reversed and any('REVERSAL' in m[1] for m in methods_triggered) else 'no'}")

        # ====================================================
        # KEPUTUSAN AKHIR
        # ====================================================
        if methods_triggered:
            # Ambil metode pertama yang trigger sebagai save_frame dan reason
            save_frame = methods_triggered[0][0]
            all_reasons = " | ".join([m[1] for m in methods_triggered])
            reason = f"SAVED ({len(methods_triggered)} metode) - {all_reasons}"

            if debug:
                print(f"[PENALTY]   >>> SAVED! {len(methods_triggered)} metode triggered:")
                for sf, sr in methods_triggered:
                    print(f"[PENALTY]       - {sr}")

            return True, save_frame, reason
        else:
            reason = "Tidak di-save (semua metode negatif)"

            if debug:
                print(f"[PENALTY]   >>> NOT SAVED (0 metode triggered)")

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
            print(f"[PENALTY] Save check window          : "
                  f"{self.save_check_window} frames")
            print(f"[PENALTY] Cooldown                   : "
                  f"{self.cooldown_frames} frames")
            print(f"[PENALTY] --- Save Detection Methods ---")
            print(f"[PENALTY]   M1-OVERLAP : enabled={self.overlap_enabled}, "
                  f"min_frames={self.overlap_min_frames}, expand={self.overlap_bbox_expand}px")
            print(f"[PENALTY]   M2-VDROP   : enabled={self.vdrop_enabled}, "
                  f"proximity={self.vdrop_proximity}px, "
                  f"ratio={self.vdrop_ratio}, abs={self.vdrop_abs_threshold}")
            print(f"[PENALTY]   M3-CONVERGE: enabled={self.converge_enabled}, "
                  f"min_dist={self.converge_min_dist}px, "
                  f"decrease={self.converge_dist_decrease}")
            print(f"[PENALTY]   M4-BOUNCE  : enabled={self.bounce_enabled}, "
                  f"frames_thr={self.bounce_back_frames_thr}, "
                  f"margin={self.bounce_back_margin}px")
            print(f"[PENALTY]   M5-REVERSAL: enabled={self.reversal_enabled}")
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
                # Deteksi otomatis: multi-method
                is_saved, save_frame, save_reason = self.check_keeper_save(
                    tracks, velocities, kick_frame, stable_gawang, debug=debug
                )
                is_goal = not is_saved
            else:
                # Off target = tidak gol
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
