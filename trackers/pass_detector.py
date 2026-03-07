# pass_detector.py
# Mendeteksi event passing dan mengevaluasi akurasi ke cone target.
#
# PERUBAHAN v2.6:
#   - Unknown BOLEH menjadi PENGIRIM (sebelumnya hanya penerima)
#   - Evaluasi akurasi Unknown: bola harus mendekati SALAH SATU dari
#     3 cone depan (cone 0, 1, 2) dengan radius lebih kecil
#   - Evaluasi akurasi #3/#19: tetap ke cone target (cone 3) radius 100px
#   - Unknown -> Unknown (track berbeda) = SKIP (bukan pass antar pemain)

import sys
sys.path.append('../')

from utils.bbox_utils import (
    measure_distance,
    get_center_of_bbox_bottom,
    get_center_of_bbox,
    extract_ball_trajectory,
    stabilize_cone_positions,
    identify_target_cone,
    check_ball_reached_target_cone
)
import numpy as np
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple


class PassDetector:
    def __init__(self, fps: int = 24):
        self.fps = fps

        # --- Parameter smoothing possession ---
        self.smoothing_window        = 5
        self.min_stable_frames       = 1

        # --- Parameter validasi pass ---
        self.min_pass_distance       = 30
        self.max_pass_distance       = 1500
        self.cooldown_frames         = 3
        self.min_possession_duration = 1

        # --- Parameter validasi gerakan bola ---
        self.ball_movement_check_radius = 25
        self.ball_movement_threshold    = 4

        # --- Parameter display ---
        self.player_search_radius  = 20
        self.pass_display_delay    = 3
        self.min_display_gap       = 3

        # --- Filter pemain ---
        # v2.6: Tidak lagi memfilter pengirim. Semua pemain bisa jadi pengirim.
        # Unknown -> Unknown tetap di-skip.
        self.known_jerseys = {"#3", "#19"}

        # --- Parameter buffer trajectory ---
        self.eval_buffer_before = 0
        self.eval_buffer_after  = 15

        # ============================================================
        # KONFIGURASI TARGET CONE (untuk #3 dan #19)
        # ============================================================
        self.manual_target_cone_id   : Optional[int] = None
        self.target_selection_mode   : str = "highest"
        self.target_proximity_radius : float = 100.0

        # ============================================================
        # KONFIGURASI FRONT CONES (untuk Unknown)
        # 3 cone di depan Unknown player, radius lebih kecil
        # ============================================================
        self.front_cone_ids          : List[int] = [0, 1, 2]
        self.front_cone_radius       : float = 65.0

        # Cache
        self._target_cone_id   : Optional[int] = None
        self._target_cone_pos  : Optional[Tuple[float, float]] = None
        self._stabilized_cones : Optional[Dict] = None
        self._front_cone_positions : Dict[int, Tuple[float, float]] = {}
        self._player_identifier = None

    # ============================================================
    # HELPER
    # ============================================================

    def set_jersey_map(self, player_identifier) -> None:
        self._player_identifier = player_identifier

    def _get_jersey(self, player_id: int) -> str:
        if self._player_identifier:
            return self._player_identifier.get_jersey_number_for_player(player_id)
        return str(player_id)

    def get_target_cone(self) -> Optional[Tuple[int, Tuple[float, float]]]:
        if self._target_cone_id is None or self._target_cone_pos is None:
            return None
        return self._target_cone_id, self._target_cone_pos

    def get_all_cones(self) -> Optional[Dict[int, Tuple[float, float]]]:
        return self._stabilized_cones

    def get_front_cones(self) -> Dict[int, Tuple[float, float]]:
        """Kembalikan posisi 3 cone depan untuk visualisasi."""
        return dict(self._front_cone_positions)

    def get_front_cone_radius(self) -> float:
        """Kembalikan radius front cone untuk visualisasi."""
        return self.front_cone_radius

    # ============================================================
    # INISIALISASI TARGET CONE + FRONT CONES
    # ============================================================

    def initialize_target_cone(
        self,
        tracks       : Dict,
        cone_key     : str = 'cones',
        sample_frames: int = 30,
        debug        : bool = True
    ) -> bool:
        if cone_key not in tracks or not tracks[cone_key]:
            print(f"[TARGET] WARNING: Key '{cone_key}' tidak ada di tracks!")
            return False

        if debug:
            print(f"\n[TARGET] === INISIALISASI TARGET CONE ===")
            print(f"[TARGET] Stabilisasi posisi cone dari {sample_frames} frame...")

        self._stabilized_cones = stabilize_cone_positions(
            tracks, cone_key=cone_key, sample_frames=sample_frames
        )

        if debug:
            print(f"[TARGET] Total cone terdeteksi: {len(self._stabilized_cones)}")
            for cid, pos in sorted(self._stabilized_cones.items()):
                marker = " <-- TARGET?" if pos[1] == min(
                    p[1] for p in self._stabilized_cones.values()
                ) else ""
                print(f"[TARGET]   Cone ID {cid:3d} -> "
                      f"({pos[0]:7.1f}, {pos[1]:7.1f}){marker}")

        if len(self._stabilized_cones) == 0:
            print("[TARGET] GAGAL: Tidak ada cone terdeteksi!")
            return False

        # --- Target cone (untuk #3/#19) ---
        result = identify_target_cone(
            stabilized_cones      = self._stabilized_cones,
            manual_target_cone_id = self.manual_target_cone_id,
            selection_mode        = self.target_selection_mode
        )

        if result is None:
            print("[TARGET] GAGAL: Tidak bisa mengidentifikasi target cone!")
            return False

        self._target_cone_id, self._target_cone_pos = result

        if debug:
            print(f"\n[TARGET] Target cone berhasil diidentifikasi!")
            print(f"[TARGET]   Cone ID      : {self._target_cone_id}")
            print(f"[TARGET]   Posisi       : ({self._target_cone_pos[0]:.1f}, "
                  f"{self._target_cone_pos[1]:.1f})")
            print(f"[TARGET]   Radius sukses: {self.target_proximity_radius:.0f} px")

        # --- Front cones (untuk Unknown) ---
        self._front_cone_positions = {}
        for cid in self.front_cone_ids:
            if cid in self._stabilized_cones:
                self._front_cone_positions[cid] = self._stabilized_cones[cid]

        if debug:
            print(f"\n[TARGET] Front cones (untuk Unknown):")
            print(f"[TARGET]   Cone IDs     : {self.front_cone_ids}")
            print(f"[TARGET]   Radius sukses: {self.front_cone_radius:.0f} px")
            for cid, pos in sorted(self._front_cone_positions.items()):
                print(f"[TARGET]   Cone {cid} -> ({pos[0]:.1f}, {pos[1]:.1f})")
            if not self._front_cone_positions:
                print(f"[TARGET]   WARNING: Tidak ada front cone ditemukan!")
            print(f"[TARGET] ========================================\n")

        return True

    # ============================================================
    # EVALUASI PASS KE TARGET CONE (#3/#19)
    # ============================================================

    def evaluate_pass_to_target(
        self,
        tracks    : Dict,
        pass_event: Dict,
        debug     : bool = False
    ) -> Tuple[bool, str]:
        if self._target_cone_pos is None:
            return True, "Target cone tidak diinisialisasi - semua pass = SUKSES"

        frame_start = pass_event['frame_start']
        frame_end   = pass_event['frame_end']

        trajectory = extract_ball_trajectory(
            tracks, frame_start, frame_end,
            buffer_before=self.eval_buffer_before,
            buffer_after=self.eval_buffer_after
        )

        if debug:
            print(f"[TARGET] Evaluasi pass frame {frame_start}-{frame_end}: "
                  f"{len(trajectory)} titik trajectory "
                  f"(buffer: -{self.eval_buffer_before}/+{self.eval_buffer_after})")

        reached, reason = check_ball_reached_target_cone(
            ball_trajectory  = trajectory,
            target_cone_pos  = self._target_cone_pos,
            proximity_radius = self.target_proximity_radius
        )

        if debug:
            status = "SUKSES" if reached else "GAGAL"
            print(f"[TARGET]   Hasil: {status} | {reason}")

        return reached, reason

    # ============================================================
    # EVALUASI PASS KE FRONT CONES (Unknown)
    # ============================================================

    def evaluate_pass_to_front_cones(
        self,
        tracks    : Dict,
        pass_event: Dict,
        debug     : bool = False
    ) -> Tuple[bool, str, Optional[int], float]:
        """
        Evaluasi apakah bola mendekati SALAH SATU dari 3 cone depan.

        Returns:
            (reached, reason, closest_cone_id, closest_dist)
            - reached       : True jika bola mendekati minimal 1 cone
            - reason        : penjelasan teks
            - closest_cone_id: ID cone terdekat (untuk display)
            - closest_dist  : jarak minimum ke cone terdekat
        """
        if not self._front_cone_positions:
            return True, "Front cones tidak diinisialisasi", None, 0.0

        frame_start = pass_event['frame_start']
        frame_end   = pass_event['frame_end']

        trajectory = extract_ball_trajectory(
            tracks, frame_start, frame_end,
            buffer_before=self.eval_buffer_before,
            buffer_after=self.eval_buffer_after
        )

        if debug:
            print(f"[FRONT] Evaluasi pass frame {frame_start}-{frame_end}: "
                  f"{len(trajectory)} titik trajectory "
                  f"(buffer: -{self.eval_buffer_before}/+{self.eval_buffer_after})")

        # Cek setiap front cone — SUKSES jika bola mendekati SALAH SATU
        overall_closest_dist = float('inf')
        overall_closest_cone = None
        best_reached = False
        best_reason  = ""

        for cid, cone_pos in self._front_cone_positions.items():
            reached, reason = check_ball_reached_target_cone(
                ball_trajectory  = trajectory,
                target_cone_pos  = cone_pos,
                proximity_radius = self.front_cone_radius
            )

            # Hitung closest distance ke cone ini
            if trajectory:
                min_dist = min(measure_distance(pt, cone_pos) for pt in trajectory)
            else:
                min_dist = float('inf')

            if min_dist < overall_closest_dist:
                overall_closest_dist = min_dist
                overall_closest_cone = cid

            if reached and not best_reached:
                best_reached = True
                best_reason  = f"Cone {cid}: {reason}"

            if debug:
                status = "SUKSES" if reached else "GAGAL"
                print(f"[FRONT]   Cone {cid} ({cone_pos[0]:.0f},{cone_pos[1]:.0f}): "
                      f"{status} | closest={min_dist:.1f}px | {reason}")

        if not best_reached:
            best_reason = (f"Bola tidak mencapai cone depan manapun: "
                          f"closest={overall_closest_dist:.1f}px ke cone {overall_closest_cone} "
                          f"(radius={self.front_cone_radius:.0f}px)")

        if debug:
            status = "SUKSES" if best_reached else "GAGAL"
            print(f"[FRONT]   Hasil akhir: {status}")

        return best_reached, best_reason, overall_closest_cone, overall_closest_dist

    # ============================================================
    # SMOOTHING & PREPROCESSING POSSESSION
    # ============================================================

    def smooth_possessions(self, raw_possessions: List[int]) -> List[int]:
        smoothed    = list(raw_possessions)
        half_window = self.smoothing_window // 2

        for i in range(half_window, len(raw_possessions) - half_window):
            window = raw_possessions[i - half_window: i + half_window + 1]
            valid  = [p for p in window if p != -1]
            if valid:
                smoothed[i] = Counter(valid).most_common(1)[0][0]
            else:
                smoothed[i] = -1

        return smoothed

    def fill_short_gaps(
        self,
        possessions: List[int],
        max_gap    : int = 20
    ) -> List[int]:
        filled     = list(possessions)
        last_valid = -1
        gap_start  = -1

        for i in range(len(filled)):
            if filled[i] != -1:
                if last_valid != -1 and gap_start != -1:
                    gap_length = i - gap_start
                    if gap_length <= max_gap:
                        for g in range(gap_start, i):
                            filled[g] = last_valid
                last_valid = filled[i]
                gap_start  = -1
            else:
                if gap_start == -1:
                    gap_start = i

        return filled

    def _normalize_possessions_by_jersey(
        self,
        possessions: List[int]
    ) -> List[int]:
        if not self._player_identifier:
            return possessions

        jersey_canonical: Dict[str, int] = {}
        normalized = list(possessions)

        for i, pid in enumerate(possessions):
            if pid == -1:
                continue
            jersey = self._get_jersey(pid)

            if jersey == "Unknown":
                continue

            if jersey not in jersey_canonical:
                jersey_canonical[jersey] = pid
            normalized[i] = jersey_canonical[jersey]

        return normalized

    def get_stable_segments(
        self,
        smoothed_possessions: List[int]
    ) -> List[Dict]:
        segments       = []
        current_player = -1
        segment_start  = 0

        for frame_num, player_id in enumerate(smoothed_possessions):
            if player_id != current_player:
                if current_player != -1:
                    duration = frame_num - segment_start
                    if duration >= self.min_stable_frames:
                        segments.append({
                            'player_id'  : current_player,
                            'frame_start': segment_start,
                            'frame_end'  : frame_num - 1
                        })
                current_player = player_id
                segment_start  = frame_num

        if current_player != -1:
            duration = len(smoothed_possessions) - segment_start
            if duration >= self.min_stable_frames:
                segments.append({
                    'player_id'  : current_player,
                    'frame_start': segment_start,
                    'frame_end'  : len(smoothed_possessions) - 1
                })

        return segments

    # ============================================================
    # VALIDASI PASS
    # ============================================================

    def validate_ball_movement(
        self,
        tracks     : Dict,
        frame_start: int,
        frame_end  : int
    ) -> float:
        check_start = max(0, frame_start - self.ball_movement_check_radius)
        check_end   = min(len(tracks['ball']),
                          frame_end + self.ball_movement_check_radius)

        ball_positions = []
        for f in range(check_start, check_end):
            ball_data = tracks['ball'][f].get(1)
            if ball_data and 'bbox' in ball_data:
                pos = get_center_of_bbox(ball_data['bbox'])
                ball_positions.append(pos)

        if len(ball_positions) < 2:
            return 0.0

        direct_distance  = measure_distance(ball_positions[0], ball_positions[-1])
        max_displacement = max(
            measure_distance(ball_positions[0], p) for p in ball_positions
        )
        return max(direct_distance, max_displacement)

    def find_player_nearby(
        self,
        tracks       : Dict,
        player_id    : int,
        target_frame : int,
        search_radius: Optional[int] = None
    ) -> Tuple[Optional[Dict], int]:
        if search_radius is None:
            search_radius = self.player_search_radius

        total_frames = len(tracks['players'])

        player_data = tracks['players'][target_frame].get(player_id)
        if player_data:
            return player_data, target_frame

        for offset in range(1, search_radius + 1):
            for delta in [-offset, offset]:
                check_frame = target_frame + delta
                if 0 <= check_frame < total_frames:
                    player_data = tracks['players'][check_frame].get(player_id)
                    if player_data:
                        return player_data, check_frame

        return None, -1

    # ============================================================
    # DETEKSI PASS UTAMA
    # ============================================================

    def _get_player_position_for_pass(
        self,
        tracks      : Dict,
        player_id   : int,
        frame_target: int,
        segment     : Dict
    ) -> Optional[Tuple[int, int]]:
        # Fallback 1: Track ID asli di frame target ± radius
        player_data, found_frame = self.find_player_nearby(
            tracks, player_id, frame_target
        )
        if player_data:
            return get_center_of_bbox_bottom(player_data['bbox'])

        # Fallback 2: Track ID asli di seluruh segmen
        seg_start = segment['frame_start']
        seg_end   = segment['frame_end']
        for f in range(seg_start, min(seg_end + 1, len(tracks['players']))):
            player_data = tracks['players'][f].get(player_id)
            if player_data:
                return get_center_of_bbox_bottom(player_data['bbox'])

        # Fallback 3: Track ID lain dengan jersey sama
        if self._player_identifier:
            jersey = self._get_jersey(player_id)
            if jersey != "Unknown" and not jersey.startswith("ID:"):
                all_track_ids = self._player_identifier.get_all_track_ids_for_jersey(jersey)

                for alt_tid in all_track_ids:
                    if alt_tid == player_id:
                        continue
                    alt_data, alt_frame = self.find_player_nearby(
                        tracks, alt_tid, frame_target
                    )
                    if alt_data:
                        return get_center_of_bbox_bottom(alt_data['bbox'])

                # Fallback 4: Track ID jersey sama di seluruh segmen
                for alt_tid in all_track_ids:
                    if alt_tid == player_id:
                        continue
                    for f in range(seg_start, min(seg_end + 1, len(tracks['players']))):
                        alt_data = tracks['players'][f].get(alt_tid)
                        if alt_data:
                            return get_center_of_bbox_bottom(alt_data['bbox'])

        return None

    def detect_passes(
        self,
        tracks           : Dict,
        ball_possessions : List[int],
        player_identifier = None,
        debug            : bool = True
    ) -> List[Dict]:
        """
        Deteksi semua event passing dan evaluasi akurasi.

        v2.6 PERUBAHAN:
        - Semua pemain bisa jadi pengirim (termasuk Unknown)
        - Unknown -> Unknown = SKIP
        - Unknown pass: evaluasi ke 3 front cones (radius kecil)
        - #3/#19 pass: evaluasi ke target cone (radius 100px)
        """
        if player_identifier:
            self._player_identifier = player_identifier

        if debug:
            valid_count    = sum(1 for p in ball_possessions if p != -1)
            unique_players = set(p for p in ball_possessions if p != -1)
            print(f"\n[PASS] === PASS DETECTION PIPELINE v2.6 ===")
            print(f"[PASS] Total frames            : {len(ball_possessions)}")
            print(f"[PASS] Frame dengan possession  : {valid_count}/{len(ball_possessions)}")
            print(f"[PASS] Unique player IDs        : {unique_players}")
            print(f"[PASS] Semua pemain bisa pengirim (termasuk Unknown)")
            print(f"[PASS] Smoothing window         : {self.smoothing_window} frames")
            print(f"[PASS] Eval buffer              : "
                  f"-{self.eval_buffer_before}/+{self.eval_buffer_after} frames")
            if self._target_cone_pos:
                print(f"[PASS] Target cone (#3/#19)     : "
                      f"({self._target_cone_pos[0]:.1f}, {self._target_cone_pos[1]:.1f}) "
                      f"r={self.target_proximity_radius:.0f}px")
            if self._front_cone_positions:
                print(f"[PASS] Front cones (Unknown)    : "
                      f"{list(self._front_cone_positions.keys())} "
                      f"r={self.front_cone_radius:.0f}px")

        if sum(1 for p in ball_possessions if p != -1) == 0:
            print("[PASS] Tidak ada possession terdeteksi!")
            return []

        if len(set(p for p in ball_possessions if p != -1)) < 2:
            print("[PASS] Hanya 1 pemain - tidak bisa ada passing!")
            return []

        # --- Preprocessing ---
        filled   = self.fill_short_gaps(ball_possessions, max_gap=20)
        filled   = self._normalize_possessions_by_jersey(filled)

        if debug:
            unique_before_smooth = set(p for p in filled if p != -1)
            print(f"[PASS] Unique IDs sebelum smooth : {unique_before_smooth}")

        smoothed = self.smooth_possessions(filled)

        if debug:
            unique_after_smooth = set(p for p in smoothed if p != -1)
            print(f"[PASS] Unique IDs setelah smooth : {unique_after_smooth}")

        segments = self.get_stable_segments(smoothed)

        if debug:
            print(f"[PASS] Stable segments          : {len(segments)}")

        if len(segments) < 2:
            return []

        # Merge segmen pendek dari pemain yang sama
        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            gap  = seg['frame_start'] - last['frame_end']
            if seg['player_id'] == last['player_id'] and gap <= 10:
                merged[-1]['frame_end'] = seg['frame_end']
            else:
                merged.append(seg)
        segments = merged

        if debug:
            print(f"[PASS] Segments setelah merge   : {len(segments)}")
            for idx, seg in enumerate(segments):
                jersey = self._get_jersey(seg['player_id'])
                print(f"[PASS]   Seg {idx:2d}: track {seg['player_id']:2d} "
                      f"({jersey:>8s}) | frame {seg['frame_start']:3d}-{seg['frame_end']:3d}")

        # --- Deteksi dan evaluasi passes ---
        passes          = []
        last_pass_frame = -999

        for i in range(len(segments) - 1):
            seg_from = segments[i]
            seg_to   = segments[i + 1]

            from_player = seg_from['player_id']
            to_player   = seg_to['player_id']

            transition_frame_start = seg_from['frame_end']
            transition_frame_end   = seg_to['frame_start']

            from_jersey = self._get_jersey(from_player)
            to_jersey   = self._get_jersey(to_player)

            # =========================================================
            # v2.6: Skip Unknown -> Unknown
            # Tidak ada pass antar pemain Unknown yang sama atau berbeda
            # =========================================================
            if from_jersey == "Unknown" and to_jersey == "Unknown":
                if debug:
                    print(f"[PASS] Skip: Unknown -> Unknown "
                          f"(track {from_player} -> {to_player})")
                continue

            # Skip jersey sama (untuk #3->#3 atau #19->#19)
            if from_jersey == to_jersey:
                if debug:
                    print(f"[PASS] Skip: jersey sama ({from_jersey}) | "
                          f"track {from_player} -> {to_player}")
                continue

            # Skip cooldown
            if (transition_frame_end - last_pass_frame) < self.cooldown_frames:
                if debug:
                    print(f"[PASS] Skip: cooldown")
                continue

            # Skip possession terlalu singkat
            from_duration = seg_from['frame_end'] - seg_from['frame_start']
            if from_duration < self.min_possession_duration:
                if debug:
                    print(f"[PASS] Skip: possession {from_jersey} "
                          f"terlalu singkat ({from_duration} frames)")
                continue

            # Cari posisi pemain
            from_pos = self._get_player_position_for_pass(
                tracks, from_player, transition_frame_start, seg_from
            )
            to_pos = self._get_player_position_for_pass(
                tracks, to_player, transition_frame_end, seg_to
            )

            if not from_pos or not to_pos:
                if debug:
                    print(f"[PASS] Skip: posisi pemain {from_jersey} atau "
                          f"{to_jersey} tidak ditemukan")
                continue

            distance = measure_distance(from_pos, to_pos)
            if not (self.min_pass_distance <= distance <= self.max_pass_distance):
                if debug:
                    print(f"[PASS] Skip: jarak {distance:.0f}px di luar range")
                continue

            ball_movement = self.validate_ball_movement(
                tracks, transition_frame_start, transition_frame_end
            )
            if ball_movement < self.ball_movement_threshold:
                if debug:
                    print(f"[PASS] Skip: gerakan bola terlalu kecil "
                          f"({ball_movement:.1f}px)")
                continue

            # =========================================================
            # v2.6: EVALUASI BERBEDA BERDASARKAN PENGIRIM
            # - #3 atau #19 mengirim → cek TARGET CONE (cone 3)
            # - Unknown mengirim    → cek FRONT CONES (cone 0, 1, 2)
            # =========================================================
            temp_event = {
                'frame_start': transition_frame_start,
                'frame_end'  : transition_frame_end,
                'from_pos'   : from_pos,
                'to_pos'     : to_pos,
            }

            closest_dist = float('inf')
            hit_cone_id  = None

            if from_jersey in self.known_jerseys:
                # #3 atau #19 → evaluasi ke target cone
                success, reason = self.evaluate_pass_to_target(
                    tracks, temp_event, debug=debug
                )
                # Hitung closest_dist ke target cone
                trajectory = extract_ball_trajectory(
                    tracks, transition_frame_start, transition_frame_end,
                    buffer_before=self.eval_buffer_before,
                    buffer_after=self.eval_buffer_after
                )
                if self._target_cone_pos and trajectory:
                    closest_dist = min(
                        measure_distance(pt, self._target_cone_pos)
                        for pt in trajectory
                    )
                    hit_cone_id = self._target_cone_id

            else:
                # Unknown → evaluasi ke front cones
                success, reason, hit_cone_id, closest_dist = \
                    self.evaluate_pass_to_front_cones(
                        tracks, temp_event, debug=debug
                    )

            # Frame display
            receiver_start     = seg_to['frame_start']
            pass_display_frame = min(
                receiver_start + self.pass_display_delay,
                seg_to['frame_end']
            )
            if passes:
                last_display = passes[-1]['frame_display']
                if pass_display_frame - last_display < self.min_display_gap:
                    pass_display_frame = last_display + self.min_display_gap

            pass_event = {
                'frame_start'  : transition_frame_start,
                'frame_end'    : transition_frame_end,
                'frame_display': pass_display_frame,
                'from_player'  : from_player,
                'to_player'    : to_player,
                'from_jersey'  : from_jersey,
                'to_jersey'    : to_jersey,
                'distance'     : distance,
                'ball_movement': ball_movement,
                'success'      : success,
                'target_reason': reason,
                'closest_dist' : closest_dist,
                'hit_cone_id'  : hit_cone_id,
                'from_pos'     : from_pos,
                'to_pos'       : to_pos,
            }
            passes.append(pass_event)
            last_pass_frame = transition_frame_end

            if debug:
                status = "SUKSES" if success else "GAGAL"
                eval_type = "target" if from_jersey in self.known_jerseys else "front"
                print(f"[PASS] Pass {from_jersey} -> {to_jersey}: "
                      f"jarak={distance:.0f}px | "
                      f"bola={ball_movement:.0f}px | "
                      f"closest={closest_dist:.0f}px | "
                      f"eval={eval_type} | "
                      f"{status}")

        if debug:
            sukses = sum(1 for p in passes if p['success'])
            gagal  = sum(1 for p in passes if not p['success'])
            pct    = sukses / len(passes) * 100 if passes else 0.0
            print(f"\n[PASS] === HASIL AKHIR ===")
            print(f"[PASS] Total pass  : {len(passes)}")
            print(f"[PASS] SUKSES      : {sukses}")
            print(f"[PASS] GAGAL       : {gagal}")
            print(f"[PASS] Akurasi     : {pct:.1f}%")
            print(f"[PASS] =========================\n")

        return passes

    # ============================================================
    # STATISTIK
    # ============================================================

    def get_pass_statistics(self, passes: List[Dict]) -> Dict:
        total  = len(passes)
        sukses = [p for p in passes if p['success']]
        gagal  = [p for p in passes if not p['success']]

        per_player: Dict[str, Dict] = {}
        for p in passes:
            jersey = p['from_jersey']
            if jersey not in per_player:
                per_player[jersey] = {
                    'total'       : 0,
                    'success'     : 0,
                    'failed'      : 0,
                    'accuracy_pct': 0.0,
                    'avg_closest' : 0.0,
                    '_closest_sum': 0.0
                }
            per_player[jersey]['total'] += 1
            per_player[jersey]['_closest_sum'] += p.get('closest_dist', 0.0)
            if p['success']:
                per_player[jersey]['success'] += 1
            else:
                per_player[jersey]['failed'] += 1

        for jersey, stat in per_player.items():
            stat['accuracy_pct'] = round(
                stat['success'] / stat['total'] * 100, 1
            ) if stat['total'] > 0 else 0.0
            stat['avg_closest'] = round(
                stat['_closest_sum'] / stat['total'], 1
            ) if stat['total'] > 0 else 0.0
            del stat['_closest_sum']

        return {
            'total_passes'           : total,
            'successful_passes'      : len(sukses),
            'failed_passes'          : len(gagal),
            'accuracy_pct'           : round(len(sukses) / total * 100, 1)
                                       if total > 0 else 0.0,
            'avg_distance'           : round(float(np.mean(
                                           [p['distance'] for p in passes])), 1)
                                       if passes else 0.0,
            'avg_distance_successful': round(float(np.mean(
                                           [p['distance'] for p in sukses])), 1)
                                       if sukses else 0.0,
            'avg_closest_dist'       : round(float(np.mean(
                                           [p.get('closest_dist', 0)
                                            for p in passes])), 1)
                                       if passes else 0.0,
            'per_player'             : per_player
        }
