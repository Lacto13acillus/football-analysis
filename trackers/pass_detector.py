# pass_detector.py
# Mendeteksi event passing dan mengevaluasi akurasi ke kaki penerima (Unknown).
#
# PERUBAHAN v3.0:
#   - Akurasi hanya dihitung untuk pengirim #3 dan #19
#   - Indikator sukses: bola sampai ke kaki player Unknown (penerima)
#   - Unknown hanya penerima, pass dari Unknown TIDAK dihitung akurasi
#   - Cone tidak lagi digunakan sebagai indikator keberhasilan
#   - Menambah evaluate_pass_to_receiver_feet()

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
        # v3.0: Hanya #3 dan #19 yang dihitung akurasi passingnya
        # Unknown hanya penerima
        self.known_jerseys = {"#3", "#19"}

        # --- Parameter buffer trajectory ---
        self.eval_buffer_before = 0
        self.eval_buffer_after  = 15

        # ============================================================
        # KONFIGURASI EVALUASI KE KAKI PENERIMA (v3.0)
        # ============================================================
        self.receiver_proximity_radius: float = 100.0  # radius sukses ke kaki penerima (px)
        self.receiver_foot_search_radius: int = 15     # frame search radius untuk posisi kaki penerima

        # ============================================================
        # KONFIGURASI TARGET CONE (OPSIONAL — untuk visualisasi saja)
        # ============================================================
        self.manual_target_cone_id   : Optional[int] = None
        self.target_selection_mode   : str = "highest"
        self.target_proximity_radius : float = 100.0

        # ============================================================
        # KONFIGURASI FRONT CONES (TIDAK DIGUNAKAN untuk evaluasi v3.0)
        # Dipertahankan agar tidak error jika dipanggil
        # ============================================================
        self.front_cone_ids          : List[int] = [0, 1, 2]
        self.front_cone_radius       : float = 125.0

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
    # INISIALISASI TARGET CONE (OPSIONAL — untuk visualisasi)
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
            print(f"\n[TARGET] === INISIALISASI CONE (visualisasi) ===")
            print(f"[TARGET] Stabilisasi posisi cone dari {sample_frames} frame...")

        self._stabilized_cones = stabilize_cone_positions(
            tracks, cone_key=cone_key, sample_frames=sample_frames
        )

        if debug:
            print(f"[TARGET] Total cone terdeteksi: {len(self._stabilized_cones)}")
            for cid, pos in sorted(self._stabilized_cones.items()):
                print(f"[TARGET]   Cone ID {cid:3d} -> "
                      f"({pos[0]:7.1f}, {pos[1]:7.1f})")

        if len(self._stabilized_cones) == 0:
            print("[TARGET] WARNING: Tidak ada cone terdeteksi!")
            return False

        # --- Target cone (untuk visualisasi) ---
        result = identify_target_cone(
            stabilized_cones      = self._stabilized_cones,
            manual_target_cone_id = self.manual_target_cone_id,
            selection_mode        = self.target_selection_mode
        )

        if result is not None:
            self._target_cone_id, self._target_cone_pos = result

        # --- Front cones (untuk visualisasi) ---
        self._front_cone_positions = {}
        for cid in self.front_cone_ids:
            if cid in self._stabilized_cones:
                self._front_cone_positions[cid] = self._stabilized_cones[cid]

        if debug:
            print(f"\n[TARGET] v3.0: Cone TIDAK digunakan untuk evaluasi akurasi")
            print(f"[TARGET] v3.0: Evaluasi = bola sampai ke kaki penerima (Unknown)")
            print(f"[TARGET] v3.0: Receiver proximity radius = {self.receiver_proximity_radius:.0f}px")
            print(f"[TARGET] ========================================\n")

        return True

    # ============================================================
    # EVALUASI PASS KE KAKI PENERIMA (v3.0 — CORE)
    # ============================================================

    def evaluate_pass_to_receiver_feet(
        self,
        tracks     : Dict,
        pass_event : Dict,
        debug      : bool = False
    ) -> Tuple[bool, str, float]:
        """
        Evaluasi apakah bola hasil passing sampai ke kaki penerima (Unknown).

        Logika:
        1. Ambil posisi kaki penerima di sekitar frame penerimaan
        2. Ambil trajectory bola selama event passing
        3. Cek apakah trajectory bola pernah masuk dalam radius
           proximity dari posisi kaki penerima

        Args:
            tracks    : dict hasil tracker
            pass_event: dict event pass (harus punya frame_start, frame_end,
                        to_player, to_pos)
            debug     : cetak info evaluasi

        Returns:
            (reached, reason, closest_dist)
            - reached      : True jika bola sampai ke kaki penerima
            - reason       : penjelasan teks
            - closest_dist : jarak minimum bola ke kaki penerima
        """
        frame_start = pass_event['frame_start']
        frame_end   = pass_event['frame_end']
        to_player   = pass_event['to_player']
        to_pos      = pass_event['to_pos']  # posisi kaki penerima (bottom-center)

        # --- Ambil posisi kaki penerima yang lebih akurat ---
        # Cari di beberapa frame sekitar frame_end untuk posisi terbaik
        receiver_positions = []
        total_frames = len(tracks['players'])

        search_start = max(0, frame_end - 3)
        search_end   = min(total_frames - 1, frame_end + self.receiver_foot_search_radius)

        for f in range(search_start, search_end + 1):
            player_data = tracks['players'][f].get(to_player)
            if player_data and 'bbox' in player_data:
                foot_pos = get_center_of_bbox_bottom(player_data['bbox'])
                receiver_positions.append(foot_pos)

        # Jika tidak ditemukan posisi penerima di sekitar frame_end,
        # cari track ID lain dengan jersey yang sama
        if not receiver_positions and self._player_identifier:
            to_jersey = self._get_jersey(to_player)
            if to_jersey != "Unknown":
                # Untuk non-Unknown, coba track ID lain
                all_tids = self._player_identifier.get_all_track_ids_for_jersey(to_jersey)
                for alt_tid in all_tids:
                    if alt_tid == to_player:
                        continue
                    for f in range(search_start, search_end + 1):
                        player_data = tracks['players'][f].get(alt_tid)
                        if player_data and 'bbox' in player_data:
                            foot_pos = get_center_of_bbox_bottom(player_data['bbox'])
                            receiver_positions.append(foot_pos)
                    if receiver_positions:
                        break

        # Gunakan posisi rata-rata kaki penerima, atau fallback ke to_pos
        if receiver_positions:
            avg_x = np.mean([p[0] for p in receiver_positions])
            avg_y = np.mean([p[1] for p in receiver_positions])
            receiver_foot = (float(avg_x), float(avg_y))
        else:
            receiver_foot = to_pos

        # --- Ambil trajectory bola ---
        trajectory = extract_ball_trajectory(
            tracks, frame_start, frame_end,
            buffer_before=self.eval_buffer_before,
            buffer_after=self.eval_buffer_after
        )

        if debug:
            print(f"[RECV] Evaluasi pass frame {frame_start}-{frame_end}: "
                  f"{len(trajectory)} titik trajectory")
            print(f"[RECV] Posisi kaki penerima: ({receiver_foot[0]:.1f}, {receiver_foot[1]:.1f})")
            print(f"[RECV] Radius sukses: {self.receiver_proximity_radius:.0f}px")

        if not trajectory:
            return False, "Tidak ada trajectory bola", float('inf')

        # --- Cek jarak minimum bola ke kaki penerima ---
        min_distance = float('inf')
        min_idx = -1
        for i, point in enumerate(trajectory):
            dist = measure_distance(point, receiver_foot)
            if dist < min_distance:
                min_distance = dist
                min_idx = i

        reached = min_distance <= self.receiver_proximity_radius

        if reached:
            reason = (f"Bola sampai ke kaki penerima: jarak minimum "
                     f"{min_distance:.1f}px di titik ke-{min_idx} "
                     f"(radius={self.receiver_proximity_radius:.0f}px)")
        else:
            reason = (f"Bola tidak sampai ke kaki penerima: jarak minimum "
                     f"{min_distance:.1f}px > radius {self.receiver_proximity_radius:.0f}px")

        if debug:
            status = "SUKSES" if reached else "GAGAL"
            print(f"[RECV] Hasil: {status} | closest={min_distance:.1f}px | {reason}")

        return reached, reason, min_distance

    # ============================================================
    # EVALUASI LAMA (dipertahankan untuk backward compat jika perlu)
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

        reached, reason = check_ball_reached_target_cone(
            ball_trajectory  = trajectory,
            target_cone_pos  = self._target_cone_pos,
            proximity_radius = self.target_proximity_radius
        )

        return reached, reason

    def evaluate_pass_to_front_cones(
        self,
        tracks    : Dict,
        pass_event: Dict,
        debug     : bool = False
    ) -> Tuple[bool, str, Optional[int], float]:
        """Dipertahankan untuk backward compatibility, tidak dipakai v3.0."""
        if not self._front_cone_positions:
            return True, "Front cones tidak diinisialisasi", None, 0.0

        frame_start = pass_event['frame_start']
        frame_end   = pass_event['frame_end']

        trajectory = extract_ball_trajectory(
            tracks, frame_start, frame_end,
            buffer_before=self.eval_buffer_before,
            buffer_after=self.eval_buffer_after
        )

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

        if not best_reached:
            best_reason = (f"Bola tidak mencapai cone depan manapun: "
                          f"closest={overall_closest_dist:.1f}px ke cone {overall_closest_cone}")

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
        player_data, found_frame = self.find_player_nearby(
            tracks, player_id, frame_target
        )
        if player_data:
            return get_center_of_bbox_bottom(player_data['bbox'])

        seg_start = segment['frame_start']
        seg_end   = segment['frame_end']
        for f in range(seg_start, min(seg_end + 1, len(tracks['players']))):
            player_data = tracks['players'][f].get(player_id)
            if player_data:
                return get_center_of_bbox_bottom(player_data['bbox'])

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

        v3.0 PERUBAHAN:
        - HANYA #3 dan #19 yang dihitung akurasi passingnya
        - Indikator SUKSES: bola sampai ke kaki player Unknown (penerima)
        - Pass dari Unknown TIDAK dihitung (Unknown hanya penerima)
        - Unknown -> Unknown = SKIP
        - #3 -> #19 atau #19 -> #3 = SKIP (bukan pass ke penerima Unknown)
        """
        if player_identifier:
            self._player_identifier = player_identifier

        if debug:
            valid_count    = sum(1 for p in ball_possessions if p != -1)
            unique_players = set(p for p in ball_possessions if p != -1)
            print(f"\n[PASS] === PASS DETECTION PIPELINE v3.0 ===")
            print(f"[PASS] Total frames            : {len(ball_possessions)}")
            print(f"[PASS] Frame dengan possession  : {valid_count}/{len(ball_possessions)}")
            print(f"[PASS] Unique player IDs        : {unique_players}")
            print(f"[PASS] Pengirim yang dihitung   : {self.known_jerseys}")
            print(f"[PASS] Penerima target          : Unknown")
            print(f"[PASS] Indikator sukses         : bola sampai ke kaki penerima")
            print(f"[PASS] Receiver radius          : {self.receiver_proximity_radius:.0f}px")
            print(f"[PASS] Smoothing window         : {self.smoothing_window} frames")
            print(f"[PASS] Eval buffer              : "
                  f"-{self.eval_buffer_before}/+{self.eval_buffer_after} frames")

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
            # v3.0: FILTER — hanya hitung pass dari #3/#19 ke Unknown
            # =========================================================

            # Skip Unknown -> Unknown
            if from_jersey == "Unknown" and to_jersey == "Unknown":
                if debug:
                    print(f"[PASS] Skip: Unknown -> Unknown "
                          f"(track {from_player} -> {to_player})")
                continue

            # Skip jersey sama
            if from_jersey == to_jersey:
                if debug:
                    print(f"[PASS] Skip: jersey sama ({from_jersey}) | "
                          f"track {from_player} -> {to_player}")
                continue

            # v3.0: HANYA hitung pass dari #3/#19 (pengirim yang dihitung)
            if from_jersey not in self.known_jerseys:
                if debug:
                    print(f"[PASS] Skip: pengirim {from_jersey} bukan target analisis "
                          f"(hanya {self.known_jerseys})")
                continue

            # v3.0: Penerima HARUS Unknown
            if to_jersey != "Unknown":
                if debug:
                    print(f"[PASS] Skip: penerima {to_jersey} bukan Unknown "
                          f"(#3/#19 -> #3/#19 bukan pass ke penerima)")
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
            # v3.0: EVALUASI KE KAKI PENERIMA (Unknown)
            # =========================================================
            temp_event = {
                'frame_start': transition_frame_start,
                'frame_end'  : transition_frame_end,
                'from_pos'   : from_pos,
                'to_pos'     : to_pos,
                'to_player'  : to_player,
            }

            success, reason, closest_dist = self.evaluate_pass_to_receiver_feet(
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
                'hit_cone_id'  : None,  # v3.0: tidak pakai cone
                'from_pos'     : from_pos,
                'to_pos'       : to_pos,
            }
            passes.append(pass_event)
            last_pass_frame = transition_frame_end

            if debug:
                status = "SUKSES" if success else "GAGAL"
                print(f"[PASS] Pass {from_jersey} -> {to_jersey}: "
                      f"jarak={distance:.0f}px | "
                      f"bola={ball_movement:.0f}px | "
                      f"closest_to_feet={closest_dist:.0f}px | "
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
