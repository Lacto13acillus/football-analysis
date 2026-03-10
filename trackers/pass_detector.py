# pass_detector.py
# Mendeteksi event passing dan mengevaluasi akurasi.
#
# PERUBAHAN v3.1:
#   - Akurasi hanya dihitung untuk pengirim #3 dan #19
#   - TANPA RADIUS — indikator sukses murni possession-based:
#     SUKSES = Unknown MENDAPAT possession setelah #3/#19 mengirim
#     GAGAL  = Unknown TIDAK mendapat possession (bola hilang/balik)
#   - Mendeteksi juga PERCOBAAN pass yang gagal (failed attempts)
#   - Unknown hanya penerima, tidak dihitung akurasi

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
        self.known_jerseys = {"#3", "#19"}

        # --- Parameter buffer trajectory ---
        self.eval_buffer_before = 0
        self.eval_buffer_after  = 15

        # --- Parameter deteksi failed attempt ---
        # Jumlah frame setelah #3/#19 kehilangan possession
        # untuk mengecek apakah ada percobaan pass (bola bergerak)
        self.failed_attempt_check_window = 30
        self.failed_attempt_min_ball_movement = 40

        # ============================================================
        # KONFIGURASI CONE (OPSIONAL — untuk visualisasi saja)
        # ============================================================
        self.manual_target_cone_id   : Optional[int] = None
        self.target_selection_mode   : str = "highest"
        self.target_proximity_radius : float = 100.0
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
        return dict(self._front_cone_positions)

    def get_front_cone_radius(self) -> float:
        return self.front_cone_radius

    # ============================================================
    # INISIALISASI CONE (OPSIONAL — visualisasi saja)
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

        result = identify_target_cone(
            stabilized_cones      = self._stabilized_cones,
            manual_target_cone_id = self.manual_target_cone_id,
            selection_mode        = self.target_selection_mode
        )

        if result is not None:
            self._target_cone_id, self._target_cone_pos = result

        self._front_cone_positions = {}
        for cid in self.front_cone_ids:
            if cid in self._stabilized_cones:
                self._front_cone_positions[cid] = self._stabilized_cones[cid]

        if debug:
            print(f"\n[TARGET] v3.1: TANPA RADIUS")
            print(f"[TARGET] v3.1: SUKSES = Unknown mendapat possession")
            print(f"[TARGET] v3.1: GAGAL  = Unknown TIDAK mendapat possession")
            print(f"[TARGET] ========================================\n")

        return True

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
    # VALIDASI
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
    # POSISI PEMAIN
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

    # ============================================================
    # HITUNG JARAK TERDEKAT BOLA KE KAKI PENERIMA (INFO SAJA)
    # ============================================================

    def _compute_closest_to_feet(
        self,
        tracks     : Dict,
        frame_start: int,
        frame_end  : int,
        to_player  : int,
        to_pos     : Tuple[int, int]
    ) -> float:
        """
        Hitung jarak terdekat bola ke kaki penerima selama trajectory.
        Ini HANYA untuk informasi/display, BUKAN untuk menentukan sukses/gagal.
        """
        # Cari posisi kaki penerima di sekitar frame penerimaan
        receiver_positions = []
        total_frames = len(tracks['players'])
        search_start = max(0, frame_end - 3)
        search_end   = min(total_frames - 1, frame_end + 15)

        for f in range(search_start, search_end + 1):
            player_data = tracks['players'][f].get(to_player)
            if player_data and 'bbox' in player_data:
                foot_pos = get_center_of_bbox_bottom(player_data['bbox'])
                receiver_positions.append(foot_pos)

        if receiver_positions:
            avg_x = np.mean([p[0] for p in receiver_positions])
            avg_y = np.mean([p[1] for p in receiver_positions])
            receiver_foot = (float(avg_x), float(avg_y))
        else:
            receiver_foot = to_pos

        trajectory = extract_ball_trajectory(
            tracks, frame_start, frame_end,
            buffer_before=self.eval_buffer_before,
            buffer_after=self.eval_buffer_after
        )

        if not trajectory:
            return float('inf')

        return min(measure_distance(pt, receiver_foot) for pt in trajectory)

    # ============================================================
    # DETEKSI PASS UTAMA v3.1 — POSSESSION-BASED
    # ============================================================

    def detect_passes(
        self,
        tracks           : Dict,
        ball_possessions : List[int],
        player_identifier = None,
        debug            : bool = True
    ) -> List[Dict]:
        """
        Deteksi semua event passing dan evaluasi akurasi.

        v3.1 — TANPA RADIUS, murni possession-based:

        1. PASS SUKSES: terdeteksi saat possession berpindah
           dari #3/#19 ke Unknown. Artinya bola TEPAT sampai
           ke kaki Unknown (karena Unknown benar-benar menerima bola).

        2. PASS GAGAL: terdeteksi saat #3/#19 memiliki possession,
           lalu kehilangan bola, dan yang mendapat bola BUKAN Unknown.
           Ini termasuk kasus:
           - #3/#19 -> (tidak ada yang dapat) -> #3/#19 lagi
           - #3/#19 -> (bola hilang lama)
           - #3/#19 -> #3/#19 yang lain (bukan ke Unknown)
        """
        if player_identifier:
            self._player_identifier = player_identifier

        if debug:
            valid_count    = sum(1 for p in ball_possessions if p != -1)
            unique_players = set(p for p in ball_possessions if p != -1)
            print(f"\n[PASS] === PASS DETECTION PIPELINE v3.1 ===")
            print(f"[PASS] Total frames            : {len(ball_possessions)}")
            print(f"[PASS] Frame dengan possession  : {valid_count}/{len(ball_possessions)}")
            print(f"[PASS] Unique player IDs        : {unique_players}")
            print(f"[PASS] Pengirim yang dihitung   : {self.known_jerseys}")
            print(f"[PASS] Penerima target          : Unknown")
            print(f"[PASS] Indikator sukses         : Unknown MENDAPAT possession")
            print(f"[PASS] Indikator gagal          : Unknown TIDAK mendapat possession")
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

        # --- Deteksi passes ---
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
            # FILTER: hanya proses jika PENGIRIM adalah #3 atau #19
            # =========================================================
            if from_jersey not in self.known_jerseys:
                if debug:
                    print(f"[PASS] Skip: pengirim {from_jersey} bukan target analisis")
                continue

            # Skip jersey sama (mis. #3 -> #3)
            if from_jersey == to_jersey:
                if debug:
                    print(f"[PASS] Skip: jersey sama ({from_jersey})")
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
                    print(f"[PASS] Skip: posisi pemain tidak ditemukan")
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
            # v3.1: EVALUASI MURNI POSSESSION-BASED
            #
            # SUKSES = penerima adalah Unknown (bola sampai ke kakinya)
            # GAGAL  = penerima BUKAN Unknown (bola tidak sampai)
            # =========================================================
            if to_jersey == "Unknown":
                success = True
                reason  = (f"SUKSES: Unknown (track {to_player}) "
                          f"menerima bola (possession confirmed)")
            else:
                success = False
                reason  = (f"GAGAL: bola diterima {to_jersey} "
                          f"(track {to_player}), bukan Unknown")

            # Hitung closest distance ke kaki penerima (info saja)
            closest_dist = self._compute_closest_to_feet(
                tracks, transition_frame_start, transition_frame_end,
                to_player, to_pos
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
                'hit_cone_id'  : None,
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
                      f"closest_feet={closest_dist:.0f}px | "
                      f"{status}")

        if debug:
            sukses = sum(1 for p in passes if p['success'])
            gagal  = sum(1 for p in passes if not p['success'])
            pct    = sukses / len(passes) * 100 if passes else 0.0
            print(f"\n[PASS] === HASIL AKHIR ===")
            print(f"[PASS] Total pass  : {len(passes)}")
            print(f"[PASS] SUKSES      : {sukses} (Unknown dapat possession)")
            print(f"[PASS] GAGAL       : {gagal} (Unknown tidak dapat)")
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
