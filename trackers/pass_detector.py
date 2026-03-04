# pass_detector.py
# Mendeteksi event passing dan mengevaluasi apakah bola mencapai cone target.
# 
# LOGIKA SUKSES (diperbarui):
#   Pass SUKSES = trajectory bola mendekati cone TARGET (cone paling atas)
#   dalam radius yang ditentukan.
#   Abaikan 3 cone yang berjejer di tengah lapangan.

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
        """
        Inisialisasi parameter pass detector.

        Args:
            fps: frame per second video
        """
        self.fps = fps

        # --- Parameter smoothing possession ---
        self.smoothing_window        = 3
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

        # ============================================================
        # KONFIGURASI TARGET CONE
        #
        # Pass dianggap SUKSES jika bola mendekati cone target
        # dalam proximity_radius pixel.
        #
        # OPSI 1 (DISARANKAN): Manual cone ID
        #   Set self.manual_target_cone_id = 3 (dari log: Cone ID 3 = paling atas)
        #
        # OPSI 2: Auto - pilih cone paling atas (Y terkecil)
        #   Set self.manual_target_cone_id = None
        #   Set self.target_selection_mode = "highest"
        # ============================================================

        # ID cone target (None = auto-detect)
        self.manual_target_cone_id : Optional[int] = None

        # Mode auto-selection: "highest", "lowest", "leftmost", "rightmost"
        self.target_selection_mode : str = "highest"

        # Radius keberhasilan: bola harus mendekati cone target
        # dalam jarak ini (pixel). Sesuaikan dengan skala video.
        # Dari video Anda: cone berukuran ~30-40px, set 120px sebagai buffer.
        self.target_proximity_radius : float = 80

        # Cache
        self._target_cone_id  : Optional[int] = None
        self._target_cone_pos : Optional[Tuple[float, float]] = None
        self._stabilized_cones: Optional[Dict] = None
        self._player_identifier = None

    # ============================================================
    # HELPER
    # ============================================================

    def set_jersey_map(self, player_identifier) -> None:
        """Simpan referensi ke PlayerIdentifier."""
        self._player_identifier = player_identifier

    def _get_jersey(self, player_id: int) -> str:
        """Dapatkan nomor jersey dari player ID."""
        if self._player_identifier:
            return self._player_identifier.get_jersey_number_for_player(player_id)
        return str(player_id)

    def get_target_cone(self) -> Optional[Tuple[int, Tuple[float, float]]]:
        """
        Kembalikan (cone_id, posisi) target cone yang sudah diidentifikasi.
        Digunakan oleh main.py untuk visualisasi.
        """
        if self._target_cone_id is None or self._target_cone_pos is None:
            return None
        return self._target_cone_id, self._target_cone_pos

    def get_all_cones(self) -> Optional[Dict[int, Tuple[float, float]]]:
        """Kembalikan semua posisi cone yang sudah distabilisasi."""
        return self._stabilized_cones

    # ============================================================
    # INISIALISASI TARGET CONE
    # ============================================================

    def initialize_target_cone(
        self,
        tracks       : Dict,
        cone_key     : str = 'cones',
        sample_frames: int = 30,
        debug        : bool = True
    ) -> bool:
        """
        Inisialisasi posisi target cone dari data tracks.
        Harus dipanggil SEBELUM detect_passes().

        Berbeda dari initialize_gate(), fungsi ini hanya
        mengidentifikasi SATU cone target, bukan pasangan cone.

        Args:
            tracks       : dict hasil tracker
            cone_key     : nama key untuk data cone di tracks
            sample_frames: jumlah frame untuk averaging posisi cone
            debug        : tampilkan info debug ke console

        Returns:
            True jika target cone berhasil diidentifikasi
        """
        if cone_key not in tracks or not tracks[cone_key]:
            print(f"[TARGET] WARNING: Key '{cone_key}' tidak ada di tracks!")
            return False

        if debug:
            print(f"\n[TARGET] === INISIALISASI TARGET CONE ===")
            print(f"[TARGET] Stabilisasi posisi cone dari {sample_frames} frame...")

        # Stabilisasi semua cone
        self._stabilized_cones = stabilize_cone_positions(
            tracks,
            cone_key      = cone_key,
            sample_frames = sample_frames
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

        # Identifikasi target cone
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
            print(f"[TARGET] ========================================\n")

        return True

    # ============================================================
    # EVALUASI PASS KE TARGET CONE
    # ============================================================

    def evaluate_pass_to_target(
        self,
        tracks    : Dict,
        pass_event: Dict,
        debug     : bool = False
    ) -> Tuple[bool, str]:
        """
        Evaluasi apakah bola mendekati cone target selama pass event.
        Menggunakan buffer_frames=0 agar trajectory tidak terkontaminasi
        oleh posisi bola dari pass event lain yang berdekatan.
        """
        if self._target_cone_pos is None:
            return True, "Target cone tidak diinisialisasi - semua pass = SUKSES"
        frame_start = pass_event['frame_start']
        frame_end   = pass_event['frame_end']
        # buffer_frames=0 -> hanya ambil frame dalam event ini saja
        trajectory = extract_ball_trajectory(
            tracks,
            frame_start,
            frame_end,
            buffer_frames=0    # <-- PENTING: tanpa buffer
        )
        if debug:
            print(f"[TARGET] Evaluasi pass frame {frame_start}-{frame_end}: "
                  f"{len(trajectory)} titik trajectory")
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
    # SMOOTHING & PREPROCESSING POSSESSION
    # ============================================================

    def smooth_possessions(self, raw_possessions: List[int]) -> List[int]:
        """Haluskan possession dengan sliding window majority vote."""
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
        max_gap    : int = 12
    ) -> List[int]:
        """Isi gap pendek (-1) di antara possession yang sama."""
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
        """
        Normalisasi possession berdasarkan jersey number.
        Mencegah ByteTrack ID switching menyebabkan false pass.
        """
        if not self._player_identifier:
            return possessions

        jersey_canonical: Dict[str, int] = {}
        normalized = list(possessions)

        for i, pid in enumerate(possessions):
            if pid == -1:
                continue
            jersey = self._get_jersey(pid)
            if jersey not in jersey_canonical:
                jersey_canonical[jersey] = pid
            normalized[i] = jersey_canonical[jersey]

        return normalized

    def get_stable_segments(
        self,
        smoothed_possessions: List[int]
    ) -> List[Dict]:
        """
        Pecah possession menjadi segmen-segmen stabil.
        Satu segmen = satu pemain menguasai bola secara konsisten.
        """
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
        """Validasi bola benar-benar bergerak selama transisi possession."""
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
        """Cari data pemain di frame target, fallback ke frame terdekat."""
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
        """
        Dapatkan posisi pemain untuk pass event dengan multiple fallback.
        Fallback order:
        1. Frame tepat di transisi
        2. Frame dalam search_radius di sekitar transisi
        3. Frame mana saja dalam segment pemain
        4. None jika benar-benar tidak ada data
        Args:
            tracks      : dict hasil tracker
            player_id   : track ID pemain
            frame_target: frame target utama
            segment     : segment possession pemain ini
        Returns:
            (x, y) posisi kaki pemain, atau None
        """
        # Fallback 1 & 2: cari di sekitar frame transisi
        player_data, found_frame = self.find_player_nearby(
            tracks, player_id, frame_target
        )
        if player_data:
            return get_center_of_bbox_bottom(player_data['bbox'])
        # Fallback 3: cari di seluruh segment
        seg_start = segment['frame_start']
        seg_end   = segment['frame_end']
        for f in range(seg_start, min(seg_end + 1, len(tracks['players']))):
            player_data = tracks['players'][f].get(player_id)
            if player_data:
                return get_center_of_bbox_bottom(player_data['bbox'])
        return None

    def detect_passes(
        self,
        tracks           : Dict,
        ball_possessions : List[int],
        player_identifier = None,
        debug            : bool = True
    ) -> List[Dict]:
        """
        Deteksi semua event passing dan evaluasi ke target cone.
        """
        if player_identifier:
            self._player_identifier = player_identifier
        if debug:
            valid_count    = sum(1 for p in ball_possessions if p != -1)
            unique_players = set(p for p in ball_possessions if p != -1)
            print(f"\n[PASS] === PASS DETECTION PIPELINE ===")
            print(f"[PASS] Total frames            : {len(ball_possessions)}")
            print(f"[PASS] Frame dengan possession  : {valid_count}/{len(ball_possessions)}")
            print(f"[PASS] Unique player IDs        : {unique_players}")
            if self._target_cone_pos:
                print(f"[PASS] Target cone posisi       : "
                      f"({self._target_cone_pos[0]:.1f}, {self._target_cone_pos[1]:.1f})")
                print(f"[PASS] Radius sukses            : "
                      f"{self.target_proximity_radius:.0f} px")
        if sum(1 for p in ball_possessions if p != -1) == 0:
            print("[PASS] Tidak ada possession terdeteksi!")
            return []
        if len(set(p for p in ball_possessions if p != -1)) < 2:
            print("[PASS] Hanya 1 pemain - tidak bisa ada passing!")
            return []
        # --- Preprocessing ---
        filled   = self.fill_short_gaps(ball_possessions, max_gap=12)
        filled   = self._normalize_possessions_by_jersey(filled)
        smoothed = self.smooth_possessions(filled)
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
            # Skip jersey sama (ID switching)
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
                    print(f"[PASS] Skip: possession #{from_jersey} "
                          f"terlalu singkat ({from_duration} frames)")
                continue
            # Cari posisi pemain dengan multi-fallback
            from_pos = self._get_player_position_for_pass(
                tracks, from_player, transition_frame_start, seg_from
            )
            to_pos = self._get_player_position_for_pass(
                tracks, to_player, transition_frame_end, seg_to
            )
            if not from_pos or not to_pos:
                if debug:
                    print(f"[PASS] Skip: posisi pemain #{from_jersey} atau "
                          f"#{to_jersey} tidak ditemukan sama sekali")
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
            # Evaluasi target cone
            temp_event = {
                'frame_start': transition_frame_start,
                'frame_end'  : transition_frame_end,
                'from_pos'   : from_pos,
                'to_pos'     : to_pos,
            }
            success, reason = self.evaluate_pass_to_target(
                tracks, temp_event, debug=debug
            )
            # Hitung closest_dist
            trajectory   = extract_ball_trajectory(
                tracks, transition_frame_start, transition_frame_end,
                buffer_frames=0    # <-- buffer=0 juga di sini
            )
            closest_dist = float('inf')
            if self._target_cone_pos and trajectory:
                closest_dist = min(
                    measure_distance(pt, self._target_cone_pos)
                    for pt in trajectory
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
                'from_pos'     : from_pos,
                'to_pos'       : to_pos,
            }
            passes.append(pass_event)
            last_pass_frame = transition_frame_end
            if debug:
                status = "SUKSES" if success else "GAGAL"
                print(f"[PASS] Pass #{from_jersey} -> #{to_jersey}: "
                      f"jarak={distance:.0f}px | "
                      f"bola={ball_movement:.0f}px | "
                      f"closest={closest_dist:.0f}px | "
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
        """
        Hitung statistik keseluruhan dan per pemain.

        Returns:
            Dict dengan keys:
            total_passes, successful_passes, failed_passes,
            accuracy_pct, avg_distance, avg_distance_successful,
            avg_closest_dist, per_player
        """
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