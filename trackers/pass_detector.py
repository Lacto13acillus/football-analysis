# pass_detector.py
# Mendeteksi event passing dan mengevaluasi apakah bola melewati gate cone.

import sys
sys.path.append('../')

from utils.bbox_utils import (
    measure_distance,
    get_center_of_bbox_bottom,
    get_center_of_bbox,
    extract_ball_trajectory,
    check_ball_passed_through_gate,
    stabilize_cone_positions,
    identify_gate_cones
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
        self.smoothing_window      = 3   # Window untuk majority vote smoothing
        self.min_stable_frames     = 1   # Minimum frame agar segment valid

        # --- Parameter validasi pass ---
        self.min_pass_distance     = 30      # Jarak minimum passing (pixel)
        self.max_pass_distance     = 1500    # Jarak maksimum passing (pixel)
        self.cooldown_frames       = 3       # Jeda minimum antar pass
        self.min_possession_duration = 1     # Minimum frame penguasaan sebelum pass

        # --- Parameter validasi gerakan bola ---
        self.ball_movement_check_radius = 25
        self.ball_movement_threshold    = 4  # Minimum pixel gerakan bola agar valid

        # --- Parameter display ---
        self.player_search_radius  = 20
        self.pass_display_delay    = 3
        self.min_display_gap       = 3

        # ============================================================
        # KONFIGURASI GATE
        # Pilih salah satu opsi (prioritas dari atas ke bawah):
        # ============================================================

        # OPSI 1: Manual - masukkan (id_cone_1, id_cone_2)
        # Jalankan debug_find_gate_ids.py untuk menemukan ID yang benar
        # Contoh: self.manual_gate_cone_ids = (3, 5)
        self.manual_gate_cone_ids: Optional[Tuple[int, int]] = None

        # OPSI 2: Hint koordinat area gate di video (pixel)
        # Contoh: self.gate_position_hint = ((400.0, 300.0), (500.0, 320.0))
        self.gate_position_hint: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

        # OPSI 3: Auto-detection berdasarkan range lebar gate (pixel)
        self.expected_gate_width_range: Tuple[float, float] = (60.0, 300.0)

        # Threshold proximity untuk Metode B fallback (pixel)
        self.gate_proximity_threshold: float = 40.0

        # Cache gate & cones yang sudah diinisialisasi
        self._gate: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self._stabilized_cones: Optional[Dict] = None
        self._player_identifier = None

    # ============================================================
    # HELPER: Jersey & Player Identifier
    # ============================================================

    def set_jersey_map(self, player_identifier) -> None:
        """Simpan referensi ke PlayerIdentifier untuk normalisasi jersey."""
        self._player_identifier = player_identifier

    def _get_jersey(self, player_id: int) -> str:
        """Dapatkan nomor jersey dari player ID."""
        if self._player_identifier:
            return self._player_identifier.get_jersey_number_for_player(player_id)
        return str(player_id)

    # ============================================================
    # INISIALISASI GATE
    # ============================================================

    def initialize_gate(
        self,
        tracks       : Dict,
        cone_key     : str = 'cones',
        sample_frames: int = 30,
        debug        : bool = True
    ) -> bool:
        """
        Inisialisasi posisi gate dari data tracks.
        Harus dipanggil SEBELUM detect_passes().
        Kamera harus statis agar stabilisasi cone akurat.

        Args:
            tracks       : dict hasil tracker (harus ada key cone_key)
            cone_key     : nama key untuk data cone di tracks
            sample_frames: jumlah frame untuk averaging posisi cone
            debug        : tampilkan info debug ke console

        Returns:
            True jika gate berhasil diidentifikasi, False jika gagal
        """
        if cone_key not in tracks or not tracks[cone_key]:
            print(f"[GATE] WARNING: Key '{cone_key}' tidak ada atau kosong di tracks!")
            print(f"[GATE] Pastikan tracker.py menyimpan cone di tracks['{cone_key}']")
            return False

        if debug:
            print(f"\n[GATE] === INISIALISASI GATE ===")
            print(f"[GATE] Stabilisasi posisi cone dari {sample_frames} frame pertama...")

        # Stabilisasi posisi semua cone dari N frame pertama
        self._stabilized_cones = stabilize_cone_positions(
            tracks,
            cone_key      = cone_key,
            sample_frames = sample_frames
        )

        if debug:
            print(f"[GATE] Total cone stabil terdeteksi: {len(self._stabilized_cones)}")
            for cid, pos in sorted(self._stabilized_cones.items()):
                print(f"[GATE]   Cone ID {cid:3d} -> posisi ({pos[0]:.1f}, {pos[1]:.1f})")

        if len(self._stabilized_cones) < 2:
            print("[GATE] GAGAL: Kurang dari 2 cone terdeteksi!")
            return False

        # Identifikasi 2 cone yang membentuk gate
        gate_result = identify_gate_cones(
            stabilized_cones          = self._stabilized_cones,
            gate_hint                 = self.gate_position_hint,
            manual_cone_ids           = self.manual_gate_cone_ids,
            expected_gate_width_range = self.expected_gate_width_range
        )

        if gate_result is None:
            print("[GATE] GAGAL: Tidak bisa mengidentifikasi gate cone!")
            print("[GATE] Tips: Jalankan debug_find_gate_ids.py lalu set")
            print("[GATE]       pass_detector.manual_gate_cone_ids = (id1, id2)")
            return False

        self._gate              = gate_result
        gate_left, gate_right   = self._gate
        gate_width              = measure_distance(gate_left, gate_right)

        if debug:
            print(f"\n[GATE] Gate berhasil diidentifikasi!")
            print(f"[GATE]   Cone Kiri : ({gate_left[0]:.1f},  {gate_left[1]:.1f})")
            print(f"[GATE]   Cone Kanan: ({gate_right[0]:.1f}, {gate_right[1]:.1f})")
            print(f"[GATE]   Lebar Gate: {gate_width:.1f} px")
            print(f"[GATE] ================================\n")

        return True

    def get_gate(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Kembalikan posisi gate yang sudah diidentifikasi."""
        return self._gate

    # ============================================================
    # EVALUASI GATE PER PASS EVENT
    # ============================================================

    def evaluate_pass_through_gate(
        self,
        tracks     : Dict,
        pass_event : Dict,
        debug      : bool = False
    ) -> Tuple[bool, str]:
        """
        Evaluasi apakah sebuah pass event berhasil melewati gate.

        Mengekstrak trajectory nyata bola selama event, lalu
        cek apakah trajectory memotong garis gate.

        Args:
            tracks    : dict hasil tracker
            pass_event: dict pass event dari detect_passes()
            debug     : tampilkan info debug

        Returns:
            (success: bool, reason: str)
        """
        # Jika gate belum diinisialisasi, anggap semua pass sukses
        if self._gate is None:
            return True, "Gate tidak diinisialisasi - semua pass dianggap sukses"

        gate_left, gate_right = self._gate
        frame_start           = pass_event['frame_start']
        frame_end             = pass_event['frame_end']

        # Ekstrak trajectory nyata bola selama event berlangsung
        trajectory = extract_ball_trajectory(tracks, frame_start, frame_end)

        if debug:
            print(f"[GATE] Evaluasi pass frame {frame_start}-{frame_end}: "
                  f"{len(trajectory)} titik trajectory")

        # Cek apakah trajectory melewati gate
        passed, reason = check_ball_passed_through_gate(
            ball_trajectory     = trajectory,
            gate_cone_left      = gate_left,
            gate_cone_right     = gate_right,
            proximity_threshold = self.gate_proximity_threshold
        )

        if debug:
            status = "SUKSES" if passed else "GAGAL"
            print(f"[GATE]   Hasil: {status} | {reason}")

        return passed, reason

    # ============================================================
    # SMOOTHING & PREPROCESSING POSSESSION
    # ============================================================

    def smooth_possessions(self, raw_possessions: List[int]) -> List[int]:
        """
        Haluskan possession menggunakan sliding window majority vote.
        Mengurangi false possession akibat noise deteksi per frame.

        Args:
            raw_possessions: list raw possession per frame

        Returns:
            List possession yang sudah dihaluskan
        """
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
        """
        Isi gap pendek (-1) di antara possession yang sama.
        Berguna untuk frame di mana bola momentan tidak terdeteksi.

        Args:
            possessions: list possession per frame
            max_gap    : maksimal frame gap yang boleh diisi

        Returns:
            List possession dengan gap pendek terisi
        """
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
        Mencegah ByteTrack ID switching menyebabkan false pass detection.
        Pemain dengan jersey sama selalu dikodekan dengan ID yang konsisten.

        Args:
            possessions: list raw possession

        Returns:
            List possession yang sudah dinormalisasi
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
        Pecah possession menjadi segmen-segmen stabil berurutan.
        Satu segmen = periode di mana satu pemain menguasai bola secara konsisten.

        Args:
            smoothed_possessions: list possession setelah smoothing

        Returns:
            List dict: [{'player_id', 'frame_start', 'frame_end'}, ...]
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

        # Tambahkan segmen terakhir
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
        tracks      : Dict,
        frame_start : int,
        frame_end   : int
    ) -> float:
        """
        Validasi bahwa bola benar-benar bergerak selama transisi possession.
        Mencegah false pass akibat ID switching tanpa gerakan bola nyata.

        Args:
            tracks     : dict hasil tracker
            frame_start: frame awal transisi
            frame_end  : frame akhir transisi

        Returns:
            float: jarak maksimal gerakan bola (pixel)
        """
        check_start = max(0, frame_start - self.ball_movement_check_radius)
        check_end   = min(len(tracks['ball']), frame_end + self.ball_movement_check_radius)

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
        """
        Cari data pemain di frame target, dengan fallback ke frame terdekat.
        Berguna saat pemain tidak terdeteksi tepat di frame transisi.

        Args:
            tracks       : dict hasil tracker
            player_id    : track ID pemain yang dicari
            target_frame : frame yang dituju
            search_radius: jumlah frame yang diperiksa ke kiri/kanan

        Returns:
            (player_data dict atau None, frame_num yang digunakan)
        """
        if search_radius is None:
            search_radius = self.player_search_radius

        total_frames = len(tracks['players'])

        # Cek frame target langsung
        player_data = tracks['players'][target_frame].get(player_id)
        if player_data:
            return player_data, target_frame

        # Cari di frame sekitar target
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

    def detect_passes(
        self,
        tracks           : Dict,
        ball_possessions : List[int],
        player_identifier = None,
        debug            : bool = True
    ) -> List[Dict]:
        """
        Deteksi semua event passing dan evaluasi keberhasilan melewati gate.

        Pipeline internal:
        1. Fill gaps pendek di possession
        2. Normalisasi jersey (anti ID-switching)
        3. Smoothing possession
        4. Bentuk stable segments
        5. Merge segment sama yang berdekatan
        6. Deteksi transisi antar pemain sebagai pass
        7. Validasi jarak & gerakan bola
        8. Evaluasi gate (SUKSES / GAGAL)

        Args:
            tracks           : dict hasil tracker
            ball_possessions : list possession per frame dari PlayerBallAssigner
            player_identifier: objek PlayerIdentifier (opsional)
            debug            : tampilkan pipeline debug ke console

        Returns:
            List dict pass event:
            {
                'frame_start'  : int,
                'frame_end'    : int,
                'frame_display': int,
                'from_player'  : int,
                'to_player'    : int,
                'from_jersey'  : str,
                'to_jersey'    : str,
                'distance'     : float,
                'ball_movement': float,
                'success'      : bool,
                'gate_reason'  : str,
                'from_pos'     : tuple,
                'to_pos'       : tuple
            }
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

        # Guard: tidak cukup data
        if sum(1 for p in ball_possessions if p != -1) == 0:
            print("[PASS] Tidak ada possession terdeteksi!")
            return []

        if len(set(p for p in ball_possessions if p != -1)) < 2:
            print("[PASS] Hanya 1 pemain yang terdeteksi, tidak bisa ada passing!")
            return []

        # ----------------------------------------------------------
        # TAHAP 1: Isi gap pendek
        # ----------------------------------------------------------
        filled = self.fill_short_gaps(ball_possessions, max_gap=12)

        # ----------------------------------------------------------
        # TAHAP 2: Normalisasi berdasarkan jersey
        # ----------------------------------------------------------
        filled = self._normalize_possessions_by_jersey(filled)

        # ----------------------------------------------------------
        # TAHAP 3: Smoothing
        # ----------------------------------------------------------
        smoothed = self.smooth_possessions(filled)

        # ----------------------------------------------------------
        # TAHAP 4: Bentuk segmen stabil
        # ----------------------------------------------------------
        segments = self.get_stable_segments(smoothed)

        if debug:
            print(f"[PASS] Stable segments          : {len(segments)}")

        if len(segments) < 2:
            print("[PASS] Kurang dari 2 segmen, tidak ada passing!")
            return []

        # ----------------------------------------------------------
        # TAHAP 5: Merge segmen pendek dari pemain yang sama
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # TAHAP 6-8: Deteksi dan evaluasi passes
        # ----------------------------------------------------------
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

            # --- Guard: jersey sama = ID switching, bukan passing ---
            if from_jersey == to_jersey:
                if debug:
                    print(f"[PASS] Skip: jersey sama ({from_jersey}) "
                          f"di frame {transition_frame_start}")
                continue

            # --- Guard: masih dalam cooldown ---
            if (transition_frame_end - last_pass_frame) < self.cooldown_frames:
                if debug:
                    print(f"[PASS] Skip: cooldown "
                          f"({transition_frame_end - last_pass_frame} frames)")
                continue

            # --- Guard: possession terlalu singkat ---
            from_duration = seg_from['frame_end'] - seg_from['frame_start']
            if from_duration < self.min_possession_duration:
                if debug:
                    print(f"[PASS] Skip: possession #{from_jersey} "
                          f"terlalu singkat ({from_duration} frames)")
                continue

            # --- Cari data posisi pemain di sekitar frame transisi ---
            from_player_data, _ = self.find_player_nearby(
                tracks, from_player, transition_frame_start
            )
            to_player_data, _ = self.find_player_nearby(
                tracks, to_player, transition_frame_end
            )

            if not from_player_data or not to_player_data:
                if debug:
                    print(f"[PASS] Skip: data pemain tidak ditemukan "
                          f"di sekitar frame {transition_frame_start}-{transition_frame_end}")
                continue

            from_pos = get_center_of_bbox_bottom(from_player_data['bbox'])
            to_pos   = get_center_of_bbox_bottom(to_player_data['bbox'])
            distance = measure_distance(from_pos, to_pos)

            # --- Guard: validasi jarak passing ---
            if not (self.min_pass_distance <= distance <= self.max_pass_distance):
                if debug:
                    print(f"[PASS] Skip: jarak {distance:.0f}px di luar range "
                          f"[{self.min_pass_distance}, {self.max_pass_distance}]")
                continue

            # --- Guard: validasi gerakan bola nyata ---
            ball_movement = self.validate_ball_movement(
                tracks, transition_frame_start, transition_frame_end
            )
            if ball_movement < self.ball_movement_threshold:
                if debug:
                    print(f"[PASS] Skip: gerakan bola terlalu kecil "
                          f"({ball_movement:.1f}px, min={self.ball_movement_threshold}px)")
                continue

            # --- Evaluasi gate: apakah bola melewati gawang? ---
            temp_pass_event = {
                'frame_start': transition_frame_start,
                'frame_end'  : transition_frame_end,
                'from_pos'   : from_pos,
                'to_pos'     : to_pos,
            }
            gate_success, gate_reason = self.evaluate_pass_through_gate(
                tracks, temp_pass_event, debug=debug
            )

            # --- Hitung frame display ---
            receiver_start     = seg_to['frame_start']
            pass_display_frame = min(
                receiver_start + self.pass_display_delay,
                seg_to['frame_end']
            )
            if passes:
                last_display = passes[-1]['frame_display']
                if pass_display_frame - last_display < self.min_display_gap:
                    pass_display_frame = last_display + self.min_display_gap

            # --- Simpan pass event ---
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
                'success'      : gate_success,
                'gate_reason'  : gate_reason,
                'from_pos'     : from_pos,
                'to_pos'       : to_pos,
            }

            passes.append(pass_event)
            last_pass_frame = transition_frame_end

            if debug:
                status = "SUKSES" if gate_success else "GAGAL"
                print(f"[PASS] Pass #{from_jersey} -> #{to_jersey}: "
                      f"jarak={distance:.0f}px | "
                      f"bola={ball_movement:.0f}px | "
                      f"{status}")

        # --- Ringkasan debug ---
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
        Hitung statistik keseluruhan dan per pemain dari list pass events.

        Args:
            passes: list pass event dari detect_passes()

        Returns:
            Dict statistik lengkap:
            {
                'total_passes', 'successful_passes', 'failed_passes',
                'accuracy_pct', 'avg_distance', 'avg_distance_successful',
                'per_player': {
                    jersey: {'total', 'success', 'failed', 'accuracy_pct'}
                }
            }
        """
        total  = len(passes)
        sukses = [p for p in passes if p['success']]
        gagal  = [p for p in passes if not p['success']]

        # Statistik per pemain (berdasarkan pengirim/sender)
        per_player: Dict[str, Dict] = {}
        for p in passes:
            jersey = p['from_jersey']
            if jersey not in per_player:
                per_player[jersey] = {
                    'total'       : 0,
                    'success'     : 0,
                    'failed'      : 0,
                    'accuracy_pct': 0.0
                }
            per_player[jersey]['total'] += 1
            if p['success']:
                per_player[jersey]['success'] += 1
            else:
                per_player[jersey]['failed'] += 1

        # Hitung akurasi per pemain
        for jersey, stat in per_player.items():
            stat['accuracy_pct'] = round(
                stat['success'] / stat['total'] * 100, 1
            ) if stat['total'] > 0 else 0.0

        return {
            'total_passes'           : total,
            'successful_passes'      : len(sukses),
            'failed_passes'          : len(gagal),
            'accuracy_pct'           : round(len(sukses) / total * 100, 1) if total > 0 else 0.0,
            'avg_distance'           : round(float(np.mean([p['distance'] for p in passes])), 1) if passes else 0.0,
            'avg_distance_successful': round(float(np.mean([p['distance'] for p in sukses])), 1) if sukses else 0.0,
            'per_player'             : per_player
        }