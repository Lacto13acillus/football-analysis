# longpass_detector.py
# ============================================================
# Long Pass Counting — State Machine:
#   IDLE → KICK_DETECTED → BALL_IN_AIR → RECEIVED / MISSED
#
# Logic:
#   SUKSES = bola long pass dari Player A diterima oleh KAKI Player B
#   GAGAL  = bola tidak diterima (timeout, keluar frame, dll)
#
# Model YOLO: 2 class (ball=0, player=1)
# ============================================================

import sys
sys.path.append('../')

from utils.bbox_utils import measure_distance, get_center_of_bbox, get_foot_position
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class LongPassDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps

        # ============================================================
        # PARAMETER POSSESSION (bola dekat kaki = memiliki bola)
        # ============================================================
        # Jarak maks bola ke kaki pemain untuk dianggap "possession"
        self.ball_possession_distance: float = 150.0

        # ============================================================
        # PARAMETER KICK DETECTION
        # ============================================================
        # Jarak minimal bola dari pengirim agar dianggap "sudah ditendang"
        self.kick_away_distance: float = 200.0

        # Min frames bola harus dekat pemain sebelum dianggap possession valid
        self.min_possession_frames: int = 3

        # ============================================================
        # PARAMETER RECEIVE DETECTION
        # ============================================================
        # Jarak maks bola ke kaki penerima agar dianggap "diterima"
        self.receive_distance: float = 200.0

        # Min frames bola harus dekat kaki penerima untuk konfirmasi
        self.min_receive_frames: int = 2

        # Adaptive scaling: skala receive_distance berdasarkan tinggi bbox pemain
        # Jika pemain bbox lebih besar (dekat kamera), threshold lebih besar.
        # Multiplier: receive_distance * (player_bbox_height / reference_height)
        self.adaptive_receive: bool = True
        self.reference_player_height: float = 200.0  # pixel tinggi bbox referensi

        # ============================================================
        # PARAMETER FLIGHT / TIMEOUT
        # ============================================================
        # Maks frame bola di udara sebelum auto-reset (timeout)
        self.max_flight_frames: int = 180  # ~6 detik @30fps

        # ============================================================
        # PARAMETER COOLDOWN
        # ============================================================
        # Cooldown setelah 1 event longpass, hindari double count
        self.cooldown_frames: int = 30

        # Min frames bola harus jauh dari semua pemain sebelum event baru
        self.min_away_frames: int = 5

    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================

    def _get_ball_position(
        self,
        tracks: Dict,
        frame_num: int
    ) -> Optional[Tuple[int, int]]:
        """Ambil posisi center bola di frame tertentu."""
        ball_data = tracks['ball'][frame_num].get(1)
        if ball_data is None or 'bbox' not in ball_data:
            return None
        return get_center_of_bbox(ball_data['bbox'])

    def _get_player_foot_positions(
        self,
        tracks: Dict,
        frame_num: int
    ) -> Dict[int, Tuple[int, int]]:
        """Ambil posisi kaki (bottom-center) semua pemain di frame tertentu."""
        positions = {}
        for pid, pdata in tracks['players'][frame_num].items():
            bbox = pdata.get('bbox')
            if bbox is None:
                continue
            positions[pid] = get_foot_position(bbox)
        return positions

    def _get_player_bbox_height(
        self,
        tracks: Dict,
        frame_num: int,
        player_id: int
    ) -> float:
        """Ambil tinggi bbox pemain di frame tertentu."""
        pdata = tracks['players'][frame_num].get(player_id)
        if pdata is None:
            return self.reference_player_height
        bbox = pdata.get('bbox')
        if bbox is None:
            return self.reference_player_height
        return max(1.0, bbox[3] - bbox[1])

    def _get_adaptive_receive_distance(
        self,
        tracks: Dict,
        frame_num: int,
        player_id: int
    ) -> float:
        """
        Hitung receive_distance adaptif berdasarkan ukuran bbox pemain.
        Pemain yang bbox-nya lebih besar (lebih dekat kamera) butuh
        threshold lebih besar karena pixel-nya lebih banyak.
        """
        if not self.adaptive_receive:
            return self.receive_distance

        player_height = self._get_player_bbox_height(tracks, frame_num, player_id)
        scale = player_height / self.reference_player_height
        # Clamp scale agar tidak terlalu ekstrem
        scale = max(0.6, min(2.5, scale))
        adaptive_dist = self.receive_distance * scale
        return adaptive_dist

    def _get_player_center_positions(
        self,
        tracks: Dict,
        frame_num: int
    ) -> Dict[int, Tuple[int, int]]:
        """Ambil posisi center semua pemain di frame tertentu."""
        positions = {}
        for pid, pdata in tracks['players'][frame_num].items():
            bbox = pdata.get('bbox')
            if bbox is None:
                continue
            positions[pid] = get_center_of_bbox(bbox)
        return positions

    def _find_closest_player_to_ball(
        self,
        ball_pos: Tuple[int, int],
        player_foot_positions: Dict[int, Tuple[int, int]],
    ) -> Tuple[int, float]:
        """
        Cari pemain terdekat ke bola (berdasarkan jarak kaki).

        Returns:
            (player_id, distance) — (-1, inf) jika tidak ada pemain
        """
        closest_pid = -1
        closest_dist = float('inf')

        for pid, foot_pos in player_foot_positions.items():
            dist = measure_distance(ball_pos, foot_pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_pid = pid

        return closest_pid, closest_dist

    def _identify_two_players(
        self,
        tracks: Dict,
        sample_frames: int = 60
    ) -> Tuple[int, int]:
        """
        Identifikasi 2 pemain utama berdasarkan tracking.
        Ambil 2 player ID yang paling sering muncul.

        Returns:
            (player_a_id, player_b_id) — dua ID pemain
        """
        from collections import Counter
        pid_counter = Counter()

        total = min(sample_frames, len(tracks['players']))
        for f in range(total):
            for pid in tracks['players'][f].keys():
                pid_counter[pid] += 1

        # Ambil 2 pemain yang paling sering muncul
        most_common = pid_counter.most_common(2)
        if len(most_common) < 2:
            print("[LONGPASS] WARNING: Kurang dari 2 pemain terdeteksi!")
            if len(most_common) == 1:
                return most_common[0][0], -1
            return -1, -1

        return most_common[0][0], most_common[1][0]

    # ============================================================
    # DETEKSI LONG PASS — CORE LOGIC
    # ============================================================

    def detect_longpasses(
        self,
        tracks: Dict,
        debug: bool = True
    ) -> List[Dict]:
        """
        Deteksi long pass dari tracking data.

        Logic (State Machine):
            IDLE:
                - Monitor bola. Jika bola dekat kaki salah satu pemain
                  (possession), tunggu sampai bola pergi.

            KICK_DETECTED:
                - Bola mulai menjauh dari pemain pengirim.
                  Masuk state BALL_IN_AIR.

            BALL_IN_AIR:
                - Tracking jarak bola ke kedua pemain.
                - Jika bola mendekati kaki pemain LAIN (bukan pengirim)
                  → cek apakah bola benar-benar diterima.

            RECEIVED (= SUKSES):
                - Bola dekat kaki penerima selama beberapa frame.
                - Longpass berhasil!

            MISSED (= GAGAL):
                - Bola tidak sampai ke kaki penerima:
                  timeout, keluar frame, atau menjauh.

        Returns:
            List[Dict] berisi event longpass
        """
        total_frames = len(tracks['players'])
        if total_frames == 0:
            return []

        # Identifikasi 2 pemain
        player_a, player_b = self._identify_two_players(tracks)

        if debug:
            print(f"\n[LONGPASS] === LONG PASS DETECTION ===")
            print(f"[LONGPASS] Total frames           : {total_frames}")
            print(f"[LONGPASS] FPS                    : {self.fps}")
            print(f"[LONGPASS] Player A (ID)          : {player_a}")
            print(f"[LONGPASS] Player B (ID)          : {player_b}")
            print(f"[LONGPASS] Possession distance    : {self.ball_possession_distance}px")
            print(f"[LONGPASS] Kick away distance     : {self.kick_away_distance}px")
            print(f"[LONGPASS] Receive distance       : {self.receive_distance}px")
            print(f"[LONGPASS] Max flight frames      : {self.max_flight_frames}")
            print(f"[LONGPASS] Cooldown frames        : {self.cooldown_frames}")
            print(f"[LONGPASS] ================================\n")

        if player_a == -1 or player_b == -1:
            print("[LONGPASS] ERROR: Tidak bisa mengidentifikasi 2 pemain!")
            return []

        # ============================================================
        # STATE MACHINE
        # ============================================================

        state = 'idle'
        sender_id = -1
        receiver_id = -1
        kick_frame = -1
        flight_frames = 0
        receive_frames = 0
        possession_frames = 0
        current_possessor = -1
        last_event_frame = -999
        away_frames = self.min_away_frames  # Mulai bisa langsung

        longpass_events: List[Dict] = []
        event_id_counter = 0

        # Track closest distance saat flight
        closest_to_receiver = float('inf')

        for frame_num in range(total_frames):
            if frame_num % 200 == 0 and debug:
                print(f"[LONGPASS] Processing frame {frame_num}/{total_frames}...")

            # --- Ambil data ---
            ball_pos = self._get_ball_position(tracks, frame_num)
            if ball_pos is None:
                # Bola tidak terdeteksi
                if state == 'ball_in_air':
                    flight_frames += 1
                    if flight_frames > self.max_flight_frames:
                        # Timeout
                        if debug:
                            print(f"[LONGPASS] Frame {frame_num}: TIMEOUT "
                                  f"— bola tidak terdeteksi selama flight")
                        state = 'idle'
                        away_frames = 0
                continue

            player_foot_positions = self._get_player_foot_positions(tracks, frame_num)

            # Jarak bola ke kaki masing-masing pemain
            dist_to_a = float('inf')
            dist_to_b = float('inf')

            if player_a in player_foot_positions:
                dist_to_a = measure_distance(ball_pos, player_foot_positions[player_a])
            if player_b in player_foot_positions:
                dist_to_b = measure_distance(ball_pos, player_foot_positions[player_b])

            # ======== STATE MACHINE ========

            if state == 'idle':
                # Cooldown check
                if (frame_num - last_event_frame) < self.cooldown_frames:
                    continue

                # Bola harus sudah jauh cukup lama sebelum event baru
                if away_frames < self.min_away_frames:
                    # Cek apakah bola jauh dari semua pemain
                    min_dist = min(dist_to_a, dist_to_b)
                    if min_dist > self.ball_possession_distance:
                        away_frames += 1
                    continue

                # Cek possession: bola dekat kaki siapa?
                if dist_to_a <= self.ball_possession_distance:
                    possession_frames += 1
                    current_possessor = player_a
                elif dist_to_b <= self.ball_possession_distance:
                    possession_frames += 1
                    current_possessor = player_b
                else:
                    possession_frames = 0
                    current_possessor = -1

                # Jika sudah possession cukup lama, tunggu kick
                if possession_frames >= self.min_possession_frames and current_possessor != -1:
                    sender_id = current_possessor
                    receiver_id = player_b if sender_id == player_a else player_a
                    state = 'waiting_kick'

                    if debug:
                        print(f"[LONGPASS] Frame {frame_num}: Player {sender_id} "
                              f"memiliki bola (possession {possession_frames}f)")

            elif state == 'waiting_kick':
                # Tunggu bola menjauh dari pengirim
                dist_sender = dist_to_a if sender_id == player_a else dist_to_b

                if dist_sender > self.kick_away_distance:
                    # Bola sudah ditendang!
                    state = 'ball_in_air'
                    kick_frame = frame_num
                    flight_frames = 0
                    receive_frames = 0
                    closest_to_receiver = float('inf')

                    if debug:
                        print(f"[LONGPASS] Frame {frame_num}: KICK DETECTED! "
                              f"Player {sender_id} → Player {receiver_id} "
                              f"(dist_sender={dist_sender:.0f}px)")
                elif dist_sender > self.ball_possession_distance:
                    # Bola mulai menjauh tapi belum cukup
                    pass
                else:
                    # Masih dekat pengirim — terus tunggu
                    # Tapi cek apakah pemain berganti
                    dist_other = dist_to_b if sender_id == player_a else dist_to_a
                    if dist_other < dist_sender and dist_other <= self.ball_possession_distance:
                        # Possession pindah tanpa kick yang jelas — reset
                        state = 'idle'
                        possession_frames = 0
                        away_frames = self.min_away_frames

            elif state == 'ball_in_air':
                flight_frames += 1

                # Hitung adaptive receive distance untuk penerima
                current_recv_dist = self._get_adaptive_receive_distance(
                    tracks, frame_num, receiver_id
                )

                # Update closest distance ke penerima
                dist_receiver = dist_to_b if receiver_id == player_b else dist_to_a
                if dist_receiver < closest_to_receiver:
                    closest_to_receiver = dist_receiver

                # Debug: print jarak setiap 10 frame saat flight
                if debug and flight_frames % 10 == 0:
                    print(f"[LONGPASS]   flight f={frame_num}: "
                          f"dist_recv={dist_receiver:.0f}px "
                          f"(threshold={current_recv_dist:.0f}px), "
                          f"closest={closest_to_receiver:.0f}px")

                # --- CEK BOLA DITERIMA ---
                if dist_receiver <= current_recv_dist:
                    receive_frames += 1

                    if debug:
                        print(f"[LONGPASS]   f={frame_num}: bola DEKAT penerima! "
                              f"dist={dist_receiver:.0f}px <= {current_recv_dist:.0f}px "
                              f"(receive_frames={receive_frames}/{self.min_receive_frames})")

                    if receive_frames >= self.min_receive_frames:
                        # LONGPASS SUKSES!
                        event_id_counter += 1
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': sender_id,
                            'receiver_id': receiver_id,
                            'frame_kick': kick_frame,
                            'frame_receive': frame_num,
                            'frame_start': kick_frame,
                            'frame_end': frame_num,
                            'success': True,
                            'flight_frames': flight_frames,
                            'flight_seconds': round(flight_frames / self.fps, 2),
                            'closest_distance': round(closest_to_receiver, 1),
                            'receive_distance': round(dist_receiver, 1),
                        }
                        longpass_events.append(event)
                        last_event_frame = frame_num
                        away_frames = 0
                        state = 'idle'
                        possession_frames = 0

                        if debug:
                            print(f"[LONGPASS] Frame {frame_num}: LONGPASS SUKSES ✓ "
                                  f"(P{sender_id}→P{receiver_id}, "
                                  f"flight={flight_frames}f/{flight_frames/self.fps:.1f}s, "
                                  f"recv_dist={dist_receiver:.0f}px, "
                                  f"threshold={current_recv_dist:.0f}px)")
                        continue
                else:
                    receive_frames = 0

                # --- CEK BOLA KEMBALI KE PENGIRIM ---
                # Gunakan ball_possession_distance (bukan receive_distance)
                # agar tidak terlalu agresif menandai GAGAL
                dist_sender = dist_to_a if sender_id == player_a else dist_to_b
                if dist_sender <= self.ball_possession_distance and flight_frames > 15:
                    # Bola benar-benar kembali dekat kaki pengirim — GAGAL
                    event_id_counter += 1
                    event = {
                        'event_id': event_id_counter,
                        'sender_id': sender_id,
                        'receiver_id': receiver_id,
                        'frame_kick': kick_frame,
                        'frame_receive': frame_num,
                        'frame_start': kick_frame,
                        'frame_end': frame_num,
                        'success': False,
                        'flight_frames': flight_frames,
                        'flight_seconds': round(flight_frames / self.fps, 2),
                        'closest_distance': round(closest_to_receiver, 1),
                        'receive_distance': round(dist_receiver, 1),
                        'reason': 'Bola kembali ke pengirim',
                    }
                    longpass_events.append(event)
                    last_event_frame = frame_num
                    away_frames = 0
                    state = 'idle'
                    possession_frames = 0

                    if debug:
                        print(f"[LONGPASS] Frame {frame_num}: LONGPASS GAGAL ✗ "
                              f"(bola kembali ke pengirim P{sender_id}, "
                              f"dist_sender={dist_sender:.0f}px, "
                              f"closest_to_recv={closest_to_receiver:.0f}px)")
                    continue

                # --- TIMEOUT ---
                if flight_frames > self.max_flight_frames:
                    # Timeout — bola di udara terlalu lama
                    event_id_counter += 1
                    event = {
                        'event_id': event_id_counter,
                        'sender_id': sender_id,
                        'receiver_id': receiver_id,
                        'frame_kick': kick_frame,
                        'frame_receive': frame_num,
                        'frame_start': kick_frame,
                        'frame_end': frame_num,
                        'success': False,
                        'flight_frames': flight_frames,
                        'flight_seconds': round(flight_frames / self.fps, 2),
                        'closest_distance': round(closest_to_receiver, 1),
                        'receive_distance': round(dist_receiver, 1),
                        'reason': 'Timeout — bola terlalu lama di udara',
                    }
                    longpass_events.append(event)
                    last_event_frame = frame_num
                    away_frames = 0
                    state = 'idle'
                    possession_frames = 0

                    if debug:
                        print(f"[LONGPASS] Frame {frame_num}: LONGPASS GAGAL ✗ "
                              f"(TIMEOUT {flight_frames}f, "
                              f"closest={closest_to_receiver:.0f}px)")
                    continue

        # ============================================================
        # HASIL
        # ============================================================
        if debug:
            sukses = sum(1 for e in longpass_events if e['success'])
            gagal = sum(1 for e in longpass_events if not e['success'])
            total = len(longpass_events)
            print(f"\n[LONGPASS] === HASIL AKHIR ===")
            print(f"[LONGPASS] Total longpass  : {total}")
            print(f"[LONGPASS] SUKSES          : {sukses}")
            print(f"[LONGPASS] GAGAL           : {gagal}")
            if total > 0:
                print(f"[LONGPASS] Akurasi         : "
                      f"{sukses/total*100:.1f}%")
            print(f"[LONGPASS] =====================\n")

        return longpass_events

    # ============================================================
    # DEBUG: Print jarak bola-kaki per frame
    # ============================================================

    def debug_distances(
        self,
        tracks: Dict,
        sample_every: int = 10,
    ) -> None:
        """
        Print jarak bola ke kaki semua pemain setiap N frame.
        Berguna untuk tuning parameter.
        """
        total_frames = len(tracks['players'])
        player_a, player_b = self._identify_two_players(tracks)

        print(f"\n[DEBUG] === JARAK BOLA-KAKI (setiap {sample_every} frame) ===")
        print(f"[DEBUG] Player A: {player_a}, Player B: {player_b}")
        print(f"{'Frame':<8} {'BallPos':<20} {'DistA':<12} {'DistB':<12} {'Possessor':<12}")
        print("-" * 70)

        for f in range(0, total_frames, sample_every):
            ball_pos = self._get_ball_position(tracks, f)
            if ball_pos is None:
                continue

            player_feet = self._get_player_foot_positions(tracks, f)

            dist_a = float('inf')
            dist_b = float('inf')

            if player_a in player_feet:
                dist_a = measure_distance(ball_pos, player_feet[player_a])
            if player_b in player_feet:
                dist_b = measure_distance(ball_pos, player_feet[player_b])

            possessor = "None"
            if dist_a <= self.ball_possession_distance:
                possessor = f"P{player_a}"
            elif dist_b <= self.ball_possession_distance:
                possessor = f"P{player_b}"

            print(f"{f:<8} "
                  f"({ball_pos[0]:.0f},{ball_pos[1]:.0f}){'':<8} "
                  f"{dist_a:<12.1f} "
                  f"{dist_b:<12.1f} "
                  f"{possessor:<12}")

        print("-" * 70)
        print()

    # ============================================================
    # STATISTIK
    # ============================================================

    def get_longpass_statistics(self, events: List[Dict]) -> Dict:
        """Hitung statistik dari event longpass."""
        total = len(events)
        sukses = [e for e in events if e['success']]
        gagal = [e for e in events if not e['success']]

        # Per-player stats (sender)
        player_stats: Dict[int, Dict] = {}
        for e in events:
            sid = e['sender_id']
            if sid not in player_stats:
                player_stats[sid] = {'total': 0, 'sukses': 0, 'gagal': 0}
            player_stats[sid]['total'] += 1
            if e['success']:
                player_stats[sid]['sukses'] += 1
            else:
                player_stats[sid]['gagal'] += 1

        # Average flight time (sukses)
        avg_flight_success = (
            round(float(np.mean([e['flight_seconds'] for e in sukses])), 2)
            if sukses else 0.0
        )

        # Average closest distance
        avg_closest = (
            round(float(np.mean([e['closest_distance'] for e in events])), 1)
            if events else 0.0
        )

        return {
            'total_longpass': total,
            'successful_longpass': len(sukses),
            'failed_longpass': len(gagal),
            'accuracy_pct': round(
                len(sukses) / total * 100, 1
            ) if total > 0 else 0.0,
            'avg_flight_time_success': avg_flight_success,
            'avg_closest_distance': avg_closest,
            'player_stats': player_stats,
        }
