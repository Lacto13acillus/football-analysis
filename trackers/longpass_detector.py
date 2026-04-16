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
        self.kick_away_distance: float = 150.0

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

    def _get_min_distance_to_player(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
        player_id: int,
    ) -> float:
        """
        Hitung jarak MINIMAL bola ke pemain.
        Cek jarak ke KAKI (bottom-center) DAN BADAN (center) bbox,
        ambil yang terkecil.

        Ini penting karena pemain bisa menerima bola dengan:
        - Kaki → bola dekat bottom-center bbox
        - Dada/badan → bola dekat center bbox
        """
        pdata = tracks['players'][frame_num].get(player_id)
        if pdata is None:
            return float('inf')
        bbox = pdata.get('bbox')
        if bbox is None:
            return float('inf')

        foot_pos = get_foot_position(bbox)
        center_pos = get_center_of_bbox(bbox)

        dist_foot = measure_distance(ball_pos, foot_pos)
        dist_center = measure_distance(ball_pos, center_pos)

        return min(dist_foot, dist_center)

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
    # DETEKSI LONG PASS — CORE LOGIC (ID-AGNOSTIC)
    # ============================================================

    def _get_nearest_player_distance(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
    ) -> Tuple[int, float]:
        """
        Cari pemain MANAPUN yang paling dekat ke bola di frame ini.
        Cek jarak ke kaki DAN badan, ambil yang paling kecil.

        Returns:
            (player_id, min_distance)
        """
        best_pid = -1
        best_dist = float('inf')

        for pid, pdata in tracks['players'][frame_num].items():
            bbox = pdata.get('bbox')
            if bbox is None:
                continue
            foot_pos = get_foot_position(bbox)
            center_pos = get_center_of_bbox(bbox)
            dist = min(
                measure_distance(ball_pos, foot_pos),
                measure_distance(ball_pos, center_pos),
            )
            if dist < best_dist:
                best_dist = dist
                best_pid = pid

        return best_pid, best_dist

    def _get_nearest_player_excluding_position(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
        exclude_position: Tuple[int, int],
        min_separation: float = 150.0,
    ) -> Tuple[int, float]:
        """
        Cari pemain terdekat ke bola, tapi BUKAN pemain yang
        berada di dekat exclude_position (= posisi pengirim saat kick).

        Ini untuk mencari penerima tanpa bergantung pada tracking ID.

        Returns:
            (player_id, min_distance_to_ball)
        """
        best_pid = -1
        best_dist = float('inf')

        for pid, pdata in tracks['players'][frame_num].items():
            bbox = pdata.get('bbox')
            if bbox is None:
                continue

            player_center = get_center_of_bbox(bbox)

            # Skip pemain yang terlalu dekat dengan posisi sender saat kick
            if measure_distance(player_center, exclude_position) < min_separation:
                continue

            foot_pos = get_foot_position(bbox)
            dist = min(
                measure_distance(ball_pos, foot_pos),
                measure_distance(ball_pos, player_center),
            )
            if dist < best_dist:
                best_dist = dist
                best_pid = pid

        return best_pid, best_dist

    def detect_longpasses(
        self,
        tracks: Dict,
        debug: bool = True
    ) -> List[Dict]:
        """
        Deteksi long pass dari tracking data.

        VERSI ID-AGNOSTIC: Tidak bergantung pada fixed player ID.
        ByteTrack sering reassign ID, jadi kita cari pemain terdekat
        ke bola di setiap frame, bukan track ID tetap.

        Logic (State Machine):
            IDLE → WAITING_KICK → BALL_IN_AIR → SUKSES/GAGAL → IDLE

        Returns:
            List[Dict] berisi event longpass
        """
        total_frames = len(tracks['players'])
        if total_frames == 0:
            return []

        if debug:
            print(f"\n[LONGPASS] === LONG PASS DETECTION (ID-AGNOSTIC) ===")
            print(f"[LONGPASS] Total frames           : {total_frames}")
            print(f"[LONGPASS] FPS                    : {self.fps}")
            print(f"[LONGPASS] Possession distance    : {self.ball_possession_distance}px")
            print(f"[LONGPASS] Kick away distance     : {self.kick_away_distance}px")
            print(f"[LONGPASS] Receive distance       : {self.receive_distance}px")
            print(f"[LONGPASS] Max flight frames      : {self.max_flight_frames}")
            print(f"[LONGPASS] Cooldown frames        : {self.cooldown_frames}")
            print(f"[LONGPASS] ================================\n")

        # ============================================================
        # STATE MACHINE (ID-AGNOSTIC)
        # ============================================================

        state = 'idle'
        sender_id = -1            # ID pemain pengirim (bisa berubah tiap event)
        sender_position = None    # Posisi pengirim saat kick (untuk identifikasi)
        kick_frame = -1
        flight_frames = 0
        receive_frames = 0
        possession_frames = 0
        current_possessor = -1
        last_event_frame = -999
        away_frames = self.min_away_frames

        longpass_events: List[Dict] = []
        event_id_counter = 0
        closest_to_receiver = float('inf')

        for frame_num in range(total_frames):
            if frame_num % 200 == 0 and debug:
                print(f"[LONGPASS] Processing frame {frame_num}/{total_frames}...")

            # --- Ambil posisi bola ---
            ball_pos = self._get_ball_position(tracks, frame_num)
            if ball_pos is None:
                if state == 'ball_in_air':
                    flight_frames += 1
                    if flight_frames > self.max_flight_frames:
                        if debug:
                            print(f"[LONGPASS] Frame {frame_num}: TIMEOUT "
                                  f"— bola tidak terdeteksi selama flight")
                        state = 'idle'
                        away_frames = 0
                continue

            # --- Cari pemain terdekat ke bola (ID-agnostic) ---
            nearest_pid, nearest_dist = self._get_nearest_player_distance(
                ball_pos, tracks, frame_num
            )

            # ======== STATE MACHINE ========

            if state == 'idle':
                # Cooldown
                if (frame_num - last_event_frame) < self.cooldown_frames:
                    continue

                # Away frames
                if away_frames < self.min_away_frames:
                    if nearest_dist > self.ball_possession_distance:
                        away_frames += 1
                    continue

                # Cek possession: bola dekat pemain manapun?
                if nearest_dist <= self.ball_possession_distance and nearest_pid != -1:
                    if current_possessor == nearest_pid:
                        possession_frames += 1
                    else:
                        # Pemain baru, reset counter
                        current_possessor = nearest_pid
                        possession_frames = 1
                else:
                    possession_frames = 0
                    current_possessor = -1

                # Possession cukup lama → siap menunggu kick
                if possession_frames >= self.min_possession_frames and current_possessor != -1:
                    sender_id = current_possessor
                    state = 'waiting_kick'

                    if debug:
                        print(f"[LONGPASS] Frame {frame_num}: Player {sender_id} "
                              f"memiliki bola (possession {possession_frames}f, "
                              f"dist={nearest_dist:.0f}px)")

            elif state == 'waiting_kick':
                # Cek jarak bola ke sender saat ini
                dist_sender = self._get_min_distance_to_player(
                    ball_pos, tracks, frame_num, sender_id
                )
                # Jika sender ID hilang, coba cari pemain dekat posisi sender
                if dist_sender == float('inf'):
                    # Sender hilang dari tracking, gunakan nearest player
                    # yang dekat dengan bola terakhir (masih holding)
                    if nearest_dist <= self.ball_possession_distance:
                        sender_id = nearest_pid
                        dist_sender = nearest_dist
                    else:
                        # Bola sudah jauh, anggap ditendang
                        dist_sender = self.kick_away_distance + 1

                if dist_sender > self.kick_away_distance:
                    # Bola sudah ditendang!
                    # Simpan posisi sender saat kick untuk identifikasi
                    pdata = tracks['players'][frame_num].get(sender_id)
                    if pdata and 'bbox' in pdata:
                        sender_position = get_center_of_bbox(pdata['bbox'])
                    else:
                        # Cari di frame sebelumnya
                        for back_f in range(frame_num - 1, max(0, frame_num - 10) - 1, -1):
                            pdata = tracks['players'][back_f].get(sender_id)
                            if pdata and 'bbox' in pdata:
                                sender_position = get_center_of_bbox(pdata['bbox'])
                                break
                        else:
                            sender_position = ball_pos  # fallback ke posisi bola

                    state = 'ball_in_air'
                    kick_frame = frame_num
                    flight_frames = 0
                    receive_frames = 0
                    closest_to_receiver = float('inf')

                    if debug:
                        print(f"[LONGPASS] Frame {frame_num}: KICK DETECTED! "
                              f"Player {sender_id} "
                              f"(dist_sender={dist_sender:.0f}px, "
                              f"sender_pos={sender_position})")

                elif dist_sender <= self.ball_possession_distance:
                    # Masih dekat sender, tapi cek jika pemain lain ambil
                    if nearest_pid != sender_id and nearest_dist < dist_sender:
                        if nearest_dist <= self.ball_possession_distance:
                            state = 'idle'
                            possession_frames = 0
                            away_frames = self.min_away_frames

            elif state == 'ball_in_air':
                flight_frames += 1

                # Cari pemain terdekat ke bola YANG BUKAN sender
                # (gunakan posisi sender saat kick untuk membedakan)
                recv_pid, dist_receiver = self._get_nearest_player_excluding_position(
                    ball_pos, tracks, frame_num,
                    exclude_position=sender_position if sender_position else (0, 0),
                    min_separation=100.0,
                )

                # Juga cek jarak ke sender (untuk deteksi bola kembali)
                dist_sender = self._get_min_distance_to_player(
                    ball_pos, tracks, frame_num, sender_id
                )
                # Jika sender ID hilang, cek nearest player dekat sender_position
                if dist_sender == float('inf') and sender_position:
                    for pid, pdata in tracks['players'][frame_num].items():
                        bbox = pdata.get('bbox')
                        if bbox is None:
                            continue
                        pc = get_center_of_bbox(bbox)
                        if measure_distance(pc, sender_position) < 100:
                            dist_sender = self._get_min_distance_to_player(
                                ball_pos, tracks, frame_num, pid
                            )
                            break

                # Update closest
                if dist_receiver < closest_to_receiver:
                    closest_to_receiver = dist_receiver

                # Debug setiap 10 frame
                if debug and flight_frames % 10 == 0:
                    print(f"[LONGPASS]   flight f={frame_num}: "
                          f"recv_pid={recv_pid}, "
                          f"dist_recv={dist_receiver:.0f}px, "
                          f"dist_sender={dist_sender:.0f}px, "
                          f"closest={closest_to_receiver:.0f}px")

                # --- CEK BOLA DITERIMA ---
                if dist_receiver <= self.receive_distance and recv_pid != -1:
                    receive_frames += 1

                    if debug:
                        print(f"[LONGPASS]   f={frame_num}: bola DEKAT penerima "
                              f"P{recv_pid}! dist={dist_receiver:.0f}px "
                              f"(receive_frames={receive_frames}/{self.min_receive_frames})")

                    if receive_frames >= self.min_receive_frames:
                        # LONGPASS SUKSES!
                        event_id_counter += 1
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': sender_id,
                            'receiver_id': recv_pid,
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
                                  f"(P{sender_id}→P{recv_pid}, "
                                  f"flight={flight_frames}f/"
                                  f"{flight_frames/self.fps:.1f}s, "
                                  f"recv_dist={dist_receiver:.0f}px)")
                        continue
                else:
                    receive_frames = 0

                # --- CEK BOLA KEMBALI KE PENGIRIM ---
                if dist_sender <= self.ball_possession_distance and flight_frames > 15:
                    event_id_counter += 1
                    event = {
                        'event_id': event_id_counter,
                        'sender_id': sender_id,
                        'receiver_id': recv_pid if recv_pid != -1 else -1,
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
                              f"(bola kembali, dist_sender={dist_sender:.0f}px)")
                    continue

                # --- TIMEOUT ---
                if flight_frames > self.max_flight_frames:
                    event_id_counter += 1
                    event = {
                        'event_id': event_id_counter,
                        'sender_id': sender_id,
                        'receiver_id': recv_pid if recv_pid != -1 else -1,
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
        Print jarak bola ke SEMUA pemain (kaki & badan) setiap N frame.
        ID-agnostic: menunjukkan semua player ID di setiap frame.
        """
        total_frames = len(tracks['players'])

        print(f"\n[DEBUG] === JARAK BOLA-PEMAIN (setiap {sample_every} frame) ===")
        print(f"{'Frame':<8} {'BallPos':<16} {'NearestPID':<12} {'NearestDist':<14} {'AllPlayers'}")
        print("-" * 90)

        for f in range(0, total_frames, sample_every):
            ball_pos = self._get_ball_position(tracks, f)
            if ball_pos is None:
                continue

            nearest_pid, nearest_dist = self._get_nearest_player_distance(
                ball_pos, tracks, f
            )

            # List semua pemain dan jaraknya
            all_players = []
            for pid, pdata in tracks['players'][f].items():
                bbox = pdata.get('bbox')
                if bbox is None:
                    continue
                d = self._get_min_distance_to_player(ball_pos, tracks, f, pid)
                all_players.append(f"P{pid}:{d:.0f}")

            all_str = ", ".join(all_players) if all_players else "None"

            print(f"{f:<8} "
                  f"({ball_pos[0]:.0f},{ball_pos[1]:.0f}){'':<6} "
                  f"P{nearest_pid:<10} "
                  f"{nearest_dist:<14.1f} "
                  f"{all_str}")

        print("-" * 90)
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
