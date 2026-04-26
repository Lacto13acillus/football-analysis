# onetouch_detector.py
# ============================================================
# One-Touch Pass Counting — State Machine (POSITION-BASED):
#   IDLE → BALL_AT_PLAYER → BALL_TRAVELING → SUKSES / GAGAL
#
# Logic:
#   SUKSES = bola dari pemain terdekat diterima oleh pemain lain
#            dengan sentuhan < max_touch_seconds (2 detik)
#   GAGAL  = bola ditahan terlalu lama (> 2 detik)
#          = bola tidak sampai ke pemain lain (timeout)
#          = bola kembali ke pengirim
#
# POSITION-BASED: Tidak bergantung pada tracking ID.
#   ByteTrack sering reassign ID, jadi kita pakai posisi pemain
#   terdekat ke bola. Pemain "terdekat" di setiap frame dianggap
#   possessor, tanpa peduli ID-nya berubah.
#
# Model YOLO: 3 class (ball=0, cone=1, player=2)
# ============================================================

import sys
sys.path.append('../')

from utils.bbox_utils import measure_distance, get_center_of_bbox, get_foot_position
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class OneTouchDetector:
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
        # Jarak minimal bola dari pemain agar dianggap "sudah dioper"
        self.kick_away_distance: float = 150.0

        # Min frames bola harus dekat pemain sebelum dianggap possession valid
        self.min_possession_frames: int = 2

        # ============================================================
        # PARAMETER ONE-TOUCH (DURASI SENTUHAN)
        # ============================================================
        # Maks durasi sentuhan (detik). Jika lebih → GAGAL (bukan one-touch)
        self.max_touch_seconds: float = 2.0

        # ============================================================
        # PARAMETER RECEIVE DETECTION
        # ============================================================
        # Jarak maks bola ke kaki/badan penerima agar dianggap "diterima"
        self.receive_distance: float = 200.0

        # Min frames bola harus dekat penerima untuk konfirmasi
        self.min_receive_frames: int = 2

        # ============================================================
        # PARAMETER TRANSIT / TIMEOUT
        # ============================================================
        # Maks frame bola di transit sebelum auto-reset (timeout)
        self.max_transit_frames: int = 90  # ~3 detik @30fps

        # ============================================================
        # PARAMETER COOLDOWN
        # ============================================================
        # Cooldown setelah 1 event, hindari double count
        self.cooldown_frames: int = 15  # Lebih pendek karena one-touch cepat

        # Min frames bola harus jauh dari semua pemain sebelum event baru
        self.min_away_frames: int = 3

        # ============================================================
        # PARAMETER POSITION-BASED PLAYER SEPARATION
        # ============================================================
        # Jarak minimal antara 2 pemain agar dianggap "berbeda"
        # Digunakan untuk membedakan sender vs receiver berdasarkan posisi
        self.player_separation_distance: float = 150.0

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

    def _get_nearest_player_distance(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
    ) -> Tuple[int, float, Optional[Tuple[int, int]]]:
        """
        Cari pemain MANAPUN yang paling dekat ke bola di frame ini.
        Cek jarak ke kaki DAN badan, ambil yang paling kecil.

        Returns:
            (player_id, min_distance, player_center_position)
        """
        best_pid = -1
        best_dist = float('inf')
        best_center = None

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
                best_center = center_pos

        return best_pid, best_dist, best_center

    def _get_nearest_player_excluding_area(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
        exclude_center: Tuple[int, int],
        min_separation: float = 150.0,
    ) -> Tuple[int, float, Optional[Tuple[int, int]]]:
        """
        Cari pemain terdekat ke bola, tapi BUKAN pemain yang
        berada di dekat exclude_center (= area posisi pengirim).

        POSITION-BASED: membedakan pemain berdasarkan posisi, bukan ID.

        Returns:
            (player_id, min_distance_to_ball, player_center)
        """
        best_pid = -1
        best_dist = float('inf')
        best_center = None

        for pid, pdata in tracks['players'][frame_num].items():
            bbox = pdata.get('bbox')
            if bbox is None:
                continue

            player_center = get_center_of_bbox(bbox)

            # Skip pemain yang terlalu dekat dengan area sender
            if measure_distance(player_center, exclude_center) < min_separation:
                continue

            foot_pos = get_foot_position(bbox)
            dist = min(
                measure_distance(ball_pos, foot_pos),
                measure_distance(ball_pos, player_center),
            )
            if dist < best_dist:
                best_dist = dist
                best_pid = pid
                best_center = player_center

        return best_pid, best_dist, best_center

    def _get_min_distance_to_nearest_at_position(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
        target_center: Tuple[int, int],
        max_match_distance: float = 200.0,
    ) -> float:
        """
        Hitung jarak bola ke pemain yang berada di sekitar target_center.
        POSITION-BASED: cari pemain manapun (ID apapun) yang dekat
        dengan target_center, lalu hitung jarak bola ke pemain itu.

        Returns:
            jarak bola ke pemain terdekat di area target_center.
            float('inf') jika tidak ada pemain di area itu.
        """
        best_dist = float('inf')

        for pid, pdata in tracks['players'][frame_num].items():
            bbox = pdata.get('bbox')
            if bbox is None:
                continue

            player_center = get_center_of_bbox(bbox)

            # Hanya cek pemain yang dekat dengan target_center
            if measure_distance(player_center, target_center) > max_match_distance:
                continue

            foot_pos = get_foot_position(bbox)
            dist = min(
                measure_distance(ball_pos, foot_pos),
                measure_distance(ball_pos, player_center),
            )
            if dist < best_dist:
                best_dist = dist

        return best_dist

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

        most_common = pid_counter.most_common(2)
        if len(most_common) < 2:
            print("[ONETOUCH] WARNING: Kurang dari 2 pemain terdeteksi!")
            if len(most_common) == 1:
                return most_common[0][0], -1
            return -1, -1

        return most_common[0][0], most_common[1][0]

    # ============================================================
    # DETEKSI ONE-TOUCH PASS — CORE LOGIC (POSITION-BASED)
    # ============================================================

    def detect_onetouch_passes(
        self,
        tracks: Dict,
        debug: bool = True
    ) -> List[Dict]:
        """
        Deteksi one-touch pass dari tracking data.

        POSITION-BASED: Tidak bergantung pada tracking ID yang fixed.
        Menggunakan posisi pemain terdekat ke bola untuk menentukan
        possessor. Pemain dibedakan berdasarkan POSISI, bukan ID.

        Logic (State Machine):
            IDLE → BALL_AT_PLAYER → BALL_TRAVELING → SUKSES/GAGAL → IDLE

        Kunci one-touch:
            - Jika bola di kaki pemain > max_touch_seconds → GAGAL
            - Jika bola tidak sampai ke pemain lain → GAGAL
            - Jika bola sampai ke pemain lain dengan sentuhan cepat → SUKSES

        Returns:
            List[Dict] berisi event one-touch pass
        """
        total_frames = len(tracks['players'])
        if total_frames == 0:
            return []

        max_touch_frames = int(self.max_touch_seconds * self.fps)

        if debug:
            print(f"\n[ONETOUCH] === ONE-TOUCH PASS DETECTION (POSITION-BASED) ===")
            print(f"[ONETOUCH] Total frames           : {total_frames}")
            print(f"[ONETOUCH] FPS                    : {self.fps}")
            print(f"[ONETOUCH] Possession distance    : {self.ball_possession_distance}px")
            print(f"[ONETOUCH] Kick away distance     : {self.kick_away_distance}px")
            print(f"[ONETOUCH] Receive distance       : {self.receive_distance}px")
            print(f"[ONETOUCH] Max touch seconds      : {self.max_touch_seconds}s "
                  f"({max_touch_frames} frames)")
            print(f"[ONETOUCH] Max transit frames     : {self.max_transit_frames}")
            print(f"[ONETOUCH] Cooldown frames        : {self.cooldown_frames}")
            print(f"[ONETOUCH] Player separation      : {self.player_separation_distance}px")
            print(f"[ONETOUCH] ================================\n")

        # ============================================================
        # STATE MACHINE (POSITION-BASED)
        # ============================================================

        state = 'idle'
        sender_position: Optional[Tuple[int, int]] = None  # Posisi pemain pengirim
        sender_pid = -1            # ID pemain terakhir (untuk logging saja)
        possession_start_frame = -1
        possession_frames = 0
        touch_frames = 0
        transit_frames = 0
        receive_frames = 0
        kick_frame = -1
        last_event_frame = -999
        away_frames = self.min_away_frames
        closest_to_receiver = float('inf')
        last_nearest_center: Optional[Tuple[int, int]] = None

        onetouch_events: List[Dict] = []
        event_id_counter = 0

        for frame_num in range(total_frames):
            if frame_num % 200 == 0 and debug:
                print(f"[ONETOUCH] Processing frame {frame_num}/{total_frames}...")

            # --- Ambil posisi bola ---
            ball_pos = self._get_ball_position(tracks, frame_num)
            if ball_pos is None:
                if state == 'ball_traveling':
                    transit_frames += 1
                    if transit_frames > self.max_transit_frames:
                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: TIMEOUT "
                                  f"— bola hilang dari deteksi")
                        event_id_counter += 1
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': sender_pid,
                            'receiver_id': -1,
                            'frame_kick': kick_frame,
                            'frame_start': possession_start_frame,
                            'frame_end': frame_num,
                            'success': False,
                            'touch_frames': touch_frames,
                            'touch_seconds': round(touch_frames / self.fps, 2),
                            'transit_frames': transit_frames,
                            'flight_seconds': round(transit_frames / self.fps, 2),
                            'closest_distance': round(closest_to_receiver, 1),
                            'reason': 'Timeout — bola hilang dari deteksi',
                        }
                        onetouch_events.append(event)
                        last_event_frame = frame_num
                        away_frames = 0
                        state = 'idle'
                        possession_frames = 0
                continue

            # --- Cari pemain terdekat ke bola (POSITION-BASED) ---
            nearest_pid, nearest_dist, nearest_center = \
                self._get_nearest_player_distance(ball_pos, tracks, frame_num)

            # ======== STATE MACHINE ========

            if state == 'idle':
                # Cooldown
                if (frame_num - last_event_frame) < self.cooldown_frames:
                    continue

                # Away frames — pastikan bola sempat jauh dari semua pemain
                if away_frames < self.min_away_frames:
                    if nearest_dist > self.ball_possession_distance:
                        away_frames += 1
                    else:
                        away_frames = 0
                    continue

                # Cek possession: bola dekat pemain manapun?
                if nearest_dist <= self.ball_possession_distance and nearest_pid != -1:
                    # Cek apakah ini pemain yang sama (berdasarkan posisi)
                    if last_nearest_center is not None and nearest_center is not None:
                        pos_dist = measure_distance(nearest_center, last_nearest_center)
                        if pos_dist < self.player_separation_distance:
                            # Pemain yang sama (posisi mirip)
                            possession_frames += 1
                        else:
                            # Pemain berbeda, reset
                            possession_frames = 1
                            sender_position = nearest_center
                            sender_pid = nearest_pid
                    else:
                        possession_frames = 1
                        sender_position = nearest_center
                        sender_pid = nearest_pid

                    last_nearest_center = nearest_center
                else:
                    possession_frames = 0
                    last_nearest_center = None

                # Possession cukup lama → masuk state BALL_AT_PLAYER
                if possession_frames >= self.min_possession_frames:
                    state = 'ball_at_player'
                    possession_start_frame = frame_num - possession_frames + 1
                    touch_frames = possession_frames
                    sender_position = nearest_center
                    sender_pid = nearest_pid

                    if debug:
                        print(f"[ONETOUCH] Frame {frame_num}: "
                              f"BALL_AT_PLAYER (P{sender_pid}, "
                              f"dist={nearest_dist:.0f}px, "
                              f"pos=({sender_position[0]:.0f},{sender_position[1]:.0f}))")

            elif state == 'ball_at_player':
                # ====================================================
                # BALL_AT_PLAYER: bola di kaki pemain, hitung touch
                # ====================================================

                # Cek jarak bola ke pemain di area sender (position-based)
                dist_at_sender = self._get_min_distance_to_nearest_at_position(
                    ball_pos, tracks, frame_num,
                    target_center=sender_position,
                    max_match_distance=250.0,
                )

                # Juga update sender_position jika ada pemain di area itu
                for pid, pdata in tracks['players'][frame_num].items():
                    bbox = pdata.get('bbox')
                    if bbox is None:
                        continue
                    pc = get_center_of_bbox(bbox)
                    if measure_distance(pc, sender_position) < 250:
                        sender_position = pc  # Update posisi sender
                        sender_pid = pid
                        break

                if dist_at_sender <= self.ball_possession_distance:
                    # Bola masih di kaki pemain
                    touch_frames += 1

                    # --- CEK: apakah sudah terlalu lama? ---
                    if touch_frames > max_touch_frames:
                        event_id_counter += 1
                        touch_seconds = round(touch_frames / self.fps, 2)
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': sender_pid,
                            'receiver_id': -1,
                            'frame_kick': frame_num,
                            'frame_start': possession_start_frame,
                            'frame_end': frame_num,
                            'success': False,
                            'touch_frames': touch_frames,
                            'touch_seconds': touch_seconds,
                            'transit_frames': 0,
                            'flight_seconds': 0.0,
                            'closest_distance': 0.0,
                            'reason': f'Bola ditahan {touch_seconds}s '
                                      f'(maks {self.max_touch_seconds}s)',
                        }
                        onetouch_events.append(event)
                        last_event_frame = frame_num
                        away_frames = 0
                        state = 'idle'
                        possession_frames = 0

                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                                  f"— held {touch_seconds}s by P{sender_pid} "
                                  f"(maks {self.max_touch_seconds}s)")
                        continue

                    # Debug touch progress
                    if debug and touch_frames % 15 == 0:
                        touch_sec = touch_frames / self.fps
                        print(f"[ONETOUCH]   touch f={frame_num}: "
                              f"P{sender_pid} holding {touch_sec:.1f}s / "
                              f"{self.max_touch_seconds}s")

                elif dist_at_sender > self.kick_away_distance:
                    # Bola sudah ditendang! → traveling
                    state = 'ball_traveling'
                    kick_frame = frame_num
                    transit_frames = 0
                    receive_frames = 0
                    closest_to_receiver = float('inf')

                    if debug:
                        touch_sec = touch_frames / self.fps
                        print(f"[ONETOUCH] Frame {frame_num}: KICK by "
                              f"P{sender_pid} "
                              f"(touch={touch_sec:.2f}s, "
                              f"dist={dist_at_sender:.0f}px)")

            elif state == 'ball_traveling':
                transit_frames += 1

                # Cari pemain terdekat ke bola YANG BUKAN di area sender
                recv_pid, dist_receiver, recv_center = \
                    self._get_nearest_player_excluding_area(
                        ball_pos, tracks, frame_num,
                        exclude_center=sender_position,
                        min_separation=self.player_separation_distance,
                    )

                # Cek jarak bola ke area sender (position-based)
                dist_sender = self._get_min_distance_to_nearest_at_position(
                    ball_pos, tracks, frame_num,
                    target_center=sender_position,
                    max_match_distance=250.0,
                )

                # Update closest
                if dist_receiver < closest_to_receiver:
                    closest_to_receiver = dist_receiver

                # Debug setiap 10 frame
                if debug and transit_frames % 10 == 0:
                    print(f"[ONETOUCH]   transit f={frame_num}: "
                          f"recv_pid={recv_pid}, "
                          f"dist_recv={dist_receiver:.0f}px, "
                          f"dist_sender={dist_sender:.0f}px")

                # --- CEK BOLA DITERIMA ---
                if dist_receiver <= self.receive_distance and recv_pid != -1:
                    receive_frames += 1

                    if debug:
                        print(f"[ONETOUCH]   f={frame_num}: bola DEKAT penerima "
                              f"P{recv_pid}! dist={dist_receiver:.0f}px "
                              f"(receive_frames={receive_frames}/"
                              f"{self.min_receive_frames})")

                    if receive_frames >= self.min_receive_frames:
                        # SUKSES!
                        event_id_counter += 1
                        touch_seconds = round(touch_frames / self.fps, 2)
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': sender_pid,
                            'receiver_id': recv_pid,
                            'frame_kick': kick_frame,
                            'frame_start': possession_start_frame,
                            'frame_end': frame_num,
                            'success': True,
                            'touch_frames': touch_frames,
                            'touch_seconds': touch_seconds,
                            'transit_frames': transit_frames,
                            'flight_seconds': round(transit_frames / self.fps, 2),
                            'closest_distance': round(closest_to_receiver, 1),
                            'receive_distance': round(dist_receiver, 1),
                        }
                        onetouch_events.append(event)
                        last_event_frame = frame_num
                        away_frames = 0
                        state = 'idle'
                        possession_frames = 0

                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: SUKSES ✓ "
                                  f"(P{sender_pid}→P{recv_pid}, "
                                  f"touch={touch_seconds}s, "
                                  f"transit={transit_frames}f/"
                                  f"{transit_frames/self.fps:.1f}s)")
                        continue
                else:
                    receive_frames = 0

                # --- CEK BOLA KEMBALI KE PENGIRIM ---
                if dist_sender <= self.ball_possession_distance and transit_frames > 10:
                    event_id_counter += 1
                    touch_seconds = round(touch_frames / self.fps, 2)
                    event = {
                        'event_id': event_id_counter,
                        'sender_id': sender_pid,
                        'receiver_id': -1,
                        'frame_kick': kick_frame,
                        'frame_start': possession_start_frame,
                        'frame_end': frame_num,
                        'success': False,
                        'touch_frames': touch_frames,
                        'touch_seconds': touch_seconds,
                        'transit_frames': transit_frames,
                        'flight_seconds': round(transit_frames / self.fps, 2),
                        'closest_distance': round(closest_to_receiver, 1),
                        'reason': 'Bola kembali ke pengirim',
                    }
                    onetouch_events.append(event)
                    last_event_frame = frame_num
                    away_frames = 0
                    state = 'idle'
                    possession_frames = 0

                    if debug:
                        print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                              f"(bola kembali ke sender, "
                              f"dist={dist_sender:.0f}px)")
                    continue

                # --- TIMEOUT ---
                if transit_frames > self.max_transit_frames:
                    event_id_counter += 1
                    touch_seconds = round(touch_frames / self.fps, 2)
                    event = {
                        'event_id': event_id_counter,
                        'sender_id': sender_pid,
                        'receiver_id': recv_pid if recv_pid != -1 else -1,
                        'frame_kick': kick_frame,
                        'frame_start': possession_start_frame,
                        'frame_end': frame_num,
                        'success': False,
                        'touch_frames': touch_frames,
                        'touch_seconds': touch_seconds,
                        'transit_frames': transit_frames,
                        'flight_seconds': round(transit_frames / self.fps, 2),
                        'closest_distance': round(closest_to_receiver, 1),
                        'reason': f'Timeout transit {transit_frames}f '
                                  f'({transit_frames/self.fps:.1f}s)',
                    }
                    onetouch_events.append(event)
                    last_event_frame = frame_num
                    away_frames = 0
                    state = 'idle'
                    possession_frames = 0

                    if debug:
                        print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                              f"(TIMEOUT transit {transit_frames}f)")
                    continue

        # ============================================================
        # HASIL
        # ============================================================
        if debug:
            sukses = sum(1 for e in onetouch_events if e['success'])
            gagal = sum(1 for e in onetouch_events if not e['success'])
            total = len(onetouch_events)
            print(f"\n[ONETOUCH] === HASIL AKHIR ===")
            print(f"[ONETOUCH] Total one-touch  : {total}")
            print(f"[ONETOUCH] SUKSES           : {sukses}")
            print(f"[ONETOUCH] GAGAL            : {gagal}")
            if total > 0:
                print(f"[ONETOUCH] Akurasi          : "
                      f"{sukses/total*100:.1f}%")
                avg_touch = np.mean([e['touch_seconds'] for e in onetouch_events
                                     if e['success']]) if sukses > 0 else 0
                print(f"[ONETOUCH] Avg touch (sukses): {avg_touch:.2f}s")
            print(f"[ONETOUCH] =====================\n")

        return onetouch_events

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
        """
        total_frames = len(tracks['players'])

        print(f"\n[DEBUG] === JARAK BOLA-PEMAIN (setiap {sample_every} frame) ===")
        print(f"{'Frame':<8} {'BallPos':<16} {'NearestPID':<12} "
              f"{'NearestDist':<14} {'AllPlayers'}")
        print("-" * 90)

        for f in range(0, total_frames, sample_every):
            ball_pos = self._get_ball_position(tracks, f)
            if ball_pos is None:
                continue

            nearest_pid, nearest_dist, _ = self._get_nearest_player_distance(
                ball_pos, tracks, f
            )

            all_players = []
            for pid, pdata in tracks['players'][f].items():
                bbox = pdata.get('bbox')
                if bbox is None:
                    continue
                foot_pos = get_foot_position(bbox)
                center_pos = get_center_of_bbox(bbox)
                d = min(
                    measure_distance(ball_pos, foot_pos),
                    measure_distance(ball_pos, center_pos),
                )
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

    def get_onetouch_statistics(self, events: List[Dict]) -> Dict:
        """Hitung statistik dari event one-touch pass."""
        total = len(events)
        sukses = [e for e in events if e['success']]
        gagal = [e for e in events if not e['success']]

        # Per-player stats (sender) — untuk kompatibilitas
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

        # Gagal per alasan
        gagal_held = sum(1 for e in gagal if 'ditahan' in e.get('reason', '').lower())
        gagal_timeout = sum(1 for e in gagal if 'timeout' in e.get('reason', '').lower())
        gagal_return = sum(1 for e in gagal if 'kembali' in e.get('reason', '').lower())
        gagal_other = len(gagal) - gagal_held - gagal_timeout - gagal_return

        # Average touch time (sukses)
        avg_touch_success = (
            round(float(np.mean([e['touch_seconds'] for e in sukses])), 2)
            if sukses else 0.0
        )

        # Average transit time (sukses)
        avg_transit_success = (
            round(float(np.mean([e['flight_seconds'] for e in sukses])), 2)
            if sukses else 0.0
        )

        # Average closest distance
        avg_closest = (
            round(float(np.mean([e['closest_distance'] for e in events])), 1)
            if events else 0.0
        )

        return {
            'total_onetouch': total,
            'successful_onetouch': len(sukses),
            'failed_onetouch': len(gagal),
            'accuracy_pct': round(
                len(sukses) / total * 100, 1
            ) if total > 0 else 0.0,
            'avg_touch_time_success': avg_touch_success,
            'avg_transit_time_success': avg_transit_success,
            'avg_closest_distance': avg_closest,
            'gagal_held_too_long': gagal_held,
            'gagal_timeout': gagal_timeout,
            'gagal_ball_return': gagal_return,
            'gagal_other': gagal_other,
            'player_stats': player_stats,
        }
