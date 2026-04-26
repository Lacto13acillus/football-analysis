# onetouch_detector.py
# ============================================================
# One-Touch Pass Counting — State Machine:
#   IDLE → POSSESSION → BALL_IN_TRANSIT → SUKSES / GAGAL
#
# Logic:
#   SUKSES = bola dari Player A diterima oleh Player B
#            dengan sentuhan < max_touch_seconds (2 detik)
#   GAGAL  = bola ditahan terlalu lama (> 2 detik)
#          = bola tidak sampai ke pemain lain (timeout)
#          = bola kembali ke pengirim
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
        self.kick_away_distance: float = 100.0

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
            print("[ONETOUCH] WARNING: Kurang dari 2 pemain terdeteksi!")
            if len(most_common) == 1:
                return most_common[0][0], -1
            return -1, -1

        return most_common[0][0], most_common[1][0]

    # ============================================================
    # CORE HELPER: Nearest player (ID-agnostic)
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

    # ============================================================
    # DETEKSI ONE-TOUCH PASS — CORE LOGIC
    # ============================================================

    def detect_onetouch_passes(
        self,
        tracks: Dict,
        debug: bool = True
    ) -> List[Dict]:
        """
        Deteksi one-touch pass dari tracking data.

        VERSI ID-AGNOSTIC: Tidak bergantung pada fixed player ID.

        Logic (State Machine):
            IDLE → POSSESSION → BALL_IN_TRANSIT → SUKSES/GAGAL → IDLE

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
            print(f"\n[ONETOUCH] === ONE-TOUCH PASS DETECTION (ID-AGNOSTIC) ===")
            print(f"[ONETOUCH] Total frames           : {total_frames}")
            print(f"[ONETOUCH] FPS                    : {self.fps}")
            print(f"[ONETOUCH] Possession distance    : {self.ball_possession_distance}px")
            print(f"[ONETOUCH] Kick away distance     : {self.kick_away_distance}px")
            print(f"[ONETOUCH] Receive distance       : {self.receive_distance}px")
            print(f"[ONETOUCH] Max touch seconds      : {self.max_touch_seconds}s "
                  f"({max_touch_frames} frames)")
            print(f"[ONETOUCH] Max transit frames     : {self.max_transit_frames}")
            print(f"[ONETOUCH] Cooldown frames        : {self.cooldown_frames}")
            print(f"[ONETOUCH] ================================\n")

        # ============================================================
        # STATE MACHINE
        # ============================================================

        state = 'idle'
        possessor_id = -1         # ID pemain yang sedang pegang bola
        possessor_position = None # Posisi possessor (untuk identifikasi)
        possession_start_frame = -1
        possession_frames = 0     # Counter untuk validasi awal
        touch_frames = 0          # Counter durasi sentuhan (untuk one-touch check)
        transit_frames = 0
        receive_frames = 0
        kick_frame = -1
        last_event_frame = -999
        away_frames = self.min_away_frames
        closest_to_receiver = float('inf')

        onetouch_events: List[Dict] = []
        event_id_counter = 0

        for frame_num in range(total_frames):
            if frame_num % 200 == 0 and debug:
                print(f"[ONETOUCH] Processing frame {frame_num}/{total_frames}...")

            # --- Ambil posisi bola ---
            ball_pos = self._get_ball_position(tracks, frame_num)
            if ball_pos is None:
                if state == 'ball_in_transit':
                    transit_frames += 1
                    if transit_frames > self.max_transit_frames:
                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: TIMEOUT "
                                  f"— bola tidak terdeteksi selama transit")
                        # GAGAL — timeout
                        event_id_counter += 1
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': possessor_id,
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

            # --- Cari pemain terdekat ke bola (ID-agnostic) ---
            nearest_pid, nearest_dist = self._get_nearest_player_distance(
                ball_pos, tracks, frame_num
            )

            # ======== STATE MACHINE ========

            if state == 'idle':
                # Cooldown
                if (frame_num - last_event_frame) < self.cooldown_frames:
                    continue

                # Away frames — pastikan bola sempat jauh dari semua pemain
                if away_frames < self.min_away_frames:
                    if nearest_dist > self.ball_possession_distance:
                        away_frames += 1
                    continue

                # Cek possession: bola dekat pemain manapun?
                if nearest_dist <= self.ball_possession_distance and nearest_pid != -1:
                    if possessor_id == nearest_pid:
                        possession_frames += 1
                    else:
                        possessor_id = nearest_pid
                        possession_frames = 1
                else:
                    possession_frames = 0
                    possessor_id = -1

                # Possession cukup lama → masuk state POSSESSION
                if possession_frames >= self.min_possession_frames and possessor_id != -1:
                    state = 'possession'
                    possession_start_frame = frame_num - possession_frames + 1
                    touch_frames = possession_frames  # Sudah dihitung dari awal

                    if debug:
                        print(f"[ONETOUCH] Frame {frame_num}: Player {possessor_id} "
                              f"POSSESSION mulai (dist={nearest_dist:.0f}px)")

            elif state == 'possession':
                # ====================================================
                # POSSESSION: bola di kaki pemain, hitung touch duration
                # ====================================================
                dist_possessor = self._get_min_distance_to_player(
                    ball_pos, tracks, frame_num, possessor_id
                )

                # Handle jika possessor ID hilang dari tracking
                if dist_possessor == float('inf'):
                    if nearest_dist <= self.ball_possession_distance:
                        possessor_id = nearest_pid
                        dist_possessor = nearest_dist
                    else:
                        dist_possessor = self.kick_away_distance + 1

                if dist_possessor <= self.ball_possession_distance:
                    # Bola masih di kaki pemain
                    touch_frames += 1

                    # --- CEK: apakah sudah terlalu lama? ---
                    if touch_frames > max_touch_frames:
                        # GAGAL — bukan one-touch, ditahan terlalu lama
                        event_id_counter += 1
                        touch_seconds = round(touch_frames / self.fps, 2)
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': possessor_id,
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
                            'reason': f'Bola ditahan {touch_seconds}s (maks {self.max_touch_seconds}s)',
                        }
                        onetouch_events.append(event)
                        last_event_frame = frame_num
                        away_frames = 0
                        state = 'idle'
                        possession_frames = 0

                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                                  f"— bola ditahan {touch_seconds}s oleh "
                                  f"Player {possessor_id} (maks {self.max_touch_seconds}s)")
                        continue

                    # Debug touch progress
                    if debug and touch_frames % 15 == 0:
                        touch_sec = touch_frames / self.fps
                        print(f"[ONETOUCH]   touch f={frame_num}: "
                              f"P{possessor_id} holding {touch_sec:.1f}s / "
                              f"{self.max_touch_seconds}s")

                elif dist_possessor > self.kick_away_distance:
                    # Bola sudah ditendang! → transit
                    # Simpan posisi possessor saat kick
                    pdata = tracks['players'][frame_num].get(possessor_id)
                    if pdata and 'bbox' in pdata:
                        possessor_position = get_center_of_bbox(pdata['bbox'])
                    else:
                        for back_f in range(frame_num - 1, max(0, frame_num - 10) - 1, -1):
                            pdata = tracks['players'][back_f].get(possessor_id)
                            if pdata and 'bbox' in pdata:
                                possessor_position = get_center_of_bbox(pdata['bbox'])
                                break
                        else:
                            possessor_position = ball_pos

                    state = 'ball_in_transit'
                    kick_frame = frame_num
                    transit_frames = 0
                    receive_frames = 0
                    closest_to_receiver = float('inf')

                    if debug:
                        touch_sec = touch_frames / self.fps
                        print(f"[ONETOUCH] Frame {frame_num}: KICK by "
                              f"Player {possessor_id} "
                              f"(touch={touch_sec:.2f}s, "
                              f"dist={dist_possessor:.0f}px)")

                else:
                    # Bola agak menjauh tapi belum cukup untuk dianggap kick
                    # Cek jika pemain lain lebih dekat
                    if nearest_pid != possessor_id and nearest_dist < dist_possessor:
                        if nearest_dist <= self.ball_possession_distance:
                            # Pemain lain ambil bola tanpa kick yang jelas
                            # Reset ke idle
                            state = 'idle'
                            possession_frames = 0
                            away_frames = self.min_away_frames

            elif state == 'ball_in_transit':
                transit_frames += 1

                # Cari pemain terdekat ke bola YANG BUKAN sender
                recv_pid, dist_receiver = self._get_nearest_player_excluding_position(
                    ball_pos, tracks, frame_num,
                    exclude_position=possessor_position if possessor_position else (0, 0),
                    min_separation=100.0,
                )

                # Cek jarak ke sender juga
                dist_sender = self._get_min_distance_to_player(
                    ball_pos, tracks, frame_num, possessor_id
                )
                if dist_sender == float('inf') and possessor_position:
                    for pid, pdata in tracks['players'][frame_num].items():
                        bbox = pdata.get('bbox')
                        if bbox is None:
                            continue
                        pc = get_center_of_bbox(bbox)
                        if measure_distance(pc, possessor_position) < 100:
                            dist_sender = self._get_min_distance_to_player(
                                ball_pos, tracks, frame_num, pid
                            )
                            break

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
                              f"(receive_frames={receive_frames}/{self.min_receive_frames})")

                    if receive_frames >= self.min_receive_frames:
                        # SUKSES — one-touch pass berhasil!
                        event_id_counter += 1
                        touch_seconds = round(touch_frames / self.fps, 2)
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': possessor_id,
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
                                  f"(P{possessor_id}→P{recv_pid}, "
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
                        'sender_id': possessor_id,
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
                              f"(bola kembali, dist_sender={dist_sender:.0f}px)")
                    continue

                # --- TIMEOUT ---
                if transit_frames > self.max_transit_frames:
                    event_id_counter += 1
                    touch_seconds = round(touch_frames / self.fps, 2)
                    event = {
                        'event_id': event_id_counter,
                        'sender_id': possessor_id,
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
                        'reason': f'Timeout — bola transit {transit_frames}f '
                                  f'({transit_frames/self.fps:.1f}s)',
                    }
                    onetouch_events.append(event)
                    last_event_frame = frame_num
                    away_frames = 0
                    state = 'idle'
                    possession_frames = 0

                    if debug:
                        print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                              f"(TIMEOUT transit {transit_frames}f, "
                              f"closest={closest_to_receiver:.0f}px)")
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
        ID-agnostic: menunjukkan semua player ID di setiap frame.
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

            nearest_pid, nearest_dist = self._get_nearest_player_distance(
                ball_pos, tracks, f
            )

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

    def get_onetouch_statistics(self, events: List[Dict]) -> Dict:
        """Hitung statistik dari event one-touch pass."""
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
