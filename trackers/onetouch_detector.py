# onetouch_detector.py
# ============================================================
# One-Touch Pass Counting — ALTERNATING State Machine:
#
# Hanya menghitung pass dari TARGET player (pemain pertama
# yang pegang bola). Pass dari partner TIDAK dihitung.
#
# Alur:
#   1. TARGET pegang bola → hitung touch duration
#   2. TARGET tendang → bola traveling
#   3. Bola sampai ke player lain → SUKSES (jika touch < 2s)
#   4. Tunggu bola kembali ke player lain (partner return)
#   5. Tunggu bola sampai ke target lagi → kembali ke step 1
#
# POSITION-BASED: Mengidentifikasi "player lain" berdasarkan
#   jarak posisi ke posisi possessor terakhir (bukan tracking ID).
#   Ini mengatasi masalah ByteTrack yang reassign ID.
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
        # PARAMETER POSSESSION
        # ============================================================
        self.ball_possession_distance: float = 150.0

        # ============================================================
        # PARAMETER KICK DETECTION
        # ============================================================
        self.kick_away_distance: float = 150.0
        self.min_possession_frames: int = 2

        # ============================================================
        # PARAMETER ONE-TOUCH (DURASI SENTUHAN)
        # ============================================================
        self.max_touch_seconds: float = 2.0

        # ============================================================
        # PARAMETER RECEIVE DETECTION
        # ============================================================
        self.receive_distance: float = 200.0
        self.min_receive_frames: int = 2

        # ============================================================
        # PARAMETER TRANSIT / TIMEOUT
        # ============================================================
        self.max_transit_frames: int = 90

        # ============================================================
        # PARAMETER COOLDOWN
        # ============================================================
        self.cooldown_frames: int = 10
        self.min_away_frames: int = 2

        # ============================================================
        # PARAMETER PLAYER SEPARATION
        # ============================================================
        # Jarak minimal posisi 2 pemain agar dianggap "beda pemain"
        self.player_separation_distance: float = 120.0

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

    def _get_nearest_player(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
    ) -> Tuple[int, float, Optional[Tuple[int, int]]]:
        """
        Cari pemain terdekat ke bola.
        Returns: (player_id, min_distance, player_center)
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

    def _get_nearest_player_far_from(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
        exclude_pos: Tuple[int, int],
        min_separation: float = 120.0,
    ) -> Tuple[int, float, Optional[Tuple[int, int]]]:
        """
        Cari pemain terdekat ke bola yang posisinya JAUH dari exclude_pos.
        Untuk mencari "pemain lain" (bukan possessor).
        """
        best_pid = -1
        best_dist = float('inf')
        best_center = None

        for pid, pdata in tracks['players'][frame_num].items():
            bbox = pdata.get('bbox')
            if bbox is None:
                continue

            center = get_center_of_bbox(bbox)

            # Skip pemain yang dekat dengan exclude_pos
            if measure_distance(center, exclude_pos) < min_separation:
                continue

            foot_pos = get_foot_position(bbox)
            dist = min(
                measure_distance(ball_pos, foot_pos),
                measure_distance(ball_pos, center),
            )
            if dist < best_dist:
                best_dist = dist
                best_pid = pid
                best_center = center

        return best_pid, best_dist, best_center

    def _identify_two_players(
        self,
        tracks: Dict,
        sample_frames: int = 60
    ) -> Tuple[int, int]:
        """Identifikasi 2 player ID paling sering muncul."""
        from collections import Counter
        pid_counter = Counter()
        total = min(sample_frames, len(tracks['players']))
        for f in range(total):
            for pid in tracks['players'][f].keys():
                pid_counter[pid] += 1
        most_common = pid_counter.most_common(2)
        if len(most_common) < 2:
            if len(most_common) == 1:
                return most_common[0][0], -1
            return -1, -1
        return most_common[0][0], most_common[1][0]

    # ============================================================
    # DETEKSI ONE-TOUCH PASS — ALTERNATING STATE MACHINE
    # ============================================================

    def detect_onetouch_passes(
        self,
        tracks: Dict,
        debug: bool = True
    ) -> List[Dict]:
        """
        Deteksi one-touch pass dari target player saja.

        ALTERNATING LOGIC:
        - Paso 1: TARGET pegang bola → kick → sampai partner = EVENT
        - Paso 2: PARTNER pegang bola → kick → sampai target = SKIP (return)
        - Paso 3: TARGET pegang bola → kick → sampai partner = EVENT
        - dst...

        Ini hanya menghitung pass dari pemain pertama (TARGET).

        States:
            wait_target   : Menunggu bola sampai di target player
            target_touch  : Bola di kaki target, hitung touch duration
            target_kick   : Bola ditendang target, menuju partner
            wait_return   : Menunggu partner return (bola kembali)
        """
        total_frames = len(tracks['players'])
        if total_frames == 0:
            return []

        max_touch_frames = int(self.max_touch_seconds * self.fps)

        if debug:
            print(f"\n[ONETOUCH] === ONE-TOUCH PASS DETECTION ===")
            print(f"[ONETOUCH] Mode                : ALTERNATING "
                  f"(count target passes only)")
            print(f"[ONETOUCH] Total frames         : {total_frames}")
            print(f"[ONETOUCH] FPS                  : {self.fps}")
            print(f"[ONETOUCH] Possession distance   : {self.ball_possession_distance}px")
            print(f"[ONETOUCH] Kick away distance    : {self.kick_away_distance}px")
            print(f"[ONETOUCH] Receive distance      : {self.receive_distance}px")
            print(f"[ONETOUCH] Max touch             : {self.max_touch_seconds}s "
                  f"({max_touch_frames}f)")
            print(f"[ONETOUCH] Max transit           : {self.max_transit_frames}f")
            print(f"[ONETOUCH] Player separation     : {self.player_separation_distance}px")
            print(f"[ONETOUCH] ================================\n")

        # ============================================================
        # STATE MACHINE VARIABLES
        # ============================================================

        state = 'wait_target'  # First we wait for target player
        is_first_possession = True  # Pemain pertama yang pegang = target

        # Possessor tracking (position-based)
        possessor_pos: Optional[Tuple[int, int]] = None
        possessor_pid = -1
        possession_start_frame = -1
        touch_frames = 0
        possession_counter = 0  # For min_possession_frames

        # Transit tracking
        kick_frame = -1
        kick_pos: Optional[Tuple[int, int]] = None
        transit_frames = 0
        receive_frames = 0
        return_frames = 0  # Berapa frame bola dekat sender (sustained check)
        closest_to_receiver = float('inf')

        # Event tracking
        last_event_frame = -999

        onetouch_events: List[Dict] = []
        event_id_counter = 0

        for frame_num in range(total_frames):
            if frame_num % 200 == 0 and debug:
                print(f"[ONETOUCH] Processing frame {frame_num}/{total_frames} "
                      f"[state={state}]...")

            ball_pos = self._get_ball_position(tracks, frame_num)
            if ball_pos is None:
                if state == 'target_kick':
                    transit_frames += 1
                    if transit_frames > self.max_transit_frames:
                        # GAGAL — timeout
                        event_id_counter += 1
                        touch_sec = round(touch_frames / self.fps, 2)
                        onetouch_events.append({
                            'event_id': event_id_counter,
                            'sender_id': possessor_pid,
                            'receiver_id': -1,
                            'frame_kick': kick_frame,
                            'frame_start': possession_start_frame,
                            'frame_end': frame_num,
                            'success': False,
                            'touch_frames': touch_frames,
                            'touch_seconds': touch_sec,
                            'transit_frames': transit_frames,
                            'flight_seconds': round(transit_frames / self.fps, 2),
                            'closest_distance': round(closest_to_receiver, 1),
                            'reason': 'Timeout — bola hilang',
                        })
                        last_event_frame = frame_num
                        state = 'wait_target'
                        touch_frames = 0
                        possession_counter = 0
                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                                  f"(timeout, bola hilang)")
                continue

            # Cari pemain terdekat ke bola
            nearest_pid, nearest_dist, nearest_center = \
                self._get_nearest_player(ball_pos, tracks, frame_num)

            # ======== STATE: WAIT_TARGET ========
            # Menunggu bola sampai di target player.
            # Jika is_first_possession: pemain pertama yang pegang = target
            # Jika bukan first: tunggu bola sampai di player manapun
            #   (ini adalah target setelah partner return)

            if state == 'wait_target':
                if (frame_num - last_event_frame) < self.cooldown_frames:
                    continue

                if nearest_dist <= self.ball_possession_distance and nearest_pid != -1:
                    possession_counter += 1

                    if possession_counter >= self.min_possession_frames:
                        state = 'target_touch'
                        possessor_pid = nearest_pid
                        possessor_pos = nearest_center
                        possession_start_frame = frame_num - possession_counter + 1
                        touch_frames = possession_counter

                        if is_first_possession:
                            is_first_possession = False
                            if debug:
                                print(f"[ONETOUCH] Frame {frame_num}: "
                                      f"TARGET identified = P{possessor_pid} "
                                      f"(first possession)")

                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: "
                                  f"TARGET_TOUCH start "
                                  f"(P{possessor_pid}, dist={nearest_dist:.0f}px)")
                else:
                    possession_counter = 0

            # ======== STATE: TARGET_TOUCH ========
            # Bola di kaki target player. Hitung touch duration.

            elif state == 'target_touch':
                # Cek jarak bola ke possessor (by position)
                dist_possessor = float('inf')
                current_pid = possessor_pid

                for pid, pdata in tracks['players'][frame_num].items():
                    bbox = pdata.get('bbox')
                    if bbox is None:
                        continue
                    center = get_center_of_bbox(bbox)
                    # Pemain ini = possessor jika dekat possessor_pos
                    if measure_distance(center, possessor_pos) < self.player_separation_distance:
                        foot = get_foot_position(bbox)
                        d = min(
                            measure_distance(ball_pos, foot),
                            measure_distance(ball_pos, center),
                        )
                        if d < dist_possessor:
                            dist_possessor = d
                            current_pid = pid
                            possessor_pos = center  # Update posisi

                if dist_possessor == float('inf'):
                    # Possessor hilang, cek nearest
                    if nearest_dist <= self.ball_possession_distance:
                        dist_possessor = nearest_dist
                        current_pid = nearest_pid
                        possessor_pos = nearest_center

                if dist_possessor <= self.ball_possession_distance:
                    touch_frames += 1
                    possessor_pid = current_pid

                    # Touch terlalu lama?
                    if touch_frames > max_touch_frames:
                        event_id_counter += 1
                        touch_sec = round(touch_frames / self.fps, 2)
                        onetouch_events.append({
                            'event_id': event_id_counter,
                            'sender_id': possessor_pid,
                            'receiver_id': -1,
                            'frame_kick': frame_num,
                            'frame_start': possession_start_frame,
                            'frame_end': frame_num,
                            'success': False,
                            'touch_frames': touch_frames,
                            'touch_seconds': touch_sec,
                            'transit_frames': 0,
                            'flight_seconds': 0.0,
                            'closest_distance': 0.0,
                            'reason': f'Bola ditahan {touch_sec}s '
                                      f'(maks {self.max_touch_seconds}s)',
                        })
                        last_event_frame = frame_num
                        state = 'wait_target'
                        touch_frames = 0
                        possession_counter = 0
                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                                  f"— held {touch_sec}s")
                        continue

                    if debug and touch_frames % 15 == 0:
                        print(f"[ONETOUCH]   touch f={frame_num}: "
                              f"P{possessor_pid} holding "
                              f"{touch_frames/self.fps:.1f}s / "
                              f"{self.max_touch_seconds}s")

                elif dist_possessor > self.kick_away_distance:
                    # Bola ditendang!
                    state = 'target_kick'
                    kick_frame = frame_num
                    kick_pos = possessor_pos
                    transit_frames = 0
                    receive_frames = 0
                    return_frames = 0
                    closest_to_receiver = float('inf')

                    if debug:
                        touch_sec = touch_frames / self.fps
                        print(f"[ONETOUCH] Frame {frame_num}: TARGET KICK "
                              f"(P{possessor_pid}, touch={touch_sec:.2f}s)")

            # ======== STATE: TARGET_KICK ========
            # Bola sudah ditendang oleh target. Cari apakah sampai ke
            # pemain lain (partner).

            elif state == 'target_kick':
                transit_frames += 1

                # Cari pemain yang JAUH dari kick position (= partner)
                recv_pid, dist_recv, recv_center = \
                    self._get_nearest_player_far_from(
                        ball_pos, tracks, frame_num,
                        exclude_pos=kick_pos,
                        min_separation=self.player_separation_distance,
                    )

                # Cek jarak bola ke possessor (apakah kembali?)
                dist_back = float('inf')
                for pid, pdata in tracks['players'][frame_num].items():
                    bbox = pdata.get('bbox')
                    if bbox is None:
                        continue
                    center = get_center_of_bbox(bbox)
                    if measure_distance(center, kick_pos) < self.player_separation_distance:
                        foot = get_foot_position(bbox)
                        d = min(
                            measure_distance(ball_pos, foot),
                            measure_distance(ball_pos, center),
                        )
                        if d < dist_back:
                            dist_back = d

                if dist_recv < closest_to_receiver:
                    closest_to_receiver = dist_recv

                if debug and transit_frames % 10 == 0:
                    print(f"[ONETOUCH]   transit f={frame_num}: "
                          f"recv_pid=P{recv_pid}, "
                          f"dist_recv={dist_recv:.0f}px, "
                          f"dist_back={dist_back:.0f}px")

                # --- BOLA DITERIMA PARTNER ---
                if dist_recv <= self.receive_distance and recv_pid != -1:
                    receive_frames += 1

                    if debug:
                        print(f"[ONETOUCH]   f={frame_num}: DEKAT partner "
                              f"P{recv_pid}! dist={dist_recv:.0f}px "
                              f"(recv={receive_frames}/{self.min_receive_frames})")

                    if receive_frames >= self.min_receive_frames:
                        # SUKSES!
                        event_id_counter += 1
                        touch_sec = round(touch_frames / self.fps, 2)
                        onetouch_events.append({
                            'event_id': event_id_counter,
                            'sender_id': possessor_pid,
                            'receiver_id': recv_pid,
                            'frame_kick': kick_frame,
                            'frame_start': possession_start_frame,
                            'frame_end': frame_num,
                            'success': True,
                            'touch_frames': touch_frames,
                            'touch_seconds': touch_sec,
                            'transit_frames': transit_frames,
                            'flight_seconds': round(transit_frames / self.fps, 2),
                            'closest_distance': round(closest_to_receiver, 1),
                            'receive_distance': round(dist_recv, 1),
                        })
                        last_event_frame = frame_num
                        # Sekarang tunggu partner return
                        state = 'wait_return'
                        touch_frames = 0
                        possession_counter = 0

                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: SUKSES ✓ "
                                  f"(P{possessor_pid}→P{recv_pid}, "
                                  f"touch={touch_sec}s, "
                                  f"transit={transit_frames}f/"
                                  f"{transit_frames/self.fps:.1f}s)")
                        continue
                else:
                    receive_frames = 0

                # --- BOLA KEMBALI KE SENDER ---
                # Require sustained proximity (5 frames) to avoid
                # false trigger when ball just passes near kick area
                if dist_back <= self.ball_possession_distance and transit_frames > 10:
                    return_frames += 1
                    if return_frames >= 5:
                        event_id_counter += 1
                        touch_sec = round(touch_frames / self.fps, 2)
                        onetouch_events.append({
                            'event_id': event_id_counter,
                            'sender_id': possessor_pid,
                            'receiver_id': -1,
                            'frame_kick': kick_frame,
                            'frame_start': possession_start_frame,
                            'frame_end': frame_num,
                            'success': False,
                            'touch_frames': touch_frames,
                            'touch_seconds': touch_sec,
                            'transit_frames': transit_frames,
                            'flight_seconds': round(transit_frames / self.fps, 2),
                            'closest_distance': round(closest_to_receiver, 1),
                            'reason': 'Bola kembali ke pengirim',
                        })
                        last_event_frame = frame_num
                        state = 'wait_target'
                        touch_frames = 0
                        possession_counter = 0
                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                                  f"(bola kembali, dist={dist_back:.0f}px, "
                                  f"return_frames={return_frames})")
                        continue
                else:
                    return_frames = 0

                # --- TIMEOUT ---
                if transit_frames > self.max_transit_frames:
                    event_id_counter += 1
                    touch_sec = round(touch_frames / self.fps, 2)
                    onetouch_events.append({
                        'event_id': event_id_counter,
                        'sender_id': possessor_pid,
                        'receiver_id': -1,
                        'frame_kick': kick_frame,
                        'frame_start': possession_start_frame,
                        'frame_end': frame_num,
                        'success': False,
                        'touch_frames': touch_frames,
                        'touch_seconds': touch_sec,
                        'transit_frames': transit_frames,
                        'flight_seconds': round(transit_frames / self.fps, 2),
                        'closest_distance': round(closest_to_receiver, 1),
                        'reason': f'Timeout transit {transit_frames}f',
                    })
                    last_event_frame = frame_num
                    state = 'wait_target'
                    touch_frames = 0
                    possession_counter = 0
                    if debug:
                        print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                              f"(TIMEOUT {transit_frames}f)")
                    continue

            # ======== STATE: WAIT_RETURN ========
            # Bola sudah diterima partner. Sekarang tunggu partner
            # memainkan bola kembali ke target. Kita TIDAK menghitung
            # pass dari partner. Kita hanya menunggu bola sampai di
            # pemain lain (= target) lagi.

            elif state == 'wait_return':
                if (frame_num - last_event_frame) < self.cooldown_frames:
                    continue

                # Cek: bola dekat pemain manapun?
                if nearest_dist <= self.ball_possession_distance and nearest_pid != -1:
                    possession_counter += 1

                    if possession_counter >= self.min_possession_frames:
                        # Pemain ini memiliki bola (partner).
                        # Set possessor dan kick position, lalu tunggu
                        # bola meninggalkan partner.
                        possessor_pos = nearest_center
                        possessor_pid = nearest_pid
                        state = 'partner_touch'
                        touch_frames = possession_counter

                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: "
                                  f"PARTNER has ball "
                                  f"(P{nearest_pid}, dist={nearest_dist:.0f}px) "
                                  f"— waiting for return...")
                else:
                    possession_counter = 0

            # ======== STATE: PARTNER_TOUCH ========
            # Partner memiliki bola. Tunggu bola meninggalkan partner,
            # lalu tunggu bola sampai ke target (= wait_target).

            elif state == 'partner_touch':
                # Cek jarak bola ke possessor (partner)
                dist_possessor = float('inf')
                for pid, pdata in tracks['players'][frame_num].items():
                    bbox = pdata.get('bbox')
                    if bbox is None:
                        continue
                    center = get_center_of_bbox(bbox)
                    if measure_distance(center, possessor_pos) < self.player_separation_distance:
                        foot = get_foot_position(bbox)
                        d = min(
                            measure_distance(ball_pos, foot),
                            measure_distance(ball_pos, center),
                        )
                        if d < dist_possessor:
                            dist_possessor = d
                            possessor_pos = center

                if dist_possessor == float('inf'):
                    if nearest_dist <= self.ball_possession_distance:
                        dist_possessor = nearest_dist
                        possessor_pos = nearest_center

                if dist_possessor <= self.ball_possession_distance:
                    # Partner masih pegang bola, tunggu
                    touch_frames += 1
                elif dist_possessor > self.kick_away_distance:
                    # Partner sudah tendang bola! → tunggu target receive
                    state = 'wait_target'
                    kick_pos = possessor_pos
                    possession_counter = 0
                    touch_frames = 0

                    if debug:
                        print(f"[ONETOUCH] Frame {frame_num}: "
                              f"PARTNER KICKED — waiting for target...")

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
    # DEBUG
    # ============================================================

    def debug_distances(
        self,
        tracks: Dict,
        sample_every: int = 10,
    ) -> None:
        """Print jarak bola ke SEMUA pemain setiap N frame."""
        total_frames = len(tracks['players'])

        print(f"\n[DEBUG] === JARAK BOLA-PEMAIN (setiap {sample_every} frame) ===")
        print(f"{'Frame':<8} {'BallPos':<16} {'NearestPID':<12} "
              f"{'NearestDist':<14} {'AllPlayers'}")
        print("-" * 90)

        for f in range(0, total_frames, sample_every):
            ball_pos = self._get_ball_position(tracks, f)
            if ball_pos is None:
                continue

            nearest_pid, nearest_dist, _ = self._get_nearest_player(
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

    def _identify_two_players(
        self,
        tracks: Dict,
        sample_frames: int = 60
    ) -> Tuple[int, int]:
        """Identifikasi 2 player ID paling sering muncul."""
        from collections import Counter
        pid_counter = Counter()
        total = min(sample_frames, len(tracks['players']))
        for f in range(total):
            for pid in tracks['players'][f].keys():
                pid_counter[pid] += 1
        most_common = pid_counter.most_common(2)
        if len(most_common) < 2:
            if len(most_common) == 1:
                return most_common[0][0], -1
            return -1, -1
        return most_common[0][0], most_common[1][0]

    # ============================================================
    # STATISTIK
    # ============================================================

    def get_onetouch_statistics(self, events: List[Dict]) -> Dict:
        """Hitung statistik dari event one-touch pass."""
        total = len(events)
        sukses = [e for e in events if e['success']]
        gagal = [e for e in events if not e['success']]

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

        gagal_held = sum(1 for e in gagal if 'ditahan' in e.get('reason', '').lower())
        gagal_timeout = sum(1 for e in gagal if 'timeout' in e.get('reason', '').lower())
        gagal_return = sum(1 for e in gagal if 'kembali' in e.get('reason', '').lower())
        gagal_other = len(gagal) - gagal_held - gagal_timeout - gagal_return

        avg_touch_success = (
            round(float(np.mean([e['touch_seconds'] for e in sukses])), 2)
            if sukses else 0.0
        )
        avg_transit_success = (
            round(float(np.mean([e['flight_seconds'] for e in sukses])), 2)
            if sukses else 0.0
        )
        avg_closest = (
            round(float(np.mean([e['closest_distance'] for e in events])), 1)
            if events else 0.0
        )

        return {
            'total_onetouch': total,
            'successful_onetouch': len(sukses),
            'failed_onetouch': len(gagal),
            'accuracy_pct': round(len(sukses) / total * 100, 1) if total > 0 else 0.0,
            'avg_touch_time_success': avg_touch_success,
            'avg_transit_time_success': avg_transit_success,
            'avg_closest_distance': avg_closest,
            'gagal_held_too_long': gagal_held,
            'gagal_timeout': gagal_timeout,
            'gagal_ball_return': gagal_return,
            'gagal_other': gagal_other,
            'player_stats': player_stats,
        }
