# throughpass_detector.py
# ============================================================
# Through Pass Counting — State Machine:
#   IDLE → POSSESSION → BALL_IN_FLIGHT → SUKSES / GAGAL
#
# Logic:
#   SUKSES = bola yang di-passing melewati celah 2 cone LAWAN
#   GAGAL  = bola tidak melewati celah cone lawan (timeout, meleset, dll)
#
# Model YOLO: 3 class (ball=0, cone=1, player=2)
#
# Setup lapangan:
#   - 2 pemain (Player A & Player B) saling berhadapan
#   - 4 cone total: 2 cone di depan Player A, 2 cone di depan Player B
#   - Through pass sukses = bola melewati celah 2 cone lawan
# ============================================================

import sys
sys.path.append('../')

from utils.bbox_utils import (
    measure_distance,
    get_center_of_bbox,
    get_foot_position,
    stabilize_cone_positions,
    check_ball_passed_through_gate,
    extract_ball_trajectory,
)
import numpy as np
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple


class ThroughPassDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps

        # ============================================================
        # PARAMETER POSSESSION (bola dekat kaki = memiliki bola)
        # ============================================================
        self.ball_possession_distance: float = 150.0

        # ============================================================
        # PARAMETER KICK DETECTION
        # ============================================================
        self.kick_away_distance: float = 150.0
        self.min_possession_frames: int = 3

        # ============================================================
        # PARAMETER FLIGHT / GATE CHECK
        # ============================================================
        self.max_flight_frames: int = 120  # ~4 detik @30fps
        self.gate_proximity_threshold: float = 50.0
        self.min_trajectory_points: int = 3

        # Buffer trajectory: frame extra setelah event untuk capture
        # bola yang masih bergerak menuju cone
        self.trajectory_buffer_after: int = 10
        self.trajectory_buffer_before: int = 0

        # ============================================================
        # PARAMETER COOLDOWN
        # ============================================================
        self.cooldown_frames: int = 30
        self.min_away_frames: int = 5

        # ============================================================
        # CONE STABILIZATION
        # ============================================================
        self.cone_stabilize_frames: int = 60

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

    def _get_min_distance_to_player(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
        player_id: int,
    ) -> float:
        """
        Hitung jarak MINIMAL bola ke pemain.
        Cek jarak ke KAKI dan BADAN, ambil yang terkecil.
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

    def _get_nearest_player_distance(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
    ) -> Tuple[int, float]:
        """
        Cari pemain terdekat ke bola di frame ini.
        Returns: (player_id, min_distance)
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

    def _identify_two_players(
        self,
        tracks: Dict,
        sample_frames: int = 60
    ) -> Tuple[int, int]:
        """
        Identifikasi 2 pemain utama berdasarkan tracking.
        Ambil 2 player ID yang paling sering muncul.
        """
        pid_counter = Counter()
        total = min(sample_frames, len(tracks['players']))
        for f in range(total):
            for pid in tracks['players'][f].keys():
                pid_counter[pid] += 1

        most_common = pid_counter.most_common(2)
        if len(most_common) < 2:
            print("[THROUGHPASS] WARNING: Kurang dari 2 pemain terdeteksi!")
            if len(most_common) == 1:
                return most_common[0][0], -1
            return -1, -1

        return most_common[0][0], most_common[1][0]

    def _get_player_avg_position(
        self,
        tracks: Dict,
        player_id: int,
        sample_frames: int = 60
    ) -> Optional[Tuple[float, float]]:
        """Hitung posisi rata-rata pemain dari N frame pertama."""
        positions = []
        total = min(sample_frames, len(tracks['players']))

        for f in range(total):
            pdata = tracks['players'][f].get(player_id)
            if pdata and 'bbox' in pdata:
                cx, cy = get_center_of_bbox(pdata['bbox'])
                positions.append((cx, cy))

        if not positions:
            return None

        avg_x = float(np.mean([p[0] for p in positions]))
        avg_y = float(np.mean([p[1] for p in positions]))
        return (avg_x, avg_y)

    # ============================================================
    # CONE GROUPING: Assign 2 cone ke setiap pemain
    # ============================================================

    def _stabilize_and_group_cones(
        self,
        tracks: Dict,
        player_a_id: int,
        player_b_id: int,
        debug: bool = True,
    ) -> Tuple[
        Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
        Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    ]:
        """
        Stabilkan posisi 4 cone, lalu group 2 cone terdekat ke Player A
        dan 2 cone terdekat ke Player B.

        Logika: cone target Player A = 2 cone yang paling DEKAT ke Player A
                (karena cone ada di DEPAN pemain yang berlawanan,
                tapi dalam setup ini, cone di depan Player A berarti
                cone yang harus dilewati bola dari Player B)

        PENTING: "2 cone di depan Player A" = gate yang harus dilewati
                 oleh bola dari Player B. Jadi:
                 - gate_a = 2 cone di depan Player A = target untuk Player B
                 - gate_b = 2 cone di depan Player B = target untuk Player A

        Returns:
            (gate_a, gate_b)
            gate_a = (cone1_pos, cone2_pos) di depan Player A (target Player B)
            gate_b = (cone1_pos, cone2_pos) di depan Player B (target Player A)
            Masing-masing bisa None jika gagal
        """
        # Stabilkan posisi cone
        stabilized = stabilize_cone_positions(
            tracks,
            cone_key='cones',
            sample_frames=self.cone_stabilize_frames,
        )

        if len(stabilized) < 4:
            print(f"[THROUGHPASS] WARNING: Hanya {len(stabilized)} cone terdeteksi "
                  f"(butuh 4)! IDs: {list(stabilized.keys())}")
            if len(stabilized) < 2:
                return None, None

        # Ambil posisi rata-rata kedua pemain
        pos_a = self._get_player_avg_position(tracks, player_a_id)
        pos_b = self._get_player_avg_position(tracks, player_b_id)

        if pos_a is None or pos_b is None:
            print("[THROUGHPASS] WARNING: Tidak bisa mendapatkan posisi pemain!")
            return None, None

        if debug:
            print(f"[THROUGHPASS] Player A (ID={player_a_id}) avg pos: "
                  f"({pos_a[0]:.0f}, {pos_a[1]:.0f})")
            print(f"[THROUGHPASS] Player B (ID={player_b_id}) avg pos: "
                  f"({pos_b[0]:.0f}, {pos_b[1]:.0f})")
            print(f"[THROUGHPASS] Stabilized cones ({len(stabilized)}):")
            for cid, cpos in stabilized.items():
                print(f"  Cone {cid}: ({cpos[0]:.0f}, {cpos[1]:.0f})")

        # Hitung jarak setiap cone ke kedua pemain
        cone_distances = []
        for cid, cpos in stabilized.items():
            dist_a = measure_distance(cpos, pos_a)
            dist_b = measure_distance(cpos, pos_b)
            cone_distances.append({
                'cone_id': cid,
                'pos': cpos,
                'dist_a': dist_a,
                'dist_b': dist_b,
                'closer_to': 'A' if dist_a < dist_b else 'B',
            })

        # Sort dan assign: 2 cone terdekat ke A → gate_a, 2 terdekat ke B → gate_b
        cones_near_a = sorted(
            cone_distances, key=lambda c: c['dist_a']
        )
        cones_near_b = sorted(
            cone_distances, key=lambda c: c['dist_b']
        )

        # Ambil 2 cone terdekat ke masing-masing pemain
        # Tapi pastikan tidak ada overlap (1 cone hanya bisa milik 1 gate)
        assigned_to_a = set()
        gate_a_cones = []
        for c in cones_near_a:
            if len(gate_a_cones) >= 2:
                break
            gate_a_cones.append(c)
            assigned_to_a.add(c['cone_id'])

        gate_b_cones = []
        for c in cones_near_b:
            if len(gate_b_cones) >= 2:
                break
            if c['cone_id'] not in assigned_to_a:
                gate_b_cones.append(c)

        if len(gate_a_cones) < 2 or len(gate_b_cones) < 2:
            print("[THROUGHPASS] WARNING: Tidak cukup cone untuk membentuk 2 gate!")
            return None, None

        gate_a = (gate_a_cones[0]['pos'], gate_a_cones[1]['pos'])
        gate_b = (gate_b_cones[0]['pos'], gate_b_cones[1]['pos'])

        if debug:
            print(f"\n[THROUGHPASS] Gate A (di depan Player A, target Player B):")
            print(f"  Cone {gate_a_cones[0]['cone_id']}: "
                  f"({gate_a[0][0]:.0f}, {gate_a[0][1]:.0f})")
            print(f"  Cone {gate_a_cones[1]['cone_id']}: "
                  f"({gate_a[1][0]:.0f}, {gate_a[1][1]:.0f})")
            print(f"  Gate width: "
                  f"{measure_distance(gate_a[0], gate_a[1]):.0f}px")

            print(f"[THROUGHPASS] Gate B (di depan Player B, target Player A):")
            print(f"  Cone {gate_b_cones[0]['cone_id']}: "
                  f"({gate_b[0][0]:.0f}, {gate_b[0][1]:.0f})")
            print(f"  Cone {gate_b_cones[1]['cone_id']}: "
                  f"({gate_b[1][0]:.0f}, {gate_b[1][1]:.0f})")
            print(f"  Gate width: "
                  f"{measure_distance(gate_b[0], gate_b[1]):.0f}px")

        return gate_a, gate_b

    # ============================================================
    # DETEKSI THROUGH PASS — CORE LOGIC
    # ============================================================

    def detect_throughpasses(
        self,
        tracks: Dict,
        debug: bool = True
    ) -> List[Dict]:
        """
        Deteksi through pass dari tracking data.

        Logic (State Machine):
            IDLE → POSSESSION → BALL_IN_FLIGHT → SUKSES/GAGAL → IDLE

        Through pass SUKSES = bola melewati celah 2 cone LAWAN.
        - Player A passing → bola melewati gate B → SUKSES untuk Player A
        - Player B passing → bola melewati gate A → SUKSES untuk Player B

        Returns:
            List[Dict] berisi event through pass
        """
        total_frames = len(tracks['players'])
        if total_frames == 0:
            return []

        # --- Identifikasi pemain dan cone ---
        player_a, player_b = self._identify_two_players(tracks)

        if debug:
            print(f"\n[THROUGHPASS] === THROUGH PASS DETECTION ===")
            print(f"[THROUGHPASS] Total frames           : {total_frames}")
            print(f"[THROUGHPASS] FPS                    : {self.fps}")
            print(f"[THROUGHPASS] Player A               : {player_a}")
            print(f"[THROUGHPASS] Player B               : {player_b}")
            print(f"[THROUGHPASS] Possession distance    : {self.ball_possession_distance}px")
            print(f"[THROUGHPASS] Kick away distance     : {self.kick_away_distance}px")
            print(f"[THROUGHPASS] Max flight frames      : {self.max_flight_frames}")
            print(f"[THROUGHPASS] Gate proximity thresh   : {self.gate_proximity_threshold}px")
            print(f"[THROUGHPASS] Cooldown frames        : {self.cooldown_frames}")

        # --- Stabilize dan group cone ---
        gate_a, gate_b = self._stabilize_and_group_cones(
            tracks, player_a, player_b, debug=debug
        )

        if gate_a is None or gate_b is None:
            print("[THROUGHPASS] ERROR: Tidak bisa membentuk gate! Abort.")
            return []

        if debug:
            print(f"[THROUGHPASS] ================================\n")

        # ============================================================
        # STATE MACHINE
        # ============================================================

        state = 'idle'
        sender_id = -1
        sender_position = None
        kick_frame = -1
        flight_frames = 0
        possession_frames = 0
        current_possessor = -1
        last_event_frame = -999
        away_frames = self.min_away_frames

        # Trajectory bola selama flight
        ball_trajectory: List[Tuple[int, int]] = []

        throughpass_events: List[Dict] = []
        event_id_counter = 0

        for frame_num in range(total_frames):
            if frame_num % 200 == 0 and debug:
                print(f"[THROUGHPASS] Processing frame {frame_num}/{total_frames}...")

            # --- Ambil posisi bola ---
            ball_pos = self._get_ball_position(tracks, frame_num)
            if ball_pos is None:
                if state == 'ball_in_flight':
                    flight_frames += 1
                    if flight_frames > self.max_flight_frames:
                        if debug:
                            print(f"[THROUGHPASS] Frame {frame_num}: TIMEOUT "
                                  f"— bola hilang selama flight")
                        # Cek gate sebelum reset
                        target_gate = gate_b if sender_id == player_a else gate_a
                        passed, reason = check_ball_passed_through_gate(
                            ball_trajectory,
                            target_gate[0], target_gate[1],
                            proximity_threshold=self.gate_proximity_threshold,
                            min_trajectory_points=self.min_trajectory_points,
                        )
                        event_id_counter += 1
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': sender_id,
                            'frame_kick': kick_frame,
                            'frame_end': frame_num,
                            'frame_start': kick_frame,
                            'success': passed,
                            'flight_frames': flight_frames,
                            'flight_seconds': round(flight_frames / self.fps, 2),
                            'reason': reason if not passed else 'Bola melewati gate (timeout)',
                            'trajectory_length': len(ball_trajectory),
                        }
                        throughpass_events.append(event)
                        last_event_frame = frame_num
                        away_frames = 0
                        state = 'idle'
                        possession_frames = 0
                        ball_trajectory = []

                        if debug:
                            status = "SUKSES ✓" if passed else "GAGAL ✗"
                            print(f"[THROUGHPASS] Frame {frame_num}: THROUGHPASS {status} "
                                  f"(timeout, {reason})")
                continue

            # --- Cari pemain terdekat ke bola ---
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

                # Cek possession
                if nearest_dist <= self.ball_possession_distance and nearest_pid != -1:
                    if current_possessor == nearest_pid:
                        possession_frames += 1
                    else:
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
                        print(f"[THROUGHPASS] Frame {frame_num}: Player {sender_id} "
                              f"possession (frames={possession_frames}, "
                              f"dist={nearest_dist:.0f}px)")

            elif state == 'waiting_kick':
                # Cek apakah bola sudah ditendang (menjauh dari sender)
                dist_sender = self._get_min_distance_to_player(
                    ball_pos, tracks, frame_num, sender_id
                )

                if dist_sender == float('inf'):
                    if nearest_dist <= self.ball_possession_distance:
                        sender_id = nearest_pid
                        dist_sender = nearest_dist
                    else:
                        dist_sender = self.kick_away_distance + 1

                if dist_sender > self.kick_away_distance:
                    # Bola ditendang!
                    pdata = tracks['players'][frame_num].get(sender_id)
                    if pdata and 'bbox' in pdata:
                        sender_position = get_center_of_bbox(pdata['bbox'])
                    else:
                        for back_f in range(frame_num - 1, max(0, frame_num - 10) - 1, -1):
                            pdata = tracks['players'][back_f].get(sender_id)
                            if pdata and 'bbox' in pdata:
                                sender_position = get_center_of_bbox(pdata['bbox'])
                                break
                        else:
                            sender_position = ball_pos

                    state = 'ball_in_flight'
                    kick_frame = frame_num
                    flight_frames = 0
                    ball_trajectory = [ball_pos]

                    if debug:
                        # Tentukan target gate
                        target_name = "Gate B" if sender_id == player_a else "Gate A"
                        print(f"[THROUGHPASS] Frame {frame_num}: KICK! "
                              f"Player {sender_id} → target {target_name} "
                              f"(dist_sender={dist_sender:.0f}px)")

                elif dist_sender <= self.ball_possession_distance:
                    # Masih dekat sender, cek apakah pemain lain ambil
                    if nearest_pid != sender_id and nearest_dist < dist_sender:
                        if nearest_dist <= self.ball_possession_distance:
                            state = 'idle'
                            possession_frames = 0
                            away_frames = self.min_away_frames

            elif state == 'ball_in_flight':
                flight_frames += 1
                ball_trajectory.append(ball_pos)

                # Tentukan target gate berdasarkan sender
                if sender_id == player_a:
                    target_gate = gate_b  # Player A passing → harus lewat gate B
                elif sender_id == player_b:
                    target_gate = gate_a  # Player B passing → harus lewat gate A
                else:
                    # Sender bukan A atau B → assign berdasarkan proximity
                    pos_a = self._get_player_avg_position(tracks, player_a)
                    pos_b = self._get_player_avg_position(tracks, player_b)
                    if sender_position and pos_a and pos_b:
                        dist_to_a = measure_distance(sender_position, pos_a)
                        dist_to_b = measure_distance(sender_position, pos_b)
                        target_gate = gate_b if dist_to_a < dist_to_b else gate_a
                    else:
                        target_gate = gate_b  # fallback

                # Cek apakah bola melewati gate setiap beberapa frame
                # (minimal butuh beberapa titik trajectory)
                if len(ball_trajectory) >= self.min_trajectory_points:
                    passed, reason = check_ball_passed_through_gate(
                        ball_trajectory,
                        target_gate[0], target_gate[1],
                        proximity_threshold=self.gate_proximity_threshold,
                        min_trajectory_points=self.min_trajectory_points,
                    )

                    if passed:
                        # THROUGH PASS SUKSES!
                        event_id_counter += 1
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': sender_id,
                            'frame_kick': kick_frame,
                            'frame_end': frame_num,
                            'frame_start': kick_frame,
                            'success': True,
                            'flight_frames': flight_frames,
                            'flight_seconds': round(flight_frames / self.fps, 2),
                            'reason': reason,
                            'trajectory_length': len(ball_trajectory),
                        }
                        throughpass_events.append(event)
                        last_event_frame = frame_num
                        away_frames = 0
                        state = 'idle'
                        possession_frames = 0
                        ball_trajectory = []

                        if debug:
                            print(f"[THROUGHPASS] Frame {frame_num}: THROUGHPASS SUKSES ✓ "
                                  f"(P{sender_id}, flight={flight_frames}f/"
                                  f"{flight_frames/self.fps:.1f}s, {reason})")
                        continue

                # --- CEK BOLA DITERIMA OLEH LAWAN (bola sudah berhenti/diterima) ---
                # Jika bola dekat pemain lawan, event selesai
                if sender_id == player_a:
                    recv_pid = player_b
                elif sender_id == player_b:
                    recv_pid = player_a
                else:
                    recv_pid = -1

                if recv_pid != -1:
                    dist_recv = self._get_min_distance_to_player(
                        ball_pos, tracks, frame_num, recv_pid
                    )
                    # Jika bola sudah dekat penerima, finalize event
                    if dist_recv <= self.ball_possession_distance and flight_frames > 5:
                        # Cek gate satu kali terakhir
                        passed, reason = check_ball_passed_through_gate(
                            ball_trajectory,
                            target_gate[0], target_gate[1],
                            proximity_threshold=self.gate_proximity_threshold,
                            min_trajectory_points=self.min_trajectory_points,
                        )
                        event_id_counter += 1
                        event = {
                            'event_id': event_id_counter,
                            'sender_id': sender_id,
                            'frame_kick': kick_frame,
                            'frame_end': frame_num,
                            'frame_start': kick_frame,
                            'success': passed,
                            'flight_frames': flight_frames,
                            'flight_seconds': round(flight_frames / self.fps, 2),
                            'reason': reason,
                            'trajectory_length': len(ball_trajectory),
                        }
                        throughpass_events.append(event)
                        last_event_frame = frame_num
                        away_frames = 0
                        state = 'idle'
                        possession_frames = 0
                        ball_trajectory = []

                        if debug:
                            status = "SUKSES ✓" if passed else "GAGAL ✗"
                            print(f"[THROUGHPASS] Frame {frame_num}: THROUGHPASS {status} "
                                  f"(P{sender_id}, bola diterima P{recv_pid}, "
                                  f"dist={dist_recv:.0f}px, {reason})")
                        continue

                # --- CEK BOLA KEMBALI KE PENGIRIM ---
                dist_sender = self._get_min_distance_to_player(
                    ball_pos, tracks, frame_num, sender_id
                )
                if dist_sender <= self.ball_possession_distance and flight_frames > 15:
                    passed, reason = check_ball_passed_through_gate(
                        ball_trajectory,
                        target_gate[0], target_gate[1],
                        proximity_threshold=self.gate_proximity_threshold,
                        min_trajectory_points=self.min_trajectory_points,
                    )
                    event_id_counter += 1
                    event = {
                        'event_id': event_id_counter,
                        'sender_id': sender_id,
                        'frame_kick': kick_frame,
                        'frame_end': frame_num,
                        'frame_start': kick_frame,
                        'success': passed,
                        'flight_frames': flight_frames,
                        'flight_seconds': round(flight_frames / self.fps, 2),
                        'reason': reason if passed else 'Bola kembali ke pengirim',
                        'trajectory_length': len(ball_trajectory),
                    }
                    throughpass_events.append(event)
                    last_event_frame = frame_num
                    away_frames = 0
                    state = 'idle'
                    possession_frames = 0
                    ball_trajectory = []

                    if debug:
                        status = "SUKSES ✓" if passed else "GAGAL ✗"
                        print(f"[THROUGHPASS] Frame {frame_num}: THROUGHPASS {status} "
                              f"(bola kembali, {reason})")
                    continue

                # --- TIMEOUT ---
                if flight_frames > self.max_flight_frames:
                    passed, reason = check_ball_passed_through_gate(
                        ball_trajectory,
                        target_gate[0], target_gate[1],
                        proximity_threshold=self.gate_proximity_threshold,
                        min_trajectory_points=self.min_trajectory_points,
                    )
                    event_id_counter += 1
                    event = {
                        'event_id': event_id_counter,
                        'sender_id': sender_id,
                        'frame_kick': kick_frame,
                        'frame_end': frame_num,
                        'frame_start': kick_frame,
                        'success': passed,
                        'flight_frames': flight_frames,
                        'flight_seconds': round(flight_frames / self.fps, 2),
                        'reason': reason if passed else f'Timeout ({flight_frames}f)',
                        'trajectory_length': len(ball_trajectory),
                    }
                    throughpass_events.append(event)
                    last_event_frame = frame_num
                    away_frames = 0
                    state = 'idle'
                    possession_frames = 0
                    ball_trajectory = []

                    if debug:
                        status = "SUKSES ✓" if passed else "GAGAL ✗"
                        print(f"[THROUGHPASS] Frame {frame_num}: THROUGHPASS {status} "
                              f"(TIMEOUT, {reason})")
                    continue

                # Debug setiap 10 frame
                if debug and flight_frames % 10 == 0:
                    print(f"[THROUGHPASS]   flight f={frame_num}: "
                          f"traj_len={len(ball_trajectory)}, "
                          f"nearest={nearest_pid}:{nearest_dist:.0f}px")

        # ============================================================
        # HASIL
        # ============================================================
        if debug:
            sukses = sum(1 for e in throughpass_events if e['success'])
            gagal = sum(1 for e in throughpass_events if not e['success'])
            total = len(throughpass_events)
            print(f"\n[THROUGHPASS] === HASIL AKHIR ===")
            print(f"[THROUGHPASS] Total through pass  : {total}")
            print(f"[THROUGHPASS] SUKSES              : {sukses}")
            print(f"[THROUGHPASS] GAGAL               : {gagal}")
            if total > 0:
                print(f"[THROUGHPASS] Akurasi             : "
                      f"{sukses/total*100:.1f}%")
            print(f"[THROUGHPASS] =====================\n")

        return throughpass_events

    # ============================================================
    # GATE INFO (untuk visualisasi)
    # ============================================================

    def get_gates(
        self,
        tracks: Dict,
    ) -> Tuple[int, int,
               Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
               Optional[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        """
        Return player IDs dan gate positions untuk visualisasi.
        Returns: (player_a_id, player_b_id, gate_a, gate_b)
        """
        player_a, player_b = self._identify_two_players(tracks)
        gate_a, gate_b = self._stabilize_and_group_cones(
            tracks, player_a, player_b, debug=False
        )
        return player_a, player_b, gate_a, gate_b

    # ============================================================
    # DEBUG: Print jarak bola-pemain per frame
    # ============================================================

    def debug_distances(
        self,
        tracks: Dict,
        sample_every: int = 10,
    ) -> None:
        """Print jarak bola ke semua pemain dan cone setiap N frame."""
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

    def get_throughpass_statistics(self, events: List[Dict]) -> Dict:
        """Hitung statistik dari event through pass."""
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

        return {
            'total_throughpass': total,
            'successful_throughpass': len(sukses),
            'failed_throughpass': len(gagal),
            'accuracy_pct': round(
                len(sukses) / total * 100, 1
            ) if total > 0 else 0.0,
            'avg_flight_time_success': avg_flight_success,
            'player_stats': player_stats,
        }
