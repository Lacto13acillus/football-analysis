# onetouch_detector.py
# ============================================================
# One-Touch Pass Counting — State Machine (POSITION-BASED):
#   IDLE → BALL_AT_TARGET → BALL_TRAVELING → SUKSES / GAGAL
#
# Logic:
#   Hanya menghitung pass DARI target player (baju hitam)
#   KE partner player (baju orange).
#   Pass dari partner → target TIDAK dihitung (hanya ball return).
#
#   SUKSES = bola dari target player diterima oleh partner
#            dengan sentuhan < max_touch_seconds (2 detik)
#   GAGAL  = bola ditahan terlalu lama (> 2 detik)
#          = bola tidak sampai ke partner (timeout)
#
# POSITION-BASED: Menggunakan spatial clustering untuk
#   mengidentifikasi 2 pemain fisik, lalu auto-detect
#   target player dari posisi pemain pertama yang pegang bola.
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
        self.cooldown_frames: int = 15
        self.min_away_frames: int = 3

        # ============================================================
        # PARAMETER SPATIAL CLUSTERING
        # ============================================================
        # Jarak maks untuk menganggap 2 posisi sebagai pemain yang sama
        self.cluster_match_distance: float = 250.0
        # Jumlah frame awal untuk sampling posisi pemain
        self.cluster_sample_frames: int = 60

        # ============================================================
        # INTERNAL STATE (diisi saat detect)
        # ============================================================
        self._target_center: Optional[Tuple[float, float]] = None
        self._partner_center: Optional[Tuple[float, float]] = None

    # ============================================================
    # SPATIAL CLUSTERING: Identifikasi 2 pemain fisik
    # ============================================================

    def _cluster_player_positions(
        self,
        tracks: Dict,
        sample_frames: int = 60,
    ) -> List[Tuple[float, float]]:
        """
        Cluster posisi semua pemain dari N frame pertama menjadi
        2 kelompok posisi (= 2 pemain fisik).

        Menggunakan simple clustering: kumpulkan semua posisi,
        lalu cluster berdasarkan jarak.

        Returns:
            List of 2 cluster centers [(x1,y1), (x2,y2)]
        """
        all_positions: List[Tuple[float, float]] = []
        total = min(sample_frames, len(tracks['players']))

        for f in range(total):
            for pid, pdata in tracks['players'][f].items():
                bbox = pdata.get('bbox')
                if bbox is None:
                    continue
                cx, cy = get_center_of_bbox(bbox)
                all_positions.append((float(cx), float(cy)))

        if len(all_positions) < 2:
            print("[ONETOUCH] WARNING: Kurang dari 2 posisi pemain!")
            return []

        # Simple 2-means clustering
        # Inisialisasi: ambil 2 posisi terjauh
        max_dist = 0
        seed_a, seed_b = all_positions[0], all_positions[1]
        # Sample beberapa pasang untuk cari yang terjauh
        step = max(1, len(all_positions) // 20)
        for i in range(0, len(all_positions), step):
            for j in range(i + 1, len(all_positions), step):
                d = measure_distance(all_positions[i], all_positions[j])
                if d > max_dist:
                    max_dist = d
                    seed_a = all_positions[i]
                    seed_b = all_positions[j]

        # Iterate clustering
        for _ in range(10):
            cluster_a: List[Tuple[float, float]] = []
            cluster_b: List[Tuple[float, float]] = []

            for pos in all_positions:
                da = measure_distance(pos, seed_a)
                db = measure_distance(pos, seed_b)
                if da <= db:
                    cluster_a.append(pos)
                else:
                    cluster_b.append(pos)

            if not cluster_a or not cluster_b:
                break

            new_a = (
                float(np.mean([p[0] for p in cluster_a])),
                float(np.mean([p[1] for p in cluster_a])),
            )
            new_b = (
                float(np.mean([p[0] for p in cluster_b])),
                float(np.mean([p[1] for p in cluster_b])),
            )

            if (measure_distance(new_a, seed_a) < 1.0 and
                    measure_distance(new_b, seed_b) < 1.0):
                break

            seed_a, seed_b = new_a, new_b

        return [seed_a, seed_b]

    def _identify_target_player(
        self,
        tracks: Dict,
        cluster_centers: List[Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Identifikasi target player (yang akan di-counting) berdasarkan
        siapa yang PERTAMA memiliki bola.

        Target = pemain pertama yang pegang bola (baju hitam).
        Partner = pemain lainnya (baju orange).

        Returns:
            (target_center, partner_center)
        """
        total = len(tracks['players'])

        for f in range(total):
            ball_pos = self._get_ball_position(tracks, f)
            if ball_pos is None:
                continue

            # Cari pemain terdekat ke bola
            nearest_pid, nearest_dist, nearest_center = \
                self._get_nearest_player_distance(ball_pos, tracks, f)

            if nearest_dist <= self.ball_possession_distance and nearest_center:
                # Pemain ini memiliki bola pertama = target
                # Tentukan cluster mana yang paling dekat
                d0 = measure_distance(nearest_center, cluster_centers[0])
                d1 = measure_distance(nearest_center, cluster_centers[1])

                if d0 <= d1:
                    target = cluster_centers[0]
                    partner = cluster_centers[1]
                else:
                    target = cluster_centers[1]
                    partner = cluster_centers[0]

                print(f"[ONETOUCH] Target player (first possession) "
                      f"at frame {f}: "
                      f"pos=({target[0]:.0f},{target[1]:.0f})")
                print(f"[ONETOUCH] Partner player: "
                      f"pos=({partner[0]:.0f},{partner[1]:.0f})")

                return target, partner

        # Fallback: cluster 0 = target
        print("[ONETOUCH] WARNING: Tidak bisa detect first possession, "
              "fallback ke cluster[0] sebagai target")
        return cluster_centers[0], cluster_centers[1]

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
        Cari pemain MANAPUN yang paling dekat ke bola.
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

    def _classify_player_cluster(
        self,
        player_center: Tuple[int, int],
    ) -> str:
        """
        Tentukan apakah pemain di posisi ini termasuk 'target' atau 'partner'.
        Berdasarkan jarak ke cluster center.

        Returns: 'target' atau 'partner'
        """
        if self._target_center is None or self._partner_center is None:
            return 'unknown'

        dt = measure_distance(player_center, self._target_center)
        dp = measure_distance(player_center, self._partner_center)

        return 'target' if dt <= dp else 'partner'

    def _get_ball_dist_to_cluster(
        self,
        ball_pos: Tuple[int, int],
        tracks: Dict,
        frame_num: int,
        cluster_center: Tuple[float, float],
    ) -> Tuple[float, int, Optional[Tuple[int, int]]]:
        """
        Hitung jarak bola ke pemain manapun yang berada di cluster ini.
        Returns: (min_distance, player_id, player_center)
        """
        best_dist = float('inf')
        best_pid = -1
        best_center = None

        for pid, pdata in tracks['players'][frame_num].items():
            bbox = pdata.get('bbox')
            if bbox is None:
                continue

            player_center = get_center_of_bbox(bbox)

            # Cek apakah pemain ini termasuk cluster yang dimaksud
            if self._classify_player_cluster(player_center) == 'target':
                is_target = True
            else:
                is_target = False

            # Match cluster
            d_to_cluster = measure_distance(player_center, cluster_center)
            dt = measure_distance(player_center, self._target_center) if self._target_center else float('inf')
            dp = measure_distance(player_center, self._partner_center) if self._partner_center else float('inf')

            # Pemain dianggap milik cluster ini jika cluster_center adalah
            # yang lebih dekat
            if cluster_center == self._target_center and dt > dp:
                continue
            elif cluster_center == self._partner_center and dp > dt:
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

        return best_dist, best_pid, best_center

    def _identify_two_players(
        self,
        tracks: Dict,
        sample_frames: int = 60
    ) -> Tuple[int, int]:
        """Identifikasi 2 player ID paling sering muncul (untuk rendering)."""
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
    # DETEKSI ONE-TOUCH PASS — CORE LOGIC
    # ============================================================

    def detect_onetouch_passes(
        self,
        tracks: Dict,
        debug: bool = True
    ) -> List[Dict]:
        """
        Deteksi one-touch pass. Hanya menghitung pass dari
        TARGET player ke PARTNER player.

        State Machine:
            IDLE → BALL_AT_TARGET → BALL_TRAVELING → SUKSES/GAGAL

        IDLE: Menunggu bola sampai di target player.
              Bola di partner player TIDAK memicu event baru.
        BALL_AT_TARGET: Bola di kaki target player, hitung touch duration.
        BALL_TRAVELING: Bola sudah dioper, cek apakah sampai ke partner.

        Returns:
            List[Dict] berisi event one-touch pass
        """
        total_frames = len(tracks['players'])
        if total_frames == 0:
            return []

        max_touch_frames = int(self.max_touch_seconds * self.fps)

        # ============================================================
        # STEP 1: Clustering — identifikasi 2 pemain fisik
        # ============================================================
        if debug:
            print(f"\n[ONETOUCH] === STEP 1: Spatial Clustering ===")

        clusters = self._cluster_player_positions(
            tracks, sample_frames=self.cluster_sample_frames
        )
        if len(clusters) < 2:
            print("[ONETOUCH] ERROR: Tidak bisa mengidentifikasi 2 pemain!")
            return []

        if debug:
            print(f"[ONETOUCH] Cluster A: ({clusters[0][0]:.0f}, {clusters[0][1]:.0f})")
            print(f"[ONETOUCH] Cluster B: ({clusters[1][0]:.0f}, {clusters[1][1]:.0f})")
            print(f"[ONETOUCH] Jarak antar cluster: "
                  f"{measure_distance(clusters[0], clusters[1]):.0f}px")

        # ============================================================
        # STEP 2: Identifikasi target player (first possession)
        # ============================================================
        if debug:
            print(f"\n[ONETOUCH] === STEP 2: Identify Target Player ===")

        self._target_center, self._partner_center = \
            self._identify_target_player(tracks, clusters)

        if debug:
            print(f"\n[ONETOUCH] === ONE-TOUCH PASS DETECTION ===")
            print(f"[ONETOUCH] Mode              : ONE-DIRECTIONAL "
                  f"(target → partner)")
            print(f"[ONETOUCH] Total frames       : {total_frames}")
            print(f"[ONETOUCH] FPS                : {self.fps}")
            print(f"[ONETOUCH] Possession dist    : {self.ball_possession_distance}px")
            print(f"[ONETOUCH] Kick away dist     : {self.kick_away_distance}px")
            print(f"[ONETOUCH] Receive dist       : {self.receive_distance}px")
            print(f"[ONETOUCH] Max touch          : {self.max_touch_seconds}s "
                  f"({max_touch_frames}f)")
            print(f"[ONETOUCH] Max transit        : {self.max_transit_frames}f")
            print(f"[ONETOUCH] Cooldown           : {self.cooldown_frames}f")
            print(f"[ONETOUCH] ================================\n")

        # ============================================================
        # STATE MACHINE
        # ============================================================

        state = 'idle'
        sender_pid = -1
        possession_start_frame = -1
        touch_frames = 0
        transit_frames = 0
        receive_frames = 0
        kick_frame = -1
        last_event_frame = -999
        closest_to_partner = float('inf')

        onetouch_events: List[Dict] = []
        event_id_counter = 0

        for frame_num in range(total_frames):
            if frame_num % 200 == 0 and debug:
                print(f"[ONETOUCH] Processing frame {frame_num}/{total_frames}...")

            ball_pos = self._get_ball_position(tracks, frame_num)
            if ball_pos is None:
                if state == 'ball_traveling':
                    transit_frames += 1
                    if transit_frames > self.max_transit_frames:
                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: TIMEOUT — bola hilang")
                        event_id_counter += 1
                        onetouch_events.append({
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
                            'closest_distance': round(closest_to_partner, 1),
                            'reason': 'Timeout — bola hilang',
                        })
                        last_event_frame = frame_num
                        state = 'idle'
                continue

            # Jarak bola ke target cluster dan partner cluster
            dist_target, target_pid, target_pos = self._get_ball_dist_to_cluster(
                ball_pos, tracks, frame_num, self._target_center
            )
            dist_partner, partner_pid, partner_pos = self._get_ball_dist_to_cluster(
                ball_pos, tracks, frame_num, self._partner_center
            )

            # Update cluster centers secara dinamis
            if target_pos and dist_target < self.ball_possession_distance:
                # Smooth update target center
                self._target_center = (
                    self._target_center[0] * 0.9 + target_pos[0] * 0.1,
                    self._target_center[1] * 0.9 + target_pos[1] * 0.1,
                )
            if partner_pos and dist_partner < self.ball_possession_distance:
                self._partner_center = (
                    self._partner_center[0] * 0.9 + partner_pos[0] * 0.1,
                    self._partner_center[1] * 0.9 + partner_pos[1] * 0.1,
                )

            # ======== STATE MACHINE ========

            if state == 'idle':
                # Cooldown
                if (frame_num - last_event_frame) < self.cooldown_frames:
                    continue

                # Cek: bola di kaki TARGET player?
                if dist_target <= self.ball_possession_distance and target_pid != -1:
                    touch_frames += 1

                    if touch_frames >= self.min_possession_frames:
                        state = 'ball_at_target'
                        possession_start_frame = frame_num - touch_frames + 1
                        sender_pid = target_pid

                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: "
                                  f"BALL_AT_TARGET (P{sender_pid}, "
                                  f"dist={dist_target:.0f}px)")
                else:
                    touch_frames = 0

            elif state == 'ball_at_target':
                # Bola di kaki target player, hitung touch duration
                if dist_target <= self.ball_possession_distance:
                    touch_frames += 1
                    sender_pid = target_pid if target_pid != -1 else sender_pid

                    # Touch terlalu lama?
                    if touch_frames > max_touch_frames:
                        event_id_counter += 1
                        touch_sec = round(touch_frames / self.fps, 2)
                        onetouch_events.append({
                            'event_id': event_id_counter,
                            'sender_id': sender_pid,
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
                        state = 'idle'
                        touch_frames = 0

                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                                  f"— held {touch_sec}s (maks "
                                  f"{self.max_touch_seconds}s)")
                        continue

                    if debug and touch_frames % 15 == 0:
                        print(f"[ONETOUCH]   touch f={frame_num}: "
                              f"P{sender_pid} holding "
                              f"{touch_frames/self.fps:.1f}s / "
                              f"{self.max_touch_seconds}s")

                elif dist_target > self.kick_away_distance:
                    # Bola ditendang!
                    state = 'ball_traveling'
                    kick_frame = frame_num
                    transit_frames = 0
                    receive_frames = 0
                    closest_to_partner = float('inf')

                    if debug:
                        touch_sec = touch_frames / self.fps
                        print(f"[ONETOUCH] Frame {frame_num}: KICK by "
                              f"P{sender_pid} (touch={touch_sec:.2f}s, "
                              f"dist_target={dist_target:.0f}px)")

            elif state == 'ball_traveling':
                transit_frames += 1

                # Update closest
                if dist_partner < closest_to_partner:
                    closest_to_partner = dist_partner

                if debug and transit_frames % 10 == 0:
                    print(f"[ONETOUCH]   transit f={frame_num}: "
                          f"dist_partner={dist_partner:.0f}px, "
                          f"dist_target={dist_target:.0f}px "
                          f"(P_partner={partner_pid})")

                # --- BOLA DITERIMA PARTNER ---
                if dist_partner <= self.receive_distance and partner_pid != -1:
                    receive_frames += 1

                    if debug:
                        print(f"[ONETOUCH]   f={frame_num}: bola DEKAT partner "
                              f"P{partner_pid}! dist={dist_partner:.0f}px "
                              f"(recv={receive_frames}/{self.min_receive_frames})")

                    if receive_frames >= self.min_receive_frames:
                        # SUKSES!
                        event_id_counter += 1
                        touch_sec = round(touch_frames / self.fps, 2)
                        onetouch_events.append({
                            'event_id': event_id_counter,
                            'sender_id': sender_pid,
                            'receiver_id': partner_pid,
                            'frame_kick': kick_frame,
                            'frame_start': possession_start_frame,
                            'frame_end': frame_num,
                            'success': True,
                            'touch_frames': touch_frames,
                            'touch_seconds': touch_sec,
                            'transit_frames': transit_frames,
                            'flight_seconds': round(transit_frames / self.fps, 2),
                            'closest_distance': round(closest_to_partner, 1),
                            'receive_distance': round(dist_partner, 1),
                        })
                        last_event_frame = frame_num
                        state = 'idle'
                        touch_frames = 0

                        if debug:
                            print(f"[ONETOUCH] Frame {frame_num}: SUKSES ✓ "
                                  f"(P{sender_pid}→P{partner_pid}, "
                                  f"touch={touch_sec}s, "
                                  f"transit={transit_frames}f/"
                                  f"{transit_frames/self.fps:.1f}s)")
                        continue
                else:
                    receive_frames = 0

                # --- BOLA KEMBALI KE TARGET ---
                if dist_target <= self.ball_possession_distance and transit_frames > 10:
                    event_id_counter += 1
                    touch_sec = round(touch_frames / self.fps, 2)
                    onetouch_events.append({
                        'event_id': event_id_counter,
                        'sender_id': sender_pid,
                        'receiver_id': -1,
                        'frame_kick': kick_frame,
                        'frame_start': possession_start_frame,
                        'frame_end': frame_num,
                        'success': False,
                        'touch_frames': touch_frames,
                        'touch_seconds': touch_sec,
                        'transit_frames': transit_frames,
                        'flight_seconds': round(transit_frames / self.fps, 2),
                        'closest_distance': round(closest_to_partner, 1),
                        'reason': 'Bola kembali ke target player',
                    })
                    last_event_frame = frame_num
                    state = 'idle'
                    touch_frames = 0

                    if debug:
                        print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                              f"(bola kembali, dist={dist_target:.0f}px)")
                    continue

                # --- TIMEOUT ---
                if transit_frames > self.max_transit_frames:
                    event_id_counter += 1
                    touch_sec = round(touch_frames / self.fps, 2)
                    onetouch_events.append({
                        'event_id': event_id_counter,
                        'sender_id': sender_pid,
                        'receiver_id': -1,
                        'frame_kick': kick_frame,
                        'frame_start': possession_start_frame,
                        'frame_end': frame_num,
                        'success': False,
                        'touch_frames': touch_frames,
                        'touch_seconds': touch_sec,
                        'transit_frames': transit_frames,
                        'flight_seconds': round(transit_frames / self.fps, 2),
                        'closest_distance': round(closest_to_partner, 1),
                        'reason': f'Timeout transit {transit_frames}f',
                    })
                    last_event_frame = frame_num
                    state = 'idle'
                    touch_frames = 0

                    if debug:
                        print(f"[ONETOUCH] Frame {frame_num}: GAGAL ✗ "
                              f"(TIMEOUT {transit_frames}f)")
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
