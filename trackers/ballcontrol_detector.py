# ballcontrol_detector.py
# Ball Control Counting — State Machine
# SUKSES = bola diterima pemain dan tetap di dalam area 4 cone
# GAGAL  = bola keluar dari area 4 cone setelah diterima
# Model YOLO: 3 class (ball=0, cone=1, player=2)

import sys
sys.path.append('../')

from utils.bbox_utils import measure_distance, get_center_of_bbox, get_foot_position, stabilize_cone_positions
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple


class BallControlDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.ball_possession_distance: float = 150.0
        self.kick_away_distance: float = 150.0
        self.receive_distance: float = 200.0
        self.min_possession_frames: int = 3
        self.min_receive_frames: int = 2
        self.control_check_frames: int = 30
        self.zone_margin: float = 30.0
        self.max_transit_frames: int = 120
        self.cooldown_frames: int = 20
        self.min_away_frames: int = 5
        self.cone_stabilize_frames: int = 60
        self.player_separation_distance: float = 150.0

    # ============================================================
    # HELPERS
    # ============================================================

    def _get_ball_position(self, tracks: Dict, frame_num: int) -> Optional[Tuple[int, int]]:
        ball_data = tracks['ball'][frame_num].get(1)
        if ball_data is None or 'bbox' not in ball_data:
            return None
        return get_center_of_bbox(ball_data['bbox'])

    def _get_nearest_player(self, ball_pos, tracks, frame_num):
        best_pid, best_dist, best_center = -1, float('inf'), None
        for pid, pdata in tracks['players'][frame_num].items():
            bbox = pdata.get('bbox')
            if bbox is None:
                continue
            foot = get_foot_position(bbox)
            center = get_center_of_bbox(bbox)
            d = min(measure_distance(ball_pos, foot), measure_distance(ball_pos, center))
            if d < best_dist:
                best_dist, best_pid, best_center = d, pid, center
        return best_pid, best_dist, best_center

    def _identify_two_players(self, tracks, sample_frames=60):
        pid_counter = Counter()
        total = min(sample_frames, len(tracks['players']))
        for f in range(total):
            for pid in tracks['players'][f].keys():
                pid_counter[pid] += 1
        most_common = pid_counter.most_common(2)
        if len(most_common) < 2:
            return (most_common[0][0], -1) if most_common else (-1, -1)
        return most_common[0][0], most_common[1][0]

    def _get_player_avg_position(self, tracks, player_id, sample_frames=60):
        positions = []
        total = min(sample_frames, len(tracks['players']))
        for f in range(total):
            pdata = tracks['players'][f].get(player_id)
            if pdata and 'bbox' in pdata:
                positions.append(get_center_of_bbox(pdata['bbox']))
        if not positions:
            return None
        return (float(np.mean([p[0] for p in positions])),
                float(np.mean([p[1] for p in positions])))

    # ============================================================
    # CONE GROUPING: 4 cone per player → polygon zone
    # ============================================================

    def _order_polygon(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Urutkan 4 titik menjadi polygon (convex hull order)."""
        cx = np.mean([p[0] for p in points])
        cy = np.mean([p[1] for p in points])
        angles = [np.arctan2(p[1] - cy, p[0] - cx) for p in points]
        ordered = [p for _, p in sorted(zip(angles, points))]
        return ordered

    def _expand_polygon(self, polygon, margin):
        """Expand polygon outward by margin pixels."""
        cx = np.mean([p[0] for p in polygon])
        cy = np.mean([p[1] for p in polygon])
        expanded = []
        for px, py in polygon:
            dx, dy = px - cx, py - cy
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                expanded.append((px + margin * dx / length, py + margin * dy / length))
            else:
                expanded.append((px, py))
        return expanded

    def _point_in_polygon(self, point, polygon) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _deduplicate_cones(self, stabilized, min_dist=50.0, debug=True):
        """Merge cone yang terlalu dekat (duplikat dari ByteTrack ID berbeda)."""
        cone_list = [(cid, cpos) for cid, cpos in stabilized.items()]
        merged = {}
        used = set()

        for i, (cid_i, pos_i) in enumerate(cone_list):
            if cid_i in used:
                continue
            # Cari semua cone yang dekat dengan cone ini
            group_ids = [cid_i]
            group_positions = [pos_i]
            for j, (cid_j, pos_j) in enumerate(cone_list):
                if j <= i or cid_j in used:
                    continue
                if measure_distance(pos_i, pos_j) < min_dist:
                    group_ids.append(cid_j)
                    group_positions.append(pos_j)
                    used.add(cid_j)
            used.add(cid_i)
            # Merge: rata-rata posisi, pakai ID terkecil
            avg_x = float(np.mean([p[0] for p in group_positions]))
            avg_y = float(np.mean([p[1] for p in group_positions]))
            merged[min(group_ids)] = (avg_x, avg_y)
            if debug and len(group_ids) > 1:
                print(f"[BALLCTRL] DEDUP: Merged cones {group_ids} → "
                      f"Cone {min(group_ids)} ({avg_x:.0f}, {avg_y:.0f})")

        return merged

    def _stabilize_and_group_cones(self, tracks, player_a_id, player_b_id, debug=True):
        """Stabilkan 8 cone, deduplikasi, kelompokkan 4 per pemain, buat polygon."""
        stabilized = stabilize_cone_positions(
            tracks, cone_key='cones', sample_frames=self.cone_stabilize_frames)

        if debug:
            print(f"[BALLCTRL] Raw stabilized cones ({len(stabilized)}):")
            for cid, cpos in stabilized.items():
                print(f"  Cone {cid}: ({cpos[0]:.0f}, {cpos[1]:.0f})")

        # Deduplicate nearby cones
        stabilized = self._deduplicate_cones(stabilized, min_dist=50.0, debug=debug)

        if debug:
            print(f"[BALLCTRL] After dedup: {len(stabilized)} unique cones")
            for cid, cpos in stabilized.items():
                print(f"  Cone {cid}: ({cpos[0]:.0f}, {cpos[1]:.0f})")

        if len(stabilized) < 8:
            if debug:
                print(f"[BALLCTRL] WARNING: Hanya {len(stabilized)} cone unik (butuh 8)")
            if len(stabilized) < 4:
                return None, None

        pos_a = self._get_player_avg_position(tracks, player_a_id)
        pos_b = self._get_player_avg_position(tracks, player_b_id)
        if pos_a is None or pos_b is None:
            return None, None

        if debug:
            print(f"[BALLCTRL] Player A (ID={player_a_id}) avg pos: ({pos_a[0]:.0f}, {pos_a[1]:.0f})")
            print(f"[BALLCTRL] Player B (ID={player_b_id}) avg pos: ({pos_b[0]:.0f}, {pos_b[1]:.0f})")

        # Hitung jarak setiap cone ke kedua pemain
        all_cones = []
        for cid, cpos in stabilized.items():
            da = measure_distance(cpos, pos_a)
            db = measure_distance(cpos, pos_b)
            all_cones.append({'id': cid, 'pos': cpos, 'dist_a': da, 'dist_b': db})

        # Sort semua cone berdasarkan jarak ke player A
        all_cones.sort(key=lambda c: c['dist_a'])
        # Ambil 4 terdekat ke A
        cones_a = all_cones[:4]
        # Sisanya sort berdasarkan jarak ke player B, ambil 4 terdekat
        remaining = all_cones[4:]
        remaining.sort(key=lambda c: c['dist_b'])
        cones_b = remaining[:4]

        if len(cones_a) < 4 or len(cones_b) < 4:
            print("[BALLCTRL] ERROR: Tidak cukup cone unik (butuh 4 per pemain)!")
            return None, None

        # Create ordered polygons
        zone_a_pts = self._order_polygon([c['pos'] for c in cones_a])
        zone_b_pts = self._order_polygon([c['pos'] for c in cones_b])

        # Expand with margin
        zone_a_expanded = self._expand_polygon(zone_a_pts, self.zone_margin)
        zone_b_expanded = self._expand_polygon(zone_b_pts, self.zone_margin)

        if debug:
            print(f"\n[BALLCTRL] Zone A ({len(cones_a)} cones):")
            for c in cones_a:
                print(f"  Cone {c['id']}: ({c['pos'][0]:.0f}, {c['pos'][1]:.0f})")
            print(f"[BALLCTRL] Zone B ({len(cones_b)} cones):")
            for c in cones_b:
                print(f"  Cone {c['id']}: ({c['pos'][0]:.0f}, {c['pos'][1]:.0f})")

        return zone_a_expanded, zone_b_expanded

    # ============================================================
    # MAIN DETECTION
    # ============================================================

    def detect_ball_controls(self, tracks, debug=True):
        total_frames = len(tracks['players'])
        if total_frames == 0:
            return []

        player_a, player_b = self._identify_two_players(tracks)
        zone_a, zone_b = self._stabilize_and_group_cones(
            tracks, player_a, player_b, debug=debug)

        if zone_a is None or zone_b is None:
            print("[BALLCTRL] ERROR: Tidak bisa membentuk zone! Abort.")
            return []

        # Store zones for visualization
        self._zone_a = zone_a
        self._zone_b = zone_b
        self._player_a = player_a
        self._player_b = player_b

        if debug:
            print(f"\n[BALLCTRL] === BALL CONTROL DETECTION ===")
            print(f"[BALLCTRL] Total frames: {total_frames}, FPS: {self.fps}")
            print(f"[BALLCTRL] Player A: {player_a}, Player B: {player_b}")
            print(f"[BALLCTRL] Possession dist: {self.ball_possession_distance}px")
            print(f"[BALLCTRL] Control check: {self.control_check_frames}f")
            print(f"[BALLCTRL] Zone margin: {self.zone_margin}px")

        state = 'idle'
        sender_id = -1
        sender_pos = None
        kick_frame = -1
        possession_frames = 0
        current_possessor = -1
        last_event_frame = -999
        away_frames = self.min_away_frames
        transit_frames = 0
        receive_frames = 0
        receiver_id = -1
        receiver_zone = None
        control_start_frame = -1
        control_frames = 0
        ball_exited = False
        exit_frame = -1

        events = []
        event_id = 0

        for frame_num in range(total_frames):
            if frame_num % 200 == 0 and debug:
                print(f"[BALLCTRL] Frame {frame_num}/{total_frames} [state={state}]")

            ball_pos = self._get_ball_position(tracks, frame_num)
            if ball_pos is None:
                if state == 'ball_transit':
                    transit_frames += 1
                    if transit_frames > self.max_transit_frames:
                        event_id += 1
                        events.append({
                            'event_id': event_id, 'sender_id': sender_id,
                            'receiver_id': -1, 'frame_kick': kick_frame,
                            'frame_start': kick_frame, 'frame_end': frame_num,
                            'success': False, 'transit_frames': transit_frames,
                            'flight_seconds': round(transit_frames/self.fps, 2),
                            'reason': 'Timeout — bola hilang',
                        })
                        last_event_frame = frame_num
                        state = 'idle'
                        if debug:
                            print(f"[BALLCTRL] F{frame_num}: GAGAL (timeout bola hilang)")
                continue

            nearest_pid, nearest_dist, nearest_center = self._get_nearest_player(
                ball_pos, tracks, frame_num)

            # === STATE: IDLE ===
            if state == 'idle':
                if (frame_num - last_event_frame) < self.cooldown_frames:
                    continue
                if away_frames < self.min_away_frames:
                    if nearest_dist > self.ball_possession_distance:
                        away_frames += 1
                    continue

                if nearest_dist <= self.ball_possession_distance and nearest_pid != -1:
                    if current_possessor == nearest_pid:
                        possession_frames += 1
                    else:
                        current_possessor = nearest_pid
                        possession_frames = 1
                else:
                    possession_frames = 0
                    current_possessor = -1

                if possession_frames >= self.min_possession_frames and current_possessor != -1:
                    sender_id = current_possessor
                    state = 'waiting_kick'
                    if debug:
                        print(f"[BALLCTRL] F{frame_num}: P{sender_id} possession")

            # === STATE: WAITING_KICK ===
            elif state == 'waiting_kick':
                dist_sender = float('inf')
                for pid, pdata in tracks['players'][frame_num].items():
                    bbox = pdata.get('bbox')
                    if bbox is None:
                        continue
                    center = get_center_of_bbox(bbox)
                    if pid == sender_id or (sender_pos and measure_distance(center, sender_pos) < self.player_separation_distance):
                        foot = get_foot_position(bbox)
                        d = min(measure_distance(ball_pos, foot),
                                measure_distance(ball_pos, center))
                        if d < dist_sender:
                            dist_sender = d
                            sender_pos = center

                if dist_sender == float('inf'):
                    if nearest_dist <= self.ball_possession_distance:
                        sender_id = nearest_pid
                        sender_pos = nearest_center
                        dist_sender = nearest_dist
                    else:
                        dist_sender = self.kick_away_distance + 1

                if dist_sender > self.kick_away_distance:
                    state = 'ball_transit'
                    kick_frame = frame_num
                    transit_frames = 0
                    receive_frames = 0
                    if debug:
                        print(f"[BALLCTRL] F{frame_num}: KICK! P{sender_id}")
                elif dist_sender <= self.ball_possession_distance:
                    if nearest_pid != sender_id and nearest_dist < dist_sender:
                        if nearest_dist <= self.ball_possession_distance:
                            state = 'idle'
                            possession_frames = 0
                            away_frames = self.min_away_frames

            # === STATE: BALL_TRANSIT ===
            elif state == 'ball_transit':
                transit_frames += 1

                # Find receiver (player far from sender)
                recv_pid, recv_dist, recv_center = -1, float('inf'), None
                for pid, pdata in tracks['players'][frame_num].items():
                    bbox = pdata.get('bbox')
                    if bbox is None:
                        continue
                    center = get_center_of_bbox(bbox)
                    if sender_pos and measure_distance(center, sender_pos) < self.player_separation_distance:
                        continue
                    foot = get_foot_position(bbox)
                    d = min(measure_distance(ball_pos, foot),
                            measure_distance(ball_pos, center))
                    if d < recv_dist:
                        recv_dist, recv_pid, recv_center = d, pid, center

                if recv_dist <= self.receive_distance and recv_pid != -1:
                    receive_frames += 1
                    if receive_frames >= self.min_receive_frames:
                        receiver_id = recv_pid
                        # Determine receiver zone
                        pos_a = self._get_player_avg_position(tracks, player_a)
                        pos_b = self._get_player_avg_position(tracks, player_b)
                        if recv_center and pos_a and pos_b:
                            da = measure_distance(recv_center, pos_a)
                            db = measure_distance(recv_center, pos_b)
                            receiver_zone = zone_a if da < db else zone_b
                        else:
                            receiver_zone = zone_b if sender_id == player_a else zone_a

                        state = 'checking_control'
                        control_start_frame = frame_num
                        control_frames = 0
                        ball_exited = False
                        exit_frame = -1
                        if debug:
                            print(f"[BALLCTRL] F{frame_num}: RECEIVED by P{recv_pid}, checking control...")
                else:
                    receive_frames = 0

                # Timeout
                if transit_frames > self.max_transit_frames:
                    event_id += 1
                    events.append({
                        'event_id': event_id, 'sender_id': sender_id,
                        'receiver_id': -1, 'frame_kick': kick_frame,
                        'frame_start': kick_frame, 'frame_end': frame_num,
                        'success': False, 'transit_frames': transit_frames,
                        'flight_seconds': round(transit_frames/self.fps, 2),
                        'reason': f'Timeout transit {transit_frames}f',
                    })
                    last_event_frame = frame_num
                    away_frames = 0
                    state = 'idle'
                    possession_frames = 0
                    if debug:
                        print(f"[BALLCTRL] F{frame_num}: GAGAL (timeout {transit_frames}f)")

                if debug and transit_frames % 15 == 0 and state == 'ball_transit':
                    print(f"[BALLCTRL]   transit f={frame_num}: recv={recv_pid} dist={recv_dist:.0f}")

            # === STATE: CHECKING_CONTROL ===
            elif state == 'checking_control':
                control_frames += 1
                in_zone = self._point_in_polygon(ball_pos, receiver_zone)

                if not in_zone and not ball_exited:
                    ball_exited = True
                    exit_frame = frame_num
                    if debug:
                        print(f"[BALLCTRL] F{frame_num}: BOLA KELUAR ZONE!")

                if control_frames >= self.control_check_frames:
                    event_id += 1
                    success = not ball_exited
                    reason = 'Bola tetap di dalam zone' if success else f'Bola keluar zone di frame {exit_frame}'
                    events.append({
                        'event_id': event_id, 'sender_id': sender_id,
                        'receiver_id': receiver_id, 'frame_kick': kick_frame,
                        'frame_start': kick_frame, 'frame_end': frame_num,
                        'success': success,
                        'transit_frames': control_start_frame - kick_frame,
                        'flight_seconds': round((control_start_frame - kick_frame)/self.fps, 2),
                        'control_frames': control_frames,
                        'ball_exited': ball_exited,
                        'exit_frame': exit_frame if ball_exited else -1,
                        'reason': reason,
                    })
                    last_event_frame = frame_num
                    away_frames = 0
                    state = 'idle'
                    possession_frames = 0
                    status = "SUKSES ✓" if success else "GAGAL ✗"
                    if debug:
                        print(f"[BALLCTRL] F{frame_num}: {status} ({reason})")

        if debug:
            s = sum(1 for e in events if e['success'])
            g = sum(1 for e in events if not e['success'])
            print(f"\n[BALLCTRL] === HASIL === Total:{len(events)} S:{s} G:{g}")

        return events

    # ============================================================
    # PUBLIC GETTERS
    # ============================================================

    def get_zones(self, tracks):
        player_a, player_b = self._identify_two_players(tracks)
        zone_a, zone_b = self._stabilize_and_group_cones(
            tracks, player_a, player_b, debug=False)
        return player_a, player_b, zone_a, zone_b

    def debug_distances(self, tracks, sample_every=10):
        total = len(tracks['players'])
        print(f"\n[DEBUG] === JARAK BOLA-PEMAIN (/{sample_every}f) ===")
        for f in range(0, total, sample_every):
            bp = self._get_ball_position(tracks, f)
            if bp is None:
                continue
            pid, dist, _ = self._get_nearest_player(bp, tracks, f)
            players = []
            for p, d in tracks['players'][f].items():
                bbox = d.get('bbox')
                if bbox:
                    dd = min(measure_distance(bp, get_foot_position(bbox)),
                             measure_distance(bp, get_center_of_bbox(bbox)))
                    players.append(f"P{p}:{dd:.0f}")
            print(f"  F{f} ball=({bp[0]},{bp[1]}) near=P{pid}:{dist:.0f} {','.join(players)}")

    def get_ballcontrol_statistics(self, events):
        total = len(events)
        sukses = [e for e in events if e['success']]
        gagal = [e for e in events if not e['success']]
        player_stats = {}
        for e in events:
            rid = e.get('receiver_id', -1)
            if rid not in player_stats:
                player_stats[rid] = {'total': 0, 'sukses': 0, 'gagal': 0}
            player_stats[rid]['total'] += 1
            if e['success']:
                player_stats[rid]['sukses'] += 1
            else:
                player_stats[rid]['gagal'] += 1

        return {
            'total_ballcontrol': total,
            'successful_ballcontrol': len(sukses),
            'failed_ballcontrol': len(gagal),
            'accuracy_pct': round(len(sukses)/total*100, 1) if total > 0 else 0.0,
            'avg_flight_success': round(float(np.mean(
                [e['flight_seconds'] for e in sukses])), 2) if sukses else 0.0,
            'player_stats': player_stats,
        }
