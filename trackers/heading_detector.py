# heading_detector.py
# Deteksi dan counting heading bola.
# SUKSES = bbox bola overlap/menyentuh bbox kepala (class "Heading" dari YOLO)
# GAGAL  = bola dekat pemain tapi TIDAK menyentuh bbox kepala

import sys
sys.path.append('../')

from utils.bbox_utils import measure_distance, get_center_of_bbox
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class HeadingDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps

        # ============================================================
        # PARAMETER KONTAK BOLA-KEPALA
        # ============================================================
        # Margin tambahan (px) di sekitar bbox kepala untuk toleransi overlap
        self.head_bbox_margin: float = 15.0

        # Jarak maksimum center bola ke center kepala (fallback jika tidak overlap)
        self.max_head_ball_center_distance: float = 60.0

        # ============================================================
        # PARAMETER HEADING ATTEMPT (kapan dianggap "mencoba heading")
        # ============================================================
        # Jarak maksimum bola ke center pemain untuk dianggap "approach"
        self.max_approach_distance: float = 180.0

        # Minimum frame bola dekat pemain sebelum dianggap attempt
        self.min_approach_frames: int = 3

        # Cooldown setelah heading terdeteksi (frame)
        self.cooldown_frames: int = 25

        # Maksimum durasi approach sebelum reset (detik)
        self.max_approach_duration_sec: float = 3.0

        # Minimum frame bola harus jauh sebelum attempt baru
        self.min_away_frames: int = 8

        # ============================================================
        # INTERNAL STATE
        # ============================================================
        self._heading_events: List[Dict] = []

    # ============================================================
    # HELPER: CEK OVERLAP / KONTAK BBOX
    # ============================================================

    def _bbox_overlap(
        self,
        bbox_a: List[float],
        bbox_b: List[float],
        margin: float = 0.0
    ) -> bool:
        """
        Cek apakah dua bounding box overlap (dengan margin tambahan).
        bbox format: [x1, y1, x2, y2]
        """
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        # Expand bbox_b dengan margin
        bx1 -= margin
        by1 -= margin
        bx2 += margin
        by2 += margin

        # Cek overlap
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    def _compute_iou(
        self,
        bbox_a: List[float],
        bbox_b: List[float]
    ) -> float:
        """Hitung Intersection over Union antara dua bbox."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _ball_touches_head(
        self,
        ball_bbox: List[float],
        head_bbox: List[float]
    ) -> Tuple[bool, float, float]:
        """
        Cek apakah bola menyentuh kepala.

        Returns:
            (is_touching, distance_centers, iou)
        """
        ball_center = get_center_of_bbox(ball_bbox)
        head_center = get_center_of_bbox(head_bbox)
        dist = measure_distance(ball_center, head_center)
        iou = self._compute_iou(ball_bbox, head_bbox)

        # Kontak jika:
        # 1. Bbox overlap (dengan margin), ATAU
        # 2. Jarak center cukup dekat
        is_overlap = self._bbox_overlap(ball_bbox, head_bbox, self.head_bbox_margin)
        is_close = dist <= self.max_head_ball_center_distance

        is_touching = is_overlap or is_close

        return is_touching, dist, iou

    def _find_nearest_head_to_player(
        self,
        player_bbox: List[float],
        heads_in_frame: Dict[int, Dict]
    ) -> Optional[Tuple[int, List[float]]]:
        """
        Cari bbox kepala (class Heading) yang paling dekat dengan pemain.
        Kepala harus berada di bagian atas bbox pemain.
        """
        if not heads_in_frame:
            return None

        player_center = get_center_of_bbox(player_bbox)
        px1, py1, px2, py2 = player_bbox
        player_top_half_y = py1 + (py2 - py1) * 0.5  # setengah atas pemain

        best_head_id = None
        best_head_bbox = None
        best_dist = float('inf')

        for head_id, head_data in heads_in_frame.items():
            head_bbox = head_data.get('bbox')
            if head_bbox is None:
                continue

            head_center = get_center_of_bbox(head_bbox)
            hcx, hcy = head_center

            # Kepala harus di bagian atas bbox pemain
            if hcy > player_top_half_y:
                continue

            # Kepala harus overlap secara horizontal dengan pemain
            if hcx < px1 - 30 or hcx > px2 + 30:
                continue

            dist = measure_distance(player_center, head_center)
            if dist < best_dist:
                best_dist = dist
                best_head_id = head_id
                best_head_bbox = head_bbox

        if best_head_id is not None:
            return best_head_id, best_head_bbox

        return None

    # ============================================================
    # DETEKSI HEADING — CORE LOGIC
    # ============================================================

    def detect_headings(
        self,
        tracks: Dict,
        debug: bool = True
    ) -> List[Dict]:
        """
        Deteksi semua heading events.

        Logic sederhana:
        1. Setiap frame, cari pasangan player ↔ head (class Heading)
        2. Cek apakah bola overlap/menyentuh bbox kepala
        3. State machine: idle → approaching → contact → resolved

        Sukses = bola mengenai bbox kepala
        Gagal  = bola dekat pemain tapi tidak mengenai kepala
        """
        total_frames = len(tracks['players'])
        if total_frames == 0:
            return []

        has_heading_key = 'heading' in tracks and len(tracks['heading']) == total_frames
        max_approach_frames = int(self.max_approach_duration_sec * self.fps)

        if debug:
            print(f"\n[HEADING] === HEADING DETECTION ===")
            print(f"[HEADING] Total frames            : {total_frames}")
            print(f"[HEADING] FPS                     : {self.fps}")
            print(f"[HEADING] Head bbox margin        : {self.head_bbox_margin}px")
            print(f"[HEADING] Max head-ball center dist: {self.max_head_ball_center_distance}px")
            print(f"[HEADING] Approach distance        : {self.max_approach_distance}px")
            print(f"[HEADING] Cooldown frames          : {self.cooldown_frames}")
            print(f"[HEADING] Class 'Heading' tersedia : {'Ya' if has_heading_key else 'Tidak'}")
            print(f"[HEADING] ================================\n")

        if not has_heading_key:
            print("[HEADING] ERROR: tracks['heading'] tidak ditemukan!")
            print("[HEADING] Pastikan Tracker mendeteksi class 'Heading' (index 0).")
            return []

        # State per player
        player_states: Dict[int, Dict[str, Any]] = {}
        heading_events: List[Dict] = []
        event_id_counter = 0

        for frame_num in range(total_frames):
            if frame_num % 200 == 0 and debug:
                print(f"[HEADING] Processing frame {frame_num}/{total_frames}...")

            # --- Data frame ini ---
            ball_data = tracks['ball'][frame_num].get(1)
            heads_in_frame = tracks['heading'][frame_num]
            players = tracks['players'][frame_num]

            ball_bbox = None
            ball_center = None
            if ball_data and 'bbox' in ball_data:
                ball_bbox = ball_data['bbox']
                ball_center = get_center_of_bbox(ball_bbox)

            # --- Proses setiap pemain ---
            for player_id, player_data in players.items():
                player_bbox = player_data.get('bbox')
                if player_bbox is None:
                    continue

                # Inisialisasi state
                if player_id not in player_states:
                    player_states[player_id] = {
                        'state': 'idle',
                        'approach_start_frame': -1,
                        'approach_frames': 0,
                        'last_event_frame': -999,
                        'away_frames': self.min_away_frames,
                        'head_touch_frames': 0,
                        'closest_head_dist': float('inf'),
                        'best_iou': 0.0,
                        'matched_head_id': None,
                    }

                ps = player_states[player_id]

                # Cari kepala yang cocok dengan pemain ini
                head_match = self._find_nearest_head_to_player(
                    player_bbox, heads_in_frame
                )
                head_bbox = head_match[1] if head_match else None
                head_id = head_match[0] if head_match else None

                # Cek kontak bola-kepala
                ball_touches_head = False
                head_ball_dist = float('inf')
                head_ball_iou = 0.0

                if ball_bbox is not None and head_bbox is not None:
                    ball_touches_head, head_ball_dist, head_ball_iou = \
                        self._ball_touches_head(ball_bbox, head_bbox)

                # Cek bola dekat pemain
                ball_near_player = False
                if ball_center is not None:
                    player_center = get_center_of_bbox(player_bbox)
                    dist_ball_player = measure_distance(ball_center, player_center)
                    ball_near_player = dist_ball_player <= self.max_approach_distance

                # ======== STATE MACHINE ========

                if ps['state'] == 'idle':
                    # Cooldown check
                    if (frame_num - ps['last_event_frame']) < self.cooldown_frames:
                        continue

                    # Bola harus sudah cukup jauh sebelumnya
                    if ps['away_frames'] < self.min_away_frames:
                        if not ball_near_player:
                            ps['away_frames'] += 1
                        continue

                    # Bola mendekat ke pemain?
                    if ball_near_player and ball_bbox is not None:
                        ps['state'] = 'approaching'
                        ps['approach_start_frame'] = frame_num
                        ps['approach_frames'] = 1
                        ps['head_touch_frames'] = 1 if ball_touches_head else 0
                        ps['closest_head_dist'] = head_ball_dist
                        ps['best_iou'] = head_ball_iou
                        ps['matched_head_id'] = head_id
                    else:
                        if not ball_near_player:
                            ps['away_frames'] += 1

                elif ps['state'] == 'approaching':
                    ps['approach_frames'] += 1

                    # Update tracking
                    if head_ball_dist < ps['closest_head_dist']:
                        ps['closest_head_dist'] = head_ball_dist
                    if head_ball_iou > ps['best_iou']:
                        ps['best_iou'] = head_ball_iou
                    if ball_touches_head:
                        ps['head_touch_frames'] += 1

                    # --- CEK KONTAK BOLA-KEPALA → SUKSES ---
                    if ball_touches_head:
                        event_id_counter += 1
                        event = self._create_event(
                            event_id=event_id_counter,
                            player_id=player_id,
                            frame_start=ps['approach_start_frame'],
                            frame_contact=frame_num,
                            success=True,
                            head_ball_dist=head_ball_dist,
                            iou=head_ball_iou,
                            head_touch_frames=ps['head_touch_frames'],
                            head_bbox=head_bbox,
                            ball_bbox=ball_bbox,
                        )
                        heading_events.append(event)

                        if debug:
                            print(f"[HEADING] Frame {frame_num}: HEADING SUKSES ✓ "
                                  f"(Player {player_id}, "
                                  f"dist={head_ball_dist:.0f}px, "
                                  f"IoU={head_ball_iou:.2f})")

                        ps['state'] = 'idle'
                        ps['last_event_frame'] = frame_num
                        ps['away_frames'] = 0
                        continue

                    # --- BOLA DEKAT PEMAIN TAPI TIDAK KENA KEPALA ---
                    # Jika bola sangat dekat badan (di bawah kepala) → GAGAL
                    if (ball_near_player and
                        ball_center is not None and
                        head_bbox is not None and
                        ps['approach_frames'] >= self.min_approach_frames):

                        # Bola di bawah area kepala = kena badan
                        head_bottom = head_bbox[3]  # y2 dari head bbox
                        ball_cy = ball_center[1]

                        if ball_cy > head_bottom and dist_ball_player < 100:
                            event_id_counter += 1
                            event = self._create_event(
                                event_id=event_id_counter,
                                player_id=player_id,
                                frame_start=ps['approach_start_frame'],
                                frame_contact=frame_num,
                                success=False,
                                head_ball_dist=head_ball_dist,
                                iou=head_ball_iou,
                                head_touch_frames=ps['head_touch_frames'],
                                head_bbox=head_bbox,
                                ball_bbox=ball_bbox,
                            )
                            heading_events.append(event)

                            if debug:
                                print(f"[HEADING] Frame {frame_num}: HEADING GAGAL ✗ "
                                      f"(Player {player_id}, bola kena badan, "
                                      f"head_dist={head_ball_dist:.0f}px)")

                            ps['state'] = 'idle'
                            ps['last_event_frame'] = frame_num
                            ps['away_frames'] = 0
                            continue

                    # --- TIMEOUT ---
                    if ps['approach_frames'] > max_approach_frames:
                        if debug:
                            print(f"[HEADING] Frame {frame_num}: TIMEOUT "
                                  f"(Player {player_id})")
                        ps['state'] = 'idle'
                        ps['away_frames'] = 0
                        continue

                    # --- BOLA MENJAUH → RESOLVE ---
                    if not ball_near_player:
                        if ps['approach_frames'] >= self.min_approach_frames:
                            # Pernah kontak kepala?
                            success = ps['head_touch_frames'] >= 2

                            event_id_counter += 1
                            event = self._create_event(
                                event_id=event_id_counter,
                                player_id=player_id,
                                frame_start=ps['approach_start_frame'],
                                frame_contact=frame_num,
                                success=success,
                                head_ball_dist=ps['closest_head_dist'],
                                iou=ps['best_iou'],
                                head_touch_frames=ps['head_touch_frames'],
                                head_bbox=head_bbox,
                                ball_bbox=ball_bbox,
                            )
                            heading_events.append(event)

                            status = "SUKSES ✓" if success else "GAGAL ✗"
                            if debug:
                                print(f"[HEADING] Frame {frame_num}: {status} "
                                      f"(Player {player_id}, bola menjauh, "
                                      f"head_touch={ps['head_touch_frames']})")

                            ps['last_event_frame'] = frame_num

                        ps['state'] = 'idle'
                        ps['away_frames'] = 0

        self._heading_events = heading_events

        if debug:
            sukses = sum(1 for e in heading_events if e['success'])
            gagal = sum(1 for e in heading_events if not e['success'])
            total = len(heading_events)
            print(f"\n[HEADING] === HASIL AKHIR ===")
            print(f"[HEADING] Total heading   : {total}")
            print(f"[HEADING] SUKSES          : {sukses}")
            print(f"[HEADING] GAGAL           : {gagal}")
            if total > 0:
                print(f"[HEADING] Akurasi         : {sukses/total*100:.1f}%")
            print(f"[HEADING] =====================\n")

        return heading_events

    def _create_event(
        self,
        event_id: int,
        player_id: int,
        frame_start: int,
        frame_contact: int,
        success: bool,
        head_ball_dist: float,
        iou: float,
        head_touch_frames: int,
        head_bbox: Optional[List[float]],
        ball_bbox: Optional[List[float]],
    ) -> Dict:
        return {
            'event_id': event_id,
            'player_id': player_id,
            'frame_start': frame_start,
            'frame_contact': frame_contact,
            'frame_end': frame_contact,
            'success': success,
            'head_ball_distance': round(head_ball_dist, 1),
            'iou': round(iou, 3),
            'head_touch_frames': head_touch_frames,
            'head_bbox': head_bbox,
            'ball_bbox': ball_bbox,
            'duration_frames': frame_contact - frame_start,
            'duration_seconds': round(
                (frame_contact - frame_start) / self.fps, 2
            ),
        }

    # ============================================================
    # STATISTIK
    # ============================================================

    def get_heading_statistics(self, events: List[Dict]) -> Dict:
        total = len(events)
        sukses = [e for e in events if e['success']]
        gagal = [e for e in events if not e['success']]

        player_stats: Dict[int, Dict] = {}
        for e in events:
            pid = e['player_id']
            if pid not in player_stats:
                player_stats[pid] = {'total': 0, 'sukses': 0, 'gagal': 0}
            player_stats[pid]['total'] += 1
            if e['success']:
                player_stats[pid]['sukses'] += 1
            else:
                player_stats[pid]['gagal'] += 1

        avg_dist_success = (
            round(float(np.mean([e['head_ball_distance'] for e in sukses])), 1)
            if sukses else 0.0
        )
        avg_iou_success = (
            round(float(np.mean([e['iou'] for e in sukses])), 3)
            if sukses else 0.0
        )

        return {
            'total_headings': total,
            'successful_headings': len(sukses),
            'failed_headings': len(gagal),
            'accuracy_pct': round(len(sukses) / total * 100, 1) if total > 0 else 0.0,
            'avg_dist_success': avg_dist_success,
            'avg_iou_success': avg_iou_success,
            'player_stats': player_stats,
        }
