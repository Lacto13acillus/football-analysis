# heading_detector.py
# ============================================================
# Logic sederhana & akurat:
#   SUKSES = bbox ball overlap/menyentuh bbox Heading (kepala)
#   GAGAL  = bola dekat pemain tapi TIDAK mengenai bbox Heading
#
# Tidak perlu match head→player, langsung cek ball↔head overlap.
# ============================================================

import sys
sys.path.append('../')

from utils.bbox_utils import measure_distance, get_center_of_bbox
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class HeadingDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps

        # ============================================================
        # PARAMETER KONTAK BOLA ↔ KEPALA
        # ============================================================
        # Margin (px) di sekitar bbox kepala untuk toleransi overlap
        self.head_bbox_margin: float = 20.0

        # Jarak maks center bola → center kepala (fallback)
        self.max_head_ball_center_distance: float = 80.0

        # ============================================================
        # PARAMETER ATTEMPT DETECTION
        # ============================================================
        # Jarak maks bola → center pemain untuk dianggap "mendekati"
        self.max_approach_distance: float = 200.0

        # Cooldown setelah 1 heading event (frame), hindari double count
        self.cooldown_frames: int = 25

        # Min frame bola harus jauh dari semua kepala sebelum attempt baru
        self.min_away_frames: int = 5

        # Maks durasi pendekatan sebelum auto-reset (detik)
        self.max_approach_duration_sec: float = 4.0

        # Min frame approach sebelum bisa dianggap attempt
        self.min_approach_frames: int = 2

    # ============================================================
    # HELPER: OVERLAP & DISTANCE
    # ============================================================

    def _bbox_overlap(
        self,
        bbox_a: List[float],
        bbox_b: List[float],
        margin: float = 0.0
    ) -> bool:
        """Cek apakah dua bbox overlap (+ margin pada bbox_b)."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        bx1 -= margin
        by1 -= margin
        bx2 += margin
        by2 += margin

        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    def _compute_iou(
        self,
        bbox_a: List[float],
        bbox_b: List[float]
    ) -> float:
        """Intersection over Union."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def _ball_touches_any_head(
        self,
        ball_bbox: List[float],
        heads_in_frame: Dict[int, Dict],
    ) -> Tuple[bool, int, float, float]:
        """
        Cek apakah bola menyentuh SALAH SATU bbox kepala di frame ini.

        Returns:
            (is_touching, best_head_id, distance, iou)
        """
        if not heads_in_frame:
            return False, -1, float('inf'), 0.0

        ball_center = get_center_of_bbox(ball_bbox)

        best_touching = False
        best_head_id = -1
        best_dist = float('inf')
        best_iou = 0.0

        for head_id, head_data in heads_in_frame.items():
            head_bbox = head_data.get('bbox')
            if head_bbox is None:
                continue

            head_center = get_center_of_bbox(head_bbox)
            dist = measure_distance(ball_center, head_center)
            iou = self._compute_iou(ball_bbox, head_bbox)

            is_overlap = self._bbox_overlap(
                ball_bbox, head_bbox, self.head_bbox_margin
            )
            is_close = dist <= self.max_head_ball_center_distance

            touching = is_overlap or is_close

            # Pilih kepala terdekat
            if dist < best_dist:
                best_dist = dist
                best_head_id = head_id
                best_iou = iou
                best_touching = touching

        return best_touching, best_head_id, best_dist, best_iou

    def _ball_near_any_head(
        self,
        ball_bbox: List[float],
        heads_in_frame: Dict[int, Dict],
    ) -> Tuple[bool, float]:
        """
        Cek apakah bola berada dekat dengan salah satu kepala
        (zona approach, lebih besar dari zona contact).
        """
        if not heads_in_frame:
            return False, float('inf')

        ball_center = get_center_of_bbox(ball_bbox)
        min_dist = float('inf')

        for head_id, head_data in heads_in_frame.items():
            head_bbox = head_data.get('bbox')
            if head_bbox is None:
                continue
            head_center = get_center_of_bbox(head_bbox)
            dist = measure_distance(ball_center, head_center)
            if dist < min_dist:
                min_dist = dist

        return min_dist <= self.max_approach_distance, min_dist

    # ============================================================
    # DETEKSI HEADING — CORE LOGIC
    # ============================================================

    def detect_headings(
        self,
        tracks: Dict,
        debug: bool = True
    ) -> List[Dict]:
        """
        Deteksi heading langsung dari overlap bbox ball ↔ bbox Heading.

        Logic (sangat sederhana):
        - Setiap frame, cek apakah bbox bola overlap dengan bbox kepala manapun
        - Jika overlap → HEADING SUKSES
        - Jika bola dekat pemain tapi tidak overlap kepala → HEADING GAGAL
        - Cooldown untuk hindari double counting
        """
        total_frames = len(tracks['players'])
        if total_frames == 0:
            return []

        has_heading = 'heading' in tracks and len(tracks['heading']) == total_frames
        max_approach_frames = int(self.max_approach_duration_sec * self.fps)

        if debug:
            print(f"\n[HEADING] === HEADING DETECTION ===")
            print(f"[HEADING] Total frames              : {total_frames}")
            print(f"[HEADING] FPS                       : {self.fps}")
            print(f"[HEADING] Head bbox margin          : {self.head_bbox_margin}px")
            print(f"[HEADING] Max head-ball center dist  : {self.max_head_ball_center_distance}px")
            print(f"[HEADING] Approach distance          : {self.max_approach_distance}px")
            print(f"[HEADING] Cooldown frames            : {self.cooldown_frames}")
            print(f"[HEADING] Min away frames            : {self.min_away_frames}")
            print(f"[HEADING] Class 'Heading' tersedia   : {'Ya' if has_heading else 'Tidak'}")
            print(f"[HEADING] ================================\n")

        if not has_heading:
            print("[HEADING] ERROR: tracks['heading'] tidak ditemukan!")
            return []

        # ============================================================
        # SIMPLE STATE MACHINE (global, bukan per-player)
        # ============================================================
        # Karena ini cek langsung ball↔head, tidak perlu per-player state.
        # Cukup global state: idle → approaching → resolved
        # ============================================================

        state = 'idle'
        approach_start_frame = -1
        approach_frames = 0
        away_frames = self.min_away_frames  # mulai bisa langsung
        last_event_frame = -999
        closest_head_dist_during_approach = float('inf')
        best_head_id_during_approach = -1

        heading_events: List[Dict] = []
        event_id_counter = 0

        for frame_num in range(total_frames):
            if frame_num % 200 == 0 and debug:
                print(f"[HEADING] Processing frame {frame_num}/{total_frames}...")

            # --- Data bola ---
            ball_data = tracks['ball'][frame_num].get(1)
            if ball_data is None or 'bbox' not in ball_data:
                # Bola tidak ada
                if state == 'idle':
                    away_frames += 1
                continue

            ball_bbox = ball_data['bbox']

            # --- Data kepala (class Heading) ---
            heads_in_frame = tracks['heading'][frame_num]

            # --- Cek kontak bola-kepala ---
            touching, touch_head_id, touch_dist, touch_iou = \
                self._ball_touches_any_head(ball_bbox, heads_in_frame)

            # --- Cek bola dekat kepala (approach zone) ---
            near_head, near_dist = self._ball_near_any_head(ball_bbox, heads_in_frame)

            # --- Cek bola dekat pemain (untuk deteksi GAGAL) ---
            ball_near_player = False
            nearest_player_id = -1
            ball_center = get_center_of_bbox(ball_bbox)

            for pid, pdata in tracks['players'][frame_num].items():
                pbbox = pdata.get('bbox')
                if pbbox is None:
                    continue
                pcenter = get_center_of_bbox(pbbox)
                pdist = measure_distance(ball_center, pcenter)
                if pdist <= self.max_approach_distance:
                    ball_near_player = True
                    nearest_player_id = pid
                    break

            # ======== STATE MACHINE ========

            if state == 'idle':
                # Cooldown
                if (frame_num - last_event_frame) < self.cooldown_frames:
                    continue

                # Bola harus sudah jauh cukup lama
                if away_frames < self.min_away_frames:
                    if not near_head and not ball_near_player:
                        away_frames += 1
                    continue

                # --- LANGSUNG CEK KONTAK ---
                # Bahkan tanpa approach, jika bola tiba-tiba overlap kepala
                if touching:
                    event_id_counter += 1
                    event = {
                        'event_id': event_id_counter,
                        'player_id': nearest_player_id,
                        'head_id': touch_head_id,
                        'frame_start': frame_num,
                        'frame_contact': frame_num,
                        'frame_end': frame_num,
                        'success': True,
                        'head_ball_distance': round(touch_dist, 1),
                        'iou': round(touch_iou, 3),
                        'duration_frames': 0,
                        'duration_seconds': 0.0,
                    }
                    heading_events.append(event)
                    last_event_frame = frame_num
                    away_frames = 0

                    if debug:
                        print(f"[HEADING] Frame {frame_num}: HEADING SUKSES ✓ "
                              f"(instant contact, head_id={touch_head_id}, "
                              f"dist={touch_dist:.0f}px, IoU={touch_iou:.3f})")
                    continue

                # Bola mendekat ke kepala?
                if near_head or ball_near_player:
                    state = 'approaching'
                    approach_start_frame = frame_num
                    approach_frames = 1
                    closest_head_dist_during_approach = near_dist
                    best_head_id_during_approach = touch_head_id
                else:
                    away_frames += 1

            elif state == 'approaching':
                approach_frames += 1

                # Update closest
                if near_dist < closest_head_dist_during_approach:
                    closest_head_dist_during_approach = near_dist

                # --- KONTAK! HEADING SUKSES ---
                if touching:
                    event_id_counter += 1
                    event = {
                        'event_id': event_id_counter,
                        'player_id': nearest_player_id,
                        'head_id': touch_head_id,
                        'frame_start': approach_start_frame,
                        'frame_contact': frame_num,
                        'frame_end': frame_num,
                        'success': True,
                        'head_ball_distance': round(touch_dist, 1),
                        'iou': round(touch_iou, 3),
                        'duration_frames': approach_frames,
                        'duration_seconds': round(approach_frames / self.fps, 2),
                    }
                    heading_events.append(event)
                    last_event_frame = frame_num
                    away_frames = 0
                    state = 'idle'

                    if debug:
                        print(f"[HEADING] Frame {frame_num}: HEADING SUKSES ✓ "
                              f"(approach {approach_frames}f, "
                              f"head_id={touch_head_id}, "
                              f"dist={touch_dist:.0f}px, IoU={touch_iou:.3f})")
                    continue

                # --- TIMEOUT ---
                if approach_frames > max_approach_frames:
                    if debug:
                        print(f"[HEADING] Frame {frame_num}: TIMEOUT "
                              f"(approach {approach_frames}f) → reset")
                    state = 'idle'
                    away_frames = 0
                    continue

                # --- BOLA MENJAUH (tidak ada kepala & tidak dekat pemain) ---
                if not near_head and not ball_near_player:
                    # Bola pergi tanpa menyentuh kepala
                    if approach_frames >= self.min_approach_frames:
                        event_id_counter += 1
                        event = {
                            'event_id': event_id_counter,
                            'player_id': nearest_player_id,
                            'head_id': best_head_id_during_approach,
                            'frame_start': approach_start_frame,
                            'frame_contact': frame_num,
                            'frame_end': frame_num,
                            'success': False,
                            'head_ball_distance': round(
                                closest_head_dist_during_approach, 1
                            ),
                            'iou': 0.0,
                            'duration_frames': approach_frames,
                            'duration_seconds': round(
                                approach_frames / self.fps, 2
                            ),
                        }
                        heading_events.append(event)
                        last_event_frame = frame_num

                        if debug:
                            print(f"[HEADING] Frame {frame_num}: HEADING GAGAL ✗ "
                                  f"(bola menjauh tanpa kontak kepala, "
                                  f"closest_dist="
                                  f"{closest_head_dist_during_approach:.0f}px)")

                    state = 'idle'
                    away_frames = 0

        # ============================================================
        # HASIL
        # ============================================================
        if debug:
            sukses = sum(1 for e in heading_events if e['success'])
            gagal = sum(1 for e in heading_events if not e['success'])
            total = len(heading_events)
            print(f"\n[HEADING] === HASIL AKHIR ===")
            print(f"[HEADING] Total heading   : {total}")
            print(f"[HEADING] SUKSES          : {sukses}")
            print(f"[HEADING] GAGAL           : {gagal}")
            if total > 0:
                print(f"[HEADING] Akurasi         : "
                      f"{sukses/total*100:.1f}%")
            print(f"[HEADING] =====================\n")

        return heading_events

    # ============================================================
    # DEBUG: Print jarak bola-kepala per frame (untuk tuning)
    # ============================================================

    def debug_distances(
        self,
        tracks: Dict,
        sample_every: int = 10,
    ) -> None:
        """
        Print jarak bola ke semua kepala setiap N frame.
        Berguna untuk tuning parameter.
        """
        total_frames = len(tracks['players'])
        print(f"\n[DEBUG] === JARAK BOLA-KEPALA (setiap {sample_every} frame) ===")
        print(f"{'Frame':<8} {'BallPos':<20} {'HeadID':<8} "
              f"{'HeadPos':<20} {'Dist':<10} {'Overlap':<10}")
        print("-" * 80)

        for f in range(0, total_frames, sample_every):
            ball_data = tracks['ball'][f].get(1)
            if not ball_data or 'bbox' not in ball_data:
                continue

            ball_bbox = ball_data['bbox']
            ball_center = get_center_of_bbox(ball_bbox)

            heads = tracks['heading'][f]
            if not heads:
                continue

            for hid, hdata in heads.items():
                hbbox = hdata.get('bbox')
                if hbbox is None:
                    continue
                hcenter = get_center_of_bbox(hbbox)
                dist = measure_distance(ball_center, hcenter)
                overlap = self._bbox_overlap(
                    ball_bbox, hbbox, self.head_bbox_margin
                )

                print(f"{f:<8} "
                      f"({ball_center[0]:.0f},{ball_center[1]:.0f}){'':<8} "
                      f"{hid:<8} "
                      f"({hcenter[0]:.0f},{hcenter[1]:.0f}){'':<8} "
                      f"{dist:<10.1f} "
                      f"{'YA' if overlap else '-':<10}")

        print("-" * 80)
        print()

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
            'accuracy_pct': round(
                len(sukses) / total * 100, 1
            ) if total > 0 else 0.0,
            'avg_dist_success': avg_dist_success,
            'avg_iou_success': avg_iou_success,
            'player_stats': player_stats,
        }
