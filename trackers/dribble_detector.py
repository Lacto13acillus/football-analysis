# dribble_detector.py

import sys
sys.path.append('../')

from utils.bbox_utils import (
    measure_distance,
    get_center_of_bbox,
    get_center_of_bbox_bottom,
    stabilize_cone_positions,
)
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class DribbleDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps

        # ============================================================
        # PARAMETER RADIUS CONE — DIKECILKAN
        # ============================================================
        self.default_cone_radius: float = 25.0
        self.cone_radius_multiplier: float = 0.8    # 1.5 → 0.8
        self.min_cone_radius: float = 15.0           # 20 → 15
        self.max_cone_radius: float = 40.0           # 80 → 40
        self.manual_cone_radii: Dict[int, float] = {}

        # ============================================================
        # PARAMETER DRIBBLE ATTEMPT DETECTION
        # ============================================================
        self.max_dribble_ball_distance: float = 130.0
        self.entry_exit_zone_radius: float = 150.0
        self.min_attempt_frames: int = 15
        self.cooldown_frames: int = 30
        self.max_attempt_duration_sec: float = 30.0
        self.min_possession_ratio: float = 0.3

        # ============================================================
        # MODE DETEKSI
        # ============================================================
        self.detection_mode: str = "auto"
        self.auto_mode_max_duration_sec: float = 10.0

        # ============================================================
        # PARAMETER TEMPORAL FILTERING
        # ============================================================
        self.min_consecutive_touch_frames: int = 2

        # ============================================================
        # PARAMETER BALL EDGE DISTANCE
        # ============================================================
        self.use_ball_edge_distance: bool = True

        # ============================================================
        # PARAMETER TRAJECTORY INTERPOLATION
        # ============================================================
        self.interpolation_substeps: int = 3

        # ============================================================
        # PARAMETER CONE ORDERING
        # ============================================================
        self.cone_order_direction: str = 'auto'

        # ============================================================
        # PARAMETER STABILISASI CONE — DIPERBAIKI
        # ============================================================
        # Minimum kemunculan cone (rasio terhadap total frame)
        self.min_cone_appearance_ratio: float = 0.01  # 5% → 1%

        # Jarak minimum antar cone unik (untuk deduplikasi)
        # Cone yang jaraknya < ini dianggap cone yang SAMA
        self.cone_dedup_distance: float = 50.0

        # ============================================================
        # INTERNAL STATE
        # ============================================================
        self._stabilized_cones: Optional[Dict[int, Tuple[float, float]]] = None
        self._cone_radii: Dict[int, float] = {}
        self._ordered_cone_ids: List[int] = []
        self._cone_bbox_sizes: Dict[int, float] = {}

    # ============================================================
    # PUBLIC GETTERS
    # ============================================================

    def get_all_cones(self) -> Optional[Dict[int, Tuple[float, float]]]:
        return self._stabilized_cones

    def get_cone_radii(self) -> Dict[int, float]:
        return self._cone_radii

    def get_ordered_cone_ids(self) -> List[int]:
        return self._ordered_cone_ids

    # ============================================================
    # INISIALISASI CONE
    # ============================================================

    def initialize_cones(
        self,
        tracks       : Dict,
        cone_key     : str = 'cones',
        sample_frames: int = -1,
        debug        : bool = True
    ) -> bool:
        if cone_key not in tracks or not tracks[cone_key]:
            print(f"[DRIBBLE] WARNING: Key '{cone_key}' tidak ada di tracks!")
            return False

        if debug:
            print(f"\n[DRIBBLE] === INISIALISASI CONE ===")

        total_frames = len(tracks[cone_key])
        if sample_frames <= 0:
            sample_frames = total_frames

        if debug:
            print(f"[DRIBBLE] Sample frames: {sample_frames}/{total_frames}")

        # LANGKAH 1: Kumpulkan semua posisi cone dari semua frame
        raw_cones = self._collect_all_cone_positions(
            tracks, cone_key, sample_frames, debug
        )

        if debug:
            print(f"[DRIBBLE] Raw cone IDs: {len(raw_cones)}")

        # LANGKAH 2: Deduplikasi — merge cone yang terlalu dekat
        self._stabilized_cones = self._deduplicate_cones(raw_cones, debug)

        if debug:
            print(f"[DRIBBLE] Setelah deduplikasi: {len(self._stabilized_cones)} cone unik")

        if len(self._stabilized_cones) < 2:
            print("[DRIBBLE] GAGAL: Minimal 2 cone diperlukan!")
            return False

        # LANGKAH 3: Hitung bbox size per cone (untuk radius)
        self._compute_cone_bbox_sizes_dedup(tracks, cone_key, sample_frames)
        self._compute_cone_radii(debug)
        self._order_cones(debug)

        if debug:
            print(f"\n[DRIBBLE] Total cone       : {len(self._stabilized_cones)}")
            print(f"[DRIBBLE] Urutan cone IDs  : {self._ordered_cone_ids}")
            print(f"[DRIBBLE] Temporal filter   : {self.min_consecutive_touch_frames} frames")
            print(f"[DRIBBLE] Ball edge dist   : {'Ya' if self.use_ball_edge_distance else 'Tidak'}")
            print(f"[DRIBBLE] Interpolation    : {self.interpolation_substeps} substeps")
            print(f"[DRIBBLE] Detection mode   : {self.detection_mode}")
            for idx, cid in enumerate(self._ordered_cone_ids):
                pos = self._stabilized_cones[cid]
                r = self._cone_radii[cid]
                role = ""
                if idx == 0:
                    role = " ← ENTRY"
                elif idx == len(self._ordered_cone_ids) - 1:
                    role = " ← EXIT"
                print(f"[DRIBBLE]   [{idx}] Cone {cid}: "
                      f"pos=({pos[0]:.0f}, {pos[1]:.0f}), "
                      f"radius={r:.0f}px{role}")
            print(f"[DRIBBLE] ================================\n")

        return True

    def _collect_all_cone_positions(
        self,
        tracks       : Dict,
        cone_key     : str,
        sample_frames: int,
        debug        : bool
    ) -> Dict[int, List[Tuple[float, float]]]:
        """Kumpulkan semua posisi cone dari semua frame."""
        cone_positions: Dict[int, List[Tuple[float, float]]] = {}
        total = min(sample_frames, len(tracks.get(cone_key, [])))

        for f in range(total):
            for cone_id, cone_data in tracks[cone_key][f].items():
                bbox = cone_data.get('bbox')
                if bbox is None:
                    continue
                pos = get_center_of_bbox_bottom(bbox)
                if cone_id not in cone_positions:
                    cone_positions[cone_id] = []
                cone_positions[cone_id].append(pos)

        # Filter: cone harus muncul di cukup banyak frame
        min_appearances = max(1, int(total * self.min_cone_appearance_ratio))
        filtered: Dict[int, List[Tuple[float, float]]] = {}

        for cone_id, positions in cone_positions.items():
            if len(positions) >= min_appearances:
                filtered[cone_id] = positions
                if debug:
                    avg_x = float(np.mean([p[0] for p in positions]))
                    avg_y = float(np.mean([p[1] for p in positions]))
                    print(f"[DRIBBLE] Cone {cone_id}: {len(positions)} appearances "
                          f"→ avg=({avg_x:.0f}, {avg_y:.0f}) ✓")
            else:
                if debug:
                    print(f"[DRIBBLE] Cone {cone_id}: {len(positions)} appearances "
                          f"< {min_appearances} → DIBUANG")

        return filtered

    def _deduplicate_cones(
        self,
        raw_cones: Dict[int, List[Tuple[float, float]]],
        debug    : bool
    ) -> Dict[int, Tuple[float, float]]:
        """
        Deduplikasi cone yang terlalu dekat satu sama lain.

        Masalah: Tracker sering membuat beberapa anchor ID untuk cone
        yang sama (karena jitter deteksi, partial occlusion, dll).
        Solusi: Merge cone yang jaraknya < cone_dedup_distance.
        Cone dengan appearances terbanyak diprioritaskan.
        """
        # Hitung posisi rata-rata per cone
        cone_avg: Dict[int, Tuple[float, float]] = {}
        cone_counts: Dict[int, int] = {}
        for cid, positions in raw_cones.items():
            avg_x = float(np.mean([p[0] for p in positions]))
            avg_y = float(np.mean([p[1] for p in positions]))
            cone_avg[cid] = (avg_x, avg_y)
            cone_counts[cid] = len(positions)

        # Sort by appearances (terbanyak dulu — ini yang paling reliable)
        sorted_ids = sorted(cone_avg.keys(), key=lambda x: -cone_counts[x])

        # Greedy dedup: ambil cone, buang semua cone lain yang terlalu dekat
        final_cones: Dict[int, Tuple[float, float]] = {}
        used = set()

        for cid in sorted_ids:
            if cid in used:
                continue

            pos = cone_avg[cid]
            too_close = False

            for existing_id, existing_pos in final_cones.items():
                dist = measure_distance(pos, existing_pos)
                if dist < self.cone_dedup_distance:
                    too_close = True
                    if debug:
                        print(f"[DRIBBLE] Cone {cid} terlalu dekat ke Cone {existing_id} "
                              f"(dist={dist:.0f}px < {self.cone_dedup_distance}px) → MERGE")
                    break

            if not too_close:
                final_cones[cid] = pos
                used.add(cid)

        # Re-index cone IDs menjadi 0, 1, 2, ... agar rapi
        reindexed: Dict[int, Tuple[float, float]] = {}
        for new_id, (old_id, pos) in enumerate(final_cones.items()):
            reindexed[new_id] = pos
            if debug:
                print(f"[DRIBBLE] Final: Cone {old_id} → Cone {new_id} "
                      f"at ({pos[0]:.0f}, {pos[1]:.0f})")

        return reindexed

    def _compute_cone_bbox_sizes_dedup(
        self, tracks: Dict, cone_key: str, sample_frames: int
    ) -> None:
        """
        Hitung rata-rata bbox size per STABILIZED cone.
        Karena cone sudah di-reindex, kita perlu match berdasarkan posisi.
        """
        if not self._stabilized_cones:
            return

        # Kumpulkan semua bbox sizes dari raw detections
        all_sizes: List[float] = []
        total = min(sample_frames, len(tracks.get(cone_key, [])))

        for f in range(total):
            for cone_id, cone_data in tracks[cone_key][f].items():
                bbox = cone_data.get('bbox')
                if bbox is None:
                    continue
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w <= 0 or h <= 0:
                    continue

                det_pos = get_center_of_bbox_bottom(bbox)
                size = max(w, h)

                # Match ke stabilized cone terdekat
                for scid, spos in self._stabilized_cones.items():
                    dist = measure_distance(det_pos, spos)
                    if dist < self.cone_dedup_distance:
                        if scid not in self._cone_bbox_sizes:
                            self._cone_bbox_sizes[scid] = []
                        # Sementara simpan sebagai list, nanti di-average
                        if not isinstance(self._cone_bbox_sizes.get(scid), list):
                            self._cone_bbox_sizes[scid] = []
                        self._cone_bbox_sizes[scid].append(size)
                        break

        # Average
        for scid in list(self._cone_bbox_sizes.keys()):
            sizes = self._cone_bbox_sizes[scid]
            if isinstance(sizes, list) and len(sizes) > 0:
                self._cone_bbox_sizes[scid] = float(np.mean(sizes))
            else:
                self._cone_bbox_sizes[scid] = 0.0

    def _compute_cone_radii(self, debug: bool = False) -> None:
        for cone_id in self._stabilized_cones:
            if cone_id in self.manual_cone_radii:
                radius = self.manual_cone_radii[cone_id]
                source = "manual"
            elif cone_id in self._cone_bbox_sizes and self._cone_bbox_sizes[cone_id] > 0:
                radius = self._cone_bbox_sizes[cone_id] * self.cone_radius_multiplier
                source = "dynamic"
            else:
                radius = self.default_cone_radius
                source = "default"

            radius = max(self.min_cone_radius, min(self.max_cone_radius, radius))
            self._cone_radii[cone_id] = radius

            if debug:
                bbox_s = self._cone_bbox_sizes.get(cone_id, 0)
                if isinstance(bbox_s, list):
                    bbox_s = float(np.mean(bbox_s)) if bbox_s else 0
                print(f"[DRIBBLE] Cone {cone_id}: bbox_size={bbox_s:.0f}px "
                      f"→ radius={radius:.0f}px ({source})")

    def _order_cones(self, debug: bool = False) -> None:
        cones = self._stabilized_cones
        cone_ids = list(cones.keys())
        positions = [cones[cid] for cid in cone_ids]

        if self.cone_order_direction == 'auto':
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            x_spread = max(xs) - min(xs)
            y_spread = max(ys) - min(ys)

            if y_spread >= x_spread:
                direction = 'top_to_bottom'
            else:
                direction = 'left_to_right'

            if debug:
                print(f"[DRIBBLE] Auto-detect: x_spread={x_spread:.0f}, "
                      f"y_spread={y_spread:.0f} → {direction}")
        else:
            direction = self.cone_order_direction

        sort_key = {
            'top_to_bottom' : lambda x: x[1][1],
            'bottom_to_top' : lambda x: -x[1][1],
            'left_to_right' : lambda x: x[1][0],
            'right_to_left' : lambda x: -x[1][0],
        }

        key_fn = sort_key.get(direction, lambda x: x[1][1])
        paired = sorted(zip(cone_ids, positions), key=key_fn)
        self._ordered_cone_ids = [cid for cid, _ in paired]

    # ============================================================
    # HELPER: BALL EDGE DISTANCE
    # ============================================================

    def _compute_ball_cone_distance(
        self,
        ball_pos : Tuple[float, float],
        cone_pos : Tuple[float, float],
        ball_bbox: Optional[List[float]] = None
    ) -> float:
        if not self.use_ball_edge_distance or ball_bbox is None:
            return measure_distance(ball_pos, cone_pos)

        bx1, by1, bx2, by2 = ball_bbox
        cx, cy = cone_pos
        closest_x = max(bx1, min(cx, bx2))
        closest_y = max(by1, min(cy, by2))
        return measure_distance((closest_x, closest_y), cone_pos)

    # ============================================================
    # HELPER: TRAJECTORY INTERPOLATION
    # ============================================================

    def _interpolate_positions(
        self,
        pos_a: Tuple[float, float],
        pos_b: Tuple[float, float],
        steps: int
    ) -> List[Tuple[float, float]]:
        if steps <= 1:
            return [pos_a, pos_b]

        result = []
        for i in range(steps + 1):
            t = i / steps
            x = pos_a[0] + (pos_b[0] - pos_a[0]) * t
            y = pos_a[1] + (pos_b[1] - pos_a[1]) * t
            result.append((x, y))
        return result

    # ============================================================
    # CEK SENTUHAN BOLA KE CONE
    # ============================================================

    def check_ball_touches_any_cone(
        self,
        ball_pos : Tuple[float, float],
        ball_bbox: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        touched = []
        if not self._stabilized_cones:
            return touched

        for cone_id, cone_pos in self._stabilized_cones.items():
            radius = self._cone_radii.get(cone_id, self.default_cone_radius)
            dist = self._compute_ball_cone_distance(ball_pos, cone_pos, ball_bbox)
            if dist <= radius:
                touched.append((cone_id, dist))

        return touched

    def evaluate_trajectory_against_cones(
        self,
        ball_trajectory: List[Tuple[float, float]],
        ball_bboxes    : Optional[List[Optional[List[float]]]] = None
    ) -> Dict[int, Dict[str, Any]]:
        results: Dict[int, Dict[str, Any]] = {}

        for cone_id in self._stabilized_cones:
            radius = self._cone_radii.get(cone_id, self.default_cone_radius)
            cone_pos = self._stabilized_cones[cone_id]

            min_dist = float('inf')
            raw_touch_count = 0
            first_touch = -1

            consecutive_count = 0
            max_consecutive = 0
            valid_touch = False

            for i, ball_pos in enumerate(ball_trajectory):
                bbox = None
                if ball_bboxes and i < len(ball_bboxes):
                    bbox = ball_bboxes[i]

                positions_to_check = [(ball_pos, bbox)]

                if i > 0 and self.interpolation_substeps > 1:
                    prev_pos = ball_trajectory[i - 1]
                    interp = self._interpolate_positions(
                        prev_pos, ball_pos, self.interpolation_substeps
                    )
                    for ip in interp[1:-1]:
                        positions_to_check.append((ip, None))

                frame_touched = False
                for pos, bb in positions_to_check:
                    dist = self._compute_ball_cone_distance(pos, cone_pos, bb)
                    if dist < min_dist:
                        min_dist = dist
                    if dist <= radius:
                        frame_touched = True

                if frame_touched:
                    raw_touch_count += 1
                    consecutive_count += 1
                    if first_touch == -1:
                        first_touch = i
                    if consecutive_count > max_consecutive:
                        max_consecutive = consecutive_count
                    if consecutive_count >= self.min_consecutive_touch_frames:
                        valid_touch = True
                else:
                    consecutive_count = 0

            results[cone_id] = {
                'touched'          : valid_touch,
                'min_distance'     : min_dist,
                'touch_count'      : raw_touch_count,
                'consecutive_max'  : max_consecutive,
                'first_touch_idx'  : first_touch,
                'radius'           : radius,
            }

        return results

    # ============================================================
    # HELPERS
    # ============================================================

    def _get_ball_pos(self, tracks: Dict, frame_num: int) -> Optional[Tuple[float, float]]:
        ball_data = tracks['ball'][frame_num].get(1)
        if ball_data and 'bbox' in ball_data:
            return get_center_of_bbox_bottom(ball_data['bbox'])
        return None

    def _get_ball_bbox(self, tracks: Dict, frame_num: int) -> Optional[List[float]]:
        ball_data = tracks['ball'][frame_num].get(1)
        if ball_data and 'bbox' in ball_data:
            return ball_data['bbox']
        return None

    def _get_player_pos(self, tracks: Dict, frame_num: int) -> Optional[Tuple[int, Tuple[float, float]]]:
        players = tracks['players'][frame_num]
        for pid, pdata in players.items():
            bbox = pdata.get('bbox')
            if bbox:
                pos = get_center_of_bbox_bottom(bbox)
                return pid, pos
        return None

    # ============================================================
    # AUTO-DETECT ARAH DRIBBLE DARI GERAKAN PEMAIN
    # ============================================================

    def _auto_detect_dribble_direction(
        self,
        tracks: Dict,
        debug : bool = True
    ) -> None:
        total_frames = len(tracks['players'])
        if total_frames < 10:
            return

        early_positions = []
        for f in range(min(10, total_frames)):
            result = self._get_player_pos(tracks, f)
            if result:
                early_positions.append(result[1])

        late_positions = []
        for f in range(max(0, total_frames - 10), total_frames):
            result = self._get_player_pos(tracks, f)
            if result:
                late_positions.append(result[1])

        if not early_positions or not late_positions:
            if debug:
                print("[DRIBBLE] Tidak bisa detect arah: pemain tidak ditemukan")
            return

        early_avg = (
            float(np.mean([p[0] for p in early_positions])),
            float(np.mean([p[1] for p in early_positions]))
        )
        late_avg = (
            float(np.mean([p[0] for p in late_positions])),
            float(np.mean([p[1] for p in late_positions]))
        )

        if debug:
            print(f"[DRIBBLE] Posisi pemain awal : ({early_avg[0]:.0f}, {early_avg[1]:.0f})")
            print(f"[DRIBBLE] Posisi pemain akhir: ({late_avg[0]:.0f}, {late_avg[1]:.0f})")

        # Cari cone terdekat ke posisi awal dan akhir
        best_entry_id = None
        best_entry_dist = float('inf')
        best_exit_id = None
        best_exit_dist = float('inf')

        for cid, cpos in self._stabilized_cones.items():
            d_early = measure_distance(early_avg, cpos)
            d_late = measure_distance(late_avg, cpos)

            if d_early < best_entry_dist:
                best_entry_dist = d_early
                best_entry_id = cid
            if d_late < best_exit_dist:
                best_exit_dist = d_late
                best_exit_id = cid

        if best_entry_id is not None and best_exit_id is not None:
            if debug:
                print(f"[DRIBBLE] Entry cone terdekat: {best_entry_id} "
                      f"(dist={best_entry_dist:.0f}px)")
                print(f"[DRIBBLE] Exit cone terdekat : {best_exit_id} "
                      f"(dist={best_exit_dist:.0f}px)")

            entry_pos = self._stabilized_cones[best_entry_id]
            exit_pos = self._stabilized_cones[best_exit_id]

            dx = exit_pos[0] - entry_pos[0]
            dy = exit_pos[1] - entry_pos[1]

            if abs(dx) >= abs(dy):
                if dx > 0:
                    self.cone_order_direction = 'left_to_right'
                else:
                    self.cone_order_direction = 'right_to_left'
            else:
                if dy > 0:
                    self.cone_order_direction = 'top_to_bottom'
                else:
                    self.cone_order_direction = 'bottom_to_top'

            if debug:
                print(f"[DRIBBLE] Arah dribble terdeteksi: {self.cone_order_direction}")

            self._order_cones(debug=False)

    # ============================================================
    # DETEKSI DRIBBLE ATTEMPTS (CORE)
    # ============================================================

    def detect_dribble_attempts(
        self,
        tracks          : Dict,
        ball_possessions: List[int],
        debug           : bool = True
    ) -> List[Dict]:
        if not self._stabilized_cones or len(self._ordered_cone_ids) < 2:
            print("[DRIBBLE] ERROR: Cone belum diinisialisasi atau < 2!")
            return []

        total_frames = len(tracks['players'])
        video_duration = total_frames / self.fps

        if debug:
            print(f"\n[DRIBBLE] === AUTO-DETECT ARAH DRIBBLE ===")
        self._auto_detect_dribble_direction(tracks, debug)

        mode = self.detection_mode
        if mode == 'auto':
            if video_duration <= self.auto_mode_max_duration_sec:
                mode = 'whole_video'
                if debug:
                    print(f"[DRIBBLE] Video pendek ({video_duration:.1f}s "
                          f"<= {self.auto_mode_max_duration_sec}s) → mode: whole_video")
            else:
                mode = 'entry_exit'
                if debug:
                    print(f"[DRIBBLE] Video panjang ({video_duration:.1f}s) → mode: entry_exit")

        if mode == 'whole_video':
            return self._detect_whole_video(tracks, ball_possessions, debug)
        else:
            return self._detect_entry_exit(tracks, ball_possessions, debug)

    def _detect_whole_video(
        self,
        tracks          : Dict,
        ball_possessions: List[int],
        debug           : bool
    ) -> List[Dict]:
        total_frames = len(tracks['players'])

        if debug:
            print(f"\n[DRIBBLE] === DRIBBLE DETECTION (WHOLE VIDEO) ===")
            print(f"[DRIBBLE] Total frames: {total_frames}")
            print(f"[DRIBBLE] Total cones : {len(self._stabilized_cones)}")

        ball_trajectory: List[Tuple[float, float]] = []
        ball_bboxes: List[Optional[List[float]]] = []
        frames_with_ball = 0
        first_possession_frame = -1
        last_possession_frame = -1
        player_id_main = -1

        for frame_num in range(total_frames):
            ball_pos = self._get_ball_pos(tracks, frame_num)
            ball_bbox = self._get_ball_bbox(tracks, frame_num)

            has_ball = (
                frame_num < len(ball_possessions) and
                ball_possessions[frame_num] != -1
            )

            if ball_pos:
                ball_trajectory.append(ball_pos)
                ball_bboxes.append(ball_bbox)
            else:
                ball_trajectory.append(ball_trajectory[-1] if ball_trajectory else (0, 0))
                ball_bboxes.append(None)

            if has_ball:
                frames_with_ball += 1
                if first_possession_frame == -1:
                    first_possession_frame = frame_num
                last_possession_frame = frame_num

                if player_id_main == -1:
                    player_result = self._get_player_pos(tracks, frame_num)
                    if player_result:
                        player_id_main = player_result[0]

        if first_possession_frame == -1 or not ball_trajectory:
            if debug:
                print("[DRIBBLE] Tidak ada possession terdeteksi!")
            return []

        cone_results = self.evaluate_trajectory_against_cones(
            ball_trajectory, ball_bboxes
        )

        touched_cones = [
            cid for cid, res in cone_results.items()
            if res['touched']
        ]

        success = len(touched_cones) == 0
        duration_frames = last_possession_frame - first_possession_frame + 1
        possession_ratio = (
            frames_with_ball / total_frames if total_frames > 0 else 0.0
        )

        attempt = {
            'attempt_id'       : 1,
            'player_id'        : player_id_main,
            'frame_start'      : first_possession_frame,
            'frame_end'        : last_possession_frame,
            'duration_frames'  : duration_frames,
            'duration_seconds' : round(duration_frames / self.fps, 2),
            'success'          : success,
            'total_cones'      : len(self._stabilized_cones),
            'touched_cones'    : touched_cones,
            'cone_details'     : cone_results,
            'possession_ratio' : round(possession_ratio, 2),
            'ball_trajectory'  : list(ball_trajectory),
        }

        if debug:
            status = "SUKSES ✓" if success else "GAGAL ✗"
            print(f"[DRIBBLE] Hasil: {status}")
            print(f"[DRIBBLE]   Frame range  : {first_possession_frame}-{last_possession_frame}")
            print(f"[DRIBBLE]   Durasi       : {duration_frames} frames "
                  f"({duration_frames/self.fps:.1f}s)")
            print(f"[DRIBBLE]   Possession   : {possession_ratio*100:.0f}%")
            print(f"[DRIBBLE]   Cone disentuh: "
                  f"{len(touched_cones)}/{len(self._stabilized_cones)}")
            for cid in self._ordered_cone_ids:
                res = cone_results[cid]
                status_c = "HIT" if res['touched'] else "AMAN"
                print(f"[DRIBBLE]     Cone {cid}: {status_c} - "
                      f"min_dist={res['min_distance']:.1f}px "
                      f"(radius={res['radius']:.0f}px, "
                      f"touch_frames={res['touch_count']}, "
                      f"consecutive={res['consecutive_max']})")

        attempts = [attempt]

        if debug:
            sukses = sum(1 for a in attempts if a['success'])
            gagal = sum(1 for a in attempts if not a['success'])
            pct = sukses / len(attempts) * 100 if attempts else 0.0
            print(f"\n[DRIBBLE] === HASIL AKHIR ===")
            print(f"[DRIBBLE] Total attempts : {len(attempts)}")
            print(f"[DRIBBLE] SUKSES         : {sukses}")
            print(f"[DRIBBLE] GAGAL          : {gagal}")
            print(f"[DRIBBLE] Akurasi        : {pct:.1f}%")
            print(f"[DRIBBLE] =========================\n")

        return attempts

    def _detect_entry_exit(
        self,
        tracks          : Dict,
        ball_possessions: List[int],
        debug           : bool
    ) -> List[Dict]:
        total_frames = len(tracks['players'])
        entry_cone_id = self._ordered_cone_ids[0]
        exit_cone_id = self._ordered_cone_ids[-1]
        entry_pos = self._stabilized_cones[entry_cone_id]
        exit_pos = self._stabilized_cones[exit_cone_id]
        max_attempt_frames = int(self.max_attempt_duration_sec * self.fps)

        if debug:
            print(f"\n[DRIBBLE] === DRIBBLE DETECTION (ENTRY/EXIT) ===")
            print(f"[DRIBBLE] Total frames           : {total_frames}")
            print(f"[DRIBBLE] Total cones             : {len(self._stabilized_cones)}")
            print(f"[DRIBBLE] Entry cone {entry_cone_id}: "
                  f"({entry_pos[0]:.0f}, {entry_pos[1]:.0f})")
            print(f"[DRIBBLE] Exit cone  {exit_cone_id}: "
                  f"({exit_pos[0]:.0f}, {exit_pos[1]:.0f})")
            print(f"[DRIBBLE] Entry/exit zone radius  : {self.entry_exit_zone_radius}px")

        attempts: List[Dict] = []
        last_attempt_end = -999

        state = 'idle'
        attempt_start_frame = -1
        attempt_player_id = -1
        ball_trajectory: List[Tuple[float, float]] = []
        ball_bboxes: List[Optional[List[float]]] = []
        player_positions: List[Tuple[float, float]] = []
        frames_with_ball = 0
        attempt_total_frames = 0

        for frame_num in range(total_frames):
            player_result = self._get_player_pos(tracks, frame_num)
            if player_result is None:
                continue
            player_id, player_pos = player_result

            ball_pos = self._get_ball_pos(tracks, frame_num)
            ball_bbox = self._get_ball_bbox(tracks, frame_num)

            has_ball = (
                frame_num < len(ball_possessions) and
                ball_possessions[frame_num] != -1
            )

            if state == 'idle':
                if has_ball and ball_pos:
                    dist_to_entry = measure_distance(player_pos, entry_pos)
                    if dist_to_entry <= self.entry_exit_zone_radius:
                        if (frame_num - last_attempt_end) >= self.cooldown_frames:
                            state = 'dribbling'
                            attempt_start_frame = frame_num
                            attempt_player_id = player_id
                            ball_trajectory = [ball_pos]
                            ball_bboxes = [ball_bbox]
                            player_positions = [player_pos]
                            frames_with_ball = 1
                            attempt_total_frames = 1

                            if debug:
                                print(f"[DRIBBLE] Frame {frame_num}: "
                                      f"MASUK entry zone → MULAI DRIBBLE")

            elif state == 'dribbling':
                attempt_total_frames += 1

                if ball_pos:
                    ball_trajectory.append(ball_pos)
                    ball_bboxes.append(ball_bbox)
                player_positions.append(player_pos)
                if has_ball:
                    frames_with_ball += 1

                dist_to_exit = measure_distance(player_pos, exit_pos)

                if dist_to_exit <= self.entry_exit_zone_radius:
                    duration_frames = frame_num - attempt_start_frame

                    if duration_frames >= self.min_attempt_frames:
                        cone_results = self.evaluate_trajectory_against_cones(
                            ball_trajectory, ball_bboxes
                        )

                        touched_cones = [
                            cid for cid, res in cone_results.items()
                            if res['touched']
                        ]

                        success = len(touched_cones) == 0
                        possession_ratio = (
                            frames_with_ball / attempt_total_frames
                            if attempt_total_frames > 0 else 0.0
                        )

                        attempt = {
                            'attempt_id'       : len(attempts) + 1,
                            'player_id'        : attempt_player_id,
                            'frame_start'      : attempt_start_frame,
                            'frame_end'        : frame_num,
                            'duration_frames'  : duration_frames,
                            'duration_seconds' : round(duration_frames / self.fps, 2),
                            'success'          : success,
                            'total_cones'      : len(self._stabilized_cones),
                            'touched_cones'    : touched_cones,
                            'cone_details'     : cone_results,
                            'possession_ratio' : round(possession_ratio, 2),
                            'ball_trajectory'  : list(ball_trajectory),
                        }
                        attempts.append(attempt)
                        last_attempt_end = frame_num

                        if debug:
                            status_text = "SUKSES ✓" if success else "GAGAL ✗"
                            print(f"[DRIBBLE] Frame {frame_num}: SELESAI → {status_text}")
                            print(f"[DRIBBLE]   Cone disentuh: "
                                  f"{len(touched_cones)}/{len(self._stabilized_cones)}")

                    state = 'idle'
                    continue

                if attempt_total_frames > max_attempt_frames:
                    if debug:
                        print(f"[DRIBBLE] Frame {frame_num}: TIMEOUT → reset")
                    state = 'idle'
                    continue

                if attempt_total_frames > 20:
                    current_ratio = frames_with_ball / attempt_total_frames
                    if current_ratio < self.min_possession_ratio:
                        if debug:
                            print(f"[DRIBBLE] Frame {frame_num}: "
                                  f"possession rendah ({current_ratio:.0%}) → BATAL")
                        state = 'idle'
                        continue

        if debug:
            sukses = sum(1 for a in attempts if a['success'])
            gagal = sum(1 for a in attempts if not a['success'])
            pct = sukses / len(attempts) * 100 if attempts else 0.0
            print(f"\n[DRIBBLE] === HASIL AKHIR ===")
            print(f"[DRIBBLE] Total attempts : {len(attempts)}")
            print(f"[DRIBBLE] SUKSES         : {sukses}")
            print(f"[DRIBBLE] GAGAL          : {gagal}")
            print(f"[DRIBBLE] Akurasi        : {pct:.1f}%")
            print(f"[DRIBBLE] =========================\n")

        return attempts

    # ============================================================
    # STATISTIK
    # ============================================================

    def get_dribble_statistics(self, attempts: List[Dict]) -> Dict:
        total = len(attempts)
        sukses = [a for a in attempts if a['success']]
        gagal = [a for a in attempts if not a['success']]

        cone_hit_count: Dict[int, int] = {}
        for a in attempts:
            for tc in a['touched_cones']:
                cone_hit_count[tc] = cone_hit_count.get(tc, 0) + 1

        cone_avg_min_dist: Dict[int, List[float]] = {}
        for a in attempts:
            for cid, detail in a['cone_details'].items():
                if cid not in cone_avg_min_dist:
                    cone_avg_min_dist[cid] = []
                cone_avg_min_dist[cid].append(detail['min_distance'])

        cone_analysis = {}
        for cid in self._ordered_cone_ids:
            dists = cone_avg_min_dist.get(cid, [])
            cone_analysis[cid] = {
                'avg_min_distance': round(float(np.mean(dists)), 1) if dists else 0.0,
                'times_touched'  : cone_hit_count.get(cid, 0),
                'radius'         : self._cone_radii.get(cid, 0.0),
            }

        return {
            'total_attempts'     : total,
            'successful_attempts': len(sukses),
            'failed_attempts'    : len(gagal),
            'accuracy_pct'       : round(len(sukses) / total * 100, 1) if total > 0 else 0.0,
            'avg_duration'       : round(float(np.mean(
                                       [a['duration_seconds'] for a in attempts])), 2)
                                   if attempts else 0.0,
            'cone_hit_frequency' : cone_hit_count,
            'cone_analysis'      : cone_analysis,
        }
