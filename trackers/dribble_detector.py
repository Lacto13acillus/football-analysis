# dribble_detector.py

import sys
sys.path.append('../')

from utils.bbox_utils import (
    measure_distance,
    get_center_of_bbox,
    get_center_of_bbox_bottom,
)
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class DribbleDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps

        # ============================================================
        # PARAMETER JUMLAH CONE
        # ============================================================
        self.expected_num_cones: Optional[int] = None  # Set ke 7 jika tahu

        # ============================================================
        # PARAMETER STABILISASI CONE
        # ============================================================
        self.min_cone_appearance_ratio: float = 0.03
        self.cone_dedup_distance: float = 80.0

        # ============================================================
        # PARAMETER CONE RADIUS (untuk visualisasi)
        # ============================================================
        self.default_cone_radius: float = 25.0
        self.cone_radius_multiplier: float = 0.8
        self.min_cone_radius: float = 15.0
        self.max_cone_radius: float = 40.0
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
        # PARAMETER CONE ORDERING
        # ============================================================
        self.cone_order_direction: str = 'auto'

        # ============================================================
        # === BARU: PARAMETER DETEKSI SENTUHAN ZIG-ZAG ===
        # ============================================================

        # --- Metode 1: Cone Displacement ---
        self.use_cone_displacement: bool = True
        # Jika posisi cone bergeser > threshold ini (px) dalam window,
        # dianggap tertabrak
        self.cone_displacement_threshold: float = 20.0
        # Window frame untuk mendeteksi displacement
        self.cone_displacement_window: int = 3
        # Bola harus dekat cone (px) saat displacement terjadi
        self.cone_displacement_ball_proximity: float = 150.0

        # --- Metode 2: BBox Overlap ---
        self.use_bbox_overlap: bool = True
        # Padding negatif pada cone bbox (shrink) — makin besar,
        # makin ketat (hanya overlap yang benar-benar nyata)
        self.bbox_overlap_shrink: float = 5.0
        # Minimum jumlah frame bbox overlap berturut-turut
        self.min_overlap_consecutive_frames: int = 3
        self.bbox_overlap_min_iou: float = 0.05

        # --- Metode 3: Ball Speed Anomaly ---
        self.use_speed_anomaly: bool = True
        # Rasio perlambatan: jika speed drop ke < ratio * avg_speed
        # saat dekat cone, dianggap menabrak
        self.speed_drop_ratio: float = 0.3
        # Bola harus dekat cone (px) saat speed drop
        self.speed_anomaly_proximity: float = 60.0
        # Window untuk hitung average speed
        self.speed_avg_window: int = 10

        # --- Kombinasi ---
        # Minimum berapa metode yang harus agree untuk declare "touch"
        # 1 = salah satu cukup, 2 = minimal 2 metode setuju
        self.min_methods_agree: int = 1

        # ============================================================
        # INTERNAL STATE
        # ============================================================
        self._stabilized_cones: Optional[Dict[int, Tuple[float, float]]] = None
        self._cone_radii: Dict[int, float] = {}
        self._ordered_cone_ids: List[int] = []
        self._cone_bbox_sizes: Dict[int, float] = {}
        # Mapping dari stabilized cone ID → raw tracker cone IDs
        self._cone_id_mapping: Dict[int, List[int]] = {}

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

        # LANGKAH 2: Deduplikasi
        self._stabilized_cones = self._deduplicate_cones(raw_cones, debug)

        if debug:
            print(f"[DRIBBLE] Setelah deduplikasi: "
                  f"{len(self._stabilized_cones)} cone unik")

        if len(self._stabilized_cones) < 2:
            print("[DRIBBLE] GAGAL: Minimal 2 cone diperlukan!")
            return False

        # LANGKAH 3: Hitung bbox size per cone (untuk radius visualisasi)
        self._compute_cone_bbox_sizes_dedup(tracks, cone_key, sample_frames)
        self._compute_cone_radii(debug)
        self._order_cones(debug)

        if debug:
            print(f"\n[DRIBBLE] Total cone           : {len(self._stabilized_cones)}")
            print(f"[DRIBBLE] Urutan cone IDs      : {self._ordered_cone_ids}")
            print(f"[DRIBBLE] Detection mode       : {self.detection_mode}")
            print(f"[DRIBBLE] --- Metode Sentuhan ---")
            print(f"[DRIBBLE]   Cone displacement  : "
                  f"{'Ya' if self.use_cone_displacement else 'Tidak'} "
                  f"(threshold={self.cone_displacement_threshold}px)")
            print(f"[DRIBBLE]   BBox overlap       : "
                  f"{'Ya' if self.use_bbox_overlap else 'Tidak'} "
                  f"(shrink={self.bbox_overlap_shrink}px)")
            print(f"[DRIBBLE]   Speed anomaly      : "
                  f"{'Ya' if self.use_speed_anomaly else 'Tidak'} "
                  f"(drop_ratio={self.speed_drop_ratio})")
            print(f"[DRIBBLE]   Min methods agree  : {self.min_methods_agree}")
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
        if not raw_cones:
            return {}

        cone_avg: Dict[int, Tuple[float, float]] = {}
        cone_counts: Dict[int, int] = {}
        for cid, positions in raw_cones.items():
            avg_x = float(np.mean([p[0] for p in positions]))
            avg_y = float(np.mean([p[1] for p in positions]))
            cone_avg[cid] = (avg_x, avg_y)
            cone_counts[cid] = len(positions)

        sorted_ids = sorted(cone_avg.keys(), key=lambda x: -cone_counts[x])

        # Greedy dedup
        final_cones: Dict[int, Tuple[float, float]] = {}
        final_counts: Dict[int, int] = {}
        # Track which raw IDs map to which final cone
        raw_to_final: Dict[int, int] = {}  # raw_id → final_id (old_id)

        for cid in sorted_ids:
            pos = cone_avg[cid]
            merged_into = None

            for existing_id, existing_pos in final_cones.items():
                dist = measure_distance(pos, existing_pos)
                if dist < self.cone_dedup_distance:
                    merged_into = existing_id
                    if debug:
                        print(f"[DRIBBLE] Cone {cid} ({cone_counts[cid]}x) "
                              f"terlalu dekat ke Cone {existing_id} "
                              f"({cone_counts[existing_id]}x) "
                              f"(dist={dist:.0f}px) → MERGE")
                    break

            if merged_into is not None:
                raw_to_final[cid] = merged_into
            else:
                final_cones[cid] = pos
                final_counts[cid] = cone_counts[cid]
                raw_to_final[cid] = cid

        # Potong ke expected_num_cones jika di-set
        if (self.expected_num_cones is not None
                and self.expected_num_cones > 0
                and len(final_cones) > self.expected_num_cones):
            sorted_final = sorted(
                final_cones.keys(), key=lambda x: -final_counts[x]
            )
            keep_ids = set(sorted_final[:self.expected_num_cones])
            removed_ids = set(sorted_final[self.expected_num_cones:])

            if debug:
                for rid in removed_ids:
                    print(f"[DRIBBLE] Cone {rid} ({final_counts[rid]}x) "
                          f"→ DIBUANG (melebihi expected="
                          f"{self.expected_num_cones})")

            final_cones = {
                k: v for k, v in final_cones.items() if k in keep_ids
            }

        # Re-index & build mapping
        sorted_by_count = sorted(
            final_cones.keys(), key=lambda x: -final_counts.get(x, 0)
        )
        reindexed: Dict[int, Tuple[float, float]] = {}
        self._cone_id_mapping = {}

        for new_id, old_id in enumerate(sorted_by_count):
            reindexed[new_id] = final_cones[old_id]
            # Kumpulkan semua raw IDs yang map ke old_id ini
            mapped_raw = [old_id]
            for raw_id, final_id in raw_to_final.items():
                if final_id == old_id and raw_id != old_id:
                    mapped_raw.append(raw_id)
            self._cone_id_mapping[new_id] = mapped_raw

            if debug:
                print(f"[DRIBBLE] Final: Cone {old_id} "
                      f"({final_counts.get(old_id, 0)}x) → Cone {new_id} "
                      f"at ({final_cones[old_id][0]:.0f}, "
                      f"{final_cones[old_id][1]:.0f}) "
                      f"[raw IDs: {mapped_raw}]")

        return reindexed

    def _compute_cone_bbox_sizes_dedup(
        self, tracks: Dict, cone_key: str, sample_frames: int
    ) -> None:
        if not self._stabilized_cones:
            return

        total = min(sample_frames, len(tracks.get(cone_key, [])))
        temp_sizes: Dict[int, List[float]] = {
            scid: [] for scid in self._stabilized_cones
        }

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

                for scid, spos in self._stabilized_cones.items():
                    dist = measure_distance(det_pos, spos)
                    if dist < self.cone_dedup_distance:
                        temp_sizes[scid].append(size)
                        break

        self._cone_bbox_sizes = {}
        for scid, sizes in temp_sizes.items():
            if sizes:
                self._cone_bbox_sizes[scid] = float(np.mean(sizes))
            else:
                self._cone_bbox_sizes[scid] = 0.0

    def _compute_cone_radii(self, debug: bool = False) -> None:
        for cone_id in self._stabilized_cones:
            if cone_id in self.manual_cone_radii:
                radius = self.manual_cone_radii[cone_id]
                source = "manual"
            elif (cone_id in self._cone_bbox_sizes
                  and self._cone_bbox_sizes[cone_id] > 0):
                radius = (self._cone_bbox_sizes[cone_id]
                          * self.cone_radius_multiplier)
                source = "dynamic"
            else:
                radius = self.default_cone_radius
                source = "default"

            radius = max(self.min_cone_radius,
                         min(self.max_cone_radius, radius))
            self._cone_radii[cone_id] = radius

            if debug:
                bbox_s = self._cone_bbox_sizes.get(cone_id, 0)
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
            'top_to_bottom': lambda x: x[1][1],
            'bottom_to_top': lambda x: -x[1][1],
            'left_to_right': lambda x: x[1][0],
            'right_to_left': lambda x: -x[1][0],
        }

        key_fn = sort_key.get(direction, lambda x: x[1][1])
        paired = sorted(zip(cone_ids, positions), key=key_fn)
        self._ordered_cone_ids = [cid for cid, _ in paired]

    # ============================================================
    # HELPERS
    # ============================================================

    def _get_ball_pos(
        self, tracks: Dict, frame_num: int
    ) -> Optional[Tuple[float, float]]:
        ball_data = tracks['ball'][frame_num].get(1)
        if ball_data and 'bbox' in ball_data:
            return get_center_of_bbox_bottom(ball_data['bbox'])
        return None

    def _get_ball_bbox(
        self, tracks: Dict, frame_num: int
    ) -> Optional[List[float]]:
        ball_data = tracks['ball'][frame_num].get(1)
        if ball_data and 'bbox' in ball_data:
            return ball_data['bbox']
        return None

    def _get_player_pos(
        self, tracks: Dict, frame_num: int
    ) -> Optional[Tuple[int, Tuple[float, float]]]:
        players = tracks['players'][frame_num]
        for pid, pdata in players.items():
            bbox = pdata.get('bbox')
            if bbox:
                pos = get_center_of_bbox_bottom(bbox)
                return pid, pos
        return None

    # ============================================================
    # === BARU: METODE 1 — CONE DISPLACEMENT DETECTION ===
    # ============================================================

    def _detect_cone_displacement(
        self,
        tracks   : Dict,
        cone_key : str,
        debug    : bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Deteksi cone yang tertabrak berdasarkan pergeseran posisi bbox.

        PERBAIKAN:
        - Track displacement per RAW cone ID secara terpisah,
          bukan campur semua raw ID dalam cluster yang sama.
        - Ini mencegah false positive dari tracker ID switching.
        - Gunakan median posisi sebagai baseline, bukan frame-to-frame.
        """
        if not self._stabilized_cones:
            return {}

        total_frames = len(tracks.get(cone_key, []))
        results: Dict[int, Dict[str, Any]] = {
            cid: {
                'displaced': False,
                'max_displacement': 0.0,
                'displacement_frame': -1,
                'ball_nearby': False,
            }
            for cid in self._stabilized_cones
        }

        # Kumpulkan posisi per RAW cone ID (bukan per stabilized)
        raw_cone_positions: Dict[int, Dict[int, Tuple[float, float]]] = {}

        for f in range(total_frames):
            if cone_key not in tracks or f >= len(tracks[cone_key]):
                continue
            for raw_cone_id, cone_data in tracks[cone_key][f].items():
                bbox = cone_data.get('bbox')
                if bbox is None:
                    continue
                pos = get_center_of_bbox_bottom(bbox)
                if raw_cone_id not in raw_cone_positions:
                    raw_cone_positions[raw_cone_id] = {}
                raw_cone_positions[raw_cone_id][f] = pos

        window = self.cone_displacement_window
        threshold = self.cone_displacement_threshold

        # Per stabilized cone, cek displacement dari SETIAP raw ID-nya
        for scid, raw_ids in self._cone_id_mapping.items():
            best_displacement = 0.0
            best_frame = -1
            best_ball_nearby = False

            for raw_id in raw_ids:
                if raw_id not in raw_cone_positions:
                    continue

                frame_positions = raw_cone_positions[raw_id]
                if len(frame_positions) < 3:
                    continue

                sorted_frames = sorted(frame_positions.keys())

                # Hitung baseline posisi (median dari semua posisi raw ID ini)
                all_x = [frame_positions[f][0] for f in sorted_frames]
                all_y = [frame_positions[f][1] for f in sorted_frames]
                baseline_pos = (float(np.median(all_x)), float(np.median(all_y)))

                # Cek setiap frame: apakah posisi menyimpang jauh dari baseline?
                for f in sorted_frames:
                    pos = frame_positions[f]
                    deviation = measure_distance(pos, baseline_pos)

                    if deviation > threshold and deviation > best_displacement:
                        # Cek apakah bola dekat saat displacement
                        ball_nearby = False
                        for bf in range(
                            max(0, f - 3),
                            min(total_frames, f + 4)
                        ):
                            ball_pos = self._get_ball_pos(tracks, bf)
                            if ball_pos:
                                ball_cone_dist = measure_distance(
                                    ball_pos,
                                    self._stabilized_cones[scid]
                                )
                                if ball_cone_dist < self.cone_displacement_ball_proximity:
                                    ball_nearby = True
                                    break

                        best_displacement = deviation
                        best_frame = f
                        best_ball_nearby = ball_nearby

            results[scid]['max_displacement'] = best_displacement
            results[scid]['displacement_frame'] = best_frame
            results[scid]['ball_nearby'] = best_ball_nearby

            if best_displacement > threshold and best_ball_nearby:
                results[scid]['displaced'] = True

        if debug:
            print(f"[DRIBBLE] --- Cone Displacement Results ---")
            print(f"[DRIBBLE]   Threshold: {threshold}px (per raw ID, baseline=median)")
            for cid in self._ordered_cone_ids:
                r = results[cid]
                status = "DISPLACED!" if r['displaced'] else "stable"
                raw_ids = self._cone_id_mapping.get(cid, [])
                print(f"[DRIBBLE]   Cone {cid} (raw={raw_ids}): {status} "
                      f"(max_shift={r['max_displacement']:.1f}px, "
                      f"frame={r['displacement_frame']}, "
                      f"ball_near={r['ball_nearby']})")

        return results

    # ============================================================
    # === BARU: METODE 2 — BBOX OVERLAP DETECTION ===
    # ============================================================

    def _detect_bbox_overlap(
        self,
        tracks   : Dict,
        cone_key : str,
        debug    : bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Deteksi sentuhan berdasarkan overlap bounding box bola dan cone.

        PERBAIKAN untuk zig-zag:
        - Gunakan IoU (Intersection over Union) bukan hanya any overlap
        - Minimum IoU threshold untuk menghindari false positive
          dari bola yang lewat sangat dekat tapi tidak benar-benar menabrak
        - Shrink cone bbox lebih agresif
        """
        if not self._stabilized_cones:
            return {}

        total_frames = len(tracks.get(cone_key, []))
        shrink = self.bbox_overlap_shrink
        min_consec = self.min_overlap_consecutive_frames
        min_iou = self.bbox_overlap_min_iou

        results: Dict[int, Dict[str, Any]] = {
            cid: {
                'overlapped': False,
                'overlap_count': 0,
                'max_consecutive': 0,
                'max_iou': 0.0,
                'first_overlap_frame': -1,
            }
            for cid in self._stabilized_cones
        }

        consecutive: Dict[int, int] = {
            cid: 0 for cid in self._stabilized_cones
        }

        for f in range(total_frames):
            ball_bbox = self._get_ball_bbox(tracks, f)
            if ball_bbox is None:
                for cid in self._stabilized_cones:
                    consecutive[cid] = 0
                continue

            bx1, by1, bx2, by2 = ball_bbox

            # Dapatkan cone bboxes di frame ini
            cone_bboxes_this_frame: Dict[int, List[float]] = {}
            if cone_key in tracks and f < len(tracks[cone_key]):
                for raw_cone_id, cone_data in tracks[cone_key][f].items():
                    cbbox = cone_data.get('bbox')
                    if cbbox is None:
                        continue
                    pos = get_center_of_bbox_bottom(cbbox)
                    for scid, spos in self._stabilized_cones.items():
                        if raw_cone_id in self._cone_id_mapping.get(scid, []):
                            cone_bboxes_this_frame[scid] = cbbox
                            break
                        dist = measure_distance(pos, spos)
                        if dist < self.cone_dedup_distance * 1.5:
                            cone_bboxes_this_frame[scid] = cbbox
                            break

            for scid in self._stabilized_cones:
                if scid not in cone_bboxes_this_frame:
                    consecutive[scid] = 0
                    continue

                cx1, cy1, cx2, cy2 = cone_bboxes_this_frame[scid]

                # Shrink cone bbox
                cx1 += shrink
                cy1 += shrink
                cx2 -= shrink
                cy2 -= shrink

                if cx1 >= cx2 or cy1 >= cy2:
                    consecutive[scid] = 0
                    continue

                # Hitung intersection
                ix1 = max(bx1, cx1)
                iy1 = max(by1, cy1)
                ix2 = min(bx2, cx2)
                iy2 = min(by2, cy2)

                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    ball_area = max(1, (bx2 - bx1) * (by2 - by1))
                    cone_area = max(1, (cx2 - cx1) * (cy2 - cy1))
                    union = ball_area + cone_area - intersection
                    iou = intersection / union if union > 0 else 0.0

                    # Track max IoU
                    if iou > results[scid]['max_iou']:
                        results[scid]['max_iou'] = iou

                    # Hanya hitung sebagai overlap jika IoU cukup tinggi
                    if iou >= min_iou:
                        results[scid]['overlap_count'] += 1
                        consecutive[scid] += 1

                        if results[scid]['first_overlap_frame'] == -1:
                            results[scid]['first_overlap_frame'] = f

                        if consecutive[scid] > results[scid]['max_consecutive']:
                            results[scid]['max_consecutive'] = consecutive[scid]

                        if consecutive[scid] >= min_consec:
                            results[scid]['overlapped'] = True
                    else:
                        consecutive[scid] = 0
                else:
                    consecutive[scid] = 0

        if debug:
            print(f"[DRIBBLE] --- BBox Overlap Results ---")
            print(f"[DRIBBLE]   Shrink={shrink}px, min_IoU={min_iou}, "
                  f"min_consec={min_consec}")
            for cid in self._ordered_cone_ids:
                r = results[cid]
                status = "OVERLAP!" if r['overlapped'] else "clear"
                print(f"[DRIBBLE]   Cone {cid}: {status} "
                      f"(count={r['overlap_count']}, "
                      f"max_consec={r['max_consecutive']}, "
                      f"max_iou={r['max_iou']:.3f}, "
                      f"first_frame={r['first_overlap_frame']})")

        return results


    # ============================================================
    # === BARU: METODE 3 — BALL SPEED ANOMALY ===
    # ============================================================

    def _detect_speed_anomaly(
        self,
        tracks: Dict,
        debug : bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Deteksi sentuhan berdasarkan perlambatan tiba-tiba bola
        saat berada dekat cone.

        PERBAIKAN:
        - Abaikan speed=0 (artefak interpolasi)
        - Gunakan median bukan mean untuk average (lebih robust)
        - Cek speed drop harus sustained (bukan 1 frame saja)
        """
        if not self._stabilized_cones:
            return {}

        total_frames = len(tracks['ball'])
        results: Dict[int, Dict[str, Any]] = {
            cid: {
                'speed_anomaly': False,
                'min_speed_near_cone': float('inf'),
                'avg_speed': 0.0,
                'anomaly_frame': -1,
            }
            for cid in self._stabilized_cones
        }

        # Hitung ball positions
        ball_positions: List[Optional[Tuple[float, float]]] = []
        for f in range(total_frames):
            ball_positions.append(self._get_ball_pos(tracks, f))

        # Hitung ball speed per frame
        ball_speeds: List[float] = [0.0]
        for f in range(1, total_frames):
            if ball_positions[f] and ball_positions[f - 1]:
                speed = measure_distance(ball_positions[f], ball_positions[f - 1])
                ball_speeds.append(speed)
            else:
                ball_speeds.append(0.0)

        # Hitung global average speed — ABAIKAN speed=0
        valid_speeds = [s for s in ball_speeds if s > 1.0]  # threshold > 1px
        if not valid_speeds or len(valid_speeds) < 5:
            if debug:
                print(f"[DRIBBLE] Speed anomaly: tidak cukup data speed valid")
            return results

        global_avg_speed = float(np.median(valid_speeds))  # Median lebih robust

        # Per cone, cek apakah ada SUSTAINED speed drop saat bola dekat
        proximity = self.speed_anomaly_proximity
        drop_ratio = self.speed_drop_ratio
        min_slow_frames = 3  # Harus lambat minimal 3 frame berturut-turut

        for scid, cone_pos in self._stabilized_cones.items():
            min_speed_near = float('inf')
            anomaly_frame = -1
            consecutive_slow = 0
            max_consecutive_slow = 0

            for f in range(total_frames):
                if ball_positions[f] is None:
                    consecutive_slow = 0
                    continue

                dist_to_cone = measure_distance(ball_positions[f], cone_pos)
                if dist_to_cone > proximity:
                    consecutive_slow = 0
                    continue

                # Bola dekat cone
                speed = ball_speeds[f]

                # Abaikan speed=0 (artefak interpolasi/duplikat frame)
                if speed < 1.0:
                    continue

                # Track minimum speed
                if speed < min_speed_near:
                    min_speed_near = speed

                # Hitung local average (window sebelumnya, abaikan 0)
                start_w = max(0, f - self.speed_avg_window)
                local_speeds = [
                    ball_speeds[k]
                    for k in range(start_w, f)
                    if ball_speeds[k] > 1.0
                ]
                local_avg = (
                    float(np.median(local_speeds))
                    if len(local_speeds) >= 3
                    else global_avg_speed
                )

                # Cek speed drop
                if local_avg > 0 and speed < local_avg * drop_ratio:
                    consecutive_slow += 1
                    if consecutive_slow > max_consecutive_slow:
                        max_consecutive_slow = consecutive_slow
                        anomaly_frame = f
                    if consecutive_slow >= min_slow_frames:
                        results[scid]['speed_anomaly'] = True
                else:
                    consecutive_slow = 0

            results[scid]['min_speed_near_cone'] = (
                min_speed_near if min_speed_near != float('inf') else 0.0
            )
            results[scid]['avg_speed'] = global_avg_speed
            results[scid]['anomaly_frame'] = anomaly_frame

        if debug:
            print(f"[DRIBBLE] --- Speed Anomaly Results ---")
            print(f"[DRIBBLE]   Global median speed: {global_avg_speed:.1f}px/frame")
            print(f"[DRIBBLE]   Min slow frames: {min_slow_frames}")
            for cid in self._ordered_cone_ids:
                r = results[cid]
                status = "ANOMALY!" if r['speed_anomaly'] else "normal"
                print(f"[DRIBBLE]   Cone {cid}: {status} "
                      f"(min_speed_near={r['min_speed_near_cone']:.1f}, "
                      f"frame={r['anomaly_frame']})")

        return results

    # ============================================================
    # === BARU: KOMBINASI SEMUA METODE ===
    # ============================================================

    def _evaluate_cone_touches_zigzag(
        self,
        tracks  : Dict,
        cone_key: str = 'cones',
        debug   : bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluasi sentuhan cone dengan menggabungkan semua metode.
        
        Setiap metode memberikan "vote". Cone dianggap disentuh jika
        jumlah vote >= min_methods_agree.
        """
        if debug:
            print(f"\n[DRIBBLE] === EVALUASI SENTUHAN CONE (ZIG-ZAG MODE) ===")

        # Jalankan semua metode yang aktif
        displacement_results = {}
        overlap_results = {}
        speed_results = {}

        if self.use_cone_displacement:
            displacement_results = self._detect_cone_displacement(
                tracks, cone_key, debug
            )

        if self.use_bbox_overlap:
            overlap_results = self._detect_bbox_overlap(
                tracks, cone_key, debug
            )

        if self.use_speed_anomaly:
            speed_results = self._detect_speed_anomaly(tracks, debug)

        # Kombinasi
        combined: Dict[int, Dict[str, Any]] = {}

        for cid in self._stabilized_cones:
            votes = 0
            methods_triggered = []

            if self.use_cone_displacement:
                if displacement_results.get(cid, {}).get('displaced', False):
                    votes += 1
                    methods_triggered.append('displacement')

            if self.use_bbox_overlap:
                if overlap_results.get(cid, {}).get('overlapped', False):
                    votes += 1
                    methods_triggered.append('bbox_overlap')

            if self.use_speed_anomaly:
                if speed_results.get(cid, {}).get('speed_anomaly', False):
                    votes += 1
                    methods_triggered.append('speed_anomaly')

            touched = votes >= self.min_methods_agree

            combined[cid] = {
                'touched': touched,
                'votes': votes,
                'methods_triggered': methods_triggered,
                'displacement': displacement_results.get(cid, {}),
                'bbox_overlap': overlap_results.get(cid, {}),
                'speed_anomaly': speed_results.get(cid, {}),
                'radius': self._cone_radii.get(cid, 0),
            }

        if debug:
            print(f"\n[DRIBBLE] --- Hasil Kombinasi ---")
            print(f"[DRIBBLE] Min methods agree: {self.min_methods_agree}")
            for cid in self._ordered_cone_ids:
                r = combined[cid]
                status = "HIT" if r['touched'] else "AMAN"
                methods_str = (", ".join(r['methods_triggered'])
                               if r['methods_triggered'] else "none")
                print(f"[DRIBBLE]   Cone {cid}: {status} "
                      f"(votes={r['votes']}/{self.min_methods_agree}, "
                      f"methods=[{methods_str}])")

        return combined

    # ============================================================
    # AUTO-DETECT ARAH DRIBBLE
    # ============================================================

    def _auto_detect_dribble_direction(
        self, tracks: Dict, debug: bool = True
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
            print(f"[DRIBBLE] Posisi pemain awal : "
                  f"({early_avg[0]:.0f}, {early_avg[1]:.0f})")
            print(f"[DRIBBLE] Posisi pemain akhir: "
                  f"({late_avg[0]:.0f}, {late_avg[1]:.0f})")

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
                self.cone_order_direction = (
                    'left_to_right' if dx > 0 else 'right_to_left'
                )
            else:
                self.cone_order_direction = (
                    'top_to_bottom' if dy > 0 else 'bottom_to_top'
                )

            if debug:
                print(f"[DRIBBLE] Arah dribble: {self.cone_order_direction}")

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
                    print(f"[DRIBBLE] Video pendek ({video_duration:.1f}s) "
                          f"→ mode: whole_video")
            else:
                mode = 'entry_exit'
                if debug:
                    print(f"[DRIBBLE] Video panjang ({video_duration:.1f}s) "
                          f"→ mode: entry_exit")

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

        # Kumpulkan data bola
        ball_trajectory: List[Tuple[float, float]] = []
        frames_with_ball = 0
        first_possession_frame = -1
        last_possession_frame = -1
        player_id_main = -1

        for frame_num in range(total_frames):
            ball_pos = self._get_ball_pos(tracks, frame_num)

            has_ball = (
                frame_num < len(ball_possessions)
                and ball_possessions[frame_num] != -1
            )

            if ball_pos:
                ball_trajectory.append(ball_pos)
            else:
                ball_trajectory.append(
                    ball_trajectory[-1] if ball_trajectory else (0, 0)
                )

            if has_ball:
                frames_with_ball += 1
                if first_possession_frame == -1:
                    first_possession_frame = frame_num
                last_possession_frame = frame_num

                if player_id_main == -1:
                    player_result = self._get_player_pos(tracks, frame_num)
                    if player_result:
                        player_id_main = player_result[0]

        if first_possession_frame == -1:
            if debug:
                print("[DRIBBLE] Tidak ada possession terdeteksi!")
            return []

        # === GUNAKAN METODE ZIG-ZAG ===
        cone_results = self._evaluate_cone_touches_zigzag(
            tracks, cone_key='cones', debug=debug
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
            'attempt_id'      : 1,
            'player_id'       : player_id_main,
            'frame_start'     : first_possession_frame,
            'frame_end'       : last_possession_frame,
            'duration_frames' : duration_frames,
            'duration_seconds': round(duration_frames / self.fps, 2),
            'success'         : success,
            'total_cones'     : len(self._stabilized_cones),
            'touched_cones'   : touched_cones,
            'cone_details'    : cone_results,
            'possession_ratio': round(possession_ratio, 2),
            'ball_trajectory' : list(ball_trajectory),
        }

        if debug:
            status = "SUKSES ✓" if success else "GAGAL ✗"
            print(f"\n[DRIBBLE] Hasil: {status}")
            print(f"[DRIBBLE]   Frame range  : "
                  f"{first_possession_frame}-{last_possession_frame}")
            print(f"[DRIBBLE]   Durasi       : {duration_frames} frames "
                  f"({duration_frames / self.fps:.1f}s)")
            print(f"[DRIBBLE]   Possession   : {possession_ratio * 100:.0f}%")
            print(f"[DRIBBLE]   Cone disentuh: "
                  f"{len(touched_cones)}/{len(self._stabilized_cones)}")

            for cid in self._ordered_cone_ids:
                res = cone_results[cid]
                status_c = "HIT" if res['touched'] else "AMAN"
                methods = res.get('methods_triggered', [])
                methods_str = ", ".join(methods) if methods else "-"
                print(f"[DRIBBLE]     Cone {cid}: {status_c} "
                      f"(votes={res.get('votes', 0)}, "
                      f"methods=[{methods_str}])")

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
        """Entry/exit mode — untuk video panjang dengan multiple attempts."""
        total_frames = len(tracks['players'])
        entry_cone_id = self._ordered_cone_ids[0]
        exit_cone_id = self._ordered_cone_ids[-1]
        entry_pos = self._stabilized_cones[entry_cone_id]
        exit_pos = self._stabilized_cones[exit_cone_id]
        max_attempt_frames = int(self.max_attempt_duration_sec * self.fps)

        if debug:
            print(f"\n[DRIBBLE] === DRIBBLE DETECTION (ENTRY/EXIT) ===")
            print(f"[DRIBBLE] Entry cone {entry_cone_id}: "
                  f"({entry_pos[0]:.0f}, {entry_pos[1]:.0f})")
            print(f"[DRIBBLE] Exit cone  {exit_cone_id}: "
                  f"({exit_pos[0]:.0f}, {exit_pos[1]:.0f})")

        # Untuk entry/exit mode, gunakan evaluasi zig-zag juga
        cone_results = self._evaluate_cone_touches_zigzag(
            tracks, cone_key='cones', debug=debug
        )

        touched_cones = [
            cid for cid, res in cone_results.items()
            if res['touched']
        ]

        # Deteksi attempt boundaries (entry → exit)
        attempts: List[Dict] = []
        last_attempt_end = -999
        state = 'idle'
        attempt_start_frame = -1
        attempt_player_id = -1
        frames_with_ball = 0
        attempt_total_frames = 0

        for frame_num in range(total_frames):
            player_result = self._get_player_pos(tracks, frame_num)
            if player_result is None:
                continue
            player_id, player_pos = player_result

            has_ball = (
                frame_num < len(ball_possessions)
                and ball_possessions[frame_num] != -1
            )

            if state == 'idle':
                if has_ball:
                    dist_to_entry = measure_distance(player_pos, entry_pos)
                    if (dist_to_entry <= self.entry_exit_zone_radius
                            and (frame_num - last_attempt_end)
                            >= self.cooldown_frames):
                        state = 'dribbling'
                        attempt_start_frame = frame_num
                        attempt_player_id = player_id
                        frames_with_ball = 1
                        attempt_total_frames = 1
                        if debug:
                            print(f"[DRIBBLE] Frame {frame_num}: "
                                  f"MASUK entry zone")

            elif state == 'dribbling':
                attempt_total_frames += 1
                if has_ball:
                    frames_with_ball += 1

                dist_to_exit = measure_distance(player_pos, exit_pos)

                if dist_to_exit <= self.entry_exit_zone_radius:
                    duration_frames = frame_num - attempt_start_frame
                    if duration_frames >= self.min_attempt_frames:
                        success = len(touched_cones) == 0
                        possession_ratio = (
                            frames_with_ball / attempt_total_frames
                            if attempt_total_frames > 0 else 0.0
                        )
                        attempt = {
                            'attempt_id'      : len(attempts) + 1,
                            'player_id'       : attempt_player_id,
                            'frame_start'     : attempt_start_frame,
                            'frame_end'       : frame_num,
                            'duration_frames' : duration_frames,
                            'duration_seconds': round(
                                duration_frames / self.fps, 2
                            ),
                            'success'         : success,
                            'total_cones'     : len(self._stabilized_cones),
                            'touched_cones'   : touched_cones,
                            'cone_details'    : cone_results,
                            'possession_ratio': round(possession_ratio, 2),
                        }
                        attempts.append(attempt)
                        last_attempt_end = frame_num

                        if debug:
                            s = "SUKSES ✓" if success else "GAGAL ✗"
                            print(f"[DRIBBLE] Frame {frame_num}: "
                                  f"SELESAI → {s}")

                    state = 'idle'
                    continue

                if attempt_total_frames > max_attempt_frames:
                    state = 'idle'
                    continue

                if attempt_total_frames > 20:
                    current_ratio = frames_with_ball / attempt_total_frames
                    if current_ratio < self.min_possession_ratio:
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

        cone_analysis = {}
        for cid in self._ordered_cone_ids:
            cone_analysis[cid] = {
                'times_touched': cone_hit_count.get(cid, 0),
                'radius': self._cone_radii.get(cid, 0.0),
            }

        return {
            'total_attempts'     : total,
            'successful_attempts': len(sukses),
            'failed_attempts'    : len(gagal),
            'accuracy_pct'       : round(
                len(sukses) / total * 100, 1
            ) if total > 0 else 0.0,
            'avg_duration'       : round(float(np.mean(
                [a['duration_seconds'] for a in attempts]
            )), 2) if attempts else 0.0,
            'cone_hit_frequency' : cone_hit_count,
            'cone_analysis'      : cone_analysis,
        }
