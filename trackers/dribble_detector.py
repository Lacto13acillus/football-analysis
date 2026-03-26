# dribble_detector.py
# Deteksi dribbling melewati cone course untuk 1 pemain.
#
# LOGIKA:
#   - Setiap cone punya radius deteksi (dynamic dari bbox atau manual).
#   - Dribble attempt dimulai saat pemain+bola masuk entry zone (cone pertama).
#   - Dribble attempt selesai saat pemain+bola sampai exit zone (cone terakhir).
#   - Selama attempt, jika bola masuk radius cone manapun → GAGAL.
#   - Jika bola tidak masuk radius cone manapun → SUKSES.
#
# PENINGKATAN v2:
#   - Temporal Filtering: sentuhan harus >= N consecutive frames
#   - Ball Edge Distance: gunakan edge bbox bola, bukan hanya center
#   - Trajectory Interpolation: interpolasi posisi antar frame

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
        # PARAMETER RADIUS CONE
        # ============================================================
        self.default_cone_radius: float = 40.0
        self.cone_radius_multiplier: float = 1.5
        self.min_cone_radius: float = 20.0
        self.max_cone_radius: float = 80.0

        # Override radius manual per cone: {cone_id: radius}
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
        # PARAMETER TEMPORAL FILTERING (BARU)
        # ============================================================
        # Minimum consecutive frames bola di dalam radius cone
        # agar dianggap sebagai sentuhan valid.
        # Nilai 1 = tanpa filtering (seperti sebelumnya)
        # Nilai 2-3 = mengurangi false positive dari noise deteksi
        self.min_consecutive_touch_frames: int = 2

        # ============================================================
        # PARAMETER BALL EDGE DISTANCE (BARU)
        # ============================================================
        # Jika True, gunakan edge bounding box bola (bukan hanya center)
        # untuk menghitung jarak ke cone. Lebih akurat karena bola
        # punya ukuran fisik.
        self.use_ball_edge_distance: bool = True

        # ============================================================
        # PARAMETER TRAJECTORY INTERPOLATION (BARU)
        # ============================================================
        # Jumlah sub-step interpolasi antar frame berurutan.
        # Berguna untuk mendeteksi sentuhan ketika bola bergerak
        # cepat dan "melompat" melewati radius cone dalam 1 frame.
        # Nilai 1 = tanpa interpolasi, 3-5 = lebih akurat.
        self.interpolation_substeps: int = 3

        # ============================================================
        # PARAMETER CONE ORDERING
        # ============================================================
        self.cone_order_direction: str = 'auto'

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
        sample_frames: int = 30,
        debug        : bool = True
    ) -> bool:
        if cone_key not in tracks or not tracks[cone_key]:
            print(f"[DRIBBLE] WARNING: Key '{cone_key}' tidak ada di tracks!")
            return False

        if debug:
            print(f"\n[DRIBBLE] === INISIALISASI CONE ===")

        self._stabilized_cones = stabilize_cone_positions(
            tracks, cone_key=cone_key, sample_frames=sample_frames
        )

        if len(self._stabilized_cones) < 2:
            print("[DRIBBLE] GAGAL: Minimal 2 cone diperlukan!")
            return False

        self._compute_cone_bbox_sizes(tracks, cone_key, sample_frames)
        self._compute_cone_radii(debug)
        self._order_cones(debug)

        if debug:
            print(f"\n[DRIBBLE] Total cone       : {len(self._stabilized_cones)}")
            print(f"[DRIBBLE] Urutan cone IDs  : {self._ordered_cone_ids}")
            print(f"[DRIBBLE] Temporal filter   : {self.min_consecutive_touch_frames} frames")
            print(f"[DRIBBLE] Ball edge dist   : {'Ya' if self.use_ball_edge_distance else 'Tidak'}")
            print(f"[DRIBBLE] Interpolation    : {self.interpolation_substeps} substeps")
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

    def _compute_cone_bbox_sizes(
        self, tracks: Dict, cone_key: str, sample_frames: int
    ) -> None:
        size_acc: Dict[int, List[float]] = {}
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
                if cone_id not in size_acc:
                    size_acc[cone_id] = []
                size_acc[cone_id].append(max(w, h))

        for cone_id, sizes in size_acc.items():
            self._cone_bbox_sizes[cone_id] = float(np.mean(sizes))

    def _compute_cone_radii(self, debug: bool = False) -> None:
        for cone_id in self._stabilized_cones:
            if cone_id in self.manual_cone_radii:
                radius = self.manual_cone_radii[cone_id]
                source = "manual"
            elif cone_id in self._cone_bbox_sizes:
                radius = self._cone_bbox_sizes[cone_id] * self.cone_radius_multiplier
                source = "dynamic"
            else:
                radius = self.default_cone_radius
                source = "default"

            radius = max(self.min_cone_radius, min(self.max_cone_radius, radius))
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
            'top_to_bottom' : lambda x: x[1][1],
            'bottom_to_top' : lambda x: -x[1][1],
            'left_to_right' : lambda x: x[1][0],
            'right_to_left' : lambda x: -x[1][0],
        }

        key_fn = sort_key.get(direction, lambda x: x[1][1])
        paired = sorted(zip(cone_ids, positions), key=key_fn)
        self._ordered_cone_ids = [cid for cid, _ in paired]

    # ============================================================
    # HELPER: BALL EDGE DISTANCE (BARU)
    # ============================================================

    def _compute_ball_cone_distance(
        self,
        ball_pos: Tuple[float, float],
        cone_pos: Tuple[float, float],
        ball_bbox: Optional[List[float]] = None
    ) -> float:
        """
        Hitung jarak bola ke cone.

        Jika use_ball_edge_distance=True dan ball_bbox tersedia,
        gunakan titik terdekat dari edge bounding box bola ke cone.
        Ini lebih akurat karena bola punya ukuran fisik — center bola
        mungkin jauh dari cone tapi edge-nya sudah menyentuh.

        Args:
            ball_pos : center position bola (x, y)
            cone_pos : center position cone (x, y)
            ball_bbox: bounding box bola [x1, y1, x2, y2] (opsional)

        Returns:
            Jarak terpendek (float)
        """
        if not self.use_ball_edge_distance or ball_bbox is None:
            return measure_distance(ball_pos, cone_pos)

        # Cari titik terdekat pada edge bbox bola ke cone_pos
        bx1, by1, bx2, by2 = ball_bbox
        cx, cy = cone_pos

        # Clamp cone position ke dalam bbox → titik terdekat di bbox
        closest_x = max(bx1, min(cx, bx2))
        closest_y = max(by1, min(cy, by2))

        return measure_distance((closest_x, closest_y), cone_pos)

    # ============================================================
    # HELPER: TRAJECTORY INTERPOLATION (BARU)
    # ============================================================

    def _interpolate_positions(
        self,
        pos_a: Tuple[float, float],
        pos_b: Tuple[float, float],
        steps: int
    ) -> List[Tuple[float, float]]:
        """
        Interpolasi linier antara dua posisi.
        Berguna saat bola bergerak cepat dan bisa "melompat"
        melewati radius cone antara dua frame berurutan.

        Returns:
            List posisi termasuk pos_a dan pos_b
        """
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
        """
        Cek apakah posisi bola saat ini menyentuh radius cone manapun.

        Returns:
            List of (cone_id, distance) untuk cone yang disentuh
        """
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
        """
        Evaluasi seluruh trajectory bola terhadap semua cone.

        PENINGKATAN:
        1. Ball edge distance (jika ball_bboxes diberikan)
        2. Trajectory interpolation antar frame
        3. Temporal filtering — sentuhan hanya valid jika
           consecutive frames >= min_consecutive_touch_frames

        Returns:
            {cone_id: {
                'touched': bool,
                'min_distance': float,
                'touch_count': int,
                'consecutive_max': int,     # max consecutive frames di dalam radius
                'first_touch_idx': int,
                'radius': float
            }}
        """
        results: Dict[int, Dict[str, Any]] = {}

        for cone_id in self._stabilized_cones:
            radius = self._cone_radii.get(cone_id, self.default_cone_radius)
            cone_pos = self._stabilized_cones[cone_id]

            min_dist = float('inf')
            raw_touch_count = 0
            first_touch = -1

            # Untuk temporal filtering
            consecutive_count = 0
            max_consecutive = 0
            valid_touch = False

            for i, ball_pos in enumerate(ball_trajectory):
                # Ambil bbox bola jika tersedia
                bbox = None
                if ball_bboxes and i < len(ball_bboxes):
                    bbox = ball_bboxes[i]

                # === Trajectory Interpolation ===
                # Jika ada posisi sebelumnya, interpolasi antar frame
                positions_to_check = [(ball_pos, bbox)]

                if (i > 0 and self.interpolation_substeps > 1):
                    prev_pos = ball_trajectory[i - 1]
                    interp = self._interpolate_positions(
                        prev_pos, ball_pos, self.interpolation_substeps
                    )
                    # Skip first (=prev_pos, sudah dicek di iterasi sebelumnya)
                    # dan last (=ball_pos, akan dicek sebagai posisi utama)
                    for ip in interp[1:-1]:
                        positions_to_check.append((ip, None))

                # Cek semua posisi (utama + interpolasi)
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

                    # === Temporal Filtering ===
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
        """Ambil bounding box bola di frame tertentu."""
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
    # DETEKSI DRIBBLE ATTEMPTS (CORE)
    # ============================================================

    def detect_dribble_attempts(
        self,
        tracks          : Dict,
        ball_possessions: List[int],
        debug           : bool = True
    ) -> List[Dict]:
        """
        Deteksi semua dribble attempt dan evaluasi sentuhan cone.

        Returns:
            List[Dict] dribble attempts
        """
        if not self._stabilized_cones or len(self._ordered_cone_ids) < 2:
            print("[DRIBBLE] ERROR: Cone belum diinisialisasi atau < 2!")
            return []

        total_frames = len(tracks['players'])
        entry_cone_id = self._ordered_cone_ids[0]
        exit_cone_id = self._ordered_cone_ids[-1]
        entry_pos = self._stabilized_cones[entry_cone_id]
        exit_pos = self._stabilized_cones[exit_cone_id]
        max_attempt_frames = int(self.max_attempt_duration_sec * self.fps)

        if debug:
            print(f"\n[DRIBBLE] === DRIBBLE DETECTION ===")
            print(f"[DRIBBLE] Total frames           : {total_frames}")
            print(f"[DRIBBLE] Total cones             : {len(self._stabilized_cones)}")
            print(f"[DRIBBLE] Entry cone {entry_cone_id}: "
                  f"({entry_pos[0]:.0f}, {entry_pos[1]:.0f})")
            print(f"[DRIBBLE] Exit cone  {exit_cone_id}: "
                  f"({exit_pos[0]:.0f}, {exit_pos[1]:.0f})")
            print(f"[DRIBBLE] Entry/exit zone radius  : {self.entry_exit_zone_radius}px")
            print(f"[DRIBBLE] Max attempt duration    : {self.max_attempt_duration_sec}s")
            print(f"[DRIBBLE] Temporal filter         : {self.min_consecutive_touch_frames} frames")

        attempts: List[Dict] = []
        last_attempt_end = -999

        # State machine
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

            # ---- STATE: IDLE ----
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

            # ---- STATE: DRIBBLING ----
            elif state == 'dribbling':
                attempt_total_frames += 1

                if ball_pos:
                    ball_trajectory.append(ball_pos)
                    ball_bboxes.append(ball_bbox)
                player_positions.append(player_pos)
                if has_ball:
                    frames_with_ball += 1

                # Cek apakah sampai exit zone
                dist_to_exit = measure_distance(player_pos, exit_pos)

                if dist_to_exit <= self.entry_exit_zone_radius:
                    duration_frames = frame_num - attempt_start_frame

                    if duration_frames >= self.min_attempt_frames:
                        # Evaluasi sentuhan cone dengan peningkatan
                        cone_results = self.evaluate_trajectory_against_cones(
                            ball_trajectory,
                            ball_bboxes
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
                            status = "SUKSES ✓" if success else "GAGAL ✗"
                            print(f"[DRIBBLE] Frame {frame_num}: "
                                  f"SELESAI → {status}")
                            print(f"[DRIBBLE]   Durasi       : "
                                  f"{duration_frames} frames "
                                  f"({duration_frames/self.fps:.1f}s)")
                            print(f"[DRIBBLE]   Cone disentuh: "
                                  f"{len(touched_cones)}/{len(self._stabilized_cones)}")
                            for tc in touched_cones:
                                d = cone_results[tc]['min_distance']
                                r = cone_results[tc]['radius']
                                cnt = cone_results[tc]['touch_count']
                                consec = cone_results[tc]['consecutive_max']
                                print(f"[DRIBBLE]     Cone {tc}: "
                                      f"min_dist={d:.1f}px "
                                      f"(radius={r:.0f}px, "
                                      f"{cnt} frame total, "
                                      f"max_consecutive={consec})")
                            print(f"[DRIBBLE]   Possession   : "
                                  f"{possession_ratio*100:.0f}%")
                    else:
                        if debug:
                            print(f"[DRIBBLE] Frame {frame_num}: "
                                  f"attempt terlalu singkat "
                                  f"({duration_frames} < {self.min_attempt_frames})")

                    state = 'idle'
                    continue

                # Timeout
                if attempt_total_frames > max_attempt_frames:
                    if debug:
                        print(f"[DRIBBLE] Frame {frame_num}: TIMEOUT → reset")
                    state = 'idle'
                    continue

                # Kehilangan bola
                if attempt_total_frames > 20:
                    current_ratio = frames_with_ball / attempt_total_frames
                    if current_ratio < self.min_possession_ratio:
                        if debug:
                            print(f"[DRIBBLE] Frame {frame_num}: "
                                  f"possession rendah ({current_ratio:.0%}) → BATAL")
                        state = 'idle'
                        continue

        # Summary
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
