# dribble_detector.py
# Deteksi dribbling melewati cone course untuk 1 pemain.
#
# LOGIKA:
#   - Setiap cone punya radius deteksi (dynamic dari bbox atau manual).
#   - Dribble attempt dimulai saat pemain+bola masuk entry zone (cone pertama).
#   - Dribble attempt selesai saat pemain+bola sampai exit zone (cone terakhir).
#   - Selama attempt, jika bola masuk radius cone manapun → GAGAL.
#   - Jika bola tidak masuk radius cone manapun → SUKSES.

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

        # Bisa override radius manual per cone: {cone_id: radius}
        # Jika diset, akan MENGGANTIKAN dynamic radius untuk cone itu
        self.manual_cone_radii: Dict[int, float] = {}

        # ============================================================
        # PARAMETER DRIBBLE ATTEMPT DETECTION
        # ============================================================
        self.max_dribble_ball_distance: float = 130.0
        self.entry_exit_zone_radius: float = 150.0
        self.min_attempt_frames: int = 15
        self.cooldown_frames: int = 30
        self.max_attempt_duration_sec: float = 30.0

        # Minimum possession ratio selama attempt (0.0 - 1.0)
        # Jika possession < threshold → attempt dibatalkan (bola lepas)
        self.min_possession_ratio: float = 0.3

        # ============================================================
        # PARAMETER CONE ORDERING
        # 'auto', 'top_to_bottom', 'bottom_to_top',
        # 'left_to_right', 'right_to_left'
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
        """
        Stabilisasi posisi cone, hitung radius, tentukan urutan.

        Returns:
            True jika berhasil (minimal 2 cone terdeteksi)
        """
        if cone_key not in tracks or not tracks[cone_key]:
            print(f"[DRIBBLE] WARNING: Key '{cone_key}' tidak ada di tracks!")
            return False

        if debug:
            print(f"\n[DRIBBLE] === INISIALISASI CONE ===")

        # Stabilisasi posisi rata-rata
        self._stabilized_cones = stabilize_cone_positions(
            tracks, cone_key=cone_key, sample_frames=sample_frames
        )

        if len(self._stabilized_cones) < 2:
            print("[DRIBBLE] GAGAL: Minimal 2 cone diperlukan!")
            return False

        # Hitung bbox size per cone (untuk dynamic radius)
        self._compute_cone_bbox_sizes(tracks, cone_key, sample_frames)

        # Hitung radius per cone
        self._compute_cone_radii(debug)

        # Tentukan urutan cone
        self._order_cones(debug)

        if debug:
            print(f"\n[DRIBBLE] Total cone       : {len(self._stabilized_cones)}")
            print(f"[DRIBBLE] Urutan cone IDs  : {self._ordered_cone_ids}")
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
        """Hitung rata-rata ukuran bbox per cone."""
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
        """
        Hitung radius deteksi per cone.
        Prioritas: manual_cone_radii > dynamic dari bbox > default
        """
        for cone_id in self._stabilized_cones:
            # Prioritas 1: manual override
            if cone_id in self.manual_cone_radii:
                radius = self.manual_cone_radii[cone_id]
                source = "manual"
            # Prioritas 2: dynamic dari bbox
            elif cone_id in self._cone_bbox_sizes:
                radius = self._cone_bbox_sizes[cone_id] * self.cone_radius_multiplier
                source = "dynamic"
            # Prioritas 3: default
            else:
                radius = self.default_cone_radius
                source = "default"

            # Clamp
            radius = max(self.min_cone_radius, min(self.max_cone_radius, radius))
            self._cone_radii[cone_id] = radius

            if debug:
                bbox_s = self._cone_bbox_sizes.get(cone_id, 0)
                print(f"[DRIBBLE] Cone {cone_id}: bbox_size={bbox_s:.0f}px "
                      f"→ radius={radius:.0f}px ({source})")

    def _order_cones(self, debug: bool = False) -> None:
        """Urutkan cone berdasarkan arah konfigurasi atau auto-detect."""
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
    # CEK SENTUHAN BOLA KE CONE
    # ============================================================

    def check_ball_touches_any_cone(
        self,
        ball_pos: Tuple[float, float]
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
            dist = measure_distance(ball_pos, cone_pos)
            if dist <= radius:
                touched.append((cone_id, dist))

        return touched

    def evaluate_trajectory_against_cones(
        self,
        ball_trajectory: List[Tuple[float, float]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluasi seluruh trajectory bola terhadap semua cone.

        Returns:
            {cone_id: {
                'touched': bool,
                'min_distance': float,
                'touch_count': int,        # berapa frame bola di dalam radius
                'first_touch_idx': int,    # index frame pertama kali menyentuh
                'radius': float
            }}
        """
        results: Dict[int, Dict[str, Any]] = {}

        for cone_id in self._stabilized_cones:
            radius = self._cone_radii.get(cone_id, self.default_cone_radius)
            cone_pos = self._stabilized_cones[cone_id]

            min_dist = float('inf')
            touch_count = 0
            first_touch = -1

            for i, ball_pos in enumerate(ball_trajectory):
                dist = measure_distance(ball_pos, cone_pos)
                if dist < min_dist:
                    min_dist = dist
                if dist <= radius:
                    touch_count += 1
                    if first_touch == -1:
                        first_touch = i

            results[cone_id] = {
                'touched'         : touch_count > 0,
                'min_distance'    : min_dist,
                'touch_count'     : touch_count,
                'first_touch_idx' : first_touch,
                'radius'          : radius,
            }

        return results

    # ============================================================
    # HELPERS
    # ============================================================

    def _get_ball_pos(self, tracks: Dict, frame_num: int) -> Optional[Tuple[float, float]]:
        """Posisi bola (bottom-center) di frame tertentu."""
        ball_data = tracks['ball'][frame_num].get(1)
        if ball_data and 'bbox' in ball_data:
            return get_center_of_bbox_bottom(ball_data['bbox'])
        return None

    def _get_player_pos(self, tracks: Dict, frame_num: int) -> Optional[Tuple[int, Tuple[float, float]]]:
        """
        Dapatkan posisi pemain pertama yang ditemukan di frame.
        Karena hanya 1 pemain, ambil siapa saja.

        Returns:
            (player_id, (x, y)) atau None
        """
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

        Algoritma (State Machine):
        ┌──────────┐    pemain+bola dekat    ┌────────────┐
        │   IDLE   │ ──── entry cone ──────► │ DRIBBLING  │
        └──────────┘                         └─────┬──────┘
              ▲                                    │
              │          pemain+bola dekat         │
              └──────── exit cone / timeout ◄──────┘
                              │
                        evaluasi cone
                        SUKSES / GAGAL

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

        attempts: List[Dict] = []
        last_attempt_end = -999

        # State
        state = 'idle'
        attempt_start_frame = -1
        attempt_player_id = -1
        ball_trajectory: List[Tuple[float, float]] = []
        player_positions: List[Tuple[float, float]] = []
        frames_with_ball = 0
        attempt_total_frames = 0

        for frame_num in range(total_frames):
            # Posisi pemain
            player_result = self._get_player_pos(tracks, frame_num)
            if player_result is None:
                continue
            player_id, player_pos = player_result

            # Posisi bola
            ball_pos = self._get_ball_pos(tracks, frame_num)

            # Possession
            has_ball = (
                frame_num < len(ball_possessions) and
                ball_possessions[frame_num] != -1
            )

            # ---- STATE: IDLE ----
            if state == 'idle':
                if has_ball and ball_pos:
                    dist_to_entry = measure_distance(player_pos, entry_pos)
                    if dist_to_entry <= self.entry_exit_zone_radius:
                        # Cooldown check
                        if (frame_num - last_attempt_end) >= self.cooldown_frames:
                            state = 'dribbling'
                            attempt_start_frame = frame_num
                            attempt_player_id = player_id
                            ball_trajectory = [ball_pos]
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
                player_positions.append(player_pos)
                if has_ball:
                    frames_with_ball += 1

                # Cek apakah sampai exit zone
                dist_to_exit = measure_distance(player_pos, exit_pos)

                if dist_to_exit <= self.entry_exit_zone_radius:
                    # ATTEMPT SELESAI → evaluasi
                    duration_frames = frame_num - attempt_start_frame

                    if duration_frames >= self.min_attempt_frames:
                        # Evaluasi sentuhan cone
                        cone_results = self.evaluate_trajectory_against_cones(
                            ball_trajectory
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
                                print(f"[DRIBBLE]     Cone {tc}: "
                                      f"min_dist={d:.1f}px "
                                      f"(radius={r:.0f}px, "
                                      f"{cnt} frame di dalam)")
                            print(f"[DRIBBLE]   Possession   : "
                                  f"{possession_ratio*100:.0f}%")
                    else:
                        if debug:
                            print(f"[DRIBBLE] Frame {frame_num}: "
                                  f"attempt terlalu singkat "
                                  f"({duration_frames} < {self.min_attempt_frames})")

                    # Reset state
                    state = 'idle'
                    continue

                # Timeout
                if attempt_total_frames > max_attempt_frames:
                    if debug:
                        print(f"[DRIBBLE] Frame {frame_num}: TIMEOUT → reset")
                    state = 'idle'
                    continue

                # Kehilangan bola → cek possession ratio
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
        """Hitung statistik dribble keseluruhan."""
        total = len(attempts)
        sukses = [a for a in attempts if a['success']]
        gagal = [a for a in attempts if not a['success']]

        # Cone yang paling sering disentuh
        cone_hit_count: Dict[int, int] = {}
        for a in attempts:
            for tc in a['touched_cones']:
                cone_hit_count[tc] = cone_hit_count.get(tc, 0) + 1

        # Rata-rata detail per cone
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
