# penalty_detector.py
# Mendeteksi event tendangan penalty dan mengevaluasi gol/tidak gol.
#
# Logika:
#   1. Deteksi saat pemain (Merah/Abu-Abu) memiliki possession lalu kehilangan bola
#   2. Setelah bola ditendang, cek apakah bola masuk ke area gawang
#   3. GOAL = bola memasuki/overlap dengan bounding box gawang
#   4. MISS = bola tidak masuk gawang

import sys
sys.path.append('../')

from utils.bbox_utils import (
    measure_distance,
    get_center_of_bbox,
    get_center_of_bbox_bottom,
)
import numpy as np
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple


class PenaltyDetector:
    def __init__(self, fps: int = 24):
        self.fps = fps

        # --- Parameter smoothing possession ---
        self.smoothing_window        = 5
        self.min_stable_frames       = 1

        # --- Parameter validasi kick ---
        self.cooldown_frames         = 30   # Cooldown antara penalty
        self.min_possession_duration = 3    # Minimal frame memiliki bola
        self.ball_movement_threshold = 30   # Minimal gerakan bola saat tendangan

        # --- Parameter deteksi gol ---
        # Jumlah frame setelah tendangan untuk mengecek apakah bola masuk
        self.goal_check_window       = 45   # ~1.5 detik pada 30fps
        self.gawang_overlap_margin   = 20   # Margin pixel untuk overlap check

        # --- Filter pemain (hanya penendang) ---
        self.kicker_jerseys = {"Merah", "Abu-Abu"}

        # --- Parameter display ---
        self.kick_display_duration   = 45   # Berapa frame hasil ditampilkan

        self._player_identifier = None

    def set_jersey_map(self, player_identifier) -> None:
        self._player_identifier = player_identifier

    def _get_jersey(self, player_id: int) -> str:
        if self._player_identifier:
            return self._player_identifier.get_jersey_number_for_player(player_id)
        return str(player_id)

    # ============================================================
    # SMOOTHING POSSESSION
    # ============================================================

    def smooth_possessions(self, raw_possessions: List[int]) -> List[int]:
        smoothed    = list(raw_possessions)
        half_window = self.smoothing_window // 2
        for i in range(half_window, len(raw_possessions) - half_window):
            window = raw_possessions[i - half_window: i + half_window + 1]
            valid  = [p for p in window if p != -1]
            if valid:
                smoothed[i] = Counter(valid).most_common(1)[0][0]
            else:
                smoothed[i] = -1
        return smoothed

    def fill_short_gaps(self, possessions: List[int], max_gap: int = 15) -> List[int]:
        filled     = list(possessions)
        last_valid = -1
        gap_start  = -1
        for i in range(len(filled)):
            if filled[i] != -1:
                if last_valid != -1 and gap_start != -1:
                    gap_length = i - gap_start
                    if gap_length <= max_gap:
                        for g in range(gap_start, i):
                            filled[g] = last_valid
                last_valid = filled[i]
                gap_start  = -1
            else:
                if gap_start == -1:
                    gap_start = i
        return filled

    def _normalize_possessions_by_jersey(self, possessions: List[int]) -> List[int]:
        if not self._player_identifier:
            return possessions
        jersey_canonical: Dict[str, int] = {}
        normalized = list(possessions)
        for i, pid in enumerate(possessions):
            if pid == -1:
                continue
            jersey = self._get_jersey(pid)
            if jersey == "Unknown":
                continue
            if jersey not in jersey_canonical:
                jersey_canonical[jersey] = pid
            normalized[i] = jersey_canonical[jersey]
        return normalized

    def get_stable_segments(self, smoothed_possessions: List[int]) -> List[Dict]:
        segments       = []
        current_player = -1
        segment_start  = 0
        for frame_num, player_id in enumerate(smoothed_possessions):
            if player_id != current_player:
                if current_player != -1:
                    duration = frame_num - segment_start
                    if duration >= self.min_stable_frames:
                        segments.append({
                            'player_id'  : current_player,
                            'frame_start': segment_start,
                            'frame_end'  : frame_num - 1
                        })
                current_player = player_id
                segment_start  = frame_num
        if current_player != -1:
            duration = len(smoothed_possessions) - segment_start
            if duration >= self.min_stable_frames:
                segments.append({
                    'player_id'  : current_player,
                    'frame_start': segment_start,
                    'frame_end'  : len(smoothed_possessions) - 1
                })
        return segments

    # ============================================================
    # STABILISASI POSISI GAWANG
    # ============================================================

    def get_stable_gawang_bbox(
        self,
        tracks: Dict,
        sample_frames: int = 0  # 0 = semua frame
    ) -> Optional[List[float]]:
        """
        Hitung posisi rata-rata gawang dari semua frame untuk stabilitas.
        """
        gawang_bboxes = []
        total = len(tracks.get('gawang', []))
        if sample_frames > 0:
            total = min(sample_frames, total)

        for f in range(total):
            gawang_data = tracks['gawang'][f].get(1)
            if gawang_data and 'bbox' in gawang_data:
                gawang_bboxes.append(gawang_data['bbox'])

        if not gawang_bboxes:
            return None

        avg_bbox = [
            float(np.mean([b[0] for b in gawang_bboxes])),
            float(np.mean([b[1] for b in gawang_bboxes])),
            float(np.mean([b[2] for b in gawang_bboxes])),
            float(np.mean([b[3] for b in gawang_bboxes])),
        ]

        print(f"[PENALTY] Gawang stabil bbox: "
              f"({avg_bbox[0]:.0f}, {avg_bbox[1]:.0f}) - "
              f"({avg_bbox[2]:.0f}, {avg_bbox[3]:.0f})")
        return avg_bbox

    # ============================================================
    # CEK BOLA MASUK GAWANG
    # ============================================================

    def check_ball_in_gawang(
        self,
        tracks: Dict,
        frame_start: int,
        frame_end: int,
        stable_gawang_bbox: Optional[List[float]] = None
    ) -> Tuple[bool, int, str]:
        """
        Cek apakah bola masuk ke area gawang antara frame_start dan frame_end.

        Returns:
            (is_goal, goal_frame, reason)
        """
        total_frames = len(tracks['ball'])
        check_end = min(frame_end, total_frames - 1)
        margin = self.gawang_overlap_margin

        for f in range(frame_start, check_end + 1):
            ball_data = tracks['ball'][f].get(1)
            if not ball_data or 'bbox' not in ball_data:
                continue

            ball_bbox = ball_data['bbox']
            ball_cx, ball_cy = get_center_of_bbox(ball_bbox)

            # Gunakan gawang bbox dari frame saat ini atau stabil
            if stable_gawang_bbox:
                gx1, gy1, gx2, gy2 = stable_gawang_bbox
            else:
                gawang_data = tracks['gawang'][f].get(1)
                if not gawang_data or 'bbox' not in gawang_data:
                    continue
                gx1, gy1, gx2, gy2 = gawang_data['bbox']

            # Cek apakah pusat bola berada di dalam area gawang (+ margin)
            if (gx1 - margin <= ball_cx <= gx2 + margin and
                gy1 - margin <= ball_cy <= gy2 + margin):
                return True, f, f"Bola masuk gawang di frame {f}"

        return False, -1, "Bola tidak masuk gawang"

    # ============================================================
    # VALIDASI GERAKAN BOLA
    # ============================================================

    def validate_ball_movement(
        self,
        tracks: Dict,
        frame_start: int,
        frame_end: int
    ) -> float:
        ball_positions = []
        total = len(tracks['ball'])
        for f in range(max(0, frame_start), min(total, frame_end + 1)):
            ball_data = tracks['ball'][f].get(1)
            if ball_data and 'bbox' in ball_data:
                pos = get_center_of_bbox(ball_data['bbox'])
                ball_positions.append(pos)
        if len(ball_positions) < 2:
            return 0.0
        direct_distance = measure_distance(ball_positions[0], ball_positions[-1])
        max_displacement = max(
            measure_distance(ball_positions[0], p) for p in ball_positions
        )
        return max(direct_distance, max_displacement)

    # ============================================================
    # DETEKSI PENALTY UTAMA
    # ============================================================

    def detect_penalties(
        self,
        tracks           : Dict,
        ball_possessions : List[int],
        player_identifier = None,
        debug            : bool = True
    ) -> List[Dict]:
        """
        Deteksi semua event tendangan penalty.

        Logika:
        1. Cari segmen di mana Merah/Abu-Abu memiliki bola
        2. Saat kehilangan bola (transisi ke segmen berikutnya), itu = tendangan
        3. Setelah tendangan, cek apakah bola masuk gawang
        """
        if player_identifier:
            self._player_identifier = player_identifier

        if debug:
            valid_count = sum(1 for p in ball_possessions if p != -1)
            print(f"\n[PENALTY] === PENALTY DETECTION PIPELINE ===")
            print(f"[PENALTY] Total frames           : {len(ball_possessions)}")
            print(f"[PENALTY] Frame dengan possession : {valid_count}")
            print(f"[PENALTY] Penendang target        : {self.kicker_jerseys}")
            print(f"[PENALTY] Goal check window       : {self.goal_check_window} frames")
            print(f"[PENALTY] Cooldown                : {self.cooldown_frames} frames")

        if sum(1 for p in ball_possessions if p != -1) == 0:
            print("[PENALTY] Tidak ada possession terdeteksi!")
            return []

        # Stabilisasi gawang
        stable_gawang = self.get_stable_gawang_bbox(tracks)
        if not stable_gawang:
            print("[PENALTY] WARNING: Gawang tidak terdeteksi! "
                  "Tidak bisa menentukan gol/tidak.")
            return []

        # Preprocessing possession
        filled   = self.fill_short_gaps(ball_possessions, max_gap=15)
        filled   = self._normalize_possessions_by_jersey(filled)
        smoothed = self.smooth_possessions(filled)
        segments = self.get_stable_segments(smoothed)

        # Merge segmen pendek dari pemain yang sama
        merged = [segments[0]] if segments else []
        for seg in segments[1:]:
            last = merged[-1]
            gap  = seg['frame_start'] - last['frame_end']
            if seg['player_id'] == last['player_id'] and gap <= 10:
                merged[-1]['frame_end'] = seg['frame_end']
            else:
                merged.append(seg)
        segments = merged

        if debug:
            print(f"[PENALTY] Stable segments: {len(segments)}")
            for idx, seg in enumerate(segments):
                jersey = self._get_jersey(seg['player_id'])
                print(f"[PENALTY]   Seg {idx:2d}: track {seg['player_id']:2d} "
                      f"({jersey:>10s}) | frame {seg['frame_start']:4d}-{seg['frame_end']:4d}")

        # Deteksi tendangan
        penalties = []
        last_kick_frame = -999

        for i in range(len(segments)):
            seg = segments[i]
            kicker_id = seg['player_id']
            kicker_jersey = self._get_jersey(kicker_id)

            # Hanya proses jika penendang = Merah atau Abu-Abu
            if kicker_jersey not in self.kicker_jerseys:
                continue

            # Frame saat bola ditendang = akhir segmen possession
            kick_frame = seg['frame_end']

            # Cek cooldown
            if (kick_frame - last_kick_frame) < self.cooldown_frames:
                if debug:
                    print(f"[PENALTY] Skip: cooldown (frame {kick_frame})")
                continue

            # Cek durasi possession
            duration = seg['frame_end'] - seg['frame_start']
            if duration < self.min_possession_duration:
                if debug:
                    print(f"[PENALTY] Skip: possession terlalu singkat "
                          f"({duration} frames) di frame {kick_frame}")
                continue

            # Cek apakah setelah segmen ini, pemain kehilangan bola
            # (ada transisi ke segmen lain atau -1)
            if i + 1 < len(segments):
                next_seg = segments[i + 1]
                next_jersey = self._get_jersey(next_seg['player_id'])
                # Jika pemain sama lanjut punya bola, bukan tendangan
                if kicker_jersey == next_jersey:
                    continue

            # Validasi gerakan bola
            check_end = min(kick_frame + self.goal_check_window,
                            len(ball_possessions) - 1)
            ball_movement = self.validate_ball_movement(
                tracks, kick_frame, check_end
            )
            if ball_movement < self.ball_movement_threshold:
                if debug:
                    print(f"[PENALTY] Skip: gerakan bola kecil "
                          f"({ball_movement:.1f}px) di frame {kick_frame}")
                continue

            # ===== CEK GOL =====
            is_goal, goal_frame, reason = self.check_ball_in_gawang(
                tracks, kick_frame, check_end, stable_gawang=stable_gawang
            )

            # Posisi pemain saat menendang
            kicker_pos = None
            player_data = tracks['players'][kick_frame].get(kicker_id)
            if player_data and 'bbox' in player_data:
                kicker_pos = get_center_of_bbox_bottom(player_data['bbox'])

            # Posisi bola saat tendangan
            ball_pos_kick = None
            ball_data = tracks['ball'][kick_frame].get(1)
            if ball_data:
                ball_pos_kick = get_center_of_bbox(ball_data['bbox'])

            penalty_event = {
                'frame_kick'     : kick_frame,
                'frame_goal'     : goal_frame if is_goal else -1,
                'frame_display'  : kick_frame + 10,
                'kicker_id'      : kicker_id,
                'kicker_jersey'  : kicker_jersey,
                'is_goal'        : is_goal,
                'reason'         : reason,
                'ball_movement'  : ball_movement,
                'kicker_pos'     : kicker_pos,
                'ball_pos_kick'  : ball_pos_kick,
                'gawang_bbox'    : stable_gawang,
            }
            penalties.append(penalty_event)
            last_kick_frame = kick_frame

            if debug:
                status = "GOL!" if is_goal else "MISS"
                print(f"[PENALTY] Tendangan {kicker_jersey}: frame {kick_frame} "
                      f"| bola={ball_movement:.0f}px | {status} | {reason}")

        if debug:
            gol  = sum(1 for p in penalties if p['is_goal'])
            miss = sum(1 for p in penalties if not p['is_goal'])
            print(f"\n[PENALTY] === HASIL AKHIR ===")
            print(f"[PENALTY] Total tendangan : {len(penalties)}")
            print(f"[PENALTY] GOL             : {gol}")
            print(f"[PENALTY] MISS            : {miss}")
            print(f"[PENALTY] =========================\n")

        return penalties

    # ============================================================
    # STATISTIK
    # ============================================================

    def get_penalty_statistics(self, penalties: List[Dict]) -> Dict:
        total = len(penalties)
        gol   = [p for p in penalties if p['is_goal']]
        miss  = [p for p in penalties if not p['is_goal']]

        per_player: Dict[str, Dict] = {}
        for p in penalties:
            jersey = p['kicker_jersey']
            if jersey not in per_player:
                per_player[jersey] = {
                    'total'       : 0,
                    'goals'       : 0,
                    'misses'      : 0,
                    'goal_pct'    : 0.0,
                }
            per_player[jersey]['total'] += 1
            if p['is_goal']:
                per_player[jersey]['goals'] += 1
            else:
                per_player[jersey]['misses'] += 1

        for jersey, stat in per_player.items():
            stat['goal_pct'] = round(
                stat['goals'] / stat['total'] * 100, 1
            ) if stat['total'] > 0 else 0.0

        return {
            'total_kicks'   : total,
            'total_goals'   : len(gol),
            'total_misses'  : len(miss),
            'goal_pct'      : round(len(gol) / total * 100, 1) if total > 0 else 0.0,
            'per_player'    : per_player
        }
