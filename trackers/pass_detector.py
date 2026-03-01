"""
Pass Detector v3 — Jersey-Based Pipeline

PERUBAHAN ARSITEKTUR UTAMA:
Semua deteksi pass sekarang bekerja di level JERSEY NUMBER,
bukan track ID. Ini menyelesaikan:
1. Track ID re-assignment oleh ByteTrack
2. False pass antar track ID yang sebenarnya pemain sama
3. Lebih mudah di-tune karena datanya lebih bersih

Pipeline:
  raw track-ID possessions
  → convert to jersey possessions
  → temporal flicker filter (hapus 1-frame noise)
  → majority-vote smoothing (window=5)
  → gap filling
  → remove short false segments (A-B-A pattern)
  → stable segments
  → merge same-jersey segments
  → validate transitions → passes
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.bbox_utils import (measure_distance, get_center_of_bbox_bottom,
                               get_center_of_bbox)
import numpy as np
from collections import Counter


class PassDetector:
    def __init__(self, fps=24):
        self.fps = fps

        # Smoothing
        self.smoothing_window = 5          # UBAH: 3 → 5 (lebih agresif hapus noise)
        self.min_stable_frames = 2

        # Pass validation
        self.min_pass_distance = 30        # UBAH: 35 → 30
        self.max_pass_distance = 1200      # UBAH: 950 → 1200 (sangat longgar)
        self.cooldown_frames = 3           # UBAH: 5 → 3 (drill cepat)
        self.min_possession_duration = 2   # UBAH: 3 → 2

        # Ball movement
        self.ball_movement_check_radius = 25
        self.ball_movement_threshold = 3   # UBAH: 4 → 3

        self.player_search_radius = 25     # UBAH: 20 → 25

        # Display
        self.pass_display_delay = 3
        self.min_display_gap = 4

    # ----------------------------------------------------------------
    #  BARU: Convert track-ID possessions → jersey possessions
    # ----------------------------------------------------------------
    def convert_to_jersey_possessions(self, raw_possessions, player_identifier):
        """
        Convert list of track IDs ke list of jersey strings.
        -1 tetap jadi None (tidak ada possession).
        """
        jersey_poss = []
        for tid in raw_possessions:
            if tid == -1:
                jersey_poss.append(None)
            else:
                jersey_poss.append(player_identifier.track_id_to_jersey(tid))
        return jersey_poss

    # ----------------------------------------------------------------
    #  BARU: Temporal flicker filter — hapus possession 1-frame
    # ----------------------------------------------------------------
    def remove_flickers(self, possessions, max_flicker=2):
        """
        Jika pemain X memiliki bola hanya 1-2 frame di antara
        pemain Y yang panjang, itu noise. Hapus.
        
        Contoh: [Unknown, Unknown, #3, Unknown, Unknown]
                → #3 di frame tengah = noise → hapus jadi Unknown
        """
        result = list(possessions)
        n = len(result)
        i = 0
        while i < n:
            # Cari segment
            j = i
            while j < n and result[j] == result[i]:
                j += 1
            seg_len = j - i
            seg_val = result[i]

            # Jika segment pendek (≤ max_flicker) dan diapit oleh segment lain
            if seg_val is not None and seg_len <= max_flicker:
                # Cek sebelum dan sesudah
                prev_val = result[i - 1] if i > 0 else None
                next_val = result[j] if j < n else None

                # Jika diapit oleh player yang SAMA → ganti ke player itu
                if prev_val is not None and prev_val == next_val:
                    for k in range(i, j):
                        result[k] = prev_val
                # Jika diapit oleh player BERBEDA tapi segment sangat pendek (1 frame)
                elif seg_len == 1 and prev_val is not None:
                    result[i] = prev_val

            i = j
        return result

    # ----------------------------------------------------------------
    #  Majority-vote smoothing
    # ----------------------------------------------------------------
    def smooth_possessions(self, possessions):
        smoothed = list(possessions)
        half_window = self.smoothing_window // 2
        for i in range(half_window, len(possessions) - half_window):
            window = possessions[i - half_window: i + half_window + 1]
            valid = [p for p in window if p is not None]
            if len(valid) > 0:
                smoothed[i] = Counter(valid).most_common(1)[0][0]
            else:
                smoothed[i] = None
        return smoothed

    # ----------------------------------------------------------------
    #  Fill short gaps (None → last known)
    # ----------------------------------------------------------------
    def fill_short_gaps(self, possessions, max_gap=15):
        """UBAH: max_gap 12 → 15"""
        filled = list(possessions)
        last_valid = None
        gap_start = -1
        for i in range(len(filled)):
            if filled[i] is not None:
                if last_valid is not None and gap_start != -1:
                    gap_length = i - gap_start
                    if gap_length <= max_gap:
                        for g in range(gap_start, i):
                            filled[g] = last_valid
                last_valid = filled[i]
                gap_start = -1
            else:
                if gap_start == -1:
                    gap_start = i
        return filled

    # ----------------------------------------------------------------
    #  BARU: Remove short false segments (A-B-A pattern)
    # ----------------------------------------------------------------
    def remove_short_false_segments(self, possessions, max_false_len=3):
        """
        Jika pattern A...A-B(1~3 frames)-A...A, ganti B → A.
        Ini menghilangkan false possession yang terjadi ketika
        bola melintas dekat pemain lain sebentar.

        Contoh: [Unknown*10, #3*2, Unknown*10] → ganti #3 → Unknown
        """
        result = list(possessions)
        n = len(result)

        # Build segments
        segments = []
        i = 0
        while i < n:
            j = i
            while j < n and result[j] == result[i]:
                j += 1
            segments.append((result[i], i, j))  # (value, start, end)
            i = j

        # Find short segments sandwiched between same value
        for idx in range(1, len(segments) - 1):
            val, start, end = segments[idx]
            seg_len = end - start
            prev_val = segments[idx - 1][0]
            next_val = segments[idx + 1][0]

            if (val is not None and seg_len <= max_false_len
                    and prev_val == next_val and prev_val is not None):
                for k in range(start, end):
                    result[k] = prev_val

        return result

    # ----------------------------------------------------------------
    #  Get stable segments
    # ----------------------------------------------------------------
    def get_stable_segments(self, possessions):
        segments = []
        current = None
        segment_start = 0
        for frame_num, jersey in enumerate(possessions):
            if jersey != current:
                if current is not None:
                    duration = frame_num - segment_start
                    if duration >= self.min_stable_frames:
                        segments.append({
                            'jersey': current,
                            'frame_start': segment_start,
                            'frame_end': frame_num - 1
                        })
                current = jersey
                segment_start = frame_num

        if current is not None:
            duration = len(possessions) - segment_start
            if duration >= self.min_stable_frames:
                segments.append({
                    'jersey': current,
                    'frame_start': segment_start,
                    'frame_end': len(possessions) - 1
                })
        return segments

    # ----------------------------------------------------------------
    #  Ball movement validation
    # ----------------------------------------------------------------
    def validate_ball_movement(self, tracks, frame_start, frame_end):
        check_start = max(0, frame_start - self.ball_movement_check_radius)
        check_end = min(len(tracks['ball']),
                        frame_end + self.ball_movement_check_radius)

        ball_positions = []
        for f in range(check_start, check_end):
            ball_data = tracks['ball'][f].get(1)
            if ball_data and 'bbox' in ball_data:
                pos = get_center_of_bbox(ball_data['bbox'])
                ball_positions.append(pos)

        if len(ball_positions) < 2:
            return 0

        direct_distance = measure_distance(ball_positions[0], ball_positions[-1])
        max_displacement = 0
        for i in range(1, len(ball_positions)):
            d = measure_distance(ball_positions[0], ball_positions[i])
            if d > max_displacement:
                max_displacement = d

        return max(direct_distance, max_displacement)

    # ----------------------------------------------------------------
    #  Find any player bbox for a given jersey near a frame
    # ----------------------------------------------------------------
    def find_jersey_player_nearby(self, tracks, jersey, target_frame,
                                  player_identifier, search_radius=None):
        """
        Cari bbox pemain dengan jersey tertentu di sekitar target_frame.
        Karena track ID bisa berganti, kita cari SEMUA track ID
        yang mapping ke jersey ini.
        """
        if search_radius is None:
            search_radius = self.player_search_radius
        total_frames = len(tracks['players'])

        # Dapatkan semua track ID untuk jersey ini
        candidate_tids = player_identifier.get_all_track_ids_for_jersey(jersey)

        # Cari di frame terdekat
        for offset in range(0, search_radius + 1):
            for direction in ([0] if offset == 0 else [-1, 1]):
                check_frame = target_frame + direction * offset
                if 0 <= check_frame < total_frames:
                    for tid in candidate_tids:
                        pdata = tracks['players'][check_frame].get(tid)
                        if pdata:
                            return pdata, check_frame
        return None, -1

    # ----------------------------------------------------------------
    #  MAIN: Detect passes (jersey-based)
    # ----------------------------------------------------------------
    def detect_passes(self, tracks, raw_possessions_track_ids,
                      player_identifier=None, debug=True):
        """
        ARSITEKTUR BARU: Jersey-based pass detection.

        Input:
            raw_possessions_track_ids: list of track IDs per frame (-1 = no possession)
            player_identifier: PlayerIdentifier instance (WAJIB)
        """
        if player_identifier is None:
            raise ValueError("player_identifier is required for jersey-based detection")

        # TAHAP 0: Convert track IDs → jersey strings
        jersey_poss = self.convert_to_jersey_possessions(
            raw_possessions_track_ids, player_identifier
        )

        if debug:
            valid_count = sum(1 for p in jersey_poss if p is not None)
            unique = set(p for p in jersey_poss if p is not None)
            print(f"\n[DEBUG] === JERSEY-BASED PASS DETECTION PIPELINE ===")
            print(f"[DEBUG] Total frames: {len(jersey_poss)}")
            print(f"[DEBUG] Frames with possession: {valid_count}/{len(jersey_poss)} "
                  f"({100 * valid_count / max(1, len(jersey_poss)):.1f}%)")
            print(f"[DEBUG] Unique jerseys: {unique}")
            if valid_count == 0 or len(unique) < 2:
                return []

        # TAHAP 1: Remove flickers (1-2 frame noise)
        deflickered = self.remove_flickers(jersey_poss, max_flicker=2)
        if debug:
            changes = sum(1 for a, b in zip(jersey_poss, deflickered) if a != b)
            print(f"[DEBUG] After de-flicker: {changes} frames changed")

        # TAHAP 2: Smoothing (majority vote, window=5)
        smoothed = self.smooth_possessions(deflickered)
        if debug:
            print(f"[DEBUG] After smoothing: "
                  f"{sum(1 for p in smoothed if p is not None)} frames with possession")

        # TAHAP 3: Fill gaps
        filled = self.fill_short_gaps(smoothed, max_gap=15)
        if debug:
            print(f"[DEBUG] After gap-fill: "
                  f"{sum(1 for p in filled if p is not None)} frames with possession")

        # TAHAP 4: Remove short false segments (A-B-A pattern)
        cleaned = self.remove_short_false_segments(filled, max_false_len=3)
        if debug:
            changes = sum(1 for a, b in zip(filled, cleaned) if a != b)
            print(f"[DEBUG] After false-segment removal: {changes} frames changed")

        # TAHAP 5: Second round smoothing (cleanup residual)
        cleaned = self.smooth_possessions(cleaned)

        # TAHAP 6: Stable segments
        segments = self.get_stable_segments(cleaned)
        if debug:
            print(f"[DEBUG] Stable segments found: {len(segments)}")
            for i, seg in enumerate(segments):
                dur = seg['frame_end'] - seg['frame_start']
                print(f"[DEBUG]   Seg #{i}: Jersey #{seg['jersey']}, "
                      f"frames {seg['frame_start']}-{seg['frame_end']} (dur={dur})")
            if len(segments) < 2:
                return []

        # TAHAP 7: Merge same-jersey segments yang berdekatan
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg['jersey'] == merged[-1]['jersey']:
                gap = seg['frame_start'] - merged[-1]['frame_end']
                if gap <= 12:
                    merged[-1]['frame_end'] = seg['frame_end']
                else:
                    merged.append(seg)
            else:
                merged.append(seg)
        segments = merged

        if debug:
            print(f"[DEBUG] After merge: {len(segments)} segments")
            for i, seg in enumerate(segments):
                dur = seg['frame_end'] - seg['frame_start']
                print(f"[DEBUG]   Merged #{i}: Jersey #{seg['jersey']}, "
                      f"frames {seg['frame_start']}-{seg['frame_end']} (dur={dur})")

        # TAHAP 8: Detect passes dari transisi antar jersey
        passes = []
        last_pass_frame = -999

        if debug:
            print(f"\n[DEBUG] === EVALUATING JERSEY TRANSITIONS ===")

        for i in range(len(segments) - 1):
            seg_from = segments[i]
            seg_to = segments[i + 1]
            from_jersey = seg_from['jersey']
            to_jersey = seg_to['jersey']
            transition_frame_start = seg_from['frame_end']
            transition_frame_end = seg_to['frame_start']

            if debug:
                print(f"\n[DEBUG] Transition #{i}: "
                      f"#{from_jersey} -> #{to_jersey}")

            # Filter: same jersey
            if from_jersey == to_jersey:
                if debug:
                    print(f"[DEBUG]   SKIP: Same jersey")
                continue

            # Filter: None jersey
            if from_jersey is None or to_jersey is None:
                if debug:
                    print(f"[DEBUG]   SKIP: None jersey")
                continue

            # Filter: cooldown
            if (transition_frame_end - last_pass_frame) < self.cooldown_frames:
                if debug:
                    print(f"[DEBUG]   SKIP: Cooldown "
                          f"({transition_frame_end - last_pass_frame} "
                          f"< {self.cooldown_frames})")
                continue

            # Filter: sender duration
            from_duration = seg_from['frame_end'] - seg_from['frame_start']
            if from_duration < self.min_possession_duration:
                if debug:
                    print(f"[DEBUG]   SKIP: Sender too short "
                          f"({from_duration} < {self.min_possession_duration})")
                continue

            # Find player positions for distance check
            from_pdata, from_frame = self.find_jersey_player_nearby(
                tracks, from_jersey, transition_frame_start, player_identifier
            )
            to_pdata, to_frame = self.find_jersey_player_nearby(
                tracks, to_jersey, transition_frame_end, player_identifier
            )

            if not from_pdata or not to_pdata:
                if debug:
                    print(f"[DEBUG]   SKIP: Player data missing")
                continue

            from_pos = get_center_of_bbox_bottom(from_pdata['bbox'])
            to_pos = get_center_of_bbox_bottom(to_pdata['bbox'])
            distance = measure_distance(from_pos, to_pos)

            # Filter: distance
            if distance < self.min_pass_distance:
                if debug:
                    print(f"[DEBUG]   SKIP: Too close "
                          f"({distance:.0f} < {self.min_pass_distance})")
                continue
            if distance > self.max_pass_distance:
                if debug:
                    print(f"[DEBUG]   SKIP: Too far "
                          f"({distance:.0f} > {self.max_pass_distance})")
                continue

            # Filter: ball movement
            ball_movement = self.validate_ball_movement(
                tracks, transition_frame_start, transition_frame_end
            )
            if ball_movement < self.ball_movement_threshold:
                if debug:
                    print(f"[DEBUG]   SKIP: Ball static "
                          f"({ball_movement:.0f} < {self.ball_movement_threshold})")
                continue

            if debug:
                print(f"[DEBUG]   *** PASS! #{from_jersey} -> #{to_jersey} | "
                      f"dist={distance:.0f}px, ball={ball_movement:.0f}px ***")

            # Display frame
            receiver_start = seg_to['frame_start']
            pass_display_frame = min(
                receiver_start + self.pass_display_delay, seg_to['frame_end']
            )
            if len(passes) > 0:
                last_display = passes[-1]['frame_display']
                if pass_display_frame - last_display < self.min_display_gap:
                    pass_display_frame = last_display + self.min_display_gap

            pass_event = {
                'frame_start': transition_frame_start,
                'frame_end': transition_frame_end,
                'frame_display': pass_display_frame,
                'from_jersey': from_jersey,
                'to_jersey': to_jersey,
                'distance': distance,
                'ball_movement': ball_movement,
                'success': True,
                'from_pos': from_pos,
                'to_pos': to_pos,
            }
            passes.append(pass_event)
            last_pass_frame = transition_frame_end

        if debug:
            print(f"\n[DEBUG] === RESULT: {len(passes)} passes detected ===")
            # Summary per jersey
            jersey_counts = {}
            for p in passes:
                j = p['from_jersey']
                jersey_counts[j] = jersey_counts.get(j, 0) + 1
            print(f"[DEBUG] Per jersey sender: {jersey_counts}")
            print()

        return passes

    def get_pass_statistics(self, passes):
        stats = {
            'total_passes': len(passes),
            'avg_distance': np.mean([p['distance'] for p in passes]) if passes else 0,
        }
        return stats
