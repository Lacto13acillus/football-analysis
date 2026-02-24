import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox_bottom, get_center_of_bbox
import numpy as np
from collections import Counter

class PassDetector:
    def __init__(self, fps=24):
        self.fps = fps
        
        # === POSSESSION SMOOTHING ===
        self.smoothing_window = 5          # Dikurangi dari 7
        self.min_stable_frames = 3         # Dikurangi DRASTIS dari 8 -> 3
        
        # === PASS VALIDATION ===
        self.min_pass_distance = 50        # Dikurangi dari 80 -> 50
        self.max_pass_distance = 700       # Diperbesar
        self.cooldown_frames = 10          # Dikurangi dari 20 -> 10
        self.min_possession_duration = 3   # Dikurangi dari 5 -> 3
        
        # === BALL MOVEMENT (lebih longgar) ===
        self.ball_movement_check_radius = 15  # Cek bola di radius frame lebih lebar
        self.ball_movement_threshold = 15     # Dikurangi dari 30 -> 15

    def smooth_possessions(self, raw_possessions):
        """
        Majority Vote sliding window untuk hilangkan flickering.
        """
        smoothed = list(raw_possessions)
        half_window = self.smoothing_window // 2
        
        for i in range(half_window, len(raw_possessions) - half_window):
            window = raw_possessions[i - half_window : i + half_window + 1]
            valid = [p for p in window if p != -1]
            if len(valid) > 0:
                smoothed[i] = Counter(valid).most_common(1)[0][0]
            else:
                smoothed[i] = -1
        
        return smoothed

    def fill_short_gaps(self, possessions, max_gap=5):
        """
        BARU: Isi gap pendek (-1) dengan possession terakhir yang valid.
        Mengatasi bola yang hilang sebentar lalu kembali ke pemain yang sama.
        """
        filled = list(possessions)
        last_valid = -1
        gap_start = -1
        
        for i in range(len(filled)):
            if filled[i] != -1:
                # Jika sebelumnya ada gap pendek, isi dengan last_valid
                if last_valid != -1 and gap_start != -1:
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

    def get_stable_segments(self, smoothed_possessions):
        """
        Identifikasi segmen possession yang stabil.
        """
        segments = []
        current_player = -1
        segment_start = 0
        
        for frame_num, player_id in enumerate(smoothed_possessions):
            if player_id != current_player:
                if current_player != -1:
                    duration = frame_num - segment_start
                    if duration >= self.min_stable_frames:
                        segments.append({
                            'player_id': current_player,
                            'frame_start': segment_start,
                            'frame_end': frame_num - 1
                        })
                current_player = player_id
                segment_start = frame_num
        
        # Segment terakhir
        if current_player != -1:
            duration = len(smoothed_possessions) - segment_start
            if duration >= self.min_stable_frames:
                segments.append({
                    'player_id': current_player,
                    'frame_start': segment_start,
                    'frame_end': len(smoothed_possessions) - 1
                })
        
        return segments

    def validate_ball_movement(self, tracks, frame_start, frame_end):
        """
        Validasi bola bergerak — cek di radius yang lebih lebar 
        (sebelum dan sesudah transisi)
        """
        check_start = max(0, frame_start - self.ball_movement_check_radius)
        check_end = min(len(tracks['ball']), frame_end + self.ball_movement_check_radius)
        
        ball_positions = []
        for f in range(check_start, check_end):
            ball_data = tracks['ball'][f].get(1)
            if ball_data and 'bbox' in ball_data:
                pos = get_center_of_bbox(ball_data['bbox'])
                ball_positions.append(pos)
        
        if len(ball_positions) < 2:
            return 0
        
        # Jarak antara posisi pertama dan terakhir bola (bukan total path)
        direct_distance = measure_distance(ball_positions[0], ball_positions[-1])
        return direct_distance

    def detect_passes(self, tracks, ball_possessions, debug=True):
        """
        Pipeline:
        1. Fill short gaps
        2. Smooth possessions
        3. Get stable segments
        4. Validate transitions as passes
        """
        
        # === DEBUG: Cek raw data ===
        if debug:
            valid_count = sum(1 for p in ball_possessions if p != -1)
            unique_players = set(p for p in ball_possessions if p != -1)
            print(f"\n[DEBUG] === PASS DETECTION PIPELINE ===")
            print(f"[DEBUG] Total frames: {len(ball_possessions)}")
            print(f"[DEBUG] Frames with possession: {valid_count}/{len(ball_possessions)} ({100*valid_count/max(1,len(ball_possessions)):.1f}%)")
            print(f"[DEBUG] Unique players detected: {unique_players}")
            
            if valid_count == 0:
                print(f"[DEBUG] *** PROBLEM: No ball possession detected at all! ***")
                print(f"[DEBUG] *** Check PlayerBallAssigner and ball detection ***")
                return []
            
            if len(unique_players) < 2:
                print(f"[DEBUG] *** PROBLEM: Only {len(unique_players)} player(s) ever had the ball! ***")
                print(f"[DEBUG] *** Need at least 2 different players for a pass ***")
                return []

        # === TAHAP 1: Fill short gaps ===
        filled = self.fill_short_gaps(ball_possessions, max_gap=5)
        
        if debug:
            filled_valid = sum(1 for p in filled if p != -1)
            print(f"[DEBUG] After gap-fill: {filled_valid} frames with possession")
        
        # === TAHAP 2: Smoothing ===
        smoothed = self.smooth_possessions(filled)
        
        if debug:
            smoothed_valid = sum(1 for p in smoothed if p != -1)
            smoothed_players = set(p for p in smoothed if p != -1)
            print(f"[DEBUG] After smoothing: {smoothed_valid} frames, players: {smoothed_players}")
        
        # === TAHAP 3: Stable segments ===
        segments = self.get_stable_segments(smoothed)
        
        if debug:
            print(f"[DEBUG] Stable segments found: {len(segments)}")
            for i, seg in enumerate(segments):
                dur = seg['frame_end'] - seg['frame_start']
                print(f"[DEBUG]   Seg #{i}: Player {seg['player_id']}, "
                      f"frames {seg['frame_start']}-{seg['frame_end']} (duration={dur})")
            
            if len(segments) < 2:
                print(f"[DEBUG] *** PROBLEM: Less than 2 stable segments! ***")
                print(f"[DEBUG] *** Try lowering min_stable_frames (currently {self.min_stable_frames}) ***")
                return []
        
        # === TAHAP 4: Detect passes from transitions ===
        passes = []
        last_pass_frame = -999
        
        if debug:
            print(f"\n[DEBUG] === EVALUATING TRANSITIONS ===")
        
        for i in range(len(segments) - 1):
            seg_from = segments[i]
            seg_to = segments[i + 1]
            
            from_player = seg_from['player_id']
            to_player = seg_to['player_id']
            
            transition_frame_start = seg_from['frame_end']
            transition_frame_end = seg_to['frame_start']
            
            if debug:
                print(f"\n[DEBUG] Transition #{i}: Player {from_player} -> Player {to_player}")
            
            # Skip jika pemain sama
            if from_player == to_player:
                if debug:
                    print(f"[DEBUG]   SKIP: Same player")
                continue
            
            # Cooldown check
            if (transition_frame_end - last_pass_frame) < self.cooldown_frames:
                if debug:
                    print(f"[DEBUG]   SKIP: Cooldown ({transition_frame_end - last_pass_frame} < {self.cooldown_frames})")
                continue
            
            # Possession duration check
            from_duration = seg_from['frame_end'] - seg_from['frame_start']
            if from_duration < self.min_possession_duration:
                if debug:
                    print(f"[DEBUG]   SKIP: From-segment too short ({from_duration} < {self.min_possession_duration})")
                continue
            
            # Posisi pemain
            from_player_data = tracks['players'][transition_frame_start].get(from_player)
            to_player_data = tracks['players'][transition_frame_end].get(to_player)
            
            if not from_player_data or not to_player_data:
                if debug:
                    print(f"[DEBUG]   SKIP: Player data missing at transition frames")
                continue
            
            from_pos = get_center_of_bbox_bottom(from_player_data['bbox'])
            to_pos = get_center_of_bbox_bottom(to_player_data['bbox'])
            distance = measure_distance(from_pos, to_pos)
            
            # Validasi jarak
            if distance < self.min_pass_distance:
                if debug:
                    print(f"[DEBUG]   SKIP: Distance too short ({distance:.0f} < {self.min_pass_distance})")
                continue
            if distance > self.max_pass_distance:
                if debug:
                    print(f"[DEBUG]   SKIP: Distance too far ({distance:.0f} > {self.max_pass_distance})")
                continue
            
            # Validasi ball movement
            ball_movement = self.validate_ball_movement(
                tracks, transition_frame_start, transition_frame_end
            )
            
            if ball_movement < self.ball_movement_threshold:
                if debug:
                    print(f"[DEBUG]   SKIP: Ball didn't move enough ({ball_movement:.0f} < {self.ball_movement_threshold})")
                continue
            
            # === PASS VALID! ===
            if debug:
                print(f"[DEBUG]   *** PASS DETECTED! dist={distance:.0f}px, ball_move={ball_movement:.0f}px ***")
            
            pass_event = {
                'frame_start': transition_frame_start,
                'frame_end': transition_frame_end,
                'from_player': from_player,
                'to_player': to_player,
                'distance': distance,
                'ball_movement': ball_movement,
                'success': True,
                'from_pos': from_pos,
                'to_pos': to_pos,
            }
            passes.append(pass_event)
            last_pass_frame = transition_frame_end
        
        if debug:
            print(f"\n[DEBUG] === RESULT: {len(passes)} passes detected ===\n")
        
        return passes

    def get_pass_statistics(self, passes):
        stats = {
            'total_passes': len(passes),
            'avg_distance': np.mean([p['distance'] for p in passes]) if passes else 0,
        }
        return stats