import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox_bottom, get_center_of_bbox, check_line_intersection
import numpy as np
from collections import Counter

class PassDetector:
    def __init__(self, fps=24):
        self.fps = fps

        self.smoothing_window = 3
        self.min_stable_frames = 1        

        self.min_pass_distance = 30       
        self.max_pass_distance = 1500     
        self.cooldown_frames = 3          
        self.min_possession_duration = 1  

        self.ball_movement_check_radius = 25
        self.ball_movement_threshold = 4

        self.player_search_radius = 20

        self.pass_display_delay = 3
        self.min_display_gap = 3          

        self.jersey_map = {} 

    def set_jersey_map(self, player_identifier):
        self._player_identifier = player_identifier

    def _get_jersey(self, player_id):
        if hasattr(self, '_player_identifier'):
            return self._player_identifier.get_jersey_number_for_player(player_id)
        return str(player_id)

    def smooth_possessions(self, raw_possessions):
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

    def fill_short_gaps(self, possessions, max_gap=12):
        filled = list(possessions)
        last_valid = -1
        gap_start = -1
        for i in range(len(filled)):
            if filled[i] != -1:
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

    def _normalize_possessions_by_jersey(self, possessions):
        if not hasattr(self, '_player_identifier'):
            return possessions

        jersey_canonical = {}
        normalized = list(possessions)

        for i, pid in enumerate(possessions):
            if pid == -1:
                continue
            jersey = self._get_jersey(pid)
            if jersey not in jersey_canonical:
                jersey_canonical[jersey] = pid
            normalized[i] = jersey_canonical[jersey]

        return normalized

    def get_stable_segments(self, smoothed_possessions):
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

        direct_distance = measure_distance(ball_positions[0], ball_positions[-1])
        max_displacement = 0
        for i in range(1, len(ball_positions)):
            d = measure_distance(ball_positions[0], ball_positions[i])
            if d > max_displacement:
                max_displacement = d

        return max(direct_distance, max_displacement)

    def find_player_nearby(self, tracks, player_id, target_frame, search_radius=None):
        if search_radius is None:
            search_radius = self.player_search_radius
        total_frames = len(tracks['players'])

        player_data = tracks['players'][target_frame].get(player_id)
        if player_data:
            return player_data, target_frame

        for offset in range(1, search_radius + 1):
            check_frame = target_frame - offset
            if 0 <= check_frame < total_frames:
                player_data = tracks['players'][check_frame].get(player_id)
                if player_data:
                    return player_data, check_frame
            check_frame = target_frame + offset
            if 0 <= check_frame < total_frames:
                player_data = tracks['players'][check_frame].get(player_id)
                if player_data:
                    return player_data, check_frame
        return None, -1

    def detect_passes(self, tracks, ball_possessions, debug=True):
        filled = self.fill_short_gaps(ball_possessions, max_gap=12)
        filled = self._normalize_possessions_by_jersey(filled)
        smoothed = self.smooth_possessions(filled)
        segments = self.get_stable_segments(smoothed)

        if len(segments) < 2:
            return []

        merged = [segments[0]]
        for seg in segments[1:]:
            if seg['player_id'] == merged[-1]['player_id']:
                gap = seg['frame_start'] - merged[-1]['frame_end']
                if gap <= 10:
                    merged[-1]['frame_end'] = seg['frame_end']
                else:
                    merged.append(seg)
            else:
                merged.append(seg)
        segments = merged

        passes = []
        last_pass_frame = -999

        for i in range(len(segments) - 1):
            seg_from = segments[i]
            seg_to = segments[i + 1]
            from_player = seg_from['player_id']
            to_player = seg_to['player_id']
            transition_frame_start = seg_from['frame_end']
            transition_frame_end = seg_to['frame_start']

            from_jersey = self._get_jersey(from_player)
            to_jersey = self._get_jersey(to_player)

            if from_jersey == to_jersey:
                continue

            if (transition_frame_end - last_pass_frame) < self.cooldown_frames:
                continue

            from_duration = seg_from['frame_end'] - seg_from['frame_start']
            if from_duration < self.min_possession_duration:
                continue

            from_player_data, _ = self.find_player_nearby(tracks, from_player, transition_frame_start)
            to_player_data, _ = self.find_player_nearby(tracks, to_player, transition_frame_end)

            if not from_player_data or not to_player_data:
                continue

            from_pos = get_center_of_bbox_bottom(from_player_data['bbox'])
            to_pos = get_center_of_bbox_bottom(to_player_data['bbox'])
            distance = measure_distance(from_pos, to_pos)

            if distance < self.min_pass_distance or distance > self.max_pass_distance:
                continue

            ball_movement = self.validate_ball_movement(tracks, transition_frame_start, transition_frame_end)
            if ball_movement < self.ball_movement_threshold:
                continue

            # === BARU: CEK PASSING SUCCESS / FAILED ===
            is_success = False
            # Ambil frame tengah saat passing terjadi untuk mengambil posisi cone stabil
            mid_frame = (transition_frame_start + transition_frame_end) // 2
            cones_in_frame = tracks.get('cones', [])
            
            if len(cones_in_frame) > mid_frame:
                cone_data = cones_in_frame[mid_frame]
                if len(cone_data) >= 2:
                    # Ambil titik tengah bawah dari semua cone
                    cone_centers = [get_center_of_bbox_bottom(c['bbox']) for c in cone_data.values()]
                    # Urutkan dari kiri ke kanan berdasarkan sumbu X
                    cone_centers = sorted(cone_centers, key=lambda x: x[0])

                    for c_idx in range(len(cone_centers) - 1):
                        cone1 = cone_centers[c_idx]
                        cone2 = cone_centers[c_idx + 1]
                        
                        if check_line_intersection(from_pos, to_pos, cone1, cone2):
                            is_success = True
                            break

            receiver_start = seg_to['frame_start']
            pass_display_frame = min(receiver_start + self.pass_display_delay, seg_to['frame_end'])

            if len(passes) > 0:
                last_display = passes[-1]['frame_display']
                if pass_display_frame - last_display < self.min_display_gap:
                    pass_display_frame = last_display + self.min_display_gap

            pass_event = {
                'frame_start': transition_frame_start,
                'frame_end': transition_frame_end,
                'frame_display': pass_display_frame,
                'from_player': from_player,
                'to_player': to_player,
                'from_jersey': from_jersey,
                'to_jersey': to_jersey,
                'distance': distance,
                'ball_movement': ball_movement,
                'success': is_success,
                'from_pos': from_pos,
                'to_pos': to_pos,
            }
            passes.append(pass_event)
            last_pass_frame = transition_frame_end

        if debug:
            print(f"\n[DEBUG] === RESULT: {len(passes)} passes detected ===\n")

        return passes