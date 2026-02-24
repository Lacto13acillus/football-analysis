import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox_bottom, get_center_of_bbox
import numpy as np
from collections import Counter

class PassDetector:
    def __init__(self, fps=24):
        self.fps = fps
        
        # === PARAMETER POSSESSION SMOOTHING ===
        self.smoothing_window = 7          # Window untuk majority vote (ganjil)
        self.min_stable_frames = 8         # Minimum frame stabil sebelum dianggap possession valid
        
        # === PARAMETER PASS VALIDATION ===
        self.min_pass_distance = 80        # Jarak minimum (pixel) agar dianggap pass
        self.max_pass_distance = 600       # Jarak maksimum (terlalu jauh = bukan pass)
        self.cooldown_frames = 20          # Jeda minimum antar pass (~0.83 detik @24fps)
        self.min_possession_duration = 5   # Minimum frame possession sebelum bisa "mengirim" pass
        
        # === PARAMETER BALL MOVEMENT VALIDATION ===
        self.ball_movement_threshold = 30  # Bola harus bergerak minimal segini pixel

    def smooth_possessions(self, raw_possessions):
        """
        Tahap 1: Smoothing possession menggunakan Majority Vote sliding window.
        Menghilangkan flickering di mana possession berubah-ubah cepat.
        """
        smoothed = list(raw_possessions)
        half_window = self.smoothing_window // 2
        
        for i in range(half_window, len(raw_possessions) - half_window):
            window = raw_possessions[i - half_window : i + half_window + 1]
            # Hanya hitung yang bukan -1
            valid = [p for p in window if p != -1]
            if len(valid) >= half_window + 1:  # Minimal setengah window harus valid
                most_common = Counter(valid).most_common(1)[0][0]
                smoothed[i] = most_common
            elif len(valid) > 0:
                smoothed[i] = Counter(valid).most_common(1)[0][0]
            else:
                smoothed[i] = -1
        
        return smoothed

    def get_stable_segments(self, smoothed_possessions):
        """
        Tahap 2: Identifikasi segmen possession yang stabil.
        Mengembalikan list of (player_id, frame_start, frame_end).
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
        Tahap 3: Validasi bahwa bola benar-benar bergerak antara dua segment.
        Jika bola tidak bergerak, ini bukan pass — hanya noise.
        """
        ball_positions = []
        for f in range(frame_start, min(frame_end + 1, len(tracks['ball']))):
            ball_data = tracks['ball'][f].get(1)
            if ball_data and 'bbox' in ball_data:
                pos = get_center_of_bbox(ball_data['bbox'])
                ball_positions.append(pos)
        
        if len(ball_positions) < 2:
            return 0
        
        # Hitung total jarak perpindahan bola
        total_movement = 0
        for i in range(1, len(ball_positions)):
            total_movement += measure_distance(ball_positions[i-1], ball_positions[i])
        
        return total_movement

    def detect_passes(self, tracks, ball_possessions):
        """
        Pipeline deteksi pass yang jauh lebih akurat:
        1. Smooth raw possessions (hilangkan flicker)
        2. Identifikasi segmen possession stabil
        3. Validasi transisi antar segmen sebagai pass
        """
        # === TAHAP 1: Smoothing ===
        smoothed = self.smooth_possessions(ball_possessions)
        
        # === TAHAP 2: Segmen stabil ===
        segments = self.get_stable_segments(smoothed)
        
        if len(segments) < 2:
            return []
        
        # === TAHAP 3: Deteksi pass dari transisi antar segmen ===
        passes = []
        last_pass_frame = -999
        
        for i in range(len(segments) - 1):
            seg_from = segments[i]
            seg_to = segments[i + 1]
            
            from_player = seg_from['player_id']
            to_player = seg_to['player_id']
            
            # Skip jika pemain sama (bukan pass)
            if from_player == to_player:
                continue
            
            # Frame transisi
            transition_frame_start = seg_from['frame_end']
            transition_frame_end = seg_to['frame_start']
            
            # Cooldown check
            if (transition_frame_end - last_pass_frame) < self.cooldown_frames:
                continue
            
            # Possession duration check (segment "from" harus cukup lama)
            from_duration = seg_from['frame_end'] - seg_from['frame_start']
            if from_duration < self.min_possession_duration:
                continue
            
            # Ambil posisi pemain
            from_player_data = tracks['players'][transition_frame_start].get(from_player)
            to_player_data = tracks['players'][transition_frame_end].get(to_player)
            
            if not from_player_data or not to_player_data:
                continue
            
            from_pos = get_center_of_bbox_bottom(from_player_data['bbox'])
            to_pos = get_center_of_bbox_bottom(to_player_data['bbox'])
            distance = measure_distance(from_pos, to_pos)
            
            # Validasi jarak
            if distance < self.min_pass_distance or distance > self.max_pass_distance:
                continue
            
            # Validasi pergerakan bola (bola harus benar-benar berpindah)
            ball_movement = self.validate_ball_movement(
                tracks, transition_frame_start, transition_frame_end
            )
            if ball_movement < self.ball_movement_threshold:
                continue
            
            # === PASS VALID ===
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
                'from_segment_duration': from_duration,
            }
            passes.append(pass_event)
            last_pass_frame = transition_frame_end
        
        return passes

    def get_pass_statistics(self, passes):
        stats = {
            'total_passes': len(passes),
            'avg_distance': np.mean([p['distance'] for p in passes]) if passes else 0,
            'avg_ball_movement': np.mean([p['ball_movement'] for p in passes]) if passes else 0,
        }
        return stats