import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox_bottom
import numpy as np

class PassDetector:
    def __init__(self):
        self.pass_threshold_frames = 5  # Sedikit dinaikkan untuk mencegah sentuhan tak sengaja
        self.max_pass_distance = 500  
        self.min_pass_distance = 90   # Ditingkatkan dari 40 ke 90 pixel agar jarak dekat tidak dihitung umpan
        
    def detect_passes(self, tracks, ball_possessions):
        passes = []
        last_possession = None
        possession_start_frame = None
        
        # FITUR BARU: Cooldown untuk mencegah perhitungan ganda (Flickering)
        last_pass_frame = -999 
        cooldown_frames = 15 # Jeda minimal 0.5 detik (@30fps) sebelum umpan baru bisa dicatat
        
        for frame_num in range(len(ball_possessions)):
            current_possession = ball_possessions[frame_num]
            
            if current_possession == -1:
                continue
            
            if last_possession is None:
                last_possession = current_possession
                possession_start_frame = frame_num
                continue
            
            if current_possession != last_possession:
                possession_duration = frame_num - possession_start_frame
                
                if possession_duration >= self.pass_threshold_frames:
                    from_player_data = tracks['players'][possession_start_frame].get(last_possession)
                    to_player_data = tracks['players'][frame_num].get(current_possession)
                    
                    if from_player_data and to_player_data:
                        from_pos = get_center_of_bbox_bottom(from_player_data['bbox'])
                        to_pos = get_center_of_bbox_bottom(to_player_data['bbox'])
                        distance = measure_distance(from_pos, to_pos)
                        
                        # Validasi: Jaraknya harus cukup jauh DAN sudah melewati masa cooldown
                        if self.min_pass_distance < distance < self.max_pass_distance:
                            if (frame_num - last_pass_frame) > cooldown_frames:
                                pass_event = {
                                    'frame_start': possession_start_frame,
                                    'frame_end': frame_num,
                                    'from_player': last_possession,
                                    'to_player': current_possession,
                                    'distance': distance,
                                    'success': True,
                                    'from_pos': from_pos,
                                    'to_pos': to_pos
                                }
                                
                                passes.append(pass_event)
                                last_pass_frame = frame_num # Catat frame umpan ini untuk cooldown berikutnya
                
                last_possession = current_possession
                possession_start_frame = frame_num
        
        return passes
    
    def get_pass_statistics(self, passes):
        """ Menghitung total umpan secara keseluruhan """
        stats = {
            'total_passes': len(passes)
        }
        return stats