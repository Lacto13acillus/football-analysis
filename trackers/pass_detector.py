import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox_bottom
import numpy as np

class PassDetector:
    def __init__(self):
        # DITURUNKAN menjadi 4 frame agar sentuhan bola yang sangat cepat tetap dihitung
        self.pass_threshold_frames = 4  
        self.max_pass_distance = 500  
        self.min_pass_distance = 40   
        
    def detect_passes(self, tracks, ball_possessions):
        passes = []
        last_possession = None
        possession_start_frame = None
        
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
                        
                        if self.min_pass_distance < distance < self.max_pass_distance:
                            
                            pass_event = {
                                'frame_start': possession_start_frame,
                                'frame_end': frame_num,
                                'from_player': last_possession,
                                'to_player': current_possession,
                                'distance': distance,
                                'success': True, # Semuanya dianggap sukses karena ini drill kawan
                                'from_pos': from_pos,
                                'to_pos': to_pos
                            }
                            
                            passes.append(pass_event)
                
                last_possession = current_possession
                possession_start_frame = frame_num
        
        return passes
    
    def get_pass_statistics(self, passes):
        """ Menghitung total umpan secara keseluruhan """
        stats = {
            'total_passes': len(passes)
        }
        return stats