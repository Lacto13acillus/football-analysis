import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox_bottom
import numpy as np

class PassDetector:
    def __init__(self):
        self.pass_threshold_frames = 15  # Minimum frames untuk dianggap pass (0.5 detik @ 30fps)
        self.max_pass_distance = 500  # Maximum pixel distance untuk pass yang valid
        self.min_pass_distance = 50   # Minimum pixel distance untuk pass yang valid
        
    def detect_passes(self, tracks, ball_possessions):
        """
        Detect passes between players
        
        Args:
            tracks: Dictionary containing player, ball, referee tracks
            ball_possessions: List of player IDs who possess the ball per frame
            
        Returns:
            List of pass events with metadata
        """
        passes = []
        last_possession = None
        possession_start_frame = None
        
        for frame_num in range(len(ball_possessions)):
            current_possession = ball_possessions[frame_num]
            
            # Skip if no one has the ball
            if current_possession == -1:
                continue
            
            # Initialize first possession
            if last_possession is None:
                last_possession = current_possession
                possession_start_frame = frame_num
                continue
            
            # Check if possession changed
            if current_possession != last_possession:
                # Calculate possession duration
                possession_duration = frame_num - possession_start_frame
                
                # Only count as pass if possession lasted minimum frames
                if possession_duration >= self.pass_threshold_frames:
                    # Get player data
                    from_player_data = tracks['players'][possession_start_frame].get(last_possession)
                    to_player_data = tracks['players'][frame_num].get(current_possession)
                    
                    if from_player_data and to_player_data:
                        # Calculate pass distance
                        from_pos = get_center_of_bbox_bottom(from_player_data['bbox'])
                        to_pos = get_center_of_bbox_bottom(to_player_data['bbox'])
                        distance = measure_distance(from_pos, to_pos)
                        
                        # Validate pass distance
                        if self.min_pass_distance < distance < self.max_pass_distance:
                            from_team = from_player_data.get('team', 0)
                            to_team = to_player_data.get('team', 0)
                            
                            # Pass is successful if same team
                            success = (from_team == to_team and from_team != 0)
                            
                            pass_event = {
                                'frame_start': possession_start_frame,
                                'frame_end': frame_num,
                                'from_player': last_possession,
                                'to_player': current_possession,
                                'from_team': from_team,
                                'to_team': to_team,
                                'distance': distance,
                                'success': success,
                                'from_pos': from_pos,
                                'to_pos': to_pos
                            }
                            
                            passes.append(pass_event)
                
                # Update for next iteration
                last_possession = current_possession
                possession_start_frame = frame_num
        
        return passes
    
    def get_pass_statistics(self, passes):
        """
        Calculate pass statistics per team
        """
        stats = {
            1: {'total': 0, 'successful': 0, 'failed': 0, 'avg_distance': 0, 'total_distance': 0},
            2: {'total': 0, 'successful': 0, 'failed': 0, 'avg_distance': 0, 'total_distance': 0}
        }
        
        for pass_event in passes:
            team = pass_event['from_team']
            if team in [1, 2]:
                stats[team]['total'] += 1
                stats[team]['total_distance'] += pass_event['distance']
                
                if pass_event['success']:
                    stats[team]['successful'] += 1
                else:
                    stats[team]['failed'] += 1
        
        # Calculate averages
        for team in [1, 2]:
            if stats[team]['total'] > 0:
                stats[team]['avg_distance'] = stats[team]['total_distance'] / stats[team]['total']
                stats[team]['success_rate'] = (stats[team]['successful'] / stats[team]['total']) * 100
            else:
                stats[team]['success_rate'] = 0
        
        return stats