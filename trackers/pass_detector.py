import numpy as np
from collections import defaultdict
from utils.bbox_utils import get_center_of_bbox, measure_distance

class PassDetector:
    def __init__(self, min_pass_distance=50, ball_possession_frames=5):
        """
        Initialize Pass Detector
        
        Args:
            min_pass_distance: Minimum distance in pixels to consider a pass
            ball_possession_frames: Number of frames a player must have ball to be considered in possession
        """
        self.min_pass_distance = min_pass_distance
        self.ball_possession_frames = ball_possession_frames
        self.passes = []  # List to store all passes
        self.current_possessor = None
        self.possession_counter = 0
        self.ball_history = []  # Track ball positions for debugging
        self.pass_events = []  # Store pass events with metadata
        
    def reset(self):
        """Reset the detector state"""
        self.passes = []
        self.current_possessor = None
        self.possession_counter = 0
        self.ball_history = []
        self.pass_events = []
    
    def find_ball_possessor(self, players_dict, ball_bbox, frame_num):
        """
        Find which player currently has the ball based on proximity
        
        Args:
            players_dict: Dictionary of players in current frame
            ball_bbox: Bounding box of the ball
            frame_num: Current frame number
            
        Returns:
            player_id of the possessor or None
        """
        if not ball_bbox or not players_dict:
            return None
        
        # Get center of ball
        ball_center = get_center_of_bbox(ball_bbox)
        self.ball_history.append({
            'frame': frame_num,
            'position': ball_center
        })
        
        # Find closest player to the ball
        min_distance = float('inf')
        closest_player = None
        
        for player_id, player_info in players_dict.items():
            if 'bbox' not in player_info:
                continue
                
            # Use foot position for more accurate ball possession
            player_foot = self.get_player_foot_position(player_info['bbox'])
            distance = measure_distance(ball_center, player_foot)
            
            # Also check distance to player center as fallback
            player_center = get_center_of_bbox(player_info['bbox'])
            distance_center = measure_distance(ball_center, player_center)
            
            # Use the smaller distance
            actual_distance = min(distance, distance_center)
            
            # Threshold for possession (adjust based on your video scale)
            if actual_distance < 50 and actual_distance < min_distance:
                min_distance = actual_distance
                closest_player = player_id
        
        return closest_player
    
    def get_player_foot_position(self, bbox):
        """Get the foot position of a player from bbox"""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int(y2))
    
    def detect_passes(self, tracks, fps=24):
        """
        Detect passes throughout the entire match
        
        Args:
            tracks: Dictionary containing player and ball tracks
            fps: Frames per second of the video
            
        Returns:
            List of detected passes with metadata
        """
        self.reset()
        
        # Group passes by team
        team_passes = {1: [], 2: []}
        total_passes = 0
        
        for frame_num in range(len(tracks['players'])):
            players_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            
            # Skip if no ball detected
            if 1 not in ball_dict:
                continue
                
            ball_bbox = ball_dict[1]['bbox']
            
            # Find current ball possessor
            possessor = self.find_ball_possessor(players_dict, ball_bbox, frame_num)
            
            if possessor is not None:
                # Get player's team
                if possessor in players_dict and 'team' in players_dict[possessor]:
                    team = players_dict[possessor]['team']
                    
                    # Check if possession changed
                    if self.current_possessor is not None and self.current_possessor != possessor:
                        # Verify it's a valid pass (not just a loose ball)
                        prev_player = self.current_possessor
                        
                        # Check if both players are from the same team (team pass)
                        if (prev_player in tracks['players'][frame_num-1] and 
                            'team' in tracks['players'][frame_num-1][prev_player] and
                            tracks['players'][frame_num-1][prev_player]['team'] == team):
                            
                            # Calculate pass distance
                            prev_pos = self.get_player_foot_position(
                                tracks['players'][frame_num-1][prev_player]['bbox']
                            )
                            curr_pos = self.get_player_foot_position(
                                players_dict[possessor]['bbox']
                            )
                            pass_distance = measure_distance(prev_pos, curr_pos)
                            
                            # Create pass event
                            pass_event = {
                                'from_player': prev_player,
                                'to_player': possessor,
                                'team': team,
                                'frame': frame_num,
                                'distance': pass_distance,
                                'time': frame_num / fps,  # Time in seconds
                                'from_position': prev_pos,
                                'to_position': curr_pos
                            }
                            
                            self.passes.append(pass_event)
                            team_passes[team].append(pass_event)
                            total_passes += 1
                            
                            print(f"⚽ Pass detected at frame {frame_num}: "
                                  f"Player {prev_player} → Player {possessor} "
                                  f"(Team {team}, Distance: {pass_distance:.1f}px)")
                    
                    self.current_possessor = possessor
                    self.possession_counter = self.ball_possession_frames
            else:
                # No possessor, decrement counter
                if self.possession_counter > 0:
                    self.possession_counter -= 1
                else:
                    self.current_possessor = None
        
        return self.passes
    
    def get_pass_statistics(self):
        """
        Calculate comprehensive pass statistics
        
        Returns:
            Dictionary with pass statistics
        """
        if not self.passes:
            return {
                'total_passes': 0,
                'passes_by_team': {1: 0, 2: 0},
                'player_passes': {},
                'average_pass_distance': 0,
                'longest_pass': None,
                'shortest_pass': None
            }
        
        # Initialize statistics
        stats = {
            'total_passes': len(self.passes),
            'passes_by_team': {1: 0, 2: 0},
            'player_passes': defaultdict(lambda: {'completed': 0, 'received': 0, 'to': defaultdict(int)}),
            'pass_distances': [],
            'longest_pass': max(self.passes, key=lambda x: x['distance']),
            'shortest_pass': min(self.passes, key=lambda x: x['distance'])
        }
        
        # Calculate statistics
        total_distance = 0
        for pass_event in self.passes:
            team = pass_event['team']
            from_p = pass_event['from_player']
            to_p = pass_event['to_player']
            
            # Team passes
            stats['passes_by_team'][team] += 1
            
            # Player statistics
            stats['player_passes'][from_p]['completed'] += 1
            stats['player_passes'][from_p]['to'][to_p] += 1
            stats['player_passes'][to_p]['received'] += 1
            
            # Distances
            stats['pass_distances'].append(pass_event['distance'])
            total_distance += pass_event['distance']
        
        # Average pass distance
        stats['average_pass_distance'] = total_distance / len(self.passes)
        
        # Find top passers
        stats['top_passers'] = sorted(
            [(pid, data['completed']) for pid, data in stats['player_passes'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return stats
    
    def visualize_pass_network(self, tracks, frame_num):
        """
        Visualize pass network for a specific frame
        Useful for debugging and visualization
        """
        if not self.passes:
            return {}
        
        # Get all passes up to current frame
        relevant_passes = [p for p in self.passes if p['frame'] <= frame_num]
        
        # Create pass network
        network = defaultdict(lambda: defaultdict(int))
        
        for pass_event in relevant_passes:
            from_p = pass_event['from_player']
            to_p = pass_event['to_player']
            network[from_p][to_p] += 1
        
        return network

    def get_team_possession_stats(self, tracks):
        """
        Calculate ball possession statistics by team
        
        Args:
            tracks: Dictionary with tracking data
            
        Returns:
            Dictionary with possession stats
        """
        team_possession_frames = {1: 0, 2: 0}
        total_frames_with_ball = 0
        
        for frame_num in range(len(tracks['players'])):
            if 'ball' not in tracks or frame_num >= len(tracks['ball']):
                continue
                
            ball_dict = tracks['ball'][frame_num]
            if 1 not in ball_dict:
                continue
                
            # Find possessor for this frame
            players_dict = tracks['players'][frame_num]
            possessor = self.find_ball_possessor(players_dict, ball_dict[1]['bbox'], frame_num)
            
            if possessor and possessor in players_dict and 'team' in players_dict[possessor]:
                team = players_dict[possessor]['team']
                team_possession_frames[team] += 1
                total_frames_with_ball += 1
        
        # Calculate percentages
        possession_stats = {}
        if total_frames_with_ball > 0:
            for team in team_possession_frames:
                possession_stats[team] = {
                    'frames': team_possession_frames[team],
                    'percentage': (team_possession_frames[team] / total_frames_with_ball) * 100
                }
        else:
            possession_stats = {1: {'frames': 0, 'percentage': 0}, 
                                2: {'frames': 0, 'percentage': 0}}
        
        return possession_stats