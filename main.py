from utils.video_utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
import numpy as np

def main():
    #read_video
    video_frames = read_video('input_videos/football_analysis.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_track(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Count players per team in each frame
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        team_1_count = 0
        team_2_count = 0
        
        for player_id, track in player_track.items():
            if track.get('team') == 1:
                team_1_count += 1
            elif track.get('team') == 2:
                team_2_count += 1
        
        if team_1_count > team_2_count:
            team_ball_control.append(1)
        elif team_2_count > team_1_count:
            team_ball_control.append(2)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
    
    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    #save_video
    save_video(output_video_frames, 'output_videos/football_analysis_output_color.avi')

if __name__ == '__main__':
    main()