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
    
    # Assign team untuk players
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #Assign goalkeeper ke tim berdasarkan posisi, tapi gunakan warna tim player
    frame_width = video_frames[0].shape[1]
    
    for frame_num, goalkeeper_track in enumerate(tracks['goalkeepers']):
        for goalkeeper_id, track in goalkeeper_track.items():
            bbox = track['bbox']
            x_center = (bbox[0] + bbox[2]) / 2
            
            # ✅ Tentukan tim berdasarkan posisi
            # Kiper kiri → Team 1 (sama dengan player tim kiri)
            # Kiper kanan → Team 2 (sama dengan player tim kanan)
            if x_center < frame_width / 2:
                team = 2
            else:
                team = 1
            
            tracks['goalkeepers'][frame_num][goalkeeper_id]['team'] = team
            
            # ✅ Gunakan warna TIM PLAYER (bukan warna jersey kiper)
            team_color = team_assigner.team_colors[team]
            
            # Konversi numpy array ke tuple
            if isinstance(team_color, np.ndarray):
                team_color = tuple(int(c) for c in team_color)
            
            tracks['goalkeepers'][frame_num][goalkeeper_id]['team_color'] = team_color
    
    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)


    #save_video
    save_video(output_video_frames, 'output_videos/football_analysis_output_color.avi')

if __name__ == '__main__':
    main()