from utils.video_utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

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

    # Assign team untuk goalkeepers berdasarkan posisi X
    for frame_num, goalkeeper_track in enumerate(tracks['goalkeepers']):
        for goalkeeper_id, track in goalkeeper_track.items():
            bbox = track['bbox']
            x_center = (bbox[0] + bbox[2]) / 2
            
            # Goalkeeper di sisi kiri → Team 1, di sisi kanan → Team 2
            if x_center < video_frames[0].shape[1] / 2:
                team = 1
            else:
                team = 2
            
            tracks['goalkeepers'][frame_num][goalkeeper_id]['team'] = team
            tracks['goalkeepers'][frame_num][goalkeeper_id]['team_color'] = team_assigner.team_colors[team]
    
    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)


    #save_video
    save_video(output_video_frames, 'output_videos/football_analysis_output_color.avi')

if __name__ == '__main__':
    main()