from trackers import Tracker
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.pass_detector import PassDetector
# from team_assigner import TeamAssigner # DIBLOKIR: Tidak perlu team assigner
import numpy as np
import cv2
import pickle
import os

def main():
    video_path = 'input_videos/passing.mp4'
    stub_path = 'stubs/track_stubs.pkl'
    output_path = 'output_videos/passing.avi'
    
    tracker = Tracker('models/best.pt')
    # team_assigner = TeamAssigner() # DIBLOKIR
    player_ball_assigner = PlayerBallAssigner()
    pass_detector = PassDetector()

    # ===== STEP 1: Get Tracks =====
    if os.path.exists(stub_path):
        print("Loading tracks from stub...")
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
        print(f"   Loaded {len(tracks['players'])} frames from stub")
    else:
        print("No stub found. Running detection...")
        from utils.video_utils import read_video
        video_frames = read_video(video_path)
        print(f"   Loaded {len(video_frames)} frames")
        
        print("   Running object detection & tracking...")
        tracks = tracker.get_object_track(video_frames,
                                           read_from_stub=False,
                                           stub_path=stub_path)
        del video_frames
        print("Tracks saved to stub")

    # ===== STEP 2: Team Assignment (DIMODIFIKASI UNTUK 1 TIM) =====
    print("Assigning all players to Training Group...")
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            tracks['players'][frame_num][player_id]['team'] = 1 
            tracks['players'][frame_num][player_id]['team_color'] = (255, 255, 255) # Warna putih untuk semua pemain
    
    print("Team assignment complete")

    # ===== STEP 3: Ball Possession =====
    print("Detecting ball possession...")
    
    ball_possessions = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])
        
        if len(ball_bbox) == 0:
            ball_possessions.append(-1)
        else:
            assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
            ball_possessions.append(assigned_player)
    
    print(f"   Processed {len(ball_possessions)} frames")

    # ===== STEP 4: Pass Detection =====
    print("Detecting passes...")
    
    passes = pass_detector.detect_passes(tracks, ball_possessions)
    pass_stats = pass_detector.get_pass_statistics(passes)
    
    print(f"   Detected {pass_stats['total_passes']} total passes")

    # ===== STEP 5: Render Video =====
    print("🎬 Rendering video with pass annotations...")
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    active_passes = []
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= len(tracks['players']):
            break
        
        player_dict = tracks["players"][frame_num]
        ball_dict = tracks["ball"][frame_num]
        
        # DIBLOKIR: Data wasit tidak lagi diperlukan
        # referee_dict = tracks["referees"][frame_num] 

        # Draw Players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (255, 255, 255))
            frame = tracker.draw_ellipse(frame, player["bbox"], color, track_id)
            
            if ball_possessions[frame_num] == track_id:
                frame = tracker.draw_traingle(frame, player["bbox"], (0, 0, 255)) # Segitiga merah untuk yang pegang bola

        # DIBLOKIR: Draw Referee
        # for _, referee in referee_dict.items():
        #     frame = tracker.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
        
        # Draw ball 
        for track_id, ball in ball_dict.items():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))
        
        for pass_event in passes:
            if pass_event['frame_end'] == frame_num:
                active_passes.append({
                    'pass': pass_event,
                    'frames_remaining': 30  
                })
        
        passes_to_remove = []
        for i, active_pass in enumerate(active_passes):
            if active_pass['frames_remaining'] > 0:
                frame = tracker.draw_pass_arrow(frame, active_pass['pass'])
                active_pass['frames_remaining'] -= 1
            else:
                passes_to_remove.append(i)
        
        for i in reversed(passes_to_remove):
            active_passes.pop(i)
        
        # Draw pass statistics
        frame = tracker.draw_pass_statistics(frame, pass_stats)

        out.write(frame)
        frame_num += 1
        
        if frame_num % 50 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"   Progress: {frame_num}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    print(f"Video saved to {output_path}")

if __name__ == '__main__':
    main()