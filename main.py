from trackers import Tracker
from team_assigner import TeamAssigner
import numpy as np
import cv2
import pickle
import os

def main():
    video_path = 'input_videos/video_ssb_1.mp4'
    stub_path = 'stubs/track_stubs.pkl'
    output_path = 'output_videos/video_ssb_1_output.avi'
    
    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # ===== STEP 1: Get Tracks (Detection & Tracking) =====
    if os.path.exists(stub_path):
        print("✅ Loading tracks from stub...")
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
        print(f"   Loaded {len(tracks['players'])} frames from stub")
    else:
        print("⚠️  No stub found. Running detection (this will take time)...")
        print("   Loading video frames...")
        
        # Load video for detection
        from utils.video_utils import read_video
        video_frames = read_video(video_path)
        print(f"   Loaded {len(video_frames)} frames")
        
        print("   Running object detection & tracking...")
        tracks = tracker.get_object_track(video_frames,
                                           read_from_stub=False,
                                           stub_path=stub_path)
        
        # Free memory
        del video_frames
        print("✅ Tracks saved to stub")

    # ===== STEP 2: Team Assignment =====
    print("🎨 Assigning team colors...")
    
    # ✅ Find first frame with players
    first_frame_with_players = None
    first_frame_idx = None
    
    cap = cv2.VideoCapture(video_path)
    
    for frame_idx, player_track in enumerate(tracks['players']):
        if len(player_track) >= 2:  # Frame has players
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                first_frame_with_players = frame
                first_frame_idx = frame_idx
                print(f"   Found players at frame {frame_idx}")
                break
    
    if first_frame_with_players is None:
        print("❌ Error: No players found in entire video!")
        cap.release()
        return
    
    # Assign team colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(first_frame_with_players, tracks['players'][first_frame_idx])
    
    print(f"   Team 1 color: {team_assigner.team_colors[1]}")
    print(f"   Team 2 color: {team_assigner.team_colors[2]}")
    
    # Assign team to each player in each frame
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(first_frame_with_players,
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    del first_frame_with_players  # Free memory
    print("✅ Team assignment complete")

    # ===== STEP 3: Render Video with Annotations =====
    print("🎬 Rendering annotated video...")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num >= len(tracks['players']):
            break
        
        # Get detections for this frame
        player_dict = tracks["players"][frame_num]
        ball_dict = tracks["ball"][frame_num]
        referee_dict = tracks["referees"][frame_num]

        # Draw Players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = tracker.draw_ellipse(frame, player["bbox"], color, track_id)

        # Draw Referee
        for _, referee in referee_dict.items():
            frame = tracker.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
        
        # Draw ball 
        for track_id, ball in ball_dict.items():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        # Write frame
        out.write(frame)
        
        frame_num += 1
        
        # Progress indicator
        if frame_num % 50 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"   Progress: {frame_num}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    print(f"✅ Video saved to {output_path}")
    print(f"   Total frames processed: {frame_num}")

if __name__ == '__main__':
    main()