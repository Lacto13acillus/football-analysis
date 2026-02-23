from trackers import Tracker
from team_assigner import TeamAssigner
from trackers.player_ball_assigner import PlayerBallAssigner
import numpy as np
import cv2
import pickle
import os
from utils.video_utils import read_video, save_video

def main():
    video_path = 'input_videos/video_ssb_1.mp4'
    stub_path = 'stubs/track_stubs.pkl'
    output_path = 'output_videos/video_ssb_output.avi'
    
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
        
        video_frames = read_video(video_path)
        print(f"   Loaded {len(video_frames)} frames")
        
        print("   Running object detection & tracking...")
        tracks = tracker.get_object_track(video_frames,
                                           read_from_stub=False,
                                           stub_path=stub_path)
        del video_frames
        print("✅ Tracks saved to stub")

    # ===== STEP 2: Team Assignment =====
    print("🎨 Assigning team colors...")
    
    best_frame_idx = None
    best_frame = None
    best_score = 0
    
    cap = cv2.VideoCapture(video_path)
    
    for frame_idx in range(len(tracks['players'])):
        num_players = len(tracks['players'][frame_idx])
        num_referees = len(tracks['referees'][frame_idx])
        
        score = num_players - (num_referees * 3) 
        
        if num_players >= 4 and score > best_score:  
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                best_frame = frame
                best_frame_idx = frame_idx
                best_score = score
    
    if best_frame is None:
        print("❌ Error: No suitable frame found!")
        cap.release()
        return
    
    print(f"   Selected frame {best_frame_idx}: {len(tracks['players'][best_frame_idx])} players, {len(tracks['referees'][best_frame_idx])} referees")
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(best_frame, tracks['players'][best_frame_idx])
    
    def is_yellow_like(color):
        b, g, r = int(color[0]), int(color[1]), int(color[2])
        return (g > 180 and r > 180 and b < 150)  
    
    team_1_is_yellow = is_yellow_like(team_assigner.team_colors[1])
    team_2_is_yellow = is_yellow_like(team_assigner.team_colors[2])
    
    if team_1_is_yellow or team_2_is_yellow:
        print("   ⚠️  Warning: One team detected as yellow (referee might be in clustering)")
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(best_frame, track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    del best_frame 
    print("✅ Team assignment complete")

    # ===== STEP 3: Render Video with Annotations & Pass Detection =====
    print("🎬 Rendering annotated video...")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0

    # --- INISIALISASI VARIABEL PASSING (DIPERBARUI) ---
    player_assigner = PlayerBallAssigner()
    team_passes = {1: 0, 2: 0}
    
    confirmed_player_with_ball = None 
    current_team_in_possession = None
    
    candidate_player = None
    consecutive_frames_with_ball = 0
    
    # Syarat: Pemain harus memegang bola minimal 7 frame berturut-turut agar dianggap SAH
    FRAMES_REQUIRED_FOR_POSSESSION = 7 
    # --------------------------------------------------
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= len(tracks['players']):
            break
        
        player_dict = tracks["players"][frame_num]
        ball_dict = tracks["ball"][frame_num]
        referee_dict = tracks["referees"][frame_num]

        # ---------------- PASS DETECTION LOGIC (DIPERBARUI) ---------------- #
        assigned_player = -1
        if 1 in ball_dict: 
            ball_bbox = ball_dict[1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_dict, ball_bbox)

        if assigned_player != -1 and assigned_player is not None:
            # Cek apakah pemain yang dekat bola SAMA dengan frame sebelumnya
            if assigned_player == candidate_player:
                consecutive_frames_with_ball += 1
            else:
                candidate_player = assigned_player
                consecutive_frames_with_ball = 1

            # Jika sudah memenuhi batas frame (bola benar-benar dikuasai)
            if consecutive_frames_with_ball >= FRAMES_REQUIRED_FOR_POSSESSION:
                player_dict[assigned_player]['has_ball'] = True
                team = player_dict[assigned_player]['team']

                # Cek apakah bola berpindah penguasaan dari pemain sebelumnya
                if confirmed_player_with_ball is not None and confirmed_player_with_ball != assigned_player:
                    # Jika timnya sama, berarti PASS SUKSES!
                    if current_team_in_possession == team:
                        team_passes[team] += 1
                
                # Update status penguasaan bola yang sah
                confirmed_player_with_ball = assigned_player
                current_team_in_possession = team
        else:
            # Bola tidak dekat siapa-siapa (bola liar), reset perhitungan sementara
            candidate_player = None
            consecutive_frames_with_ball = 0
        # ------------------------------------------------------------------- #

        # Draw Players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = tracker.draw_ellipse(frame, player["bbox"], color, track_id)
            
            # Jika player bawa bola (sudah lolos filter frame), gambar segitiga merah
            if player.get('has_ball', False):
                 frame = tracker.draw_traingle(frame, player["bbox"], (0,0,255))

        # Draw Referee
        for _, referee in referee_dict.items():
            frame = tracker.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
        
        # Draw ball 
        for track_id, ball in ball_dict.items():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        # ---------------- STATISTIK PASSING ---------------- #
        cv2.rectangle(frame, (10, 10), (320, 100), (255, 255, 255), -1)
        cv2.putText(frame, f"Team 1 Passes: {team_passes[1]}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.putText(frame, f"Team 2 Passes: {team_passes[2]}", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        # --------------------------------------------------- #

        out.write(frame)
        frame_num += 1
        
        if frame_num % 50 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"   Progress: {frame_num}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    out.release()
    print(f"✅ Video saved to {output_path}")

if __name__ == '__main__':
    main()