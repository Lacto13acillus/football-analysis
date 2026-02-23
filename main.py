from trackers import Tracker
from team_assigner import TeamAssigner
from trackers.player_ball_assigner import PlayerBallAssigner
from utils.bbox_utils import get_center_of_bbox, measure_distance 
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

    # ===== STEP 1: Get Tracks =====
    if os.path.exists(stub_path):
        print("✅ Loading tracks from stub...")
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
        print(f"   Loaded {len(tracks['players'])} frames from stub")
    else:
        print("⚠️  No stub found. Running detection...")
        video_frames = read_video(video_path)
        tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path=stub_path)
        del video_frames
        print("✅ Tracks saved to stub")

    # ===== STEP 2: Team Assignment =====
    print("🎨 Assigning team colors...")
    best_frame_idx, best_frame, best_score = None, None, 0
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
        return
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(best_frame, tracks['players'][best_frame_idx])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(best_frame, track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    del best_frame 
    print("✅ Team assignment complete")

    # ===== STEP 3: Render Video & 3-PHASE PASS DETECTION =====
    print("🎬 Rendering annotated video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0

    # --- INISIALISASI VARIABEL 3-PHASE LOGIC ---
    player_assigner = PlayerBallAssigner()
    team_passes = {1: 0, 2: 0}
    
    last_possessor = None
    last_possessor_team = None
    last_possessor_pos = None
    
    candidate_player = None
    consecutive_frames_with_ball = 0
    frames_without_ball = 0
    
    # State Bola: 'FREE' (Bebas) atau 'POSSESSED' (Dikuasai)
    ball_state = 'FREE' 
    
    # PARAMETER KETAT (AKURASI TINGGI)
    FRAMES_REQUIRED_FOR_POSSESSION = 2 # Butuh 5 frame konstan untuk kuasai bola
    FRAMES_FOR_TRANSIT = 2             # Bola harus lepas >5 frame agar sah disebut sedang dioper
    MIN_PASS_DISTANCE = 65             # Jarak minimal operan (mencegah ID Switch saat dribble)
    PASS_LINE_DURATION = 25 
    
    active_pass_lines = []
    # -------------------------------------------
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= len(tracks['players']):
            break
        
        player_dict = tracks["players"][frame_num]
        ball_dict = tracks["ball"][frame_num]
        referee_dict = tracks["referees"][frame_num]

        # ---------------- 3-PHASE PASS DETECTION LOGIC ---------------- #
        assigned_player = -1
        if 1 in ball_dict: 
            ball_bbox = ball_dict[1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_dict, ball_bbox)

        if assigned_player != -1 and assigned_player is not None:
            frames_without_ball = 0 # Bola sedang ada di kaki seseorang
            
            if assigned_player == candidate_player:
                consecutive_frames_with_ball += 1
            else:
                candidate_player = assigned_player
                consecutive_frames_with_ball = 1

            if consecutive_frames_with_ball >= FRAMES_REQUIRED_FOR_POSSESSION:
                player_dict[assigned_player]['has_ball'] = True
                team = player_dict[assigned_player]['team']
                curr_pos = get_center_of_bbox(player_dict[assigned_player]['bbox'])

                # Logika Operan: Hanya hitung jika sebelumnya bola berstatus 'FREE' (Melayang/Menggulir jauh)
                if ball_state == 'FREE' and last_possessor is not None:
                    if last_possessor != assigned_player and last_possessor_team == team:
                        pass_distance = measure_distance(last_possessor_pos, curr_pos)
                        
                        if pass_distance > MIN_PASS_DISTANCE:
                            team_passes[team] += 1
                            active_pass_lines.append({
                                'p1': last_possessor_pos,
                                'p2': curr_pos,
                                'color': team_assigner.team_colors[team],
                                'frames_left': PASS_LINE_DURATION
                            })
                
                # Update Status Pemegang Bola Terakhir
                ball_state = 'POSSESSED'
                last_possessor = assigned_player
                last_possessor_team = team
                last_possessor_pos = curr_pos
        else:
            candidate_player = None
            consecutive_frames_with_ball = 0
            frames_without_ball += 1
            
            # Jika bola tidak tersentuh siapa pun selama 5 frame berturut-turut,
            # berarti bola sedang ditendang (dioper) atau lepas dari penguasaan.
            if frames_without_ball >= FRAMES_FOR_TRANSIT:
                ball_state = 'FREE'
        # -------------------------------------------------------------- #

        # Draw Annotations
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = tracker.draw_ellipse(frame, player["bbox"], color, track_id)
            if player.get('has_ball', False):
                 frame = tracker.draw_traingle(frame, player["bbox"], (0,0,255))

        for _, referee in referee_dict.items():
            frame = tracker.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
        
        for track_id, ball in ball_dict.items():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        # Render Pass Lines
        active_pass_lines = [line for line in active_pass_lines if line['frames_left'] > 0]
        for line_info in active_pass_lines:
            p1, p2 = line_info['p1'], line_info['p2']
            color = (int(line_info['color'][0]), int(line_info['color'][1]), int(line_info['color'][2]))
            cv2.line(frame, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
            cv2.circle(frame, p1, radius=6, color=color, thickness=-1)
            cv2.circle(frame, p2, radius=8, color=(255, 255, 255), thickness=-1)
            cv2.circle(frame, p2, radius=8, color=color, thickness=2)
            line_info['frames_left'] -= 1

        # Draw Stats
        cv2.rectangle(frame, (10, 10), (320, 100), (255, 255, 255), -1)
        cv2.putText(frame, f"Team 1 Passes: {team_passes[1]}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.putText(frame, f"Team 2 Passes: {team_passes[2]}", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        out.write(frame)
        frame_num += 1
        
        if frame_num % 50 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"   Progress: {frame_num}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    out.release()
    print(f"✅ Video saved to {output_path}. Siap untuk presentasi besok!")

if __name__ == '__main__':
    main()