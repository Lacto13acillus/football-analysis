from trackers import Tracker
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.pass_detector import PassDetector
from utils.bbox_utils import interpolate_ball_positions
from utils.video_utils import read_video
import cv2
import pickle
import os

def main():
    video_path = 'input_videos/passing_drill.mp4' 
    stub_path = 'stubs/track_stubs.pkl'
    output_path = 'output_videos/passing_drill_final.avi'

    # UBAH: Menggunakan model bawaan YOLO (Pre-trained COCO)
    # Model ini akan otomatis didownload jika belum ada di direktori Anda
    tracker = Tracker('yolov8m.pt')
    player_ball_assigner = PlayerBallAssigner()
    
    # === STEP 1: Get Tracks ===
    print("Membaca video dan memproses tracking...")
    video_frames = read_video(video_path)
    if os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
    else:
        tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path=stub_path)

    # === STEP 1.5: Interpolasi Posisi Bola ===
    tracks['ball'] = interpolate_ball_positions(tracks['ball'])

    # === STEP 2: DYNAMIC MAPPING DENGAN FOREGROUND FILTER ===
    print("Mengeksekusi Dynamic Mapping per frame...")
    id_votes = {} 

    for frame_num, players in enumerate(tracks['players']):
        current_players = []
        for track_id, data in players.items():
            t_id = int(track_id)
            bbox = data['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            x_center = (bbox[0] + bbox[2]) / 2
            current_players.append({'track_id': t_id, 'area': area, 'x_center': x_center})
        
        if not current_players:
            continue
            
        max_area = max([p['area'] for p in current_players])
        foreground_players = [p for p in current_players if p['area'] > max_area * 0.3]
        
        if len(foreground_players) >= 3:
            foreground_players.sort(key=lambda x: x['area'], reverse=True)
            top_3 = foreground_players[:3]
            top_3.sort(key=lambda x: x['x_center'])
            
            roles = ["Player A", "Player B", "Player C"]
            for i in range(3):
                t_id = top_3[i]['track_id']
                role = roles[i]
                
                if t_id not in id_votes:
                    id_votes[t_id] = {"Player A": 0, "Player B": 0, "Player C": 0}
                id_votes[t_id][role] += 1

    player_names = {}
    for track_id, votes in id_votes.items():
        best_role = max(votes, key=votes.get)
        if votes[best_role] > 5: 
            player_names[int(track_id)] = best_role

    print(f"Hasil Dynamic Mapping: {player_names}")

    stats_per_player = {"Player A": 0, "Player B": 0, "Player C": 0, "Other": 0}

    # === STEP 3: Possession Mapping ===
    print("Memproses penguasaan bola (possession)...")
    raw_ball_possessions = []
    for frame_num, frame_img in enumerate(video_frames):
        player_track = tracks['players'][frame_num]
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])

        if len(ball_bbox) == 0:
            raw_ball_possessions.append(-1)
        else:
            assigned_player, _ = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
            raw_ball_possessions.append(int(assigned_player) if assigned_player != -1 else -1)

    # === STEP 4: Pass Detection ===
    print("Mendeteksi passing...")
    cap_temp = cv2.VideoCapture(video_path)
    fps = int(cap_temp.get(cv2.CAP_PROP_FPS)) or 24
    cap_temp.release()

    pass_detector = PassDetector(fps=fps)
    detected_passes = pass_detector.detect_passes(tracks, raw_ball_possessions, debug=False)

    # === STEP 5: Render & Real-time Counter ===
    print("Merender video akhir...")
    height, width, _ = video_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for frame_num, frame in enumerate(video_frames):
        player_dict = tracks["players"][frame_num]
        
        for pass_event in detected_passes:
            if pass_event['frame_display'] == frame_num:
                sender_id = int(pass_event['from_player'])
                name = player_names.get(sender_id, "Other")
                
                if name in stats_per_player:
                    stats_per_player[name] += 1
                else:
                    stats_per_player["Other"] += 1

        for pass_event in detected_passes:
            if 0 <= frame_num - pass_event['frame_display'] <= 15:
                frame = tracker.draw_pass_arrow(frame, pass_event)

        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (300, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, "PASSING STATS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Player A : {stats_per_player['Player A']}", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player B : {stats_per_player['Player B']}", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player C : {stats_per_player['Player C']}", (40, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for track_id, player in player_dict.items():
            t_id = int(track_id)
            name = player_names.get(t_id, "Other")
            
            if name != "Other":
                color = (0, 255, 0)
                frame = tracker.draw_ellipse(frame, player["bbox"], color)
                cv2.putText(frame, name, (int(player["bbox"][0]), int(player["bbox"][1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                color = (200, 200, 200)
                frame = tracker.draw_ellipse(frame, player["bbox"], color)

            if raw_ball_possessions[frame_num] == t_id:
                frame = tracker.draw_traingle(frame, player["bbox"], (0, 0, 255))

        ball_dict = tracks["ball"][frame_num]
        for _, ball in ball_dict.items():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        out.write(frame)

    out.release()
    print(f"Selesai! Video berhasil disimpan di: {output_path}")

if __name__ == '__main__':
    main()