from trackers import Tracker
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.pass_detector import PassDetector
from team_assigner.player_identifier import PlayerIdentifier
from utils.bbox_utils import interpolate_ball_positions
from utils.video_utils import read_video
import cv2
import pickle
import os

def main():
    video_path = 'input_videos/passing_ssb.mp4'
    stub_path = 'stubs/track_stubs.pkl'
    output_path = 'output_videos/passing_ssb.avi'

    tracker = Tracker('models/best.pt')
    player_ball_assigner = PlayerBallAssigner()
    player_identifier = PlayerIdentifier()
    
    # Statistik passing target
    stats_per_jersey = {"9": 0, "15": 0, "30": 0, "Other": 0}

    # === STEP 1: Get Tracks ===
    video_frames = read_video(video_path)
    if os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
    else:
        tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path=stub_path)

    # === STEP 1.5: Interpolasi & Identifikasi ===
    tracks['ball'] = interpolate_ball_positions(tracks['ball'])

    # === STEP 2: Possession & OCR Mapping ===
    print("Processing identities and possession...")
    raw_ball_possessions = []
    
    for frame_num, frame_img in enumerate(video_frames):
        player_track = tracks['players'][frame_num]
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])

        # Sambil jalan, identifikasi nomor punggung pemain di tiap frame
        player_identifier.update_identities(frame_img, player_track)

        if len(ball_bbox) == 0:
            raw_ball_possessions.append(-1)
        else:
            assigned_player, _ = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
            raw_ball_possessions.append(assigned_player)

    # === STEP 3: Pass Detection ===
    cap_temp = cv2.VideoCapture(video_path)
    fps = int(cap_temp.get(cv2.CAP_PROP_FPS)) or 24
    cap_temp.release()

    pass_detector = PassDetector(fps=fps)
    detected_passes = pass_detector.detect_passes(tracks, raw_ball_possessions)

    # === STEP 4: Render & Real-time Counter ===
    print("Rendering final video...")
    height, width, _ = video_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for frame_num, frame in enumerate(video_frames):
        player_dict = tracks["players"][frame_num]
        
        # Logika Penambahan Score Berdasarkan Nomor Punggung
        for pass_event in detected_passes:
            if pass_event['frame_display'] == frame_num:
                sender_id = pass_event['from_player']
                jersey_no = player_identifier.player_numbers_map.get(sender_id, "Unknown")
                
                if jersey_no in stats_per_jersey:
                    stats_per_jersey[jersey_no] += 1
                else:
                    stats_per_jersey["Other"] += 1

        # Gambar UI Dashboard
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (350, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, "LIVE PASS COUNT", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Player #9  : {stats_per_jersey['9']}", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player #15 : {stats_per_jersey['15']}", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player #30 : {stats_per_jersey['30']}", (40, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Gambar Trackers (Ellipse & ID)
        for track_id, player in player_dict.items():
            # Tampilkan nomor punggung di atas kepala jika terdeteksi
            jersey_no = player_identifier.player_numbers_map.get(track_id, "???")
            color = (0, 255, 0) if jersey_no != "???" else (255, 255, 255)
            
            frame = tracker.draw_ellipse(frame, player["bbox"], color)
            cv2.putText(frame, f"#{jersey_no}", (int(player["bbox"][0]), int(player["bbox"][1]-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if raw_ball_possessions[frame_num] == track_id:
                frame = tracker.draw_traingle(frame, player["bbox"], (0, 0, 255))

        # Ball
        ball_dict = tracks["ball"][frame_num]
        for _, ball in ball_dict.items():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        out.write(frame)

    out.release()
    print(f"Selesai! Video disimpan di: {output_path}")

if __name__ == '__main__':
    main()