import os
import cv2
import pickle
import numpy as np
from trackers.tracker import Tracker
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.pass_detector import PassDetector
from team_assigner.player_identifier import PlayerIdentifier
from utils.bbox_utils import get_center_of_bbox_bottom, interpolate_ball_positions, bbox_area
from utils.video_utils import read_video

def main():
    video_path = "input_videos/passing_number.mp4"
    output_path = "output_videos/passing_number.avi"
    stub_path = "stubs/track_stubs.pkl"

    # Inisialisasi
    tracker = Tracker("yolov8m.pt")
    player_ball_assigner = PlayerBallAssigner(max_player_ball_distance=350)
    player_id_module = PlayerIdentifier()
    
    video_frames = read_video(video_path)
    if not video_frames: return

    # Load/Gat Tracks
    if os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            tracks = pickle.load(f)
    else:
        tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path=stub_path)

    tracks["ball"] = interpolate_ball_positions(tracks["ball"])

    # Proses Identifikasi Nomor Punggung (Sampling setiap 3 frame untuk efisiensi)
    print("Menganalisis Nomor Punggung 3 & 19...")
    for f in range(0, len(video_frames), 3):
        player_id_module.update_identities(video_frames[f], tracks["players"][f])

    # Deteksi Possession
    ball_possessions = []
    for f in range(len(video_frames)):
        frame_players = tracks["players"][f]
        ball_bbox = tracks["ball"][f].get(1, {}).get("bbox", [])
        
        if not ball_bbox:
            ball_possessions.append(-1)
            continue

        assigned_id, _ = player_ball_assigner.assign_ball_to_player(frame_players, ball_bbox)
        ball_possessions.append(assigned_id)

    # Deteksi Pass
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    pass_detector = PassDetector(fps=fps)
    detected_passes = pass_detector.detect_passes(tracks, ball_possessions)

    # Hitung Statistik Berdasarkan Nomor Jersey
    pass_stats = {"3": 0, "19": 0}
    
    # Mapping pass ke nomor jersey
    for ev in detected_passes:
        from_track_id = ev["from_player"]
        jersey_num = player_id_module.get_assigned_number(from_track_id)
        
        if jersey_num in pass_stats:
            pass_stats[jersey_num] += 1

    # Visualisasi & Simpan Video
    h, w = video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

    # Untuk animasi counter yang bertambah
    current_display_stats = {"3": 0, "19": 0}

    for f_idx, frame in enumerate(video_frames):
        # Update counter saat frame mencapai display_frame dari event pass
        for ev in detected_passes:
            if f_idx == ev["frame_display"]:
                j_num = player_id_module.get_assigned_number(ev["from_player"])
                if j_num in current_display_stats:
                    current_display_stats[j_num] += 1

        # Draw Overlay Statis
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (350, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, "PASS COUNTER", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Jersey #3  : {current_display_stats['3']} passes", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Jersey #19 : {current_display_stats['19']} passes", (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw Player Labels
        for tid, pdata in tracks["players"][f_idx].items():
            bbox = pdata["bbox"]
            j_num = player_id_module.get_assigned_number(tid)
            
            if j_num in ['3', '19']:
                color = (0, 255, 0) if j_num == '3' else (255, 0, 0)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, f"Player {j_num}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Pass Arrow
        for ev in detected_passes:
            if 0 <= f_idx - ev["frame_display"] <= 20:
                cv2.arrowedLine(frame, ev["from_pos"], ev["to_pos"], (0, 255, 0), 3)

        out.write(frame)

    out.release()
    print(f"Proses Selesai. Hasil: {pass_stats}")

if __name__ == "__main__":
    main()