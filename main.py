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
    video_path = 'input_videos/passing_number.mp4'
    stub_path = 'stubs/track_stubs.pkl'
    output_path = 'output_videos/passing_number.avi'

    tracker = Tracker('yolov8s.pt')
    player_ball_assigner = PlayerBallAssigner()
    player_identifier = PlayerIdentifier()

    # === Statistik passing per nomor punggung ===
    stats_per_jersey = {"3": 0, "19": 0, "Unknown": 0}

    # === STEP 1: Get Tracks ===
    video_frames = read_video(video_path)
    if os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
    else:
        tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path=stub_path)

    # === STEP 1.5: Interpolasi Bola ===
    tracks['ball'] = interpolate_ball_positions(tracks['ball'])

    # === STEP 2: Possession & OCR Mapping ===
    print("Processing identities and possession...")
    raw_ball_possessions = []

    for frame_num, frame_img in enumerate(video_frames):
        player_track = tracks['players'][frame_num]
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])

        # Identifikasi nomor punggung (dengan voting system)
        player_identifier.update_identities(frame_img, player_track)

        if len(ball_bbox) == 0:
            raw_ball_possessions.append(-1)
        else:
            assigned_player, _ = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
            raw_ball_possessions.append(assigned_player)

    # Debug: Print mapping hasil OCR
    # print("\n=== JERSEY NUMBER MAPPING ===")
    # for tid, jnum in player_identifier.player_numbers_map.items():
    #     history = player_identifier.detection_history.get(tid, {})
    #     print(f"  Track ID {tid} => #{jnum}  (votes: {history})")
    # print("=============================\n")

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

        # === Hitung pass berdasarkan jersey number ===
        for pass_event in detected_passes:
            if pass_event['frame_display'] == frame_num:
                sender_id = pass_event['from_player']
                jersey_no = player_identifier.get_jersey_number_for_player(sender_id)

                if jersey_no in stats_per_jersey:
                    stats_per_jersey[jersey_no] += 1
                else:
                    stats_per_jersey["Unknown"] += 1

                # Debug log
                receiver_id = pass_event['to_player']
                receiver_jersey = player_identifier.get_jersey_number_for_player(receiver_id)
                print(f"[PASS] Frame {frame_num}: #{jersey_no} (ID:{sender_id}) -> #{receiver_jersey} (ID:{receiver_id})")

        # === UI DASHBOARD ===
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (380, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "LIVE PASS COUNT", (40, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
        cv2.line(frame, (40, 65), (340, 65), (0, 255, 255), 1)

        total_passes = sum(stats_per_jersey.values())
        cv2.putText(frame, f"Player #3  : {stats_per_jersey['3']}",
                    (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player #19 : {stats_per_jersey['19']}",
                    (40, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Other      : {stats_per_jersey['Unknown']}",
                    (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        # === Gambar Player Trackers ===
        for track_id, player in player_dict.items():
            jersey_no = player_identifier.get_jersey_number_for_player(track_id)

            # Warna berdasarkan identifikasi
            if jersey_no == "3":
                color = (0, 255, 0)       # Hijau
            elif jersey_no == "19":
                color = (255, 165, 0)     # Oranye
            else:
                color = (200, 200, 200)   # Abu-abu untuk Unknown

            frame = tracker.draw_ellipse(frame, player["bbox"], color, track_id)

            # Label nomor punggung di atas kepala
            label = f"#{jersey_no}" if jersey_no != "Unknown" else f"?{track_id}"
            cv2.putText(frame, label,
                        (int(player["bbox"][0]), int(player["bbox"][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Segitiga possession
            if raw_ball_possessions[frame_num] == track_id:
                frame = tracker.draw_triangle(frame, player["bbox"], (0, 0, 255))

        # === Ball ===
        ball_dict = tracks["ball"][frame_num]
        for _, ball in ball_dict.items():
            frame = tracker.draw_triangle(frame, ball["bbox"], (0, 255, 0))

        out.write(frame)

    out.release()

    # === Final Summary ===
    print("\n" + "=" * 40)
    print("  PASSING STATISTICS SUMMARY")
    print("=" * 40)
    print(f"  Player #3  : {stats_per_jersey['3']} passes")
    print(f"  Player #19 : {stats_per_jersey['19']} passes")
    print(f"  Unknown    : {stats_per_jersey['Unknown']} passes")
    print(f"  TOTAL      : {sum(stats_per_jersey.values())} passes")
    print("=" * 40)
    print(f"\nVideo saved: {output_path}")


if __name__ == '__main__':
    main()
