from trackers import Tracker
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.pass_detector import PassDetector
from utils.bbox_utils import interpolate_ball_positions
import numpy as np
import cv2
import pickle
import os

def main():
    video_path = 'input_videos/passing.mp4'
    stub_path = 'stubs/track_stubs.pkl'
    output_path = 'output_videos/passing.avi'

    tracker = Tracker('models/best.pt')
    player_ball_assigner = PlayerBallAssigner()

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

    # ===== STEP 1.5: INTERPOLASI BOLA (BARU!) =====
    print("Interpolating missing ball positions...")
    original_ball_count = sum(1 for b in tracks['ball'] if 1 in b)
    tracks['ball'] = interpolate_ball_positions(tracks['ball'])
    interpolated_ball_count = sum(1 for b in tracks['ball'] if 1 in b)
    print(f"   Ball detections: {original_ball_count} -> {interpolated_ball_count} frames (interpolated {interpolated_ball_count - original_ball_count})")

    # ===== STEP 2: Team Assignment =====
    print("Assigning all players to Training Group...")
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            tracks['players'][frame_num][player_id]['team'] = 1
            tracks['players'][frame_num][player_id]['team_color'] = (255, 255, 255)
    print("   Team assignment complete")

    # ===== STEP 3: Ball Possession (DITINGKATKAN) =====
    print("Detecting ball possession...")
    
    # Dapatkan FPS video untuk konfigurasi PassDetector
    cap_temp = cv2.VideoCapture(video_path)
    fps = int(cap_temp.get(cv2.CAP_PROP_FPS))
    cap_temp.release()
    print(f"   Video FPS: {fps}")

    raw_ball_possessions = []
    possession_distances = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])

        if len(ball_bbox) == 0:
            raw_ball_possessions.append(-1)
            possession_distances.append(99999)
        else:
            assigned_player, distance = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
            raw_ball_possessions.append(assigned_player)
            possession_distances.append(distance)

    print(f"   Processed {len(raw_ball_possessions)} frames")
    valid_possessions = sum(1 for p in raw_ball_possessions if p != -1)
    print(f"   Valid possessions: {valid_possessions}/{len(raw_ball_possessions)} frames")

    # ===== STEP 4: Pass Detection (PIPELINE BARU) =====
    print("Detecting passes with improved pipeline...")
    
    pass_detector = PassDetector(fps=fps)
    passes = pass_detector.detect_passes(tracks, raw_ball_possessions)
    pass_stats = pass_detector.get_pass_statistics(passes)

    print(f"   Detected {pass_stats['total_passes']} total passes")
    if pass_stats['total_passes'] > 0:
        print(f"   Average pass distance: {pass_stats['avg_distance']:.1f} px")
        print(f"   Average ball movement: {pass_stats['avg_ball_movement']:.1f} px")

    # Print detail setiap pass untuk debugging
    for idx, p in enumerate(passes):
        print(f"   Pass #{idx+1}: Player {p['from_player']} -> Player {p['to_player']} "
              f"(frame {p['frame_start']}-{p['frame_end']}, "
              f"dist={p['distance']:.0f}px, ball_move={p['ball_movement']:.0f}px)")

    # ===== STEP 5: Render Video =====
    print("Rendering video with pass annotations...")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    active_passes = []
    current_pass_count = 0

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= len(tracks['players']):
            break

        player_dict = tracks["players"][frame_num]
        ball_dict = tracks["ball"][frame_num]

        # Draw Players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (255, 255, 255))
            frame = tracker.draw_ellipse(frame, player["bbox"], color, track_id)

            if raw_ball_possessions[frame_num] == track_id:
                frame = tracker.draw_traingle(frame, player["bbox"], (0, 0, 255))

        # Draw ball
        for track_id, ball in ball_dict.items():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        # Cek pass yang selesai di frame ini
        for pass_event in passes:
            if pass_event['frame_end'] == frame_num:
                current_pass_count += 1
                active_passes.append({
                    'pass': pass_event,
                    'frames_remaining': int(fps * 1.2)  # Tampilkan arrow selama 1.2 detik
                })

        # Draw active pass arrows
        passes_to_remove = []
        for i, active_pass in enumerate(active_passes):
            if active_pass['frames_remaining'] > 0:
                frame = tracker.draw_pass_arrow(frame, active_pass['pass'])
                active_pass['frames_remaining'] -= 1
            else:
                passes_to_remove.append(i)

        for i in reversed(passes_to_remove):
            active_passes.pop(i)

        # Draw statistics
        frame = tracker.draw_pass_statistics(frame, current_pass_count)

        out.write(frame)
        frame_num += 1

        if frame_num % 100 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"   Progress: {frame_num}/{total_frames} ({progress:.1f}%)")

    cap.release()
    out.release()

    print(f"\nVideo saved to {output_path}")
    print(f"Final stats: {current_pass_count} passes detected")

if __name__ == '__main__':
    main()