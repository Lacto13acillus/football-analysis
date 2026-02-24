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
        print("[STEP 1] Loading tracks from stub...")
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
        print(f"   Loaded {len(tracks['players'])} frames")
    else:
        print("[STEP 1] Running detection (no stub found)...")
        from utils.video_utils import read_video
        video_frames = read_video(video_path)
        print(f"   {len(video_frames)} frames loaded")
        tracks = tracker.get_object_track(video_frames,
                                           read_from_stub=False,
                                           stub_path=stub_path)
        del video_frames

    # ===== STEP 1.5: Interpolasi Bola =====
    print("\n[STEP 1.5] Interpolating ball...")
    original_ball_count = sum(1 for b in tracks['ball'] if 1 in b)
    tracks['ball'] = interpolate_ball_positions(tracks['ball'])
    new_ball_count = sum(1 for b in tracks['ball'] if 1 in b)
    print(f"   Ball: {original_ball_count} -> {new_ball_count} frames (+{new_ball_count - original_ball_count} interpolated)")
    
    if new_ball_count == 0:
        print("   *** FATAL: Ball NEVER detected! Check your YOLO model. ***")
        return

    # ===== STEP 2: Skip Team Assignment (latihan = 1 tim) =====
    print("\n[STEP 2] Assigning team (all = team 1)...")
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            tracks['players'][frame_num][player_id]['team'] = 1
            tracks['players'][frame_num][player_id]['team_color'] = (255, 255, 255)

    # ===== STEP 3: Ball Possession =====
    print("\n[STEP 3] Detecting ball possession...")
    
    cap_temp = cv2.VideoCapture(video_path)
    fps = int(cap_temp.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 24
    cap_temp.release()
    print(f"   FPS: {fps}")

    raw_ball_possessions = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])

        if len(ball_bbox) == 0:
            raw_ball_possessions.append(-1)
        else:
            assigned_player, distance = player_ball_assigner.assign_ball_to_player(
                player_track, ball_bbox
            )
            raw_ball_possessions.append(assigned_player)

    # === DEBUG: Analisis possession ===
    valid_count = sum(1 for p in raw_ball_possessions if p != -1)
    unique_players = set(p for p in raw_ball_possessions if p != -1)
    print(f"   Possession valid: {valid_count}/{len(raw_ball_possessions)} frames")
    print(f"   Unique players with ball: {unique_players}")
    
    if valid_count == 0:
        print("   *** FATAL: No player ever possesses the ball! ***")
        print("   *** Try increasing max_player_ball_distance (currently 70) ***")
        
        # Debug: cek jarak terdekat bola-pemain di beberapa frame
        print("\n   Checking closest player-ball distances in sample frames:")
        sample_frames = list(range(0, len(raw_ball_possessions), max(1, len(raw_ball_possessions) // 10)))
        for sf in sample_frames[:10]:
            ball_data = tracks['ball'][sf].get(1)
            if ball_data and 'bbox' in ball_data:
                from utils.bbox_utils import get_center_of_bbox
                ball_pos = get_center_of_bbox(ball_data['bbox'])
                min_dist = 99999
                closest_player = -1
                for pid, pdata in tracks['players'][sf].items():
                    pbbox = pdata['bbox']
                    from utils.bbox_utils import measure_distance
                    d = measure_distance((pbbox[0], pbbox[-1]), ball_pos)
                    if d < min_dist:
                        min_dist = d
                        closest_player = pid
                print(f"   Frame {sf}: closest player={closest_player}, distance={min_dist:.1f}px")
        return

    # ===== STEP 4: Pass Detection =====
    print("\n[STEP 4] Detecting passes...")
    
    pass_detector = PassDetector(fps=fps)
    passes = pass_detector.detect_passes(tracks, raw_ball_possessions, debug=True)
    pass_stats = pass_detector.get_pass_statistics(passes)

    print(f"\n   TOTAL PASSES: {pass_stats['total_passes']}")

    # ===== STEP 5: Render =====
    print("\n[STEP 5] Rendering video...")

    cap = cv2.VideoCapture(video_path)
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
            
            # Segitiga merah di atas pemain yang pegang bola
            if raw_ball_possessions[frame_num] == track_id:
                frame = tracker.draw_traingle(frame, player["bbox"], (0, 0, 255))

        # Draw ball
        for track_id, ball in ball_dict.items():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        # Pass events
        for pass_event in passes:
            if pass_event['frame_end'] == frame_num:
                current_pass_count += 1
                active_passes.append({
                    'pass': pass_event,
                    'frames_remaining': int(fps * 1.2)
                })

        # Draw arrows
        new_active = []
        for ap in active_passes:
            if ap['frames_remaining'] > 0:
                frame = tracker.draw_pass_arrow(frame, ap['pass'])
                ap['frames_remaining'] -= 1
                new_active.append(ap)
        active_passes = new_active

        # Draw stats
        frame = tracker.draw_pass_statistics(frame, current_pass_count)

        out.write(frame)
        frame_num += 1

        if frame_num % 200 == 0:
            print(f"   Rendering: {frame_num}/{total_frames}")

    cap.release()
    out.release()
    print(f"\nDone! Output: {output_path}")
    print(f"Total passes: {current_pass_count}")

if __name__ == '__main__':
    main()