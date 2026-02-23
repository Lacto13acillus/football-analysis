from trackers import Tracker
from trackers.pass_detector import PassDetector
from team_assigner import TeamAssigner
import numpy as np
import cv2
import pickle
import os

def main():
    video_path = 'input_videos/video_ssb_1.mp4'
    stub_path = 'stubs/track_stubs.pkl'
    output_path = 'output_videos/video_ssb_output_pass.avi'
    
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
    
    # ✅ Find best frame: many players, few referees
    best_frame_idx = None
    best_frame = None
    best_score = 0
    
    cap = cv2.VideoCapture(video_path)
    
    for frame_idx in range(len(tracks['players'])):
        num_players = len(tracks['players'][frame_idx])
        num_referees = len(tracks['referees'][frame_idx])
        
        # Score: prioritize frames with many players and few referees
        score = num_players - (num_referees * 3)  # Heavy penalty for referees
        
        if num_players >= 4 and score > best_score:  # Minimum 4 players
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
    
    # Assign team colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(best_frame, tracks['players'][best_frame_idx])
    
    print(f"   Team 1 color (BGR): {team_assigner.team_colors[1]}")
    print(f"   Team 2 color (BGR): {team_assigner.team_colors[2]}")
    
    # ✅ Check if any team is yellow-like (referee might be in clustering)
    def is_yellow_like(color):
        b, g, r = int(color[0]), int(color[1]), int(color[2])
        return (g > 180 and r > 180 and b < 150)  # Yellow: high green+red, low blue
    
    team_1_is_yellow = is_yellow_like(team_assigner.team_colors[1])
    team_2_is_yellow = is_yellow_like(team_assigner.team_colors[2])
    
    if team_1_is_yellow or team_2_is_yellow:
        print("   ⚠️  Warning: One team detected as yellow (referee might be in clustering)")
        print("   💡 Tip: Try using a different frame with more players and fewer referees")
    
    # Assign team to each player in each frame
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(best_frame,
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    del best_frame  # Free memory
    print("✅ Team assignment complete")
    
    # ===== STEP 3: Pass Detection =====
    print("⚽ Detecting passes...")
    
    # Initialize pass detector
    pass_detector = PassDetector(min_pass_distance=50, ball_possession_frames=5)
    
    # Get video FPS for accurate timing
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Detect passes
    passes = pass_detector.detect_passes(tracks, fps)
    
    # Calculate pass statistics
    pass_stats = pass_detector.get_pass_statistics()
    
    # Calculate possession statistics
    possession_stats = pass_detector.get_team_possession_stats(tracks)
    
    # Print pass summary
    print("\n" + "="*50)
    print("PASS DETECTION SUMMARY")
    print("="*50)
    print(f"Total passes detected: {pass_stats['total_passes']}")
    print(f"Team 1 passes: {pass_stats['passes_by_team'][1]}")
    print(f"Team 2 passes: {pass_stats['passes_by_team'][2]}")
    print(f"Average pass distance: {pass_stats['average_pass_distance']:.1f} pixels")
    
    if pass_stats['longest_pass']:
        lp = pass_stats['longest_pass']
        print(f"Longest pass: Player {lp['from_player']} → Player {lp['to_player']} "
              f"({lp['distance']:.1f}px at {lp['time']:.1f}s)")
    
    if pass_stats['shortest_pass']:
        sp = pass_stats['shortest_pass']
        print(f"Shortest pass: Player {sp['from_player']} → Player {sp['to_player']} "
              f"({sp['distance']:.1f}px)")
    
    print("\nPossession Statistics:")
    for team in [1, 2]:
        print(f"  Team {team}: {possession_stats[team]['percentage']:.1f}% "
              f"({possession_stats[team]['frames']} frames)")
    
    print("\nTop Passers:")
    for player_id, passes_count in pass_stats.get('top_passers', []):
        team = None
        # Find player's team from first frame
        for frame_data in tracks['players']:
            if player_id in frame_data and 'team' in frame_data[player_id]:
                team = frame_data[player_id]['team']
                break
        team_str = f"Team {team}" if team else "Unknown"
        print(f"  Player {player_id} ({team_str}): {passes_count} passes")
    
    print("="*50 + "\n")
    
    # Save pass data to file
    pass_data_path = 'stubs/pass_data.pkl'
    with open(pass_data_path, 'wb') as f:
        pickle.dump({
            'passes': passes,
            'stats': pass_stats,
            'possession': possession_stats
        }, f)
    print(f"✅ Pass data saved to {pass_data_path}")

    # ===== STEP 4: Render Video with Annotations =====
    print("🎬 Rendering annotated video...")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    last_pass_frame = -30  # Show pass connections for 30 frames
    
    # Dictionary to track active passes to display
    active_passes = []
    
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
        
        # Draw recent passes (last 30 frames)
        active_passes = [p for p in passes 
                        if abs(p['frame'] - frame_num) < 30]
        
        for pass_event in active_passes:
            if abs(pass_event['frame'] - frame_num) < 15:  # Show for 15 frames
                alpha = 1.0 - (abs(pass_event['frame'] - frame_num) / 15)
                if pass_event['frame'] <= frame_num:
                    # Pass already happened, show full line
                    frame = tracker.draw_pass_connection(
                        frame, 
                        pass_event['from_position'],
                        pass_event['to_position'],
                        pass_event['team']
                    )
                    
                    # Add pass label
                    mid_point = (
                        (pass_event['from_position'][0] + pass_event['to_position'][0]) // 2,
                        (pass_event['from_position'][1] + pass_event['to_position'][1]) // 2
                    )
                    cv2.putText(frame, "PASS!", 
                               (mid_point[0]-30, mid_point[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                               (255, 255, 255), 2)
        
        # Draw pass statistics on frame
        frame = tracker.draw_pass_statistics(frame, pass_stats, possession_stats)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_num}", 
                   (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL PASS ANALYSIS")
    print("="*50)
    print(f"Total passes: {pass_stats['total_passes']}")
    print(f"Pass completion rate by team:")
    # Note: This is approximate as we don't have attempted passes
    print(f"  Team 1: {pass_stats['passes_by_team'][1]} completed passes")
    print(f"  Team 2: {pass_stats['passes_by_team'][2]} completed passes")
    
    # Calculate passes per minute if video length is known
    video_length_seconds = total_frames / fps
    video_length_minutes = video_length_seconds / 60
    if video_length_minutes > 0:
        ppm = pass_stats['total_passes'] / video_length_minutes
        print(f"Passes per minute: {ppm:.1f}")
    
    print("="*50)

if __name__ == '__main__':
    main()