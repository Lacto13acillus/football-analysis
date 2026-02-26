from trackers import Tracker
from trackers.player_ball_assigner import PlayerBallAssigner
from team_assigner.player_identifier import PlayerIdentifier
from utils.bbox_utils import interpolate_ball_positions
from utils.video_utils import read_video
import cv2
import os

def debug():
    video_path = 'input_videos/passing_number.mp4'
    stub_path = 'stubs/track_stubs.pkl'

    # === STEP 1: Read Video ===
    video_frames = read_video(video_path)
    print(f"[DEBUG] Total frames loaded: {len(video_frames)}")
    if len(video_frames) == 0:
        print("[FATAL] No frames loaded! Check video path.")
        return

    # === STEP 2: Get Tracks (FORCE re-detect, no stub) ===
    print("[DEBUG] Running YOLO detection... (this may take a while)")
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path=stub_path)

    # === STEP 3: Debug Player Tracks ===
    total_player_frames = 0
    all_track_ids = set()
    for frame_num in range(len(tracks['players'])):
        player_dict = tracks['players'][frame_num]
        if len(player_dict) > 0:
            total_player_frames += 1
            for tid in player_dict:
                all_track_ids.add(tid)

    print(f"\n{'='*50}")
    print(f"[DEBUG] PLAYER DETECTION RESULTS:")
    print(f"  Frames with players: {total_player_frames}/{len(video_frames)}")
    print(f"  Unique track IDs: {sorted(all_track_ids)}")
    print(f"  Total unique players: {len(all_track_ids)}")

    if total_player_frames == 0:
        print("[FATAL] NO PLAYERS DETECTED! Your YOLO model might not detect 'player' class.")
        print("[FATAL] Check model classes with: model.names")
        return

    # Print track_id appearances
    print(f"\n[DEBUG] TRACK ID FREQUENCY:")
    for tid in sorted(all_track_ids):
        count = sum(1 for f in tracks['players'] if tid in f)
        # Get a sample bbox
        for f in tracks['players']:
            if tid in f:
                bbox = [int(b) for b in f[tid]['bbox']]
                break
        print(f"  Track ID {tid}: appears in {count} frames, sample bbox={bbox}")

    # === STEP 4: Debug Ball Tracks ===
    total_ball_frames = 0
    for frame_num in range(len(tracks['ball'])):
        ball_dict = tracks['ball'][frame_num]
        if 1 in ball_dict:
            total_ball_frames += 1

    print(f"\n{'='*50}")
    print(f"[DEBUG] BALL DETECTION RESULTS:")
    print(f"  Frames with ball: {total_ball_frames}/{len(video_frames)} "
          f"({100*total_ball_frames/len(video_frames):.1f}%)")

    if total_ball_frames == 0:
        print("[FATAL] NO BALL DETECTED! Your YOLO model might not detect 'ball' class.")
        return

    # === STEP 5: Interpolate Ball ===
    tracks['ball'] = interpolate_ball_positions(tracks['ball'])
    total_ball_after = sum(1 for f in tracks['ball'] if 1 in f)
    print(f"  After interpolation: {total_ball_after}/{len(video_frames)} "
          f"({100*total_ball_after/len(video_frames):.1f}%)")

    # === STEP 6: Debug Ball Possession ===
    player_ball_assigner = PlayerBallAssigner()
    raw_possessions = []
    for frame_num in range(len(video_frames)):
        player_track = tracks['players'][frame_num]
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])

        if len(ball_bbox) == 0:
            raw_possessions.append(-1)
        else:
            assigned, dist = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
            raw_possessions.append(assigned)

    valid_possessions = sum(1 for p in raw_possessions if p != -1)
    unique_possessors = set(p for p in raw_possessions if p != -1)
    print(f"\n{'='*50}")
    print(f"[DEBUG] BALL POSSESSION RESULTS:")
    print(f"  Frames with possession: {valid_possessions}/{len(video_frames)} "
          f"({100*valid_possessions/len(video_frames):.1f}%)")
    print(f"  Unique possessors (track_ids): {sorted(unique_possessors)}")

    if valid_possessions == 0:
        print("[FATAL] NO BALL POSSESSION! max_player_ball_distance might be too small.")
        print("[HINT] Try increasing max_player_ball_distance in PlayerBallAssigner")
        # Show min distances for debugging
        print("\n[DEBUG] Checking distances between players and ball...")
        sample_count = 0
        for frame_num in range(0, len(video_frames), 30):  # Check every 30 frames
            player_track = tracks['players'][frame_num]
            ball_data = tracks['ball'][frame_num].get(1, {})
            if 'bbox' not in ball_data or len(player_track) == 0:
                continue
            ball_bbox = ball_data['bbox']
            from utils.bbox_utils import get_center_of_bbox
            ball_pos = get_center_of_bbox(ball_bbox)
            for tid, pdata in player_track.items():
                pbbox = pdata['bbox']
                left_foot = (pbbox[0], pbbox[3])
                right_foot = (pbbox[2], pbbox[3])
                from utils.bbox_utils import measure_distance
                d = min(measure_distance(left_foot, ball_pos),
                        measure_distance(right_foot, ball_pos))
                print(f"  Frame {frame_num}: Track {tid} -> ball distance = {d:.0f}px")
            sample_count += 1
            if sample_count >= 5:
                break
        return

    # === STEP 7: Debug OCR (Sample frames) ===
    print(f"\n{'='*50}")
    print(f"[DEBUG] JERSEY OCR TEST (sampling every 50 frames):")
    identifier = PlayerIdentifier()
    for frame_num in range(0, len(video_frames), 50):
        player_track = tracks['players'][frame_num]
        if len(player_track) == 0:
            continue
        identifier.update_identities(video_frames[frame_num], player_track)

    print(f"  Mapping result: {identifier.player_numbers_map}")
    print(f"  Vote history: {identifier.detection_history}")

    if not identifier.player_numbers_map or all(v == "Unknown" for v in identifier.player_numbers_map.values()):
        print("[WARNING] OCR failed to detect any jersey numbers!")
        print("[HINT] Consider using MANUAL MAPPING instead (see below)")

    # === FINAL SUMMARY ===
    print(f"\n{'='*50}")
    print(f"SUMMARY — WHAT TO DO NEXT:")
    print(f"{'='*50}")

    if len(unique_possessors) >= 2:
        print(f"✅ Pass detection SHOULD work ({len(unique_possessors)} players had possession)")
        print(f"   Possessor track IDs: {sorted(unique_possessors)}")
        print(f"\n   Use MANUAL MAPPING in player_identifier.py:")
        print(f"   self.manual_map = {{")
        for tid in sorted(unique_possessors):
            print(f"       {tid}: '??',  # <-- Look at output video to identify this player")
        print(f"   }}")
    else:
        print(f"❌ Need at least 2 players with possession, got {len(unique_possessors)}")


if __name__ == '__main__':
    debug()
