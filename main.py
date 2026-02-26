import os
import cv2
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from trackers.tracker import Tracker
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.pass_detector import PassDetector
from utils.bbox_utils import (
    bbox_area,
    get_center_of_bbox_bottom,
    measure_distance,
    interpolate_ball_positions,
)
from utils.video_utils import read_video
@dataclass
class CandidatePlayer:
    track_id: int
    bbox: List[float]
    area: float
    foot_pos: Tuple[int, int]
    x: int
def build_locked_roles(
    tracks: Dict[str, List[Dict[int, Dict[str, Any]]]],
    video_frames: List[np.ndarray],
    max_jump_px: int = 160,
    min_area_ratio: float = 0.0025,
    lost_tolerance: int = 12,
) -> Tuple[Dict[int, Dict[int, str]], List[Dict[int, int]]]:
    """
    Bulletproof temporal locking for exactly 3 foreground players (A/B/C).
    WHY this works better than "top-3 area per frame":
    - Background players sometimes become "top-3" when a foreground player turns/occludes (bbox shrinks).
    - Camera panning breaks ByteTrack IDs (new IDs appear), so relying on track_id continuity is fragile.
    - We instead lock 3 roles using *spatial continuity* of foot positions + hysteresis.
      This makes the system resistant to brief occlusion, bbox jitter, and background distractions.
    Returns:
      frame_roles: dict[frame_idx] -> dict[track_id] = "Player A/B/C" (for visualization)
      locked_ids_per_frame: list[frame_idx] -> {0: track_id_for_A, 1: track_id_for_B, 2: track_id_for_C}
        where roles are integers 0=A,1=B,2=C (for role-based possession and pass arrow resolution)
    """
    if not video_frames:
        raise ValueError("video_frames is empty. Check input video path.")
    if "players" not in tracks or len(tracks["players"]) != len(video_frames):
        raise ValueError("tracks['players'] must exist and have same length as video_frames.")
    height, width = video_frames[0].shape[:2]
    min_area = float(height * width) * float(min_area_ratio)
    ROLE_A, ROLE_B, ROLE_C = 0, 1, 2
    role_names = {ROLE_A: "Player A", ROLE_B: "Player B", ROLE_C: "Player C"}
    roles = [ROLE_A, ROLE_B, ROLE_C]
    # Role state stores the last known track and foot position for continuity.
    role_state: Dict[int, Dict[str, Any]] = {
        r: {"track_id": -1, "foot_pos": None, "lost": 10_000} for r in roles
    }
    frame_roles: Dict[int, Dict[int, str]] = {}
    locked_ids_per_frame: List[Dict[int, int]] = []
    def make_candidates(frame_players: Dict[int, Dict[str, Any]]) -> List[CandidatePlayer]:
        candidates: List[CandidatePlayer] = []
        for tid_raw, pdata in frame_players.items():
            tid = int(tid_raw)
            bbox = pdata.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            area = bbox_area(bbox)
            # Foreground gating: reject small detections that are very likely background.
            if area < min_area:
                continue
            foot = get_center_of_bbox_bottom(bbox)
            candidates.append(CandidatePlayer(track_id=tid, bbox=bbox, area=area, foot_pos=foot, x=foot[0]))
        return candidates
    def reinitialize_all_roles(candidates: List[CandidatePlayer]) -> bool:
        # Initialize by taking the three largest foreground candidates, then sorting by X (left->right).
        top3 = sorted(candidates, key=lambda c: c.area, reverse=True)[:3]
        if len(top3) < 3:
            return False
        top3 = sorted(top3, key=lambda c: c.x)
        for role, cand in zip(roles, top3):
            role_state[role]["track_id"] = cand.track_id
            role_state[role]["foot_pos"] = cand.foot_pos
            role_state[role]["lost"] = 0
        return True
    for f in range(len(video_frames)):
        players_f = tracks["players"][f]
        candidates = make_candidates(players_f)
        # If we have no candidates (e.g., detection glitch), mark lost and continue.
        if not candidates:
            for r in roles:
                role_state[r]["lost"] += 1
            frame_roles[f] = {}
            locked_ids_per_frame.append({ROLE_A: -1, ROLE_B: -1, ROLE_C: -1})
            continue
        # If never initialized or everything has been missing too long, reinitialize from scratch.
        never_initialized = all(role_state[r]["track_id"] == -1 for r in roles)
        all_lost_long = all(role_state[r]["lost"] > lost_tolerance for r in roles)
        if never_initialized or all_lost_long:
            ok = reinitialize_all_roles(candidates)
            if not ok:
                frame_roles[f] = {}
                locked_ids_per_frame.append({ROLE_A: -1, ROLE_B: -1, ROLE_C: -1})
                continue
        # Update each role by nearest foot position with a max jump threshold.
        # WHY: even if ByteTrack gives a new ID, the real player is usually close in image space.
        used_track_ids: Set[int] = set()
        for role in roles:
            prev_pos: Optional[Tuple[int, int]] = role_state[role]["foot_pos"]
            best_cand: Optional[CandidatePlayer] = None
            best_cost = float("inf")
            for cand in candidates:
                if cand.track_id in used_track_ids:
                    continue
                if prev_pos is None:
                    # If no previous position (should be rare), prefer larger area.
                    cost = -cand.area
                else:
                    d = measure_distance(prev_pos, cand.foot_pos)
                    if d > max_jump_px:
                        # Hard gate: prevents swapping to background or wrong athlete during jitter.
                        continue
                    # Prefer closer position; add tiny area preference to break ties toward foreground.
                    cost = d - 1e-5 * cand.area
                if cost < best_cost:
                    best_cost = cost
                    best_cand = cand
            if best_cand is None:
                role_state[role]["lost"] += 1
            else:
                role_state[role]["track_id"] = best_cand.track_id
                role_state[role]["foot_pos"] = best_cand.foot_pos
                role_state[role]["lost"] = 0
                used_track_ids.add(best_cand.track_id)
        # If any role has been lost too long, reinitialize all roles.
        # WHY: after hard panning, the three players may re-enter with new IDs far away.
        if any(role_state[r]["lost"] > lost_tolerance for r in roles):
            reinitialize_all_roles(candidates)
        # Build outputs for this frame.
        roles_dict_for_drawing: Dict[int, str] = {}
        locked_ids: Dict[int, int] = {ROLE_A: -1, ROLE_B: -1, ROLE_C: -1}
        for role in roles:
            tid = int(role_state[role]["track_id"])
            locked_ids[role] = tid
            if tid != -1:
                roles_dict_for_drawing[tid] = role_names[role]
        frame_roles[f] = roles_dict_for_drawing
        locked_ids_per_frame.append(locked_ids)
    return frame_roles, locked_ids_per_frame
def main() -> None:
    video_path = "input_videos/passing_drill.mp4"
    stub_path = "stubs/track_stubs.pkl"
    output_path = "output_videos/passing_drill_final.avi"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    tracker = Tracker("yolov8m.pt")
    player_ball_assigner = PlayerBallAssigner(max_player_ball_distance=70)
    video_frames = read_video(video_path)
    if not video_frames:
        raise RuntimeError("No frames read from video. The file may be corrupted or codec unsupported.")
    # Track objects (load stub if available).
    if os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            tracks = pickle.load(f)
    else:
        tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path=stub_path)
    if "ball" not in tracks:
        tracks["ball"] = [{} for _ in range(len(video_frames))]
    # Interpolate missing ball bboxes to reduce pass false negatives.
    tracks["ball"] = interpolate_ball_positions(tracks["ball"])
    # Lock A/B/C temporally.
    frame_roles, locked_ids_per_frame = build_locked_roles(tracks, video_frames)
    # Convert possession to role-based IDs: 0=A, 1=B, 2=C; -1 = unknown/no ball.
    role_based_possessions: List[int] = []
    role_to_trackid_per_frame: List[Dict[int, int]] = locked_ids_per_frame
    for frame_num in range(len(video_frames)):
        frame_players = tracks["players"][frame_num]
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox", [])
        if not ball_bbox:
            role_based_possessions.append(-1)
            continue
        locked_map = locked_ids_per_frame[frame_num]  # {0:tidA,1:tidB,2:tidC}
        locked_ids = {tid for tid in locked_map.values() if tid != -1}
        if not locked_ids:
            role_based_possessions.append(-1)
            continue
        # Filter to only the locked 3 players before assigning ball.
        filtered_players = {tid: pdata for tid, pdata in frame_players.items() if int(tid) in locked_ids}
        assigned_track_id, _ = player_ball_assigner.assign_ball_to_player(filtered_players, ball_bbox)
        if assigned_track_id == -1:
            role_based_possessions.append(-1)
            continue
        # Map track_id -> role integer using current frame locking.
        role_id = -1
        for r, tid in locked_map.items():
            if tid == assigned_track_id:
                role_id = int(r)
                break
        role_based_possessions.append(role_id)
    # Get FPS for time-based thresholds.
    cap_temp = cv2.VideoCapture(video_path)
    fps = float(cap_temp.get(cv2.CAP_PROP_FPS)) or 24.0
    cap_temp.release()
    pass_detector = PassDetector(fps=fps)
    allowed_role_ids = {0, 1, 2}
    detected_passes = pass_detector.detect_passes(
        tracks=tracks,
        ball_possessions=role_based_possessions,
        debug=False,
        allowed_player_ids=allowed_role_ids,
        role_to_trackid_per_frame=role_to_trackid_per_frame,
    )
    # Aggregate stats by role name.
    role_name = {0: "Player A", 1: "Player B", 2: "Player C"}
    stats_per_player = {"Player A": 0, "Player B": 0, "Player C": 0}
    for ev in detected_passes:
        sender_role = int(ev["from_player"])
        stats_per_player[role_name.get(sender_role, "Player A")] += 1  # safe fallback
    # Prepare writer.
    h, w = video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), float(fps), (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
    # For on-video display: progressive counts as the video plays.
    display_stats = {"Player A": 0, "Player B": 0, "Player C": 0}
    # Speed-up: create index of pass events by display frame.
    events_by_display_frame: Dict[int, List[Dict[str, Any]]] = {}
    for ev in detected_passes:
        fd = int(ev["frame_display"])
        events_by_display_frame.setdefault(fd, []).append(ev)
    for frame_num, frame in enumerate(video_frames):
        player_dict = tracks["players"][frame_num]
        locked_map = locked_ids_per_frame[frame_num]  # {0:tidA,1:tidB,2:tidC}
        # Update progressive stats when a pass is displayed.
        for ev in events_by_display_frame.get(frame_num, []):
            sender_role = int(ev["from_player"])
            nm = role_name.get(sender_role, None)
            if nm in display_stats:
                display_stats[nm] += 1
        # Draw recent pass arrows for a short duration.
        for ev in detected_passes:
            if 0 <= frame_num - int(ev["frame_display"]) <= 15:
                frame = tracker.draw_pass_arrow(frame, ev)
        # Draw overlay stats box.
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (320, 190), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "PASSING STATS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Player A : {display_stats['Player A']}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player B : {display_stats['Player B']}", (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player C : {display_stats['Player C']}", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Draw locked foreground players only, with role labels.
        # WHY: visualization should match exactly what the pass system "believes".
        for role_id, tid in locked_map.items():
            if tid == -1:
                continue
            pdata = player_dict.get(tid, None)
            if pdata is None:
                continue
            bbox = pdata.get("bbox", None)
            if not bbox:
                continue
            nm = role_name.get(int(role_id), "Player ?")
            color = (0, 255, 0)
            frame = tracker.draw_ellipse(frame, bbox, color)
            cv2.putText(
                frame,
                nm,
                (int(bbox[0]), max(0, int(bbox[1]) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            # Highlight if current frame possession belongs to this role.
            if role_based_possessions[frame_num] == int(role_id):
                frame = tracker.draw_triangle(frame, bbox, (0, 0, 255))
        # Draw ball.
        ball_dict = tracks["ball"][frame_num]
        for _, ball in ball_dict.items():
            bb = ball.get("bbox", None)
            if bb:
                frame = tracker.draw_triangle(frame, bb, (0, 255, 0))
        out.write(frame)
    out.release()
    print(f"Done. Output saved to: {output_path}")
    print("Final pass stats:", stats_per_player)
if __name__ == "__main__":
    main()