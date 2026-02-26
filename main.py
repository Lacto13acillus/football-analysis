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


# ──────────────────────────────────────────────────────────────
# Data class for cleaner candidate handling
# ──────────────────────────────────────────────────────────────
@dataclass
class CandidatePlayer:
    track_id: int
    bbox: List[float]
    area: float
    foot_pos: Tuple[int, int]
    x: int


# ──────────────────────────────────────────────────────────────
# STICKY ID LOCKING  (replaces old spatial-only build_locked_roles)
# ──────────────────────────────────────────────────────────────
def build_locked_roles(
    tracks: Dict[str, List[Dict[int, Dict[str, Any]]]],
    video_frames: List[np.ndarray],
    max_jump_px: int = 180,
    min_area_ratio: float = 0.0025,
    lost_tolerance: int = 15,
) -> Tuple[Dict[int, Dict[int, str]], List[Dict[int, int]]]:
    """
    Sticky ID Locking: uses ByteTrack track_id as PRIMARY key for role persistence.

    Strategy:
    1. On first frame with 3+ foreground players, sort by X -> lock track_ids to roles.
    2. On subsequent frames, look up each locked track_id directly in the detection dict.
       - If found  -> role stays, update foot position.
       - If NOT found (ByteTrack lost it) -> fallback to spatial nearest match.
    3. If ALL roles are lost for too long -> full reinitialization.

    This prevents identity swaps during crossing because ByteTrack's internal
    Kalman filter + Hungarian matching keeps IDs stable through brief occlusions.

    Returns:
      frame_roles:          dict[frame_idx] -> dict[track_id] = "Player A/B/C"
      locked_ids_per_frame: list[frame_idx] -> {0: tidA, 1: tidB, 2: tidC}
    """
    if not video_frames:
        raise ValueError("video_frames is empty. Check input video path.")
    if "players" not in tracks or len(tracks["players"]) != len(video_frames):
        raise ValueError("tracks['players'] must exist and match video_frames length.")

    height, width = video_frames[0].shape[:2]
    min_area = float(height * width) * float(min_area_ratio)

    ROLE_A, ROLE_B, ROLE_C = 0, 1, 2
    role_names = {ROLE_A: "Player A", ROLE_B: "Player B", ROLE_C: "Player C"}
    roles = [ROLE_A, ROLE_B, ROLE_C]

    # ── Role state ──
    role_state: Dict[int, Dict[str, Any]] = {
        r: {"track_id": -1, "foot_pos": None, "lost_count": 10_000} for r in roles
    }
    initialized = False

    frame_roles: Dict[int, Dict[int, str]] = {}
    locked_ids_per_frame: List[Dict[int, int]] = []

    # ── Helpers ──
    def get_foreground_candidates(
        frame_players: Dict[int, Dict[str, Any]],
    ) -> List[CandidatePlayer]:
        candidates: List[CandidatePlayer] = []
        for tid_raw, pdata in frame_players.items():
            tid = int(tid_raw)
            bbox = pdata.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            area = bbox_area(bbox)
            if area < min_area:
                continue
            foot = get_center_of_bbox_bottom(bbox)
            candidates.append(
                CandidatePlayer(track_id=tid, bbox=bbox, area=area, foot_pos=foot, x=foot[0])
            )
        return candidates

    def initialize_roles(candidates: List[CandidatePlayer]) -> bool:
        """Pick 3 largest, sort by X (left->right), lock their track_ids."""
        top3 = sorted(candidates, key=lambda c: c.area, reverse=True)[:3]
        if len(top3) < 3:
            return False
        top3 = sorted(top3, key=lambda c: c.x)
        for role, cand in zip(roles, top3):
            role_state[role]["track_id"] = cand.track_id
            role_state[role]["foot_pos"] = cand.foot_pos
            role_state[role]["lost_count"] = 0
        return True

    def find_spatial_fallback(
        candidates: List[CandidatePlayer],
        prev_foot: Tuple[int, int],
        used_ids: Set[int],
    ) -> Optional[CandidatePlayer]:
        """When a track_id disappears, find the nearest unassigned candidate."""
        best: Optional[CandidatePlayer] = None
        best_dist = float("inf")
        for cand in candidates:
            if cand.track_id in used_ids:
                continue
            d = measure_distance(prev_foot, cand.foot_pos)
            if d < best_dist and d <= max_jump_px:
                best_dist = d
                best = cand
        return best

    # ── Main frame loop ──
    for f in range(len(video_frames)):
        players_f = tracks["players"][f]
        candidates = get_foreground_candidates(players_f)

        # No candidates at all
        if not candidates:
            for r in roles:
                role_state[r]["lost_count"] += 1
            frame_roles[f] = {}
            locked_ids_per_frame.append({ROLE_A: -1, ROLE_B: -1, ROLE_C: -1})
            continue

        # ── Initialization gate ──
        all_lost = all(role_state[r]["lost_count"] > lost_tolerance for r in roles)
        if not initialized or all_lost:
            ok = initialize_roles(candidates)
            if ok:
                initialized = True
            else:
                frame_roles[f] = {}
                locked_ids_per_frame.append({ROLE_A: -1, ROLE_B: -1, ROLE_C: -1})
                continue

        # ── Sticky ID Matching ──
        # Build quick lookup: track_id -> CandidatePlayer
        cand_by_tid: Dict[int, CandidatePlayer] = {c.track_id: c for c in candidates}
        used_ids: Set[int] = set()

        for role in roles:
            locked_tid = role_state[role]["track_id"]

            # PRIMARY: check if locked track_id still exists in this frame
            if locked_tid != -1 and locked_tid in cand_by_tid and locked_tid not in used_ids:
                cand = cand_by_tid[locked_tid]
                role_state[role]["track_id"] = cand.track_id
                role_state[role]["foot_pos"] = cand.foot_pos
                role_state[role]["lost_count"] = 0
                used_ids.add(cand.track_id)

            # FALLBACK: track_id gone -> spatial nearest
            else:
                prev_pos = role_state[role]["foot_pos"]
                if prev_pos is not None:
                    fallback = find_spatial_fallback(candidates, prev_pos, used_ids)
                    if fallback is not None:
                        role_state[role]["track_id"] = fallback.track_id
                        role_state[role]["foot_pos"] = fallback.foot_pos
                        role_state[role]["lost_count"] = 0
                        used_ids.add(fallback.track_id)
                    else:
                        role_state[role]["lost_count"] += 1
                else:
                    role_state[role]["lost_count"] += 1

        # ── If any single role lost too long, try full reinit ──
        if any(role_state[r]["lost_count"] > lost_tolerance for r in roles):
            ok = initialize_roles(candidates)
            if ok:
                used_ids = {role_state[r]["track_id"] for r in roles}

        # ── Build output for this frame ──
        roles_dict: Dict[int, str] = {}
        locked_ids: Dict[int, int] = {}
        for role in roles:
            tid = int(role_state[role]["track_id"])
            locked_ids[role] = tid
            if tid != -1:
                roles_dict[tid] = role_names[role]
        frame_roles[f] = roles_dict
        locked_ids_per_frame.append(locked_ids)

    return frame_roles, locked_ids_per_frame


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main() -> None:
    video_path = "input_videos/passing_close.mp4"
    stub_path = "stubs/track_stubs.pkl"
    output_path = "output_videos/passing_close.avi"

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)

    # ── Initialize components ──
    tracker = Tracker("yolov8m.pt")
    player_ball_assigner = PlayerBallAssigner(max_player_ball_distance=150)  # CHANGED: 70 -> 150

    # ── Read video ──
    video_frames = read_video(video_path)
    if not video_frames:
        raise RuntimeError("No frames read from video.")

    # ── Track objects ──
    if os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            tracks = pickle.load(f)
    else:
        tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path=stub_path)

    if "ball" not in tracks:
        tracks["ball"] = [{} for _ in range(len(video_frames))]

    # ── Interpolate missing ball bboxes ──
    tracks["ball"] = interpolate_ball_positions(tracks["ball"])

    # ── Lock A/B/C with Sticky ID Locking ──
    frame_roles, locked_ids_per_frame = build_locked_roles(tracks, video_frames)

    # ── Build role-based possession array ──
    role_based_possessions: List[int] = []
    role_to_trackid_per_frame: List[Dict[int, int]] = locked_ids_per_frame
    role_name = {0: "Player A", 1: "Player B", 2: "Player C"}

    for frame_num in range(len(video_frames)):
        frame_players = tracks["players"][frame_num]
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox", [])

        if not ball_bbox:
            role_based_possessions.append(-1)
            continue

        locked_map = locked_ids_per_frame[frame_num]
        locked_ids = {tid for tid in locked_map.values() if tid != -1}

        if not locked_ids:
            role_based_possessions.append(-1)
            continue

        # Filter to only the locked 3 players before assigning ball
        filtered_players = {
            tid: pdata for tid, pdata in frame_players.items() if int(tid) in locked_ids
        }
        assigned_track_id, _ = player_ball_assigner.assign_ball_to_player(filtered_players, ball_bbox)

        if assigned_track_id == -1:
            role_based_possessions.append(-1)
            continue

        # Map track_id -> role integer
        role_id = -1
        for r, tid in locked_map.items():
            if tid == assigned_track_id:
                role_id = int(r)
                break
        role_based_possessions.append(role_id)

    # ── Get FPS ──
    cap_temp = cv2.VideoCapture(video_path)
    fps = float(cap_temp.get(cv2.CAP_PROP_FPS)) or 24.0
    cap_temp.release()

    # ── Detect passes ──
    pass_detector = PassDetector(fps=fps)
    allowed_role_ids = {0, 1, 2}
    detected_passes = pass_detector.detect_passes(
        tracks=tracks,
        ball_possessions=role_based_possessions,
        debug=False,
        allowed_player_ids=allowed_role_ids,
        role_to_trackid_per_frame=role_to_trackid_per_frame,
    )

    # ── DEBUG: Print possession timeline & detected passes ──
    print(f"\n{'='*60}")
    print(f"Total frames: {len(video_frames)}, FPS: {fps:.1f}")
    print(f"Detected passes: {len(detected_passes)}")
    for i, ev in enumerate(detected_passes):
        rn = role_name.get
        print(
            f"  Pass #{i+1}: {rn(ev['from_player'],'?')} -> {rn(ev['to_player'],'?')} "
            f"| frame {ev['frame_start']}-{ev['frame_end']} "
            f"| display@{ev['frame_display']}"
        )

    print(f"\nPossession changes (role-based):")
    prev = -1
    for f, p in enumerate(role_based_possessions):
        if p != prev:
            print(f"  Frame {f:>5d}: {role_name.get(prev, 'None'):>10s} -> {role_name.get(p, 'None')}")
            prev = p
    print(f"{'='*60}\n")

    # ── Aggregate stats ──
    stats_per_player = {"Player A": 0, "Player B": 0, "Player C": 0}
    for ev in detected_passes:
        sender_role = int(ev["from_player"])
        stats_per_player[role_name.get(sender_role, "Player A")] += 1

    # ── Prepare video writer ──
    h, w = video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), float(fps), (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    # Progressive display counts
    display_stats = {"Player A": 0, "Player B": 0, "Player C": 0}

    # Index pass events by display frame for O(1) lookup
    events_by_display_frame: Dict[int, List[Dict[str, Any]]] = {}
    for ev in detected_passes:
        fd = int(ev["frame_display"])
        events_by_display_frame.setdefault(fd, []).append(ev)

    # ── Render loop ──
    for frame_num, frame in enumerate(video_frames):
        player_dict = tracks["players"][frame_num]
        locked_map = locked_ids_per_frame[frame_num]

        # Update progressive stats
        for ev in events_by_display_frame.get(frame_num, []):
            sender_role = int(ev["from_player"])
            nm = role_name.get(sender_role, None)
            if nm in display_stats:
                display_stats[nm] += 1

        # Draw recent pass arrows (visible for 15 frames)
        for ev in detected_passes:
            if 0 <= frame_num - int(ev["frame_display"]) <= 15:
                frame = tracker.draw_pass_arrow(frame, ev)

        # Draw overlay stats box
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (320, 190), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(
            frame, "PASSING STATS", (40, 50),
            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2,
        )
        cv2.putText(
            frame, f"Player A : {display_stats['Player A']}", (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, f"Player B : {display_stats['Player B']}", (40, 125),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, f"Player C : {display_stats['Player C']}", (40, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        # Draw locked foreground players with role labels
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

            # Triangle on current possessor
            if role_based_possessions[frame_num] == int(role_id):
                frame = tracker.draw_triangle(frame, bbox, (0, 0, 255))

        # Draw ball
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
