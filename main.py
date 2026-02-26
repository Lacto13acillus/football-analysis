import os
import cv2
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque
from itertools import permutations

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


@dataclass
class RoleState:
    track_id: int = -1
    foot_pos: Optional[Tuple[float, float]] = None
    lost_count: int = 10_000
    pos_history: deque = field(default_factory=lambda: deque(maxlen=8))

    def predicted_pos(self) -> Optional[Tuple[float, float]]:
        if len(self.pos_history) < 2:
            return self.foot_pos
        p_old = self.pos_history[-2]
        p_new = self.pos_history[-1]
        vx = p_new[0] - p_old[0]
        vy = p_new[1] - p_old[1]
        return (p_new[0] + vx, p_new[1] + vy)

    def update(self, tid: int, foot: Tuple[int, int]):
        self.track_id = tid
        self.foot_pos = (float(foot[0]), float(foot[1]))
        self.lost_count = 0
        self.pos_history.append(self.foot_pos)

    def mark_lost(self):
        self.lost_count += 1


def optimal_assignment_3(
    roles: List[int],
    role_states: Dict[int, "RoleState"],
    candidates: List[CandidatePlayer],
    max_jump_px: int,
) -> Dict[int, Optional[CandidatePlayer]]:
    n_roles = len(roles)
    n_cands = len(candidates)

    if n_cands == 0:
        return {r: None for r in roles}

    LARGE_COST = 1e9

    pred_positions: Dict[int, Optional[Tuple[float, float]]] = {}
    for r in roles:
        pred_positions[r] = role_states[r].predicted_pos()

    best_assignment: Dict[int, Optional[CandidatePlayer]] = {r: None for r in roles}
    best_total_cost = float("inf")

    if n_cands >= n_roles:
        cand_indices = list(range(n_cands))
        for perm in permutations(cand_indices, n_roles):
            total_cost = 0.0
            valid = True
            for ri, ci in enumerate(perm):
                role = roles[ri]
                cand = candidates[ci]
                pred = pred_positions[role]
                if pred is None:
                    total_cost += LARGE_COST / 2 - cand.area * 0.001
                else:
                    d = measure_distance(
                        (int(pred[0]), int(pred[1])),
                        cand.foot_pos,
                    )
                    if d > max_jump_px:
                        valid = False
                        break
                    total_cost += d
            if valid and total_cost < best_total_cost:
                best_total_cost = total_cost
                best_assignment = {}
                for ri, ci in enumerate(perm):
                    best_assignment[roles[ri]] = candidates[ci]
    else:
        for perm in permutations(range(n_roles), n_cands):
            total_cost = 0.0
            valid = True
            assignment: Dict[int, Optional[CandidatePlayer]] = {r: None for r in roles}
            for ci, ri in enumerate(perm):
                role = roles[ri]
                cand = candidates[ci]
                pred = pred_positions[role]
                if pred is None:
                    total_cost += LARGE_COST / 2 - cand.area * 0.001
                else:
                    d = measure_distance(
                        (int(pred[0]), int(pred[1])),
                        cand.foot_pos,
                    )
                    if d > max_jump_px:
                        valid = False
                        break
                    total_cost += d
                assignment[role] = cand
            if valid and total_cost < best_total_cost:
                best_total_cost = total_cost
                best_assignment = assignment

    if best_total_cost >= float("inf"):
        return {r: None for r in roles}

    return best_assignment


def build_locked_roles(
    tracks: Dict[str, List[Dict[int, Dict[str, Any]]]],
    video_frames: List[np.ndarray],
    max_jump_px: int = 200,
    min_area_ratio: float = 0.0025,
    lost_tolerance: int = 20,
) -> Tuple[Dict[int, Dict[int, str]], List[Dict[int, int]]]:
    if not video_frames:
        raise ValueError("video_frames is empty.")
    if "players" not in tracks or len(tracks["players"]) != len(video_frames):
        raise ValueError("tracks['players'] must match video_frames length.")

    height, width = video_frames[0].shape[:2]
    min_area = float(height * width) * float(min_area_ratio)

    ROLE_A, ROLE_B, ROLE_C = 0, 1, 2
    role_names = {ROLE_A: "Player A", ROLE_B: "Player B", ROLE_C: "Player C"}
    roles = [ROLE_A, ROLE_B, ROLE_C]

    role_states: Dict[int, RoleState] = {r: RoleState() for r in roles}
    initialized = False

    frame_roles: Dict[int, Dict[int, str]] = {}
    locked_ids_per_frame: List[Dict[int, int]] = []

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
        top3 = sorted(candidates, key=lambda c: c.area, reverse=True)[:3]
        if len(top3) < 3:
            return False
        top3 = sorted(top3, key=lambda c: c.x)
        for role, cand in zip(roles, top3):
            role_states[role] = RoleState()
            role_states[role].update(cand.track_id, cand.foot_pos)
        return True

    for f in range(len(video_frames)):
        players_f = tracks["players"][f]
        candidates = get_foreground_candidates(players_f)

        if not candidates:
            for r in roles:
                role_states[r].mark_lost()
            frame_roles[f] = {}
            locked_ids_per_frame.append({ROLE_A: -1, ROLE_B: -1, ROLE_C: -1})
            continue

        all_lost = all(role_states[r].lost_count > lost_tolerance for r in roles)
        if not initialized or all_lost:
            ok = initialize_roles(candidates)
            if ok:
                initialized = True
                roles_dict: Dict[int, str] = {}
                locked_ids: Dict[int, int] = {}
                for role in roles:
                    tid = role_states[role].track_id
                    locked_ids[role] = tid
                    if tid != -1:
                        roles_dict[tid] = role_names[role]
                frame_roles[f] = roles_dict
                locked_ids_per_frame.append(locked_ids)
                continue
            else:
                frame_roles[f] = {}
                locked_ids_per_frame.append({ROLE_A: -1, ROLE_B: -1, ROLE_C: -1})
                continue

        assignments = optimal_assignment_3(roles, role_states, candidates, max_jump_px)

        for role in roles:
            cand = assignments[role]
            if cand is not None:
                role_states[role].update(cand.track_id, cand.foot_pos)
            else:
                role_states[role].mark_lost()

        if any(role_states[r].lost_count > lost_tolerance for r in roles):
            initialize_roles(candidates)

        roles_dict = {}
        locked_ids = {}
        for role in roles:
            tid = role_states[role].track_id
            locked_ids[role] = tid
            if tid != -1:
                roles_dict[tid] = role_names[role]
        frame_roles[f] = roles_dict
        locked_ids_per_frame.append(locked_ids)

    return frame_roles, locked_ids_per_frame


def main() -> None:
    video_path = "input_videos/passing_close.mp4"
    stub_path = "stubs/track_stubs.pkl"
    output_path = "output_videos/passing_close.avi"

    # ──────────────────────────────────────────────────────────
    # CONFIGURABLE: Force initial possession for N frames
    # WHY: At frame 0, Player A has the ball at his feet but the ball
    # is detected closer to Player B (center of frame) due to Player A's
    # enormous bbox (area=212K, very close to camera).
    # The ball is physically at A's feet but the foot position (bbox bottom)
    # is far from the ball center because the bbox is so tall.
    #
    # Set to 0 to disable this override.
    # Set to ~30 if you know Player A starts with the ball.
    # ──────────────────────────────────────────────────────────
    INITIAL_POSSESSION_ROLE = 0     # 0 = Player A
    INITIAL_POSSESSION_FRAMES = 30  # Force first 30 frames as Player A

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)

    tracker = Tracker("yolov8m.pt")
    player_ball_assigner = PlayerBallAssigner(max_player_ball_distance=350)

    video_frames = read_video(video_path)
    if not video_frames:
        raise RuntimeError("No frames read from video.")

    if os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            tracks = pickle.load(f)
    else:
        tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path=stub_path)

    if "ball" not in tracks:
        tracks["ball"] = [{} for _ in range(len(video_frames))]

    tracks["ball"] = interpolate_ball_positions(tracks["ball"])

    frame_roles, locked_ids_per_frame = build_locked_roles(tracks, video_frames)

    role_name = {0: "Player A", 1: "Player B", 2: "Player C"}
    print(f"\n{'='*60}")
    print("INITIAL ROLE ASSIGNMENT (first valid frame):")
    for f in range(min(10, len(locked_ids_per_frame))):
        lm = locked_ids_per_frame[f]
        if all(tid != -1 for tid in lm.values()):
            for role_id, tid in lm.items():
                bbox = tracks["players"][f].get(tid, {}).get("bbox", [])
                foot = get_center_of_bbox_bottom(bbox) if bbox else "N/A"
                area = bbox_area(bbox) if bbox else 0
                print(f"  {role_name[role_id]}: track_id={tid}, foot={foot}, area={area:.0f}")
            print(f"  (at frame {f})")
            break
    print(f"{'='*60}")

    # Build role-based possession array
    role_based_possessions: List[int] = []
    role_to_trackid_per_frame: List[Dict[int, int]] = locked_ids_per_frame

    for frame_num in range(len(video_frames)):
        frame_players = tracks["players"][frame_num]
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox", [])

        # ── INITIAL POSSESSION OVERRIDE ──
        # Force Player A for first N frames (ball at A's feet but
        # detected closer to B due to A's massive bbox)
        if frame_num < INITIAL_POSSESSION_FRAMES:
            role_based_possessions.append(INITIAL_POSSESSION_ROLE)
            continue

        if not ball_bbox:
            role_based_possessions.append(-1)
            continue

        locked_map = locked_ids_per_frame[frame_num]
        locked_ids = {tid for tid in locked_map.values() if tid != -1}

        if not locked_ids:
            role_based_possessions.append(-1)
            continue

        filtered_players = {
            tid: pdata for tid, pdata in frame_players.items() if int(tid) in locked_ids
        }
        assigned_track_id, assigned_dist = player_ball_assigner.assign_ball_to_player(
            filtered_players, ball_bbox
        )

        if assigned_track_id == -1:
            role_based_possessions.append(-1)
            continue

        role_id = -1
        for r, tid in locked_map.items():
            if tid == assigned_track_id:
                role_id = int(r)
                break
        role_based_possessions.append(role_id)

    # Get FPS
    cap_temp = cv2.VideoCapture(video_path)
    fps = float(cap_temp.get(cv2.CAP_PROP_FPS)) or 24.0
    cap_temp.release()

    # Detect passes
    pass_detector = PassDetector(fps=fps)
    allowed_role_ids = {0, 1, 2}
    detected_passes = pass_detector.detect_passes(
        tracks=tracks,
        ball_possessions=role_based_possessions,
        debug=False,
        allowed_player_ids=allowed_role_ids,
        role_to_trackid_per_frame=role_to_trackid_per_frame,
    )

    # ── DEBUG OUTPUT ──
    print(f"\nTotal frames: {len(video_frames)}, FPS: {fps:.1f}")
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

    print(f"\nRole-to-TrackID changes:")
    prev_map: Dict[int, int] = {}
    for f, lm in enumerate(locked_ids_per_frame):
        if lm != prev_map:
            parts = [f"{role_name[r]}=tid{tid}" for r, tid in sorted(lm.items())]
            print(f"  Frame {f:>5d}: {', '.join(parts)}")
            prev_map = dict(lm)
    print(f"{'='*60}\n")

    # Aggregate stats
    stats_per_player = {"Player A": 0, "Player B": 0, "Player C": 0}
    for ev in detected_passes:
        sender_role = int(ev["from_player"])
        stats_per_player[role_name.get(sender_role, "Player A")] += 1

    # Prepare video writer
    h, w = video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), float(fps), (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    display_stats = {"Player A": 0, "Player B": 0, "Player C": 0}

    events_by_display_frame: Dict[int, List[Dict[str, Any]]] = {}
    for ev in detected_passes:
        fd = int(ev["frame_display"])
        events_by_display_frame.setdefault(fd, []).append(ev)

    for frame_num, frame in enumerate(video_frames):
        player_dict = tracks["players"][frame_num]
        locked_map = locked_ids_per_frame[frame_num]

        for ev in events_by_display_frame.get(frame_num, []):
            sender_role = int(ev["from_player"])
            nm = role_name.get(sender_role, None)
            if nm in display_stats:
                display_stats[nm] += 1

        for ev in detected_passes:
            if 0 <= frame_num - int(ev["frame_display"]) <= 15:
                frame = tracker.draw_pass_arrow(frame, ev)

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

            if role_based_possessions[frame_num] == int(role_id):
                frame = tracker.draw_triangle(frame, bbox, (0, 0, 255))

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
