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
from team_assigner.player_identifier import PlayerIdentifier  # ← TAMBAHAN
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
    video_path = "input_videos/passing_number.mp4"
    stub_path = "stubs/track_stubs.pkl"
    output_path = "output_videos/passing_number.avi"

    INITIAL_POSSESSION_ROLE = 0
    INITIAL_POSSESSION_FRAMES = 30

    # ── KONFIGURASI JERSEY NUMBER ──
    # Nomor punggung yang diharapkan ada di video
    EXPECTED_JERSEY_NUMBERS = ['3', '19']
    # Interval frame untuk menjalankan OCR (setiap N frame, hemat resource)
    OCR_INTERVAL = 5
    # Jumlah frame awal yang diproses untuk deteksi jersey
    OCR_SCAN_FRAMES = 150  # scan 150 frame pertama

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)

    tracker = Tracker("yolov8m.pt")
    player_ball_assigner = PlayerBallAssigner(max_player_ball_distance=350)
    player_identifier = PlayerIdentifier(expected_numbers=EXPECTED_JERSEY_NUMBERS)  # ← BARU

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

    # ════════════════════════════════════════════════════════════
    # JERSEY NUMBER DETECTION PHASE
    # Scan frame-frame awal untuk mendeteksi nomor punggung
    # ════════════════════════════════════════════════════════════
    print("\n[INFO] Scanning jersey numbers...")
    scan_limit = min(OCR_SCAN_FRAMES, len(video_frames))
    for f in range(0, scan_limit, OCR_INTERVAL):
        frame = video_frames[f]
        player_tracks_f = tracks["players"][f]
        if player_tracks_f:
            player_identifier.update_identities(frame, player_tracks_f)

    # Jika masih ada Unknown, scan lebih banyak frame
    all_identified = all(
        v != "Unknown" for v in player_identifier.player_numbers_map.values()
    )
    if not all_identified:
        print("[INFO] Some players still Unknown, scanning more frames...")
        for f in range(scan_limit, min(len(video_frames), scan_limit * 2), OCR_INTERVAL):
            frame = video_frames[f]
            player_tracks_f = tracks["players"][f]
            if player_tracks_f:
                player_identifier.update_identities(frame, player_tracks_f)

    # Buat mapping role_id -> jersey_number
    role_jersey = player_identifier.get_role_jersey_mapping(locked_ids_per_frame, tracks)

    # Buat display name berdasarkan jersey number
    role_display_name = {}
    for role_id in [0, 1, 2]:
        jersey = role_jersey.get(role_id, "Unknown")
        if jersey != "Unknown":
            role_display_name[role_id] = f"#{jersey}"
        else:
            role_display_name[role_id] = f"Player {'ABC'[role_id]}"

    print(f"\n{'='*60}")
    print("JERSEY NUMBER DETECTION RESULTS:")
    for role_id in [0, 1, 2]:
        print(f"  Role {role_id} -> Jersey: {role_jersey.get(role_id, 'Unknown')} -> Display: {role_display_name[role_id]}")
    print(f"\nAll track_id -> jersey mappings:")
    for tid, jersey in sorted(player_identifier.player_numbers_map.items()):
        print(f"  track_id {tid} -> #{jersey}")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print("INITIAL ROLE ASSIGNMENT (first valid frame):")
    role_name_old = {0: "Player A", 1: "Player B", 2: "Player C"}
    for f in range(min(10, len(locked_ids_per_frame))):
        lm = locked_ids_per_frame[f]
        if all(tid != -1 for tid in lm.values()):
            for role_id, tid in lm.items():
                bbox = tracks["players"][f].get(tid, {}).get("bbox", [])
                foot = get_center_of_bbox_bottom(bbox) if bbox else "N/A"
                area = bbox_area(bbox) if bbox else 0
                jersey = player_identifier.player_numbers_map.get(tid, "Unknown")
                print(f"  {role_display_name[role_id]}: track_id={tid}, jersey=#{jersey}, foot={foot}, area={area:.0f}")
            print(f"  (at frame {f})")
            break
    print(f"{'='*60}")

    # Build role-based possession array
    role_based_possessions: List[int] = []
    role_to_trackid_per_frame: List[Dict[int, int]] = locked_ids_per_frame

    for frame_num in range(len(video_frames)):
        frame_players = tracks["players"][frame_num]
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox", [])

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
        from_name = role_display_name.get(ev['from_player'], '?')
        to_name = role_display_name.get(ev['to_player'], '?')
        print(
            f"  Pass #{i+1}: {from_name} -> {to_name} "
            f"| frame {ev['frame_start']}-{ev['frame_end']} "
            f"| display@{ev['frame_display']}"
        )

    print(f"\nPossession changes (role-based):")
    prev = -1
    for f, p in enumerate(role_based_possessions):
        if p != prev:
            prev_name = role_display_name.get(prev, 'None')
            curr_name = role_display_name.get(p, 'None')
            print(f"  Frame {f:>5d}: {prev_name:>10s} -> {curr_name}")
            prev = p

    print(f"{'='*60}\n")

    # Aggregate stats per jersey number
    stats_per_player = {}
    for role_id in [0, 1, 2]:
        stats_per_player[role_display_name[role_id]] = 0
    for ev in detected_passes:
        sender_role = int(ev["from_player"])
        name = role_display_name.get(sender_role, "Unknown")
        if name in stats_per_player:
            stats_per_player[name] += 1

    # ════════════════════════════════════════════════════════════
    # VIDEO RENDERING
    # ════════════════════════════════════════════════════════════
    h, w = video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), float(fps), (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    display_stats = {}
    for role_id in [0, 1, 2]:
        display_stats[role_display_name[role_id]] = 0

    events_by_display_frame: Dict[int, List[Dict[str, Any]]] = {}
    for ev in detected_passes:
        fd = int(ev["frame_display"])
        events_by_display_frame.setdefault(fd, []).append(ev)

    # Warna per role untuk visualisasi
    role_colors = {
        0: (0, 255, 0),    # Hijau
        1: (255, 165, 0),  # Oranye (BGR: biru=0, hijau=165, merah=255 → sebenarnya BGR)
        2: (255, 0, 255),  # Magenta
    }
    # BGR format untuk OpenCV
    role_colors_bgr = {
        0: (0, 255, 0),
        1: (0, 165, 255),
        2: (255, 0, 255),
    }

    for frame_num, frame in enumerate(video_frames):
        player_dict = tracks["players"][frame_num]
        locked_map = locked_ids_per_frame[frame_num]

        # Update display stats ketika pass event terjadi
        for ev in events_by_display_frame.get(frame_num, []):
            sender_role = int(ev["from_player"])
            nm = role_display_name.get(sender_role, None)
            if nm in display_stats:
                display_stats[nm] += 1

        # Draw pass arrows
        for ev in detected_passes:
            if 0 <= frame_num - int(ev["frame_display"]) <= 15:
                frame = tracker.draw_pass_arrow(frame, ev)

        # ── STATS OVERLAY dengan Jersey Number ──
        # Hitung tinggi panel berdasarkan jumlah pemain
        panel_height = 70 + len(display_stats) * 35
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (350, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(
            frame, "PASSING STATS", (40, 55),
            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2,
        )

        y_offset = 95
        for role_id in [0, 1, 2]:
            name = role_display_name[role_id]
            count = display_stats.get(name, 0)
            color = role_colors_bgr.get(role_id, (255, 255, 255))
            cv2.putText(
                frame, f"{name} : {count} passes", (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            y_offset += 35

        # ── Draw player annotations dengan jersey number ──
        for role_id, tid in locked_map.items():
            if tid == -1:
                continue
            pdata = player_dict.get(tid, None)
            if pdata is None:
                continue
            bbox = pdata.get("bbox", None)
            if not bbox:
                continue

            nm = role_display_name.get(int(role_id), "?")
            color = role_colors_bgr.get(int(role_id), (0, 255, 0))
            frame = tracker.draw_ellipse(frame, bbox, color)

            # Label dengan jersey number
            label = nm
            jersey = role_jersey.get(int(role_id), "Unknown")
            if jersey != "Unknown":
                label = f"#{jersey}"

            cv2.putText(
                frame,
                label,
                (int(bbox[0]), max(0, int(bbox[1]) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Triangle untuk pemain yang memiliki bola
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
    print(f"\nFinal pass stats by jersey number:")
    for name, count in stats_per_player.items():
        print(f"  {name}: {count} passes")


if __name__ == "__main__":
    main()
