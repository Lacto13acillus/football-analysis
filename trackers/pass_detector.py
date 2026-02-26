from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set
import numpy as np
from utils.bbox_utils import measure_distance, get_center_of_bbox_bottom


class PassDetector:
    """
    FINAL TUNING:
    - smoothing_window:  9 → 13  (kill ALL flicker < 13 frames)
    - min_stable_frames: 4 → 6   (require solid possession segment)
    - cooldown_frames:  15 → 20  (prevent any double-count)
    - min_display_gap:  10 → 15  (visual spacing between arrows)
    """

    def __init__(self, fps: float = 24.0):
        self.fps = float(fps)
        self.smoothing_window = 13
        self.min_stable_frames = 6
        self.min_possession_duration = 1
        self.min_pass_distance = 40
        self.max_pass_distance = 900
        self.cooldown_frames = 20
        self.ball_movement_check_radius = 30
        self.ball_movement_threshold = 15
        self.pass_display_delay = 5
        self.min_display_gap = 15

    def smooth_possessions(self, raw_possessions: Sequence[int]) -> List[int]:
        smoothed = list(raw_possessions)
        half_window = self.smoothing_window // 2
        for i in range(half_window, len(raw_possessions) - half_window):
            window = raw_possessions[i - half_window : i + half_window + 1]
            valid = [p for p in window if p != -1]
            if valid:
                smoothed[i] = Counter(valid).most_common(1)[0][0]
            else:
                smoothed[i] = -1
        return smoothed

    def fill_short_gaps(self, possessions: Sequence[int], max_gap: int = 6) -> List[int]:
        filled = list(possessions)
        last_valid = -1
        gap_start = -1
        for i in range(len(filled)):
            if filled[i] != -1:
                if last_valid != -1 and gap_start != -1:
                    gap_len = i - gap_start
                    if gap_len <= max_gap:
                        for g in range(gap_start, i):
                            filled[g] = last_valid
                last_valid = filled[i]
                gap_start = -1
            else:
                if gap_start == -1:
                    gap_start = i
        return filled

    def get_stable_segments(self, smoothed_possessions: Sequence[int]) -> List[Dict[str, int]]:
        segments: List[Dict[str, int]] = []
        current = -1
        start = 0
        for f, pid in enumerate(smoothed_possessions):
            if pid != current:
                if current != -1:
                    duration = f - start
                    if duration >= self.min_stable_frames:
                        segments.append({
                            "player_id": int(current),
                            "frame_start": int(start),
                            "frame_end": int(f - 1),
                        })
                current = pid
                start = f
        if current != -1:
            duration = len(smoothed_possessions) - start
            if duration >= self.min_stable_frames:
                segments.append({
                    "player_id": int(current),
                    "frame_start": int(start),
                    "frame_end": int(len(smoothed_possessions) - 1),
                })
        return segments

    def validate_ball_movement(self, tracks: Dict[str, Any], frame_idx: int) -> bool:
        if "ball" not in tracks:
            return False
        start = max(0, frame_idx - self.ball_movement_check_radius)
        end = min(len(tracks["ball"]) - 1, frame_idx + self.ball_movement_check_radius)

        positions: List[Tuple[int, int]] = []
        for f in range(start, end + 1):
            bb = tracks["ball"][f].get(1, {}).get("bbox", None)
            if not bb:
                continue
            x = int((bb[0] + bb[2]) / 2)
            y = int((bb[1] + bb[3]) / 2)
            positions.append((x, y))

        if len(positions) < 2:
            return False

        disp = measure_distance(positions[0], positions[-1])
        return disp >= self.ball_movement_threshold

    def find_player_position(
        self,
        tracks: Dict[str, Any],
        frame_idx: int,
        player_id: int,
        role_to_trackid_per_frame: Optional[List[Dict[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        if frame_idx < 0 or frame_idx >= len(tracks.get("players", [])):
            return None

        if role_to_trackid_per_frame is not None:
            if frame_idx >= len(role_to_trackid_per_frame):
                return None
            track_id = role_to_trackid_per_frame[frame_idx].get(int(player_id), -1)
            if track_id == -1:
                return None
            pdata = tracks["players"][frame_idx].get(int(track_id), None)
            if pdata is None:
                return None
            bbox = pdata.get("bbox", None)
            if not bbox:
                return None
            return get_center_of_bbox_bottom(bbox)

        pdata = tracks["players"][frame_idx].get(int(player_id), None)
        if pdata is None:
            return None
        bbox = pdata.get("bbox", None)
        if not bbox:
            return None
        return get_center_of_bbox_bottom(bbox)

    def detect_passes(
        self,
        tracks: Dict[str, Any],
        ball_possessions: Sequence[int],
        debug: bool = False,
        allowed_player_ids: Optional[Set[int]] = None,
        role_to_trackid_per_frame: Optional[List[Dict[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        filled = self.fill_short_gaps(ball_possessions, max_gap=6)
        smoothed = self.smooth_possessions(filled)

        if allowed_player_ids is not None:
            allowed = set(int(x) for x in allowed_player_ids)
            smoothed = [p if p in allowed else -1 for p in smoothed]

        segments = self.get_stable_segments(smoothed)

        if len(segments) < 2:
            return []

        passes: List[Dict[str, Any]] = []
        last_pass_frame = -10_000
        last_display_frame = -10_000

        for i in range(len(segments) - 1):
            s1 = segments[i]
            s2 = segments[i + 1]
            from_id = int(s1["player_id"])
            to_id = int(s2["player_id"])

            if from_id == -1 or to_id == -1 or from_id == to_id:
                continue

            transition_frame = int(s2["frame_start"])

            if transition_frame - last_pass_frame < self.cooldown_frames:
                continue

            if not self.validate_ball_movement(tracks, transition_frame):
                continue

            from_pos = self.find_player_position(tracks, transition_frame, from_id, role_to_trackid_per_frame)
            to_pos = self.find_player_position(tracks, transition_frame, to_id, role_to_trackid_per_frame)

            if from_pos is None or to_pos is None:
                continue

            d = measure_distance(from_pos, to_pos)
            if d < self.min_pass_distance or d > self.max_pass_distance:
                continue

            frame_display = transition_frame + self.pass_display_delay
            if frame_display - last_display_frame < self.min_display_gap:
                frame_display = last_display_frame + self.min_display_gap

            event = {
                "from_player": from_id,
                "to_player": to_id,
                "frame_start": int(s1["frame_start"]),
                "frame_end": int(s2["frame_end"]),
                "frame_display": int(frame_display),
                "from_pos": (int(from_pos[0]), int(from_pos[1])),
                "to_pos": (int(to_pos[0]), int(to_pos[1])),
            }
            passes.append(event)
            last_pass_frame = transition_frame
            last_display_frame = frame_display

        return passes
