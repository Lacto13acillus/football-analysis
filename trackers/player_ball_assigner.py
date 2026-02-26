from typing import Any, Dict, List, Optional, Tuple
from utils.bbox_utils import get_center_of_bbox, get_center_of_bbox_bottom, measure_distance
class PlayerBallAssigner:
    """
    Assign ball -> nearest player (by ball center to player foot position).
    WHY foot position:
    - For eye-level camera, the ball is usually near the ground.
    - Foot position is more stable than bbox center when player raises arms or turns.
    """
    def __init__(self, max_player_ball_distance: float = 70.0):
        self.max_player_ball_distance = float(max_player_ball_distance)
    def assign_ball_to_player(
        self,
        player_track: Dict[int, Dict[str, Any]],
        ball_bbox: List[float],
    ) -> Tuple[int, Optional[float]]:
        if not ball_bbox or len(ball_bbox) != 4:
            return -1, None
        ball_center = get_center_of_bbox(ball_bbox)
        best_player = -1
        best_dist: Optional[float] = None
        for tid_raw, pdata in player_track.items():
            tid = int(tid_raw)
            bbox = pdata.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            foot = get_center_of_bbox_bottom(bbox)
            d = measure_distance(ball_center, foot)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_player = tid
        if best_dist is None or best_dist > self.max_player_ball_distance:
            return -1, None
        return best_player, best_dist