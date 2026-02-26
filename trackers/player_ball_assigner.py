from typing import Any, Dict, List, Optional, Tuple
from utils.bbox_utils import get_center_of_bbox, get_center_of_bbox_bottom, measure_distance


class PlayerBallAssigner:
    """
    Assign ball -> nearest player.
    
    CHANGE: Uses DUAL distance (foot + bbox center), takes the MINIMUM.
    WHY: Player A's foot position might be far from ball (e.g., ball at waist level
    or ball between legs), but bbox center might be closer. Taking min of both
    gives more robust assignment.
    
    CHANGE: max_player_ball_distance 150 -> 200
    WHY: In eye-level camera during passing drills, ball can be 150-200px away
    from the nearest body reference point, especially during wind-up or follow-through.
    """

    def __init__(self, max_player_ball_distance: float = 200.0):
        self.max_player_ball_distance = float(max_player_ball_distance)

    def assign_ball_to_player(
        self,
        player_track: Dict[int, Dict[str, Any]],
        ball_bbox: List[float],
    ) -> Tuple[int, Optional[float]]:
        if not ball_bbox or len(ball_bbox) != 4:
            return -1, None

        ball_center = get_center_of_bbox(ball_bbox)
        ball_bottom = get_center_of_bbox_bottom(ball_bbox)

        best_player = -1
        best_dist: Optional[float] = None

        for tid_raw, pdata in player_track.items():
            tid = int(tid_raw)
            bbox = pdata.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue

            foot = get_center_of_bbox_bottom(bbox)
            center = get_center_of_bbox(bbox)

            # Dual distance: take the minimum of foot and center distances
            d_foot = measure_distance(ball_center, foot)
            d_center = measure_distance(ball_center, center)
            d_foot_bottom = measure_distance(ball_bottom, foot)
            d = min(d_foot, d_center, d_foot_bottom)

            if best_dist is None or d < best_dist:
                best_dist = d
                best_player = tid

        if best_dist is None or best_dist > self.max_player_ball_distance:
            return -1, None

        return best_player, best_dist
