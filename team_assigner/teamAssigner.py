import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 180  # DIUBAH: 120 -> 180

    def assign_ball_to_player(self, player_track, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        min_distance = float('inf')
        assigned_player = -1

        for player_id, player_data in player_track.items():
            player_bbox = player_data['bbox']

            # Cek jarak dari kaki, tengah bawah, dan center
            left_foot = (player_bbox[0], player_bbox[3])
            right_foot = (player_bbox[2], player_bbox[3])
            center_bottom = ((player_bbox[0] + player_bbox[2]) / 2, player_bbox[3])
            center = get_center_of_bbox(player_bbox)

            dist_left = measure_distance(left_foot, ball_position)
            dist_right = measure_distance(right_foot, ball_position)
            dist_bottom = measure_distance(center_bottom, ball_position)
            dist_center = measure_distance(center, ball_position)

            distance = min(dist_left, dist_right, dist_bottom, dist_center)

            if distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        if min_distance > self.max_player_ball_distance:
            return -1, min_distance

        return assigned_player, min_distance