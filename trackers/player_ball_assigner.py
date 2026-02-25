import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox

class PlayerBallAssigner:
    def __init__(self):
        # DIUBAH: dari 70 -> 120 pixel
        # Alasan: saat pemain mengontrol bola, jarak bbox kaki ke pusat bola
        # bisa > 70px tergantung postur tubuh (header, chest control, dll)
        self.max_player_ball_distance = 120

    def assign_ball_to_player(self, player_track, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        min_distance = float('inf')
        assigned_player = -1

        for player_id, player_data in player_track.items():
            player_bbox = player_data['bbox']

            # Gunakan titik bawah bbox (kaki) untuk jarak ke bola
            # Kaki kiri bawah dan kaki kanan bawah
            left_foot = (player_bbox[0], player_bbox[3])
            right_foot = (player_bbox[2], player_bbox[3])

            dist_left = measure_distance(left_foot, ball_position)
            dist_right = measure_distance(right_foot, ball_position)
            distance = min(dist_left, dist_right)

            if distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        if min_distance > self.max_player_ball_distance:
            return -1, min_distance

        return assigned_player, min_distance