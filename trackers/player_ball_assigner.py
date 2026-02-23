from utils.bbox_utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        # Toleransi dinaikkan drastis, tapi kita menggunakan logika "Pemain Terdekat" (Nearest Neighbor)
        self.max_player_ball_distance = 150

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Ukur jarak bola ke ujung kaki kiri, kaki kanan, dan titik tengah selangkangan
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance_center = measure_distance(((player_bbox[0] + player_bbox[2])/2, player_bbox[-1]), ball_position)
            
            # Ambil jarak terdekat dari ketiga titik tersebut
            distance = min(distance_left, distance_right, distance_center)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player