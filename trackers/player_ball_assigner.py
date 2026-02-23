from utils.bbox_utils import get_center_of_bbox

class PlayerBallAssigner():
    def __init__(self):
        pass

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        bx, by = ball_position

        assigned_player = -1
        min_horizontal_distance = float('inf')

        for player_id, player in players.items():
            px1, py1, px2, py2 = player['bbox']
            p_width = px2 - px1
            p_height = py2 - py1

            # --- LOGIKA BARU: AREA KAKI (FOOT AREA) ---
            # Kita buat kotak imajiner khusus di sekitar kaki pemain
            # (30% bagian bawah tinggi pemain, dengan sedikit padding ke samping dan bawah)
            foot_x1 = px1 - (p_width * 0.2)
            foot_x2 = px2 + (p_width * 0.2)
            foot_y1 = py2 - (p_height * 0.3) # Mulai dari betis/lutut
            foot_y2 = py2 + (p_height * 0.2) # Padding sedikit ke bawah rumput

            # Cek apakah titik tengah bola MASUK ke dalam kotak area kaki ini
            if foot_x1 <= bx <= foot_x2 and foot_y1 <= by <= foot_y2:
                # Jika ada dua pemain berebut, pilih yang titik tengah horizontalnya paling dekat dengan bola
                px_center = (px1 + px2) / 2
                horizontal_distance = abs(px_center - bx)

                if assigned_player == -1 or horizontal_distance < min_horizontal_distance:
                    assigned_player = player_id
                    min_horizontal_distance = horizontal_distance

        return assigned_player