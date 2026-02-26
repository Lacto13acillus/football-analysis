from trackers import Tracker
from trackers.player_ball_assigner import PlayerBallAssigner
from trackers.pass_detector import PassDetector
from utils.bbox_utils import interpolate_ball_positions
from utils.video_utils import read_video
import cv2
import pickle
import os

def main():
    # Pastikan path video input dan output Anda sesuai dengan struktur folder Anda
    video_path = 'input_videos/passing_drill.mp4' 
    stub_path = 'stubs/track_stubs.pkl'
    output_path = 'output_videos/passing_drill_output.avi'

    # Inisialisasi Tracker dan Assigner
    tracker = Tracker('models/best.pt')
    player_ball_assigner = PlayerBallAssigner()
    
    # === STEP 1: Get Tracks ===
    print("Membaca video dan memproses tracking...")
    video_frames = read_video(video_path)
    if os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
    else:
        tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path=stub_path)

    # === STEP 1.5: Interpolasi Posisi Bola ===
    tracks['ball'] = interpolate_ball_positions(tracks['ball'])

    # === STEP 2: Identifikasi 3 Pemain Utama Berdasarkan Area & Posisi X ===
    print("Mengidentifikasi Player A, B, dan C berdasarkan posisi spasial...")
    player_stats = {} 
    
    # Kumpulkan data ukuran (area) dan posisi sumbu X dari semua player yang terdeteksi
    for frame_num, players in enumerate(tracks['players']):
        for track_id, data in players.items():
            bbox = data['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            x_center = (bbox[0] + bbox[2]) / 2
            
            if track_id not in player_stats:
                player_stats[track_id] = {'areas': [], 'x_positions': []}
            
            player_stats[track_id]['areas'].append(area)
            player_stats[track_id]['x_positions'].append(x_center)

    # Hitung rata-rata area dan median posisi X untuk setiap ID
    id_summaries = []
    for track_id, stats in player_stats.items():
        avg_area = sum(stats['areas']) / len(stats['areas'])
        median_x = sorted(stats['x_positions'])[len(stats['x_positions'])//2]
        id_summaries.append({'track_id': track_id, 'avg_area': avg_area, 'median_x': median_x})

    # Ambil 3 ID player dengan rata-rata area terbesar (menyaring pemain kecil di background)
    id_summaries.sort(key=lambda x: x['avg_area'], reverse=True)
    top_3_players = id_summaries[:3]
    
    # Urutkan ketiga ID tersebut berdasarkan posisi X (dari kiri ke kanan layar)
    top_3_players.sort(key=lambda x: x['median_x'])

    # Petakan ID ke Nama Pemain
    player_names = {}
    if len(top_3_players) >= 3:
        player_names[top_3_players[0]['track_id']] = "Player A" # Kiri (Baju Hitam)
        player_names[top_3_players[1]['track_id']] = "Player B" # Tengah (Baju Putih)
        player_names[top_3_players[2]['track_id']] = "Player C" # Kanan (Baju Biru/Hitam)
    
    print(f"Mapping Pemain Berhasil: {player_names}")

    # Inisialisasi variabel untuk menghitung jumlah passing
    stats_per_player = {"Player A": 0, "Player B": 0, "Player C": 0, "Other": 0}

    # === STEP 3: Possession Mapping ===
    print("Memproses penguasaan bola (possession)...")
    raw_ball_possessions = []
    for frame_num, frame_img in enumerate(video_frames):
        player_track = tracks['players'][frame_num]
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])

        if len(ball_bbox) == 0:
            raw_ball_possessions.append(-1)
        else:
            assigned_player, _ = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
            raw_ball_possessions.append(assigned_player)

    # === STEP 4: Pass Detection ===
    print("Mendeteksi passing...")
    cap_temp = cv2.VideoCapture(video_path)
    fps = int(cap_temp.get(cv2.CAP_PROP_FPS)) or 24
    cap_temp.release()

    pass_detector = PassDetector(fps=fps)
    detected_passes = pass_detector.detect_passes(tracks, raw_ball_possessions, debug=False)

    # === STEP 5: Render & Real-time Counter ===
    print("Merender video akhir...")
    height, width, _ = video_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for frame_num, frame in enumerate(video_frames):
        player_dict = tracks["players"][frame_num]
        
        # Logika Penambahan Score Passing
        for pass_event in detected_passes:
            if pass_event['frame_display'] == frame_num:
                sender_id = pass_event['from_player']
                name = player_names.get(sender_id, "Other")
                
                if name in stats_per_player:
                    stats_per_player[name] += 1
                else:
                    stats_per_player["Other"] += 1

        # Gambar UI Dashboard di Kiri Atas
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (300, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, "PASSING STATS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Player A : {stats_per_player['Player A']}", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player B : {stats_per_player['Player B']}", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player C : {stats_per_player['Player C']}", (40, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Gambar Trackers (Ellipse & Nama) untuk setiap pemain
        for track_id, player in player_dict.items():
            name = player_names.get(track_id, "Other")
            
            # Hanya tampilkan highlight hijau dan nama untuk Player A, B, dan C
            if name != "Other":
                color = (0, 255, 0) # Warna hijau
                frame = tracker.draw_ellipse(frame, player["bbox"], color)
                cv2.putText(frame, name, (int(player["bbox"][0]), int(player["bbox"][1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Jika pemain sedang menguasai bola, beri marker segitiga merah di atasnya
            if raw_ball_possessions[frame_num] == track_id:
                frame = tracker.draw_traingle(frame, player["bbox"], (0, 0, 255))

        # Ball Tracker (Segitiga hijau di atas bola)
        ball_dict = tracks["ball"][frame_num]
        for _, ball in ball_dict.items():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        # Tulis frame yang sudah digambar ke video output
        out.write(frame)

    out.release()
    print(f"Selesai! Video berhasil disimpan di: {output_path}")

if __name__ == '__main__':
    main()