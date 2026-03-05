# debug_all_tracks_position.py
import pickle
from utils.bbox_utils import get_center_of_bbox, measure_distance

with open('stubs/tracks_cache.pkl', 'rb') as f:
    tracks = pickle.load(f)

total_frames = len(tracks['players'])

# Track kemunculan dan posisi rata-rata setiap track
track_info = {}
for f in range(total_frames):
    for tid, tdata in tracks['players'][f].items():
        bbox = tdata['bbox']
        cx, cy = get_center_of_bbox(bbox)
        if tid not in track_info:
            track_info[tid] = {
                'first_frame': f, 'last_frame': f, 'count': 0,
                'xs': [], 'ys': [], 'positions': []
            }
        track_info[tid]['last_frame'] = f
        track_info[tid]['count'] += 1
        track_info[tid]['xs'].append(cx)
        track_info[tid]['ys'].append(cy)

print("=" * 80)
print("ANALISIS SEMUA TRACK ID")
print("=" * 80)

import numpy as np
for tid in sorted(track_info.keys()):
    info = track_info[tid]
    avg_x = np.mean(info['xs'])
    avg_y = np.mean(info['ys'])
    min_y = min(info['ys'])
    max_y = max(info['ys'])
    print(f"\nTrack {tid}:")
    print(f"  Frame range : {info['first_frame']} - {info['last_frame']} ({info['count']} frame)")
    print(f"  Posisi rata2: ({avg_x:.0f}, {avg_y:.0f})")
    print(f"  Range Y     : {min_y:.0f} - {max_y:.0f}")

    # Cek kedekatan dengan target cone (932.4, 276.4)
    dist_to_cone = measure_distance((avg_x, avg_y), (932.4, 276.4))
    print(f"  Jarak ke cone target: {dist_to_cone:.0f}px")

    # Cek kedekatan dengan Track 2 (#19) dan Track 3 (#3)
    if tid not in [2, 3]:
        if 2 in track_info:
            avg_t2 = (np.mean(track_info[2]['xs']), np.mean(track_info[2]['ys']))
            dist_t2 = measure_distance((avg_x, avg_y), avg_t2)
            print(f"  Jarak ke Track 2 (#19): {dist_t2:.0f}px")
        if 3 in track_info:
            avg_t3 = (np.mean(track_info[3]['xs']), np.mean(track_info[3]['ys']))
            dist_t3 = measure_distance((avg_x, avg_y), avg_t3)
            print(f"  Jarak ke Track 3 (#3) : {dist_t3:.0f}px")

# Cek possession di frame 260-490 (area gap)
print("\n" + "=" * 80)
print("CEK POSSESSION FRAME 260-490 (gap area)")
print("=" * 80)
for f in range(260, 490, 10):
    ball_data = tracks['ball'][f].get(1)
    if not ball_data:
        continue
    ball_pos = get_center_of_bbox(ball_data['bbox'])
    dists = []
    for tid, tdata in tracks['players'][f].items():
        bbox = tdata['bbox']
        foot_pos = ((bbox[0]+bbox[2])/2, bbox[3])
        dist = measure_distance(ball_pos, foot_pos)
        dists.append((tid, dist))
    dists.sort(key=lambda x: x[1])
    top3 = dists[:4]
    parts = [f"T{t}:{d:.0f}px" for t, d in top3]
    print(f"  Frame {f:3d}: ball=({ball_pos[0]:.0f},{ball_pos[1]:.0f}) | {', '.join(parts)}")
