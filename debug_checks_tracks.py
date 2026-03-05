# debug_possession_gaps.py
import pickle
from utils.bbox_utils import get_center_of_bbox

with open('stubs/tracks_cache.pkl', 'rb') as f:
    tracks = pickle.load(f)

print("=== POSISI SEMUA TRACK DI FRAME 0 ===")
for tid, tdata in sorted(tracks['players'][0].items()):
    bbox = tdata['bbox']
    cx, cy = get_center_of_bbox(bbox)
    print(f"  Track {tid}: bbox={[round(b) for b in bbox]} center=({cx}, {cy})")

print("\n=== FREKUENSI KEMUNCULAN SETIAP TRACK ===")
track_counts = {}
for f in range(len(tracks['players'])):
    for tid in tracks['players'][f]:
        track_counts[tid] = track_counts.get(tid, 0) + 1
for tid, count in sorted(track_counts.items()):
    print(f"  Track {tid}: {count} frame")

print("\n=== CEK JARAK BOLA KE SETIAP PLAYER (frame 225-260) ===")
print("  Ini adalah gap di mana #19 seharusnya menerima bola")
from utils.bbox_utils import measure_distance
for f in range(225, 260):
    ball_data = tracks['ball'][f].get(1)
    if not ball_data:
        continue
    ball_pos = get_center_of_bbox(ball_data['bbox'])
    dists = []
    for tid, tdata in tracks['players'][f].items():
        bbox = tdata['bbox']
        foot_pos = ((bbox[0]+bbox[2])/2, bbox[3])  # center bottom
        dist = measure_distance(ball_pos, foot_pos)
        dists.append((tid, dist))
    dists.sort(key=lambda x: x[1])
    closest = dists[:3] if dists else []
    closest_str = ", ".join([f"T{t}:{d:.0f}px" for t,d in closest])
    print(f"  Frame {f}: ball=({ball_pos[0]:.0f},{ball_pos[1]:.0f}) | closest: {closest_str}")
