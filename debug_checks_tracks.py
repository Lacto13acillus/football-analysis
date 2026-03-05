# debug_check_tracks.py
import pickle

with open('stubs/tracks_cache.pkl', 'rb') as f:
    tracks = pickle.load(f)

print("=== Track 1 (#3 lama) ===")
count1 = sum(1 for f in tracks['players'] if 1 in f)
print(f"Muncul di {count1} frame")
if count1 > 0:
    for fn, pdata in enumerate(tracks['players']):
        if 1 in pdata:
            bbox = pdata[1]['bbox']
            print(f"  Frame {fn}: bbox={[round(b) for b in bbox]}")
            break

print("\n=== Track 3 (Unknown) ===")
count3 = sum(1 for f in tracks['players'] if 3 in f)
print(f"Muncul di {count3} frame")
if count3 > 0:
    for fn, pdata in enumerate(tracks['players']):
        if 3 in pdata:
            bbox = pdata[3]['bbox']
            print(f"  Frame {fn}: bbox={[round(b) for b in bbox]}")
            break
