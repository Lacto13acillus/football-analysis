from ultralytics import YOLO

model = YOLO('/home/dika/football-analysis/models/dribbling-models.pt')

results = model.predict('./input_videos/dribbling_count.mp4', save=True)
print(results[0])
separator = '==================================='
print(separator)

for box in results[0].boxes:
    print(box)
