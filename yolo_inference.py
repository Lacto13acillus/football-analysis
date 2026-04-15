from ultralytics import YOLO

model = YOLO('/home/dika/football-analysis/models/longpass.pt')

results = model.predict('./input_videos/longpass.mp4', save=True)
print(results[0])
separator = '==================================='
print(separator)

for box in results[0].boxes:
    print(box)
