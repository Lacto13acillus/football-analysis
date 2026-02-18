from ultralytics import YOLO

model = YOLO(' /home/dika/football-analysis/models/best.pt')

results = model.predict('./input_videos/football_analysis.mp4', save=True)
print(results[0])
separator = '==================================='
print(separator)

for box in results[0].boxes:
    print(box)