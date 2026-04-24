from ultralytics import YOLO

model = YOLO('/home/dika/football-analysis/models/one_touch_pass.pt')

results = model.predict('./input_videos/one_touch_pass.mp4', save=True)
print(results[0])
separator = '==================================='
print(separator)

for box in results[0].boxes:
    print(box)
