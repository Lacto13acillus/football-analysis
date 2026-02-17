from ultralytics import YOLO

model = YOLO('yolov8x')

results = model.predict('', save=True)
print = results([0])
print = ('===================================')
for box in results[0].boxes:
    print(box)