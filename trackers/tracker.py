from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.15)
            detections += detections_batch
        return detections

    def get_object_track(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["ball"].append({})

            # --- MENDETEKSI PEMAIN ---
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                confidence = frame_detection[2] if len(frame_detection) > 2 else 1.0

                if cls_id == cls_names_inv.get('person', -1):
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "confidence": float(confidence) if not isinstance(confidence, float) else confidence
                    }

            # --- MENDETEKSI BOLA ---
            ball_detections = []
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                conf = frame_detection[2] if len(frame_detection) > 2 else 0.5

                if cls_id == cls_names_inv.get('sports ball', -1):
                    # Filter: Hanya akui bola jika AI sangat yakin (> 30%)
                    if conf > 0.3: 
                        ball_detections.append((bbox, conf))

            if ball_detections:
                # Jika ada beberapa objek mirip bola, ambil 1 yang paling meyakinkan
                best_ball = max(ball_detections, key=lambda x: x[1])
                tracks["ball"][frame_num][1] = {
                    "bbox": best_ball[0],
                    "confidence": float(best_ball[1])
                }

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_pass_arrow(self, frame, pass_event, show_for_frames=30):
        from_pos = pass_event['from_pos']
        to_pos = pass_event['to_pos']
        color = (0, 255, 0)
        cv2.arrowedLine(frame, from_pos, to_pos, color, 3, tipLength=0.15)
        cv2.circle(frame, from_pos, 8, color, -1)
        cv2.circle(frame, to_pos, 8, (0, 200, 255), -1)
        return frame

    def draw_pass_statistics(self, frame, current_pass_count):
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (300, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        text = f"Passes: {current_pass_count}"
        cv2.putText(frame, text, (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            cv2.putText(frame, str(track_id), (x_center - 10, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame