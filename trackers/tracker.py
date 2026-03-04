import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from utils.bbox_utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames: List[np.ndarray],
                      batch_size: int = 20, conf: float = 0.15):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i: i + batch_size]
            preds = self.model.predict(batch, conf=conf, verbose=False)
            detections += preds
        return detections

    def get_object_track(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> Dict[str, List[Dict[int, Dict[str, Any]]]]:
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        
        # BARU: Tambahkan "cones" ke dalam dictionary
        tracks: Dict[str, List[Dict[int, Dict[str, Any]]]] = {
            "players": [], "ball": [], "cones": []
        }

        for frame_num, det in enumerate(detections):
            cls_names = det.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            det_sv = sv.Detections.from_ultralytics(det)
            det_with_tracks = self.tracker.update_with_detections(det_sv)

            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["cones"].append({})

            for row in det_with_tracks:
                bbox = row[0].tolist()
                confidence = float(row[2]) if len(row) > 2 else 1.0
                cls_id = int(row[3])
                track_id = int(row[4])
                if cls_id == cls_names_inv.get("person", -1):
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox, "confidence": confidence
                    }

            ball_candidates: List[Tuple[List[float], float]] = []
            cone_idx = 0 # Counter sederhana untuk cone tanpa tracking ID
            
            for row in det_sv:
                bbox = row[0].tolist()
                conf = float(row[2]) if len(row) > 2 else 0.5
                cls_id = int(row[3])
                class_name = cls_names.get(cls_id, "").lower()

                # Cek bola
                if cls_id == cls_names_inv.get("sports ball", -1) and conf > 0.3:
                    ball_candidates.append((bbox, conf))
                
                # BARU: Cek cone (Pastikan nama class-nya benar, disini di-set "cone")
                if "cone" in class_name and "circle" not in class_name: 
                    # Jika ingin circle cone juga dihitung, hilangkan kondisi 'and "circle" not in class_name'
                    tracks["cones"][frame_num][cone_idx] = {
                        "bbox": bbox, "confidence": conf
                    }
                    cone_idx += 1

            if ball_candidates:
                best_bbox, best_conf = max(ball_candidates, key=lambda x: x[1])
                tracks["ball"][frame_num][1] = {
                    "bbox": best_bbox, "confidence": float(best_conf)
                }

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_pass_arrow(self, frame: np.ndarray,
                        pass_event: Dict[str, Any]) -> np.ndarray:
        # BARU: Warna disesuaikan dengan status success/failed
        is_success = pass_event.get("success", True)
        color = (0, 255, 0) if is_success else (0, 0, 255) # Hijau jika sukses, Merah jika gagal
        
        from_pos = tuple(int(v) for v in pass_event["from_pos"])
        to_pos = tuple(int(v) for v in pass_event["to_pos"])
        
        cv2.arrowedLine(frame, from_pos, to_pos, color, 3, tipLength=0.05)
        cv2.circle(frame, from_pos, 8, color, -1)
        cv2.circle(frame, to_pos, 8, (0, 200, 255), -1)
        
        # Tambahkan teks Success/Failed pada garisnya
        mid_point = ((from_pos[0] + to_pos[0]) // 2, (from_pos[1] + to_pos[1]) // 2)
        status_text = "SUCCESS" if is_success else "FAILED"
        cv2.putText(frame, status_text, (mid_point[0], mid_point[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame, center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0, startAngle=-45, endAngle=235,
            color=color, thickness=2, lineType=cv2.LINE_4
        )
        if track_id is not None:
            cv2.putText(frame, str(track_id), (x_center - 10, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_triangle(self, frame: np.ndarray, bbox: List[float],
                      color: Tuple[int, int, int]) -> np.ndarray:
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y], [x - 10, y - 20], [x + 10, y - 20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame