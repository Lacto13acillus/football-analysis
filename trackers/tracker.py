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
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_track(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            # "referees":[], # DIBLOKIR: Tidak ada wasit
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # DIBLOKIR: Konversi Goalkeeper ke player ditiadakan karena tidak ada kiper
            # for object_ind , class_id in enumerate(detection_supervision.class_id):
            #     if cls_names[class_id] == "goalkeeper":
            #         detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            # tracks["referees"].append({}) # DIBLOKIR
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                # DIBLOKIR: Tracker untuk wasit
                # if cls_id == cls_names_inv['referee']:
                #     tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_pass_arrow(self, frame, pass_event, show_for_frames=30):
        from_pos = pass_event['from_pos']
        to_pos = pass_event['to_pos']
        
        # Selalu hijau karena ini latihan passing (semua umpan dianggap sukses ke kawan)
        color = (0, 255, 0) 

        cv2.arrowedLine(frame, from_pos, to_pos, color, 3, tipLength=0.2)
        cv2.circle(frame, from_pos, 8, color, -1)
        cv2.circle(frame, to_pos, 8, color, -1)

        return frame

    def draw_pass_statistics(self, frame, pass_stats):
        """ Draw total pass statistics on frame """
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (280, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Cukup tampilkan total umpan
        text = f"Total Passes: {pass_stats['total_passes']}"
        cv2.putText(frame, text, (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    # ... (Fungsi draw_ellipse dan draw_triangle tetap sama seperti aslinya) ...
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        return frame
    
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame