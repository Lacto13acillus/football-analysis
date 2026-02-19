from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import numpy as np
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

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
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

             # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def get_team_color_name(self, bgr_color):
        """Convert BGR color to readable color name"""
        b, g, r = int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])
        
        # Simple color classification
        if g > r and g > b and g > 150:
            return "Hijau"
        elif r > g and r > b and r > 150:
            return "Merah"
        elif r > 200 and g > 200 and b > 200:
            return "Putih"
        elif r < 100 and g < 100 and b < 100:
            return "Hitam"
        elif b > r and b > g and b > 150:
            return "Biru"
        elif r > 150 and g > 150 and b < 100:
            return "Kuning"
        else:
            return "Lainnya"
    
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

        # rectangle_width = 40
        # rectangle_height=20
        # x1_rect = x_center - rectangle_width//2
        # x2_rect = x_center + rectangle_width//2
        # y1_rect = (y2- rectangle_height//2) +15
        # y2_rect = (y2+ rectangle_height//2) +15

        # if track_id is not None:
        #     cv2.rectangle(frame,
        #                   (int(x1_rect),int(y1_rect) ),
        #                   (int(x2_rect),int(y2_rect)),
        #                   color,
        #                   cv2.FILLED)
            
        #     x1_text = x1_rect+12
        #     if track_id > 99:
        #         x1_text -=10
            
        #     cv2.putText(
        #         frame,
        #         f"{track_id}",
        #         (int(x1_text),int(y1_rect+15)),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.6,
        #         (0,0,0),
        #         2
        #     )

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

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, team_colors):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Get team color names
        team_1_color_name = self.get_team_color_name(team_colors[1])
        team_2_color_name = self.get_team_color_name(team_colors[2])
        
        # Calculate team frames
        team_1_frames = team_ball_control[:frame_num+1].count(1)
        team_2_frames = team_ball_control[:frame_num+1].count(2)
        
        # Draw team 1 count with color name
        cv2.putText(frame, f"Team {team_1_color_name}: {team_1_frames}", 
                    (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw team 2 count with color name
        cv2.putText(frame, f"Team {team_2_color_name}: {team_2_frames}", 
                    (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control, team_colors):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color,track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, team_colors)

            output_video_frames.append(frame)

        return output_video_frames     