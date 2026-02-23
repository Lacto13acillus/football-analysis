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
    
    def draw_pass_statistics(self, frame, pass_stats, possession_stats):
        """
        Draw pass statistics on the frame
        """
        # Create semi-transparent overlay
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Draw statistics panel in top-left corner
        panel_x, panel_y = 20, 20
        panel_width, panel_height = 300, 200

        # Draw background
        cv2.rectangle(overlay, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), 
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), 
                      (255, 255, 255), 2)

        # Title
        cv2.putText(frame, "PASS STATISTICS", 
                    (panel_x + 10, panel_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Total passes
        cv2.putText(frame, f"Total Passes: {pass_stats.get('total_passes', 0)}", 
                    (panel_x + 10, panel_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Team passes
        y_offset = 75
        for team in [1, 2]:
            color = (0, 0, 255) if team == 1 else (255, 0, 0)
            passes = pass_stats.get('passes_by_team', {}).get(team, 0)
            cv2.putText(frame, f"Team {team} Passes: {passes}", 
                        (panel_x + 10, panel_y + y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20

        # Possession
        if possession_stats:
            cv2.putText(frame, "Possession:", 
                        (panel_x + 10, panel_y + 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for team in [1, 2]:
                color = (0, 0, 255) if team == 1 else (255, 0, 0)
                percentage = possession_stats.get(team, {}).get('percentage', 0)
                cv2.putText(frame, f"Team {team}: {percentage:.1f}%", 
                            (panel_x + 20, panel_y + 135 + (team-1)*20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Average pass distance
        avg_distance = pass_stats.get('average_pass_distance', 0)
        cv2.putText(frame, f"Avg Pass Distance: {avg_distance:.1f}px", 
                    (panel_x + 10, panel_y + 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def draw_pass_connection(self, frame, from_pos, to_pos, team):
        """
        Draw a line showing a pass connection
        """
        color = (0, 0, 255) if team == 1 else (255, 0, 0)
        
        # Draw line with arrow effect
        cv2.line(frame, from_pos, to_pos, color, 2, cv2.LINE_AA)
        
        # Draw circles at endpoints
        cv2.circle(frame, from_pos, 5, color, -1)
        cv2.circle(frame, to_pos, 5, color, -1)
        
        # Draw small arrow
        direction = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        length = np.sqrt(direction[0]**2 + direction[1]**2)
        if length > 0:
            direction = (direction[0]/length, direction[1]/length)
            arrow_pos = (int(to_pos[0] - direction[0]*15), int(to_pos[1] - direction[1]*15))
            cv2.arrowedLine(frame, arrow_pos, to_pos, color, 2, tipLength=0.3)
        
        return frame

    def draw_annotations(self,video_frames, tracks):
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


            # # Draw Team Ball Control
            # frame = self.draw_team_ball_control(frame, frame_num)

            output_video_frames.append(frame)

        return output_video_frames
    
    