from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import numpy as np
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position, measure_distance

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.ball_last_position = None
        self.ball_transfer_count = 0
    
    def detect_frames(self, frames):
        batch_size = 20 
        detections = [] 
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
    
    def count_ball_transfer(self, frame_num, ball_dict, player_dict):
        # Checking if the ball has transferred to a new player
        if frame_num == 0:
            self.ball_last_position = None
            self.ball_transfer_count = 0
        
        current_ball = ball_dict.get(1)  # Assuming the ball has track ID 1
        if current_ball:
            current_ball_bbox = current_ball["bbox"]
            ball_center = get_center_of_bbox(current_ball_bbox)

            if self.ball_last_position:
                # Check if the ball moved between players
                for track_id, player in player_dict.items():
                    player_center = get_center_of_bbox(player["bbox"])
                    if measure_distance(ball_center, player_center) < 50:  # Threshold distance
                        if self.ball_last_position != track_id:
                            # Ball transferred to a new player
                            print(f"⚽ Ball transferred to player {track_id} in frame {frame_num}")
                            self.ball_transfer_count += 1
                            self.ball_last_position = track_id
                        break
            else:
                self.ball_last_position = None  # Ball is not with anyone initially
        return self.ball_transfer_count
    
    def draw_ball_transfer_count(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Ball Transfers: {self.ball_transfer_count}", (10, 30),
                    font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
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
    
    def draw_traingle(self, frame, bbox, color):
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

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0,255,0))

            # Count ball transfers
            ball_transfer_count = self.count_ball_transfer(frame_num, ball_dict, player_dict)
            frame = self.draw_ball_transfer_count(frame)

            output_video_frames.append(frame)

        return output_video_frames