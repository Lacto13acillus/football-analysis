from typing import List
import cv2
import numpy as np
def read_video(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames