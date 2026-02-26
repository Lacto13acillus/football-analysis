import easyocr
import cv2
import numpy as np
from collections import Counter

class PlayerIdentifier:
    def __init__(self):
        # Menggunakan GPU=True jika ada NVIDIA, jika tidak False
        self.reader = easyocr.Reader(['en'], gpu=False) 
        self.track_voting = {} # {track_id: [list_nomor_terdeteksi]}
        self.final_mapping = {} # {track_id: "nomor_punggung"}

    def get_jersey_number(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        # Ambil area punggung (tengah atas baju)
        roi = frame[y1+int(height*0.15):y1+int(height*0.6), x1:x2]
        
        if roi.size == 0: return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray, allowlist='0123456789')
        
        for (bbox_ocr, text, prob) in results:
            if prob > 0.30: # Confidence threshold
                return text
        return None

    def update_identities(self, frame, player_tracks):
        for track_id, data in player_tracks.items():
            number = self.get_jersey_number(frame, data['bbox'])
            
            # Hanya proses jika nomor adalah 3 atau 19
            if number in ['3', '19']:
                if track_id not in self.track_voting:
                    self.track_voting[track_id] = []
                self.track_voting[track_id].append(number)
                
                # Voting: Ambil nomor yang paling sering muncul untuk track_id ini
                most_common = Counter(self.track_voting[track_id]).most_common(1)[0][0]
                self.final_mapping[track_id] = most_common

    def get_assigned_number(self, track_id):
        return self.final_mapping.get(track_id, "Unknown")