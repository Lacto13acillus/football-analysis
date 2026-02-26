import easyocr
import cv2
import numpy as np
from collections import Counter
import os

# Mematikan NNPACK di dalam script agar tidak bentrok dengan CPU
os.environ['ONNXRUNTIME_DISABLE_NNPACK'] = '1'

class PlayerIdentifier:
    def __init__(self):
        # Tambahkan recognizer agar tidak download ulang & matikan akselerasi yang tidak perlu
        self.reader = easyocr.Reader(['en'], gpu=False, download_enabled=True) 
        self.track_voting = {} 
        self.final_mapping = {} 

    def get_jersey_number(self, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            height = y2 - y1
            width = x2 - x1
            
            # Crop area punggung lebih spesifik agar OCR tidak berat memproses seluruh badan
            # Ambil 25% dari atas sampai 60% tinggi badan
            roi = frame[y1+int(height*0.2):y1+int(height*0.6), x1:x2]
            
            if roi.size == 0: return None

            # Pre-processing sederhana untuk membantu CPU
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Resize sedikit lebih besar agar angka lebih jelas dibaca
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Batasi deteksi agar hanya mencari angka
            results = self.reader.readtext(gray, allowlist='0123456789', detail=0)
            
            if results:
                # Ambil teks pertama yang terdeteksi
                return results[0]
            return None
        except Exception:
            return None

    def update_identities(self, frame, player_tracks):
        for track_id, data in player_tracks.items():
            number = self.get_jersey_number(frame, data['bbox'])
            
            if number in ['3', '19']:
                if track_id not in self.track_voting:
                    self.track_voting[track_id] = []
                self.track_voting[track_id].append(number)
                
                # Update mapping dengan suara terbanyak
                votes = Counter(self.track_voting[track_id]).most_common(1)
                if votes:
                    self.final_mapping[track_id] = votes[0][0]

    def get_assigned_number(self, track_id):
        return self.final_mapping.get(track_id, "Unknown")