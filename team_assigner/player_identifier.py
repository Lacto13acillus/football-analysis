import easyocr
import cv2
import numpy as np

class PlayerIdentifier:
    def __init__(self):
        # Menggunakan angka saja agar lebih cepat dan akurat
        self.reader = easyocr.Reader(['en'], gpu=False) 
        self.player_numbers_map = {} # {track_id: "nomor_punggung"}

    def get_jersey_number(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        # Ambil area punggung (sekitar 2/3 bagian atas tubuh)
        height = y2 - y1
        roi = frame[y1:y1+int(height*0.7), x1:x2]
        
        if roi.size == 0: return None

        # Pre-processing untuk memperjelas angka
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray, allowlist='0123456789')
        
        for (bbox_ocr, text, prob) in results:
            if prob > 0.35: # Confidence threshold
                return text
        return None

    def update_identities(self, frame, player_tracks):
        for track_id, data in player_tracks.items():
            # Jika track_id belum punya nomor atau masih "Unknown", coba deteksi
            if track_id not in self.player_numbers_map or self.player_numbers_map[track_id] == "Unknown":
                number = self.get_jersey_number(frame, data['bbox'])
                if number in ['9', '15', '30']: # Hanya simpan nomor yang relevan dengan video
                    self.player_numbers_map[track_id] = number
                elif track_id not in self.player_numbers_map:
                    self.player_numbers_map[track_id] = "Unknown"