import pytesseract
import cv2
import numpy as np

class PlayerIdentifier:
    def __init__(self):
        self.player_numbers_map = {}
        self.detection_history = {}
        # Konfigurasi Tesseract untuk digit saja
        self.tess_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'

    def get_jersey_number(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        width = x2 - x1

        if height < 20 or width < 10:
            return None

        # Beberapa ROI area punggung
        rois = [
            frame[y1:y1 + int(height * 0.6), x1:x2],
            frame[y1 + int(height * 0.1):y1 + int(height * 0.65), x1:x2],
            frame[y1:y1 + int(height * 0.5),
                  x1 + int(width * 0.15):x2 - int(width * 0.15)],
        ]

        for roi in rois:
            if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                continue

            # Preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)

            # Resize agar lebih mudah dibaca OCR
            scale = max(1, 80 // max(roi.shape[:2]))
            if scale > 1:
                enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)

            # Thresholding untuk memisahkan angka dari jersey
            _, thresh = cv2.threshold(enhanced, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR dengan Tesseract
            try:
                text = pytesseract.image_to_string(thresh,
                                                    config=self.tess_config).strip()
                if text in ['3', '19']:
                    return text

                # Coba invert (angka gelap di jersey terang atau sebaliknya)
                thresh_inv = cv2.bitwise_not(thresh)
                text_inv = pytesseract.image_to_string(thresh_inv,
                                                        config=self.tess_config).strip()
                if text_inv in ['3', '19']:
                    return text_inv
            except Exception:
                continue

        return None

    def update_identities(self, frame, player_tracks):
        for track_id, data in player_tracks.items():
            # Skip jika sudah yakin (vote >= 10)
            if track_id in self.player_numbers_map \
               and self.player_numbers_map[track_id] in ['3', '19']:
                if track_id in self.detection_history:
                    total_votes = sum(self.detection_history[track_id].values())
                    if total_votes >= 10:
                        continue

            number = self.get_jersey_number(frame, data['bbox'])

            if number is not None:
                if track_id not in self.detection_history:
                    self.detection_history[track_id] = {}
                self.detection_history[track_id][number] = \
                    self.detection_history[track_id].get(number, 0) + 1

                best_number = max(self.detection_history[track_id],
                                  key=self.detection_history[track_id].get)
                best_count = self.detection_history[track_id][best_number]

                if best_count >= 2:
                    self.player_numbers_map[track_id] = best_number

            elif track_id not in self.player_numbers_map:
                self.player_numbers_map[track_id] = "Unknown"

    def get_jersey_number_for_player(self, track_id):
        return self.player_numbers_map.get(track_id, "Unknown")
