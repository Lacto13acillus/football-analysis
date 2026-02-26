import easyocr
import cv2
import numpy as np

class PlayerIdentifier:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.player_numbers_map = {}  # {track_id: "nomor_punggung"}
        self.detection_history = {}   # {track_id: {nomor: count}} — voting system

    def get_jersey_number(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        width = x2 - x1

        # Ambil bagian punggung atas (area nomor punggung)
        # Coba beberapa ROI untuk meningkatkan deteksi
        rois = [
            frame[y1:y1 + int(height * 0.6), x1:x2],                    # Atas 60%
            frame[y1 + int(height * 0.1):y1 + int(height * 0.65), x1:x2], # 10%-65%
            frame[y1:y1 + int(height * 0.5), x1 + int(width * 0.15):x2 - int(width * 0.15)],  # Cropped sides
        ]

        for roi in rois:
            if roi.size == 0:
                continue

            # Preprocessing untuk OCR lebih baik
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # CLAHE untuk contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)

            # Resize untuk OCR lebih akurat pada angka kecil
            scale = max(1, 60 // max(roi.shape[:2]))
            if scale > 1:
                enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            results = self.reader.readtext(enhanced, allowlist='0123456789', paragraph=False)

            for (bbox_ocr, text, prob) in results:
                text = text.strip()
                if prob > 0.3 and text in ['3', '19']:
                    return text

        return None

    def update_identities(self, frame, player_tracks):
        for track_id, data in player_tracks.items():
            # Selalu coba deteksi untuk voting (kecuali sudah sangat yakin)
            if track_id in self.player_numbers_map and self.player_numbers_map[track_id] in ['3', '19']:
                # Sudah teridentifikasi, tapi tetap kumpulkan vote untuk konfirmasi
                if track_id in self.detection_history:
                    total_votes = sum(self.detection_history[track_id].values())
                    if total_votes >= 10:  # Cukup yakin, skip
                        continue

            number = self.get_jersey_number(frame, data['bbox'])

            if number is not None:
                # Voting system — akumulasi deteksi per track_id
                if track_id not in self.detection_history:
                    self.detection_history[track_id] = {}
                self.detection_history[track_id][number] = self.detection_history[track_id].get(number, 0) + 1

                # Tentukan nomor berdasarkan vote terbanyak (minimal 2 kali terdeteksi)
                best_number = max(self.detection_history[track_id], key=self.detection_history[track_id].get)
                best_count = self.detection_history[track_id][best_number]

                if best_count >= 2:  # Minimal 2x terdeteksi baru assign
                    self.player_numbers_map[track_id] = best_number

            elif track_id not in self.player_numbers_map:
                self.player_numbers_map[track_id] = "Unknown"

    def get_jersey_number_for_player(self, track_id):
        """Helper untuk mendapatkan nomor punggung player."""
        return self.player_numbers_map.get(track_id, "Unknown")
