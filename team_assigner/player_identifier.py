import easyocr
import cv2
import numpy as np
from collections import Counter

class PlayerIdentifier:
    def __init__(self, expected_numbers=None):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.player_numbers_map = {}  # {track_id: "nomor_punggung"}
        # Nomor yang diharapkan ada di video
        self.expected_numbers = expected_numbers or ['3', '19']
        # History deteksi untuk voting (lebih robust)
        self._detection_history = {}  # {track_id: [list of detected numbers]}

    def get_jersey_number(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        width = x2 - x1

        # Ambil bagian punggung/torso (atas 60% dari bbox)
        roi = frame[y1:y1 + int(height * 0.6), x1:x2]

        if roi.size == 0:
            return None

        # Preprocessing untuk OCR yang lebih baik
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Coba beberapa preprocessing
        results_all = []

        # 1. Original grayscale
        results = self.reader.readtext(gray, allowlist='0123456789')
        results_all.extend(results)

        # 2. CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results = self.reader.readtext(enhanced, allowlist='0123456789')
        results_all.extend(results)

        # 3. Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = self.reader.readtext(binary, allowlist='0123456789')
        results_all.extend(results)

        for (bbox_ocr, text, prob) in results_all:
            text = text.strip()
            if prob > 0.25 and text in self.expected_numbers:
                return text

        # Jika tidak cocok dengan expected, return angka apapun dengan confidence tinggi
        for (bbox_ocr, text, prob) in results_all:
            text = text.strip()
            if prob > 0.5 and len(text) <= 2 and text.isdigit():
                return text

        return None

    def update_identities(self, frame, player_tracks):
        """Update jersey number untuk setiap track_id menggunakan majority voting."""
        for track_id, data in player_tracks.items():
            track_id = int(track_id)

            # Skip jika sudah teridentifikasi dengan confident
            if track_id in self.player_numbers_map and self.player_numbers_map[track_id] != "Unknown":
                continue

            number = self.get_jersey_number(frame, data['bbox'])

            if number is not None:
                if track_id not in self._detection_history:
                    self._detection_history[track_id] = []
                self._detection_history[track_id].append(number)

                # Majority voting setelah beberapa deteksi
                history = self._detection_history[track_id]
                if len(history) >= 3:
                    most_common = Counter(history).most_common(1)[0][0]
                    self.player_numbers_map[track_id] = most_common
                elif len(history) >= 1:
                    # Sementara pakai yang ada
                    self.player_numbers_map[track_id] = Counter(history).most_common(1)[0][0]

            elif track_id not in self.player_numbers_map:
                self.player_numbers_map[track_id] = "Unknown"

    def get_role_jersey_mapping(self, locked_ids_per_frame, tracks):
        """
        Buat mapping role_id -> jersey_number berdasarkan track_id yang paling
        sering di-assign ke setiap role.
        """
        from collections import defaultdict

        role_track_history = defaultdict(list)  # {role_id: [track_ids...]}

        for lm in locked_ids_per_frame:
            for role_id, track_id in lm.items():
                if track_id != -1:
                    role_track_history[role_id].append(track_id)

        role_jersey = {}
        for role_id, track_ids in role_track_history.items():
            # Ambil track_id yang paling sering
            most_common_tid = Counter(track_ids).most_common(1)[0][0]
            jersey = self.player_numbers_map.get(most_common_tid, "Unknown")

            # Jika Unknown, coba track_id lain yang punya jersey number
            if jersey == "Unknown":
                for tid, count in Counter(track_ids).most_common():
                    j = self.player_numbers_map.get(tid, "Unknown")
                    if j != "Unknown":
                        jersey = j
                        break

            role_jersey[role_id] = jersey

        return role_jersey
