import easyocr
import cv2
import numpy as np

class PlayerIdentifier:
    def __init__(self):
        self.player_numbers_map = {}
        self.manual_map = {
            1: '3',
            150: '3',
            2: '19',
            151: '19',
            3: 'Unknown',
            152: 'Unknown',
        }

    def update_identities(self, frame, player_tracks):
        for track_id in player_tracks:
            if track_id in self.manual_map:
                self.player_numbers_map[track_id] = self.manual_map[track_id]
            elif track_id not in self.player_numbers_map:
                self.player_numbers_map[track_id] = "Unknown"

    def get_jersey_number_for_player(self, track_id):
        return self.player_numbers_map.get(track_id, "Unknown")

