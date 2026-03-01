class PlayerIdentifier:
    """
    Manual jersey-number mapping based on track IDs.
    ByteTrack may re-assign IDs mid-video (e.g. 1->150), so we map both.
    
    DITAMBAHKAN: are_same_physical_player() untuk mendeteksi
    track ID re-assignment yang bukan pass sesungguhnya.
    """

    def __init__(self):
        self.player_numbers_map = {}
        self.manual_map = {
            1: "3",
            150: "3",
            2: "19",
            151: "19",
            3: "Unknown",
            152: "Unknown",
        }

    def update_identities(self, frame, player_tracks):
        for track_id in player_tracks:
            if track_id in self.manual_map:
                self.player_numbers_map[track_id] = self.manual_map[track_id]
            elif track_id not in self.player_numbers_map:
                self.player_numbers_map[track_id] = "Unknown"

    def get_jersey_number_for_player(self, track_id):
        return self.player_numbers_map.get(track_id, "Unknown")

    def are_same_physical_player(self, track_id_a, track_id_b):
        """
        FIX KRITIS: Cek apakah dua track ID adalah pemain fisik yang sama.
        Contoh: track ID 1 dan 150 keduanya = #3 (pemain yang sama).
        Jika True, transisi antara mereka BUKAN pass, hanya re-ID.
        """
        jersey_a = self.get_jersey_number_for_player(track_id_a)
        jersey_b = self.get_jersey_number_for_player(track_id_b)
        return jersey_a == jersey_b
