class PlayerIdentifier:
    """
    Jersey number mapper.
    Maps track IDs (yang bisa berubah mid-video oleh ByteTrack)
    ke nomor punggung tetap.

    BARU:
    - track_id_to_jersey(): convert track ID → jersey string
    - are_same_physical_player(): cek apakah 2 track ID = 1 pemain fisik
    - get_all_track_ids_for_jersey(): kebalikan mapping
    """

    def __init__(self):
        self.player_numbers_map = {}
        # ByteTrack bisa re-assign ID mid-video, jadi map keduanya
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

    def track_id_to_jersey(self, track_id):
        """Alias for get_jersey_number_for_player"""
        return self.get_jersey_number_for_player(track_id)

    def are_same_physical_player(self, tid_a, tid_b):
        """
        Cek apakah dua track ID adalah pemain fisik yang sama.
        Contoh: track ID 1 dan 150 → keduanya #3 → True
        """
        return self.get_jersey_number_for_player(tid_a) == \
               self.get_jersey_number_for_player(tid_b)

    def get_all_track_ids_for_jersey(self, jersey):
        """Dapatkan semua track ID yang pernah di-assign ke jersey tertentu"""
        return [tid for tid, j in self.player_numbers_map.items() if j == jersey]
