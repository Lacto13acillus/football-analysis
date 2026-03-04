# player_identifier.py
# Memetakan ByteTrack ID ke nomor Jersey (#3, #19, Unknown)
# Karena pengenalan jersey otomatis kompleks, gunakan mapping manual.

from typing import Dict, Optional


class PlayerIdentifier:
    def __init__(self, track_id_to_jersey: Optional[Dict[int, str]] = None):
        """
        Inisialisasi mapping track ID ke nomor jersey.

        Args:
            track_id_to_jersey: dict {track_id: jersey_string}
                Contoh: {1: "#3", 2: "#19", 3: "Unknown"}
                Jika None, semua player akan diberi label berdasarkan track ID.
        """
        # Mapping default: kosong, akan terisi saat runtime
        self._track_to_jersey: Dict[int, str] = track_id_to_jersey or {}

        # Reverse mapping untuk pencarian jersey -> track_id
        self._jersey_to_track: Dict[str, int] = {
            v: k for k, v in self._track_to_jersey.items()
        }

        print(f"[IDENTIFIER] Mapping jersey dimuat: {self._track_to_jersey}")

    def set_mapping(self, track_id: int, jersey: str) -> None:
        """
        Set mapping untuk satu pemain.

        Args:
            track_id: ByteTrack track ID
            jersey  : nomor jersey, contoh '#3', '#19', atau 'Unknown'
        """
        self._track_to_jersey[track_id] = jersey
        self._jersey_to_track[jersey] = track_id
        print(f"[IDENTIFIER] Mapping baru: Track ID {track_id} -> Jersey {jersey}")

    def get_jersey_number_for_player(self, track_id: int) -> str:
        """
        Dapatkan nomor jersey dari track ID.

        Args:
            track_id: ByteTrack track ID

        Returns:
            String jersey (e.g., '#3', '#19', 'Unknown', atau 'ID:{track_id}')
        """
        return self._track_to_jersey.get(track_id, f"ID:{track_id}")

    def get_track_id_for_jersey(self, jersey: str) -> Optional[int]:
        """
        Dapatkan track ID dari nomor jersey.

        Args:
            jersey: string jersey (e.g., '#3')

        Returns:
            Track ID atau None jika tidak ditemukan
        """
        return self._jersey_to_track.get(jersey, None)

    def is_same_player(self, track_id_a: int, track_id_b: int) -> bool:
        """
        Cek apakah dua track ID merujuk ke pemain yang sama (jersey sama).
        Berguna untuk handle ID switching dari ByteTrack.
        """
        jersey_a = self.get_jersey_number_for_player(track_id_a)
        jersey_b = self.get_jersey_number_for_player(track_id_b)

        # Jika keduanya "ID:xxx" (tidak dikenal), anggap berbeda
        if jersey_a.startswith("ID:") or jersey_b.startswith("ID:"):
            return track_id_a == track_id_b

        return jersey_a == jersey_b

    def get_all_mappings(self) -> Dict[int, str]:
        """Kembalikan semua mapping yang ada."""
        return dict(self._track_to_jersey)

    def print_mappings(self) -> None:
        """Tampilkan semua mapping ke console."""
        print("\n[IDENTIFIER] === PLAYER MAPPINGS ===")
        if not self._track_to_jersey:
            print("[IDENTIFIER] Belum ada mapping. Jalankan debug_find_gate_ids.py dulu.")
        for tid, jersey in sorted(self._track_to_jersey.items()):
            print(f"[IDENTIFIER]   Track ID {tid:3d} -> Jersey {jersey}")
        print("[IDENTIFIER] ========================\n")