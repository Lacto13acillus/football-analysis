# player_identifier.py
# Memetakan ByteTrack ID ke nomor Jersey (#3, #19, Unknown)
# V2: Dynamic Spatial Re-Identification
#   - Saat ByteTrack mereset ID, posisi terakhir jersey yang dikenal
#     digunakan untuk me-match track ID baru ke jersey lama.
#   - Kamera statis => posisi pemain antar frame konsisten.

from typing import Dict, Optional, Tuple, List
import numpy as np


class PlayerIdentifier:
    def __init__(
        self,
        track_id_to_jersey: Optional[Dict[int, str]] = None,
        reassign_distance_threshold: float = 150.0,
        lost_timeout_frames: int = 60
    ):
        """
        Inisialisasi mapping track ID ke nomor jersey dengan
        kemampuan re-identifikasi spasial otomatis.

        Args:
            track_id_to_jersey:
                Mapping awal (seed) {track_id: jersey_string}.
                Contoh: {1: "#3", 2: "#19", 3: "Unknown"}
                Jika None, semua player akan diberi label "Unknown".

            reassign_distance_threshold:
                Jarak maksimum (pixel) antara posisi terakhir jersey
                yang hilang dan track_id baru agar di-reassign.
                Sesuaikan dengan skala video (150px cukup untuk
                resolusi 1080p dengan kamera statis).

            lost_timeout_frames:
                Jumlah frame sebelum jersey yang hilang dianggap
                benar-benar tidak aktif dan tidak bisa di-reassign.
                Default 60 frame = 2 detik pada 30 FPS.
        """
        # --- Mapping aktif ---
        self._track_to_jersey: Dict[int, str] = {}
        self._jersey_to_tracks: Dict[str, List[int]] = {}

        # --- Spatial state ---
        # Posisi terakhir yang diketahui per track_id
        self._last_positions: Dict[int, Tuple[float, float]] = {}
        # Frame terakhir track_id terlihat
        self._last_seen_frame: Dict[int, int] = {}
        # Track ID yang sedang aktif per jersey
        self._active_track_per_jersey: Dict[str, int] = {}

        # --- Parameter re-identification ---
        self._reassign_threshold = reassign_distance_threshold
        self._lost_timeout = lost_timeout_frames

        # --- Muat seed mapping ---
        if track_id_to_jersey:
            for tid, jersey in track_id_to_jersey.items():
                self._register_mapping(tid, jersey)

        print(f"[IDENTIFIER] Mapping jersey dimuat: {self._track_to_jersey}")
        print(f"[IDENTIFIER] Reassign threshold: {self._reassign_threshold}px")
        print(f"[IDENTIFIER] Lost timeout: {self._lost_timeout} frames")

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _register_mapping(self, track_id: int, jersey: str) -> None:
        """Register mapping internal tanpa print."""
        self._track_to_jersey[track_id] = jersey
        if jersey not in self._jersey_to_tracks:
            self._jersey_to_tracks[jersey] = []
        if track_id not in self._jersey_to_tracks[jersey]:
            self._jersey_to_tracks[jersey].append(track_id)
        # Update active track
        self._active_track_per_jersey[jersey] = track_id

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Hitung jarak Euclidean."""
        return float(np.linalg.norm(
            np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
        ))

    # ============================================================
    # DYNAMIC RE-IDENTIFICATION (CORE)
    # ============================================================

    def update_frame(
        self,
        frame_num: int,
        player_tracks: Dict[int, Dict],
        debug: bool = False
    ) -> None:
        """
        Update state re-identifikasi berdasarkan data tracking frame ini.
        HARUS dipanggil setiap frame SEBELUM menggunakan get_jersey_number_for_player.

        Logika:
        1. Update posisi terakhir semua track_id yang terlihat di frame ini.
        2. Identifikasi track_id BARU (belum ada di mapping).
        3. Identifikasi jersey yang HILANG (track_id aktifnya tidak terlihat
           selama beberapa frame).
        4. Cocokkan track_id baru ke jersey yang hilang berdasarkan
           kedekatan posisi (nearest neighbor dengan threshold).

        Args:
            frame_num    : nomor frame saat ini
            player_tracks: tracks['players'][frame_num] = {track_id: {'bbox': [...]}}
            debug        : cetak info re-assignment
        """
        current_track_ids = set(player_tracks.keys())

        # --- Step 1: Update posisi terakhir track_id yang terlihat ---
        for tid, data in player_tracks.items():
            bbox = data.get('bbox')
            if bbox is None:
                continue
            # Posisi = bottom-center (posisi kaki)
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[3]
            self._last_positions[tid] = (cx, cy)
            self._last_seen_frame[tid] = frame_num

        # --- Step 2: Identifikasi track_id baru (belum di-map) ---
        new_track_ids = [
            tid for tid in current_track_ids
            if tid not in self._track_to_jersey
        ]

        if not new_track_ids:
            return

        # --- Step 3: Identifikasi jersey yang hilang ---
        lost_jerseys: Dict[str, Tuple[float, float]] = {}

        for jersey, active_tid in list(self._active_track_per_jersey.items()):
            # Skip "Unknown" — tidak perlu re-assign
            if jersey == "Unknown":
                continue

            # Cek apakah track aktif masih terlihat
            if active_tid in current_track_ids:
                continue

            # Track aktif hilang — cek apakah sudah timeout
            last_frame = self._last_seen_frame.get(active_tid, -999)
            frames_since_lost = frame_num - last_frame

            if frames_since_lost > self._lost_timeout:
                continue  # Sudah terlalu lama, skip

            # Jersey ini hilang dan masih dalam timeout window
            last_pos = self._last_positions.get(active_tid)
            if last_pos is not None:
                lost_jerseys[jersey] = last_pos

        if not lost_jerseys:
            # Tidak ada jersey yang hilang → track baru = Unknown
            for tid in new_track_ids:
                if tid not in self._track_to_jersey:
                    self._register_mapping(tid, "Unknown")
                    if debug:
                        print(f"[IDENTIFIER] Frame {frame_num}: "
                              f"Track {tid} -> Unknown (tidak ada jersey hilang)")
            return

        # --- Step 4: Match track baru ke jersey hilang (nearest neighbor) ---
        # Bangun cost matrix: jarak antara setiap new_tid dan setiap lost_jersey
        assignments: List[Tuple[int, str, float]] = []

        for tid in new_track_ids:
            pos = self._last_positions.get(tid)
            if pos is None:
                continue

            best_jersey = None
            best_dist = float('inf')

            for jersey, lost_pos in lost_jerseys.items():
                dist = self._distance(pos, lost_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_jersey = jersey

            if best_jersey and best_dist <= self._reassign_threshold:
                assignments.append((tid, best_jersey, best_dist))

        # Sort by distance (greedy assignment — closest first)
        assignments.sort(key=lambda x: x[2])
        assigned_jerseys = set()

        for tid, jersey, dist in assignments:
            if jersey in assigned_jerseys:
                continue  # Jersey sudah di-assign ke track lain yang lebih dekat

            self._register_mapping(tid, jersey)
            assigned_jerseys.add(jersey)

            if debug:
                print(f"[IDENTIFIER] Frame {frame_num}: "
                      f"RE-ID Track {tid} -> {jersey} "
                      f"(jarak={dist:.0f}px)")

        # Track baru yang tidak ter-match → Unknown
        for tid in new_track_ids:
            if tid not in self._track_to_jersey:
                self._register_mapping(tid, "Unknown")
                if debug:
                    print(f"[IDENTIFIER] Frame {frame_num}: "
                          f"Track {tid} -> Unknown (tidak cocok)")

    # ============================================================
    # PUBLIC API (kompatibel dengan versi lama)
    # ============================================================

    def set_mapping(self, track_id: int, jersey: str) -> None:
        """
        Set mapping manual untuk satu pemain.
        Berguna untuk override atau koreksi.
        """
        self._register_mapping(track_id, jersey)
        print(f"[IDENTIFIER] Mapping manual: Track ID {track_id} -> Jersey {jersey}")

    def get_jersey_number_for_player(self, track_id: int) -> str:
        """
        Dapatkan nomor jersey dari track ID.

        Returns:
            '#3', '#19', 'Unknown', atau 'ID:{track_id}' jika belum ter-map
        """
        return self._track_to_jersey.get(track_id, f"ID:{track_id}")

    def get_track_id_for_jersey(self, jersey: str) -> Optional[int]:
        """Dapatkan track ID AKTIF dari nomor jersey."""
        return self._active_track_per_jersey.get(jersey, None)

    def is_same_player(self, track_id_a: int, track_id_b: int) -> bool:
        """
        Cek apakah dua track ID merujuk ke pemain yang sama (jersey sama).
        Menangani ID switching dari ByteTrack.
        """
        jersey_a = self.get_jersey_number_for_player(track_id_a)
        jersey_b = self.get_jersey_number_for_player(track_id_b)

        if jersey_a.startswith("ID:") or jersey_b.startswith("ID:"):
            return track_id_a == track_id_b

        # "Unknown" dengan track ID berbeda = pemain berbeda
        if jersey_a == "Unknown" or jersey_b == "Unknown":
            return track_id_a == track_id_b

        return jersey_a == jersey_b

    def get_all_mappings(self) -> Dict[int, str]:
        """Kembalikan semua mapping yang ada."""
        return dict(self._track_to_jersey)

    def get_all_track_ids_for_jersey(self, jersey: str) -> List[int]:
        """
        Kembalikan SEMUA track ID yang pernah di-map ke jersey ini.
        Berguna untuk debugging ID switching history.
        """
        return list(self._jersey_to_tracks.get(jersey, []))

    def print_mappings(self) -> None:
        """Tampilkan semua mapping ke console."""
        print("\n[IDENTIFIER] === PLAYER MAPPINGS ===")
        if not self._track_to_jersey:
            print("[IDENTIFIER] Belum ada mapping.")
        for tid, jersey in sorted(self._track_to_jersey.items()):
            active_marker = ""
            if self._active_track_per_jersey.get(jersey) == tid:
                active_marker = " (AKTIF)"
            print(f"[IDENTIFIER]   Track ID {tid:3d} -> Jersey {jersey}{active_marker}")
        print("[IDENTIFIER] ========================\n")
