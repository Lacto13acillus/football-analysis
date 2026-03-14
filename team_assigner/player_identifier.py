# player_identifier.py
# V4: Color-Based Player Identification — LOCKED MAPPING
#   - Sekali track ID teridentifikasi warna bajunya, mapping di-lock
#   - Tidak bisa berubah lagi (stabil)
#   - Menggunakan voting window sebelum lock

from typing import Dict, Optional, Tuple, List
from collections import Counter
import numpy as np
import cv2


class PlayerIdentifier:
    def __init__(
        self,
        track_id_to_jersey: Optional[Dict[int, str]] = None,
        reassign_distance_threshold: float = 150.0,
        lost_timeout_frames: int = 60,
        color_vote_window: int = 15,
        lock_after_votes: int = 8  # Lock setelah N votes non-Unknown
    ):
        self._track_to_jersey: Dict[int, str] = {}
        self._locked_tracks: set = set()  # Track ID yang sudah di-lock
        self._jersey_to_tracks: Dict[str, List[int]] = {}
        self._last_positions: Dict[int, Tuple[float, float]] = {}
        self._last_seen_frame: Dict[int, int] = {}
        self._active_track_per_jersey: Dict[str, int] = {}

        self._reassign_threshold = reassign_distance_threshold
        self._lost_timeout = lost_timeout_frames

        # Histori warna per track_id untuk voting stabil
        self._color_votes: Dict[int, List[str]] = {}
        self._color_vote_window = color_vote_window
        self._lock_after_votes = lock_after_votes

        if track_id_to_jersey:
            for tid, jersey in track_id_to_jersey.items():
                self._register_mapping(tid, jersey)
                self._locked_tracks.add(tid)  # Manual mapping = langsung lock

        print(f"[IDENTIFIER] Mapping jersey dimuat: {self._track_to_jersey}")
        print(f"[IDENTIFIER] Lock after {self._lock_after_votes} votes")
        print(f"[IDENTIFIER] Reassign threshold: {self._reassign_threshold}px")
        print(f"[IDENTIFIER] Lost timeout: {self._lost_timeout} frames")

    def _register_mapping(self, track_id: int, jersey: str) -> None:
        self._track_to_jersey[track_id] = jersey
        if jersey not in self._jersey_to_tracks:
            self._jersey_to_tracks[jersey] = []
        if track_id not in self._jersey_to_tracks[jersey]:
            self._jersey_to_tracks[jersey].append(track_id)
        self._active_track_per_jersey[jersey] = track_id

    def is_locked(self, track_id: int) -> bool:
        return track_id in self._locked_tracks

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.linalg.norm(
            np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
        ))

    # ============================================================
    # DETEKSI WARNA BAJU
    # ============================================================

    @staticmethod
    def detect_shirt_color(frame: np.ndarray, bbox: List[float]) -> str:
        """
        Deteksi warna baju dari area upper body pemain.
        Returns: "Merah", "Abu-Abu", atau "Unknown"
        """
        x1, y1, x2, y2 = map(int, bbox)
        h_box = y2 - y1
        shirt_y1 = y1 + int(h_box * 0.10)
        shirt_y2 = y1 + int(h_box * 0.45)
        margin_x = int((x2 - x1) * 0.15)
        shirt_x1 = max(0, x1 + margin_x)
        shirt_x2 = min(frame.shape[1], x2 - margin_x)

        if shirt_x2 <= shirt_x1 or shirt_y2 <= shirt_y1:
            return "Unknown"

        shirt_region = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
        if shirt_region.size == 0:
            return "Unknown"

        hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)

        # Deteksi MERAH
        mask_red1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([12, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 60, 50]), np.array([180, 255, 255]))
        mask_red = mask_red1 | mask_red2
        red_ratio = np.count_nonzero(mask_red) / max(mask_red.size, 1)

        # Deteksi ABU-ABU
        mask_gray = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 60, 180]))
        gray_ratio = np.count_nonzero(mask_gray) / max(mask_gray.size, 1)

        if red_ratio > 0.25:
            return "Merah"
        elif gray_ratio > 0.35:
            return "Abu-Abu"
        else:
            return "Unknown"

    def identify_players_by_color(
        self,
        frame: np.ndarray,
        frame_num: int,
        player_tracks: Dict[int, Dict],
        debug: bool = False
    ) -> None:
        """
        Identifikasi semua player di frame berdasarkan warna baju.
        Sekali mapping di-lock, tidak berubah lagi.
        """
        for tid, data in player_tracks.items():
            # Skip jika sudah di-lock
            if tid in self._locked_tracks:
                continue

            bbox = data.get('bbox')
            if bbox is None:
                continue

            color = self.detect_shirt_color(frame, bbox)

            # Simpan vote
            if tid not in self._color_votes:
                self._color_votes[tid] = []
            self._color_votes[tid].append(color)
            if len(self._color_votes[tid]) > self._color_vote_window:
                self._color_votes[tid].pop(0)

            # Voting: ambil warna paling sering (exclude "Unknown")
            votes = [v for v in self._color_votes[tid] if v != "Unknown"]
            if votes:
                best_color = Counter(votes).most_common(1)[0][0]
            else:
                best_color = "Unknown"

            # Update mapping
            current = self._track_to_jersey.get(tid, None)
            if current != best_color:
                self._register_mapping(tid, best_color)
                if debug and best_color != "Unknown":
                    print(f"[IDENTIFIER] Frame {frame_num}: "
                          f"Track {tid} -> {best_color} (warna baju)")

            # LOCK jika sudah cukup votes
            if best_color != "Unknown" and len(votes) >= self._lock_after_votes:
                self._locked_tracks.add(tid)
                if debug:
                    print(f"[IDENTIFIER] Frame {frame_num}: "
                          f"Track {tid} LOCKED sebagai '{best_color}' "
                          f"({len(votes)} votes)")

    # ============================================================
    # DYNAMIC RE-ID (handle track ID reset)
    # ============================================================

    def update_frame(
        self,
        frame_num: int,
        player_tracks: Dict[int, Dict],
        debug: bool = False
    ) -> None:
        current_track_ids = set(player_tracks.keys())

        for tid, data in player_tracks.items():
            bbox = data.get('bbox')
            if bbox is None:
                continue
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[3]
            self._last_positions[tid] = (cx, cy)
            self._last_seen_frame[tid] = frame_num

        new_track_ids = [
            tid for tid in current_track_ids
            if tid not in self._track_to_jersey
        ]

        if not new_track_ids:
            return

        lost_jerseys: Dict[str, Tuple[float, float]] = {}
        for jersey, active_tid in list(self._active_track_per_jersey.items()):
            if jersey == "Unknown":
                continue
            if active_tid in current_track_ids:
                continue
            last_frame = self._last_seen_frame.get(active_tid, -999)
            frames_since_lost = frame_num - last_frame
            if frames_since_lost > self._lost_timeout:
                continue
            last_pos = self._last_positions.get(active_tid)
            if last_pos is not None:
                lost_jerseys[jersey] = last_pos

        if not lost_jerseys:
            for tid in new_track_ids:
                if tid not in self._track_to_jersey:
                    self._register_mapping(tid, "Unknown")
            return

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

        assignments.sort(key=lambda x: x[2])
        assigned_jerseys = set()
        for tid, jersey, dist in assignments:
            if jersey in assigned_jerseys:
                continue
            self._register_mapping(tid, jersey)
            # Lock juga re-assigned track
            self._locked_tracks.add(tid)
            assigned_jerseys.add(jersey)
            if debug:
                print(f"[IDENTIFIER] Frame {frame_num}: "
                      f"RE-ID Track {tid} -> {jersey} (jarak={dist:.0f}px, LOCKED)")

        for tid in new_track_ids:
            if tid not in self._track_to_jersey:
                self._register_mapping(tid, "Unknown")

    # ============================================================
    # PUBLIC API
    # ============================================================

    def set_mapping(self, track_id: int, jersey: str, lock: bool = True) -> None:
        self._register_mapping(track_id, jersey)
        if lock:
            self._locked_tracks.add(track_id)
        print(f"[IDENTIFIER] Mapping manual: Track ID {track_id} -> {jersey} "
              f"(locked={lock})")

    def get_jersey_number_for_player(self, track_id: int) -> str:
        return self._track_to_jersey.get(track_id, f"ID:{track_id}")

    def get_track_id_for_jersey(self, jersey: str) -> Optional[int]:
        return self._active_track_per_jersey.get(jersey, None)

    def is_same_player(self, track_id_a: int, track_id_b: int) -> bool:
        jersey_a = self.get_jersey_number_for_player(track_id_a)
        jersey_b = self.get_jersey_number_for_player(track_id_b)
        if jersey_a.startswith("ID:") or jersey_b.startswith("ID:"):
            return track_id_a == track_id_b
        if jersey_a == "Unknown" or jersey_b == "Unknown":
            return track_id_a == track_id_b
        return jersey_a == jersey_b

    def get_all_mappings(self) -> Dict[int, str]:
        return dict(self._track_to_jersey)

    def get_all_track_ids_for_jersey(self, jersey: str) -> List[int]:
        return list(self._jersey_to_tracks.get(jersey, []))

    def print_mappings(self) -> None:
        print("\n[IDENTIFIER] === PLAYER MAPPINGS ===")
        if not self._track_to_jersey:
            print("[IDENTIFIER] Belum ada mapping.")
        for tid, jersey in sorted(self._track_to_jersey.items()):
            active_marker = ""
            if self._active_track_per_jersey.get(jersey) == tid:
                active_marker = " (AKTIF)"
            lock_marker = " [LOCKED]" if tid in self._locked_tracks else ""
            print(f"[IDENTIFIER]   Track ID {tid:3d} -> {jersey}"
                  f"{active_marker}{lock_marker}")
        print("[IDENTIFIER] ========================\n")
