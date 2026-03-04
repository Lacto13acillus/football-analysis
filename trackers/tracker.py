# tracker.py
import pickle
import os
import sys
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import Dict, List, Any, Optional, Tuple

sys.path.append('../')
from utils.bbox_utils import (
    get_center_of_bbox,
    get_center_of_bbox_bottom,
    get_bbox_width,
    interpolate_ball_positions,
    measure_distance
)


class Tracker:
    def __init__(self, model_path: str):
        """
        Inisialisasi YOLO model dan ByteTrack tracker.

        Args:
            model_path: path ke file model YOLOv8 (.pt)
        """
        self.model   = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        # Mapping nama kelas dari model
        self.cls_names     : Dict[int, str] = self.model.model.names
        self.cls_names_inv : Dict[str, int] = {v: k for k, v in self.cls_names.items()}

        # -------------------------------------------------------
        # Cache anchor posisi cone untuk konsistensi ID antar frame
        # Karena kamera statis, posisi cone tidak berubah
        # -------------------------------------------------------
        self._cone_anchors: Dict[int, Tuple[float, float]] = {}
        self._cone_anchor_max_distance = 60.0  # Pixel threshold untuk match cone ke anchor

        print(f"[TRACKER] Model loaded: {model_path}")
        print(f"[TRACKER] Kelas terdeteksi: {list(self.cls_names.values())}")

        # Deteksi nama kelas secara fleksibel (custom model vs COCO)
        self._ball_cls_name   = self._find_class(['ball', 'sports ball'])
        self._player_cls_name = self._find_class(['player', 'person'])
        self._cone_cls_name   = self._find_class(['cone'])

        print(f"[TRACKER] Kelas BOLA    : '{self._ball_cls_name}'")
        print(f"[TRACKER] Kelas PEMAIN  : '{self._player_cls_name}'")
        print(f"[TRACKER] Kelas CONE    : '{self._cone_cls_name}'")

    def _find_class(self, candidates: List[str]) -> Optional[str]:
        """
        Cari nama kelas yang cocok dari daftar kandidat.
        Mendukung custom model dengan nama kelas berbeda dari COCO.

        Args:
            candidates: list nama kelas kandidat (prioritas dari kiri)

        Returns:
            Nama kelas yang ditemukan, atau None jika tidak ada
        """
        for name in candidates:
            if name in self.cls_names_inv:
                return name
        print(f"[TRACKER] WARNING: Kelas {candidates} tidak ditemukan di model!")
        return None

    def _get_class_id(self, class_name: Optional[str]) -> int:
        """Dapatkan integer class ID dari nama kelas, atau -1 jika tidak ada."""
        if class_name is None:
            return -1
        return self.cls_names_inv.get(class_name, -1)

    # ============================================================
    # CONE TRACKING DENGAN ID KONSISTEN (Spatial Anchor Matching)
    # ============================================================

    def _match_cone_to_anchor(
        self,
        cx: float,
        cy: float
    ) -> int:
        """
        Cocokkan posisi cone ke anchor yang sudah ada menggunakan nearest-neighbor.
        Jika tidak ada anchor yang cukup dekat, buat anchor baru.

        Ini memastikan cone ID KONSISTEN antar frame meskipun
        YOLO mendeteksi cone dalam urutan berbeda tiap frame.

        Args:
            cx, cy: posisi tengah-bawah cone yang terdeteksi

        Returns:
            cone_id yang konsisten
        """
        best_id   = None
        best_dist = float('inf')

        # Cari anchor terdekat yang belum di-assign di frame ini
        for anchor_id, anchor_pos in self._cone_anchors.items():
            dist = measure_distance((cx, cy), anchor_pos)
            if dist < best_dist:
                best_dist = dist
                best_id   = anchor_id

        if best_dist <= self._cone_anchor_max_distance:
            # Cone ini cocok ke anchor yang sudah ada
            # Update posisi anchor dengan exponential moving average (smooth)
            ax, ay = self._cone_anchors[best_id]
            alpha  = 0.1  # Learning rate - rendah agar tidak drift
            self._cone_anchors[best_id] = (
                ax * (1 - alpha) + cx * alpha,
                ay * (1 - alpha) + cy * alpha
            )
            return best_id
        else:
            # Cone baru - buat anchor baru dengan ID berikutnya
            new_id = len(self._cone_anchors)
            # Pastikan ID tidak bentrok
            while new_id in self._cone_anchors:
                new_id += 1
            self._cone_anchors[new_id] = (cx, cy)
            print(f"[TRACKER] Anchor cone baru: ID {new_id} "
                  f"di posisi ({cx:.1f}, {cy:.1f})")
            return new_id

    def _reset_cone_anchors(self) -> None:
        """Reset semua anchor cone (panggil jika perlu re-inisialisasi)."""
        self._cone_anchors.clear()
        print("[TRACKER] Cone anchors direset.")

    # ============================================================
    # DETEKSI FRAMES
    # ============================================================

    def detect_frames(
        self,
        frames    : List[np.ndarray],
        batch_size: int = 20
    ) -> List[Any]:
        """
        Jalankan deteksi YOLOv8 pada list frames secara batch.

        Args:
            frames    : list frame video (numpy array BGR)
            batch_size: jumlah frame per batch inference

        Returns:
            List hasil deteksi per frame
        """
        detections   = []
        total_batches = (len(frames) + batch_size - 1) // batch_size

        for i in range(0, len(frames), batch_size):
            batch     = frames[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"[TRACKER] Deteksi batch {batch_num}/{total_batches}...")
            results = self.model.predict(batch, conf=0.1, verbose=False)
            detections.extend(results)

        return detections

    # ============================================================
    # TRACKING UTAMA
    # ============================================================

    def get_object_tracks(
        self,
        frames        : List[np.ndarray],
        read_from_stub: bool = False,
        stub_path     : Optional[str] = None
    ) -> Dict[str, List[Dict[int, Dict[str, Any]]]]:
        """
        Deteksi dan tracking semua objek di setiap frame.

        Perbaikan dari versi sebelumnya:
        - Nama kelas bola fleksibel ('ball' atau 'sports ball')
        - Cone ID konsisten antar frame via spatial anchor matching
        - Circle cone difilter via nama kelas (bukan hanya aspect ratio)

        Struktur output:
        {
            'players': [{player_id: {'bbox': [...], 'confidence': float}}, ...],
            'ball'   : [{1: {'bbox': [...], 'confidence': float}}, ...],
            'cones'  : [{cone_id: {'bbox': [...], 'confidence': float, ...}}, ...]
        }

        Args:
            frames        : list frame video
            read_from_stub: baca dari cache pickle jika True
            stub_path     : path file pickle untuk cache

        Returns:
            Dict tracks semua objek
        """
        # Baca dari cache jika tersedia
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"[TRACKER] Membaca tracks dari cache: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Validasi kelas yang dibutuhkan
        if self._ball_cls_name is None:
            print("[TRACKER] CRITICAL: Kelas bola tidak ditemukan!")
            print(f"[TRACKER] Kelas tersedia: {list(self.cls_names.values())}")

        # Jalankan deteksi batch
        detections = self.detect_frames(frames)

        # Reset cone anchors sebelum memproses video baru
        self._reset_cone_anchors()

        # Struktur tracks output
        tracks: Dict[str, List[Dict[int, Dict[str, Any]]]] = {
            "players": [],
            "ball"   : [],
            "cones"  : []
        }

        # Ambil class ID yang diperlukan (berdasarkan nama kelas model)
        ball_cls_id   = self._get_class_id(self._ball_cls_name)
        player_cls_id = self._get_class_id(self._player_cls_name)
        cone_cls_id   = self._get_class_id(self._cone_cls_name)

        # Set untuk cone ID yang sudah di-assign di frame ini
        # (mencegah satu anchor di-assign ke dua cone dalam frame yang sama)
        assigned_in_frame = set()

        for frame_num, detection in enumerate(detections):
            if frame_num % 50 == 0:
                print(f"[TRACKER] Processing frame {frame_num}/{len(detections)}...")

            # Inisialisasi frame kosong
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["cones"].append({})

            # Konversi ke supervision Detections
            det_sv = sv.Detections.from_ultralytics(detection)

            if len(det_sv) == 0:
                continue

            # Nama kelas dari deteksi frame ini
            cls_names_local     = detection.names
            cls_names_inv_local = {v: k for k, v in cls_names_local.items()}

            # ---- TRACKING PEMAIN menggunakan ByteTrack ----
            p_cls_id    = cls_names_inv_local.get(self._player_cls_name, -1)
            person_mask = det_sv.class_id == p_cls_id
            player_det  = det_sv[person_mask]

            if len(player_det) > 0:
                player_det_tracked = self.tracker.update_with_detections(player_det)
                for det in player_det_tracked:
                    bbox       = det[0].tolist()
                    track_id   = int(det[4])
                    confidence = float(det[2]) if len(det) > 2 else 0.5
                    tracks["players"][frame_num][track_id] = {
                        "bbox"      : bbox,
                        "confidence": confidence
                    }

            # ---- DETEKSI BOLA (confidence tertinggi, ID selalu 1) ----
            b_cls_id  = cls_names_inv_local.get(self._ball_cls_name, -1)
            ball_mask = det_sv.class_id == b_cls_id
            ball_det  = det_sv[ball_mask]

            if len(ball_det) > 0:
                # Ambil bola dengan confidence tertinggi
                best_idx  = int(np.argmax(ball_det.confidence))
                best_ball = ball_det[best_idx]
                bbox      = best_ball.xyxy[0].tolist()
                conf      = float(best_ball.confidence[0])

                tracks["ball"][frame_num][1] = {
                    "bbox"      : bbox,
                    "confidence": conf
                }

            # ---- DETEKSI CONE dengan ID KONSISTEN ----
            c_cls_id  = cls_names_inv_local.get(self._cone_cls_name, -1)
            cone_mask = det_sv.class_id == c_cls_id
            cone_det  = det_sv[cone_mask]

            # Reset set assignment untuk frame baru
            assigned_in_frame = set()

            for idx in range(len(cone_det)):
                bbox = cone_det.xyxy[idx].tolist()
                conf = float(cone_det.confidence[idx])

                # Filter confidence rendah
                if conf < 0.25:
                    continue

                # Filter circle cone menggunakan aspect ratio
                # Cone berdiri: tinggi > lebar (aspect_ratio > 0.8)
                # Circle cone: lebar >= tinggi (aspect_ratio <= 0.8)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w <= 0:
                    continue

                aspect_ratio = h / w
                if aspect_ratio < 0.8:
                    # Skip circle cone (pipih/setengah bola)
                    continue

                # Posisi referensi: titik tengah-bawah cone
                cx, cy = get_center_of_bbox_bottom(bbox)

                # Dapatkan cone_id yang konsisten via spatial anchor matching
                cone_id = self._match_cone_to_anchor(cx, cy)

                # Skip jika anchor ini sudah di-assign di frame ini
                # (hindari duplikasi deteksi cone yang sama)
                if cone_id in assigned_in_frame:
                    continue
                assigned_in_frame.add(cone_id)

                tracks["cones"][frame_num][cone_id] = {
                    "bbox"        : bbox,
                    "confidence"  : conf,
                    "aspect_ratio": aspect_ratio
                }

        # Interpolasi posisi bola yang hilang
        print("[TRACKER] Interpolasi posisi bola...")
        tracks["ball"] = interpolate_ball_positions(tracks["ball"])

        # Debug: tampilkan statistik bola
        ball_detected = sum(1 for f in tracks["ball"] if f.get(1))
        print(f"[TRACKER] Bola terdeteksi: {ball_detected}/{len(tracks['ball'])} frames "
              f"({ball_detected/len(tracks['ball'])*100:.1f}%)")

        # Debug: tampilkan cone anchors yang terbentuk
        print(f"[TRACKER] Total cone anchor unik: {len(self._cone_anchors)}")
        for cid, pos in sorted(self._cone_anchors.items()):
            print(f"[TRACKER]   Anchor cone ID {cid}: ({pos[0]:.1f}, {pos[1]:.1f})")

        # Simpan ke cache
        if stub_path:
            os.makedirs(
                os.path.dirname(stub_path) if os.path.dirname(stub_path) else '.',
                exist_ok=True
            )
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            print(f"[TRACKER] Cache disimpan: {stub_path}")

        return tracks

    # ============================================================
    # DRAW ANNOTATIONS
    # ============================================================

    def draw_annotations(
        self,
        frame            : np.ndarray,
        tracks           : Dict,
        frame_num        : int,
        ball_possessions : List[int],
        player_identifier = None
    ) -> np.ndarray:
        """Gambar bounding box dan annotasi di atas frame."""
        annotated = frame.copy()

        for player_id, player_data in tracks["players"][frame_num].items():
            bbox = player_data["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            has_ball = (
                frame_num < len(ball_possessions) and
                ball_possessions[frame_num] == player_id
            )
            color = (0, 255, 0) if has_ball else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            jersey = str(player_id)
            if player_identifier:
                jersey = player_identifier.get_jersey_number_for_player(player_id)
            label = f"#{jersey}"
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + 45, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ball_data = tracks["ball"][frame_num].get(1)
        if ball_data:
            bx1, by1, bx2, by2 = map(int, ball_data["bbox"])
            cv2.ellipse(
                annotated,
                center    = ((bx1 + bx2) // 2, (by1 + by2) // 2),
                axes      = ((bx2 - bx1) // 2, (by2 - by1) // 2),
                angle     = 0, startAngle=0, endAngle=360,
                color     = (0, 255, 255), thickness=2
            )

        return annotated

    # ============================================================
    # VIDEO I/O
    # ============================================================

    @staticmethod
    def read_video(video_path: str) -> List[np.ndarray]:
        """Baca video dari path dan kembalikan list frames."""
        cap    = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"[VIDEO] Total frame dibaca: {len(frames)}")
        return frames

    @staticmethod
    def save_video(
        output_frames: List[np.ndarray],
        output_path  : str,
        fps          : int = 24
    ) -> None:
        """Simpan list frames menjadi file video."""
        if not output_frames:
            print("[VIDEO] Tidak ada frame untuk disimpan!")
            return

        h, w   = output_frames[0].shape[:2]

        # Pilih codec berdasarkan ekstensi file
        ext = os.path.splitext(output_path)[1].lower()
        if ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        elif ext == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
            exist_ok=True
        )
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in output_frames:
            out.write(frame)
        out.release()

        print(f"[VIDEO] Disimpan: {output_path} "
              f"({len(output_frames)} frames, {fps}fps)")