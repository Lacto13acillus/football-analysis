# tracker.py
# Menjalankan YOLOv8 untuk deteksi objek dan ByteTrack untuk tracking pemain.
# Deteksi: 'person', 'sports ball', 'cone' (standing cone saja, bukan circle cone)

import pickle
import os
import sys
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import Dict, List, Any, Optional

sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, interpolate_ball_positions


class Tracker:
    def __init__(self, model_path: str):
        """
        Inisialisasi YOLO model dan ByteTrack tracker.

        Args:
            model_path: path ke file model YOLOv8 (.pt)
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        # Mapping nama kelas dari YOLO ke index
        self.cls_names: Dict[int, str] = self.model.model.names
        self.cls_names_inv: Dict[str, int] = {v: k for k, v in self.cls_names.items()}

        print(f"[TRACKER] Model loaded: {model_path}")
        print(f"[TRACKER] Kelas yang dideteksi: {list(self.cls_names.values())}")

    def detect_frames(self, frames: List[np.ndarray], batch_size: int = 20) -> List[Any]:
        """
        Jalankan deteksi YOLOv8 pada list frames secara batch.

        Args:
            frames    : list frame video (numpy array BGR)
            batch_size: jumlah frame per batch inference

        Returns:
            List hasil deteksi per frame
        """
        detections = []
        total_batches = (len(frames) + batch_size - 1) // batch_size

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"[TRACKER] Deteksi batch {batch_num}/{total_batches}...")

            results = self.model.predict(batch, conf=0.1, verbose=False)
            detections.extend(results)

        return detections

    def get_object_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None
    ) -> Dict[str, List[Dict[int, Dict[str, Any]]]]:
        """
        Deteksi dan tracking semua objek di setiap frame.

        Struktur output tracks:
        {
            'players': [{player_id: {'bbox': [...], 'confidence': float}}, ...],
            'ball'   : [{1: {'bbox': [...], 'confidence': float}}, ...],
            'cones'  : [{cone_id: {'bbox': [...], 'confidence': float}}, ...]
        }

        Args:
            frames        : list frame video
            read_from_stub: jika True, baca hasil dari cache pickle
            stub_path     : path file pickle untuk cache

        Returns:
            Dict tracks berisi data tracking semua objek
        """
        # Baca dari cache jika tersedia
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"[TRACKER] Membaca tracks dari cache: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Jalankan deteksi
        detections = self.detect_frames(frames)

        # Struktur tracks
        tracks: Dict[str, List[Dict[int, Dict[str, Any]]]] = {
            "players": [],
            "ball"   : [],
            "cones"  : []
        }

        # ID kelas yang dibutuhkan
        person_cls_id   = self.cls_names_inv.get("person", -1)
        ball_cls_id     = self.cls_names_inv.get("sports ball", -1)
        cone_cls_id     = self.cls_names_inv.get("cone", -1)

        for frame_num, detection in enumerate(detections):
            if frame_num % 50 == 0:
                print(f"[TRACKER] Processing frame {frame_num}/{len(detections)}...")

            cls_names = detection.names
            cls_names_inv_local = {v: k for k, v in cls_names.items()}

            # Inisialisasi frame kosong
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["cones"].append({})

            # Konversi ke supervision Detection object
            det_sv = sv.Detections.from_ultralytics(detection)

            # ---- TRACKING PEMAIN (menggunakan ByteTrack) ----
            person_mask = det_sv.class_id == cls_names_inv_local.get("person", -1)
            player_det = det_sv[person_mask]

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

            # ---- TRACKING BOLA (ID selalu 1, ambil confidence tertinggi) ----
            ball_mask = det_sv.class_id == cls_names_inv_local.get("sports ball", -1)
            ball_det = det_sv[ball_mask]

            if len(ball_det) > 0:
                # Ambil deteksi bola dengan confidence tertinggi
                best_idx = int(np.argmax(ball_det.confidence))
                best_ball = ball_det[best_idx]
                bbox = best_ball.xyxy[0].tolist()
                conf = float(best_ball.confidence[0])

                tracks["ball"][frame_num][1] = {
                    "bbox"      : bbox,
                    "confidence": conf
                }

            # ---- DETEKSI CONE (tanpa ByteTrack, posisi statis) ----
            cone_mask = det_sv.class_id == cls_names_inv_local.get("cone", -1)
            cone_det = det_sv[cone_mask]

            cone_static_id = 0
            for idx in range(len(cone_det)):
                bbox = cone_det.xyxy[idx].tolist()
                conf = float(cone_det.confidence[idx])

                # Filter confidence rendah
                if conf < 0.25:
                    continue

                # Filter circle cone berdasarkan aspect ratio
                # Cone berdiri: tinggi > lebar (aspect_ratio > 0.8)
                # Circle cone: lebar > tinggi (aspect_ratio < 0.6)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w <= 0:
                    continue
                aspect_ratio = h / w

                if aspect_ratio < 0.8:
                    # Skip circle cone (pipih/setengah bola)
                    continue

                tracks["cones"][frame_num][cone_static_id] = {
                    "bbox"        : bbox,
                    "confidence"  : conf,
                    "aspect_ratio": aspect_ratio
                }
                cone_static_id += 1

        # Interpolasi posisi bola yang hilang
        print("[TRACKER] Interpolasi posisi bola...")
        tracks["ball"] = interpolate_ball_positions(tracks["ball"])

        # Simpan ke cache jika diminta
        if stub_path:
            os.makedirs(os.path.dirname(stub_path) if os.path.dirname(stub_path) else '.', exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            print(f"[TRACKER] Tracks disimpan ke cache: {stub_path}")

        return tracks

    def draw_annotations(
        self,
        frame: np.ndarray,
        tracks: Dict,
        frame_num: int,
        ball_possessions: List[int],
        player_identifier=None
    ) -> np.ndarray:
        """
        Gambar bounding box dan annotasi di atas frame.

        Args:
            frame            : frame video asli
            tracks           : dict hasil tracking
            frame_num        : nomor frame saat ini
            ball_possessions : list possession per frame
            player_identifier: objek PlayerIdentifier untuk label jersey

        Returns:
            Frame dengan annotasi
        """
        annotated = frame.copy()

        # ---- Gambar bounding box PEMAIN ----
        for player_id, player_data in tracks["players"][frame_num].items():
            bbox = player_data["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            # Tentukan warna berdasarkan possession
            has_ball = (
                frame_num < len(ball_possessions) and
                ball_possessions[frame_num] == player_id
            )
            color = (0, 255, 0) if has_ball else (255, 0, 0)  # Hijau jika punya bola

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label jersey
            jersey = str(player_id)
            if player_identifier:
                jersey = player_identifier.get_jersey_number_for_player(player_id)

            label = f"#{jersey}"
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + 40, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ---- Gambar bola ----
        ball_data = tracks["ball"][frame_num].get(1)
        if ball_data:
            bx1, by1, bx2, by2 = map(int, ball_data["bbox"])
            cv2.ellipse(
                annotated,
                center=((bx1 + bx2) // 2, (by1 + by2) // 2),
                axes=((bx2 - bx1) // 2, (by2 - by1) // 2),
                angle=0, startAngle=0, endAngle=360,
                color=(0, 255, 255), thickness=2
            )

        return annotated

    @staticmethod
    def read_video(video_path: str) -> List[np.ndarray]:
        """Baca video dari path dan kembalikan list frames."""
        cap = cv2.VideoCapture(video_path)
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
        output_path: str,
        fps: int = 24
    ) -> None:
        """Simpan list frames menjadi file video MP4."""
        if not output_frames:
            print("[VIDEO] Tidak ada frame untuk disimpan!")
            return

        h, w = output_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in output_frames:
            out.write(frame)
        out.release()

        print(f"[VIDEO] Video disimpan: {output_path} ({len(output_frames)} frames, {fps}fps)")