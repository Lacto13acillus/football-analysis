# trackers/es_batu_tracker.py
# Tracker khusus untuk proyek ES BATU.
#
# Class mapping dari data.yaml:
#   0: es-batu
#   1: motor
#   2: orang
#   3: truk

import os
import pickle
import numpy as np
import cv2
from ultralytics import YOLO
from typing import List, Dict, Optional, Any

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False
    print("[ES_BATU_TRACKER] WARNING: supervision belum terinstall. "
          "Tracking akan menggunakan mode sederhana (tanpa ByteTrack).")


# ============================================================
# CLASS ID — sesuai data.yaml es_batu
# ============================================================
CLASS_ES_BATU = 0
CLASS_MOTOR   = 1
CLASS_ORANG   = 2
CLASS_TRUK    = 3

CLASS_NAMES = {
    CLASS_ES_BATU : 'es-batu',
    CLASS_MOTOR   : 'motor',
    CLASS_ORANG   : 'orang',
    CLASS_TRUK    : 'truk',
}


class EsBatuTracker:
    """
    Tracker untuk deteksi & tracking objek es batu, motor, orang, truk.

    Tracks format output:
    {
        'es_batu' : [{id: {'bbox': [x1,y1,x2,y2], 'conf': float}, ...}, ...],
        'motor'   : [{id: {'bbox': [x1,y1,x2,y2], 'conf': float}, ...}, ...],
        'orang'   : [{id: {'bbox': [x1,y1,x2,y2], 'conf': float}, ...}, ...],
        'truk'    : [{id: {'bbox': [x1,y1,x2,y2], 'conf': float}, ...}, ...],
    }
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.30,
        iou_threshold : float = 0.45,
    ):
        self.model          = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold

        if HAS_SUPERVISION:
            # Tracker terpisah per class supaya ID tidak bertabrakan
            self.tracker_es_batu = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.7,
                frame_rate=30,
            )
            self.tracker_orang = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
            self.tracker_motor = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=60,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
            self.tracker_truk = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=60,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
        else:
            self.tracker_es_batu = None
            self.tracker_orang   = None
            self.tracker_motor   = None
            self.tracker_truk    = None

        print(f"[ES_BATU_TRACKER] Model loaded  : {model_path}")
        print(f"[ES_BATU_TRACKER] Conf threshold: {self.conf_threshold}")
        print(f"[ES_BATU_TRACKER] IoU  threshold: {self.iou_threshold}")
        print(f"[ES_BATU_TRACKER] ByteTrack     : "
              f"{'Aktif' if HAS_SUPERVISION else 'Tidak aktif'}")

    # -------------------------------------------------------
    # BACA & SIMPAN VIDEO
    # -------------------------------------------------------

    @staticmethod
    def read_video(video_path: str) -> List[np.ndarray]:
        """Baca semua frame dari video."""
        if not os.path.exists(video_path):
            print(f"[ES_BATU_TRACKER] ERROR: Video tidak ditemukan: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ES_BATU_TRACKER] ERROR: Tidak bisa membuka video: {video_path}")
            return []

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        print(f"[ES_BATU_TRACKER] Video dibaca: {len(frames)} frames dari {video_path}")
        return frames

    @staticmethod
    def save_video(
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 30,
    ) -> None:
        """Simpan list frame menjadi video."""
        if not frames:
            print("[ES_BATU_TRACKER] WARNING: Tidak ada frame untuk disimpan!")
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"[ES_BATU_TRACKER] Video disimpan: {output_path} "
              f"({len(frames)} frames, {fps} FPS)")

    # -------------------------------------------------------
    # DETEKSI YOLO
    # -------------------------------------------------------

    def _detect_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 16,
    ) -> List[Any]:
        """Jalankan deteksi YOLO pada semua frame."""
        detections = []
        total = len(frames)
        print(f"[ES_BATU_TRACKER] Menjalankan deteksi YOLO pada {total} frames...")

        for i in range(0, total, batch_size):
            batch = frames[i:i + batch_size]
            results = self.model.predict(
                batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            detections.extend(results)

            if (i // batch_size) % 5 == 0:
                pct = min(i + batch_size, total) / total * 100
                print(f"[ES_BATU_TRACKER] Deteksi progress: "
                      f"{min(i + batch_size, total)}/{total} ({pct:.1f}%)")

        print(f"[ES_BATU_TRACKER] Deteksi selesai: {len(detections)} frames.")
        return detections

    # -------------------------------------------------------
    # HELPER: track satu class dengan ByteTrack
    # -------------------------------------------------------

    def _track_class(
        self,
        bboxes: List[List[float]],
        confs : List[float],
        tracker,           # sv.ByteTrack instance atau None
        fallback_start: int = 1,
    ) -> Dict[int, Dict]:
        """
        Jalankan tracking untuk satu class.
        Kembalikan dict: {tracker_id: {'bbox': [...], 'conf': float}}
        """
        result = {}
        if not bboxes:
            return result

        if HAS_SUPERVISION and tracker is not None:
            sv_det = sv.Detections(
                xyxy=np.array(bboxes, dtype=np.float32),
                confidence=np.array(confs, dtype=np.float32),
            )
            sv_tracked = tracker.update_with_detections(sv_det)
            for j in range(len(sv_tracked)):
                tid  = int(sv_tracked.tracker_id[j])
                bbox = sv_tracked.xyxy[j].tolist()
                conf = float(sv_tracked.confidence[j])
                result[tid] = {'bbox': bbox, 'conf': conf}
        else:
            # Fallback tanpa tracking
            for j, (bbox, conf) in enumerate(zip(bboxes, confs)):
                result[j + fallback_start] = {'bbox': bbox, 'conf': conf}

        return result

    # -------------------------------------------------------
    # KONVERSI DETEKSI → TRACKS
    # -------------------------------------------------------

    def _detections_to_tracks(
        self,
        detections: List[Any],
    ) -> Dict[str, List[Dict]]:
        """
        Konversi hasil deteksi YOLO menjadi tracks dictionary.
        """
        tracks: Dict[str, List[Dict]] = {
            'es_batu': [],
            'motor'  : [],
            'orang'  : [],
            'truk'   : [],
        }
        total = len(detections)

        for frame_num, detection in enumerate(detections):
            if frame_num % 100 == 0:
                print(f"[ES_BATU_TRACKER] Konversi tracks: frame {frame_num}/{total}...")

            boxes = detection.boxes

            es_batu_bboxes: List = []
            es_batu_confs : List = []
            motor_bboxes  : List = []
            motor_confs   : List = []
            orang_bboxes  : List = []
            orang_confs   : List = []
            truk_bboxes   : List = []
            truk_confs    : List = []

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls  = int(boxes.cls[i].cpu().numpy())

                if cls == CLASS_ES_BATU:
                    es_batu_bboxes.append(bbox)
                    es_batu_confs.append(conf)
                elif cls == CLASS_MOTOR:
                    motor_bboxes.append(bbox)
                    motor_confs.append(conf)
                elif cls == CLASS_ORANG:
                    orang_bboxes.append(bbox)
                    orang_confs.append(conf)
                elif cls == CLASS_TRUK:
                    truk_bboxes.append(bbox)
                    truk_confs.append(conf)

            tracks['es_batu'].append(
                self._track_class(es_batu_bboxes, es_batu_confs, self.tracker_es_batu)
            )
            tracks['motor'].append(
                self._track_class(motor_bboxes, motor_confs, self.tracker_motor)
            )
            tracks['orang'].append(
                self._track_class(orang_bboxes, orang_confs, self.tracker_orang)
            )
            tracks['truk'].append(
                self._track_class(truk_bboxes, truk_confs, self.tracker_truk)
            )

        print(f"[ES_BATU_TRACKER] Konversi selesai: {total} frames diproses.")
        self._print_summary(tracks)
        return tracks

    def _print_summary(self, tracks: Dict) -> None:
        total = len(tracks['es_batu'])
        for key in ('es_batu', 'motor', 'orang', 'truk'):
            frames_with = sum(1 for f in tracks[key] if len(f) > 0)
            all_ids = set()
            for f in tracks[key]:
                all_ids.update(f.keys())
            print(f"[ES_BATU_TRACKER] {key:10s}: "
                  f"{frames_with}/{total} frames terdeteksi, "
                  f"unique IDs: {len(all_ids)}")

    # -------------------------------------------------------
    # MAIN PIPELINE
    # -------------------------------------------------------

    def get_object_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str = "stubs/es_batu_cache.pkl",
    ) -> Dict[str, List[Dict]]:
        """
        Pipeline utama: deteksi + tracking semua objek es batu.

        Returns:
            tracks: {
                'es_batu': [{id: {'bbox': [...], 'conf': float}, ...}, ...],
                'motor'  : [...],
                'orang'  : [...],
                'truk'   : [...],
            }
        """
        # Cek cache
        if read_from_stub and os.path.exists(stub_path):
            print(f"[ES_BATU_TRACKER] Membaca cache dari: {stub_path}")
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            self._print_summary(tracks)
            return tracks

        # Deteksi
        detections = self._detect_frames(frames)

        # Konversi
        tracks = self._detections_to_tracks(detections)

        # Simpan cache
        stub_dir = os.path.dirname(stub_path)
        if stub_dir and not os.path.exists(stub_dir):
            os.makedirs(stub_dir, exist_ok=True)

        with open(stub_path, 'wb') as f:
            pickle.dump(tracks, f)
        print(f"[ES_BATU_TRACKER] Cache disimpan ke: {stub_path}")

        return tracks
