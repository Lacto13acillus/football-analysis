# trackers/tracker.py
# Tracker untuk heading detection project.
# Mendukung 3 class dari YOLO:
#   0: Heading (kepala pemain)
#   1: ball
#   2: player

import os
import sys
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional, Any

sys.path.append('../')

# ============================================================
# Supervision untuk tracking (opsional, fallback manual jika tidak ada)
# ============================================================
try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False
    print("[TRACKER] WARNING: supervision belum terinstall. "
          "Tracking akan menggunakan mode sederhana (tanpa ByteTrack).")


class Tracker:
    """
    Tracker utama: deteksi objek dengan YOLO + tracking per frame.

    Class mapping default (sesuai data.yaml heading):
        0 → Heading (bbox kepala pemain)
        1 → ball
        2 → player

    Bisa di-override via parameter class_mapping untuk model lain.
    Contoh longpass (2 class): {'ball': 0, 'player': 1}
    """

    # Class ID mapping (default heading project)
    CLASS_HEADING = 0
    CLASS_BALL    = 1
    CLASS_PLAYER  = 2
    CLASS_CONE    = -1  # -1 = tidak ada (default)

    CLASS_NAMES = {
        0: 'Heading',
        1: 'ball',
        2: 'player',
    }

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        class_mapping: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            model_path: Path ke model YOLO (.pt)
            conf_threshold: Confidence threshold untuk deteksi
            iou_threshold: IoU threshold untuk NMS
            class_mapping: Override class ID mapping.
                           Dict dari nama objek ke class ID.
                           Contoh heading (default): {'heading': 0, 'ball': 1, 'player': 2}
                           Contoh longpass: {'ball': 0, 'player': 1}
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Apply class mapping
        if class_mapping:
            self.CLASS_BALL = class_mapping.get('ball', self.CLASS_BALL)
            self.CLASS_PLAYER = class_mapping.get('player', self.CLASS_PLAYER)
            self.CLASS_HEADING = class_mapping.get('heading', -1)  # -1 = tidak ada
            self.CLASS_CONE = class_mapping.get('cone', -1)  # -1 = tidak ada
        self.has_heading_class = (self.CLASS_HEADING >= 0 and
                                  (class_mapping is None or 'heading' in class_mapping))
        self.has_cone_class = (self.CLASS_CONE >= 0)

        # Inisialisasi ByteTrack tracker (supervision) untuk player
        if HAS_SUPERVISION:
            self.player_tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
            # Tracker terpisah untuk heading (kepala) agar ID-nya independen
            self.heading_tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=15,
                minimum_matching_threshold=0.7,
                frame_rate=30,
            )
            # Tracker terpisah untuk cone
            self.cone_tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=60,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
        else:
            self.player_tracker = None
            self.heading_tracker = None
            self.cone_tracker = None

        print(f"[TRACKER] Model loaded: {model_path}")
        print(f"[TRACKER] Confidence threshold: {self.conf_threshold}")
        print(f"[TRACKER] IoU threshold: {self.iou_threshold}")
        print(f"[TRACKER] ByteTrack: {'Aktif' if HAS_SUPERVISION else 'Tidak aktif'}")
        print(f"[TRACKER] Class mapping: ball={self.CLASS_BALL}, player={self.CLASS_PLAYER}, "
              f"heading={'N/A' if not self.has_heading_class else self.CLASS_HEADING}, "
              f"cone={'N/A' if not self.has_cone_class else self.CLASS_CONE}")

    # ============================================================
    # BACA & SIMPAN VIDEO
    # ============================================================

    @staticmethod
    def read_video(video_path: str) -> List[np.ndarray]:
        """Baca semua frame dari video."""
        if not os.path.exists(video_path):
            print(f"[TRACKER] ERROR: Video tidak ditemukan: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[TRACKER] ERROR: Tidak bisa membuka video: {video_path}")
            return []

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        print(f"[TRACKER] Video dibaca: {len(frames)} frames dari {video_path}")
        return frames

    @staticmethod
    def save_video(
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 30
    ) -> None:
        """Simpan list frame menjadi video."""
        if not frames:
            print("[TRACKER] WARNING: Tidak ada frame untuk disimpan!")
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"[TRACKER] Video disimpan: {output_path} "
              f"({len(frames)} frames, {fps} FPS)")

    # ============================================================
    # DETEKSI YOLO
    # ============================================================

    def _detect_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 20,
    ) -> List[Any]:
        """
        Jalankan deteksi YOLO pada semua frame.

        Returns:
            List of YOLO detection results per frame.
        """
        detections = []
        total = len(frames)

        print(f"[TRACKER] Menjalankan deteksi YOLO pada {total} frames...")

        for i in range(0, total, batch_size):
            batch = frames[i:i + batch_size]
            batch_results = self.model.predict(
                batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            detections.extend(batch_results)

            if (i // batch_size) % 5 == 0:
                pct = min(i + batch_size, total) / total * 100
                print(f"[TRACKER] Deteksi progress: "
                      f"{min(i + batch_size, total)}/{total} ({pct:.1f}%)")

        print(f"[TRACKER] Deteksi selesai: {len(detections)} frames diproses.")
        return detections

    # ============================================================
    # KONVERSI DETEKSI KE TRACKS DICTIONARY
    # ============================================================

    def _detections_to_tracks(
        self,
        detections: List[Any],
    ) -> Dict[str, List[Dict]]:
        """
        Konversi hasil deteksi YOLO menjadi tracks dictionary.

        Tracks format:
            {
                'players': [
                    {player_id: {'bbox': [x1,y1,x2,y2]}, ...},  # frame 0
                    {player_id: {'bbox': [x1,y1,x2,y2]}, ...},  # frame 1
                    ...
                ],
                'ball': [
                    {1: {'bbox': [x1,y1,x2,y2]}},  # frame 0
                    {1: {'bbox': [x1,y1,x2,y2]}},  # frame 1
                    ...
                ],
                'heading': [
                    {head_id: {'bbox': [x1,y1,x2,y2], 'confidence': c}, ...},
                    ...
                ],
            }
        """
        tracks: Dict[str, List[Dict]] = {
            'players': [],
            'ball':    [],
        }
        if self.has_heading_class:
            tracks['heading'] = []
        if self.has_cone_class:
            tracks['cones'] = []

        total_frames = len(detections)

        for frame_num, detection in enumerate(detections):
            if frame_num % 200 == 0:
                print(f"[TRACKER] Konversi tracks: "
                      f"frame {frame_num}/{total_frames}...")

            # ----- Pisahkan deteksi per class -----
            boxes = detection.boxes

            player_bboxes = []
            player_confs  = []
            ball_bboxes   = []
            ball_confs    = []
            cone_bboxes   = []
            cone_confs    = []
            heading_bboxes = []
            heading_confs  = []

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls  = int(boxes.cls[i].cpu().numpy())

                if cls == self.CLASS_PLAYER:
                    player_bboxes.append(bbox)
                    player_confs.append(conf)
                elif cls == self.CLASS_BALL:
                    ball_bboxes.append(bbox)
                    ball_confs.append(conf)
                elif self.has_cone_class and cls == self.CLASS_CONE:
                    cone_bboxes.append(bbox)
                    cone_confs.append(conf)
                elif cls == self.CLASS_HEADING:
                    heading_bboxes.append(bbox)
                    heading_confs.append(conf)

            # ----- PLAYER TRACKING (ByteTrack) -----
            players_dict = {}

            if player_bboxes:
                if HAS_SUPERVISION and self.player_tracker is not None:
                    # Gunakan supervision ByteTrack
                    sv_detections = sv.Detections(
                        xyxy=np.array(player_bboxes, dtype=np.float32),
                        confidence=np.array(player_confs, dtype=np.float32),
                    )
                    sv_tracked = self.player_tracker.update_with_detections(
                        sv_detections
                    )

                    for j in range(len(sv_tracked)):
                        tracker_id = int(sv_tracked.tracker_id[j])
                        bbox = sv_tracked.xyxy[j].tolist()
                        players_dict[tracker_id] = {
                            'bbox': bbox,
                        }
                else:
                    # Fallback: tanpa tracking, pakai index sebagai ID
                    for j, bbox in enumerate(player_bboxes):
                        players_dict[j + 1] = {
                            'bbox': bbox,
                        }

            tracks['players'].append(players_dict)

            # ----- BALL -----
            ball_dict = {}

            if ball_bboxes:
                # Ambil bola dengan confidence tertinggi
                best_idx = int(np.argmax(ball_confs))
                ball_dict[1] = {
                    'bbox': ball_bboxes[best_idx],
                    'confidence': ball_confs[best_idx],
                }

            tracks['ball'].append(ball_dict)

            # ----- HEADING (kepala) TRACKING -----
            if self.has_heading_class:
                heading_dict = {}

                if heading_bboxes:
                    if HAS_SUPERVISION and self.heading_tracker is not None:
                        sv_head_det = sv.Detections(
                            xyxy=np.array(heading_bboxes, dtype=np.float32),
                            confidence=np.array(heading_confs, dtype=np.float32),
                        )
                        sv_head_tracked = self.heading_tracker.update_with_detections(
                            sv_head_det
                        )

                        for j in range(len(sv_head_tracked)):
                            tracker_id = int(sv_head_tracked.tracker_id[j])
                            bbox = sv_head_tracked.xyxy[j].tolist()
                            conf = float(sv_head_tracked.confidence[j])
                            heading_dict[tracker_id] = {
                                'bbox': bbox,
                                'confidence': conf,
                            }
                    else:
                        # Fallback: tanpa tracking
                        for j, bbox in enumerate(heading_bboxes):
                            heading_dict[j + 1] = {
                                'bbox': bbox,
                                'confidence': heading_confs[j],
                            }

                tracks['heading'].append(heading_dict)

            # ----- CONE TRACKING -----
            if self.has_cone_class:
                cone_dict = {}

                if cone_bboxes:
                    if HAS_SUPERVISION and self.cone_tracker is not None:
                        sv_cone_det = sv.Detections(
                            xyxy=np.array(cone_bboxes, dtype=np.float32),
                            confidence=np.array(cone_confs, dtype=np.float32),
                        )
                        sv_cone_tracked = self.cone_tracker.update_with_detections(
                            sv_cone_det
                        )

                        for j in range(len(sv_cone_tracked)):
                            tracker_id = int(sv_cone_tracked.tracker_id[j])
                            bbox = sv_cone_tracked.xyxy[j].tolist()
                            conf = float(sv_cone_tracked.confidence[j])
                            cone_dict[tracker_id] = {
                                'bbox': bbox,
                                'confidence': conf,
                            }
                    else:
                        # Fallback: tanpa tracking
                        for j, bbox in enumerate(cone_bboxes):
                            cone_dict[j + 1] = {
                                'bbox': bbox,
                                'confidence': cone_confs[j],
                            }

                tracks['cones'].append(cone_dict)

        print(f"[TRACKER] Konversi tracks selesai: {total_frames} frames.")
        self._print_track_summary(tracks)

        return tracks

    def _print_track_summary(self, tracks: Dict) -> None:
        """Print ringkasan tracking."""
        total = len(tracks['players'])

        # Hitung statistik
        frames_with_players = sum(
            1 for f in tracks['players'] if len(f) > 0
        )
        frames_with_ball = sum(
            1 for f in tracks['ball'] if len(f) > 0
        )
        frames_with_heading = sum(
            1 for f in tracks.get('heading', []) if len(f) > 0
        )
        frames_with_cones = sum(
            1 for f in tracks.get('cones', []) if len(f) > 0
        )

        # Unique IDs
        all_player_ids = set()
        all_heading_ids = set()
        all_cone_ids = set()
        for f in tracks['players']:
            all_player_ids.update(f.keys())
        for f in tracks.get('heading', []):
            all_heading_ids.update(f.keys())
        for f in tracks.get('cones', []):
            all_cone_ids.update(f.keys())

        print(f"\n[TRACKER] === TRACKING SUMMARY ===")
        print(f"[TRACKER] Total frames         : {total}")
        print(f"[TRACKER] Frames dengan player : "
              f"{frames_with_players}/{total} "
              f"({frames_with_players/total*100:.1f}%)")
        print(f"[TRACKER] Frames dengan ball   : "
              f"{frames_with_ball}/{total} "
              f"({frames_with_ball/total*100:.1f}%)")
        print(f"[TRACKER] Frames dengan heading: "
              f"{frames_with_heading}/{total} "
              f"({frames_with_heading/total*100:.1f}%)")
        print(f"[TRACKER] Frames dengan cone   : "
              f"{frames_with_cones}/{total} "
              f"({frames_with_cones/total*100:.1f}%)")
        print(f"[TRACKER] Unique player IDs    : {len(all_player_ids)} "
              f"({sorted(all_player_ids)[:10]}{'...' if len(all_player_ids) > 10 else ''})")
        print(f"[TRACKER] Unique heading IDs   : {len(all_heading_ids)} "
              f"({sorted(all_heading_ids)[:10]}{'...' if len(all_heading_ids) > 10 else ''})")
        print(f"[TRACKER] Unique cone IDs      : {len(all_cone_ids)} "
              f"({sorted(all_cone_ids)[:10]}{'...' if len(all_cone_ids) > 10 else ''})")
        print(f"[TRACKER] ============================\n")

    # ============================================================
    # INTERPOLASI POSISI BOLA (gap filling)
    # ============================================================

    def _interpolate_ball_positions(
        self,
        tracks: Dict,
        max_gap: int = 15,
    ) -> Dict:
        """
        Interpolasi posisi bola untuk frame-frame di mana bola tidak terdeteksi.
        Menggunakan linear interpolation antara deteksi terakhir dan berikutnya.

        Args:
            tracks: Dictionary tracks
            max_gap: Maksimum gap (frame) yang akan di-interpolasi

        Returns:
            tracks dengan posisi bola yang sudah di-interpolasi
        """
        ball_frames = tracks['ball']
        total = len(ball_frames)

        if total == 0:
            return tracks

        # Cari semua frame yang punya bola
        detected_frames = []
        for i, bf in enumerate(ball_frames):
            if bf.get(1) and 'bbox' in bf[1]:
                detected_frames.append(i)

        if len(detected_frames) < 2:
            return tracks

        interpolated_count = 0

        # Interpolasi gap antar frame yang terdeteksi
        for idx in range(len(detected_frames) - 1):
            start_frame = detected_frames[idx]
            end_frame = detected_frames[idx + 1]
            gap = end_frame - start_frame - 1

            if gap <= 0 or gap > max_gap:
                continue

            start_bbox = ball_frames[start_frame][1]['bbox']
            end_bbox = ball_frames[end_frame][1]['bbox']

            for g in range(1, gap + 1):
                t = g / (gap + 1)
                interp_bbox = [
                    start_bbox[j] + (end_bbox[j] - start_bbox[j]) * t
                    for j in range(4)
                ]

                frame_idx = start_frame + g
                ball_frames[frame_idx][1] = {
                    'bbox': interp_bbox,
                    'confidence': 0.0,  # interpolated
                    'interpolated': True,
                }
                interpolated_count += 1

        if interpolated_count > 0:
            print(f"[TRACKER] Ball interpolasi: {interpolated_count} frames diisi.")

        tracks['ball'] = ball_frames
        return tracks

    # ============================================================
    # MAIN: GET OBJECT TRACKS
    # ============================================================

    def get_object_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str = "stubs/tracks_cache.pkl",
        interpolate_ball: bool = True,
        ball_interpolation_max_gap: int = 15,
    ) -> Dict[str, List[Dict]]:
        """
        Pipeline utama: deteksi + tracking semua objek.

        Args:
            frames: List frame video
            read_from_stub: Baca dari cache jika ada
            stub_path: Path file cache (.pkl)
            interpolate_ball: Apakah interpolasi posisi bola
            ball_interpolation_max_gap: Maks gap interpolasi bola

        Returns:
            tracks: {
                'players': [{id: {'bbox': [x1,y1,x2,y2]}, ...}, ...],
                'ball':    [{1:  {'bbox': [x1,y1,x2,y2]}, ...}, ...],
                'heading': [{id: {'bbox': [x1,y1,x2,y2], 'confidence': c}, ...}, ...],
            }
        """
        # ----- Cek cache -----
        if read_from_stub and os.path.exists(stub_path):
            print(f"[TRACKER] Membaca cache dari: {stub_path}")
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)

            # Validasi cache punya semua key yang diperlukan
            required_keys = ['players', 'ball']
            if self.has_heading_class:
                required_keys.append('heading')
            missing = [k for k in required_keys if k not in tracks]

            if missing:
                print(f"[TRACKER] WARNING: Cache tidak punya key: {missing}")
                # Tambahkan key yang missing dengan placeholder
                for key in missing:
                    total = len(tracks.get('players', tracks.get('ball', [])))
                    tracks[key] = [{} for _ in range(total)]
                    print(f"[TRACKER] Ditambahkan placeholder untuk '{key}': "
                          f"{total} frames")

            self._print_track_summary(tracks)
            print(f"[TRACKER] Cache berhasil dimuat: {len(tracks['players'])} frames")
            return tracks

        # ----- Deteksi -----
        detections = self._detect_frames(frames)

        # ----- Konversi ke tracks -----
        tracks = self._detections_to_tracks(detections)

        # ----- Interpolasi bola -----
        if interpolate_ball:
            tracks = self._interpolate_ball_positions(
                tracks, max_gap=ball_interpolation_max_gap
            )

        # ----- Simpan cache -----
        stub_dir = os.path.dirname(stub_path)
        if stub_dir and not os.path.exists(stub_dir):
            os.makedirs(stub_dir, exist_ok=True)

        with open(stub_path, 'wb') as f:
            pickle.dump(tracks, f)
        print(f"[TRACKER] Cache disimpan ke: {stub_path}")

        return tracks

    # ============================================================
    # UTILITY: DRAW ANNOTATIONS (opsional, untuk debugging)
    # ============================================================

    def draw_annotations(
        self,
        frames: List[np.ndarray],
        tracks: Dict,
    ) -> List[np.ndarray]:
        """
        Gambar bounding box untuk semua objek (debug/testing).
        """
        output = []
        total = len(frames)

        for frame_num, frame in enumerate(frames):
            annotated = frame.copy()

            # Players
            for pid, pdata in tracks['players'][frame_num].items():
                bbox = pdata.get('bbox')
                if bbox is None:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 50), 2)
                cv2.putText(annotated, f"Player {pid}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 200, 50), 1)

            # Ball
            ball_data = tracks['ball'][frame_num].get(1)
            if ball_data and 'bbox' in ball_data:
                bx1, by1, bx2, by2 = map(int, ball_data['bbox'])
                bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
                r = max(6, (bx2 - bx1) // 2)
                cv2.circle(annotated, (bcx, bcy), r, (0, 230, 255), 2)
                cv2.putText(annotated, "Ball",
                            (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            (0, 230, 255), 1)

            # Heading (kepala)
            for hid, hdata in tracks['heading'][frame_num].items():
                bbox = hdata.get('bbox')
                if bbox is None:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
                conf = hdata.get('confidence', 0)
                cv2.putText(annotated, f"Head {hid} ({conf:.2f})",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            (255, 180, 0), 1)

            output.append(annotated)

        return output
