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
        self.model   = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        self.cls_names     : Dict[int, str] = self.model.model.names
        self.cls_names_inv : Dict[str, int] = {v: k for k, v in self.cls_names.items()}

        self._cone_anchors: Dict[int, Tuple[float, float]] = {}
        self._cone_anchor_max_distance = 60.0

        print(f"[TRACKER] Model loaded: {model_path}")
        print(f"[TRACKER] Kelas terdeteksi: {list(self.cls_names.values())}")

        self._ball_cls_name    = self._find_class(['ball', 'sports ball'])
        self._player_cls_name  = self._find_class(['player', 'person'])
        self._keeper_cls_name  = self._find_class(['keeper'])        # BARU
        self._gawang_cls_name  = self._find_class(['gawang', 'goal']) # BARU
        self._cone_cls_name    = self._find_class(['cone'])

        print(f"[TRACKER] Kelas BOLA    : '{self._ball_cls_name}'")
        print(f"[TRACKER] Kelas PEMAIN  : '{self._player_cls_name}'")
        print(f"[TRACKER] Kelas KEEPER  : '{self._keeper_cls_name}'")
        print(f"[TRACKER] Kelas GAWANG  : '{self._gawang_cls_name}'")
        print(f"[TRACKER] Kelas CONE    : '{self._cone_cls_name}'")

    def _find_class(self, candidates: List[str]) -> Optional[str]:
        for name in candidates:
            if name in self.cls_names_inv:
                return name
        return None

    def _get_class_id(self, class_name: Optional[str]) -> int:
        if class_name is None:
            return -1
        return self.cls_names_inv.get(class_name, -1)

    def _match_cone_to_anchor(self, cx: float, cy: float) -> int:
        best_id   = None
        best_dist = float('inf')
        for anchor_id, anchor_pos in self._cone_anchors.items():
            dist = measure_distance((cx, cy), anchor_pos)
            if dist < best_dist:
                best_dist = dist
                best_id   = anchor_id
        if best_dist <= self._cone_anchor_max_distance:
            ax, ay = self._cone_anchors[best_id]
            alpha  = 0.1
            self._cone_anchors[best_id] = (
                ax * (1 - alpha) + cx * alpha,
                ay * (1 - alpha) + cy * alpha
            )
            return best_id
        else:
            new_id = len(self._cone_anchors)
            while new_id in self._cone_anchors:
                new_id += 1
            self._cone_anchors[new_id] = (cx, cy)
            return new_id

    def _reset_cone_anchors(self) -> None:
        self._cone_anchors.clear()

    def detect_frames(self, frames: List[np.ndarray], batch_size: int = 20) -> List[Any]:
        detections   = []
        total_batches = (len(frames) + batch_size - 1) // batch_size
        for i in range(0, len(frames), batch_size):
            batch     = frames[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"[TRACKER] Deteksi batch {batch_num}/{total_batches}...")
            results = self.model.predict(batch, conf=0.1, verbose=False)
            detections.extend(results)
        return detections

    def get_object_tracks(
        self,
        frames        : List[np.ndarray],
        read_from_stub: bool = False,
        stub_path     : Optional[str] = None
    ) -> Dict[str, List[Dict[int, Dict[str, Any]]]]:
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"[TRACKER] Membaca tracks dari cache: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        self._reset_cone_anchors()

        tracks: Dict[str, List[Dict[int, Dict[str, Any]]]] = {
            "players": [],
            "ball"   : [],
            "keeper" : [],    # BARU
            "gawang" : [],    # BARU
            "cones"  : []
        }

        assigned_in_frame = set()

        for frame_num, detection in enumerate(detections):
            if frame_num % 50 == 0:
                print(f"[TRACKER] Processing frame {frame_num}/{len(detections)}...")

            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["keeper"].append({})   # BARU
            tracks["gawang"].append({})   # BARU
            tracks["cones"].append({})

            det_sv = sv.Detections.from_ultralytics(detection)
            if len(det_sv) == 0:
                continue

            cls_names_local     = detection.names
            cls_names_inv_local = {v: k for k, v in cls_names_local.items()}

            # ---- TRACKING PEMAIN ----
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

            # ---- DETEKSI BOLA ----
            b_cls_id  = cls_names_inv_local.get(self._ball_cls_name, -1)
            ball_mask = det_sv.class_id == b_cls_id
            ball_det  = det_sv[ball_mask]

            if len(ball_det) > 0:
                best_idx  = int(np.argmax(ball_det.confidence))
                best_ball = ball_det[best_idx]
                bbox      = best_ball.xyxy[0].tolist()
                conf      = float(best_ball.confidence[0])
                tracks["ball"][frame_num][1] = {
                    "bbox"      : bbox,
                    "confidence": conf
                }

            # ---- DETEKSI KEEPER (BARU) ----
            k_cls_id    = cls_names_inv_local.get(self._keeper_cls_name, -1)
            keeper_mask = det_sv.class_id == k_cls_id
            keeper_det  = det_sv[keeper_mask]

            if len(keeper_det) > 0:
                best_idx    = int(np.argmax(keeper_det.confidence))
                best_keeper = keeper_det[best_idx]
                bbox = best_keeper.xyxy[0].tolist()
                conf = float(best_keeper.confidence[0])
                tracks["keeper"][frame_num][1] = {
                    "bbox"      : bbox,
                    "confidence": conf
                }

            # ---- DETEKSI GAWANG (BARU) ----
            g_cls_id    = cls_names_inv_local.get(self._gawang_cls_name, -1)
            gawang_mask = det_sv.class_id == g_cls_id
            gawang_det  = det_sv[gawang_mask]

            if len(gawang_det) > 0:
                best_idx    = int(np.argmax(gawang_det.confidence))
                best_gawang = gawang_det[best_idx]
                bbox = best_gawang.xyxy[0].tolist()
                conf = float(best_gawang.confidence[0])
                tracks["gawang"][frame_num][1] = {
                    "bbox"      : bbox,
                    "confidence": conf
                }

            # ---- DETEKSI CONE ----
            c_cls_id  = cls_names_inv_local.get(self._cone_cls_name, -1)
            cone_mask = det_sv.class_id == c_cls_id
            cone_det  = det_sv[cone_mask]

            assigned_in_frame = set()
            for idx in range(len(cone_det)):
                bbox = cone_det.xyxy[idx].tolist()
                conf = float(cone_det.confidence[idx])
                if conf < 0.25:
                    continue
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w <= 0:
                    continue
                aspect_ratio = h / w
                if aspect_ratio < 0.8:
                    continue
                cx, cy = get_center_of_bbox_bottom(bbox)
                cone_id = self._match_cone_to_anchor(cx, cy)
                if cone_id in assigned_in_frame:
                    continue
                assigned_in_frame.add(cone_id)
                tracks["cones"][frame_num][cone_id] = {
                    "bbox"        : bbox,
                    "confidence"  : conf,
                    "aspect_ratio": aspect_ratio
                }

        # Interpolasi bola
        print("[TRACKER] Interpolasi posisi bola...")
        tracks["ball"] = interpolate_ball_positions(tracks["ball"])

        ball_detected = sum(1 for f in tracks["ball"] if f.get(1))
        print(f"[TRACKER] Bola terdeteksi: {ball_detected}/{len(tracks['ball'])} frames")

        gawang_detected = sum(1 for f in tracks["gawang"] if f.get(1))
        print(f"[TRACKER] Gawang terdeteksi: {gawang_detected}/{len(tracks['gawang'])} frames")

        keeper_detected = sum(1 for f in tracks["keeper"] if f.get(1))
        print(f"[TRACKER] Keeper terdeteksi: {keeper_detected}/{len(tracks['keeper'])} frames")

        if stub_path:
            os.makedirs(
                os.path.dirname(stub_path) if os.path.dirname(stub_path) else '.',
                exist_ok=True
            )
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            print(f"[TRACKER] Cache disimpan: {stub_path}")

        return tracks

    def draw_annotations(
        self, frame, tracks, frame_num, ball_possessions, player_identifier=None
    ) -> np.ndarray:
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
                center=((bx1 + bx2) // 2, (by1 + by2) // 2),
                axes=((bx2 - bx1) // 2, (by2 - by1) // 2),
                angle=0, startAngle=0, endAngle=360,
                color=(0, 255, 255), thickness=2
            )
        return annotated

    @staticmethod
    def read_video(video_path: str) -> List[np.ndarray]:
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
    def save_video(output_frames: List[np.ndarray], output_path: str, fps: int = 24) -> None:
        if not output_frames:
            print("[VIDEO] Tidak ada frame untuk disimpan!")
            return
        h, w = output_frames[0].shape[:2]
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
        print(f"[VIDEO] Disimpan: {output_path} ({len(output_frames)} frames, {fps}fps)")
