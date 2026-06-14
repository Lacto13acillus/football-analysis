# es_batu_main.py
# Pipeline utama untuk menghitung es batu yang masuk ke motor dan truk.
#
# Cara pakai:
#   python es_batu_main.py --input input_videos/es_batu.mp4 --output output_video/es_batu_out.avi
#
# Untuk pakai cache (lebih cepat setelah run pertama):
#   python es_batu_main.py --input ... --output ... --stub
#
# Untuk run real-time dari webcam (index 0):
#   python es_batu_main.py --webcam 0

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(__file__))

from trackers.es_batu_tracker import EsBatuTracker
from trackers.es_batu_counter import EsBatuCounter
from utils.draw_es_batu import (
    annotate_frame,
    draw_counter_panel,
    draw_entry_flash,
)


# ============================================================
# KONFIGURASI DEFAULT
# ============================================================

CONFIG = {
    # ---- Path ----
    "input_video"  : "input_videos/es_batu.mp4",
    "output_video" : "output_video/es_batu_output.avi",
    "model_path"   : "/home/server/models/es_batu.pt",   # <-- sesuaikan path model server
    "stub_path"    : "stubs/es_batu_cache.pkl",
    "use_stub"     : False,

    # ---- Video ----
    "fps"          : 30,

    # ---- Counter params ----
    "carry_iou_thresh"   : 0.10,   # Overlap min es-batu dalam bbox orang agar "dibawa"
    "vehicle_iou_thresh" : 0.10,   # Overlap min es-batu dalam bbox kendaraan agar "masuk"
    "min_carry_frames"   : 3,      # Min frame harus dibawa orang sebelum bisa dihitung
    "cooldown_frames"    : 60,     # Frame cooldown per ID setelah terhitung
    "require_person_first": True,  # Harus melewati orang dulu sebelum masuk kendaraan

    # ---- Tracker params ----
    "conf_threshold" : 0.30,
    "iou_threshold"  : 0.45,

    # ---- Visualisasi ----
    "flash_duration_frames": 45,   # Berapa frame animasi flash ditampilkan
    "show_all_bboxes"      : True,
}


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Es Batu Counter — Hitung es batu masuk motor/truk"
    )
    parser.add_argument("--input",   type=str, help="Path video input")
    parser.add_argument("--output",  type=str, help="Path video output")
    parser.add_argument("--model",   type=str, help="Path model YOLO (.pt)")
    parser.add_argument("--stub",    action="store_true", help="Gunakan cache")
    parser.add_argument("--no-stub", action="store_true", help="Jangan pakai cache")
    parser.add_argument("--webcam",  type=int, default=-1,
                        help="Index webcam untuk mode real-time (misal: 0)")
    parser.add_argument("--conf",    type=float, help="Confidence threshold")
    parser.add_argument("--carry-thresh",   type=float,
                        help="Overlap threshold es-batu dalam orang")
    parser.add_argument("--vehicle-thresh", type=float,
                        help="Overlap threshold es-batu dalam kendaraan")
    return parser.parse_args()


# ============================================================
# BUILD FLASH MAP
# ============================================================

def build_flash_map(
    results      : List[Dict],
    flash_duration: int,
) -> Dict[int, Dict]:
    """
    Buat mapping frame_num → flash info untuk semua event counting.

    Returns:
        {frame_num: {'vehicle': str, 'count': int, 'progress': float}}
    """
    flash_map: Dict[int, Dict] = {}
    total = len(results)

    for result in results:
        fn = result['frame_num']
        for vehicle in ('motor', 'truk'):
            key = f'new_{vehicle}'
            if result.get(key):
                count = result[f'count_{vehicle}']
                for offset in range(flash_duration):
                    fnum = fn + offset
                    if fnum >= total:
                        break
                    progress = 1.0 - offset / flash_duration
                    # Jika ada 2 event di frame berdekatan, yang terakhir menang
                    if fnum not in flash_map or flash_map[fnum]['progress'] < progress:
                        flash_map[fnum] = {
                            'vehicle' : vehicle,
                            'count'   : count,
                            'progress': progress,
                        }

    return flash_map


# ============================================================
# RENDER SEMUA FRAME
# ============================================================

def render_frames(
    frames      : List[np.ndarray],
    tracks      : Dict,
    results     : List[Dict],
    flash_map   : Dict,
) -> List[np.ndarray]:
    """Render semua frame dengan anotasi lengkap."""
    output = []
    total  = len(frames)
    print(f"\n[RENDER] Mulai merender {total} frames...")

    for frame_num, frame in enumerate(frames):
        if frame_num % 100 == 0:
            pct = frame_num / total * 100
            print(f"[RENDER] Progress: {frame_num}/{total} ({pct:.1f}%)...")

        annotated = annotate_frame(
            frame        = frame,
            frame_num    = frame_num,
            tracks       = tracks,
            frame_result = results[frame_num],
            flash_map    = flash_map,
        )
        output.append(annotated)

    print(f"[RENDER] Selesai: {len(output)}/{total} frames dirender.")
    return output


# ============================================================
# MODE REAL-TIME (WEBCAM / RTSP)
# ============================================================

def run_realtime(
    source : int | str,
    tracker: EsBatuTracker,
    counter: EsBatuCounter,
    config : Dict,
) -> None:
    """
    Mode real-time: baca frame satu per satu dari kamera/RTSP,
    deteksi + hitung langsung, tampilkan ke layar.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[REALTIME] ERROR: Tidak bisa membuka source: {source}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or config['fps']
    print(f"[REALTIME] Stream dibuka. FPS={fps}")
    print("[REALTIME] Tekan 'q' untuk keluar, 'r' untuk reset counter.")

    frame_num    = 0
    flash_map    = {}
    flash_dur    = config['flash_duration_frames']
    model        = tracker.model
    conf_thresh  = tracker.conf_threshold
    iou_thresh   = tracker.iou_threshold

    # Tracker instances (ByteTrack)
    try:
        import supervision as sv
        _tracker_es  = sv.ByteTrack(track_activation_threshold=0.25,
                                     lost_track_buffer=30, frame_rate=fps)
        _tracker_org = sv.ByteTrack(track_activation_threshold=0.25,
                                     lost_track_buffer=30, frame_rate=fps)
        _tracker_mtr = sv.ByteTrack(track_activation_threshold=0.25,
                                     lost_track_buffer=60, frame_rate=fps)
        _tracker_trk = sv.ByteTrack(track_activation_threshold=0.25,
                                     lost_track_buffer=60, frame_rate=fps)
        has_sv = True
    except ImportError:
        has_sv = False
        _fallback_ids = {'es': 0, 'orang': 0, 'motor': 0, 'truk': 0}

    CLASS_ES_BATU = 0
    CLASS_MOTOR   = 1
    CLASS_ORANG   = 2
    CLASS_TRUK    = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[REALTIME] Stream berakhir.")
            break

        # Deteksi dengan YOLO
        det_results = model.predict(
            frame, conf=conf_thresh, iou=iou_thresh, verbose=False
        )
        boxes = det_results[0].boxes

        # Pisahkan per class
        def _split_class(cls_id):
            bbs, cfs = [], []
            for i in range(len(boxes)):
                if int(boxes.cls[i]) == cls_id:
                    bbs.append(boxes.xyxy[i].cpu().numpy().tolist())
                    cfs.append(float(boxes.conf[i].cpu().numpy()))
            return bbs, cfs

        es_bbs, es_cfs   = _split_class(CLASS_ES_BATU)
        mtr_bbs, mtr_cfs = _split_class(CLASS_MOTOR)
        org_bbs, org_cfs = _split_class(CLASS_ORANG)
        trk_bbs, trk_cfs = _split_class(CLASS_TRUK)

        def _to_dict_sv(bbs, cfs, sv_tracker):
            if not bbs:
                return {}
            d = sv.Detections(
                xyxy=np.array(bbs, dtype=np.float32),
                confidence=np.array(cfs, dtype=np.float32),
            )
            t = sv_tracker.update_with_detections(d)
            return {int(t.tracker_id[j]): {'bbox': t.xyxy[j].tolist(),
                                            'conf': float(t.confidence[j])}
                    for j in range(len(t))}

        def _to_dict_fallback(bbs, cfs, key):
            result = {}
            for j, (b, c) in enumerate(zip(bbs, cfs)):
                result[j + 1] = {'bbox': b, 'conf': c}
            return result

        if has_sv:
            es_dict  = _to_dict_sv(es_bbs,  es_cfs,  _tracker_es)
            org_dict = _to_dict_sv(org_bbs, org_cfs, _tracker_org)
            mtr_dict = _to_dict_sv(mtr_bbs, mtr_cfs, _tracker_mtr)
            trk_dict = _to_dict_sv(trk_bbs, trk_cfs, _tracker_trk)
        else:
            es_dict  = _to_dict_fallback(es_bbs,  es_cfs,  'es')
            org_dict = _to_dict_fallback(org_bbs, org_cfs, 'orang')
            mtr_dict = _to_dict_fallback(mtr_bbs, mtr_cfs, 'motor')
            trk_dict = _to_dict_fallback(trk_bbs, trk_cfs, 'truk')

        # Counter
        result = counter.process_frame(
            frame_num    = frame_num,
            es_batu_dict = es_dict,
            orang_dict   = org_dict,
            motor_dict   = mtr_dict,
            truk_dict    = trk_dict,
        )

        # Flash map update
        for vehicle in ('motor', 'truk'):
            if result.get(f'new_{vehicle}'):
                count = result[f'count_{vehicle}']
                for offset in range(flash_dur):
                    fnum = frame_num + offset
                    progress = 1.0 - offset / flash_dur
                    if fnum not in flash_map or flash_map[fnum]['progress'] < progress:
                        flash_map[fnum] = {
                            'vehicle' : vehicle,
                            'count'   : count,
                            'progress': progress,
                        }

        # Build mini-tracks untuk annotate_frame
        mini_tracks = {
            'es_batu': [es_dict],
            'orang'  : [org_dict],
            'motor'  : [mtr_dict],
            'truk'   : [trk_dict],
        }
        mini_result = {**result, 'frame_num': 0}
        mini_flash  = {0: flash_map[frame_num]} if frame_num in flash_map else {}

        from utils.draw_es_batu import annotate_frame as _af
        annotated = _af(
            frame        = frame,
            frame_num    = 0,
            tracks       = mini_tracks,
            frame_result = mini_result,
            flash_map    = mini_flash,
        )

        cv2.imshow("Es Batu Counter [q=quit, r=reset]", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.reset()
            flash_map = {}
            print("[REALTIME] Counter direset.")

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
    counter.print_event_log()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # Override config dengan args
    if args.input:
        CONFIG["input_video"] = args.input
    if args.output:
        CONFIG["output_video"] = args.output
    if args.model:
        CONFIG["model_path"] = args.model
    if args.stub:
        CONFIG["use_stub"] = True
    if args.no_stub:
        CONFIG["use_stub"] = False
    if args.conf:
        CONFIG["conf_threshold"] = args.conf
    if args.carry_thresh:
        CONFIG["carry_iou_thresh"] = args.carry_thresh
    if args.vehicle_thresh:
        CONFIG["vehicle_iou_thresh"] = args.vehicle_thresh

    print("\n" + "=" * 70)
    print("   ES BATU COUNTER v1.0")
    print("   Counting Es Batu yang Masuk Motor & Truk")
    print("=" * 70)
    print(f"  Model          : {CONFIG['model_path']}")
    print(f"  Conf threshold : {CONFIG['conf_threshold']}")
    print(f"  Carry threshold: {CONFIG['carry_iou_thresh']}")
    print(f"  Vehicle thresh : {CONFIG['vehicle_iou_thresh']}")
    print(f"  Min carry frames: {CONFIG['min_carry_frames']}")
    print(f"  Cooldown frames: {CONFIG['cooldown_frames']}")
    print("=" * 70)

    # ---- Inisialisasi tracker & counter ----
    tracker = EsBatuTracker(
        model_path     = CONFIG["model_path"],
        conf_threshold = CONFIG["conf_threshold"],
        iou_threshold  = CONFIG["iou_threshold"],
    )

    counter = EsBatuCounter(
        carry_iou_thresh    = CONFIG["carry_iou_thresh"],
        vehicle_iou_thresh  = CONFIG["vehicle_iou_thresh"],
        min_carry_frames    = CONFIG["min_carry_frames"],
        cooldown_frames     = CONFIG["cooldown_frames"],
        require_person_first= CONFIG["require_person_first"],
    )

    # ---- Mode REAL-TIME ----
    if args.webcam >= 0:
        print(f"\n[MAIN] MODE REAL-TIME — Webcam index: {args.webcam}")
        run_realtime(
            source  = args.webcam,
            tracker = tracker,
            counter = counter,
            config  = CONFIG,
        )
        return

    # ---- Mode VIDEO FILE ----
    # TAHAP 1: Baca Video
    print(f"\n[MAIN] TAHAP 1: Membaca video input...")
    print(f"  Input : {CONFIG['input_video']}")
    if not os.path.exists(CONFIG["input_video"]):
        print(f"[MAIN] ERROR: File tidak ditemukan: {CONFIG['input_video']}")
        return

    frames = EsBatuTracker.read_video(CONFIG["input_video"])
    if not frames:
        print("[MAIN] ERROR: Video kosong!")
        return

    cap = cv2.VideoCapture(CONFIG["input_video"])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        fps = CONFIG["fps"]
    CONFIG["fps"] = fps
    print(f"  FPS: {fps}, Total frames: {len(frames)}")

    # TAHAP 2: Deteksi & Tracking
    print(f"\n[MAIN] TAHAP 2: Deteksi & Tracking objek...")
    tracks = tracker.get_object_tracks(
        frames         = frames,
        read_from_stub = CONFIG["use_stub"],
        stub_path      = CONFIG["stub_path"],
    )

    # TAHAP 3: Counting
    print(f"\n[MAIN] TAHAP 3: Proses counting es batu...")
    results = counter.process_all_frames(tracks)

    # TAHAP 4: Statistik
    print(f"\n[MAIN] TAHAP 4: Statistik akhir...")
    stats = counter.get_statistics()
    counter.print_event_log()

    # TAHAP 5: Build Flash Map
    print(f"\n[MAIN] TAHAP 5: Build flash map...")
    flash_map = build_flash_map(results, CONFIG["flash_duration_frames"])
    print(f"  Flash map: {len(flash_map)} frames dengan animasi")

    # TAHAP 6: Render
    print(f"\n[MAIN] TAHAP 6: Render video output...")
    output_frames = render_frames(
        frames    = frames,
        tracks    = tracks,
        results   = results,
        flash_map = flash_map,
    )

    # TAHAP 7: Simpan
    print(f"\n[MAIN] TAHAP 7: Menyimpan video output...")
    out_dir = os.path.dirname(CONFIG["output_video"])
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    EsBatuTracker.save_video(output_frames, CONFIG["output_video"], fps=fps)

    # SELESAI
    print("\n" + "=" * 70)
    print("   PIPELINE SELESAI!")
    print("=" * 70)
    print(f"  Output video   : {CONFIG['output_video']}")
    print(f"  Total frames   : {len(output_frames)}")
    print(f"  Durasi         : {len(output_frames) / fps:.1f} detik")
    print(f"  Es batu → Motor: {stats['count_motor']}")
    print(f"  Es batu → Truk : {stats['count_truk']}")
    print(f"  Grand Total    : {stats['grand_total']}")
    print(f"  Total events   : {stats['total_events']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
