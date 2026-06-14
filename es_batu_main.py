# es_batu_main.py
# Pipeline utama untuk menghitung es batu yang masuk ke motor dan truk.
#
# Cara pakai:
#   python es_batu_main.py --input input_videos/es_batu_truk.mp4 --output output_video/result.avi
#
# Untuk run real-time dari webcam:
#   python es_batu_main.py --webcam 0

import os
import sys
import cv2
import argparse
import numpy as np
from typing import List, Dict, Optional, Union

sys.path.append(os.path.dirname(__file__))

from trackers.es_batu_counter import EsBatuCounter
from utils.draw_es_batu import (
    draw_es_batu_bbox,
    draw_orang_bbox,
    draw_motor_bbox,
    draw_truk_bbox,
    draw_counter_panel,
    draw_entry_flash,
    _intersection_ratio,
    _center,
    _point_in_box,
)


# ============================================================
# KONFIGURASI DEFAULT
# ============================================================

CONFIG = {
    # ---- Path ----
    "input_video"  : "input_videos/es_batu_truk.mp4",
    "output_video" : "output_video/es_batu_truk.avi",
    "model_path"   : "/home/dika/football-analysis/models/es_batu.pt",

    # ---- Video ----
    "fps"          : 30,

    # ---- Counter params ----
    "carry_iou_thresh"    : 0.10,   # Overlap min es-batu dalam bbox orang agar "dibawa"
    "vehicle_iou_thresh"  : 0.10,   # Overlap min es-batu dalam bbox kendaraan agar "masuk"
    "min_carry_frames"    : 3,      # Min frame harus dibawa orang sebelum bisa dihitung
    "cooldown_frames"     : 60,     # Frame cooldown per ID setelah terhitung
    "require_person_first": True,   # Harus melewati orang dulu sebelum masuk kendaraan

    # ---- Tracker params ----
    "conf_threshold" : 0.30,
    "iou_threshold"  : 0.45,

    # ---- Visualisasi ----
    "flash_duration_frames": 45,

    # ---- Streaming ----
    # Batch YOLO: berapa frame diproses sekaligus. Turunkan ke 4 jika VRAM habis.
    "batch_size": 8,
}

# Class ID sesuai data.yaml es_batu
CLASS_ES_BATU = 0
CLASS_MOTOR   = 1
CLASS_ORANG   = 2
CLASS_TRUK    = 3


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Es Batu Counter — Hitung es batu masuk motor/truk"
    )
    parser.add_argument("--input",          type=str,   help="Path video input")
    parser.add_argument("--output",         type=str,   help="Path video output")
    parser.add_argument("--model",          type=str,   help="Path model YOLO (.pt)")
    parser.add_argument("--webcam",         type=int,   default=-1,
                        help="Index webcam untuk mode real-time (misal: 0)")
    parser.add_argument("--conf",           type=float, help="Confidence threshold")
    parser.add_argument("--carry-thresh",   type=float,
                        help="Overlap threshold es-batu di dalam orang")
    parser.add_argument("--vehicle-thresh", type=float,
                        help="Overlap threshold es-batu di dalam kendaraan")
    parser.add_argument("--batch-size",     type=int,
                        help="Batch size YOLO inference (default 8)")
    return parser.parse_args()


# ============================================================
# HELPER: PARSING DETEKSI YOLO → DICT PER CLASS
# ============================================================

def _parse_detection(boxes) -> Dict[str, Dict]:
    """Pisahkan boxes YOLO menjadi list bbox+conf per class."""
    result = {
        'es_batu': {'bbs': [], 'cfs': []},
        'motor'  : {'bbs': [], 'cfs': []},
        'orang'  : {'bbs': [], 'cfs': []},
        'truk'   : {'bbs': [], 'cfs': []},
    }
    cls_map = {
        CLASS_ES_BATU: 'es_batu',
        CLASS_MOTOR  : 'motor',
        CLASS_ORANG  : 'orang',
        CLASS_TRUK   : 'truk',
    }
    for i in range(len(boxes)):
        cls  = int(boxes.cls[i].cpu().numpy())
        bbox = boxes.xyxy[i].cpu().numpy().tolist()
        conf = float(boxes.conf[i].cpu().numpy())
        key  = cls_map.get(cls)
        if key:
            result[key]['bbs'].append(bbox)
            result[key]['cfs'].append(conf)
    return result


def _track_class(
    bbs      : List,
    cfs      : List,
    sv_tracker,
    has_sv   : bool,
) -> Dict[int, Dict]:
    """Jalankan ByteTrack untuk satu class. Return {id: {'bbox', 'conf'}}."""
    if not bbs:
        return {}
    if has_sv:
        import supervision as sv
        det     = sv.Detections(
            xyxy=np.array(bbs, dtype=np.float32),
            confidence=np.array(cfs, dtype=np.float32),
        )
        tracked = sv_tracker.update_with_detections(det)
        return {
            int(tracked.tracker_id[j]): {
                'bbox': tracked.xyxy[j].tolist(),
                'conf': float(tracked.confidence[j]),
            }
            for j in range(len(tracked))
        }
    else:
        return {j + 1: {'bbox': b, 'conf': c} for j, (b, c) in enumerate(zip(bbs, cfs))}


# ============================================================
# ANNOTASI SATU FRAME (standalone)
# ============================================================

def annotate_single_frame(
    frame       : np.ndarray,
    frame_num   : int,
    es_dict     : Dict,
    orang_dict  : Dict,
    motor_dict  : Dict,
    truk_dict   : Dict,
    frame_result: Dict,
    flash_info  : Optional[Dict],
) -> np.ndarray:
    """Render anotasi lengkap pada satu frame tanpa tracks dictionary global."""
    annotated = frame.copy()
    es_states = frame_result.get('es_states', {})
    new_motor = set(frame_result.get('new_motor', []))
    new_truk  = set(frame_result.get('new_truk', []))

    # Orang yang sedang membawa es batu
    carrying_persons: set = set()
    for es_id, st in es_states.items():
        if st == 'CARRIED':
            eb = es_dict.get(es_id)
            if eb:
                ec = _center(eb['bbox'])
                for pid, pdata in orang_dict.items():
                    if _point_in_box(ec, pdata['bbox']):
                        carrying_persons.add(pid)

    # Kendaraan yang sedang menerima es batu (highlight)
    motor_hl: set = set()
    truk_hl : set = set()
    for es_id in new_motor:
        eb = es_dict.get(es_id)
        if eb:
            for mid, mdata in motor_dict.items():
                if _intersection_ratio(eb['bbox'], mdata['bbox']) > 0.05:
                    motor_hl.add(mid)
    for es_id in new_truk:
        eb = es_dict.get(es_id)
        if eb:
            for tid, tdata in truk_dict.items():
                if _intersection_ratio(eb['bbox'], tdata['bbox']) > 0.05:
                    truk_hl.add(tid)

    # Gambar TRUK
    for tid, tdata in truk_dict.items():
        annotated = draw_truk_bbox(annotated, tdata['bbox'], tid,
                                   highlight=(tid in truk_hl))

    # Gambar MOTOR
    for mid, mdata in motor_dict.items():
        annotated = draw_motor_bbox(annotated, mdata['bbox'], mid,
                                    highlight=(mid in motor_hl))

    # Gambar ORANG
    for pid, pdata in orang_dict.items():
        annotated = draw_orang_bbox(annotated, pdata['bbox'], pid,
                                    is_carrying=(pid in carrying_persons))

    # Gambar ES BATU
    for es_id, eb_data in es_dict.items():
        state = es_states.get(es_id, 'IDLE')
        annotated = draw_es_batu_bbox(
            annotated, eb_data['bbox'], es_id, state, eb_data.get('conf', 0)
        )

    # Panel counter
    annotated = draw_counter_panel(
        annotated,
        count_motor=frame_result['count_motor'],
        count_truk =frame_result['count_truk'],
    )

    # Flash animasi
    if flash_info:
        annotated = draw_entry_flash(
            annotated,
            vehicle =flash_info['vehicle'],
            count   =flash_info['count'],
            progress=flash_info['progress'],
        )

    # Frame label
    h, w = annotated.shape[:2]
    cv2.putText(annotated, f"Frame: {frame_num}",
                (w - 140, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1)

    return annotated


# ============================================================
# PIPELINE STREAMING SATU-PASS (hemat RAM)
# ============================================================

def run_video_streaming(
    model,
    counter  : EsBatuCounter,
    config   : Dict,
    trackers : Dict,
    has_sv   : bool,
) -> None:
    """
    Pipeline hemat memori:
      Baca batch kecil → deteksi YOLO → counting → tulis ke VideoWriter.
    Tidak ada penumpukan seluruh frames ke RAM.
    """
    input_path  = config["input_video"]
    output_path = config["output_video"]
    batch_size  = config.get("batch_size", 8)
    flash_dur   = config["flash_duration_frames"]
    conf_thresh = config["conf_threshold"]
    iou_thresh  = config["iou_threshold"]

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[STREAM] ERROR: Tidak bisa membuka: {input_path}")
        return

    fps          = int(cap.get(cv2.CAP_PROP_FPS)) or config["fps"]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  FPS          : {fps}")
    print(f"  Total frames : {total_frames}")
    print(f"  Resolusi     : {width}x{height}")
    print(f"  Batch size   : {batch_size}")

    # Output VideoWriter
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    flash_map     : Dict[int, Dict] = {}
    frame_num     : int = 0          # nomor frame yang baru dibaca
    batch_buf     : List = []
    batch_start_fn: int = 0          # frame_num dari frame pertama di buffer saat ini

    def _flush_batch(buf: List, start_fn: int) -> None:
        """Proses satu batch frame: deteksi → hitung → tulis."""
        nonlocal flash_map

        yolo_results = model.predict(
            buf, conf=conf_thresh, iou=iou_thresh, verbose=False
        )

        for i, det in enumerate(yolo_results):
            fn    = start_fn + i
            frame = buf[i]

            parsed   = _parse_detection(det.boxes)
            es_dict  = _track_class(parsed['es_batu']['bbs'], parsed['es_batu']['cfs'],
                                    trackers['es'], has_sv)
            org_dict = _track_class(parsed['orang']['bbs'],   parsed['orang']['cfs'],
                                    trackers['orang'], has_sv)
            mtr_dict = _track_class(parsed['motor']['bbs'],   parsed['motor']['cfs'],
                                    trackers['motor'], has_sv)
            trk_dict = _track_class(parsed['truk']['bbs'],    parsed['truk']['cfs'],
                                    trackers['truk'], has_sv)

            # Counting
            result = counter.process_frame(
                frame_num    = fn,
                es_batu_dict = es_dict,
                orang_dict   = org_dict,
                motor_dict   = mtr_dict,
                truk_dict    = trk_dict,
            )

            # Update flash map
            for vehicle in ('motor', 'truk'):
                if result.get(f'new_{vehicle}'):
                    count = result[f'count_{vehicle}']
                    for offset in range(flash_dur):
                        fnum     = fn + offset
                        progress = 1.0 - offset / flash_dur
                        if fnum not in flash_map or flash_map[fnum]['progress'] < progress:
                            flash_map[fnum] = {
                                'vehicle' : vehicle,
                                'count'   : count,
                                'progress': progress,
                            }

            # Bersihkan flash lama agar flash_map tidak membengkak
            old = [k for k in flash_map if k < fn - flash_dur]
            for k in old:
                del flash_map[k]

            # Render & tulis
            annotated = annotate_single_frame(
                frame        = frame,
                frame_num    = fn,
                es_dict      = es_dict,
                orang_dict   = org_dict,
                motor_dict   = mtr_dict,
                truk_dict    = trk_dict,
                frame_result = result,
                flash_info   = flash_map.get(fn),
            )
            writer.write(annotated)

            if fn % 100 == 0:
                pct = (fn / total_frames * 100) if total_frames > 0 else 0
                print(f"[STREAM] Frame {fn}/{total_frames} ({pct:.1f}%) "
                      f"| Motor: {counter.count_motor} | Truk: {counter.count_truk}")

    # ---- Loop baca frame ----
    print(f"\n[STREAM] Memulai streaming pipeline...\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(batch_buf) == 0:
            batch_start_fn = frame_num   # catat frame pertama batch ini

        batch_buf.append(frame)
        frame_num += 1

        if len(batch_buf) >= batch_size:
            _flush_batch(batch_buf, batch_start_fn)
            batch_buf = []

    # Proses sisa batch terakhir
    if batch_buf:
        _flush_batch(batch_buf, batch_start_fn)

    cap.release()
    writer.release()
    print(f"\n[STREAM] Selesai! Video disimpan ke: {output_path}")


# ============================================================
# MODE REAL-TIME (WEBCAM / RTSP)
# ============================================================

def run_realtime(
    source   : Union[int, str],
    model,
    counter  : EsBatuCounter,
    config   : Dict,
    trackers : Dict,
    has_sv   : bool,
) -> None:
    """Mode real-time: deteksi frame per frame dari kamera/RTSP."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[REALTIME] ERROR: Tidak bisa membuka source: {source}")
        return

    fps       = int(cap.get(cv2.CAP_PROP_FPS)) or config['fps']
    flash_dur = config['flash_duration_frames']
    flash_map : Dict[int, Dict] = {}
    frame_num : int = 0

    print(f"[REALTIME] Stream dibuka. FPS={fps}")
    print("[REALTIME] Tekan 'q' untuk keluar, 'r' untuk reset counter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[REALTIME] Stream berakhir.")
            break

        det_result = model.predict(
            frame,
            conf=config['conf_threshold'],
            iou=config['iou_threshold'],
            verbose=False,
        )
        parsed   = _parse_detection(det_result[0].boxes)
        es_dict  = _track_class(parsed['es_batu']['bbs'], parsed['es_batu']['cfs'],
                                 trackers['es'], has_sv)
        org_dict = _track_class(parsed['orang']['bbs'],   parsed['orang']['cfs'],
                                 trackers['orang'], has_sv)
        mtr_dict = _track_class(parsed['motor']['bbs'],   parsed['motor']['cfs'],
                                 trackers['motor'], has_sv)
        trk_dict = _track_class(parsed['truk']['bbs'],    parsed['truk']['cfs'],
                                 trackers['truk'], has_sv)

        result = counter.process_frame(
            frame_num    = frame_num,
            es_batu_dict = es_dict,
            orang_dict   = org_dict,
            motor_dict   = mtr_dict,
            truk_dict    = trk_dict,
        )

        for vehicle in ('motor', 'truk'):
            if result.get(f'new_{vehicle}'):
                count = result[f'count_{vehicle}']
                for offset in range(flash_dur):
                    fnum     = frame_num + offset
                    progress = 1.0 - offset / flash_dur
                    if fnum not in flash_map or flash_map[fnum]['progress'] < progress:
                        flash_map[fnum] = {'vehicle': vehicle, 'count': count,
                                           'progress': progress}

        annotated = annotate_single_frame(
            frame        = frame,
            frame_num    = frame_num,
            es_dict      = es_dict,
            orang_dict   = org_dict,
            motor_dict   = mtr_dict,
            truk_dict    = trk_dict,
            frame_result = result,
            flash_info   = flash_map.get(frame_num),
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
    if args.input:          CONFIG["input_video"]       = args.input
    if args.output:         CONFIG["output_video"]       = args.output
    if args.model:          CONFIG["model_path"]         = args.model
    if args.conf:           CONFIG["conf_threshold"]     = args.conf
    if args.carry_thresh:   CONFIG["carry_iou_thresh"]   = args.carry_thresh
    if args.vehicle_thresh: CONFIG["vehicle_iou_thresh"] = args.vehicle_thresh
    if args.batch_size:     CONFIG["batch_size"]         = args.batch_size

    print("\n" + "=" * 70)
    print("   ES BATU COUNTER v1.1  (Streaming — Low Memory)")
    print("   Counting Es Batu yang Masuk Motor & Truk")
    print("=" * 70)
    print(f"  Model           : {CONFIG['model_path']}")
    print(f"  Input           : {CONFIG['input_video']}")
    print(f"  Output          : {CONFIG['output_video']}")
    print(f"  Conf threshold  : {CONFIG['conf_threshold']}")
    print(f"  Carry threshold : {CONFIG['carry_iou_thresh']}")
    print(f"  Vehicle thresh  : {CONFIG['vehicle_iou_thresh']}")
    print(f"  Min carry frames: {CONFIG['min_carry_frames']}")
    print(f"  Cooldown frames : {CONFIG['cooldown_frames']}")
    print(f"  Batch size      : {CONFIG['batch_size']}")
    print("=" * 70)

    # ---- Load YOLO ----
    from ultralytics import YOLO
    print(f"\n[MAIN] Loading model: {CONFIG['model_path']} ...")
    model = YOLO(CONFIG["model_path"])
    print("[MAIN] Model berhasil dimuat.")

    # ---- Setup ByteTrack ----
    try:
        import supervision as sv
        has_sv = True
        trackers = {
            'es'   : sv.ByteTrack(track_activation_threshold=0.25,
                                   lost_track_buffer=30, frame_rate=30),
            'orang': sv.ByteTrack(track_activation_threshold=0.25,
                                   lost_track_buffer=30, frame_rate=30),
            'motor': sv.ByteTrack(track_activation_threshold=0.25,
                                   lost_track_buffer=60, frame_rate=30),
            'truk' : sv.ByteTrack(track_activation_threshold=0.25,
                                   lost_track_buffer=60, frame_rate=30),
        }
        print("[MAIN] ByteTrack aktif (supervision terinstall).")
    except ImportError:
        has_sv   = False
        trackers = {'es': None, 'orang': None, 'motor': None, 'truk': None}
        print("[MAIN] WARNING: supervision tidak ada. Tracking tanpa ByteTrack.")

    # ---- Inisialisasi Counter ----
    counter = EsBatuCounter(
        carry_iou_thresh    = CONFIG["carry_iou_thresh"],
        vehicle_iou_thresh  = CONFIG["vehicle_iou_thresh"],
        min_carry_frames    = CONFIG["min_carry_frames"],
        cooldown_frames     = CONFIG["cooldown_frames"],
        require_person_first= CONFIG["require_person_first"],
    )

    # ---- Mode REAL-TIME ----
    if args.webcam >= 0:
        print(f"\n[MAIN] MODE REAL-TIME — Source: {args.webcam}")
        run_realtime(
            source   = args.webcam,
            model    = model,
            counter  = counter,
            config   = CONFIG,
            trackers = trackers,
            has_sv   = has_sv,
        )
        return

    # ---- Mode VIDEO FILE (Streaming) ----
    if not os.path.exists(CONFIG["input_video"]):
        print(f"[MAIN] ERROR: File tidak ditemukan: {CONFIG['input_video']}")
        return

    print(f"\n[MAIN] MODE VIDEO FILE — Streaming (Low Memory)")
    run_video_streaming(
        model    = model,
        counter  = counter,
        config   = CONFIG,
        trackers = trackers,
        has_sv   = has_sv,
    )

    # ---- Statistik Akhir ----
    stats = counter.get_statistics()
    counter.print_event_log()

    print("\n" + "=" * 70)
    print("   PIPELINE SELESAI!")
    print("=" * 70)
    print(f"  Output video   : {CONFIG['output_video']}")
    print(f"  Es batu → Motor: {stats['count_motor']}")
    print(f"  Es batu → Truk : {stats['count_truk']}")
    print(f"  Grand Total    : {stats['grand_total']}")
    print(f"  Total events   : {stats['total_events']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
