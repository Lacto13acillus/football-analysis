# trackers/es_batu_counter.py
# Logic penghitungan es batu yang dibawa orang masuk ke motor atau truk.
#
# LOGIKA UTAMA:
#   1. Deteksi "orang membawa es batu": bbox es batu overlap dengan bbox orang
#      (IoU > threshold ATAU center es batu berada di dalam bbox orang).
#
#   2. Deteksi "masuk kendaraan": es batu yang sebelumnya terdeteksi dibawa orang
#      kemudian overlap dengan bbox motor/truk (association berdasarkan IoU).
#
#   3. Counting: Setiap es batu yang melakukan transisi
#      (carried_by_person → overlap_vehicle) dihitung 1x per ID.
#      Cooldown per ID mencegah penghitungan ganda.

from typing import Dict, List, Tuple, Optional, Set
import numpy as np


# ============================================================
# UTILITY BBOX
# ============================================================

def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Hitung Intersection-over-Union antara dua bbox [x1,y1,x2,y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _intersection_ratio(box_small: List[float], box_large: List[float]) -> float:
    """
    Hitung rasio: luas irisan / luas box_small.
    Berguna untuk mendeteksi apakah objek kecil (es batu) berada di dalam objek besar.
    """
    xa1, ya1, xa2, ya2 = box_small
    xb1, yb1, xb2, yb2 = box_large

    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_small = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)

    return inter / area_small if area_small > 0 else 0.0


def _center(bbox: List[float]) -> Tuple[float, float]:
    """Kembalikan titik tengah bbox."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _point_in_box(point: Tuple[float, float], box: List[float]) -> bool:
    """True jika titik berada di dalam bbox."""
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


# ============================================================
# MAIN COUNTER CLASS
# ============================================================

class EsBatuCounter:
    """
    Menghitung es batu yang dibawa orang dan masuk ke motor atau truk.

    State machine per es-batu ID:
        IDLE → CARRIED → ENTERING_VEHICLE → COUNTED (cooldown)

    Parameter:
        carry_iou_thresh      : Min intersection_ratio (es-batu di dalam orang) → "dibawa"
        vehicle_iou_thresh    : Min intersection_ratio (es-batu di dalam kendaraan) → "masuk"
        min_carry_frames      : Minimum frame es-batu harus terdeteksi dibawa sebelum bisa dihitung
        cooldown_frames       : Setelah terhitung, ID ini di-lock selama N frame
        require_person_first  : Jika True, es-batu HARUS melewati fase "dibawa orang" dulu
    """

    # State
    STATE_IDLE    = 'IDLE'
    STATE_CARRIED = 'CARRIED'
    STATE_COUNTED = 'COUNTED'

    def __init__(
        self,
        carry_iou_thresh   : float = 0.10,
        vehicle_iou_thresh : float = 0.10,
        min_carry_frames   : int   = 3,
        cooldown_frames    : int   = 60,
        require_person_first: bool = True,
    ):
        self.carry_iou_thresh    = carry_iou_thresh
        self.vehicle_iou_thresh  = vehicle_iou_thresh
        self.min_carry_frames    = min_carry_frames
        self.cooldown_frames     = cooldown_frames
        self.require_person_first = require_person_first

        # Counters
        self.count_motor: int = 0
        self.count_truk : int = 0

        # State per es-batu ID
        #   state         : STATE_IDLE | STATE_CARRIED | STATE_COUNTED
        #   carry_frames  : berapa frame sudah dalam kondisi carried
        #   cooldown_left : sisa frame cooldown
        #   carried_by    : set of person IDs yang membawa es batu ini
        self._state: Dict[int, Dict] = {}

        # Event log
        self.events: List[Dict] = []

    # -------------------------------------------------------
    # RESET
    # -------------------------------------------------------

    def reset(self) -> None:
        """Reset semua state dan counter."""
        self.count_motor = 0
        self.count_truk  = 0
        self._state      = {}
        self.events      = []

    # -------------------------------------------------------
    # HELPER: inisialisasi state untuk ID baru
    # -------------------------------------------------------

    def _init_state(self, es_id: int) -> None:
        if es_id not in self._state:
            self._state[es_id] = {
                'state'       : self.STATE_IDLE,
                'carry_frames': 0,
                'cooldown_left': 0,
                'carried_by'  : set(),
            }

    # -------------------------------------------------------
    # PROCESS SATU FRAME
    # -------------------------------------------------------

    def process_frame(
        self,
        frame_num   : int,
        es_batu_dict: Dict[int, Dict],   # {id: {'bbox': [...], 'conf': float}}
        orang_dict  : Dict[int, Dict],
        motor_dict  : Dict[int, Dict],
        truk_dict   : Dict[int, Dict],
    ) -> Dict:
        """
        Proses satu frame.

        Returns:
            Dict dengan info frame ini:
            {
                'frame_num'     : int,
                'count_motor'   : int,
                'count_truk'    : int,
                'es_states'     : {id: state_str},
                'new_motor'     : list of es_batu IDs yang baru dihitung ke motor,
                'new_truk'      : list of es_batu IDs yang baru dihitung ke truk,
            }
        """
        new_motor: List[int] = []
        new_truk : List[int] = []
        es_states: Dict[int, str] = {}

        # Tick cooldown semua ID
        for es_id, st in self._state.items():
            if st['cooldown_left'] > 0:
                st['cooldown_left'] -= 1
                if st['cooldown_left'] == 0:
                    # Reset ke IDLE setelah cooldown selesai
                    st['state']        = self.STATE_IDLE
                    st['carry_frames'] = 0
                    st['carried_by']   = set()

        # Ambil semua bbox kendaraan
        motor_bboxes = [v['bbox'] for v in motor_dict.values()]
        truk_bboxes  = [v['bbox'] for v in truk_dict.values()]
        orang_bboxes = [(pid, v['bbox']) for pid, v in orang_dict.items()]

        # Proses setiap es-batu yang terdeteksi di frame ini
        for es_id, es_data in es_batu_dict.items():
            self._init_state(es_id)
            st   = self._state[es_id]
            bbox = es_data['bbox']

            # Skip jika sedang cooldown
            if st['cooldown_left'] > 0:
                es_states[es_id] = f"COOLDOWN({st['cooldown_left']})"
                continue

            # ---- Cek apakah es-batu dibawa orang ----
            is_carried = False
            carrying_persons: Set[int] = set()

            for pid, pbbox in orang_bboxes:
                ratio = _intersection_ratio(bbox, pbbox)
                if ratio >= self.carry_iou_thresh:
                    is_carried = True
                    carrying_persons.add(pid)
                elif _point_in_box(_center(bbox), pbbox):
                    is_carried = True
                    carrying_persons.add(pid)

            if is_carried:
                st['state']        = self.STATE_CARRIED
                st['carry_frames'] += 1
                st['carried_by']   |= carrying_persons
            else:
                if st['state'] == self.STATE_CARRIED:
                    # es-batu baru lepas dari orang, cek apakah langsung ke kendaraan
                    pass
                elif st['state'] == self.STATE_IDLE:
                    st['carry_frames'] = 0

            # ---- Cek apakah es-batu masuk kendaraan ----
            # Syarat: carry_frames >= min_carry_frames (sudah cukup lama dibawa)
            #         ATAU require_person_first=False

            carried_enough = (not self.require_person_first or
                              st['carry_frames'] >= self.min_carry_frames)

            if carried_enough and st['state'] in (self.STATE_CARRIED, self.STATE_IDLE):
                # Cek overlap dengan MOTOR
                for mbbox in motor_bboxes:
                    ratio = _intersection_ratio(bbox, mbbox)
                    if ratio >= self.vehicle_iou_thresh or _point_in_box(_center(bbox), mbbox):
                        # ES BATU MASUK MOTOR!
                        self.count_motor += 1
                        new_motor.append(es_id)
                        st['state']        = self.STATE_COUNTED
                        st['cooldown_left'] = self.cooldown_frames
                        self.events.append({
                            'frame'      : frame_num,
                            'es_batu_id' : es_id,
                            'vehicle'    : 'motor',
                            'carry_frames': st['carry_frames'],
                            'total_motor': self.count_motor,
                            'total_truk' : self.count_truk,
                        })
                        break

                # Cek overlap dengan TRUK (hanya jika belum terhitung ke motor)
                if st['state'] != self.STATE_COUNTED:
                    for tbbox in truk_bboxes:
                        ratio = _intersection_ratio(bbox, tbbox)
                        if ratio >= self.vehicle_iou_thresh or _point_in_box(_center(bbox), tbbox):
                            # ES BATU MASUK TRUK!
                            self.count_truk += 1
                            new_truk.append(es_id)
                            st['state']        = self.STATE_COUNTED
                            st['cooldown_left'] = self.cooldown_frames
                            self.events.append({
                                'frame'      : frame_num,
                                'es_batu_id' : es_id,
                                'vehicle'    : 'truk',
                                'carry_frames': st['carry_frames'],
                                'total_motor': self.count_motor,
                                'total_truk' : self.count_truk,
                            })
                            break

            es_states[es_id] = st['state']

        return {
            'frame_num'  : frame_num,
            'count_motor': self.count_motor,
            'count_truk' : self.count_truk,
            'es_states'  : es_states,
            'new_motor'  : new_motor,
            'new_truk'   : new_truk,
        }

    # -------------------------------------------------------
    # PROCESS SEMUA FRAME SEKALIGUS
    # -------------------------------------------------------

    def process_all_frames(
        self,
        tracks: Dict[str, List[Dict]],
    ) -> List[Dict]:
        """
        Proses semua frame tracks dan kembalikan list hasil per frame.

        Args:
            tracks: Output dari EsBatuTracker.get_object_tracks()

        Returns:
            List[Dict] — hasil per frame (gunakan untuk render)
        """
        total = len(tracks['es_batu'])
        results: List[Dict] = []

        print(f"\n[COUNTER] Memulai proses counting pada {total} frames...")

        for frame_num in range(total):
            if frame_num % 200 == 0:
                print(f"[COUNTER] Progress: {frame_num}/{total} "
                      f"| Motor: {self.count_motor} | Truk: {self.count_truk}")

            result = self.process_frame(
                frame_num    = frame_num,
                es_batu_dict = tracks['es_batu'][frame_num],
                orang_dict   = tracks['orang'][frame_num],
                motor_dict   = tracks['motor'][frame_num],
                truk_dict    = tracks['truk'][frame_num],
            )
            results.append(result)

        print(f"\n[COUNTER] ============================")
        print(f"[COUNTER] HASIL COUNTING ES BATU")
        print(f"[COUNTER] ============================")
        print(f"[COUNTER] Total es batu → Motor : {self.count_motor}")
        print(f"[COUNTER] Total es batu → Truk  : {self.count_truk}")
        print(f"[COUNTER] Grand Total           : {self.count_motor + self.count_truk}")
        print(f"[COUNTER] Total events          : {len(self.events)}")
        print(f"[COUNTER] ============================\n")

        return results

    # -------------------------------------------------------
    # STATISTIK
    # -------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Kembalikan statistik keseluruhan."""
        return {
            'count_motor'  : self.count_motor,
            'count_truk'   : self.count_truk,
            'grand_total'  : self.count_motor + self.count_truk,
            'total_events' : len(self.events),
            'events'       : self.events,
        }

    def print_event_log(self) -> None:
        """Print log semua event counting."""
        sep = "=" * 65
        print(f"\n{sep}")
        print("   LOG EVENT ES BATU MASUK KENDARAAN")
        print(sep)
        print(f"  {'No':<4} {'Frame':<8} {'ES-ID':<8} "
              f"{'Kendaraan':<12} {'Carry(f)':<10} {'Motor':<8} {'Truk':<8}")
        print("  " + "-" * 58)
        for i, ev in enumerate(self.events):
            print(f"  {i+1:<4} {ev['frame']:<8} {ev['es_batu_id']:<8} "
                  f"{ev['vehicle']:<12} {ev['carry_frames']:<10} "
                  f"{ev['total_motor']:<8} {ev['total_truk']:<8}")
        print(f"\n  Total Motor : {self.count_motor}")
        print(f"  Total Truk  : {self.count_truk}")
        print(f"  Grand Total : {self.count_motor + self.count_truk}")
        print(sep + "\n")
