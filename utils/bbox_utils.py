# utils/bbox_utils.py
# Fungsi-fungsi helper matematika untuk bounding box, gate, dan trajectory bola

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


# ============================================================
# FUNGSI DASAR BOUNDING BOX
# ============================================================

def get_center_of_bbox(bbox: List[float]) -> Tuple[int, int]:
    """Hitung titik tengah dari bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_center_of_bbox_bottom(bbox: List[float]) -> Tuple[int, int]:
    """Hitung titik tengah bagian bawah bounding box (posisi kaki)."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def get_bbox_width(bbox: List[float]) -> int:
    """Hitung lebar bounding box."""
    return int(max(0.0, bbox[2] - bbox[0]))


def get_foot_position(bbox: List[float]) -> Tuple[int, int]:
    """Alias untuk get_center_of_bbox_bottom - posisi kaki pemain."""
    return get_center_of_bbox_bottom(bbox)


def measure_distance(p1: Tuple, p2: Tuple) -> float:
    """Hitung jarak Euclidean antara dua titik."""
    return float(np.linalg.norm(
        np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
    ))


def bbox_area(bbox: List[float]) -> float:
    """Hitung luas area bounding box."""
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


# ============================================================
# FUNGSI GATE & TRAJECTORY
# ============================================================

def segments_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float]
) -> bool:
    """
    Cek apakah segmen garis P1-P2 berpotongan dengan segmen garis P3-P4.
    Menggunakan metode Cross Product (orientasi) - lebih akurat dari formula parametrik.
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def on_segment(p, q, r):
        # Cek apakah titik q berada di antara p dan r (kasus collinear)
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    # Kasus umum: segmen saling berpotongan
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Handle kasus collinear (titik tepat di garis)
    if d1 == 0 and on_segment(p3, p1, p4): return True
    if d2 == 0 and on_segment(p3, p2, p4): return True
    if d3 == 0 and on_segment(p1, p3, p2): return True
    if d4 == 0 and on_segment(p1, p4, p2): return True

    return False


def point_to_segment_distance(
    point: Tuple[float, float],
    seg_a: Tuple[float, float],
    seg_b: Tuple[float, float]
) -> float:
    """
    Hitung jarak tegak lurus terpendek dari sebuah titik ke segmen garis.
    Digunakan sebagai fallback proximity check saat trajectory tidak lengkap.
    """
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b

    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy

    # Jika gate adalah titik (panjang = 0), return jarak ke titik itu
    if len_sq == 0:
        return measure_distance(point, seg_a)

    # Hitung parameter proyeksi (t=0 di titik A, t=1 di titik B)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy

    return measure_distance(point, (proj_x, proj_y))


def extract_ball_trajectory(
    tracks: Dict,
    frame_start: int,
    frame_end: int,
    buffer_frames: int = 5
) -> List[Tuple[int, int]]:
    """
    Ambil semua titik tengah bola dari frame_start hingga frame_end.
    Menggunakan trajectory NYATA (titik per frame), bukan garis lurus.

    Args:
        tracks       : dict hasil tracker berisi tracks['ball'][frame_num]
        frame_start  : frame awal event passing
        frame_end    : frame akhir event passing
        buffer_frames: frame tambahan sebelum/sesudah event untuk akurasi

    Returns:
        List of (x, y) koordinat titik tengah bola per frame
    """
    trajectory = []
    total_frames = len(tracks['ball'])

    # Tambah buffer agar trajectory lebih lengkap
    start = max(0, frame_start - buffer_frames)
    end = min(total_frames - 1, frame_end + buffer_frames)

    for f in range(start, end + 1):
        ball_data = tracks['ball'][f].get(1)
        if ball_data and 'bbox' in ball_data:
            cx, cy = get_center_of_bbox(ball_data['bbox'])
            trajectory.append((cx, cy))

    return trajectory


def stabilize_cone_positions(
    tracks: Dict,
    cone_key: str = 'cones',
    sample_frames: int = 30
) -> Dict[int, Tuple[float, float]]:
    """
    Hitung posisi rata-rata (stabilized) semua cone dari N frame pertama.
    Karena kamera statis, posisi cone tidak berubah - ini menghilangkan flickering.

    Args:
        tracks       : dict hasil tracker
        cone_key     : key untuk data cone di tracks
        sample_frames: jumlah frame awal yang diambil untuk averaging

    Returns:
        Dict {cone_id: (x_avg, y_avg)} - posisi stabil setiap cone
    """
    cone_positions_raw: Dict[int, List[Tuple[float, float]]] = {}

    total = min(sample_frames, len(tracks.get(cone_key, [])))

    for f in range(total):
        frame_cones = tracks[cone_key][f]
        for cone_id, cone_data in frame_cones.items():
            bbox = cone_data.get('bbox', None)
            if bbox is None:
                continue
            # Gunakan titik bawah tengah cone sebagai referensi posisi
            cx, cy = get_center_of_bbox_bottom(bbox)
            if cone_id not in cone_positions_raw:
                cone_positions_raw[cone_id] = []
            cone_positions_raw[cone_id].append((cx, cy))

    # Rata-ratakan posisi per cone
    stabilized: Dict[int, Tuple[float, float]] = {}
    for cone_id, positions in cone_positions_raw.items():
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        stabilized[cone_id] = (float(np.mean(xs)), float(np.mean(ys)))

    return stabilized


def identify_gate_cones(
    stabilized_cones: Dict[int, Tuple[float, float]],
    gate_hint: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    manual_cone_ids: Optional[Tuple[int, int]] = None,
    expected_gate_width_range: Tuple[float, float] = (60.0, 300.0)
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Identifikasi 2 cone yang membentuk gawang dari semua cone yang terdeteksi.

    STRATEGI (prioritas berurutan):
    1. Manual cone IDs  - langsung gunakan ID yang ditentukan (PALING AKURAT)
    2. Gate hint        - cari 2 cone terdekat ke titik perkiraan gate
    3. Auto isolation   - cari pasangan cone paling terisolasi dalam range lebar valid

    Args:
        stabilized_cones        : output dari stabilize_cone_positions()
        gate_hint               : perkiraan 2 titik gate ((x1,y1),(x2,y2)) atau None
        manual_cone_ids         : tuple 2 cone ID jika sudah diketahui, atau None
        expected_gate_width_range: (min_px, max_px) lebar gate yang valid

    Returns:
        Tuple (pos_cone_kiri, pos_cone_kanan) diurutkan kiri->kanan, atau None
    """
    if not stabilized_cones or len(stabilized_cones) < 2:
        return None

    cone_ids = list(stabilized_cones.keys())
    cone_positions = list(stabilized_cones.values())

    # --- STRATEGI 1: Manual ID override ---
    if manual_cone_ids is not None:
        id_a, id_b = manual_cone_ids
        if id_a in stabilized_cones and id_b in stabilized_cones:
            pos_a = stabilized_cones[id_a]
            pos_b = stabilized_cones[id_b]
            print(f"[GATE] Menggunakan manual cone IDs: {id_a} dan {id_b}")
            print(f"[GATE] Posisi gate: {pos_a} <-> {pos_b}")
            if pos_a[0] < pos_b[0]:
                return pos_a, pos_b
            return pos_b, pos_a
        else:
            print(f"[GATE] WARNING: Manual cone ID {id_a} atau {id_b} tidak ditemukan!")

    # --- STRATEGI 2: Gate hint (titik perkiraan lokasi gate) ---
    if gate_hint is not None:
        hint_center_x = (gate_hint[0][0] + gate_hint[1][0]) / 2
        hint_center_y = (gate_hint[0][1] + gate_hint[1][1]) / 2
        hint_center = (hint_center_x, hint_center_y)

        distances_to_hint = []
        for cid, pos in stabilized_cones.items():
            d = measure_distance(pos, hint_center)
            distances_to_hint.append((cid, pos, d))
        distances_to_hint.sort(key=lambda x: x[2])

        if len(distances_to_hint) >= 2:
            c1_id, c1_pos, _ = distances_to_hint[0]
            c2_id, c2_pos, _ = distances_to_hint[1]
            gate_width = measure_distance(c1_pos, c2_pos)
            min_w, max_w = expected_gate_width_range

            if min_w <= gate_width <= max_w:
                print(f"[GATE] Terdeteksi via hint: ID {c1_id} & {c2_id}, lebar={gate_width:.1f}px")
                if c1_pos[0] < c2_pos[0]:
                    return c1_pos, c2_pos
                return c2_pos, c1_pos
            else:
                print(f"[GATE] WARNING: 2 cone terdekat hint lebar tidak valid: {gate_width:.1f}px")

    # --- STRATEGI 3: Otomatis berbasis isolasi ---
    best_pair = None
    best_isolation_score = -1
    min_w, max_w = expected_gate_width_range

    for i in range(len(cone_ids)):
        for j in range(i + 1, len(cone_ids)):
            pos_i = cone_positions[i]
            pos_j = cone_positions[j]
            pair_dist = measure_distance(pos_i, pos_j)

            if not (min_w <= pair_dist <= max_w):
                continue

            pair_center = (
                (pos_i[0] + pos_j[0]) / 2,
                (pos_i[1] + pos_j[1]) / 2
            )
            other_distances = []
            for k in range(len(cone_ids)):
                if k == i or k == j:
                    continue
                d = measure_distance(pair_center, cone_positions[k])
                other_distances.append(d)

            isolation = min(other_distances) if other_distances else 9999.0

            if isolation > best_isolation_score:
                best_isolation_score = isolation
                best_pair = (pos_i, pos_j)

    if best_pair:
        pos_a, pos_b = best_pair
        print(f"[GATE] Terdeteksi via isolasi otomatis, isolation_score={best_isolation_score:.1f}px")
        if pos_a[0] < pos_b[0]:
            return pos_a, pos_b
        return pos_b, pos_a

    print("[GATE] GAGAL mengidentifikasi gate cone!")
    return None


def check_ball_passed_through_gate(
    ball_trajectory: List[Tuple[int, int]],
    gate_cone_left: Tuple[float, float],
    gate_cone_right: Tuple[float, float],
    proximity_threshold: float = 40.0,
    min_trajectory_points: int = 3
) -> Tuple[bool, str]:
    """
    Cek apakah trajectory bola melewati celah di antara dua cone gate.

    METODE A - Intersection Check:
        Cek setiap segmen berurutan di trajectory apakah memotong garis gate.
        Ini adalah metode utama yang paling akurat.

    METODE B - Proximity Fallback:
        Cek apakah titik bola pernah sangat dekat dengan garis gate DAN
        berada di zona gate (antara kedua cone). Berguna saat trajectory
        tidak lengkap akibat missed detection.

    Args:
        ball_trajectory       : list (x,y) dari extract_ball_trajectory()
        gate_cone_left        : posisi (x,y) cone kiri gate
        gate_cone_right       : posisi (x,y) cone kanan gate
        proximity_threshold   : jarak maks bola ke garis gate untuk Metode B (pixel)
        min_trajectory_points : minimum titik trajectory agar pengecekan valid

    Returns:
        (passed: bool, reason: str)
    """
    if len(ball_trajectory) < min_trajectory_points:
        return False, f"Trajectory terlalu pendek ({len(ball_trajectory)} titik, min={min_trajectory_points})"

    gate_left = gate_cone_left
    gate_right = gate_cone_right

    gate_min_x = min(gate_left[0], gate_right[0])
    gate_max_x = max(gate_left[0], gate_right[0])
    gate_min_y = min(gate_left[1], gate_right[1])
    gate_max_y = max(gate_left[1], gate_right[1])

    # --- METODE A: Segment Intersection ---
    for i in range(len(ball_trajectory) - 1):
        p1 = ball_trajectory[i]
        p2 = ball_trajectory[i + 1]

        if segments_intersect(p1, p2, gate_left, gate_right):
            # Verifikasi: titik perpotongan harus berada dalam batas gate
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2

            x_tolerance = (gate_max_x - gate_min_x) * 0.2
            y_tolerance = max((gate_max_y - gate_min_y) * 0.2, 15.0)

            in_x_range = (gate_min_x - x_tolerance) <= mid_x <= (gate_max_x + x_tolerance)
            in_y_range = (gate_min_y - y_tolerance) <= mid_y <= (gate_max_y + y_tolerance)

            if in_x_range or in_y_range:
                return True, f"Metode A: Intersection di segmen trajectory ke-{i}"

    # --- METODE B: Proximity Fallback ---
    for i, point in enumerate(ball_trajectory):
        dist = point_to_segment_distance(point, gate_left, gate_right)

        if dist <= proximity_threshold:
            px, py = point
            in_gate_zone = (
                (gate_min_x - 10 <= px <= gate_max_x + 10) or
                (gate_min_y - 10 <= py <= gate_max_y + 10)
            )
            if in_gate_zone:
                return True, f"Metode B: Proximity {dist:.1f}px di titik trajectory ke-{i}"

    return False, "Bola tidak melewati gate"


def interpolate_ball_positions(
    ball_positions: List[Dict[int, Dict[str, Any]]]
) -> List[Dict[int, Dict[str, Any]]]:
    """
    Interpolasi posisi bola untuk mengisi frame di mana bola tidak terdeteksi.
    Menggunakan pandas interpolate() untuk smooth filling.
    """
    if not ball_positions:
        return ball_positions

    n = len(ball_positions)
    xs1, ys1, xs2, ys2 = [], [], [], []

    for f in range(n):
        bb = ball_positions[f].get(1, {}).get("bbox", None)
        if bb is None:
            xs1.append(np.nan); ys1.append(np.nan)
            xs2.append(np.nan); ys2.append(np.nan)
        else:
            xs1.append(float(bb[0])); ys1.append(float(bb[1]))
            xs2.append(float(bb[2])); ys2.append(float(bb[3]))

    df = pd.DataFrame({"x1": xs1, "y1": ys1, "x2": xs2, "y2": ys2})
    df = df.interpolate(limit_direction="both")

    out: List[Dict[int, Dict[str, Any]]] = []
    for f in range(n):
        x1, y1, x2, y2 = df.loc[f, ["x1", "y1", "x2", "y2"]].tolist()
        if np.isnan(x1):
            out.append({})
            continue
        out.append({1: {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": 1.0
        }})
    return out

def identify_target_cone(
    stabilized_cones     : Dict[int, Tuple[float, float]],
    manual_target_cone_id: Optional[int] = None,
    selection_mode       : str = "highest"  # "highest" = Y terkecil = paling atas
) -> Optional[Tuple[int, Tuple[float, float]]]:
    """
    Identifikasi satu cone target dari semua cone yang terdeteksi.
    STRATEGI (prioritas berurutan):
    1. Manual ID  - langsung gunakan cone ID yang ditentukan (PALING AKURAT)
    2. Auto       - pilih cone berdasarkan selection_mode:
                    "highest" = cone dengan Y terkecil (paling atas di layar)
                    "lowest"  = cone dengan Y terbesar (paling bawah)
                    "leftmost"= cone dengan X terkecil (paling kiri)
    Args:
        stabilized_cones     : output dari stabilize_cone_positions()
        manual_target_cone_id: cone ID jika sudah diketahui, atau None
        selection_mode       : metode auto-selection jika manual tidak diset
    Returns:
        (cone_id, (x, y)) atau None jika gagal
    """
    if not stabilized_cones:
        return None
    # --- STRATEGI 1: Manual ID ---
    if manual_target_cone_id is not None:
        if manual_target_cone_id in stabilized_cones:
            pos = stabilized_cones[manual_target_cone_id]
            print(f"[TARGET] Menggunakan manual target cone ID: {manual_target_cone_id}")
            print(f"[TARGET] Posisi target: ({pos[0]:.1f}, {pos[1]:.1f})")
            return manual_target_cone_id, pos
        else:
            print(f"[TARGET] WARNING: Manual cone ID {manual_target_cone_id} "
                  f"tidak ditemukan!")
    # --- STRATEGI 2: Auto berdasarkan selection_mode ---
    if selection_mode == "highest":
        # Y terkecil = paling atas di layar (koordinat image)
        best_id  = min(stabilized_cones, key=lambda k: stabilized_cones[k][1])
    elif selection_mode == "lowest":
        best_id  = max(stabilized_cones, key=lambda k: stabilized_cones[k][1])
    elif selection_mode == "leftmost":
        best_id  = min(stabilized_cones, key=lambda k: stabilized_cones[k][0])
    elif selection_mode == "rightmost":
        best_id  = max(stabilized_cones, key=lambda k: stabilized_cones[k][0])
    else:
        print(f"[TARGET] WARNING: selection_mode '{selection_mode}' tidak dikenal!")
        return None
    pos = stabilized_cones[best_id]
    print(f"[TARGET] Auto-select cone ID {best_id} via mode='{selection_mode}'")
    print(f"[TARGET] Posisi target: ({pos[0]:.1f}, {pos[1]:.1f})")
    return best_id, pos
def check_ball_reached_target_cone(
    ball_trajectory  : List[Tuple[int, int]],
    target_cone_pos  : Tuple[float, float],
    proximity_radius : float = 120.0,
    min_points       : int = 3
) -> Tuple[bool, str]:
    """
    Cek apakah trajectory bola mendekati cone target dalam radius tertentu.
    Berbeda dari gate intersection, di sini kita cek apakah
    ADA TITIK MANAPUN dalam trajectory yang jaraknya <= proximity_radius
    dari posisi cone target.
    Juga menghitung 'closest approach distance' untuk debugging.
    Args:
        ball_trajectory : list (x, y) dari extract_ball_trajectory()
        target_cone_pos : posisi (x, y) cone target
        proximity_radius: radius keberhasilan dalam pixel
        min_points      : minimum titik trajectory agar valid
    Returns:
        (reached: bool, reason: str)
    """
    if len(ball_trajectory) < min_points:
        return False, (f"Trajectory terlalu pendek "
                       f"({len(ball_trajectory)} titik, min={min_points})")
    # Cari jarak terdekat bola ke target cone sepanjang trajectory
    min_distance = float('inf')
    min_idx      = -1
    for i, point in enumerate(ball_trajectory):
        dist = measure_distance(point, target_cone_pos)
        if dist < min_distance:
            min_distance = dist
            min_idx      = i
    if min_distance <= proximity_radius:
        return True, (f"Bola mendekati target: jarak minimum "
                      f"{min_distance:.1f}px di titik ke-{min_idx} "
                      f"(radius={proximity_radius:.0f}px)")
    else:
        return False, (f"Bola tidak mencapai target: jarak minimum "
                       f"{min_distance:.1f}px > radius {proximity_radius:.0f}px")