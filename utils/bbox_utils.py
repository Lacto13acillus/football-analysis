import numpy as np

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_center_of_bbox_bottom(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

def interpolate_ball_positions(ball_tracks):
    """
    Interpolasi posisi bola ketika YOLO gagal mendeteksinya.
    Menggunakan linear interpolation antara frame terdeteksi.
    """
    # Kumpulkan frame yang ada deteksi bola
    ball_positions = []
    for frame_num, ball_dict in enumerate(ball_tracks):
        if 1 in ball_dict and 'bbox' in ball_dict[1]:
            bbox = ball_dict[1]['bbox']
            ball_positions.append({
                'frame': frame_num,
                'bbox': bbox
            })
    
    if len(ball_positions) < 2:
        return ball_tracks
    
    # Interpolasi antar gap
    interpolated = [dict(bt) for bt in ball_tracks]  # deep copy sederhana
    
    for i in range(len(ball_positions) - 1):
        start = ball_positions[i]
        end = ball_positions[i + 1]
        gap = end['frame'] - start['frame']
        
        if gap <= 1:
            continue
        
        # Hanya interpolasi jika gap tidak terlalu besar (max 20 frame ~ 0.67 detik)
        if gap > 20:
            continue
        
        for f in range(start['frame'] + 1, end['frame']):
            t = (f - start['frame']) / gap
            interp_bbox = [
                start['bbox'][0] + t * (end['bbox'][0] - start['bbox'][0]),
                start['bbox'][1] + t * (end['bbox'][1] - start['bbox'][1]),
                start['bbox'][2] + t * (end['bbox'][2] - start['bbox'][2]),
                start['bbox'][3] + t * (end['bbox'][3] - start['bbox'][3]),
            ]
            interpolated[f][1] = {"bbox": interp_bbox}
    
    return interpolated