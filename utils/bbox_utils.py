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



def interpolate_ball_positions(ball_positions):
    """
    Interpolasi posisi bola yang hilang antar frame.
    Diperluas: gap limit dari 20 -> 50 untuk menangani motion blur saat tendangan kencang.
    """
    # Kumpulkan frame yang ada datanya
    ball_dict = {}
    for i, frame_data in enumerate(ball_positions):
        if 1 in frame_data and 'bbox' in frame_data[1]:
            ball_dict[i] = frame_data[1]['bbox']

    if len(ball_dict) < 2:
        return ball_positions

    frames = sorted(ball_dict.keys())
    
    # Interpolasi per komponen bbox (x1, y1, x2, y2)
    interpolated = dict(ball_dict)
    
    for idx in range(len(frames) - 1):
        start_frame = frames[idx]
        end_frame = frames[idx + 1]
        gap = end_frame - start_frame

        # DIUBAH: dari 20 -> 50 frame, agar bola yang hilang karena
        # motion blur saat tendangan kencang tetap bisa di-interpolasi
        if gap <= 1 or gap > 50:
            continue

        start_bbox = np.array(ball_dict[start_frame])
        end_bbox = np.array(ball_dict[end_frame])

        for f in range(start_frame + 1, end_frame):
            t = (f - start_frame) / gap
            interp_bbox = start_bbox + t * (end_bbox - start_bbox)
            interpolated[f] = interp_bbox.tolist()

    # Rebuild ball_positions
    result = []
    for i in range(len(ball_positions)):
        if i in interpolated:
            result.append({1: {'bbox': interpolated[i]}})
        else:
            result.append(ball_positions[i])

    return result