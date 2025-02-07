import numpy as np

def get_effective_position(bolt):
    x,y,x2,y2 = bolt[:4]
    x = x+bolt[4]
    y = y+bolt[5]
    x2 = x2+bolt[4]
    y2 = y2+bolt[5]
    return [x,y,x2,y2]


def adjust_results(detections):
    adjusted_detections = []
    print(detections)
    for detection in detections:
        adjusted_detections.append(get_effective_position(detection))
    return adjusted_detections

def findIoU(rect1, rect2):
    intersection_area = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) * max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    
    iou = intersection_area / float(rect1_area + rect2_area - intersection_area)
    return iou


def find_displacement(rect1, rect2 ):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    dx = (x2 + x1) / 2 - (x4 + x3) / 2
    dy = (y2 + y1) / 2 - (y4 + y3) / 2
    displacement = np.sqrt(dx**2 + dy**2)
    return displacement


def check_if_repeat(candidate_bbox, bolts_dict, IoUThreshold = 0.0001, disp_margin = 100):
    for key, value in bolts_dict.items():
        iou = findIoU(candidate_bbox, value['bbox'])
        if iou > IoUThreshold:
            return True
    else:
        disp = find_displacement(candidate_bbox, value['bbox'])
        if disp < disp_margin:
            return True

    return False

def update_dict(bolts_dict, detections, current_index):
    
    for bolt in detections:
        if len(bolts_dict) == 0:
            bolts_dict[current_index] = {'bbox': bolt}
            current_index += 1
            continue
        else:
            if check_if_repeat(bolt, bolts_dict):
                continue
            else:
                bolts_dict[current_index] = {'bbox': bolt}
                current_index += 1
                continue
    return bolts_dict, current_index
