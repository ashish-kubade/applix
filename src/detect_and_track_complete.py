import cv2
import numpy as np
import cv2
import sys

from ultralytics import YOLO
from collections import OrderedDict
# Ensure that the track_utils module is correctly imported by checking the following:
# 1. The track_utils directory should contain an __init__.py file to be recognized as a package.
# 2. The path to the track_utils module should be correctly set in PYTHONPATH or relative to the script.



# Now, you should be able to import track_util functions
from track_utils import track_utils

ckpt_path = sys.argv[1]
video_path = sys.argv[2]
model = YOLO(ckpt_path)

# Open the video file




method = 'lucas_kanade'
class KalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)

    def update(self, measurement):
        prediction = self.kalman.predict()
        estimate = self.kalman.correct(np.array(measurement, dtype=np.float32).reshape(2, 1))
        return estimate[:2].flatten()

def draw_optical_flow(flow, frame, step=16):
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.4)
    
    return vis

def optical_flow(video_path, method='farneback'):
    bolts_dict = OrderedDict()
    bbox_logs = []
    prev_object_index = 0
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    if not ret:
        print("Failed to read video")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    center = (h // 2, w // 2)
    initial_center_flow = None
    tracker = KalmanTracker()
    center_flow = np.zeros(2)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True)
        boxes = []
        for i, box in enumerate(results[0].boxes.xywh):
            x, y = int(box[0]), int(box[1])
            w, h = int(box[2]), int(box[3])
            x2, y2 = x + w, y + h
            boxes.append([x, y, x2, y2, center_flow[0], center_flow[1]])
        adjusted_results = track_utils.adjust_results(boxes)
        bolts_dict, new_object_index = track_utils.update_dict(bolts_dict, adjusted_results, prev_object_index)
        prev_object_index = new_object_index
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif method == 'lucas_kanade':
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            flow = np.zeros_like(prev_gray, dtype=np.float32)
            for i, (new, old) in enumerate(zip(next_pts, prev_pts)):
                a, b = new.ravel()
                c, d = old.ravel()
                flow[int(d), int(c)] = [a - c, b - d]
            prev_pts = next_pts[status == 1].reshape(-1, 1, 2)
        else:
            print("Invalid method specified")
            break
        
        # vis = draw_optical_flow(flow, gray)
        
        center_flow = flow[center]
        if initial_center_flow is None:
            initial_center_flow = center_flow
        
        movement = np.linalg.norm(center_flow - initial_center_flow)
        estimated_position = tracker.update(center_flow)
        print(f"Center movement from first frame: {movement}, Flow vector: {center_flow}, Estimated Position: {estimated_position}")
        
        # cv2.imshow("Optical Flow", vis)
        
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break
        

        prev_gray = gray
    
    cap.release()
    cv2.destroyAllWindows()
    return bolts_dict, bbox_logs

# Example usage
bolts_dict, bbox_logs = optical_flow(video_path, method='farneback')  # Change method to 'lucas_kanade' for LK method
np.save('bolts_dict', bolts_dict)
print('len ', len(bolts_dict))