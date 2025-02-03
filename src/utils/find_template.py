import cv2
import numpy as np
import os
import sys

def preprocess_frame(frame):
    """Convert to grayscale, apply CLAHE and Gaussian blur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
    return blurred

def template_matching(frame, template):
    """Perform multi-scale template matching."""
    best_match = None
    best_val = -np.inf
    
    for scale in np.linspace(0.5, 1.5, 10):  # Scale variations
        resized_template = cv2.resize(template, None, fx=scale, fy=scale)
        res = cv2.matchTemplate(frame, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_val:
            best_val = max_val
            best_match = max_loc
            best_scale = scale
    
    return best_match, best_scale, best_val

def detect_edges_and_contours(frame):
    """Detect edges and contours for refining detection."""
    edges = cv2.Canny(frame, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def process_video(video_path, template_path, dump_path):
    """Process video frame-by-frame."""
    cap = cv2.VideoCapture(video_path)
    template = cv2.imread(template_path, 0)
    template = cv2.resize(template, None, fx=5, fy=5)
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = preprocess_frame(frame)
        match_loc, scale, confidence = template_matching(processed, template)
        contours = detect_edges_and_contours(processed)
        
        if confidence > 0.88:  # Confidence threshold
            h, w = template.shape
            h, w = int(h * scale), int(w * scale)
            cv2.rectangle(frame, match_loc, (match_loc[0] + w, match_loc[1] + h), (0, 255, 0), 2)
            print('located ', match_loc, scale, confidence, template_path)
            output_name = os.path.join(dump_path, os.path.basename(video_path) + '_' + os.path.basename(template_path) + 
                                       f'_{match_loc[0]}_{match_loc[1]}_{scale}_{confidence}.jpg')
            cv2.imwrite(output_name, frame)
        cv2.imshow("Bolt Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = sys.argv[1]
template_root = sys.argv[2]
dump_path = sys.argv[3]
templates = os.listdir(template_root)

for template in templates:
    template_path = os.path.join(template_root, template)
    
    process_video(video_path, template_path, dump_path)
