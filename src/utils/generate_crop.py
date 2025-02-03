import cv2
import os
import sys


def select_and_crop(image_path, output_folder):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    image = cv2.resize(image, (w//4, h//4))
    clone = image.copy()
    rois = cv2.selectROIs("Select Regions", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for i, (x, y, w, h) in enumerate(rois):
        crop = clone[y:y+h, x:x+w]
        crop_filename = os.path.join(output_folder, f"{base_name}_crop_{i}.png")
        cv2.imwrite(crop_filename, crop)
        print(f"Saved: {crop_filename}")


def process_multiple_images(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing: {image_path}")
        select_and_crop(image_path, output_folder)
    
if __name__ == "__main__":
    image_folder = sys.argv[1]  
    output_folder = sys.argv[2] 
    process_multiple_images(image_folder, output_folder)
