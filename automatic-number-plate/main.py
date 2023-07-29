from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
import util

results = {}

mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]

# Read frames
frame_nmr = -1
ret = True
while ret and frame_nmr < 5:  # Process only first 5 frames
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        print(f"Processing frame {frame_nmr}")
        results[frame_nmr] = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, track_ids)

            if car_id != -1:
                print(f"License plate detected for car ID {car_id}")

                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'frame_nmr': frame_nmr,
                        'car_id': car_id,
                        'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                        'license_plate_bbox': [x1, y1, x2, y2],
                        'license_plate_bbox_score': score,
                        'license_number': license_plate_text,
                        'license_number_score': license_plate_text_score
                    }
                    print(f"License plate text: {license_plate_text}")

# Write results to CSV
output_file = './test.csv'
column_names = [
    'frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score',
    'license_number', 'license_number_score'
]

with open(output_file, 'w') as csv_file:
    csv_file.write(','.join(column_names) + '\n')
    for frame_nmr, frame_results in results.items():
        for car_id, car_data in frame_results.items():
            row_data = [str(frame_nmr)]
            row_data.append(str(car_id))
            row_data.append(','.join(map(str, car_data['car_bbox'])))
            row_data.append(','.join(map(str, car_data['license_plate_bbox'])))
            row_data.append(str(car_data['license_plate_bbox_score']))
            row_data.append(car_data['license_number'])
            row_data.append(str(car_data['license_number_score']))
            csv_file.write(','.join(row_data) + '\n')
