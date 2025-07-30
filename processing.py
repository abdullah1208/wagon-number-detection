import cv2
import os
import pytesseract
import re
# import numpy
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
from collections import defaultdict

class VideoProcessor:
    def __init__(self):
        self.wagon_model = YOLO("models/segment.pt")
        self.number_model = YOLO("models/wagon_number.pt")
        self.tracker = DeepSort(max_age=30, n_init=5)
        self.processed_wagons = {}
        self.track_history = defaultdict(list)
        self.direction_info = defaultdict(str)
        self.direction_counts = {"Left": 0, "Right": 0}

        os.makedirs("saved_wagons", exist_ok=True)
        os.makedirs("saved_numbers", exist_ok=True)

    def detect_wagons(self, frame):
        class_list = ["wagon"]
        results = self.wagon_model(frame)
        detections = []
        a = results[0].boxes.data
        a = a.detach().cpu().numpy()
        px = pd.DataFrame(a).astype("float")
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            conf = row[4]
            d = int(row[5])
            c = class_list[d]
            if 'wagon' in c:
                detections.append(([x1, y1+20, x2, y2-30], conf, d))
        return detections

    def track_wagons(self, frame, detections):
        return self.tracker.update_tracks(detections, frame=frame)

    def detect_numbers(self, frame, wagon_bbox):
        x1, y1, x2, y2 = wagon_bbox
        wagon_roi = frame[y1:y2, x1:x2]
        if wagon_roi.size == 0:
            return []
        
        results = self.number_model(wagon_roi)
        numbers = []
        for result in results:
            for box in result.boxes:
                nx1, ny1, nx2, ny2 = map(int, box.xyxy[0].cpu().numpy())
                numbers.append([
                    x1 + nx1, 
                    y1 + ny1, 
                    x1 + nx2, 
                    y1 + ny2
                ])
        return numbers
    
    def is_fully_visible(self, bbox, frame_shape):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        buffer = 15
        return (x1 > buffer and y1 > buffer and 
                x2 < w - buffer and y2 < h - buffer)

    def calculate_direction(self, track_id, bbox):
        # Get center coordinates
        center_x = (bbox[0] + bbox[2]) / 2
        
        # Store last 10 positions
        self.track_history[track_id].append(center_x)
        if len(self.track_history[track_id]) > 10:
            self.track_history[track_id].pop(0)
        
        # Calculate direction only when we have enough history
        if len(self.track_history[track_id]) >= 5:
            first_x = self.track_history[track_id][0]
            last_x = self.track_history[track_id][-1]
            
            # Calculate movement difference
            dx = last_x - first_x
            
            # Only register significant movement
            if abs(dx) > 25:  # 25 pixel threshold
                direction = "Right" if dx > 0 else "Left"
                if self.direction_info[track_id] != direction:
                    self.direction_counts[direction] += 1
                    if self.direction_info[track_id]:
                        self.direction_counts[self.direction_info[track_id]] -= 1
                    self.direction_info[track_id] = direction

    def process_ocr(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(
            thresh, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        )
        return re.sub(r'\D', '', text)

    def process_wagon(self, frame, track_id, bbox, results):
        x1, y1, x2, y2 = bbox
        
        # Validate bounding box coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Ensure valid region
        if x2 <= x1 or y2 <= y1:
            return
        
        wagon_img = frame[y1:y2, x1:x2]
        
        # Check if we actually got an image
        if wagon_img.size == 0:
            return
        
        try:
            wagon_path = f"saved_wagons/wagon_{track_id}.jpg"
            resized_img = cv2.resize(wagon_img, (300, 150))
            cv2.imwrite(wagon_path, resized_img)
        except Exception as e:
            print(f"Error processing wagon {track_id}: {str(e)}")
            return

        numbers = self.detect_numbers(frame, bbox)
        best_number = "missing"
        num_path = "missing"

        for num_bbox in numbers:
            nx1, ny1, nx2, ny2 = num_bbox
            # Validate number bounding box
            nx1 = max(x1, nx1)
            ny1 = max(y1, ny1)
            nx2 = min(x2, nx2)
            ny2 = min(y2, ny2)
            
            if nx2 <= nx1 or ny2 <= ny1:
                continue
                
            num_img = frame[ny1:ny2, nx1:nx2]
            if num_img.size == 0:
                continue

            try:
                number_text = self.process_ocr(num_img)
                if number_text and 9 <= len(number_text) <= 11:
                    best_number = number_text
                    num_path = f"saved_numbers/wagon_{track_id}_num_{number_text}.jpg"
                    cv2.imwrite(num_path, cv2.resize(num_img, (100, 50)))
                    break
            except Exception as e:
                print(f"Error processing number for wagon {track_id}: {str(e)}")

        results.append({
            "track_id": track_id,
            "wagon_image": wagon_path,
            "number_image": num_path if num_path != "missing" else "missing",
            "wagon_number": best_number,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "direction": ""  # Leave empty, we'll populate later
        })
        self.processed_wagons[track_id] = True


    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        results = []
        frame_skip = 3  # Process every 3rd frame
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for better tracking
            for _ in range(frame_skip):
                cap.read()

            detections = self.detect_wagons(frame)
            tracks = self.track_wagons(frame, detections)
            processed_frame = frame.copy()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_ltrb()
                bbox = list(map(int, bbox))

                self.calculate_direction(track_id, bbox)
                cv2.rectangle(processed_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                if track_id not in self.processed_wagons:
                    self.process_wagon(frame, track_id, bbox, results)

            yield processed_frame, results

        cap.release()
        # Determine final direction
        self.train_direction = "Right" if self.direction_counts["Right"] > self.direction_counts["Left"] else "Left"


    