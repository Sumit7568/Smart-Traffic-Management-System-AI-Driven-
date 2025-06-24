import cv2
import numpy as np
import time
import math

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

vehicle_classes = ['car', 'bus', 'truck', 'motorbike']
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
conversion_factor = 0.05  # pixels to meters (adjust based on video perspective)

# Number of lanes (adjust based on video)
NUM_LANES = 4
lane_length_meters = 20  # Estimate of the visible lane length in meters

# Load video
video_path = '2165-155327596_small.mp4'  # Replace with correct path if needed
cap = cv2.VideoCapture(video_path)

vehicle_tracker = {}

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    height, width, _ = img.shape

    # Dynamic lane setup
    lane_width = width // NUM_LANES
    lanes = [(i * lane_width, 0, (i + 1) * lane_width, height) for i in range(NUM_LANES)]
    lane_counts = [0] * NUM_LANES
    lane_speeds = [[] for _ in range(NUM_LANES)]

    # YOLO Detection
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    current_time = time.time()

    for output in detections:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                label = classes[class_id]
                if label in vehicle_classes:
                    center_x = int(det[0] * width)
                    center_y = int(det[1] * height)
                    w = int(det[2] * width)
                    h = int(det[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cx, cy = x + w // 2, y + h // 2

            # Assign to lane
            lane_idx = None
            for idx, (lx1, _, lx2, _) in enumerate(lanes):
                if lx1 <= cx < lx2:
                    lane_counts[idx] += 1
                    lane_idx = idx
                    break

            # Speed estimation
            object_id = f"{label}_{i}"
            if object_id in vehicle_tracker:
                px, py, pt = vehicle_tracker[object_id]
                dist_px = math.hypot(cx - px, cy - py)
                dist_m = dist_px * conversion_factor
                dt = current_time - pt
                speed_kmh = (dist_m / dt) * 3.6 if dt > 0 else 0
                vehicle_tracker[object_id] = (cx, cy, current_time)
            else:
                speed_kmh = 0
                vehicle_tracker[object_id] = (cx, cy, current_time)

            # Save speed to lane
            if lane_idx is not None:
                lane_speeds[lane_idx].append(speed_kmh)

            # Draw
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {int(speed_kmh)} km/h", (x, y - 10), font, 2, (255, 255, 255), 2)

    # Draw lanes and vehicle count
    for i, (x1, y1, x2, y2) in enumerate(lanes):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, f"Lane {i + 1}: {lane_counts[i]}", (x1 + 10, 40), font, 2, (0, 255, 0), 2)

    # Calculate and print time to clear each lane
    for i, speeds in enumerate(lane_speeds):
        if speeds:
            avg_speed_kmh = sum(speeds) / len(speeds)
            avg_speed_mps = avg_speed_kmh * (1000 / 3600)
            clearing_time = lane_length_meters / avg_speed_mps if avg_speed_mps > 0 else float('inf')
            print(f"Lane {i + 1}: Avg Speed = {avg_speed_kmh:.2f} km/h, Time to Clear = {clearing_time:.2f} sec")
        else:
            print(f"Lane {i + 1}: No vehicles detected.")

    # Show output
    cv2.imshow("Dynamic Lane Detection with Speed", img)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
