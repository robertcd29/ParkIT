import os
import torch
import cv2
import time
import requests
from ultralytics import YOLO
from torchvision.ops import nms
from typing import Dict, Any, List
import numpy

API_BASE_URL = "http://56.228.19.103:8000"

PARKING_CONFIG = {
    "P1": {"parking_number": "100000058", "window_name": "Parking Spot 1 (Upper-Right)"},
    "P2": {"parking_number": "100000081", "window_name": "Parking Spot 2 (Lower-Left)"},
    "P3": {"parking_number": "100000088", "window_name": "Parking Spot 3 (Lower-Right)"},
}

runs_folder = "C:/Unihack2025/ObDetector/venv/runs/detect"
fallback_video_path = "C:/Unihack2025/ObDetector/venv/test_videos/videoccc.mp4"

CAMERA_INDEX = 1

TARGET_DISPLAY_WIDTH = 960
TARGET_DISPLAY_HEIGHT = 720
DETECTION_RESOLUTION = 1280

train_folders = [f for f in os.listdir(runs_folder) if f.startswith("train")]
train_folders.sort()
if not train_folders:
    raise ValueError(f"Niciun folder 'train' găsit în {runs_folder}")

last_train = train_folders[-1]
model_path = os.path.join(runs_folder, last_train, "weights", "best.pt")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Modelul {model_path} nu există!")

print(f"Foloseste modelul: {model_path}")

model = YOLO(model_path)
print(model.names)

def simplify_class_name(name: str) -> str:
    name = name.lower()
    if "empty" in name:
        return "empty"
    elif "occupied" in name:
        return "occupied"
    return name

def send_to_api(parking_number: str, free_count: int, occupied_count: int):
    total_spots = free_count + occupied_count
    payload = {
        "parking_number": int(parking_number),
        "free_spots": free_count,
        "total_spots": total_spots
    }
    try:
        response = requests.post(f"{API_BASE_URL}/api/detection", json=payload, timeout=5)
        if response.status_code == 200:
            print(f"Date trimise pentru Parking ID {parking_number}: {payload}")
        else:
            print(f"Eroare API ({response.status_code}) pentru {parking_number}: {response.text}")
    except Exception as e:
        print(f"Eroare conexiune API pentru {parking_number}: {e}")

def open_camera_source():
    print(f"Deschid camera la index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    time.sleep(0.3)
    if cap.isOpened():
        print("Camera conectata.")
        return cap

    print("Fallback CAP_DSHOW...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    time.sleep(0.3)
    if cap.isOpened():
        print("Camera conectata CAP_DSHOW.")
        return cap

    print("Folosesc fallback video.")
    return cv2.VideoCapture(fallback_video_path)

def process_and_display_parking(frame_crop: Any, parking_id: str, is_update_time: bool) -> Dict[str, Any]:
    global last_results

    current_results = last_results.get(
        parking_id,
        {"free_count": 0, "occupied_count": 0, "detections": []}
    )

    h_orig, w_orig = frame_crop.shape[:2]

    detection_w = DETECTION_RESOLUTION
    detection_h = int(h_orig * (DETECTION_RESOLUTION / w_orig))

    frame_for_detection = cv2.resize(frame_crop, (detection_w, detection_h), interpolation=cv2.INTER_AREA)

    if is_update_time:
        print(f"YOLO pentru {parking_id}...")

        results = model.predict(source=frame_for_detection, conf=0.3, verbose=False)

        free_count = occupied_count = 0
        detections = []

        scale_x = w_orig / detection_w
        scale_y = h_orig / detection_h

        for r in results:
            boxes = r.boxes.xyxy.cpu()
            scores = r.boxes.conf.cpu()
            classes = r.boxes.cls.cpu()

            keep = nms(boxes, scores, iou_threshold=0.4)
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

            for box, score, cls in zip(boxes, scores, classes):
                simple = simplify_class_name(model.names[int(cls)])

                x1, y1, x2, y2 = (box * torch.tensor([scale_x, scale_y, scale_x, scale_y])).int().tolist()

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "score": float(score),
                    "label": f"{simple} {float(score):.2f}",
                    "simple_name": simple
                })

                if simple == "empty":
                    free_count += 1
                elif simple == "occupied":
                    occupied_count += 1

        send_to_api(PARKING_CONFIG[parking_id]["parking_number"], free_count, occupied_count)

        current_results = {
            "free_count": free_count,
            "occupied_count": occupied_count,
            "detections": detections
        }
        last_results[parking_id] = current_results

        del results
        torch.cuda.empty_cache()

    info_bar_height = 90
    total_width = TARGET_DISPLAY_WIDTH
    total_height = TARGET_DISPLAY_HEIGHT + info_bar_height

    display_frame = numpy.full((total_height, total_width, 3), 230, numpy.uint8)
    resized_frame = cv2.resize(frame_crop, (TARGET_DISPLAY_WIDTH, TARGET_DISPLAY_HEIGHT))
    display_frame[info_bar_height:info_bar_height+TARGET_DISPLAY_HEIGHT] = resized_frame

    free_c = current_results["free_count"]
    occ_c = current_results["occupied_count"]
    total_c = free_c + occ_c

    text = f"[{parking_id}] Free: {free_c} | Occupied: {occ_c} | Total: {total_c}"

    cv2.putText(display_frame, text, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)

    scale_x_disp = TARGET_DISPLAY_WIDTH / w_orig
    scale_y_disp = TARGET_DISPLAY_HEIGHT / h_orig

    for det in current_results["detections"]:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        color = (0, 255, 0) if det["simple_name"] == "empty" else (0, 0, 255)

        x1d = int(x1 * scale_x_disp)
        y1d = int(y1 * scale_y_disp) + info_bar_height
        x2d = int(x2 * scale_x_disp)
        y2d = int(y2 * scale_y_disp) + info_bar_height

        cv2.rectangle(display_frame, (x1d, y1d), (x2d, y2d), color, 3)
        cv2.putText(display_frame, label, (x1d, y1d - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(PARKING_CONFIG[parking_id]["window_name"], display_frame)

    return current_results

cap = open_camera_source()
if not cap.isOpened():
    raise ValueError("Nu s-a putut deschide nicio sursă video.")

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
update_interval = 1
last_update_time = 0

last_results = {
    "P1": {"free_count": 0, "occupied_count": 0, "detections": []},
    "P2": {"free_count": 0, "occupied_count": 0, "detections": []},
    "P3": {"free_count": 0, "occupied_count": 0, "detections": []},
}

cv2.namedWindow(PARKING_CONFIG["P1"]["window_name"], cv2.WINDOW_NORMAL)
cv2.namedWindow(PARKING_CONFIG["P2"]["window_name"], cv2.WINDOW_NORMAL)
cv2.namedWindow(PARKING_CONFIG["P3"]["window_name"], cv2.WINDOW_NORMAL)

print("Pornim analiza în timp real...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Eroare frame.")
        break

    h, w = frame.shape[:2]
    current_time = time.time()

    feed1 = frame[0:int(h/2), int(w/2):w]
    feed2 = frame[int(h/2):h, 0:int(w/2)]
    feed3 = frame[int(h/2):h, int(w/2):w]

    is_update_time = (current_time - last_update_time >= update_interval)
    if is_update_time:
        last_update_time = current_time

    process_and_display_parking(feed1, "P1", is_update_time)
    process_and_display_parking(feed2, "P2", is_update_time)
    process_and_display_parking(feed3, "P3", is_update_time)

    key = cv2.waitKey(int(1000/fps)) & 0xFF
    if key in [ord('q'), 27]:
        print("Stop.")
        break

cap.release()
cv2.destroyAllWindows()
print("Analiza finalizata.")
