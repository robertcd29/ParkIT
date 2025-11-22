import os
import torch
import cv2
import time
import requests
from ultralytics import YOLO
from torchvision.ops import nms

API_BASE_URL = "http://192.168.1.165:8000"
PARKING_NUMBER = "5"

runs_folder = "C:/Unihack2025/ObDetector/venv/runs/detect"
video_path = "C:/Unihack2025/ObDetector/venv/test_videos/videoccc.mp4"

train_folders = [f for f in os.listdir(runs_folder) if f.startswith("train")]
train_folders.sort()
if not train_folders:
    raise ValueError(f"Niciun folder 'train' găsit în {runs_folder}")

last_train = train_folders[-1]
model_path = os.path.join(runs_folder, last_train, "weights", "best.pt")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Modelul {model_path} nu există!")

print(f"Folosește modelul: {model_path}")

model = YOLO(model_path)
print("Clase disponibile în model:")
print(model.names)

def simplify_class_name(name: str) -> str:
    name = name.lower()
    if "empty" in name:
        return "empty"
    elif "occupied" in name:
        return "occupied"
    else:
        return name

def send_to_api(free_count, occupied_count):
    payload = {
        "empty_spots": free_count,
        "occupied_spots": occupied_count
    }
    try:
        response = requests.put(f"{API_BASE_URL}/spots/{PARKING_NUMBER}", json=payload, timeout=5)
        if response.status_code == 200:
            print(f"Date trimise: {payload}")
        else:
            print(f"Eroare API ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"Eroare conexiune cu API: {e}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Nu s-a putut deschide videoclipul: {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
update_interval = 15
frame_count = 0
last_update_time = 0
free_count = occupied_count = total_count = 0
detected_frame = None

cv2.namedWindow("YOLO Parking Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Parking Detection", 1280, 720)

print("Pornim analiza în timp real... (apasă 'q' sau 'ESC' pentru oprire)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Sfârșitul videoclipului sau eroare la citire.")
        break

    frame_count += 1
    current_time = time.time()

    if current_time - last_update_time >= update_interval or detected_frame is None:
        last_update_time = current_time

        results = model.predict(source=frame, conf=0.3, verbose=False)
        free_count = occupied_count = total_count = 0
        detected_frame = frame.copy()

        for r in results:
            boxes = r.boxes.xyxy.cpu()
            scores = r.boxes.conf.cpu()
            classes = r.boxes.cls.cpu()

            keep = nms(boxes, scores, iou_threshold=0.4)
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

            for box, score, cls in zip(boxes, scores, classes):
                class_name = model.names[int(cls)]
                simple_name = simplify_class_name(class_name)
                x1, y1, x2, y2 = map(int, box)

                color = (0, 255, 0) if simple_name == "empty" else (0, 0, 255)
                label = f"{simple_name} {score:.2f}"

                cv2.rectangle(detected_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(detected_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if simple_name in ["empty", "occupied"]:
                    total_count += 1
                    if simple_name == "empty":
                        free_count += 1
                    elif simple_name == "occupied":
                        occupied_count += 1

        print(f"Actualizare: Free={free_count}, Occupied={occupied_count}, Total={total_count}")
        send_to_api(free_count, occupied_count)

        del results
        torch.cuda.empty_cache()

    display_frame = detected_frame.copy() if detected_frame is not None else frame.copy()
    cv2.rectangle(display_frame, (10, 10), (350, 80), (255, 255, 255), -1)
    text = f"Free: {free_count} | Occupied: {occupied_count} | Total: {total_count}"
    cv2.putText(display_frame, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("YOLO Parking Detection", display_frame)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF
    if key in [ord('q'), 27]:
        print("Oprire manuală...")
        break

cap.release()
cv2.destroyAllWindows()

print("\nAnaliza videoclipului și trimiterea către API au fost finalizate!")
