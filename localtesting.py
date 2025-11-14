import os
import torch
from ultralytics import YOLO
from torchvision.ops import nms

# 🔧 Căi principale
runs_folder = "C:/Unihack2025/ObDetector/venv/runs/detect"
test_images_folder = "C:/Unihack2025/ObDetector/venv/test_images"
output_folder = "C:/Unihack2025/ObDetector/venv/test_results"
os.makedirs(output_folder, exist_ok=True)

# 🔍 Găsește ultimul folder de antrenare
train_folders = [f for f in os.listdir(runs_folder) if f.startswith("train")]
train_folders.sort()
if not train_folders:
    raise ValueError(f"Niciun folder 'train' găsit în {runs_folder}")

last_train = train_folders[-1]
model_path = os.path.join(runs_folder, last_train, "weights", "best.pt")

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Modelul {model_path} nu există!")

print(f"✅ Folosește modelul: {model_path}")

# 🔹 Încarcă modelul YOLO
model = YOLO(model_path)

print("🔎 Clase disponibile în model:")
print(model.names)

# 🔧 Funcție pentru a unifica etichetele
def simplify_class_name(name: str) -> str:
    name = name.lower()
    if "empty" in name:
        return "empty"
    elif "occupied" in name:
        return "occupied"
    else:
        return name

# 📷 Găsește imaginile de test
image_files = [f for f in os.listdir(test_images_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    raise ValueError(f"Nicio imagine găsită în {test_images_folder}")

# 🔁 Parcurge fiecare imagine
for img_file in image_files:
    image_path = os.path.join(test_images_folder, img_file)
    print(f"\n🔍 Analizează imaginea: {image_path}")

    # 🧠 Rulează predicția
    results = model.predict(
        source=image_path,
        conf=0.25,  # prag de încredere mai strict
        save=True,  # salvăm ca înainte
        project=output_folder,
        name=''     # salvează direct în test_results/predict
    )

    for r in results:
        boxes = r.boxes.xyxy.cpu()
        scores = r.boxes.conf.cpu()
        classes = r.boxes.cls.cpu()

        # 🔹 Aplică Non-Maximum Suppression suplimentar (pentru a elimina suprapunerile)
        keep = nms(boxes, scores, iou_threshold=0.4)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        free_count = 0
        occupied_count = 0
        total_count = 0

        for box, score, cls in zip(boxes, scores, classes):
            class_name = model.names[int(cls)]
            simple_name = simplify_class_name(class_name)

            print(f"→ Detectat: {class_name} ({simple_name}) cu scor {score:.2f}")

            if simple_name in ["empty", "occupied"]:
                total_count += 1
                if simple_name == "empty":
                    free_count += 1
                elif simple_name == "occupied":
                    occupied_count += 1

        print(f"🟢 Libere: {free_count} | 🔴 Ocupate: {occupied_count} | 🔢 Total locuri: {total_count}")

print("\n✅ Analiza imaginilor a fost finalizată!")
