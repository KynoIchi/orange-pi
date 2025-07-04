from ultralytics import YOLO
import cv2
import random

# === Warna class ===
class_colors = {}
def get_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))  # BGR

def draw_boxes(image, results, class_colors):
    class_counts = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            class_id = int(box.cls.item())
            class_name = result.names[class_id]

            # Count
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Color
            if class_name not in class_colors:
                class_colors[class_name] = get_random_color()
            bbox_color = class_colors[class_name]

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)

            # Label
            label = f"{class_name} {conf:.2f}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(y1, label_size[1])
            cv2.rectangle(image, (x1, top - label_size[1]), (x1 + label_size[0], top + base_line), bbox_color, cv2.FILLED)
            cv2.putText(image, label, (x1, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return image, class_counts

# === Load model ===
model_path = "C:/Users/fadel/Downloads/BackUpSyarif/onnx/yolov5xu.pt"
model = YOLO(model_path)

# === Load image ===
img_path = 'orange/Kabinet.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640))

# === Detect ===
results = model(img)

# === Draw boxes ===
img_out, class_counts = draw_boxes(img.copy(), results, class_colors)

# === Show + save ===
cv2.imshow('YOLO Deteksi', img_out)
# cv2.imwrite('hasil_ultralytics_custom.jpg', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print summary
print("Class counts:", class_counts)
