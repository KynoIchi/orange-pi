from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime
import onnxruntime as ort

app = Flask(__name__)

# === Konfigurasi Model ===
MODEL_PATH = "orange/yolov5n.onnx"
session = ort.InferenceSession(MODEL_PATH)
CONF_THRESH = 0.4
NMS_THRESH = 0.5
INPUT_SIZE = 640
SAVE_DIR = "orange/statics"

# === COCO class (ambil class_id == 0 untuk "person")
class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
               "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
               "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
               "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
               "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
               "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
               "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
               "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
               "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
               "toothbrush"]

# === Deteksi dan Gambar BBox Orang ===
def draw_yolo_detections(frame, preds):
    h, w, _ = frame.shape
    boxes, scores, class_ids = [], [], []

    for pred in preds:
        conf = pred[4]
        if conf < CONF_THRESH:
            continue
        class_conf = pred[5:]
        class_id = np.argmax(class_conf)
        if class_conf[class_id] < CONF_THRESH or class_id != 0:
            continue  # hanya deteksi 'person'

        cx, cy, bw, bh = pred[:4]
        cx *= w / INPUT_SIZE
        cy *= h / INPUT_SIZE
        bw *= w / INPUT_SIZE
        bh *= h / INPUT_SIZE

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        boxes.append([x1, y1, int(bw), int(bh)])
        scores.append(float(class_conf[class_id]))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH)
    person_count = len(indices)

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, bw, bh = boxes[i]
        label = f"{class_names[class_ids[i]]}: {scores[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, person_count

# === Endpoint Deteksi Orang ===
@app.route("/detect", methods=["POST"])
def detect_people():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        # Baca gambar dan konversi ke RGB
        image = Image.open(file.stream).convert("RGB")
        img_np = np.array(image)  # RGB (H, W, 3)

        # Preprocessing untuk YOLOv5n ONNX
        img_resized = cv2.resize(img_np, (INPUT_SIZE, INPUT_SIZE))
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        img_input = img_input.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))  # HWC → CHW
        img_input = np.expand_dims(img_input, 0).astype(np.float16)

        # Inference
        outputs = session.run(None, {'images': img_input})
        preds = outputs[0][0]

        # Gambar hasil dan hitung jumlah orang
        result_img, person_count = draw_yolo_detections(img_np.copy(), preds)

        # Simpan hasil ke file
        os.makedirs(SAVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detected_{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

        return jsonify({
            "status": "success",
            "timestamp": timestamp,
            "person_count": person_count,
            "image_url": f"/static/{filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Serve Gambar Output ===
@app.route("/static/<filename>")
def serve_detected_image(filename):
    return send_file(os.path.join(SAVE_DIR, filename), mimetype='image/jpeg')

# === Run Flask App ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
