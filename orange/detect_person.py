import cv2
import os
import time
import onnxruntime as ort
import numpy as np
from datetime import datetime
import requests

# === KONFIGURASI ===
RTSP_URL = "rtsp://admin:Damin1234@192.168.12.16:554/"
MODEL_PATH = "yolov5n.onnx"
POST_URL = "http://192.168.12.217:5000/datamasuk/add_deteksi"
SAVE_DIR = "captures"
INPUT_SIZE = 640
CONF_THRESH = 0.4
NMS_THRESH = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)
last_capture_time = 0

# === Class COCO yang kita pakai (hanya person) ===
class_names = ["person"]

# === Kirim JSON ke server Beacon ===
def send_json(timestamp_str, person_count):
    payload = {
        "waktu": timestamp_str,
        "registered_user": person_count
    }
    try:
        res = requests.post(POST_URL, json=payload, timeout=5)
        res.raise_for_status()
        print(f"📡 JSON terkirim ke Beacon: {res.status_code} | {payload}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Gagal kirim JSON ke Beacon: {e}")

# === Gambar bounding box & hitung person ===
def draw_yolo_detections(frame, preds, input_size=640, conf_thresh=0.4, nms_thresh=0.5):
    h, w, _ = frame.shape
    boxes, scores, class_ids = [], [], []

    for pred in preds:
        conf = pred[4]
        if conf < conf_thresh:
            continue
        class_conf = pred[5:]
        class_id = np.argmax(class_conf)

        # Deteksi hanya untuk "person"
        if class_id != 0 or class_conf[class_id] < conf_thresh:
            continue

        cx, cy, bw, bh = pred[:4]
        cx *= w / input_size
        cy *= h / input_size
        bw *= w / input_size
        bh *= h / input_size

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        boxes.append([x1, y1, int(bw), int(bh)])
        scores.append(float(class_conf[class_id]))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)
    person_count = len(indices)

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, bw, bh = boxes[i]
        label = f"{class_names[class_ids[i]]}: {scores[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan jumlah person
    cv2.putText(frame, f"Person count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, person_count

# === Fungsi utama ===
def run():
    global last_capture_time
    print("🚀 Memulai deteksi dari RTSP...")
    session = ort.InferenceSession(MODEL_PATH)
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Tidak bisa membuka RTSP stream")
        return

    while True:
        now = time.time()
        if now - last_capture_time < 5:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Gagal membaca frame")
            time.sleep(1)
            continue

        # Preprocess input
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float16)

        preds = session.run(None, {'images': img})[0][0]
        frame_with_boxes, person_count = draw_yolo_detections(frame, preds)

        # Simpan frame hasil deteksi ke folder
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        filename_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        image_path = os.path.join(SAVE_DIR, f"frame_{filename_str}_count{person_count}.jpg")

        success = cv2.imwrite(image_path, frame_with_boxes)
        if not success:
            print("❌ Gagal menyimpan gambar")
            continue

        # Kirim hasil deteksi ke server eksternal
        send_json(timestamp_str, person_count)
        last_capture_time = now

if __name__ == "__main__":
    run()
