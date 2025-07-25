# orangepi_detector_uploader.py
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

# === Fungsi Kirim JSON ke server eksternal ===
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

# === Fungsi Deteksi Person ===
def detect_person(image, session):
    orig_h, orig_w = image.shape[:2]
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float16)

    preds = session.run(None, {'images': img})[0][0]

    count = 0
    for pred in preds:
        if pred[4] < CONF_THRESH:
            continue
        class_conf = pred[5:]
        class_id = np.argmax(class_conf)
        if class_id != 0 or class_conf[class_id] < CONF_THRESH:
            continue
        count += 1

    return count

# === Fungsi utama (loop) ===
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

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        filename_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        image_path = os.path.join(SAVE_DIR, f"frame_{filename_str}.jpg")

        # Simpan gambar
        success = cv2.imwrite(image_path, frame)
        if not success:
            print("❌ Gagal menyimpan gambar")
            continue

        # Deteksi person
        person_count = detect_person(frame, session)

        # Kirim hasil deteksi
        send_json(timestamp_str, person_count)

        last_capture_time = now

if __name__ == "__main__":
    run()
