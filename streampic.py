import cv2
import os

# Path folder untuk menyimpan frame
save_dir = "/home/beacon/orange-pi/image"
os.makedirs(save_dir, exist_ok=True)  # Buat folder jika belum ada

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame")
        break

    # Simpan ke folder 'image'
    save_path = os.path.join(save_dir, "frame.jpg")
    cv2.imwrite(save_path, frame)
    print(f"✅ Frame disimpan di {save_path}")
    break

cap.release()
