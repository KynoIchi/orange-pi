import os
import sys
import time
import platform
import subprocess
import cv2
import numpy as np
from datetime import datetime
import multiprocessing
import requests
import re

def ping_ipcam(ip: str, timeout: int = 1000) -> dict:
    """
    Ping IP dan kembalikan status, time_ms, dan ttl.
    """
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    timeout_param = '-w' if platform.system().lower() == 'windows' else '-W'
    try:
        result = subprocess.run(
            ['ping', param, '1', timeout_param, str(timeout), ip],
            capture_output=True,
            text=True
        )
        output = result.stdout
        status = 1 if result.returncode == 0 else 0

        time_match = re.search(r'time[=<]?([\d\.]+)ms', output)
        ttl_match = re.search(r'TTL[=|:](\d+)', output, re.IGNORECASE)

        time_ms = float(time_match.group(1)) if time_match else None
        ttl = int(ttl_match.group(1)) if ttl_match else None

        return {"status": status, "time_ms": time_ms, "ttl": ttl}
    except Exception as e:
        print(f"❌ Error pinging {ip}: {e}")
        return {"status": 0, "time_ms": None, "ttl": None}

def detect_image_quality_from_frame(image, patch_size=64, fog_threshold=12.0, fog_ratio_threshold=0.3,
                                    blur_threshold=100.0, contrast_threshold=30.0):
    if image is None:
        raise ValueError("❌ Gambar tidak valid.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = laplacian_var < blur_threshold
    global_contrast = np.std(gray)
    is_low_contrast = global_contrast < contrast_threshold
    h, w = gray.shape
    fog_patches = 0
    total_patches = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                continue
            patch_std = np.std(patch)
            total_patches += 1
            if patch_std < fog_threshold:
                fog_patches += 1
    fog_ratio = fog_patches / total_patches
    foggy = fog_ratio > fog_ratio_threshold
    if foggy and is_blurry:
        quality = "Buruk (Kabut & Buram)"
    elif foggy:
        quality = "Kabut"
    elif is_blurry:
        quality = "Buram"
    elif is_low_contrast:
        quality = "Kontras Rendah"
    else:
        quality = "Baik"
    return {
        "blur_score": round(laplacian_var, 2),
        "contrast_score": round(global_contrast, 2),
        "fog_patch_ratio": round(fog_ratio, 2),
        "quality_status": quality
    }

def capture_rtsp_frame(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Tidak bisa membuka stream")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Gagal ambil frame")
        return None
    return frame

def monitor_ipcam(ip: str, rtsp_url: str, nama_keterangan: str, interval_sec: int = 60):
    API_ENDPOINT = "http://localhost:5000/api/ipcam/quality"
    while True:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        ping_result = ping_ipcam(ip)
        status_ping = ping_result["status"]
        print(f"[{timestamp}] [{nama_keterangan}] Ping: {'✅ Online' if status_ping else '❌ Offline'} | Time={ping_result['time_ms']} ms TTL={ping_result['ttl']}")
        
        result = {
            "blur_score": "-",
            "contrast_score": "-",
            "fog_patch_ratio": "-",
            "quality_status": "-"
        }

        if status_ping == 0:
            print(f"[{timestamp}] [{nama_keterangan}] 🔄 IPCam offline! Restarting process...")
            time.sleep(3)
            os.execl(sys.executable, sys.executable, *sys.argv)

        frame = capture_rtsp_frame(rtsp_url)
        if frame is not None:
            result = detect_image_quality_from_frame(frame)
            print(f"[{nama_keterangan}] 📸 Kualitas: {result['quality_status']} | Blur={result['blur_score']}, Kontras={result['contrast_score']}, Fog={result['fog_patch_ratio']}")

        payload = {
            "timestamp": timestamp,
            "ip": ip,
            "nama_keterangan": nama_keterangan,
            "status_ping": "Online" if status_ping else "Offline",
            "ping_time_ms": ping_result["time_ms"],
            "ping_ttl": ping_result["ttl"],  # ✅ TTL ditambahkan
            "quality": result["quality_status"],
            "blur_score": result["blur_score"],
            "contrast_score": result["contrast_score"],
            "fog_patch_ratio": result["fog_patch_ratio"]
        }


        try:
            response = requests.post(API_ENDPOINT, json=payload, timeout=10)
            if response.status_code != 200:
                print(f"⚠️ Gagal kirim ke API [{nama_keterangan}]: {response.status_code}")
        except Exception as e:
            print(f"❌ Error kirim data ke API [{nama_keterangan}]: {e}")

        print("-" * 60)
        time.sleep(interval_sec)

# === List Kamera
camera_list = [
    {"nama_keterangan": "Parkiran Mobil", "ip": "192.168.11.2", "port": "554", "username": "admin", "password": "damin1234"},
    {"nama_keterangan": "Parkiran Motor", "ip": "192.168.11.4", "port": "554", "username": "admin", "password": "damin1234"},
    {"nama_keterangan": "Pintu Depan (gudang)", "ip": "192.168.11.5", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Lorong Tengah", "ip": "192.168.11.8", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Workshop", "ip": "192.168.11.10", "port": "554", "username": "admin", "password": "damin1234"},
    {"nama_keterangan": "Lorong RND", "ip": "192.168.11.11", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Gudang Teknisi", "ip": "192.168.11.12", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Lorong Kamar Mandi", "ip": "192.168.11.13", "port": "554", "username": "admin", "password": "damin1234"},
    {"nama_keterangan": "Gudang", "ip": "192.168.11.14", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Smoking Area", "ip": "192.168.11.15", "port": "554", "username": "admin", "password": "damin1234"},
    {"nama_keterangan": "Timur Pabrik", "ip": "192.168.11.16", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Selatan Pabrik", "ip": "192.168.11.17", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Parkiran Mobil Selatan", "ip": "192.168.11.18", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Lorong Depan K.Direksi", "ip": "192.168.11.19", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Pintu Masuk", "ip": "192.168.11.35", "port": "560", "username": "admin", "password": "damin1234"},
    {"nama_keterangan": "Pintu Masuk Luar", "ip": "192.168.11.20", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Pintu Masuk Timur", "ip": "192.168.11.21", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Pos Satpam", "ip": "192.168.11.22", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Gudang", "ip": "192.168.11.23", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Gudang + K3", "ip": "192.168.11.24", "port": "554", "username": "admin", "password": "Damin3001"},
    {"nama_keterangan": "Kamar Mandi Belakang", "ip": "192.168.11.25", "port": "554", "username": "admin", "password": "Damin3001"},
]

if __name__ == "__main__":
    multiprocessing.freeze_support()
    processes = []
    for cam in camera_list:
        rtsp_url = f"rtsp://{cam['username']}:{cam['password']}@{cam['ip']}:{cam['port']}/"
        p = multiprocessing.Process(
            target=monitor_ipcam,
            args=(cam["ip"], rtsp_url, cam["nama_keterangan"])
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
