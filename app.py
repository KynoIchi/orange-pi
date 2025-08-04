# flask_dummy_sender.py
from flask import Flask
import requests
import threading
import time
import random

app = Flask(__name__)

TARGET_IP = '192.168.12.49'
TARGET_PORT = 5000  # Port target server Flask
SEND_INTERVAL = 5  # Detik antar pengiriman

import numpy as np
import pandas as pd
import random

def generate_minute_voltage_series():
    # Ambil 45 nilai per jam dari grafik
    hourly_pattern = [
        12.72, 12.71, 12.68, 12.66, 12.69, 12.75, 12.91, 13.52,
        14.19, 14.34, 14.45, 14.52, 14.57, 14.48, 13.30, 12.96,
        12.85, 12.80, 12.77, 12.76, 12.75, 12.70, 12.64, 12.61,
        12.63, 12.67, 12.75, 12.91, 13.55, 14.23, 14.36, 14.44,
        14.39, 14.01, 13.14, 12.89, 12.82, 12.74, 12.76, 12.73,
        12.74, 12.74, 12.74, 12.74, 12.73
    ]

    # Upsample ke per menit (dengan noise kecil)
    minute_data = []
    for volt in hourly_pattern:
        # Untuk 60 menit, tambahkan variasi kecil
        noise = np.random.normal(loc=0.0, scale=0.05, size=60)
        minute_data.extend([round(volt + n, 2) for n in noise])

    return minute_data  # panjang 45*60 = 2700 menit

# Contoh generate 2700 menit data
voltages = generate_minute_voltage_series()

# Generate dummy full data per menit
def generate_dummy_data_series():
    volt_series = generate_minute_voltage_series()
    data = []
    for v in volt_series:
        data.append({
            "temperature": round(random.uniform(25.0, 35.0), 2),
            "humidity": round(random.uniform(40.0, 60.0), 2),
            "voltage": v
        })
    return data

def send_data_loop():
    while True:
        data = generate_dummy_data()
        try:
            res = requests.post(f"http://{TARGET_IP}:{TARGET_PORT}/receive", json=data, timeout=3)
            print(f"[✓] Sent: {data} | Status: {res.status_code}")
        except Exception as e:
            print(f"[x] Failed to send data: {e}")
        time.sleep(SEND_INTERVAL)

@app.route('/')
def index():
    return "STM32MP257F Dummy Sender is Running..."

if __name__ == '__main__':
    # Start background thread to send data
    threading.Thread(target=send_data_loop, daemon=True).start()
    # Run local server (not required unless you want to test locally too)
    app.run(host='0.0.0.0', port=8080)
