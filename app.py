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

def generate_dummy_data():
    return {
        "temperature": round(random.uniform(25.0, 35.0), 2),
        "humidity": round(random.uniform(40.0, 60.0), 2),
        "voltage": round(random.uniform(3.0, 4.2), 2)
    }

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
