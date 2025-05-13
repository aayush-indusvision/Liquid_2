import socket
import requests
import time
from datetime import datetime
import os
# Configuration
# API_URL = "http://localhost:8000/api/params_graph/"  # Replace with your actual API endpoint
FRAME_THRESHOLD = 500  # API hit after 500 object detections
TIME_WINDOW = 0.3  # Time window (200ms) to group simultaneous captures

# os.system("title Monitor")

class Dashboard_crqs:
    def __init__(self, url):
        self.url = url

    def send_params(self, params_count):
        try:
            # Current system date and time
            recorded_date_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

            # JSON payload
            # payload = {
            #     "params_count": params_count,
            #     "date_time": recorded_date_time,
            #     "parameter": 1,
            # }

            payload = {
                "params_count": params_count,
                "recorded_date_time": recorded_date_time,
                "plant_id":9,
                "parameter": 1,
                "machine_id":24
            }

            # Send POST request
            response = requests.post(self.url, json=payload)

            # Check response status
            if response.status_code == 200:
                print("Params sent successfully.")
            else:
                print("Failed to send params. Status code:", response.status_code)
        except Exception as e:
            print("Error sending params:", str(e))

# UDP Setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5005))  # Listen on port 5005

frame_count = 0  # Object detection counter
last_detection_time = 0  # Track last object detection time

print("Monitoring for camera activity...")

while True:
    data, addr = sock.recvfrom(1024)  # Wait for a signal
    current_time = time.time()

    # If this frame is received within TIME_WINDOW of the last detection, consider it the same object
    if current_time - last_detection_time > TIME_WINDOW:
        frame_count += 1  # Count a new object detection
        last_detection_time = current_time  # Update timestamp
        # print(f"Object captured. Total count: {frame_count}")

    if frame_count >= FRAME_THRESHOLD:
        try:
            params_count = Dashboard_crqs(url="http://localhost:8000/api/params_graph/")
            params_count.send_params(str(frame_count))
        except requests.exceptions.RequestException as e:
            print(f"Error hitting API: {e}")

        frame_count = 0  # Reset counter after API hit
