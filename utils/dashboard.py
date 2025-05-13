import base64
import requests
from datetime import datetime
import cv2
import os
import random

class Dashboard:
    def __init__(self, url, machine, department):
        self.url = url
        self.machine = machine
        self.department = department
        

    def send_image(self, image_path, defect,product,rca,original_frame):
        try:
            # Read image
            # with open(image_path, "rb") as image_file:
            #     image_data = image_file.read()

            # # Encode image to base64
            # image_b64 = base64.b64encode(image_data).decode()
            # print(f"Base64-encoded image: {image_b64[:100]}...")

            # Current system date and time
            recorded_date_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            if product==1:
                product_id=10
            elif product==3:
                product_id=12
            elif product==2:
                product_id=11
            elif product==4:
                product_id=13
            elif product==5:
                product_id=15
            rca_list=[None,"rca2"]
            # JSON payload
            save_file_name=f"9_24_11_{product_id}_{defect}_{recorded_date_time}_{random.randint(000000000,999999999)}.png"
            save_path = os.path.join(r"defect_frames", save_file_name)
            cv2.imwrite(save_path, original_frame)
            payload= {
                "base64_image": image_path,
                # "file_name": None,
                "machines_id": self.machine,
                "department_id": self.department,
                "product_id": product_id,
                "defects_id": defect,
                "plant_id": 9,
                "recorded_date_time": recorded_date_time,
                "rca": rca_list[rca]
            }
 


            # Send POST request
            response = requests.post(self.url, json=payload)

            # Check response status
            if response.status_code == 201:
                print("Image sent successfully.")
            else:
                print("Failed to send image. Status code:", response.status_code)
        except Exception as e:
            print("Error sending image:", str(e))
