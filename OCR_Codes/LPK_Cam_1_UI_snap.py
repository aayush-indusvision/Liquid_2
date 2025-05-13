import os
import cv2
import numpy as np
from utils import mvsdk
import platform
import sys
import snap7
import base64
from cv2 import imencode
import socket
import threading
from ultralytics import YOLO
from PyQt5.QtCore import Qt
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout,QMessageBox,QLabel
from PyQt5.QtCore import QPropertyAnimation, QRect,QEasingCurve
from pycomm3 import LogixDriver
from paddleocr import PaddleOCR
import logging
import time
from PIL import Image
from utils.dashboard import Dashboard
import random
import datetime
from datetime import timedelta
from ping3 import ping, verbose_ping

def send_signal():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(b"ACTIVE", ("127.0.0.1", 5005))  # Send a signal to the monitor

os.system("title LPK_Cam_1_Text")

# Initial setup
hello_count = 0
output_dir = "output_dry_run7"
os.makedirs(output_dir, exist_ok=True)
output_normal_dir = "output_normal_cam_1"
if not os.path.exists(output_normal_dir):
    os.makedirs(output_normal_dir)
os.makedirs(output_dir, exist_ok=True)

def initialize_plc(ip='192.168.21.10', rack=0, slot=1):
    """Initialize PLC connection."""
    try:
        plc = snap7.client.Client()
        
        result = plc.connect(ip, rack, slot)
        if plc.get_connected():
            logging.info(f"Connected to PLC at {ip}")
            return plc
        else:
            logging.error("Failed to connect to PLC")
        return None
    except Exception as e:
        logging.error(f"Error connecting to PLC: {e}")
    return None 

def trigger_plc(db_number, start_offset, bit_offset, value,plc):
    """ writing True to %M2.5. :) snaping one two where are you"""

    try:
        # Prepare data to write (1 byte) with value True
        start_address = 100 # starting address
        length = 4
    
        reading = plc.db_read(db_number, start_offset, 1) # (db number, start offset, read 1 byte)
        snap7.util.set_bool(reading, 0, bit_offset, value) # (value 1= true;0=false) (bytearray_: bytearray, byte_index: int, bool_index: int, value: bool)
        plc.db_write(db_number, start_offset, reading)
        verify_reading = plc.db_read(db_number, start_offset, 1)
        written_value = snap7.util.get_bool(verify_reading, 0, bit_offset)
        # if written_value==verify_reading:
        #     print("TURE")
        # else:
        #     print("FALSE")
        logging.info("Triggered camera: %M2.5 set to True")
    except Exception as e:
        logging.error(f"Failed to trigger camera: {e}") # write back the bytearray and now the boolean value is changed in the PLC.
    return written_value==value

def rejected_and_sent_to_dashboard(frame, plc_reject, defect_id,rca_id):

    start=time.time()
    state = True
    db_number = 1 
    start_offset = 7
    bit_offset = 0 
    value = 1
    trigger_plc(db_number, start_offset, bit_offset, value,plc_reject)
    print(time.time()-start)
    print("REJECTED TEXT CAM 1")

    dashboard = Dashboard(url="http://localhost:8000/api/dashboard/", machine=24, department=11)
            
    # dashboard.send_image(save_path, defect=defect_id)
    _, buffer = imencode('.png', frame)
    
    # Convert the buffer to a byte array
    byte_data = buffer.tobytes()

    # Encode the byte data to Base64
    encoded_image = base64.b64encode(byte_data).decode('utf-8')
    dashboard.send_image(encoded_image,sel, rca_id,defect=defect_id,original_image=frame)
    return

def reset_plc(plc, plc_ip):
    try:
        plc = snap7.client.Client()
        
        result = plc.connect(plc_ip, 0, 1)
        if plc.get_connected():
            logging.info(f"Reconnected to PLC at {plc_ip}")
            return plc
        else:
            logging.error("Failed to reconnect to PLC")
        return None
    except Exception as e:
        logging.error(f"Error reconnecting to PLC: {e}")
    return None

def toggle_plc_trigger(plc, plc_ip, trigger_tag):
    max_retries = 3
    delay_between_attempts = timedelta(milliseconds=10)

    for retry in range(max_retries):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            result = plc.write(trigger_tag, True)
            print("Defect came", result)
            if result.value == True:
                logging.debug(f"Wrote 1 to {trigger_tag} - {timestamp}")
                return plc
            else:
                logging.warning(f"Failed to write to {trigger_tag}")
        except Exception as e:
            logging.error(f"Error writing to PLC at {timestamp}: {e}")

        logging.info(f"Retry {retry + 1}/{max_retries}")
        wait_until = datetime.datetime.now() + delay_between_attempts
        while datetime.datetime.now() < wait_until:
            pass
        plc = reset_plc(plc, plc_ip)
    return plc

dashboard = Dashboard(url="http://localhost:8000/api/dashboard/", machine=24, department=11)

def save_defect_frame(frame, output_dir, defect_number=54):
    dashboard = Dashboard(url="http://localhost:8000/api/dashboard/", machine=24, department=11)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    filename = os.path.join(output_dir, f"cl_{timestamp}.jpg")
    # cv2.imwrite(filename, frame)
    dashboard.send_image(filename, sel,defect=defect_number)
    logging.info(f"Defect image saved: {filename}")

def rotate_img(image, angle=6, scale=0.5):  # Adjust scale to 1.0
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def save_normal_frame(frame, output_normal_dir):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    filename = os.path.join(output_normal_dir, f"cl_{timestamp}.jpg")
    # logging.info(f"Normal image saved: {filename}")

def initialize_camera():
    cam_flag=0
    DevList = mvsdk.CameraEnumerateDevice()
    if not DevList:
        logging.error("No camera found!")
        return None, None
    
    machine_ip_list="11"
    camera_number=100
    # DevInfo = DevList[2]  # Auto-select first camera for simplicity
    for i, DevInfo in enumerate(DevList):
        # logging.info(f"{i}: {DevInfo.GetFriendlyName()} {DevInfo.GetPortType()}")
        friendly_name = DevInfo.GetPortType()

        # print("-----", friendly_name[-1:-4])

        # print(machine_ip_list == friendly_name.split('.')[-1] )
        if machine_ip_list == friendly_name.split('.')[-1]:
            camera_number = i
            cam_flag=1
            print(camera_number)
            break
    if cam_flag==0:
        print("Camera Not Found")
        # sys.exit()

    # selected_cam = 0 if len(DevList) == 1 else int(input("Select camera: "))
    DevInfo = DevList[camera_number]
    logging.info(DevInfo)
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        logging.error(f"CameraInit Failed({e.error_code}): {e.message}")
        return None, None

    cap = mvsdk.CameraGetCapability(hCamera)
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
    out_format = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if monoCamera else mvsdk.CAMERA_MEDIA_TYPE_BGR8
    mvsdk.CameraSetIspOutFormat(hCamera, out_format)
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraPlay(hCamera)

    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
    
    return hCamera, pFrameBuffer

def main_loop(stop_event):  # Pass the stop_event
    no_frame_count=0
    status_flag=0
    new_flag=0
    status_flag_2=0
    plc=None
    plc_ip = '192.168.21.10'
    if ((isinstance(ping('192.168.21.11'), (int, float))) and (isinstance(ping('192.168.21.10'), (int, float)))):
    # if ((isinstance(ping('192.168.21.11'), (int, float)))): # USE IN-CASE PLC NOT TO USED
        plc = initialize_plc(plc_ip)
        hCamera, pFrameBuffer = initialize_camera()
        status_flag_2=1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(r"models\07_02_25_LPK_Cam_1_Text.pt")
    model.to(device)
    rca=0
    defect_count=0
    ocr = PaddleOCR(rec_model_dir="en_PP-OCRv3_rec_infer", precision='fp16', show_log=False, use_fast=True,savefile=False, lang='en', output="out", use_space_char=True, use_angle_cls=False, ocr_order_method='horizontal', gpu_mem=1000)

    try:
        while not stop_event.is_set():  # Check for stop_event to break the loop
            try:
                if ((not isinstance(ping('192.168.21.11'), (int, float))) or (not isinstance(ping('192.168.21.10'), (int, float)))):
                # if ((not isinstance(ping('192.168.21.11'), (int, float)))): # USE IN-CASE PLC NOT TO USED
                    new_flag=0
                    no_frame_count+=1
                    height, width = 500, 320  # Dimensions of the image
                    image = np.zeros((height, width, 3), dtype=np.uint8)  # Black image

                    # Define the custom text and its properties
                    text_1 = "Camera Disconnected"
                    org_1 = (25, 100)  # Bottom-left corner of the text in the image
                    text_2="or PLC Disconnected"
                    org_2=(25,150)
                    text_4="or Production Stopped"
                    org_4=(25,200)
                    text_3=f"Reconnecting..."
                    org_3=(25,250)
                    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
                    font_scale = 0.5  # Font scale (size)
                    color = (255, 255, 255)  # White color in BGR
                    thickness = 2  # Thickness of the lines in the text

                    # Put the text on the image
                    cv2.putText(image, text_1, org_1, font, font_scale, color, thickness)
                    cv2.putText(image, text_2, org_2, font, font_scale, color, thickness)
                    cv2.putText(image, text_4, org_4, font, font_scale, color, thickness)
                    cv2.putText(image, text_3, org_3, font, font_scale, color, thickness)

                    # Display the image with the text
                    cv2.imshow('LPK_Cam_1', image)
                    cv2.waitKey(1)
                    if no_frame_count==30:
                        status_flag=1
                        if status_flag_2==1:
                            plc.destroy()
                        # if hCamera:
                            mvsdk.CameraUnInit(hCamera)
                        # if pFrameBuffer:
                            mvsdk.CameraAlignFree(pFrameBuffer)
                            status_flag_2=0
                        
                        no_frame_count=0
                    continue
                if status_flag==1:
                    if ((isinstance(ping('192.168.21.11'), (int, float))) and (isinstance(ping('192.168.21.10'), (int, float)))):
                    # if ((isinstance(ping('192.168.21.11'), (int, float)))): # USE IN-CASE PLC NOT TO USED
                        hCamera, pFrameBuffer = initialize_camera()
                        plc=initialize_plc(plc_ip)
                    else:
                            status_flag=0
                            continue
                    status_flag=0
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
                mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                if frame is None:
                    continue
                db_number = 1 
                start_offset = 7 
                bit_offset = 3 
                value = 1
                trigger_plc(db_number, start_offset, bit_offset, value,plc=plc)

                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
                new_flag=1
                save_normal_frame(frame, output_normal_dir)
                

                results = model.predict(frame, show=False, conf=0.5, device=0,verbose=False)
                no_frame_count=0
                img = frame.copy()
                count_text = 0
                count_other = 0
                global hello_count
                # hello_count += 1
                send_signal()
                for result in results:
                    boxes = result.boxes
                    if boxes:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            clas = box.cls
                            if clas == 1:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (66, 66, 66), 2)
                                count_text += 1
                                cropped_image = img[y1:y2, x1:x2]
                                cropped_image = rotate_img(cropped_image)
                                hello = ocr.ocr(cropped_image, cls=True)

                                if hello[0] is not None:
                                    txts = [line[1][0] for line in hello[0]]
                                    for idx, i in enumerate(txts):
                                        cv2.putText(img, i, (100, ((idx * 100) + 100)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                            elif clas == 0:
                                count_other += 1

                if count_other == 0 and count_text == 0:
                    defect_count+=1
                    if defect_count%3==0:
                        rca=1
                    start=time.time()
                    rejected_and_sent_to_dashboard(frame=img,plc_reject=plc,defect_id=54,rca_id=rca)
                    print(time.time()-start)
                else:
                    defect_count=0
                    rca=0

                r_size = cv2.resize(img, (320, 500))
                cv2.imshow("LPK_Cam_1", r_size)
                cv2.waitKey(1)

            except Exception as e:
                # print(f"Error in main_loop: {e}")
                continue

    finally:
        if os.path.exists(r"C:\Users\pc\Desktop\liquid_2_ppocr\Text_Codes\script_1.lock"):
            os.remove(r"C:\Users\pc\Desktop\liquid_2_ppocr\Text_Codes\script_1.lock")
        if new_flag==1:
            if hCamera:
                mvsdk.CameraUnInit(hCamera)
            if pFrameBuffer:
                mvsdk.CameraAlignFree(pFrameBuffer)
        elif status_flag==0:
            cv2.imshow('LPK_Cam_1', image)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()



# Initial setup

# Add your existing PLC, camera, and processing functions here...
sel=0
prod_select = ['Surf Excel Top Load', 'Surf Excel Front Load', 'Rin Top Load', 'Rin Front Load', 'Vim']
class CameraApp(QWidget):
    
    def __init__(self):
        super().__init__()

        # Set the window title and size
        self.setWindowTitle("LPK_Cam_1")
        self.setGeometry(300, 300, 400, 400)

        # Set a blue background color for the window
        self.setStyleSheet("background-color: blue;")

        # Create option buttons
        self.option_buttons = []
        self.selected_option = None
        prod_select=['Surf Excel Top Load','Surf Excel Front Load','Rin Top Load','Rin Front Load','Vim']
        for i in range(1, 6):
            
            button = QPushButton(f"{prod_select[i-1]}")
            button.setCheckable(True)
            button.setStyleSheet("""
                QPushButton {
                    background-color: gray;
                    color: white;
                    border-radius: 15px;
                    font-size: 16px;
                }
                QPushButton:checked {
                    background-color: lightgreen;
                }
            """)
            button.clicked.connect(lambda checked, option=i: self.select_option(option))
            self.option_buttons.append(button)

        # Create Start and Stop buttons
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)  # Disable stop button initially

        # Style the Start and Stop buttons
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: green;
                color: white;
                border-radius: 10px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: lightgreen;
            }
        """)

        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: red;
                color: white;
                border-radius: 10px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: lightcoral;
            }
        """)

        # Connect button clicks to their respective functions
        self.start_button.clicked.connect(self.start_execution)
        self.stop_button.clicked.connect(self.stop_execution)

        # Set layout
        layout = QVBoxLayout()
        for button in self.option_buttons:
            layout.addWidget(button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.thread = None
        self.stop_event = threading.Event()

    def closeEvent(self, event):
        """Disable the close button by ignoring the close event."""
        event.ignore()

    def select_option(self, option):
        global sel
        
        """Handle option selection."""
        self.selected_option = option
        for idx, button in enumerate(self.option_buttons):
            if idx == option - 1:
                sel = option
                button.setChecked(True)
                button.setStyleSheet("""
                    QPushButton {
                        background-color: lightgreen; /* Light blue for selection */
                        color: #002147; /* Dark blue text */
                        border: 2px solid #4682B4; /* Steel blue border */
                        font-size: 16px;
                        border-radius: 15px; /* Rounded corners */
                    }
                """)
                self.add_button_click_animation(button)  # Add click animation to the selected button
                
            else:
                button.setChecked(False)
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #D3D3D3; /* Light gray for unselected */
                        color: black;
                        border: 1px solid #A9A9A9; /* Dark gray border */
                        border-radius: 15px; /* Rounded corners */
                        font-size: 16px;
                    }
                """)

    def add_button_click_animation(self, button):
        """Add click animation to a button."""
        animation = QPropertyAnimation(button, b"geometry")
        animation.setDuration(150)
        animation.setStartValue(button.geometry())
        animation.setEndValue(QRect(button.x() - 5, button.y() - 5, button.width() + 10, button.height() + 10))
        animation.setEasingCurve(QEasingCurve.OutBounce)
        animation.finished.connect(lambda: self.reset_button_geometry(button))
        animation.start()

    def reset_button_geometry(self, button):
        """Reset the button geometry after animation."""
        button.setGeometry(button.x() + 5, button.y() + 5, button.width() - 10, button.height() - 10)


    def start_execution(self):
        if self.selected_option is None:
            QMessageBox.warning(self, "Warning", "Please select an option before starting!")
            return

        self.stop_event.clear()  # Reset the stop event
        self.start_button.setVisible(False)  # Make the start button disappear
        self.thread = threading.Thread(target=main_loop, args=(self.stop_event,))
        print(f"{prod_select[sel - 1]} selected")
        if not hasattr(self, 'selected_label'):
            self.selected_label = QLabel(self)
            self.layout().addWidget(self.selected_label)

        self.selected_label.setText(f"Selected Option: {prod_select[sel - 1]}")
        self.selected_label.setStyleSheet("font-size: 18px; color: white;")
        self.selected_label.setAlignment(Qt.AlignCenter)
        self.thread.start()
        self.stop_button.setEnabled(True)

        # Add button click animation
        self.add_button_click_animation(self.start_button)
        for i in range(5):
            self.option_buttons[i].setVisible(False)

    def stop_execution(self):
        self.stop_event.set()  # Trigger the stop event
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        logging.info("Execution stopped.")

        # Add button click animation
        self.add_button_click_animation(self.stop_button)

        self.close()  # Close the UI window
        sys.exit()  # Exit the terminal/command line

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())

