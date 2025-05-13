import os
import cv2
import torch
import numpy as np
import platform
import multiprocessing
import time
import sys
from cv2 import imencode,imwrite
import base64
import signal
# import schedule
import socket
from utils import mvsdk
from datetime import datetime, timedelta
from ultralytics import YOLO
from pycomm3 import LogixDriver
import logging
import snap7
from utils.dashboard import Dashboard
from crqs import Dashboard_crqs
import random
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO
from datetime import datetime
from functools import partial
from ping3 import ping, verbose_ping

def send_signal():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(b"ACTIVE", ("127.0.0.1", 5005))  # Send a signal to the monitor

os.system("title LPK_Cam_3_Open_Pouch_1")

# Product and Size Button Definitions
product_buttons = [("Surf Excel Top Load", 1), ("Surf Excel Front Load", 2), ("Rin Front Load", 3), ("Rin Top Load", 4), ("Vim",5)]
size_buttons = [("1L", 6), ("2L", 7), ("3.2L", 8)]

# Initialize logging
# logging.basicConfig(level=logging.DEBUG)
frame_count = 0

def initialize_camera():
    cam_flag=0
    DevList = mvsdk.CameraEnumerateDevice()
    if not DevList:
        logging.error("No camera found!")
        return None, None
    
    machine_ip_list="13"
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

def initialize_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(r"models\07_02_25_LPK_Open_Pouch.engine",task='segment')
    logging.info(f"Using device: {device}")
    return model

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
        logging.info("Triggered camera: %M2.5 set to True")
    except Exception as e:
        logging.error(f"Failed to trigger camera: {e}") # write back the bytearray and now the boolean value is changed in the PLC.
    return None

def rejected_and_sent_to_dashboard(frame, plc_reject, defect_id,product,rca_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    recorded_date_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    # save_filename = f"defect_detected_{timestamp}.jpg"
    dashboard = Dashboard(url="http://localhost:8000/api/dashboard/", machine=24, department=11)
    save_file_name=f"9_24_11_{product}_{defect_id}_{recorded_date_time}_{random.randint(000000000,999999999)}.png"
    save_path = os.path.join(r"defect_frames", save_file_name)
    cv2.imwrite(save_path, frame)

    state = True
    db_number = 1 
    start_offset = 6 
    bit_offset = 4 
    value = 1
    # trigger_plc(db_number, start_offset, bit_offset, value,plc_reject) #TEMPORARILY DISABLED

    _, buffer = imencode('.png', frame)
    
    # Convert the buffer to a byte array
    byte_data = buffer.tobytes()

    # Encode the byte data to Base64
    encoded_image = base64.b64encode(byte_data).decode('utf-8')
    dashboard.send_image(encoded_image, defect=defect_id,product=product,rca=rca_id)
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

def create_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def capture_frame(hCamera, pFrameBuffer):
    try:
        pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        if platform.system() == "Windows":
            mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
            (FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
        
        return frame
    except mvsdk.CameraException as e:
        if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
            logging.error(f"CameraGetImageBuffer failed({e.error_code}): {e.message}")
        return None
    
def shoelace_area(pts):
        n = len(pts)
        area = 0
        for i in range(n):
            j = (i + 1) % n  # Next point (wrap around)
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2

def create_binary_mask(image_shape, mask_coords):
    """Creates a binary mask from a set of coordinates (polygon vertices)."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask = cv2.fillPoly(mask, [np.array(mask_coords, dtype=np.int32)], 1)
    return mask

def divide_bbox_and_calculate_mask_percentage(bbox, mask_coords, image_shape):
    # Extract the bounding box coordinates
    x_min, y_min, x_max, y_max = [int(val) for val in bbox]
    # Create the binary mask
    mask = create_binary_mask(image_shape, mask_coords)
    
    # Divide the bounding box into 2x2 grid (exactly 2 rows, 2 columns)
    grid_cells = []
    points = []
    points_2 = []
    
    grid_width = (x_max - x_min) // 2  # width of each grid cell
    grid_height = (y_max - y_min) // 2  # height of each grid cell
    
    # Calculate grid cell coordinates (ensuring 2 rows and 2 columns)
    for i in range(2):
        for j in range(2):
            cell_x_min = x_min + (i * grid_width)
            cell_y_min = y_min + (j * grid_height)
            cell_x_max = cell_x_min + grid_width
            cell_y_max = cell_y_min + grid_height
            
            # Ensure last cell reaches the bbox boundary (avoid off-by-one error)
            if i == 1:
                cell_x_max = x_max
            if j == 1:
                cell_y_max = y_max
            
            grid_cells.append((cell_x_min, cell_y_min, cell_x_max, cell_y_max))
    
    # List to store the percentage of mask in each grid cell
    mask_percentages = []
    
    # For each grid cell, calculate the percentage of mask area
    for cell in grid_cells:
        cell_x_min, cell_y_min, cell_x_max, cell_y_max = cell
        
        # Mask the region of interest (ROI) inside the bounding box cell
        cell_mask = mask[cell_y_min:cell_y_max, cell_x_min:cell_x_max]
        
        # Calculate the area of the cell
        cell_area = (cell_x_max - cell_x_min) * (cell_y_max - cell_y_min)
        
        # Calculate the number of pixels in the mask inside the grid cell
        mask_area_in_cell = np.sum(cell_mask > 0)  # Mask is binary, so pixels > 0 are part of the mask
        
        # Calculate the percentage of the cell covered by the mask
        mask_percentage = (mask_area_in_cell / cell_area) * 100
        mask_percentages.append(mask_percentage)
        
        # Store the top-left corner points of each cell
        points.append((cell_x_min, cell_y_min))
        
        # Store the full bounding box of each grid cell
        points_2.append((cell_x_min, cell_y_min, cell_x_max, cell_y_max))
    
    return mask_percentages, points, points_2

csv_file = 'data_2L_analyse.csv'
image_folder = 'images_2L_analyse'

os.makedirs(image_folder, exist_ok=True)

def calculate_extremes(points):
        
    # Find left-most and right-most points based on x-coordinate
    
    area = shoelace_area(points)
    
    # Calculate the corner centroid

    # Display results
    return area



def detect_defects(model, frame,size):

    
    global area_1
    global area_2
    global area_3
    global area_4
    global Total_area
    global frame_count
    frame_count += 1

    if frame_count >= 500:
        params_count = Dashboard_crqs(url="http://localhost:8000/api/params_graph/")
        params_count.send_params(str(frame_count))
        frame_count = 0

    results = model.predict(frame, conf=0.55, verbose=False,classes=[0])
    org_frame = frame.copy()

    # Get pouch coordinates (class '1' pouches)
    # pouch_coords = [box.xyxy[0].cpu().tolist() for box in results[0].boxes if int(box.cls.cpu().item()) == 1]
    
    # If no pouch is detected, do nothing and return False 

    # flag for class 0 detection
    flag_1 = 0  

    # Check for class 0 (open pouch) inside detected pouches
    if 0 not in results[0].boxes.cls:
        print("-------------CLOSED POUCH------------")
        return True,frame, org_frame
    for detection in results[0].boxes:
        cls = int(detection.cls.cpu().item())
        x1, y1, x2, y2 = map(int, detection.xyxy[0].cpu().tolist())

        # Check if the detection is class 0
        if cls == 0:
            masks = results[0].masks
            op_x1=x1
            op_x2=x2
            op_y1=y1
            op_y2=y2
            per,points,points_2= divide_bbox_and_calculate_mask_percentage(results[0].boxes.xyxy[0], results[0].masks.xy[0], frame.shape)
            for i in points_2:
                frame=cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), color=(0,255,255), thickness=2)
            area=shoelace_area(results[0].masks.xy[0])
            if((per[0]<area_1) or (per[1]<area_2) or (per[2]<area_3) or (per[3]<area_4) or (area<Total_area)):
                
                flag_1 = 1  
            
            color =  (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Only send to dashboard if class 0 (open pouch) is NOT detected inside any pouches
    if flag_1 == 1:  # No open pouch detected inside pouches
        # save_defect_frame(frame, output_dir)
        print(f"Area_1: {per[0]}\n Area_2: {per[1]}\n Area_3: {per[2]}\n Area_4: {per[3]}\n Total_Area: {area}")
        cv2.rectangle(frame, (op_x1, op_y1), (op_x2, op_y2), (0,0,255), 2)
        return True,frame, org_frame  # Send images to dashboard
 # Class 0 is detected inside pouch, do not send images
    else:
        return False,frame,org_frame
    


dashboard = Dashboard(url="http://localhost:8000/api/dashboard/", machine=24, department=11)
def save_defect_frame(frame, output_dir,product):
    dashboard = Dashboard(url="http://localhost:8000/api/dashboard/", machine=24, department=11)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    filename = os.path.join(output_dir, f"cl_{timestamp}.jpg")
    # cv2.imwrite(filename, frame)
    dashboard.send_image(filename,product=product, defect=56)
    logging.info(f"Defect image saved: {filename}")

def toggle_plc_trigger(plc, plc_ip, trigger_tag):
    max_retries = 3
    delay_between_attempts = timedelta(milliseconds=10)

    for retry in range(max_retries):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            result = plc.write(trigger_tag, True)
            if result:
                logging.debug(f"Wrote 1 to {trigger_tag} - {timestamp}")
                return plc
            else:
                logging.warning(f"Failed to write to {trigger_tag}")
        except Exception as e:
            logging.error(f"Error writing to PLC at {timestamp}: {e}")

        # logging.info(f"Retry {retry + 1}/{max_retries}")
        wait_until = datetime.now() + delay_between_attempts
        while datetime.now() < wait_until:
            pass
        plc = reset_plc(plc, plc_ip)
    return plc


def set_parameters(size):
    # Size parameter logic from pouch_new_15_11_24.py
    global area_1
    global area_2
    global area_3
    global area_4
    global Total_area
    if size==6:
        area_1=35
        area_2=30
        area_3=31
        area_4=13
        Total_area=30000
    elif size==7:
        area_1=33
        area_2=24
        area_3=15   
        area_4=18
        Total_area=56000
    elif size==8:
        area_1=32
        area_2=21
        area_3=33
        area_4=26
        Total_area=50000



def main_loop(size, product, stop_event, frame_queue):
    # Main defect detection logic from pouch_new_15_11_24.py
    """
    Main loop for defect detection processing in parallel.

    Arguments:
    - size: The size parameter, used for detection settings.
    - product: The product type, used for processing logic.
    - stop_event: A multiprocessing Event used to signal stopping the loop.
    - frame_queue: A multiprocessing Queue used to send frames back to the main process.
    """
    set_parameters(size)
    no_frame_count=0
    status_flag=0
    # Capture the parent process ID (i.e., the terminal process that launched this script)
    parent_pid = os.getppid()
    no_frame_count=0
    status_flag_2=0
    plc_ip = '192.168.21.10'
    plc = None
    if ((isinstance(ping('192.168.21.13'), (int, float))) and (isinstance(ping('192.168.21.10'), (int, float)))):
    # if ((isinstance(ping('192.168.21.13'), (int, float)))):
        hCamera, pFrameBuffer = initialize_camera()
        
        status_flag_2=1
    
        plc = initialize_plc()
    model = initialize_model()
    # schedule.every(1).hours.do(res_plc)
    output_dir = "5_10_test"
    create_output_dir(output_dir)
    rca=0
    defect_count=0
    output_dir_new = "new_5_10_test"
    create_output_dir(output_dir_new)
    create_output_dir("annotated_frames")
    create_output_dir("all_frames")
    try:
        while not stop_event.is_set():
            # schedule.run_pending()
            try:
                # Check the stop event regularly
                    if ((not isinstance(ping('192.168.21.13'), (int, float))) or (not isinstance(ping('192.168.21.10'), (int, float)))):
                    # if ((not isinstance(ping('192.168.21.13'), (int, float)))): # USE IN-CASE PLC IS NOT TO BE USED
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
                        cv2.imshow('LPK_Open_Pouch_1', image)
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
                        if ((isinstance(ping('192.168.21.13'), (int, float))) and (isinstance(ping('192.168.21.10'), (int, float)))):
                        # if (isinstance(ping('192.168.21.13'), (int, float))): # USE IN-CASE PLC IS NOT TO BE USED
                            hCamera, pFrameBuffer = initialize_camera()
                            plc=initialize_plc()
                        else:
                            status_flag=0
                            continue
                        status_flag=0
                    # Capture a frame from the camera
                    frame = capture_frame(hCamera, pFrameBuffer)
                    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
                    filename = os.path.join(output_dir_new, f"cl_{timestamp}.jpg")

                    if frame is None:
                        continue
                    db_number = 1 
                    start_offset = 7 
                    bit_offset = 5 
                    value = 1
                    send_signal()
                    trigger_plc(db_number, start_offset, bit_offset, value,plc=plc)


                    # Detect defects and get the annotated frame
                    send_trigger, annotated_frame, org_frame = detect_defects(model, frame, size)
                    no_frame_count=0

                    resized_frame = cv2.resize(annotated_frame, (320, 500))

                    cv2.imshow('LPK_Open_Pouch_1', resized_frame)
                    cv2.waitKey(1)  
                    # Send frame to the main process via the queue
                    frame_queue.put(resized_frame)

                    if send_trigger:
                        defect_count+=1
                        if defect_count%3==0:
                            rca=1
                        rejected_and_sent_to_dashboard(frame=annotated_frame,plc_reject=plc,defect_id=56,product=product,rca_id=rca)
                        print("----------------Defect-----------------------")
                    else:
                        defect_count=0
                        rca=0

                    # Sleep for a short period to manage CPU load
                    
                    time.sleep(0.1)

            except Exception as e:
                continue

    finally:
        if os.path.exists(r"C:\Users\pc\Desktop\Open_pouch_model-20240830T104935Z-001\Open_Pouch\Open_pouch_model\script_1.lock"):
            os.remove(r"C:\Users\pc\Desktop\Open_pouch_model-20240830T104935Z-001\Open_Pouch\Open_pouch_model\script_1.lock")

        if status_flag==0:
            cv2.imshow('LPK_Open_Pouch_1', image)
            cv2.waitKey(0)
        else:
            
        # Cleanup resources even if the loop is interrupted
            print("Cleaning up resources...")

            # Ensure camera and PLC resources are released
            if hCamera:
                mvsdk.CameraUnInit(hCamera)
            if pFrameBuffer:
                mvsdk.CameraAlignFree(pFrameBuffer)

            os.kill(parent_pid, signal.SIGTERM)
            print("Resources cleaned up and terminal closed successfully.")


class Worker(QThread):
    """Worker thread for running defect detection."""
    frame_signal = pyqtSignal(object)

    def __init__(self, size, product, stop_event, frame_queue):
        super().__init__()
        self.size = size
        self.product = product
        self.stop_event = stop_event
        self.frame_queue = frame_queue

    def run(self):
        main_loop(self.size, self.product, self.stop_event, self.frame_queue)


class FunctionSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LPK Open Pouch 1")
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(30, 58, 138, 255), stop:1 rgba(49, 130, 206, 255));
            }
        """)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.init_ui()
        self.worker_thread = None
        self.stop_event = None
        self.frame_queue = multiprocessing.Queue()

        self.selected_product_button = None
        self.selected_size_button = None

    def closeEvent(self, event):
        """Disable the close button by ignoring the close event."""
        event.ignore()

    def init_ui(self):
        grid_layout = QGridLayout()

        heading_font = QFont("Arial", 12, QFont.Bold)
        heading_product = QLabel("Product")
        heading_product.setFont(heading_font)
        heading_product.setStyleSheet("color: white;")
        heading_size = QLabel("Size")
        heading_size.setFont(heading_font)
        heading_size.setStyleSheet("color: white;")

        grid_layout.addWidget(heading_product, 0, 0, alignment=Qt.AlignCenter)
        grid_layout.addWidget(heading_size, 0, 1, alignment=Qt.AlignCenter)

        button_font = QFont("Arial", 10)
        self.selected_product = None
        self.selected_size = None

        for i, (text, value) in enumerate(product_buttons):
            button = self.create_button(text, button_font)
            button.clicked.connect(partial(self.select_button, "product", value, button))
            grid_layout.addWidget(button, i + 1, 0)

        for i, (text, value) in enumerate(size_buttons):
            button = self.create_button(text, button_font)
            button.clicked.connect(partial(self.select_button, "size", value, button))
            grid_layout.addWidget(button, i + 1, 1)

        self.main_layout.addLayout(grid_layout)

        self.start_button = self.create_button("Start", QFont("Arial", 14, QFont.Bold), "#4CAF50", "#66BB6A")
        self.start_button.clicked.connect(self.start_processing)

        self.stop_button = self.create_button("Stop", QFont("Arial", 14, QFont.Bold), "#F44336", "#EF5350")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_processing)

        self.main_layout.addWidget(self.start_button)
        self.main_layout.addWidget(self.stop_button)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.video_label)

    def create_button(self, text, font, color="#3B82F6", hover_color="#63A3F7"):
        button = QPushButton(text)
        button.setFont(font)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """)
        return button

    def select_button(self, button_type, value, button):
        if button_type == "product":
            self.selected_product = value
            if self.selected_product_button:
                self.reset_button_style(self.selected_product_button)
            button.setStyleSheet("background-color: #4CAF50; color: white;")
            self.selected_product_button = button
        elif button_type == "size":
            self.selected_size = value
            if self.selected_size_button:
                self.reset_button_style(self.selected_size_button)
            button.setStyleSheet("background-color: #4CAF50; color: white;")
            self.selected_size_button = button

    def reset_button_style(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #63A3F7;
            }
        """)

    def start_processing(self):
        if self.selected_product is None or self.selected_size is None:
            self.show_popup("Error", "Please select one option from both Product and Size columns before starting.")
        else:
            self.stop_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self.stop_event = multiprocessing.Event()
            self.worker_thread = Worker(self.selected_size, self.selected_product, self.stop_event, self.frame_queue)
            self.worker_thread.start()

    def stop_processing(self):
        self.stop_event.set()  # Signal the thread to stop
        if self.thread:
            self.thread.join()  # Wait for the thread to terminate
        logging.info("Execution stopped.")
        QApplication.quit()  # Cleanly exit the QApplication
        sys.exit()
        
        

    def show_popup(self, title, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    selector = FunctionSelector()
    selector.show()
    sys.exit(app.exec_())
