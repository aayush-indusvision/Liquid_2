# from pycomm3 import LogixDriver
# import time

# plc_ip = '192.168.20.21'
# reject_counter=0
# with LogixDriver(plc_ip) as plc:
#     while True:
        
#         reject_counter+=1
#         print(reject_counter)
#         plc.write("CAMERA_REJECT_SIGNAL3", True)
#         print("hi turned on rejection for Machine 3")
#         time.sleep(10)

from datetime import datetime, timedelta
import os
from cv2 import imwrite, imencode
import snap7
import logging
import base64
# from siemens_plc_reject import trigger_plc

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
    # return None

def rejected_and_sent_to_dashboard( plc_reject):
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # save_filename = f"defect_detected_{timestamp}.jpg"
    # save_path = os.path.join(output_dir, save_filename)
    # cv2.imwrite(save_path, frame)
    # grayscale = 255 - frame.copy()
    # imwrite(save_path, frame)
    # cv2.imshow("m1_vertical_defect", frame)
    
    #dashboard_glob.send_image_glob(save_path, defect=23)

    # if defect_id != 20:

    state = True
    db_number = 1 
    start_offset = 7 
    bit_offset =  1
    value = 1
    trigger_plc(db_number, start_offset, bit_offset, value,plc_reject)

            
    # dashboard.send_image(save_path, defect=defect_id)
    # _, buffer = imencode('.png', frame)
    
    # Convert the buffer to a byte array
    # byte_data = buffer.tobytes()

    # Encode the byte data to Base64
    # encoded_image = base64.b64encode(byte_data).decode('utf-8')
    # dashboard.send_image(encoded_image, defect=defect_id)
    return

def initialize_plc_connection(ip='192.168.21.10', rack=0, slot=1):
    """Initialize PLC connection."""
    try:
        plc = snap7.client.Client()
        
        result = plc.connect(ip, rack, slot)
        # print(result)
        # print(plc.get_connected())
        if plc.get_connected():
            # print("jhello")
            logging.info(f"Connected to PLC at {ip}")
            return plc
        else:
            logging.error("Failed to connect to PLC")
        return None
    except Exception as e:
        logging.error(f"Error connecting to PLC: {e}")
    return None 

plc_ip = '192.168.21.10'
reject_counter = 0
time_delta = timedelta(milliseconds=50)  
next_trigger_time = datetime.now()

plc=initialize_plc_connection()
# with LogixDriver(plc_ip) as plc:
# while True:
    # current_time = datetime.now()

    # if current_time >= next_trigger_time:
reject_counter += 1
print(f"Reject Counter: {reject_counter}")
# plc.write("DIGITAL_EYE_POUCH_OPEN_1", True)
# plc.write("DIGITAL_EYE_POUCH_OPEN_2", True)
rejected_and_sent_to_dashboard(plc)
print("Hi, turned on rejection for Machine 3")

# next_trigger_time = current_time + time_delta
