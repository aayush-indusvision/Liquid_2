from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r"C:\Users\pc\Desktop\Open_pouch_model-20240830T104935Z-001\Open_Pouch\Open_pouch_model\07_02_25_LPK_Open_Pouch.pt")
print(model)

# Export the model to TensorRT format
model.export(format="engine", batch=1,half=True)  # creates 'yolov8n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO(r"C:\Users\pc\Desktop\Open_pouch_model-20240830T104935Z-001\Open_Pouch\Open_pouch_model\07_02_25_LPK_Open_Pouch.engine",task='segment')

# Run inference
# results = tensorrt_model(r"C:\Users\pc\Desktop\Open_pouch_model-20240830T104935Z-001\Open_pouch_model\images_0609")