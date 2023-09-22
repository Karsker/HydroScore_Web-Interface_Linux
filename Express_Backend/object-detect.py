import sys
import torch
import pandas
import logging
YOLO_VERBOSE=False
logging.getLogger("utils.general").setLevel(logging.WARNING)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
img = sys.argv[1]  # or file, Path, PIL, OpenCV, numpy, list
results = model(img)
names = model.names

for obj in results.pandas().xyxy[0]['name']:
    print(obj)