from ultralytics import YOLO

model = YOLO("weights/detection/model_017.pt")

print(model.names)