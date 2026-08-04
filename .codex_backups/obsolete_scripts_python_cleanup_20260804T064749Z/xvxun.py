from ultralytics import YOLO

# 加载断点，自动续训
model = YOLO("/home/rom305/zzf/yolov13/runs/train/exp3/weights/last.pt")
model.train(resume=True)