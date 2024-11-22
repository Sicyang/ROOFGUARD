from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\models\yolov8n-seg.yaml")
    model = YOLO(r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\models\yolov8n-seg.pt")

    model.train(data=r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\models\coco128-seg.yaml', epochs=500, imgsz=640)
