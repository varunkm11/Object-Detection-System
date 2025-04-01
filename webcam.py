from ultralytics import YOLO
import cv2
import cvzone
import math
cap = cv2.VideoCapture(0) # for Webcam
cap.set(3,1280)
cap.set(4,720)
#cap = cv2.VideoCapture("../videos/cars2.mp4") #for video
model = YOLO("../Yolo-Weights/yolo8n.pt")

classNames = ["person","bicycle","car","motorbike","aeroplane","bus","train","boat","traffic light",
              "file hydrant","parking meter","bench","bird","cat","dog","horse","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
              "sports ball","kite","baseball","baseball glove","skateboard",
              "surfboard","tennis racket","bottle","wine glass","cup","fork","knife",
              "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
              "hot dog","pizza","donut","cake","chair","sofa","pottedplant","flower","bed",
              "dinning table","toilet","Tv Monitor","laptop","mouse","remote","keyboard",
              "cellphone","microwave","oven","toaster","sink","refrigerator","book",
              "clock","vase","scissors","teddy bear","hair drier","Toothbrush"]
while True:
    success, img = cap.read()  # read the frame