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
    results = model(img, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # for opeen cv
            # for bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            #cv.2rectangle(imag, (x1,y1), (x2,y2),(255,0,255), 3)
            #print(x1,y1,x2,y2)
            
            # for cvzone
            w,h = x2 - x1, y2 - y1
            cvzone.cornerRect(img,((x1,y1,w,h)))
            # For Confidence
            conf = math.ceil((box[0]*100))/100
            print(conf)
            #cvzone.putTextrect(img,f'{conf}',(max(0,x1),max(35,y1)))
            #for class(object detection)
            cls = int(box.cls[0])
            print()