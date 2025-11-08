from ultralytics import YOLO
import cv2
import cvzone
import math
import os

# Use a more accurate model for better detection
# Options: yolov8n.pt (fastest), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (extra large)
model_name = "yolov8m.pt"  # Using medium model for better accuracy
weights_path = f"../Yolo-Weights/{model_name}"

# Download model if it doesn't exist
if not os.path.exists(weights_path):
    print(f"Downloading YOLOv8 {model_name.replace('.pt', '')} model for better accuracy...")
    model = YOLO(model_name)  # This will auto-download
    # Save it to the Yolo-Weights directory
    os.makedirs("../Yolo-Weights", exist_ok=True)
    import shutil
    shutil.copy(model_name, weights_path)
else:
    model = YOLO(weights_path)

# Webcam settings - Higher resolution for better detection
cap = cv2.VideoCapture(0) # for Webcam
cap.set(3, 1920)  # Increased width
cap.set(4, 1080)  # Increased height
#cap = cv2.VideoCapture("../videos/cars2.mp4") #for video

# Detection parameters for better precision
CONFIDENCE_THRESHOLD = 0.5  # Only show detections above 50% confidence
IOU_THRESHOLD = 0.4  # Lower IOU for better separation of overlapping objects

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Color palette for different object classes (BGR format)
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
]

print("🚀 Starting Real-Time Object Detection...")
print(f"📊 Model: {model_name}")
print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD * 100}%")
print(f"📹 Resolution: 1920x1080")
print("Press 'q' to quit\n")

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Run detection with improved parameters
    results = model(img, stream=True, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Only display if confidence is above threshold
            if conf >= CONFIDENCE_THRESHOLD:
                # Calculate dimensions
                w, h = x2 - x1, y2 - y1
                
                # Choose color based on class
                color = colors[cls % len(colors)]
                
                # Draw enhanced bounding box with cvzone
                cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=3, colorR=color, colorC=color)
                
                # Create label with class name and confidence
                label = f'{classNames[cls]} {conf}'
                
                # Draw text with background for better visibility
                cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), 
                                 scale=2, thickness=2, colorR=color, 
                                 colorT=(255, 255, 255), offset=10)
    
    # Display the frame
    cv2.imshow("YOLOv8 Real-Time Object Detection", img)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✅ Detection stopped successfully!")
            