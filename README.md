# Real-Time Object Detection with YOLO

A computer vision project that performs real-time object detection using YOLOv8 and webcam feed. The application can detect and classify 80 different types of objects including people, vehicles, animals, and everyday items.

## Features

- **Real-time Detection**: Live object detection using your webcam
- **80 Object Classes**: Detects a wide variety of objects including:
  - People and animals
  - Vehicles (cars, bicycles, motorbikes, buses, etc.)
  - Household items (chairs, bottles, laptops, etc.)
  - Food items (pizza, banana, sandwich, etc.)
- **Visual Feedback**: Bounding boxes with confidence scores and class labels
- **High Performance**: Optimized with YOLOv8 for fast inference

## Requirements

### Dependencies
```
ultralytics
opencv-python
cvzone
```

### Hardware
- Webcam or USB camera
- Python 3.7+

### Model Files
- YOLOv8 weights file: `yolo8n.pt` (should be placed in `../Yolo-Weights/` directory)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Object-Detection
   ```

2. **Install required packages**:
   ```bash
   pip install ultralytics opencv-python cvzone
   ```

3. **Download YOLO weights**:
   - Download the YOLOv8 nano model (`yolo8n.pt`) from [Ultralytics](https://github.com/ultralytics/ultralytics)
   - Place it in the `../Yolo-Weights/` directory relative to the project folder

## Usage

Run the object detection script:

```bash
python webcam.py
```

### Controls
- The application will open your default camera (camera index 0)
- Press any key to exit the application
- Objects will be automatically detected and highlighted with:
  - Colored bounding boxes
  - Object class names
  - Confidence scores

## Configuration

### Camera Settings
The webcam is configured with:
- Resolution: 1280x720
- Camera index: 0 (default camera)

To use a different camera, modify the camera index:
```python
cap = cv2.VideoCapture(1)  # Use camera index 1
```

### Video File Input
To use a video file instead of webcam, uncomment and modify:
```python
cap = cv2.VideoCapture("path/to/your/video.mp4")
```

## Detected Object Classes

The system can detect 80 different object classes:

**People & Animals**: person, bird, cat, dog, horse, etc.

**Vehicles**: bicycle, car, motorbike, aeroplane, bus, train, boat

**Household Items**: chair, sofa, bed, toilet, TV monitor, laptop, etc.

**Food Items**: banana, apple, sandwich, pizza, donut, cake, etc.

**Sports & Recreation**: frisbee, skis, surfboard, tennis racket, etc.

*And many more...*

## Project Structure

```
Object-Detection/
├── README.md
├── webcam.py          # Main detection script
└── ../Yolo-Weights/
    └── yolo8n.pt      # YOLOv8 model weights
```

## Technical Details

- **Model**: YOLOv8 Nano (yolo8n.pt) - optimized for speed
- **Framework**: Ultralytics YOLO implementation
- **Computer Vision**: OpenCV for camera handling and display
- **UI Elements**: CVZone for enhanced bounding box visualization

## Troubleshooting

### Common Issues

1. **Camera not found**: Check if your camera is connected and not being used by another application
2. **Model file not found**: Ensure `yolo8n.pt` is in the correct `../Yolo-Weights/` directory
3. **Import errors**: Install all required dependencies using pip

### Performance Optimization

- For better performance on slower devices, consider using a smaller input resolution
- For higher accuracy, you can use larger YOLO models (yolo8s.pt, yolo8m.pt, yolo8l.pt, yolo8x.pt)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** This project uses the YOLOv8 model from Ultralytics. Please refer to their [license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for usage terms regarding the YOLO model.

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLOv8 implementation
- [CVZone](https://github.com/cvzone/cvzone) for computer vision utilities
- [OpenCV](https://opencv.org/) for computer vision operations

---

## Author

**Developed by:** Varun Kumar Singh  
**Project:** Real-Time Object Detection System  
**Year:** 2025

