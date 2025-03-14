# YOLO Object Detection Web UI

This repository contains a Flask web application that allows users to upload images and perform object detection using a custom-trained YOLO model. The application displays the results with bounding boxes, class labels, and confidence scores.

## Features

- Upload images through a user-friendly web interface
- Real-time image preview before submission
- Object detection using a custom-trained YOLO model
- Visualization of detection results with bounding boxes
- Color-coded classes for better visual distinction
- Display of confidence scores for each detected object

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/yolo-web-detector.git
   cd yolo-web-detector
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the YOLO model weights in the correct location:
   ```
   runs/detect/train4/weights/best.pt
   ```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. Upload an image using the web interface and click "Detect Objects" to see the results.

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates for the web interface
- `uploads/`: Directory to store uploaded images
- `runs/detect/train4/weights/`: Directory containing the YOLO model weights

## Requirements

- Python 3.8+
- Flask
- OpenCV
- NumPy
- Ultralytics YOLO
- Matplotlib
- PyYAML
- Roboflow

## Credits

This project uses the Ultralytics YOLO implementation for object detection.

## License

MIT 