from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import random
from roboflow import Roboflow

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO('runs/detect/train4/weights/best.pt')

# Define a color mapping for each class
def get_color(class_id):
    random.seed(class_id)  # Ensure consistent color for each class
    return [random.randint(0, 255) for _ in range(3)]  # Random color

# Initialize Roboflow and get dataset
def initialize_roboflow():
    rf = Roboflow(api_key="cVwQuhMy2QpJ93F1yGEx")
    project = rf.workspace("captan").project("aie223_project")
    version = project.version(3)
    dataset = version.download("yolov11")
    return dataset

# Get dataset and class names
dataset = initialize_roboflow()
data_file = 'datasets/AIE223_Project-3/data.yaml'

# Load class names
with open(data_file, 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['names']

def process_image(image_path):
    # Read image from path
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    
    # Resize image for processing
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    
    # Make predictions
    results = model.predict(frame, conf=0.7, show=False)
    
    # Process results and draw bounding boxes
    for result in results:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)  # Get coordinates
            class_id = int(box.cls[0])  # Get class ID
            confidence = box.conf[0]  # Get confidence score
            
            # Get class name and color
            if class_id < len(class_names):
                label = class_names[class_id]
                color = get_color(class_id)
            else:
                label = "Unknown"
                color = [255, 0, 0]  # Default color
            
            # Create label text with confidence score
            text = f"{label}: {confidence:.2%}"
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert image to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create a BytesIO object to save the plot
    buf = BytesIO()
    plt.figure(figsize=(10, 6))
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    
    # Encode the image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Process the image
        img_str = process_image(file_path)
        
        if img_str is None:
            flash('Error processing image')
            return redirect(request.url)
        
        # Pass the result to the template
        return render_template('result.html', img_data=img_str, filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True) 