import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import yaml  # นำเข้าไลบรารี yaml
from roboflow import Roboflow  # นำเข้าไลบรารี Roboflow
import random  # นำเข้าไลบรารี random

# Load YOLO model
model = YOLO('runs/detect/train4/weights/best.pt')  # เปลี่ยนเส้นทางไปยังโมเดลที่ฝึก

# Download dataset from Roboflow
rf = Roboflow(api_key="cVwQuhMy2QpJ93F1yGEx")  # ใช้ API key ของคุณ
project = rf.workspace("captan").project("aie223_project")
version = project.version(3)
dataset = version.download("yolov11")

# กำหนดเส้นทางไปยังไฟล์ data.yaml
data_file = dataset.location + '/data.yaml'  # อัปเดตเส้นทางไปยัง data.yaml ที่ดาวน์โหลด

# อ่านชื่อคลาสจากไฟล์ data.yaml
with open(data_file, 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['names']  # ดึงชื่อคลาส

# Open webcam
cap = cv2.VideoCapture(0)  # เปิดกล้องเว็บแคม

# Define a color mapping for each class
def get_color(class_id):
    random.seed(class_id)  # Ensure consistent color for each class
    return [random.randint(0, 255) for _ in range(3)]  # Random color

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # ออกจากลูปหากไม่สามารถอ่านเฟรมได้

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    results = model.predict(frame, conf=0.7, show=False)  # ปรับค่า confidence ตามที่ต้องการ

    # วาด bounding box บนเฟรม
    for result in results:
        boxes = result.boxes  # ดึงข้อมูล bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # รับพิกัดของ bounding box
            class_id = int(box.cls[0])  # รับ label ของวัตถุ (แปลงเป็น int)
            confidence = box.conf[0]  # รับค่า confidence

            # ตรวจสอบว่าหมายเลขคลาสอยู่ในช่วงของชื่อคลาส
            if class_id < len(class_names):
                label = class_names[class_id]  # รับชื่อคลาสจากลิสต์
                color = get_color(class_id)  # Get color for the class
            else:
                label = "Unknown"  # ถ้าไม่พบชื่อคลาส
                color = [255, 0, 0]  # Default color for unknown

            # สร้างข้อความสำหรับ label และ confidence
            text = f"{label}: {confidence:.2%}"  # รูปแบบข้อความ

            # วาด bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # วาด bounding box
            cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # วาดข้อความ

    # Convert image to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display image using Matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()

cap.release()  # ปล่อยกล้องเว็บแคมเมื่อเสร็จสิ้น
