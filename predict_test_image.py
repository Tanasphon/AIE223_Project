import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import yaml
import os
from roboflow import Roboflow
import random  # นำเข้าไลบรารี random

# ติดตั้ง Roboflow
# !pip install roboflow  # Uncomment this line if you haven't installed Roboflow

# ดึงข้อมูลจาก Roboflow
rf = Roboflow(api_key="cVwQuhMy2QpJ93F1yGEx")
project = rf.workspace("captan").project("aie223_project")
version = project.version(3)
dataset = version.download("yolov11")

# Load YOLO model
model = YOLO('runs/detect/train4/weights/best.pt')  # เปลี่ยนเส้นทางไปยังโมเดลที่ฝึก

# กำหนดเส้นทางไปยังไฟล์ data.yaml
data_file = 'datasets/AIE223_Project-3/data.yaml'

# อ่านชื่อคลาสจากไฟล์ data.yaml
with open(data_file, 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['names']  # ดึงชื่อคลาส

# กำหนดโฟลเดอร์ที่มีไฟล์ภาพ
image_folder = 'test_image'  # ใส่ path ของfolder ที่มี รูปภาพที่ต้องการจะ test
result_folder = 'predict_result'  # ใส่ path ของfolder สำหรับบันทึกผลลัพธ์

# สร้างโฟลเดอร์สำหรับบันทึกผลลัพธ์ถ้ายังไม่มี
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# รับรายชื่อไฟล์ภาพในโฟลเดอร์
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Define a color mapping for each class
def get_color(class_id):
    random.seed(class_id)  # Ensure consistent color for each class
    return [random.randint(0, 255) for _ in range(3)]  # Random color

for image_file in image_files:
    # โหลดภาพจากไฟล์
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    results = model.predict(frame, conf=0.7, show=False)

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
    plt.title(image_file)  # แสดงชื่อไฟล์ภาพ
    plt.show(block=False)

    # รอให้ผู้ใช้กดปุ่ม
    plt.waitforbuttonpress()

    # บันทึกภาพที่มี bounding box ลงในโฟลเดอร์ predict_result
    result_path = os.path.join(result_folder, image_file)
    cv2.imwrite(result_path, frame)  # บันทึกภาพ

    plt.clf()  # เคลียร์ภาพก่อนแสดงภาพถัดไป

print("Processing complete.") 