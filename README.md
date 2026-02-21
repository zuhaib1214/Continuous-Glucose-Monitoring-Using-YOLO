CGM Positioning System using YOLOv8 and Flutter Integration
🩺 Project Overview

This project focuses on real-time detection and localization of Continuous Glucose Monitoring (CGM) devices using a YOLOv8-based deep learning model. The system identifies and tracks CGM sensors on the human body from live camera input, transmitting position data through WebSockets to a Flutter mobile application for visualization and monitoring.

The goal of this project is to enhance the accuracy and automation of CGM device placement validation, providing healthcare professionals and patients with an efficient AI-assisted solution.

🚀 Key Features

YOLOv8 Object Detection for CGM device recognition and localization.

WebSocket Communication for real-time data transmission between backend and mobile client.

Flutter-Based Mobile Application for seamless live visualization of detection results.

Optimized Model Integration for low-latency detection and efficient performance on edge devices.

Scalable Backend Architecture that allows multi-client connections.

🧠 System Architecture
[YOLOv8 Model] --> [WebSocket Server] --> [Flutter Mobile App]
         ↑
   [Live Camera Feed / Input Stream]

The YOLOv8 model processes input frames to detect CGM device positions.

Detection results (bounding boxes, coordinates, confidence scores) are sent to the backend.

WebSocket server transmits data in real-time to the Flutter mobile client.



🧩 Tech Stack
Component	Technology
Deep Learning	Ultralytics YOLOv8

Backend	Python (FastAPI / Flask) + WebSockets
Mobile App	Flutter (Dart)
Communication	WebSocket Protocol
Model Training	PyTorch
Dataset	Custom dataset for CGM device positioning
⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/<your-username>/CGM-Positioning-YOLOv8.git
cd CGM-Positioning-YOLOv8
2. Install Dependencies
pip install -r requirements.txt
3. Run the Backend Server
python web.py
4. Connect the Flutter Application

Update the WebSocket URL in the Flutter app to match your backend IP.

📊 Model Workflow

Data Collection & Annotation — Custom dataset containing CGM device images was prepared and annotated using LabelImg.

Training — YOLOv8 model fine-tuned using transfer learning for precise localization.

Validation — Model evaluated on unseen test samples for accuracy and bounding box precision.

Integration — WebSocket server deployed to transmit real-time detection results to the Flutter app.

🔬 Results

High detection accuracy for CGM positioning.

Low latency in real-time communication.

Smooth mobile visualization of live detection results.

🧭 Future Enhancements

Integration with medical data analytics for glucose trend prediction.

Cloud-based synchronization for long-term tracking.

Model compression for on-device inference.
