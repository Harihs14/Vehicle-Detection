# Vehicle-Detection
Vehicle Detection using YOLOv8 and OpenCV
This project utilizes the YOLOv8 object detection model and OpenCV to perform real-time vehicle detection from a webcam feed. The YOLO (You Only Look Once) model is known for its efficiency and accuracy in object detection tasks.

Key Features:

Real-time vehicle detection from a webcam stream.
Classifies vehicles into categories such as Ambulance, Bus, Car, Motorcycle, and Truck.
Displays bounding boxes and confidence percentages for detected vehicles.
No vehicle count display for a cleaner output.

Usage:
Install the required dependencies: ultralytics and opencv.

pip install ultralytics opencv-python

Download the YOLOv8 model weights (replace 'final.pt' with your actual model path).
Run the script: python vehicle_detection.py

Note:

Adjust the webcam index in the code if you are using a different webcam.
Ensure the correct path to the YOLOv8 model weights file.
