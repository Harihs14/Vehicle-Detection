from ultralytics import YOLO
import cv2
import math

# Running real-time from webcam
cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam, you can change it based on your webcam index

# Load YOLOv5 model (replace 'best.pt' with your actual model path)
model = YOLO('final.pt')

# Class names for vehicle detection
classnames = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

# Initialize vehicle counters for each class
vehicle_count = {class_name: 0 for class_name in classnames}


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    
    # Perform inference with YOLOv5
    result = model(frame, stream=True)

    # Process bounding boxes and display results
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            class_index = int(box.cls[0])

            if confidence > 50 and classnames[class_index] in classnames:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Increment the vehicle count for each detected vehicle by class
                vehicle_count[classnames[class_index]] += 1

                # Display bounding box and class label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(frame, f'{classnames[class_index]} {confidence}%', (x1 + 8, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame with vehicle count by class
    for class_name, count in vehicle_count.items():
        cv2.putText(frame, f'{class_name}: {count}', (10, 50 + classnames.index(class_name) * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Vehicle Detection', frame)

    # Press 'Esc' key to exit the loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
