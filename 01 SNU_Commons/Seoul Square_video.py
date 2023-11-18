import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cvzone
import matplotlib.pyplot as plt
import math
import time

# 동영상 불러오기
# image = cv2.imread('Images/square_2.png')
vid = cv2.VideoCapture('Video/Square_1.mp4')

# 이미지 구경
# cv2.imshow('Loaded Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from homography_for_square import get_homography_matrix

# h 행렬 가져오기
homog = get_homography_matrix()


# YOLO 모델 불러오기
model = YOLO("Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

'''
# 이미지를 모델에 전달하여 객체 감지 수행
results = model(image)
detected_classes = results[0].boxes.cls

# 발밑 센터포인트 찍기
datas = results[0].boxes.xywh
xydatas = results[0].boxes.xyxy

center_points = []

# 원본 이미지에 발밑 센터포인트 표시
for i, data in enumerate(xydatas):
    # 클래스가 사람인 경우만 처리
    if detected_classes[i] == 0:
        x1, y1, x2, y2 = data
        x_center = int((x1 + x2)/2)
        y_bottom = int(y2)
        cv2.circle(image, (x_center, y_bottom), 5, (127, 0, 255), -1)
        # 좌표를 리스트에 추가
        center_points.append((x_center, y_bottom))
'''

desired_fps = 20
frame_time = 1.0 / desired_fps
center_points = []

while True:
    frame_start_time = time.time()
    success, img = vid.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))
            x_center = int((x1 + x2) / 2)
            y_bottom = int(y2)
            cv2.circle(img, (x_center, y_bottom), 3, (10, 255, 255), -1)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            center_points.append((x_center, y_bottom))

            # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.5, thickness=1)

    # fps_display = f"FPS: {int(fps)}"
    # cv2.putText(img, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    frame_end_time = time.time()
    elapsed_time = frame_end_time - frame_start_time

    # 대기 시간 계산 및 적용
    sleep_time = frame_time - elapsed_time
    if sleep_time > 0:
        time.sleep(sleep_time)
        final_time =time.time()
        fps = 1/(final_time-frame_start_time)
        print("Final fps :", fps)
    else:
        print("Sleep_time :", sleep_time)

    # 호모그래피 변환 적용
    center_points = np.array(center_points)
    center_points_np = np.array(center_points, dtype=np.float32).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(center_points_np, homog)
    # 시각화
    plt.figure(figsize=(10, 10))

    for point in transformed_points:
        plt.scatter(point[0][0], point[0][1], c='hotpink', s=100, marker='X')
    plt.title("Seoul Square Transformed Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    # plt.savefig('Output_Images/SecondPlott!!!!.jpg')
    plt.show()







