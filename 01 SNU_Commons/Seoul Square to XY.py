import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np

# 이미지 불러오기
image = cv2.imread('Images/square_2.png')

# 이미지 구경
# cv2.imshow('Loaded Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# YOLO 모델 초기화
model = YOLO("Yolo-Weights/yolov8l.pt")

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

# # 원본 이미지에 발밑 센터포인트 표시
# for data in xydatas:
#     # x1, y1, w, h = data
#     x1, y1, x2, y2 = data
#     x_center = int((x1 + x2)/2)
#     y_bottom = int(y2)
#     # 바운딩 박스 그리기
#     # cv2.circle(image, (int(x2), int(y2)), 5, (0, 255, 0), 2)  # 초록색 사각형 그리기
#     cv2.circle(image, (x_center, y_bottom), 5, (0, 0, 255), -1)  # 빨간색 원을 그립니다.

# 이미지를 화면에 표시
cv2.imshow('Image with Center Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 저장
cv2.imwrite('Output_Images/output_image.png', image)

from homography_for_square import get_homography_matrix

center_points = np.array(center_points)
center_points_np = np.array(center_points, dtype=np.float32).reshape(-1, 1, 2)

import matplotlib.pyplot as plt

# h 행렬 가져오기
h = get_homography_matrix()

# 호모그래피 변환 적용
transformed_points = cv2.perspectiveTransform(center_points_np, h)

# 시각화
plt.figure(figsize=(10, 10))

for point in transformed_points:
    plt.scatter(point[0][0], point[0][1], c='hotpink', s=100, marker='X')

plt.title("Seoul Square Transformed Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.savefig('Output_Images/FirstPlott!!!!.jpg')
plt.show()