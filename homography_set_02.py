## CODE DESCRIPTION ##

# THIS SCRIPT IS THE SECOND SCRIPT TO BE RUN IN THE 'pedestrian_mapping' SET.
# FIRST YOU NEED TO RUN:
# 1. 'video_to_images_01.py'

# AFTER THIS SCRIPT YOU CAN RUN:
# 3. 'detect_and_map_03.py'


# THIS SCRIPT CREATES A HOMOGRAPHY MATRIX TO TRANSFORM ONE IMAGE TO ANOTHER.
# IT'S PURPOSE IS TRANSFORM POINTS OF DETECTIONS OF PEOPLE FROM A CAMERA (APPROX 15DEG ANGLE) TO A TOP VIEW SATELLITE
# IMAGE.
# THESE POINTS ARE THEN PLOTTED TOGETHER TO CREATE A HEATMAP OF PEDESTRIAN MOVEMENT THROUGH A SPACE.



## STEPS
# 1. CREATE FUNCTION TO MANUALLY SELECT POINTS IN IMAGE.
# 2. READ IN SOURCE IMAGE (FROM VIDEO USED TO TRACK PEDESTRIANS) AND CHOOSE 4 POINTS.
# 3. READ IN THE DESTINATION IMAGE (TOP VIEW SATELLITE IMAGE) AND CHOOSE 4 CORRESPONDING POINTS THAT MATCH THE LOCATIONS
#    IN THE SOURCE IMAGE.
# 4. CONVERT EACH SET OF THE COORDINATES OF THESE POINTS TO A USABLE ARRAY.
# 5. CREATE HOMOGRAPHY MATRIX FROM THESE TWO ARRAYS.

## PREPARATION

# STEP 0.1. LOAD REQUIRED PACKAGES
import numpy as np
from ultralytics import YOLO
import cv2

# 1. CREATE FUNCTION TO MANUALLY SELECT POINTS IN IMAGE.
def select_point(event, x, y, flags, param):
    global points, img_name

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(img_name, param)
        points.append((x, y))

        if len(points) == 4:
            cv2.destroyAllWindows()

# 2. READ IN SOURCE IMAGE (IMAGE FROM VIDEO USED TO TRACK PEDESTRIANS)
img1 = cv2.imread('input_images/video_image_3.jpg')
img_src = img1
w, h = img1.shape[:2]
nw = int(w *2/3)
nh = int(h *2/3)
img_src = cv2.resize(img1, (nh,nw))

# 3. READ IN DESTINATION IMAGE (TOP VIEW SATELLITE IMAGE YOU WANT TO CONVERT TO)
img2 = cv2.imread('input_images/satellite_image_3.jpg')
img_dst = img2
#img_dst = cv2.resize(img2, (600,800))

# 이미지 1에서 꼭지점 선택
points = []
img_name = 'Image 1'
cv2.imshow(img_name, img_src)
cv2.setMouseCallback(img_name, select_point, img_src)
cv2.waitKey(0)

points_img1 = points.copy()

# 이미지 2에서 꼭지점 선택
points = []
img_name = 'Image 2'
cv2.imshow(img_name, img_dst)
cv2.setMouseCallback(img_name, select_point, img_dst)
cv2.waitKey(0)

points_img2 = points.copy()

# print("Selected points in Image 1:", points_img1)
# print("Selected points in Image 2:", points_img2)

# 4. CONVERT EACH SET OF THE COORDINATES OF THESE POINTS TO A USABLE ARRAY.
point_src = np.array(points_img1).reshape(4,2)
point_dst = np.array(points_img2).reshape(4,2)

print(point_src)
print(point_dst)

# 5. CREATE HOMOGRAPHY MATRIX FROM THESE TWO ARRAYS.
# 호모그라피에서 변환된 행렬 찾기
# h가 변환에 이용된 바로, 3 X 3 의 행렬이다.

h, status = cv2.findHomography(point_src, point_dst)
# 호모그라피 행렬 호출용
def get_homography_matrix():
    return h

img_output = cv2.warpPerspective(img_src, h, (img_src.shape[1], img_src.shape[0]))

cv2.imshow('SRC', img_src)
cv2.imshow('DST', img_dst)
cv2.imshow('Warp', img_output)

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('output_images/source_image_hom_points.jpg', img_src)
cv2.imwrite('output_images/destination_image_hom_points.jpg', img_dst)
cv2.imwrite('output_images/transformation_image_hom_points.jpg', img_output)