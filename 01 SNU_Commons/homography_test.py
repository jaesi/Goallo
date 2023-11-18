import cv2
import numpy as np

# 두개의 사진 가지고 같은 방향으로 맞추기
# 네개의 좌표점이 필요

# 첫번째 이미지.
import cv2

# 마우스 콜백 함수
def select_point(event, x, y, flags, param):
    global points, img_name

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(img_name, param)
        points.append((x, y))

        if len(points) == 4:
            cv2.destroyAllWindows()

# 이미지 불러오기
img1 = cv2.imread('Images/square_1.jpg')
img_src = img1
# img_src = cv2.resize(img1, (600, 800))


img2 = cv2.imread('Images/square_satellite2.jpg')
img_dst = img2
# img_dst = cv2.resize(img2, (600,800))

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


point_src = np.array(points_img1).reshape(4,2)

point_dst = np.array(points_img2).reshape(4,2)

# 호모그라피에서 변환된 행렬 찾기
# h가 변환에 이용된 바로, 3 X 3 의 행렬이다.

h, status = cv2.findHomography(point_src, point_dst)

img_output = cv2.warpPerspective(img_src, h, (img_src.shape[1], img_src.shape[0]))

cv2.imshow('SRC', img_src)
cv2.imshow('DST', img_dst)
cv2.imshow('Warp', img_output)

cv2.waitKey()
cv2.destroyAllWindows()

print(point_src)
print()
print(point_dst)