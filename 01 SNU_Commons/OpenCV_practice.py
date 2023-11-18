import cv2
import numpy as np

# 3D 포인트 좌표 (원본 3D 이미지)
src_points_3d = np.array([
    [0, 0, 0],  # 예시 3D 포인트 1
    [1, 0, 0],  # 예시 3D 포인트 2
    [0, 1, 0],  # 예시 3D 포인트 3
    [1, 1, 0]   # 예시 3D 포인트 4
], dtype=np.float32)

# 2D 포인트 좌표 (변환된 2D 이미지)
dst_points_2d = np.array([
    [100, 100],  # 예시 2D 포인트 1
    [200, 100],  # 예시 2D 포인트 2
    [100, 200],  # 예시 2D 포인트 3
    [200, 200]   # 예시 2D 포인트 4
], dtype=np.float32)

# Homography 변환 행렬 계산
homography_matrix, _ = cv2.findHomography(src_points_3d, dst_points_2d)


print("Homography 변환 행렬:")
print(homography_matrix)
print("\n3D 포인트 좌표:")
print(src_points_3d)
