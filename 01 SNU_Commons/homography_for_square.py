import cv2
import numpy as np

point_src = [[380,173],
             [151, 368],
             [541, 301],
             [546, 188]]



point_dst = [[308, 166],
 [308, 715],
 [535, 704],
 [850, 311]]

point_src = np.array(point_src)
point_dst = np.array(point_dst)

# 호모그라피에서 변환된 행렬 찾기

h, status = cv2.findHomography(point_src, point_dst)

# 호모그라피 행렬 호출용
def get_homography_matrix():
    return h