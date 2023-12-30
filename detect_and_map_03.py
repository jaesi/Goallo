'''
< CODE DESCRIPTION >

TO RUN THIS SCRIPT YOU MUST FIRST RUN:
1. 'video_to_images_01.py'
2. 'homography_set_02.py'

THE PURPOSE OF THIS SCRIPT IS TO EXTRACT THE COORDINATES OF DETECTED PEOPLE IN IMAGES, AND THEN PLOT THEIR TRANSFORMED
POINTS ON A SINGLE HEAT MAP. THUS MAPPING PEDESTRIAN MOVEMENT.

# STEPS
1. READ IN IMAGE
2. PERFORM OBJECT DETECTION
3. SAVE BOUNDING BOX COORD RESULTS TO A DATAFRAME
4. RUN FUNCTION OVER ALL IMAGES IN FOLDER, WHILST APPENDING DATAFRAME WITH EVERY BOUNDING BOC COORD
5. CREATE NEW VARIABLE IN DATAFRAME THAT CALCULATES CENTRE POINTS FROM COORDS
6. CREATE NEW VARIABLE IN DATAFRAME THAT CALCULATES HOMOGRAPHY TRANSFORMATION OF CENTRE POINTS
7. PLOT ALL TRANSFORMED CENTRE POINTS ON HEATMAP
'''

import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from homography_set_02 import get_homography_matrix

# STEP 0.2. LOAD DESIRED YOLO MODEL (MEDIUM, LARGE, ETC.)
model = YOLO("Yolo-Weights/yolov8l.pt")

# STEP 0.3. SET DEVICE (GPU vs CPU)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device).eval()

# STEP 0.4. ASSIGN IMAGE FOLDER
images_folder = Path("video_frames")

# STEP 0.5. CREATE AN EMPTY DATAFRAME FOR THE BOUNDING BOX COORDS TO BE ADDED TO
data = []


## BEGIN PROCESS

# CREATE FUNCTION TO PERFORM OBJECT DETECTION ON IMAGE
def detect_people(image_path):
    img = Image.open(image_path)  # Load image using PIL
    img = np.array(img)  # Convert to numpy array

    results = model(img)  # Perform detection
    classes = results[0].boxes.cls  # Save list of detected classes
    # datas = results[0].boxes.xywh  # Bounding box XY and WH of person detections

    # Extract bounding box coordinates for people
    bboxes = results[0].boxes.xyxy

    return bboxes, classes, str(image_path)


# PERFORM DETECTION ON IMAGES IN THE FOLDER
for image_path in tqdm(images_folder.glob("*.png")):  # Iterate through each image
    bounding_boxes, class_labels, image_path_str = detect_people(image_path)  # Detect objects and get info
    for bbox, label in zip(bounding_boxes, class_labels):
        data.append({
            'Image': image_path_str,
            'xmin': bbox[0],
            'ymax': bbox[1], # I've changed
            'xmax': bbox[2],
            'ymin': bbox[3], # this two variables
            'Class': label
        })

# Create a DataFrame from the list of dictionaries
data = pd.DataFrame(data)

# Function to extract integer value from tensor : 왜 필요한지 체크해보기
def extract_integer(tensor_value):
    return int(tensor_value.item()) if isinstance(tensor_value, torch.Tensor) else int(tensor_value)

# Convert tensor values to integers in the DataFrame for specified columns
for col in ['xmin', 'ymin', 'xmax', 'ymax', 'Class']:
    data[col] = data[col].apply(extract_integer)

# FILTER DATA FOR DETECTED PERSONS ONLY
data = data[data['Class'] == 0]

# CREATE AN X-MID VALUE FOR THE CENTRE POINT
data['x_center'] = (data['xmin'] + data['xmax']) / 2

test_data = data.iloc[::3]

# PERFORM HOMOGRAPHY TRANSFORMATION ON POINTS
# Separate center points for easy computation
center_points = data[['x_center', 'ymin']].values
center_points = np.array(center_points)
# center_points_np = np.array(center_points, dtype=np.float32).reshape(-1, 1, 2)

h = get_homography_matrix()

# # 호모그래피 변환 적용
# transformed_points = cv2.perspectiveTransform(center_points_np, h)

# Apply homography transformation using matrix multiplication (NOTE: same results as cv2 method)
# Add a column of ones to represent homogeneous coordinates
ones_column = np.ones((len(center_points), 1))
homogeneous_points = np.hstack((center_points, ones_column))
transformed_points = np.dot(homogeneous_points, h.T)

# Normalize the transformed points by dividing by the third coordinate (homogeneous coordinate)
normalized_transformed_points = transformed_points[:, :-1] / transformed_points[:, [-1]]

# Convert transformed points array to joinable dataframe
transformed_data = pd.DataFrame(normalized_transformed_points, columns=['transformed_x', 'transformed_y'])

# Display the transformed DataFrame
print(transformed_data)

# Join the transformed_data with the original data
merged_data = pd.concat([data.reset_index(drop=True), transformed_data.reset_index(drop=True)], axis=1)


# REMOVE OUTLIERS (THERE ARE SOME VERY EXTREME VALUES AFTER TRANSFORMATION)
# Calculate Z-scores for 'transformed_x' and 'transformed_y'
z_scores_x = np.abs((merged_data['transformed_x'] - merged_data['transformed_x'].mean()) / merged_data['transformed_x'].std())
z_scores_y = np.abs((merged_data['transformed_y'] - merged_data['transformed_y'].mean()) / merged_data['transformed_y'].std())

# Define a threshold for outlier detection (e.g., z-score greater than 3)
threshold = 3

# Filter the DataFrame to exclude rows with outliers in 'transformed_x' and 'transformed_y'
filtered_data = merged_data[(z_scores_x <= threshold) & (z_scores_y <= threshold)]

# Display the filtered DataFrame
print(filtered_data)



# SAVE DATAFRAME TO .csv
filtered_data.to_csv('detections_data/detections_data.csv', index=False)

# MAP UN-TRANSFORMED POINTS ON HEATMAP
plt.figure(figsize=(6, 8))
plt.scatter(filtered_data['x_center'], filtered_data['ymin'], alpha=0.6, cmap='viridis')
# plt.colorbar()  # Add colorbar
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter-plot of Center Points (untransformed)')
plt.grid(True)
plt.savefig('output_images/center_point_plot.jpg')
plt.show()

# MAP TRANSFORMED POINTS ON HEATMAP
plt.figure(figsize=(6, 8))
plt.scatter(filtered_data['transformed_x'], filtered_data['transformed_y'], alpha=0.6, cmap='viridis')
# plt.colorbar()  # Add colorbar
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
# plt.xlim(-1000, 1000)
# plt.ylim(-10000, 10000)
plt.title('Scatter-plot of Transformed Points')
plt.grid(True)
plt.savefig('output_images/transformed_points_plot.jpg')
plt.show()


# 03_HEATMAP
plt.figure(figsize=(6, 8))
# hist2d
plt.hist2d(filtered_data['transformed_x'], filtered_data['transformed_y'], bins=(20,35), cmap='Reds')

plt.colorbar()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
# plt.xlim(-1000, 1000)
# plt.ylim(-10000, 10000)
plt.title('Heatmap of Transformed Points')
plt.grid(True)

plt.savefig('output_images/transformed_points_heatmap.jpg')
plt.show()