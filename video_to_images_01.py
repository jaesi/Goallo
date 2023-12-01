## CODE DESCRIPTION ##

# THIS SCRIPT IS THE FIRST SCRIPT TO BE RUN IN THE 'pedestrian_mapping' SET.
# AFTERWARD YOU CAN RUN:
# 2. 'homography_set_02.py'
# 3. 'detect_and_map_03.py'

# THIS SCRIPT EXTRACTS AND SAVES FRAMES FROM A VIDEO
# IT'S PURPOSE IS TO CREATE A SET OF IMAGES FOR THE FOLLOWING SCRIPTS TO USE:
# 1. 'homography_set_02.py'
# 2. 'detect_and_map_03.py'



## STEPS
# 1. IMPORT VIDEO
# 2. ASSIGN FRAME INTERVAL
# 3. EXPORT IMAGES


import cv2
import os

def export_frames(video_path, output_folder, interval_sec):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get frames per second and calculate frames to skip for the desired interval
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * interval_sec)

    # Read and export frames
    frame_count = 0
    for i in range(total_frames):
        ret, frame = cap.read()

        # if not ret:
        #     break

        # Export frame every 'frame_skip' frames
        if ret & (frame_count % frame_skip == 0):
            frame_filename = f"frame_{int(frame_count / frame_skip):04d}.png"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    video_path = "input_video/GH013662.mp4"  # Replace with your video file path
    output_folder = "video_frames"  # Replace with your desired output folder

    export_frames(video_path, output_folder, 1) # Set the frame rate that you want to extract