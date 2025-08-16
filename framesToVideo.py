import cv2
import os
import glob

frame_folder = r"D:/Moving Crowd Anomaly/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test018"
output_video = "test018.mp4"
fps = 25

# Get all tif frames (case-insensitive)
frame_paths = sorted(glob.glob(os.path.join(frame_folder, "*.tif")))
if not frame_paths:  # Try uppercase extension too
    frame_paths = sorted(glob.glob(os.path.join(frame_folder, "*.TIF")))

if not frame_paths:
    raise FileNotFoundError(f"No .tif images found in {frame_folder}")

# Read the first frame
first_frame = cv2.imread(frame_paths[0])
if first_frame is None:
    raise ValueError(f"Could not read first frame: {frame_paths[0]}")

height, width, layers = first_frame.shape

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write frames
for frame_path in frame_paths:
    frame = cv2.imread(frame_path)
    if frame is not None:
        video_writer.write(frame)

video_writer.release()
print(f"Video saved as: {output_video}")
