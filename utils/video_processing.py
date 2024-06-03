import cv2
from utils.file_processing import save_image

def extract_frames(video_path, output_folder, video_type, fps=1):
    print(f"\033[94m[Log]\033[0m Extracting frames from {video_path}")
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    frame_count = 0
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))

    while success:
        if frame_count % frame_interval == 0:
            frame_filename = f"{output_folder}/{video_type}_frame_{frame_count:04d}.jpg"
            save_image(frame_filename, frame)

        success, frame = video_capture.read()
        frame_count += 1
    video_capture.release()