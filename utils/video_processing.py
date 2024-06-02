import cv2

def extract_frames(video_path, output_folder, video_type, fps=1):
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    frame_count = 0
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))

    while success:
        if frame_count % frame_interval == 0:
            frame_filename = f"{output_folder}/{video_type}_frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_filename, frame)

        success, frame = video_capture.read()
        frame_count += 1
    video_capture.release()