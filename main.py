import os
import sys
import cv2
from invariant_features import ImageStitcher, load_frames

def list_files(directory):
    file_directories = []
    for file in sorted(os.listdir(directory)):
        file_directories.append(directory+file)
    return file_directories

def stitch_image(file_directories: list, output_dir: str, skip_size = 5):
    stitcher = ImageStitcher()
    for idx, frame in enumerate(load_frames(file_directories)):
        print(f"[Log] Working on index {idx+1} of {int(len(file_directories)/skip_size)}")
        stitcher.add_image(frame)
        stitched_image = stitcher.image()
        cv2.imwrite(output_dir+str(idx)+".png", stitched_image)
    cv2.imwrite(output_dir+"0_final_image.png", stitched_image)

if __name__ == "__main__":
    try:
        print(sys.argv)
        stitch_image(list_files(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
    except Exception as e:
        print("[Log] Error!")
        print(e)