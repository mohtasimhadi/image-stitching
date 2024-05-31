import os
import sys
import cv2
import traceback
from utils.file_processing import load_frames
from invariant_features import ImageStitcher

def list_files(directory):
    file_directories = []
    for file in sorted(os.listdir(directory)):
        file_directories.append(directory+file)
    return file_directories

def stitch_image(file_directories: list, output_dir: str, skip_size = 5):
    stitcher = ImageStitcher()
    for idx, frame in enumerate(load_frames(file_directories)):
        print(f"\033[94m[Log]\033[0m Stitching image {idx+1} of {int(len(file_directories)/skip_size)} images.")
        stitcher.add_image(frame)
        stitched_image = stitcher.image()
        cv2.imwrite(output_dir+str(idx)+".png", stitched_image)
    cv2.imwrite(output_dir+"0_final_image.png", stitched_image)

if __name__ == "__main__":
    try:
        stitch_image(list_files(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
    except Exception as e:
        frame = traceback.extract_tb(e.__traceback__)[-1]
        print("\033[91m[Error]\033[0m    {}\n    Line: {}\n    File: {}".format(e, frame.lineno, os.path.relpath(frame.filename)))