import os, sys, traceback
from iterative_closest_point import icp
from invariant_features import ImageStitcher
from scale_invariant_feature_transform import sift
from utils.file_processing import load_frames, list_files, save_image
from utils.video_processing import extract_frames

def stitch_image(file_directories: list, output_dir: str, skip_size = 5):
    stitcher = ImageStitcher()
    interval_frames = []
    for idx, frame in enumerate(load_frames(file_directories)):
        print(f"\033[94m[Log]\033[0m Stitching image {idx+1} of {int(len(file_directories))} images.")
        stitcher.add_image(frame)
        stitched_image = stitcher.image()
        if idx % skip_size == 0:
            stitcher = ImageStitcher()
            interval_frames.append(stitched_image)
            save_image(output_dir+str(idx)+".png", stitched_image)
    stitcher = ImageStitcher()
    for idx, frame in enumerate(interval_frames):
        print(f"\033[94m[Log]\033[0m Stitching interval image {idx+1} of {int(len(interval_frames))} images.")
        stitcher.add_image(frame)
        stitched_image = stitcher.image()
        save_image(output_dir+"interval"+str(idx)+".png", stitched_image)
    save_image(output_dir+"0_final_image.png", stitched_image)

if __name__ == "__main__":
    try:
        if sys.argv[1] == 'extract_frames':
            extract_frames(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
        elif sys.argv[1] == 'stitch_image':
            stitch_image(list_files(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    except Exception as e:
        frame = traceback.extract_tb(e.__traceback__)[-1]
        print("\033[91m[Error]\033[0m    {}\n    Line: {}\n    File: {}".format(e, frame.lineno, os.path.relpath(frame.filename)))