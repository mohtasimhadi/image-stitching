import cv2
import os
import matplotlib.pyplot as plt
from iterative_closest_point import icp
from invariant_features import ImageStitcher
from scale_invariant_feature_transform import sift

directory = 'frames'
images = []
images_directories = os.listdir(directory)
images_directories.sort()

for filename in images_directories:
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to load image: {img_path}")

def stitch_image_list(images):
    stitcher = ImageStitcher()
    for image in images:
        stitcher.add_image(image)
        cv2.imwrite("latest.png", stitcher.image())
    return stitcher.image()

def recursive_stitching(lst):
    if len(lst) == 1:
        return lst[0]
    
    new_lst = []
    i = 0
    while i < len(lst) - 1:
        new_lst.append(stitch_image_list([lst[i], lst[i + 1]]))
        i += 2
    
    if i == len(lst) - 1:
        new_lst.append(lst[i])
    
    return recursive_stitching(new_lst)

cv2.imwrite(f'out/300-500-20.png', recursive_stitching([images[i] for i in range(300, 500, 20)]))
cv2.imwrite(f'out/300-1800-20.png', recursive_stitching([images[i] for i in range(300, 1850, 20)]))
cv2.imwrite(f'out/300-1800-15.png', recursive_stitching([images[i] for i in range(300, 1850, 15)]))
cv2.imwrite(f'out/300-1800-10.png', recursive_stitching([images[i] for i in range(300, 1850, 10)]))
cv2.imwrite(f'out/300-1850-7.png', recursive_stitching([images[i] for i in range(300, 1850, 7)]))
cv2.imwrite(f'out/300-1850-5.png', recursive_stitching([images[i] for i in range(300, 1850, 5)]))
cv2.imwrite(f'out/300-1850-3.png', recursive_stitching([images[i] for i in range(300, 1850, 3)]))
cv2.imwrite(f'out/300-1850-2.png', recursive_stitching([images[i] for i in range(300, 1850, 2)]))
cv2.imwrite(f'out/300-1850-1.png', recursive_stitching([images[i] for i in range(300, 1850, 1)]))
