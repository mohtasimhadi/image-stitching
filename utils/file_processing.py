import os, cv2, numpy, pathlib
from typing import List, Generator
from matplotlib import pyplot as plt

def load_frames(paths: List[str]) -> Generator[numpy.ndarray, None, None]:
    for path in paths:
        path = pathlib.Path(path)
        if path.is_dir():
            yield from load_frames(path.rglob('*'))
        elif path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            yield cv2.imread(str(path))

def load_images(image_paths):
    images = []
    for path in image_paths:
        images.append(cv2.imread(path))
    return images

def list_files(directory):
    file_directories = []
    for file in sorted(os.listdir(directory)):
        file_directories.append(directory+file)
    return file_directories

def save_image(path, image):
    cv2.imwrite(path, image)
    print(f"\033[94m[Log]\033[0m Final image saved at \033[94m{path}\033[0m")

def show_image(image):
    plt.imshow(image)