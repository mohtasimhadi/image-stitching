import cv2
import numpy
import pathlib
from typing import List
from typing import Generator
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

def save_image(image, path):
    cv2.imwrite(path, image)

def show_image(image):
    plt.imshow(image)