import cv2
from matplotlib import pyplot as plt

def load_images(image_paths):
    images = []
    for path in image_paths:
        images.append(cv2.imread(path))
    return images

def save_image(image, path):
    cv2.imwrite(path, image)

def show_image(image):
    plt.imshow(image)