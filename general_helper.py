import cv2
from matplotlib import pyplot as plt

def load_images(image_paths):
    images = []
    for path in image_paths:
        images.append(cv2.imread(path))
    return images

def convert_to_grayscale(images):
    grayscale_images = []
    for image in images:
        grayscale_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return grayscale_images

def save_image(image, path):
    plt.imshow(image)
    cv2.imwrite(path, image)