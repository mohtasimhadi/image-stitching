import cv2
import math
from matplotlib import pyplot as plt

def convert_to_grayscale(images):
    grayscale_images = []
    for image in images:
        grayscale_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return grayscale_images

def calculate_fov(width, focal_length):
    return 2 * math.degrees(math.atan((2 * width) / (2 * focal_length)))