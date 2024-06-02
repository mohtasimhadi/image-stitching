import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

def convert_to_grayscale(images):
    grayscale_images = []
    for image in images:
        grayscale_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return grayscale_images

def calculate_fov(width, focal_length):
    return 2 * math.degrees(math.atan(width / (2 * focal_length)))

def calculate_dfov(hfov, vfov):
    return math.sqrt(hfov ** 2 + vfov ** 2)

def combine_images(depth_image1, depth_image2, rgb_image):
    depth1 = cv2.imread(depth_image1, cv2.IMREAD_GRAYSCALE)
    depth2 = cv2.imread(depth_image2, cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(rgb_image, cv2.IMREAD_COLOR)

    depth1 = cv2.resize(depth1, (rgb.shape[1], rgb.shape[0]))
    depth2 = cv2.resize(depth2, (rgb.shape[1], rgb.shape[0]))

    depth_stack = np.stack((depth1, depth2, np.zeros_like(depth1)), axis=-1)

    rgb_d = np.concatenate((rgb, depth_stack), axis=-1)

    return rgb_d