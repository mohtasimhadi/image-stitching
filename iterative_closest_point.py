import numpy as np
import cv2

def find_corresponding_points(image1, image2):
    # Detect keypoints and compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
    
    # Match descriptors between the images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    
    # Extract matched keypoints
    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    return matched_keypoints1, matched_keypoints2

def stitch_images_icp(images):
    image1 = images[0]
    image2 = images[1]

    # Find corresponding points
    matched_keypoints1, matched_keypoints2 = find_corresponding_points(image1, image2)
    
    # Estimate the transformation matrix using the ICP algorithm
    transformation_matrix, _ = cv2.estimateAffinePartial2D(matched_keypoints2, matched_keypoints1)
    
    # Warp image2 to image1
    height, width = image1.shape[:2]
    stitched_image = cv2.warpAffine(image2, transformation_matrix, (width, height))
    
    # Blend the two images
    panorama = cv2.addWeighted(image1, 0.5, stitched_image, 0.5, 0)
    
    return panorama