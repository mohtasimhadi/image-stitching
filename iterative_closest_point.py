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

def stitch_images(image1, image2):
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

if __name__ == "__main__":
    # Load two images
    image1 = cv2.imread('s1.jpeg')
    image2 = cv2.imread('s2.jpeg')
    
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Stitch images
    panorama = stitch_images(image1, image2)
    
    # Display the panorama
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
