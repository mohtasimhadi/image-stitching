import cv2
import numpy as np

def detect_keypoints_and_descriptors(images):
    sift = cv2.SIFT_create()
    keypoints_and_descriptors = []
    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints_and_descriptors.append((keypoints, descriptors))
    return keypoints_and_descriptors

def match_descriptors(descriptor1, descriptor2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def draw_matching_result(image1, keypoints1, image2, keypoints2, matches):
    matching_result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matching_result

def stitch_images(image1, image2, matches, keypoints1, keypoints2):
    # Extract matching points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp image2 to image1
    h, w = image1.shape[:2]
    warped_image2 = cv2.warpPerspective(image2, M, (w + image2.shape[1], h))

    # Combine image1 and warped image2
    stitched_image = np.zeros_like(warped_image2)
    stitched_image[:h, :w] = image1
    stitched_image = cv2.addWeighted(stitched_image, 0.5, warped_image2, 0.5, 0)

    return stitched_image