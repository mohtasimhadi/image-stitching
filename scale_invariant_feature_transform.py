import cv2

def detect_keypoints_and_descriptors(images):
    """Detect keypoints and compute descriptors using SIFT."""
    sift = cv2.SIFT_create()
    keypoints_and_descriptors = []
    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints_and_descriptors.append((keypoints, descriptors))
    return keypoints_and_descriptors

def match_descriptors(descriptor1, descriptor2):
    """Match descriptors using a brute force matcher."""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def draw_matching_result(image1, keypoints1, image2, keypoints2, matches):
    """Draw matching result."""
    matching_result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matching_result