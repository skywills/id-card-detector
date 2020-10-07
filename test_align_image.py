import cv2
import numpy as np
import os

MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15

CWD_PATH = os.getcwd()
IMAGE_NAME = 'test_images/000448.jpg'
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
PATH_TO_OUTPUT = os.path.join(CWD_PATH,'output')

def alignImages(im1, im2):
    # convert image to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    #Detect ORB features and comouter descriptors.
    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by score
    matches.sort(key=lambda  x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodmatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodmatches]

    # Draw top matchers
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite(os.path.join(PATH_TO_OUTPUT, "matches.jpg"), imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches),2), dtype=np.float32)
    points2 = np.zeros((len(matches),2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i,:] = keypoints1[match.queryIdx].pt
        points2[i,:] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    img1Reg = cv2.warpPerspective(im1, h, (width, height))

    return img1Reg, h

if __name__ == '__main__':

    # Read reference image
    refFilename = os.path.join(CWD_PATH, 'test_images/ref.jpeg')
    print("reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = PATH_TO_IMAGE
    print("reading image image to align : ", imFilename) 
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    imgReg, h = alignImages(im, imReference)

    cv2.imshow('reg', imgReg)
    cv2.waitKey(0)
    print("Estimated homgraphy: \n", h)
    



