from typing import List, Any, Tuple
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import cv2 as cv2
from scipy.ndimage import map_coordinates as interp2
import scipy.optimize as opt
# from Oxford_dataset import ReadCameraModel
from Oxford_dataset import UndistortImage
from matplotlib import pyplot as plt
import glob
import os
import random

from numpy.linalg import matrix_rank


def UndistortImage(image, LUT):
    reshaped_lut = LUT[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
    undistorted = np.rollaxis(
        np.array([interp2(image[:, :, channel], reshaped_lut, order=1) for channel in range(0, image.shape[2])]), 0, 3)

    return undistorted.astype(image.dtype)


def ReadCameraModel():
    models_dir = "Oxford_dataset/model"

    intrinsics_path = models_dir + "/stereo_narrow_left.txt"
    lut_path = models_dir + "/stereo_narrow_left_distortion_lut.bin"

    intrinsics = np.loadtxt(intrinsics_path)
    # Intrinsics
    fx = intrinsics[0, 0]
    fy = intrinsics[0, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[0, 3]
    # 4x4 matrix that transforms x-forward coordinate frame
    # at camera origin and image frame for specific lens
    G_camera_image = intrinsics[1:5, 0:4]
    # LUT for undistortion
    # LUT consists of (u,v) pair for each pixel)
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size // 2])
    LUT = lut.transpose()

    return fx, fy, cx, cy, G_camera_image, LUT


def findOnlyPoints(img_old, img_new):
    SIFTimg = img_new.copy
    refinedSIFTimg = img_new.copy
    old = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

    # sift
    orb = cv2.ORB_create(nfeatures=500)
    # kp,descriptors = orb.detectAndCompute(old, None)

    keypoints_1, descriptors_1 = orb.detectAndCompute(old, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(new, None)
    SIFTimg = cv2.drawKeypoints(img_new, keypoints_2, None)

    cv2.imshow("points", SIFTimg)
    cv2.moveWindow("points", 10, 10)

    # Matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    im3 = cv2.drawMatches(img_old, keypoints_1, img_new, keypoints_2, matches[0:20], None, matchColor=(20, 20, 20),
                          singlePointColor=(200, 200, 200), matchesMask=None, flags=0)
    cv2.imshow("matches", im3)
    cv2.moveWindow("matches", 10, 400)
    idx = matches[1:20]
    # Initialize keypoints list
    list_kp1 = []  # old frame
    list_kp2 = []  # new frame
    # Ratio test to see which keypoints are the best
    for i in idx:
        list_kp1.append(keypoints_1[i.queryIdx].pt)
        list_kp2.append(keypoints_2[i.trainIdx].pt)

    return np.array(list_kp1), np.array(list_kp2)

    # # Initialize key points list lists
    # list_kp1 = []
    # list_kp2: List[Tuple[Any, Any]] = []
    # # For each match...
    # for mat in matches:
    #
    #     # Get the matching keypoints for each of the images
    #     img1_idx = mat.queryIdx
    #     img2_idx = mat.trainIdx
    #
    #     # x - columns
    #     # y - rows
    #     # Get the coordinates
    #     (x1, y1) = keypoints_1[img1_idx].pt
    #     (x2, y2) = keypoints_2[img2_idx].pt
    #
    #     # Append to each list
    #     list_kp1.append((x1, y1))
    #     list_kp2.append((x2, y2))
    #
    #     if len(list_kp1)>7:
    #         return list_kp1,list_kp2



def loadImages(path=".png"):
    return [os.path.join(path, ima) for ima in os.listdir(path)]

files = loadImages("Oxford_dataset/stereo/centre")
files.sort()
images = []
count = 0
fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel()
x = []
y = []
z = []
u = []
v = []
w = []
fig = plt.figure()
ax = fig.gca(projection='3d')

for file in files:
    images = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    # cv2.imshow("video", images)

    color_image = cv2.cvtColor(images, cv2.COLOR_BayerGR2BGR)
    # Camera intrinsic parameters
    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    if count > 0:
        img_old = img_new.copy()

    # img_new=color_image
    img_new = UndistortImage(color_image, LUT)
    img_new = cv2.resize(img_new, (400, 300))

    if count > 0:
        # Perform all matrix operations
        P1, P2 = findOnlyPoints(img_new, img_old)
        k = np.array(K)
        E, mask = cv2.findEssentialMat(P1, P2, k, cv2.FM_RANSAC)
        E = np.mat(E)
        N, R, C, mask = cv2.recoverPose(E, P1, P2)
        print("R", R)
        print("C", C)

        x.append(float(C[0]))
        y.append(float(C[1]))
        z.append(float(C[2]))
        # Make the direction data for the arrows
        print(float(np.sin(np.pi * x[-1]) * np.cos(np.pi * y[-1]) * np.cos(np.pi * z[-1])))
        u.append(float(np.sin(np.pi * x[-1]) * np.cos(np.pi * y[-1]) * np.cos(np.pi * z[-1])))
        v.append(float(-np.cos(np.pi * x[-1]) * np.sin(np.pi * y[-1]) * np.cos(np.pi * z[-1])))
        w.append(float((np.sqrt(2.0 / 3.0) * np.cos(np.pi * x[-1]) * np.cos(np.pi * y[-1]) * np.sin(np.pi * z[-1]))))

    cv2.imshow("Undistorted Img", img_new)
    cv2.moveWindow("Undistorted Img", 425,10)

    count += 1
    print(count)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        np.savez('mat.npz', x, y, z, u, v, w)
        break

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
plt.show()

# https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift-create-not-working-even-though-have-contrib-instal
