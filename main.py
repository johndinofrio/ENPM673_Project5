import numpy as np
import cv2 as cv2
from scipy.ndimage import map_coordinates as interp2
# from Oxford_dataset import ReadCameraModel
from Oxford_dataset import UndistortImage
from matplotlib import pyplot as plt
import glob
import os
import numpy as np


def UndistortImage(image, LUT):
    print(image.shape)
    print(LUT.shape)
    reshaped_lut = LUT[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
    undistorted = np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1) for channel in range(0, image.shape[2])]), 0, 3)

    return undistorted.astype(image.dtype)

def ReadCameraModel():

    models_dir="Oxford_dataset/model"

    intrinsics_path =models_dir  + "/stereo_narrow_left.txt"
    lut_path =models_dir + "/stereo_narrow_left_distortion_lut.bin"

    intrinsics = np.loadtxt(intrinsics_path)
    # Intrinsics
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]
    # 4x4 matrix that transforms x-forward coordinate frame at camera origin and image frame for specific lens
    G_camera_image = intrinsics[1:5,0:4]
    # LUT for undistortion
    # LUT consists of (u,v) pair for each pixel)
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size//2])
    LUT = lut.transpose()

    return fx, fy, cx, cy, G_camera_image, LUT

def loadImages(path = ".png"):

    return [os.path.join(path, ima) for ima in os.listdir(path)]
files=loadImages("Oxford_dataset/stereo/centre")
images = []
for file in files:
    images=cv2.imread(file, cv2.IMREAD_UNCHANGED)
    cv2.imshow("video", images)
    cv2.waitKey(10)
    color_image = cv2.cvtColor(images, cv2.COLOR_BayerGR2BGR)
    cv2.imshow("Original", color_image)
    cv2.waitKey(10)
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel()
    undistorted_image = UndistortImage(color_image, LUT)
