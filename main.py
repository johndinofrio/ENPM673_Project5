import numpy as np
import cv2 as cv2
from Oxford_dataset import ReadCameraModel
from Oxford_dataset import UndistortImage
from matplotlib import pyplot as plt
import glob
import os

def loadImages(path = ".png"):

    return [os.path.join(path, ima) for ima in os.listdir(path)]

files=loadImages("Oxford_dataset/stereo/centre")
images = []
for file in files:
    images=cv2.imread(file, cv2.IMREAD_UNCHANGED)
    print(images)
    cv2.imshow("video", images)
    cv2.waitKey(10)
    color_image = cv2.cvtColor(images, cv2.COLOR_BayerGR2BGR)
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel("./Oxford_dataset/model")
    undistorted_image = UndistortImage(images, LUT)
