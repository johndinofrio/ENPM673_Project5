import numpy as np
import cv2 as cv2
from Oxford_dataset import ReadCameraModel
from Oxford_dataset import UndistortImage
from matplotlib import pyplot as plt
import glob

#get the image
img1 =[cv2.imread(File) for File in glob.glob("./Oxford_dataset/stereo/centre_demo/*.png")]
img2 = np.array(img1)

for img_num in range(0,302):
    img = img2[img_num]
    cv2.imshow('thing1',img)
    cv2.waitKey(0)
    color_image = cv2.cvtColor(img,cv2.COLOR_BayerGR2BGR)
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel("./Oxford_dataset/model")
    undistorted_image = UndistortImage(img,LUT)

    
   
