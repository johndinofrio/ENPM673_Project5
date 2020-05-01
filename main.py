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
##    print(image.shape)
##    print(LUT.shape)
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
    # 4x4 matrix that transforms x-forward coordinate frame
    # at camera origin and image frame for specific lens
    G_camera_image = intrinsics[1:5,0:4]
    # LUT for undistortion
    # LUT consists of (u,v) pair for each pixel)
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size//2])
    LUT = lut.transpose()

    return fx, fy, cx, cy, G_camera_image, LUT


def findPoints(img_old,img_new):
    old = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

    #sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(old,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(new,None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    # Initialize lists
    list_kp1 = []
    list_kp2 = []

    # For each match...
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = keypoints_1[img1_idx].pt
        (x2, y2) = keypoints_2[img2_idx].pt

        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))

        if len(list_kp1)>7:
            return list_kp1,list_kp2
        
# Compute the Fundamental Matrix
def estimateF(p1,p2):
    A = []
    # Append values in A matrix for homography Ax=0
    for i in range(len(p1)):
##                  x1x1'               x1y1'         x1        y1x1'
        row = [p1[i][0]*p2[i][0], p1[i][0]*p2[i][1], p1[i][0], p1[i][1]*p2[i][0], \
##                  y1y1'          y1       x1'       y1'   1
               p1[i][1]*p2[i][1], p1[i][1], p2[i][0], p2[i][1], 1]
        A.append(row)

    # Compute SVD of A matrix to find the 3x3 matrix
    u, s, v = np.linalg.svd(A)
    # 3x3 matrix is the last row of V transpose
    x = v[:][-1]
    F = np.reshape(x,(3,3))
    # Compute SVD again to force rank to be 2
    u, s, v = np.linalg.svd(F,full_matrices=False)
    # Set the last singular value to 0
    s[-1] = 0
    s = np.diag(s)
    F = np.matmul(np.matmul(u,s),v)

    # Return Fundamental Matrix
    return F

# Compute the Essential Matrix
def estimateE(F,K):
    E = np.matmul(np.matmul(np.transpose(K),F),K)

    # Compute SVD to force rank to be 2
    u, s, v = np.linalg.svd(E,full_matrices=False)
    # Set the last singular value to 0
    s[-1] = 0
    s = np.diag(s)
    E = np.matmul(np.matmul(u,s),v)

    # Return Essential Matrix
    return E

def estimateC(E):
    # Compute SVD of the Essential Matrix
    u, s, v = np.linalg.svd(E,full_matrices=False)
    w = [[0,-1,0],[1,0,0],[0,0,1]]

    # Four possible Camera Poses (C) and Rotation Matrices (R)
    C1 = u[:][-1]
    R1 = np.matmul(np.matmul(u,w),v)
    # Correct pose and matrix if determinant is -1
    if np.linalg.det(R1)<0:
        C1 = -C1
        R1 = -R1

    C2 = -u[:][-1]
    R2 = np.matmul(np.matmul(u,w),v)
    if np.linalg.det(R2)<0:
        C2 = -C2
        R2 = -R2
    
    C3 = u[:][-1]
    R3 = np.matmul(np.matmul(u,np.transpose(w)),v)
    if np.linalg.det(R3)<0:
        C3 = -C3
        R3 = -R3

    C4 = -u[:][-1]
    R4 = np.matmul(np.matmul(u,np.transpose(w)),v)
    if np.linalg.det(R4)<0:
        C4 = -C4
        R4 = -R4

    return C1,C2,C3,C4,R1,R2,R3,R4
    
    
    
    

def loadImages(path = ".png"):

    return [os.path.join(path, ima) for ima in os.listdir(path)]
files=loadImages("Oxford_dataset/stereo/centre")
images = []
count = 0
for file in files:
    images=cv2.imread(file, cv2.IMREAD_UNCHANGED)
    cv2.imshow("video", images)
    
    color_image = cv2.cvtColor(images, cv2.COLOR_BayerGR2BGR)
    

    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel()
    # Camera intrinsic parameters 
    K = [[fx,0,cx],[0,fy,cy],[0,0,1]]

    if count>0:
        img_old = img_new
    img_new = UndistortImage(color_image, LUT)

    if count>0:
    # Perform all matrix operations
        points1, points2 = findPoints(img_old,img_new)
        #=========STILL NEED TO DO RANSAC FOR F=========
        F = estimateF(points1, points2)
        E = estimateE(F,K)
        C1,C2,C3,C4,R1,R2,R3,R4 = estimateC(E)
        
    cv2.imshow("Undistorted Img", img_new)


    count+=1
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

#https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift-create-not-working-even-though-have-contrib-instal
