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

# Verify if Fundamental Matrix is below required threshold
def threshold(x1,x2,F):
    x1_new=np.array([x1[0],x1[1],1]).T
    x2_new=np.array([x2[0],x2[1],1])
    return abs(np.squeeze(np.matmul((np.matmul(x2_new,F)),x1_new)))

def findPoints(img_old,img_new):
    old = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

    #sift
    sift = cv2.xfeatures2d.SIFT_create()
    # kp,descriptors = orb.detectAndCompute(old, None)

    keypoints_1, descriptors_1 = sift.detectAndCompute(old,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(new,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    # Find matching points
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)
    

    # Initialize keypoints list
    list_kp1 = [] # old frame
    list_kp2 = [] # new frame
    
    # Ratio test to see which keypoints are the best
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            list_kp1.append(keypoints_1[m.queryIdx].pt)
            list_kp2.append(keypoints_2[m.trainIdx].pt)

    inlier_count = 0
    # Initialize list of inliers for new and old frame
    inlier1 = [] # old frame
    inlier2 = [] # new frame

    # RANSAC Algorithm - 100 iterations
    for i in range(0, 100):
        count = 0
        randomPoints = [] 
        # Random corresponding points from old and new frame
        correspondingPoints1 = [] 
        correspondingPoints2 = [] 
        # Best corresponding matching points 
        bestPoints1 = [] 
        bestPoints2 = []
        
        while(True): # Loop runs while we do not get eight distinct random points
            num = random.randint(0, len(list_kp1)-1)
            if num not in randomPoints:
                randomPoints.append(num)
            if len(randomPoints) == 8:
                break

        for point in randomPoints: # Looping over eight random points
            correspondingPoints1.append([list_kp1[point][0], list_kp1[point][1]])
            correspondingPoints2.append([list_kp2[point][0], list_kp2[point][1]])

        # Computing Fundamentals Matrix from current frame to next frame
        F = estimateF(correspondingPoints1, correspondingPoints2)


        for number in range(0, len(list_kp1)):
            # If x2.T * F * x1 is less than threshold (0.01) then it is considered as Inlier
            if threshold(list_kp1[number], list_kp2[number], F) < 0.01:
                count = count + 1 
                bestPoints1.append(list_kp1[number])
                bestPoints2.append(list_kp2[number])

        # Check to see if this F matrix has the most inliers
        if count > inlier_count: 
            inlier_count = count
            BestF = F
            inlier1 = bestPoints1
            inlier2 = bestPoints2

    return BestF, inlier1, inlier2




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


    
        
# Compute the Fundamental Matrix
# Input - 8 Corresponding points old frame, 8 Corresponding points new frame
# Output - Fundamental Matrix (F), Intrinsic Parameters (K)
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
# Input - Fundamental Matrix (F), Intrinsic Parameters (K)
# Output - Essential Matrix (E)
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

# Compute the Camera Pose
# Input - Essential matrix (E)
# Output - Camera position (x,y,z) and rotation matrix
def estimateC(E):
    # Compute SVD of the Essential Matrix
    u, s, v = np.linalg.svd(E,full_matrices=False)
    w = [[0,-1,0],[1,0,0],[0,0,1]]

    # Four possible Camera Poses (C) and Rotation Matrices (R)
    C1 = u[:,2]
    R1 = np.matmul(np.matmul(u,w),v)
    # Correct pose and matrix if determinant is -1
    if np.linalg.det(R1)<0:
        C1 = -C1

    C2 = -u[:,2]
    R2 = np.matmul(np.matmul(u,w),v)
    if np.linalg.det(R2)<0:
        C2 = -C2
    
    C3 = u[:,2]
    R3 = np.matmul(np.matmul(u,np.transpose(w)),v)
    if np.linalg.det(R3)<0:
        C3 = -C3

    C4 = -u[:,2]
    R4 = np.matmul(np.matmul(u,np.transpose(w)),v)
    if np.linalg.det(R4)<0:
        C4 = -C4

    return C1,C2,C3,C4,R1,R2,R3,R4

def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])


def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    x1=np.asarray(x1)
    I = np.identity(3)
    sz = x1.shape[0]
    C_1 = np.reshape(C1, (3, 1))
    C_2 = np.reshape(C2, (3, 1))
    P1 = np.dot(K, np.dot(R1, np.concatenate((I, -C_1),axis=1)))
    P2 = np.dot(K, np.dot(R2, np.concatenate((I, -C_2),axis=1)))

    X1 = np.concatenate((x1, np.ones((sz, 1))), axis=1)
    X2 = np.concatenate((x2, np.ones((sz, 1))), axis=1)

    X = np.zeros((sz, 3))

    for i in range(sz):
        skew1 = skew(X1[i, :])
        skew2 = skew(X2[i, :])
        A = np.concatenate((np.dot(skew1, P1), np.dot(skew2, P2)))
        _, _, v = np.linalg.svd(A)
        x = v[-1] / v[-1, -1]
        x = np.reshape(x, (len(x), -1))
        X[i, :] = x[0:3].T

    return X

def Cheirality(C, R, X): #TODO
    best = 0
    R_best, C_best = R[0], C[0]
    for i in range(4):
        n = 0
        for j in range(len(X)):
            if np.matmul(R[i][2],np.transpose(X[j]-C[i])) > 0:
                n = n + 1
        if n > best:
            C_best = C[i]
            R_best = R[i]
            best = n

    return R_best, C_best

def NonLinear(K, x1, x2, X_0, C1, R1, C2, R2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    X = np.zeros((x1.shape[0], 3))
    X0 = X_0.flatten()
    #     Tracer()()
    optimized_params = opt.least_squares(
        fun=Function,
        x0=X0,
        args=[K,C1, R1, R2, C2, x1,x2])

    X = np.reshape(optimized_params.x, (x1.shape[0], 3))

    return X


def Function(init, K, C1, R1, R2, C2, inliers1, inliers2):

    X = np.hstack((np.reshape(init, (inliers1.shape[0], 3)), np.ones((inliers1.shape[0], 1))))
    I = np.identity(3)
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))
    P1 = np.dot(K, np.dot(R1, np.concatenate((I, -C1), axis=1)))
    P2 = np.dot(K, np.dot(R2, np.concatenate((I, -C2), axis=1)))

    error1 = 0
    error2 = 0
    error = []

    u1 = np.divide((np.dot(P1[0, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    v1 = np.divide((np.dot(P1[1, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    u2 = np.divide((np.dot(P2[0, :], X.T).T), (np.dot(P2[2, :], X.T).T))
    v2 = np.divide((np.dot(P2[1, :], X.T).T), (np.dot(P2[2, :], X.T).T))

    error1 = ((inliers1[:, 0] - u1) + (inliers1[:, 1] - v1))
    error2 = ((inliers2[:, 0] - u2) + (inliers2[:, 1] - v2))
    #     print(error1.shape)
    error = sum(error1, error2)

    return sum(error)

def loadImages(path = ".png"):
    return [os.path.join(path, ima) for ima in os.listdir(path)]


files=loadImages("Oxford_dataset/stereo/centre")
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
    images=cv2.imread(file, cv2.IMREAD_UNCHANGED)
    #cv2.imshow("video", images)
    
    color_image = cv2.cvtColor(images, cv2.COLOR_BayerGR2BGR)
    # Camera intrinsic parameters
    K = [[fx,0,cx],[0,fy,cy],[0,0,1]]

    if count>0:
        img_old = img_new.copy()

    #img_new=color_image
    img_new = UndistortImage(color_image, LUT)
    img_new = cv2.resize(img_new, (400, 300))


    if count>0:
        # Perform all matrix operations
        F, inliers1, inliers2 = findPoints(img_old,img_new)
        E = estimateE(F,K)
        C1,C2,C3,C4,R1,R2,R3,R4 = estimateC(E)
        X = LinearTriangulation(K, C1, R1, C2, R2, inliers1, inliers2)
        X = NonLinear(K,inliers1,inliers2,X,C1,R1,C2,R2)
        C=np.vstack((C1,C2,C3,C4))
        R = R1,R2,R3,R4
        R,C=Cheirality(C,R,X)
        C = np.matmul(R,C)

        x.append(float(C[0]))
        y.append(float(C[1]))
        z.append(float(C[2]))
        # Make the direction data for the arrows
        u.append(float(np.sin(np.pi * x[-1]) * np.cos(np.pi * y[-1]) * np.cos(np.pi * z[-1])))
        v.append(float(-np.cos(np.pi * x[-1]) * np.sin(np.pi * y[-1]) * np.cos(np.pi * z[-1])))
        w.append(float((np.sqrt(2.0 / 3.0) * np.cos(np.pi * x[-1]) * np.cos(np.pi * y[-1]) * np.sin(np.pi * z[-1]))))


    cv2.imshow("Undistorted Img", img_new)

    count+=1
    print(count)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        np.savez('mat.npz',x,y,z,u,v,w)
        break

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
plt.show()

#https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift-create-not-working-even-though-have-contrib-instal
