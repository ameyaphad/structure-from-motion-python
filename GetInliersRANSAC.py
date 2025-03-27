import cv2
import numpy as np
from EstimateFundamentalMatrix import *
from os import replace
import random


def compute_epipolar_distance(F, pts1, pts2):
    """ Computes the epipolar constraint |x_2^T F x_1| for all points. """
    ones = np.ones((pts1.shape[0], 1))
    pts1_h = np.hstack([pts1, ones])  # Convert to homogeneous coordinates
    pts2_h = np.hstack([pts2, ones])

    # print(pts1_h.shape)
    # print(pts2_h.shape)

    # arr = pts2_h @ (F @ pts1_h.T)
    # print(arr.shape)
    # distances = np.abs(np.diag(pts2_h.T @ F @ pts1_h))
    distances = np.abs(np.diag(pts1_h @ F @ pts2_h.T))
    # distances = np.abs(np.sum(pts1_h @ F @ pts2_h.T)
    return distances

def ransac_fundamental_matrix(pts1, pts2, num_iterations=1000, threshold=0.5):
    """ Estimates the Fundamental Matrix using RANSAC to remove outliers. """
    best_F = None
    max_inliers = 0
    best_inlier_indices = []

    for _ in range(num_iterations):
        # Randomly select 8 correspondences
        sample_indices = random.sample(range(len(pts1)), 8)
        # print(sample_indices)
        sample_pts1 = pts1[sample_indices]
        sample_pts2 = pts2[sample_indices]

        # Compute F using the 8-point algorithm
        F = compute_fundamental_matrix(sample_pts1, sample_pts2)  # Use earlier function

        # Compute inliers using the epipolar constraint
        distances = compute_epipolar_distance(F, pts1, pts2)
        # print(distances)
        inlier_indices = np.where(distances < threshold)[0]

        # Update best model if more inliers are found
        if len(inlier_indices) > max_inliers:
            max_inliers = len(inlier_indices)
            best_F = F
            best_inlier_indices = inlier_indices

    # Recompute F using all inliers
    # print(best_inlier_indices)
    if best_inlier_indices is not None:
        # print(best_inlier_indices)
        pts1_inliers = pts1[best_inlier_indices]
        pts2_inliers = pts2[best_inlier_indices]
        best_F = compute_fundamental_matrix(pts1_inliers,pts2_inliers)

    return best_F, best_inlier_indices

