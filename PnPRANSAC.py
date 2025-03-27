import numpy as np
import cv2
import random
from LinearPnP import *

def ransac_pnp(X, x, K, num_iterations=1000, reprojection_threshold=2.0):
    """
    Perform RANSAC to find the best camera pose using Linear PnP.

    Parameters:
        X: (N, 3) array of 3D points.
        x: (N, 2) array of corresponding 2D points.
        K: (3, 3) camera intrinsic matrix.
        num_iterations: Number of RANSAC iterations.
        reprojection_threshold: Maximum allowable reprojection error for an inlier.

    Returns:
        best_C: Best camera center.
        best_R: Best rotation matrix.
        best_inliers: Indices of inliers for the best model.
    """
    best_inliers = []
    best_C, best_R = None, None

    N = X.shape[0]  # Number of correspondences
    n = 0  # Best inlier count

    for _ in range(num_iterations):
        # Randomly sample 6 correspondences
        sample_indices = random.sample(range(N), 6)
        X_sampled = X[sample_indices]
        x_sampled = x[sample_indices]

        # Compute camera pose using Linear PnP
        R,C = PnP(X_sampled, x_sampled, K)

        # Compute projection matrix P = K[R|t]
        t = -R @ C.reshape(-1, 1)
        P = K @ np.hstack((R, t))

        inliers = []

        for j in range(N):
            X_h = np.append(X[j], 1)  # Convert to homogeneous coordinates
            projected_x_h = P @ X_h  # Project 3D point
            projected_x = projected_x_h[:2] / projected_x_h[2]  # Normalize to get (u, v)

            # Compute reprojection error
            error = np.linalg.norm(x[j] - projected_x)

            if error < reprojection_threshold:
                inliers.append(j)

        # Update best model if more inliers are found
        if len(inliers) > n:
            n = len(inliers)
            best_inliers = inliers
            best_C, best_R = C, R

    return best_C, best_R, best_inliers