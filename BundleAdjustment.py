import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from BuildVisibilityMatrix import *



def project_points(X, R, C, K):
    """Projects 3D points into the camera's image plane using intrinsic matrix K."""
    X_cam = R @ (X - C)  # Convert to camera coordinates
    X_proj = K @ X_cam  # Project onto image plane
    return X_proj[:2] / X_proj[2]  # Normalize to get pixel coordinates (u, v)

def reprojection_error(params, num_cameras, num_points, K, visibility, observed_2d):
    """
    Compute the reprojection error considering the visibility matrix.
    params: Flattened array of camera parameters (R, C) and 3D points (X).
    visibility: Binary matrix indicating if a point is visible in a camera.
    observed_2d: Observed 2D points for visible correspondences.
    """
    R_vecs = params[:num_cameras * 3].reshape((num_cameras, 3))  # Rotation vectors
    C_vecs = params[num_cameras * 3:num_cameras * 6].reshape((num_cameras, 3))  # Camera centers
    X_3d = params[num_cameras * 6:].reshape((num_points, 3))  # 3D points
    
    error = []
    for i in range(num_cameras):
        R, _ = cv2.Rodrigues(R_vecs[i])  # Convert rotation vector to matrix
        C = C_vecs[i]

        for j in range(num_points):
            if visibility[i, j]:  # Only compute error for visible points
                u_proj = project_points(X_3d[j], R, C, K)
                u_obs = observed_2d[i, j]  # Observed 2D point
                error.append(u_proj - u_obs)  # Compute error

    return np.array(error).ravel()  # Flatten error array for optimization

def bundle_adjustment(cameras, X_3d, K, visibility, observed_2d):
    """
    Optimizes camera poses (R, C) and 3D points (X_3d) using Bundle Adjustment.
    
    cameras: List of (R, C) for each camera.
    X_3d: Initial 3D points.
    K: Intrinsic matrix.
    visibility: Binary matrix (cameras x points).
    observed_2d: 2D points observed in each camera.
    
    Returns: Optimized cameras and 3D points.
    """
    num_cameras = len(cameras)
    num_points = len(X_3d)

    # Flatten initial parameters (convert R to rotation vector)
    R_vecs = np.array([cv2.Rodrigues(R)[0].flatten() for R, C in cameras])
    C_vecs = np.array([C for R, C in cameras])
    X_flat = X_3d.flatten()
    
    # Stack all parameters into a single array
    params_init = np.hstack((R_vecs.flatten(), C_vecs.flatten(), X_flat))

    # Optimize using least squares
    result = least_squares(
        reprojection_error, params_init,
        args=(num_cameras, num_points, K, visibility, observed_2d),
        method='trf', verbose=2
    )

    # Extract optimized parameters
    optimized_params = result.x
    R_vecs_opt = optimized_params[:num_cameras * 3].reshape((num_cameras, 3))
    C_vecs_opt = optimized_params[num_cameras * 3:num_cameras * 6].reshape((num_cameras, 3))
    X_opt = optimized_params[num_cameras * 6:].reshape((num_points, 3))

    optimized_cameras = [(cv2.Rodrigues(R_vecs_opt[i])[0], C_vecs_opt[i]) for i in range(num_cameras)]
    return optimized_cameras, X_opt