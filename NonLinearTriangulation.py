import numpy as np
import cv2
import scipy.optimize as opt

def project(P, X):
    """Projects a 3D point X into 2D using projection matrix P."""
    X_h = np.append(X, 1)  # Convert to homogeneous coordinates
    x_proj = P @ X_h
    return x_proj[:2] / x_proj[2]  # Convert back to 2D (normalize by z)

def reprojection_error(X, P1, P2, x1, x2):
    """Computes the reprojection error for a single 3D point."""
    X = X[:3]  # Extract 3D coordinates
    x1_proj = project(P1, X)
    x2_proj = project(P2, X)

    # Compute error between observed and projected 2D points
    error1 = x1_proj - x1
    error2 = x2_proj - x2
    return np.hstack([error1, error2])

def non_linear_triangulation(X_init, P1, P2, x1, x2):
    """Optimizes the 3D point using non-linear least squares."""
    result = opt.least_squares(reprojection_error, X_init, method='lm', args=(P1, P2, x1, x2))
    return result.x[:3]  # Return optimized 3D coordinates