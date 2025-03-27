import numpy as np
import cv2


def normalize_points(points):
    """ Normalize points using isotropic scaling. Returns normalized points and transformation matrix. """
    mean = np.mean(points, axis=0)
    std = np.std(points)

    scale = np.sqrt(2) / std
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])

    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_norm = (T @ points_h.T).T

    return points_norm[:, :2], T

def compute_fundamental_matrix(pts1, pts2):
    """ Computes the Fundamental Matrix using the eight-point algorithm. """
    # Normalize points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Construct matrix A
    A = np.array([
        [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]
        for (x1, y1), (x2, y2) in zip(pts1_norm, pts2_norm)
    ])
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)  # Last row of Vt reshaped to 3x3 matrix

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Force the smallest singular value to zero
    F_rank2 = U @ np.diag(S) @ Vt

    # Denormalize F
    F_final = T2.T @ F_rank2 @ T1

    # F_final = F_rank2

    return F_final