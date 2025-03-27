import numpy as np

def compute_essential_matrix(F, K1, K2):
    """
    Compute the Essential Matrix from the Fundamental Matrix and camera intrinsics.

    Args:
        F (numpy.ndarray): Fundamental Matrix (3x3)
        K1 (numpy.ndarray): Intrinsic matrix of first camera (3x3)
        K2 (numpy.ndarray): Intrinsic matrix of second camera (3x3)

    Returns:
        numpy.ndarray: Essential Matrix (3x3)
    """
    E = K2.T @ F @ K1  # Compute the essential matrix
    U, S, Vt = np.linalg.svd(E)  # Enforce rank-2 constraint
    S = np.diag([1, 1, 0])  # Essential matrix has two equal singular values and one zero
    E = U @ S @ Vt  # Reconstruct the essential matrix with the constraint
    return E