import numpy as np


def decompose_essential_matrix(E):
    """
    Decomposes the Essential Matrix into four possible camera poses (R, t).

    Args:
        E (numpy.ndarray): Essential Matrix (3x3)

    Returns:
        list: Four possible (R, t) pairs
    """
    # Perform SVD on E
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotation matrices
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Define the W matrix
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Extract translation (third column of U)
    t = U[:, 2]

    # Four possible solutions (R, t)
    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    return solutions