import numpy as np

def linear_triangulation(x1, x2, P1, P2):
    """Triangulates a 3D point using Linear Least Squares."""
    A = np.array([
        x1[0] * P1[2, :] - P1[0, :],
        x1[1] * P1[2, :] - P1[1, :],
        x2[0] * P2[2, :] - P2[0, :],
        x2[1] * P2[2, :] - P2[1, :]
    ])

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]  # Convert to Euclidean coordinates