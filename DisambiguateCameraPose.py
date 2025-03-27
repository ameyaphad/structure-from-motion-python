import numpy as np
from LinearTriangulation import *


def count_valid_points(R, t, K, points1, points2):
    """Counts valid 3D points that are in front of both cameras."""
    # print("k matrix:",K)
    # print("R matrix:",R)
    # print("t matrix:",t)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera at origin
    # P2 = K @ np.hstack((R, t.reshape(3, 1)))           # Second camera pose
    C = np.reshape(t, (3, 1))
    I = np.identity(3)
    P2 = np.dot(K, np.dot(R, np.hstack((I, -C))))

    valid_count = 0
    for x1, x2 in zip(points1, points2):
        X = linear_triangulation(x1, x2, P1, P2)
        if X[2] > 0 and (R @ X + t)[2] > 0:  # Check depth in both cameras
            valid_count += 1

    return valid_count

def select_best_pose(K, poses,points1,points2):
    """Selects the correct (R, t) by triangulating points and using the cheirality check."""

    max_valid = 0
    best_pose = None

    for R, t in poses:
        valid_count = count_valid_points(R, t, K, points1, points2)
        if valid_count > max_valid:
            max_valid = valid_count
            best_pose = (R, t)

    return best_pose