import numpy as np


def ProjectionMatrix(R, C, K):
    C = np.reshape(C, (3, 1))
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P


def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def PnP(X_set, x_set, K):
    N = X_set.shape[0]

    X_4 = homo(X_set)
    x_3 = homo(x_set)

    # normalize x
    K_inv = np.linalg.inv(K)
    x_n = K_inv.dot(x_3.T).T

    for i in range(N):
        X = X_4[i].reshape((1, 4))
        zeros = np.zeros((1, 4))

        u, v, _ = x_n[i]

        u_cross = np.array([[0, -1, v],
                            [1, 0, -u],
                            [-v, u, 0]])
        X_tilde = np.vstack((np.hstack((X, zeros, zeros)),
                             np.hstack((zeros, X, zeros)),
                             np.hstack((zeros, zeros, X))))
        a = u_cross.dot(X_tilde)

        if i > 0:
            A = np.vstack((A, a))
        else:
            A = a

    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape((3, 4))
    R = P[:, :3]
    U_r, D, V_rT = np.linalg.svd(R)  # to enforce Orthonormality
    R = U_r.dot(V_rT)

    C = P[:, 3]
    C = - np.linalg.inv(R).dot(C)

    if np.linalg.det(R) < 0:
        R = -R
        C = -C

    return R, C