import numpy as np


def getVisibilityMatrix(X_final_merged, X_optimized, indices_impt,present_indices):
    V = np.zeros((3, X_final_merged.shape[0]))

    print(V.shape)

    for i in range(len(X_final_merged[1])):
        V[0,i] = 1
    for i in range(X_optimized.shape[0]):
        V[1,i] = 1

    # V[1,:] = np.ones(X_optimized.shape[0])
    for i in range(len(indices_impt)):
        V[1,X_optimized.shape[0]+indices_impt[i]] = 1
        
    for i in range(len(X_optimized)):
        if i in present_indices:
            V[2,i] = 1

    return V