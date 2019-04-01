import numpy as np


def similarity_matrix(weights):
    from numpy.linalg import norm
    from matplotlib import pyplot as plt

    weights = np.transpose(weights)

    nclasses = weights.shape[0]
    similarity = np.zeros((nclasses, nclasses))

    for i in range(nclasses):
        for j in range(nclasses):
            span = weights[i] - weights[j]
            dist = norm(span)
            similarity[i][j] = dist
    plt.imshow(similarity, cmap='hot', interpolation='nearest')
    plt.show()