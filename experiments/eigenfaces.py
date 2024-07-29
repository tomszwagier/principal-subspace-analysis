""" This file implements the eigenface experiment with Principal Subspace Analysis.
We first extract a dataset from the CMU Face images database (https://archive.ics.uci.edu/dataset/124/cmu+face+images).
Then we compute the BIC of PPCA model of type (1, 1, 1, 1, 1, 1, 1, 1, 1, 3831) and compare it to a PSA model of type (1, 3, 5, 3831).
The PSA model has a lower BIC, therefore we choose it. Eventually, we perform subspace exploration by sampling from
the second 3D principal subspace via the PSA generative model, and plot a few samples to gain intuition about the principal subspace.
While eigenfaces are fuzzy being linear mixtures of images, principal subspace analysis enables to extract much more interpretable
eigenfaces within the 3D principal subspace, corresponding to head rotation movements.
We use image entropy to help extracting meaningful eigenfaces within the samples.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from skimage.io import imread
from skimage.filters.rank import entropy
from skimage.morphology import disk

from utils import evd, bic


if __name__ == "__main__":
    np.random.seed(42)

    # Load dataset
    dir_name = "../data/cmu_choon/"
    X_img = []
    for filename in os.listdir(dir_name):
        image = imread(dir_name+filename)
        X_img.append(image)
    X = np.array(X_img)
    X = X.astype('float') / 255

    # Plot dataset
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    plt.set_cmap('gray')
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X_img[3 * i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.set_aspect('equal')
        ax.axis('off')
    fig.subplots_adjust(wspace=.1, hspace=.1)
    plt.show()

    # Compute BIC of PPCA and PSA models
    X = X.reshape((X.shape[0], -1))
    n, p = X.shape
    mu = np.mean(X, axis=0)
    eigval, eigvec = evd(X)
    bic_ppca = bic((1, 1, 1, 1, 1, 1, 1, 1, 1) + (p - 9,), eigval, n)
    bic_psa = bic((1, 3, 5) + (p - 9,), eigval, n)
    fig, axes = plt.subplots(1, 3)
    for j in range(1, 4):
        axes[j-1].imshow((mu + eigval[j] * eigvec[:, j]).reshape(60, 64))

    # Generate samples from the second principal subspace (3D)
    subspace_samples = np.random.multivariate_normal(mean=mu, cov=np.mean(eigval[1:4]) * eigvec[:, 1:4] @ eigvec[:, 1:4].T, size=100)
    entropies = [np.mean(entropy((2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1).reshape(60, 64), disk(10))) for x in subspace_samples]
    argsrt = np.argsort(entropies)
    subspace_samples = np.array(subspace_samples)[argsrt] ; samples_entropies = np.array(entropies)[argsrt]
    fig, axes = plt.subplots(10, 10)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(subspace_samples[i].reshape(60, 64))
        ax.set_title(f"{samples_entropies[i]:.2f}", fontsize=8)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
    plt.show()
