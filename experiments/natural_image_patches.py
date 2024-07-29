""" This file implements the natural image patches experiment with Principal Subspace Analysis.
We first extract a some patches of flower images from the Natural Images database (https://www.kaggle.com/datasets/prasunroy/natural-images).
Then we compute the BIC of PPCA model of type (1, 1, 1, 1, 1, 251) and compare it to a PSA model of type (2, 3, 251).
The PSA model has a lower BIC, therefore we choose it.
Eventually, we perform subspace exploration by sampling uniformly from the 2-sphere and 3-sphere inside the principal subspaces.
We notice the emergence of low-frequency feature subspaces with rotational invariance.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from skimage import color
from skimage import io
from sklearn.feature_extraction.image import extract_patches_2d

from utils import evd, bic


if __name__ == "__main__":
    np.random.seed(42)

    # Generate dataset
    dir_name = "../data/flower_10/"
    patchsize = 8
    n = 500
    X_img = []
    for i, file in enumerate(os.listdir(dir_name)):
        img = color.rgb2gray(io.imread(dir_name+file))
        patches = extract_patches_2d(img, (patchsize, patchsize))
        patches = patches - np.mean(patches, axis=(1, 2))[:, np.newaxis, np.newaxis]  # remove DC component
        X_img.append(patches[np.random.choice(np.arange(patches.shape[0]), size=n//10, replace=False)])
    X = np.concatenate(X_img, axis=0)

    # Plot dataset
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X[n//10 * i], cmap="gray")
        ax.axis('off')
    fig.subplots_adjust(wspace=.1, hspace=.1)
    plt.show()

    # Compute BIC of PPCA and PSA models
    X = X.reshape((n, patchsize*patchsize))
    n, p = X.shape
    eigval, eigvec = evd(X)
    plt.figure()
    plt.bar(np.arange(25), eigval[:25], color='k')
    plt.show()
    bic_psa = bic((2, 3, p - 5), eigval, n)
    bic_ppca = bic((1, 1, 1, 1, 1, p - 5), eigval, n)

    # Plot principal components
    plt.set_cmap('gray')
    fig, axes = plt.subplots(1, 9)
    for j, ax in enumerate(axes):
        ax.imshow((eigvec[:, j]).reshape(patchsize, patchsize))
        ax.axis('off')

    # Generate samples from the first principal subspace (2D)
    fig, axes = plt.subplots(1, 25)
    for ax, theta in zip(axes, np.linspace(0, 2 * np.pi, 25)):
        x, y = np.cos(theta), np.sin(theta)
        ax.imshow((x * eigvec[:, 0] + y * eigvec[:, 1]).reshape(patchsize, patchsize))
        ax.axis('off')

    # Generate samples from the second principal subspace (3D)
    fig, axes = plt.subplots(10, 25)
    for i, phi in enumerate(np.linspace(0, np.pi, 10)):
        for ax, theta in zip(axes[i], np.linspace(0, 2 * np.pi, 25)):
            x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
            ax.imshow((x * eigvec[:, 2] + y * eigvec[:, 3] + z * eigvec[:, 4]).reshape(patchsize, patchsize))
            ax.axis('off')
