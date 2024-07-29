""" This file implements the Laplacian eigenfunction experiment with Principal Subspace Analysis.
We first generate a dataset similarly as in North et al.'s paper, i.e. linear combinations of Laplacian eigenmodes
with variance being a decreasing function of the Laplacian eigenvalue.
Then we compute the BIC of PPCA model of type (1, 1, 1, 1, 1, 1, 1, 1, 1, 4087) and compare it to a PSA model of type (1, 2, 1, 2, 2, 1, 4087).
The PSA model has a lower BIC, therefore we choose it.
Eventually, we perform factor rotation by projecting the original Laplacian eigenmodes onto the multidimensional principal subspaces.
While principal components degenerate into quasimodes (random mixtures of eigenmodes), principal subspace analysis finds back
components much closer to the original eigenmodes. We also perform subspace exploration by sampling uniformly from the 2-spheres inside
the principal subspaces and get low-frequency feature subspaces.

"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from utils import evd, bic


def generate_laplacian_data(size=500, L=64, order=3):
    """ Generate a dataset consisting in a linear combination of square-domain eigenmodes, with variance being a decreasing function of the Laplacian eigenvalue.
    The formulae for the vibration modes of rectangular surfaces can be found on LibreTexts Mathematics - Vibrations of Rectangular Membranes (courtesy of Russell Herman).
    """
    X_img = np.zeros((size, L, L))
    grid = np.meshgrid(np.arange(L), np.arange(L))
    modes, variances = [], []
    for n in range(1, order + 1):
        for m in range(1, order + 1):
            lambda_nm = (n * np.pi / L) ** 2 + (m * np.pi / L) ** 2  # Laplacian eigenvalue
            u_nm = np.sin(n * np.pi * grid[0] / L) * np.sin(m * np.pi * grid[1] / L)  # Laplacian eigenfunction
            u_nm /= np.linalg.norm(u_nm)
            modes.append(u_nm); variances.append(np.exp(- 50 * lambda_nm) ** 2 + 0.005 ** 2)
            coeff = np.random.normal(0, np.exp(- 50 * lambda_nm), size=size)  # coefficient is drawn from a normal distribution with particular variance
            X_img += coeff[:, None, None] * u_nm
    X_img += np.random.normal(0, 0.005, size=X_img.shape)  # we add Gaussian noise with small variance
    modes = np.array(modes)[np.argsort(variances)[::-1]]
    return X_img, modes


if __name__ == "__main__":
    np.random.seed(42)

    # Generate dataset
    L = 64
    n = 500
    order = 3
    X_img, modes = generate_laplacian_data(size=n, L=L, order=order)

    # Plot samples from dataset
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    plt.set_cmap('coolwarm')
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X_img[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.axis('off')
    fig.subplots_adjust(wspace=.1, hspace=.1)
    plt.show()

    # Compute BIC of PPCA and PSA models
    X = X_img.reshape((X_img.shape[0], -1))
    eigval, eigvec = evd(X)
    plt.figure()
    plt.bar(np.arange(25), eigval[:25], color='k')
    plt.show()
    bic_ppca = bic((1,) * 9 + (L * L - 9,), eigval, n)
    bic_psa = bic((1, 2, 1, 2, 2, 1) + (L * L - 9,), eigval, n)

    # Plot true Laplacian eigenmodes
    fig, axes = plt.subplots(3, 9)
    for j, ax in enumerate(axes[0]):
        ax.imshow((modes[j]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')

    # Plot principal components
    for j, ax in enumerate(axes[1]):
        ax.imshow((eigvec[:, j]).reshape(L, L))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')

    # Plot rotated principal components (via orthogonal projection of true eigenmodes into principal subspaces)
    axes[2, 0].imshow((eigvec[:, 0]).reshape(L, L))
    axes[2, 1].imshow((eigvec[:, 1:3] @ eigvec[:, 1:3].T @ modes[1].flatten()).reshape(L, L))
    axes[2, 2].imshow((eigvec[:, 1:3] @ eigvec[:, 1:3].T @ modes[2].flatten()).reshape(L, L))
    axes[2, 3].imshow((-eigvec[:, 3]).reshape(L, L))
    axes[2, 4].imshow((eigvec[:, 4:6] @ eigvec[:, 4:6].T @ modes[4].flatten()).reshape(L, L))
    axes[2, 5].imshow((eigvec[:, 4:6] @ eigvec[:, 4:6].T @ modes[5].flatten()).reshape(L, L))
    axes[2, 6].imshow((eigvec[:, 6:8] @ eigvec[:, 6:8].T @ modes[6].flatten()).reshape(L, L))
    axes[2, 7].imshow((eigvec[:, 6:8] @ eigvec[:, 6:8].T @ modes[7].flatten()).reshape(L, L))
    axes[2, 8].imshow((eigvec[:, 8]).reshape(L, L))
    for ax in axes[2]:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')

    # Generate samples from the second, fourth and fifth principal subspace (2D)
    fig, axes = plt.subplots(3, 25)
    for ax, theta in zip(axes[0], np.linspace(0, 2 * np.pi, 25)):
        x, y = np.cos(theta), np.sin(theta)
        ax.imshow((x * eigvec[:, 1] + y * eigvec[:, 2]).reshape(L, L))
        ax.axis('off')
    for ax, theta in zip(axes[1], np.linspace(0, 2 * np.pi, 25)):
        x, y = np.cos(theta), np.sin(theta)
        ax.imshow((x * eigvec[:, 4] + y * eigvec[:, 5]).reshape(L, L))
        ax.axis('off')
    for ax, theta in zip(axes[2], np.linspace(0, 2 * np.pi, 25)):
        x, y = np.cos(theta), np.sin(theta)
        ax.imshow((x * eigvec[:, 6] + y * eigvec[:, 7]).reshape(L, L))
        ax.axis('off')
