import matplotlib.pyplot as plt
import numpy as np


def evd(X, plot_scree=False):
    """ Perform the eigenvalue decomposition (EVD) of the sample covariance matrix of X.
    Return sample eigenvalues and eigenvectors with decreasing amplitude.
    """
    n, p = X.shape
    mu = np.mean(X, axis=0)
    S = 1 / n * ((X - mu).T @ (X - mu))
    eigval, eigvec = np.linalg.eigh(S)
    eigval, eigvec = np.flip(eigval, -1), np.flip(eigvec, -1)
    if plot_scree:
        fig = plt.figure()
        plt.bar(np.arange(1, p + 1), eigval, color='k')
        plt.title("Eigenvalue scree plot")
        plt.show()
    return eigval, eigvec


def kappa(model):
    """ Compute the number of free parameters of a PSA model.
    """
    p = np.sum(model)
    kappa_mu = p
    kappa_eigvals = len(model)
    kappa_eigenspaces = int(p * (p - 1) / 2 - np.sum(np.array(model) * (np.array(model) - 1) / 2))
    return kappa_mu + kappa_eigvals + kappa_eigenspaces


def ll(model, eigval, n):
    """ Compute the maximum log-likelihood of a PSA model from the sample eigenvalues.
    eigval must be sorted in decreasing order.
    """
    p = np.sum(model)
    q_list = (0,) + tuple(np.cumsum(model))
    eigval_mle = np.concatenate([[np.mean(eigval[qk:qk_])] * gamma_k for (qk, qk_, gamma_k) in zip(q_list[:-1], q_list[1:], model)])
    return - (n / 2) * (p * np.log(2 * np.pi) + np.sum(np.log(eigval_mle)) + p)


def bic(model, eigval, n):
    """ Compute the Bayesian Information Criterion (BIC) of a PSA model from the sample eigenvalues.
    eigval must be sorted in decreasing order.
    """
    return kappa(model) * np.log(n) - 2 * ll(model, eigval, n)


def aic(model, eigval, n):
    """ Compute the Akaike Information Criterion (AIC) of a PSA model from the sample eigenvalues.
    eigval must be sorted in decreasing order.
    """
    return 2 * kappa(model) - 2 * ll(model, eigval, n)