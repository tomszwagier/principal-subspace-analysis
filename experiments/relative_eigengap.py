""" This file implements relative eigengap thresholds under different model selection criteria: AIC, AICc, BIC and North's rule-of-thumb (NRT-1).
Those are first compared theoretically via curve plots and second practically on classical real datasets from the UCI ML Repository (https://archive.ics.uci.edu/).
We can see that NRT-1 has globally the lowest relative eigengaps, followed by AIC, AICc, NRT-2 and BIC.
Therefore, the BIC more frequently equalizes close-eigenvalues than other methods and has the biggest impact on the PCA methodology.
On real data, all the methods suggest to consider principal subspaces of dimension greater than $1$, including North's rule-of-thumb.
Moreover, the number of eigenvalues to equalize seems to increase from NRT to AIC to BIC, but AICc seems to equalize even more eigenvalues than the BIC due to the low-sample correction.
"""

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rc('font', size=20)
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo

from utils import evd


if __name__ == "__main__":

    ##########
    # CURVES
    ##########
    plt.figure(figsize=(8, 8))
    n_list = np.logspace(1, 6.0, num=100)
    cmap = matplotlib.colormaps["coolwarm"]

    # BIC
    phi_bic = 2 * np.log(n_list) / n_list
    plt.loglog(n_list, 2 * (1 - np.exp(phi_bic) + np.exp(phi_bic / 2) * np.sqrt(np.exp(phi_bic) - 1)), linewidth=3, label="BIC", color=cmap(0/10))

    # North's rule-of-thumb
    plt.loglog(n_list, (2 * np.sqrt(2 / n_list)) / (1 + np.sqrt(2 / n_list)), linewidth=3, label="NRT", color=cmap(10/10))
    plt.loglog(n_list, (4 * np.sqrt(2 / n_list)) / (1 + 2 * np.sqrt(2 / n_list)), linewidth=3, color=cmap(10/10))

    # AIC
    phi_aic = 4 / n_list
    plt.loglog(n_list, 2 * (1 - np.exp(phi_aic) + np.exp(phi_aic / 2) * np.sqrt(np.exp(phi_aic) - 1)), linewidth=3, label="AIC", color=cmap(5/10))

    # AICc
    phi_aicc = lambda p: (4 * n_list - 4) / ((n_list - p * (p + 3) / 2) ** 2 - 1)
    for i, p in enumerate([10, 100, 1000]):
        dom = (n_list > p * (p + 3) / 2 + 1)  #  AICc is defined only for n large enough
        plt.loglog(n_list[dom], 2 * (1 - np.exp(phi_aicc(p)[dom]) + np.exp(phi_aicc(p)[dom] / 2) * np.sqrt(np.exp(phi_aicc(p)[dom]) - 1)), linewidth=3, label="AICc" if p==10 else None, color=cmap(3/10))

    plt.xlabel('Number of samples')
    plt.ylabel('Relative eigengap (%)')
    plt.legend()
    plt.grid(False, which='both')
    plt.show()

    ##########
    # TABLES
    ##########
    # UCI ML repository informations
    dataset_names = np.array(['Iris', 'Wisconsin', 'Wine', 'Spambase', 'Glass', 'Covertype', 'Ionosphere', 'Digits'])
    dataset_ids = np.array([53, 17, 109, 94, 42, 31, 52, 80])
    dataset_sizes = np.array([150, 569, 178, 4601, 214, 581012, 351, 5620])
    dataset_dimensions = np.array([4, 30, 13, 57, 9, 54, 34, 64])
    sorting = np.argsort(dataset_sizes/dataset_dimensions)
    dataset_names = dataset_names[sorting] ; dataset_ids = dataset_ids[sorting] ; dataset_sizes = dataset_sizes[sorting] ; dataset_dimensions = dataset_dimensions[sorting]
    q = 25

    colors = {"AIC": [], "AICc": [], "BIC": [], "North": []}
    for name, id in zip(dataset_names, dataset_ids):

        # Preprocessing
        print(name)
        repo = fetch_ucirepo(id=int(id))
        X = repo.data.features.to_numpy()
        if name == "Digits":
            X = X.reshape(5620, 64)
        if name not in ['Iris', 'Ionosphere', 'Digits']:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        elif name in ['Iris', 'Ionosphere']:
            X = X - np.mean(X, axis=0)

        # Eigendecomposition and computation of relative eigengaps
        n, p = X.shape
        eigval, eigvec = evd(X)
        releigengap = (eigval[:-1] - eigval[1:]) / eigval[:-1]

        # Comparison with relative eigengap thresholds for different criteria
        for criterion in colors.keys():
            if criterion in ["AIC", "AICc", "BIC"]:
                if criterion == "AIC":
                    phi = 4 / n
                elif criterion == "AICc":
                    phi = (4 * n - 4) / ((n - p * (p + 3) / 2) ** 2 - 1)
                else:
                    phi = 2 * np.log(n) / n
                delta = 2 * (1 - np.exp(phi) + np.exp(phi / 2) * np.sqrt(np.exp(phi) - 1))
            elif criterion == "North":
                delta = (2 * np.sqrt(2 / n)) / (1 + np.sqrt(2 / n))
            else:
                raise NotImplementedError
            if criterion != "AICc":
                below_threshold = (releigengap < delta).astype('int')
            else:
                below_threshold = ((releigengap < delta) | (n <= p * (p + 3) / 2 + 1)).astype('int')
            below_threshold = np.append(below_threshold, [.5] * (q - len(releigengap)))[:q]
            colors[criterion].append(plt.colormaps['coolwarm'](below_threshold))

    # Table plot
    for criterion in colors.keys():
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellColours=colors[criterion], rowLabels=dataset_names, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        cells = table.properties()["celld"]
        for i in range(len(dataset_names)):
            cells[i, 0]._loc = 'center'
            cells[i, 0].set_text_props(ha='center')
        plt.title(criterion)
        plt.show()
