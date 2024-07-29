""" This file implements the structured data experiment with Principal Subspace Analysis.
We first load and preprocess the Glass dataset of UCI ML repository https://archive.ics.uci.edu/dataset/42/glass+identification.
Then we compute the BIC of PPCA model of type (1, 1, 1, 1, 1, 4) and compare it to a PSA model of type (5, 4).
The PSA model has a lower BIC, therefore we choose it. Eventually, we perform varimax rotation in the first 5D principal subspace.
The rotated factors are much sparser and therefore much more interpretable.
Also, as a bonus, we run model selection by minimizing the BIC over the whole set of models and within the hierarchical clustering strategy.
Both yield the same result: the best model is (2, 3, 2, 1, 1).
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from sklearn.decomposition._factor_analysis import _ortho_rotation
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt


from utils import bic, evd


if __name__ == '__main__':
    np.random.seed(42)

    # Load and preprocess dataset
    glass = fetch_ucirepo(id=42)
    X = glass.data.features.to_numpy()
    n, p = X.shape
    mu = np.mean(X, axis=0)
    y = glass.data.targets
    X = (X - mu) / np.std(X, axis=0)

    # Compute BIC of PPCA and PSA models
    eigval, eigvec = evd(X)
    bic_ppca = bic((1, 1, 1, 1, 1, 4), eigval, n)
    bic_psa = bic((5, 4), eigval, n)
    model = (5, 4)

    # Rotate factors within first 5D principal subspace
    rotated_factors = _ortho_rotation(eigvec[:, :5], method="varimax", tol=1e-6, max_iter=100).T

    # Plot the results, with categorical colors like in [Jolliffe2002, Section 4.1]
    eigvec_interp = np.sign(eigvec) * (np.abs(eigvec) > .5 * np.max(np.abs(eigvec), axis=0)[None, :]).astype('int') + \
        .25 * np.sign(eigvec) * ((.5 * np.max(np.abs(eigvec), axis=0)[None, :] >= np.abs(eigvec)) & (np.abs(eigvec) >= .25 * np.max(np.abs(eigvec), axis=0)[None, :])).astype('int')
    rotated_factors_interp = np.sign(rotated_factors) * (np.abs(rotated_factors) > .5 * np.max(np.abs(rotated_factors), axis=0)[None, :]).astype('int') + \
        .25 * np.sign(rotated_factors) * ((.5 * np.max(np.abs(rotated_factors), axis=0)[None, :] >= np.abs(rotated_factors)) & (np.abs(rotated_factors) >= .25 * np.max(np.abs(rotated_factors), axis=0)[None, :])).astype('int')

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    rowLabels = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
    table = ax.table(cellText=np.round(np.concatenate([eigvec[:, :5], rotated_factors], axis=1)*100)/100, cellColours=plt.colormaps['coolwarm']((np.concatenate([eigvec_interp[:, :5], rotated_factors_interp], axis=1) + 1) / 2),
    rowLabels=rowLabels, colLabels=['E1', 'E2', 'E3', 'E4', 'E5', 'R1', 'R2', 'R3', 'R4', 'R5'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    cells = table.properties()["celld"]
    for i in range(len(rowLabels)):
        cells[i, 0]._loc = 'center'
        cells[i, 0].set_text_props(ha='center')
    plt.show()

    # BONUS: full model selection
    # from model_selection import candidate_models, candidate_models_hierarchical, model_selection
    # models = candidate_models(p, family="PSA")
    # models_ = candidate_models_hierarchical(eigval, distance="relative", linkage="single")
    # model_best, bic_best = model_selection(X, models, criterion="bic")
    # model_best_, bic_best_ = model_selection(X, models_, criterion="bic")
