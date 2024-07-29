import itertools
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from utils import aic, bic, evd


def candidate_models(p, family="PSA"):
    """ Generate a family of probabilistic models, among Probabilistic PCA (PPCA), Isotropic PPCA (IPPCA) and Principal Subspace Analysis (PSA).
    """
    if family == "PPCA":
        models = [(1,) * q + (p - q,) for q in range(p)]
    elif family == "IPPCA":
        models = [(q, p - q) for q in range(1, p)]
    elif family == "PSA":
        models = []
        for d in range(1, p + 1):  # number of distinct eigenvalues
            l_max = p - d + 1
            models_ = itertools.product(*[np.arange(1, l_max + 1).tolist() for _ in range(d)])
            for model in models_:
                if np.sum(model) == p:
                    models.append(tuple(model))
    else:
        raise(NotImplementedError(f"The family {family} has not been implemented yet."))
    return models


def candidate_models_hierarchical(eigval, distance="relative", linkage="single"):
    """ Perform a hierarchical clustering of the sample eigenvalues for model selection.
    """
    p = len(eigval)
    updiag = np.array([[j == (i + 1) for j in range(p)] for i in range(p)]).astype('int')
    if distance == "absolute":
        metric = "euclidean"
        Z = eigval[:, None]
    elif distance == "relative":
        metric = "precomputed"
        Z = np.zeros((p, p))
        for j in range(p-1):
            Z[j, j+1] = (eigval[j] - eigval[j+1]) / eigval[j]
        Z = Z + Z.T
    else:
        raise NotImplementedError()
    clustering = AgglomerativeClustering(connectivity=updiag+updiag.T, metric=metric, linkage=linkage).fit(Z)
    merges = clustering.children_
    models = [[1,] * p]
    nodes_locations = [i for i in range(p)]
    for j, merge in enumerate(merges):
        model = models[-1].copy()
        k = min(nodes_locations.index(merge[0]), nodes_locations.index(merge[1]))
        model[k] += model[k+1]
        del model[k+1]
        models.append(model)
        nodes_locations[k] = p + j
        del nodes_locations[k+1]
    return models


def model_selection(X, candidate_models, criterion="bic"):
    """ Perform model selection by minimizing a criterion (AIC or BIC) among a family of candidate models.
    eigval must be sorted in decreasing order.
    """
    n, p = X.shape
    eigval, _ = evd(X)
    model_best = None; crit_best = np.inf
    for model in candidate_models:
        if criterion == "bic":
            crit_model = bic(model, eigval, n)
        elif criterion == "aic":
            crit_model = aic(model, eigval, n)
        else:
            raise NotImplementedError
        if crit_model < crit_best:
            model_best = model
            crit_best = crit_model
    return model_best, crit_best

