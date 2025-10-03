import numpy as np
from sklearn.metrics import pairwise_distances

def _minmax01(X):
    X = X.astype(float, copy=True)
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    rng = np.where(mx > mn, mx - mn, 1.0)
    return (X - mn) / rng

def crossrelief(
    Xs, ys, n_neighbors=1, L=None, metric="euclidean", normalize=True, random_state=0
):
    """
    CrossRelief with unified scoring across datasets.
    Xs: list of (n_d x p) arrays, same p in each dataset
    ys: list of (n_d,) binary arrays {0,1}
    n_neighbors: k hits and k misses per dataset
    L: number of anchor samples per dataset (None => use all)
    Returns: 1D array of length p with CrossRelief scores.
    """
    rng = np.random.default_rng(random_state)
    D = len(Xs)
    p = Xs[0].shape[1]
    assert all(X.shape[1] == p for X in Xs), "All datasets must share feature space"

    # Optional per-dataset 0-1 scaling for stable per-feature diffs
    if normalize:
        Xs_proc = [_minmax01(X) for X in Xs]
    else:
        Xs_proc = Xs

    scores = np.zeros(p, dtype=float)

    # Precompute class masks
    class_masks = []
    for y in ys:
        y = np.asarray(y)
        class_masks.append({0: np.where(y == 0)[0], 1: np.where(y == 1)[0]})

    for d_src, (X_src, y_src) in enumerate(zip(Xs_proc, ys)):
        n_src = X_src.shape[0]
        if L is None or L >= n_src:
            anchors = np.arange(n_src)
        else:
            anchors = rng.choice(n_src, size=L, replace=False)

        for i in anchors:
            xi = X_src[i]
            yi = y_src[i]

            # Accumulate Relief-style diffs across *all* datasets
            for d_tgt, (X_tgt, y_tgt) in enumerate(zip(Xs_proc, ys)):
                # Find k nearest hits/misses in the *target* dataset
                # Compute distances to all samples in target
                dists = np.linalg.norm(X_tgt - xi, axis=1) if metric == "euclidean" \
                        else pairwise_distances(X_tgt, xi.reshape(1,-1), metric=metric).ravel()

                hit_idx = class_masks[d_tgt][yi]
                miss_idx = class_masks[d_tgt][1 - yi]

                # Guard for small class sizes
                k_hit  = min(n_neighbors, len(hit_idx))
                k_miss = min(n_neighbors, len(miss_idx))
                if k_hit == 0 or k_miss == 0:
                    continue

                nn_hits = hit_idx[np.argsort(dists[hit_idx])[:k_hit]]
                nn_miss = miss_idx[np.argsort(dists[miss_idx])[:k_miss]]

                hit_avg  = X_tgt[nn_hits].mean(axis=0)
                miss_avg = X_tgt[nn_miss].mean(axis=0)

                # Relief update: higher when feature separates misses from hits
                # diff = |x - miss| - |x - hit|
                scores += np.abs(xi - miss_avg) - np.abs(xi - hit_avg)

    # Normalize by total anchor counts * datasets
    denom = sum((min(L if L else X.shape[0], X.shape[0])) for X in Xs) * len(Xs)
    if denom > 0:
        scores /= denom

    return scores
