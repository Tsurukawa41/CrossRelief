# CrossRelief
In this paper, we introduce CrossRelief, which is an extension of the ReliefF algorithm, but designed for feature selection in multi-cohort settings.

# CrossRelief

CrossRelief is an extension of the ReliefF algorithm designed for **feature selection across multiple datasets**. While ReliefF focuses on nearest-neighbor comparisons within a single dataset, CrossRelief generalizes this approach to operate in multi-cohort settings. This enables identification of features that discriminate between classes across heterogeneous studies, such as those commonly encountered in transcriptomics and biomedical research.

## How it Works

* For each dataset in a collection of datasets, CrossRelief samples *anchor* instances.
* For each anchor, the algorithm identifies nearest neighbors (hits: same class, misses: different class) **across every dataset**.
* Feature weights are updated based on the differences between the anchor and its nearest hits and misses.

## Inputs

CrossRelief requires the following inputs:

* **Xs**: A list of `D` datasets, where each dataset is an array of shape `(n_d, p)` with `n_d` samples and `p` shared features.
* **ys**: A list of `D` label vectors, where each vector has length `n_d` with binary class labels {0,1}.
* **n_neighbors (k)**: The number of nearest hits and misses to consider per dataset.
* **L**: The number of anchor samples to draw per dataset (if `None`, use all samples).
* **metric**: Distance metric (e.g., `euclidean`).
* **normalize**: Boolean flag; if `True`, applies per-dataset min–max scaling to [0,1].
* **random_state**: Seed for reproducibility.

## Output

* **scores**: A one-dimensional NumPy array of length `p`, where each entry corresponds to the feature relevance score for a single feature. Larger values indicate stronger evidence that the feature consistently discriminates between classes across datasets.

## Example Usage (Python)

```python
scores = crossrelief(Xs, ys, n_neighbors=5, L=None, metric="euclidean", normalize=True, random_state=42)
print(scores.shape)  # (p,)
```
