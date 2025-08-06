
from typing import Tuple
import numpy as np

__all__ = ["euclidean_distance", "kmeans"]

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance between each row of a and b.
    Args:
        a: np.ndarray of shape (n_samples, n_features)
        b: np.ndarray of shape (n_features,) or (n_samples, n_features)
    Returns: np.ndarray of distances
    """
    return np.linalg.norm(a - b, axis=1)

def kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    n_iters: int = 100,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run K-means clustering on X.
    Args:
        X: Data points, shape (n_samples, n_features)
        n_clusters: Number of clusters
        n_iters: Maximum number of iterations
        random_state: Random seed
    Returns:
        labels: Cluster labels for each point
        centers: Final cluster centers
    """
    np.random.seed(random_state)
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[indices]
    for _ in range(n_iters):
        distances = np.array([euclidean_distance(X, center) for center in centers]).T
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(n_clusters)
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels, centers
