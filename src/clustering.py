import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def kmeans(X, n_clusters=3, n_iters=100, random_state=42):
    np.random.seed(random_state)
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[indices]
    for _ in range(n_iters):
        distances = np.array([euclidean_distance(X, center) for center in centers]).T
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(n_clusters)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels, centers
