import numpy as np
import pytest
from src import main

def test_generate_people_data_shape():
    data = main.generate_people_data(num_samples=50)
    assert data.shape == (50, 2)
    assert data.dtype == np.int_

def test_generate_people_data_ranges():
    data = main.generate_people_data(num_samples=100, age_range=(20, 30), income_range=(50000, 60000))
    assert np.all((data[:, 0] >= 20) & (data[:, 0] <= 30))
    assert np.all((data[:, 1] >= 50000) & (data[:, 1] <= 60000))

def test_euclidean_distance():
    a = np.array([[0, 0], [3, 4]])
    b = np.array([0, 0])
    dists = main.euclidean_distance(a, b)
    assert np.allclose(dists, [0, 5])

def test_kmeans_basic():
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    labels, centers = main.kmeans(X, n_clusters=2, n_iters=100, random_state=0)
    assert set(labels) == {0, 1}
    assert centers.shape == (2, 2)
    # Each cluster should have 3 points
    unique, counts = np.unique(labels, return_counts=True)
    assert all(count == 3 for count in counts)

def test_kmeans_convergence():
    X = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]])
    labels, centers = main.kmeans(X, n_clusters=2, n_iters=100, random_state=42)
    # Should converge to two clusters: near (0,0) and near (10,10)
    assert np.allclose(sorted(centers[:, 0]), [0.33333333, 10.33333333], atol=1e-1)

