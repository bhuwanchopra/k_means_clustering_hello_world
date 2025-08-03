import numpy as np
import pytest
from src import main

def test_generate_people_data_shape():
    data = main.generate_people_data(num_samples=50)
    assert data.shape == (50, 4)
    assert data.dtype == np.int_
    
def test_generate_people_data_default_ranges():
    data = main.generate_people_data(num_samples=1000)
    # Check age range (18-70)
    assert np.all((data[:, 0] >= 18) & (data[:, 0] <= 70))
    # Check income range (20000-120000)
    assert np.all((data[:, 1] >= 20000) & (data[:, 1] <= 120000))
    # Check purchase history range (100-50000)
    assert np.all((data[:, 2] >= 100) & (data[:, 2] < 50000))
    # Check frequency range (1-100)
    assert np.all((data[:, 3] >= 1) & (data[:, 3] < 100))

def test_generate_people_data_ranges():
    data = main.generate_people_data(num_samples=100, age_range=(20, 30), income_range=(50000, 60000))
    assert np.all((data[:, 0] >= 20) & (data[:, 0] <= 30))
    assert np.all((data[:, 1] >= 50000) & (data[:, 1] <= 60000))
    assert np.all((data[:, 2] >= 100) & (data[:, 2] < 50000))
    assert np.all((data[:, 3] >= 1) & (data[:, 3] < 100))

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

def test_kmeans_reproducibility():
    """Test that the same random_state produces the same results"""
    X = np.array([[1, 1], [2, 1], [1, 2], [8, 8], [9, 8], [8, 9]])
    labels1, centers1 = main.kmeans(X, n_clusters=2, random_state=42)
    labels2, centers2 = main.kmeans(X, n_clusters=2, random_state=42)
    assert np.array_equal(labels1, labels2)
    assert np.array_equal(centers1, centers2)

def test_kmeans_cluster_sizes():
    """Test that clusters are assigned reasonably for well-separated data"""
    # Create three well-separated clusters
    np.random.seed(42)  # Set seed for reproducibility
    cluster1 = np.random.normal(0, 0.1, (100, 2))
    cluster2 = np.random.normal(5, 0.1, (100, 2))
    cluster3 = np.random.normal(10, 0.1, (100, 2))
    X = np.vstack([cluster1, cluster2, cluster3])
    
    labels, _ = main.kmeans(X, n_clusters=3, random_state=42)
    unique, counts = np.unique(labels, return_counts=True)
    # Check that we have exactly 3 clusters
    assert len(counts) == 3
    # Check total number of points
    assert sum(counts) == 300
    # Check that no cluster is empty or has all points
    assert all(0 < count < 300 for count in counts)

def test_euclidean_distance_edge_cases():
    """Test edge cases for euclidean distance calculation"""
    # Test with zero distance
    a = np.array([[0, 0], [1, 1]])
    b = np.array([0, 0])
    dists = main.euclidean_distance(a, b)
    assert dists[0] == 0
    assert np.allclose(dists[1], np.sqrt(2))
    
    # Test with negative values
    a = np.array([[-1, -1], [-2, -2]])
    b = np.array([0, 0])
    dists = main.euclidean_distance(a, b)
    assert np.allclose(dists, [np.sqrt(2), np.sqrt(8)])
