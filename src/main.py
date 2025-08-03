import numpy as np
import matplotlib.pyplot as plt

def generate_people_data(num_samples=100, age_range=(18, 70), income_range=(20000, 120000)):
    ages = np.random.randint(age_range[0], age_range[1]+1, num_samples)
    incomes = np.random.randint(income_range[0], income_range[1]+1, num_samples)
    # Purchase history: total amount spent, correlated with income but random
    purchase_history = np.random.randint(100, 50000, num_samples)
    # Frequency of purchase: number of purchases, random but could be correlated with age
    frequency = np.random.randint(1, 100, num_samples)
    data = np.column_stack((ages, incomes, purchase_history, frequency))
    return data

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def kmeans(X, n_clusters=3, n_iters=100, random_state=42):
    np.random.seed(random_state)
    # Randomly choose initial centers
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[indices]
    
    for _ in range(n_iters):
        # Assign clusters
        distances = np.array([euclidean_distance(X, center) for center in centers]).T
        labels = np.argmin(distances, axis=1)
        # Update centers
        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(n_clusters)])
        # Check for convergence
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels, centers

def compare_kmeans(X, cluster_range=(2, 5), random_state=42):
    plt.figure(figsize=(15, 10))
    for i, k in enumerate(range(cluster_range[0], cluster_range[1] + 1), 1):
        plt.subplot(2, 2, i)
        labels, centers = kmeans(X, n_clusters=k, random_state=random_state)
        # For visualization, plot only the first two features (Age, Income)
        for j in range(k):
            cluster_points = X[labels == j]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {j}')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label='Centers')
        plt.xlabel('Age')
        plt.ylabel('Income')
        plt.title(f'KMeans with {k} Clusters (Age vs Income)')
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    people_data = generate_people_data(num_samples=100000)
    print("Random people data (first few rows):")
    print(people_data[:5])
    print("\nColumns: Age, Income, Purchase History, Frequency of Purchase")
    # Compare different numbers of clusters (using Age and Income for visualization)
    compare_kmeans(people_data, cluster_range=(2, 5))
