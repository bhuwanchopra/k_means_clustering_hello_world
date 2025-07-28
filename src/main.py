import numpy as np
import matplotlib.pyplot as plt

def generate_people_data(num_samples=100, age_range=(18, 70), income_range=(20000, 120000)):
    ages = np.random.randint(age_range[0], age_range[1]+1, num_samples)
    incomes = np.random.randint(income_range[0], income_range[1]+1, num_samples)
    data = np.column_stack((ages, incomes))
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

def plot_clusters(X, labels, centers):
    n_clusters = centers.shape[0]
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label='Centers')
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('KMeans Clusters')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    people_data = generate_people_data(num_samples=100)
    print("Random people data (age, income):")
    print(people_data)

    # Fit custom KMeans model
    labels, centers = kmeans(people_data, n_clusters=3, random_state=42)

    print("\nKMeans cluster labels:")
    print(labels)
    print("\nKMeans cluster centers (age, income):")
    print(centers)

    plot_clusters(people_data, labels, centers)