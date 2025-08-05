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

def plot_2d_clusters(X, labels, centers, title="2D Cluster Visualization"):
    plt.figure(figsize=(8, 6))
    n_clusters = len(np.unique(labels))
    colors = plt.get_cmap('tab10', n_clusters)
    import mplcursors  # You may need to install this: pip install mplcursors
    scatter_objs = []
    for j in range(n_clusters):
        cluster_points = X[labels == j]
        sc = plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {j}', color=colors(j), alpha=0.7)
        scatter_objs.append((sc, cluster_points))
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label='Centers')
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    # Add hover tooltips
    for sc, cluster_points in scatter_objs:
        cursor = mplcursors.cursor(sc, hover=True)
        @cursor.connect("add")
        def on_add(sel, points=cluster_points):
            idx = sel.index
            age, income, purchase, freq = points[idx]
            sel.annotation.set(text=f"Age: {age}\nIncome: {income}\nPurchase: {purchase}\nFreq: {freq}")
    plt.show()

def compare_kmeans(X, cluster_range=(2, 5), random_state=42):
    for k in range(cluster_range[0], cluster_range[1] + 1):
        labels, centers = kmeans(X, n_clusters=k, random_state=random_state)
        plot_2d_clusters(X, labels, centers, title=f'KMeans with {k} Clusters (Age vs Income)')

def plot_3d_clusters(X, labels, centers, title="3D Cluster Visualization"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    n_clusters = len(np.unique(labels))
    colors = plt.get_cmap('tab10', n_clusters)
    import mplcursors  # You may need to install this: pip install mplcursors
    scatter_objs = []
    for j in range(n_clusters):
        cluster_points = X[labels == j]
        sc = ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 3], label=f'Cluster {j}', alpha=0.6, color=colors(j))
        scatter_objs.append((sc, cluster_points))
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 3], c='black', marker='x', s=100, label='Centers')
    ax.set_xlabel('Age')
    ax.set_ylabel('Income')
    ax.set_zlabel('Frequency')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    # Add hover tooltips
    for sc, cluster_points in scatter_objs:
        cursor = mplcursors.cursor(sc, hover=True)
        @cursor.connect("add")
        def on_add(sel, points=cluster_points):
            idx = sel.index
            age, income, purchase, freq = points[idx]
            sel.annotation.set(text=f"Age: {age}\nIncome: {income}\nPurchase: {purchase}\nFreq: {freq}")
    plt.show()


def cluster_high_value_young_adults(people_data, age_range=(25, 35), high_income_threshold=100000, high_frequency_threshold=80):
    """
    Filter people with given age range, high income, and frequent purchases, then cluster and visualize them (2D and 3D).
    """
    mask = (
        (people_data[:, 0] >= age_range[0]) & (people_data[:, 0] <= age_range[1]) &
        (people_data[:, 1] >= high_income_threshold) &
        (people_data[:, 3] >= high_frequency_threshold)
    )
    filtered_data = people_data[mask]
    print(f"\nFiltered people (age {age_range[0]}-{age_range[1]}, high income >= {high_income_threshold}, frequency >= {high_frequency_threshold}): {filtered_data.shape[0]} found")
    if filtered_data.shape[0] > 0:
        print(filtered_data[:5])
        # 2D visualization (Age vs Income)
        labels_2d, centers_2d = kmeans(filtered_data, n_clusters=3, random_state=42)
        plot_2d_clusters(filtered_data, labels_2d, centers_2d, title="2D Clusters: Age vs Income")
        # 3D visualization (Age, Income, Frequency)
        plot_3d_clusters(filtered_data, labels_2d, centers_2d, title="3D Clusters: Age, Income, Frequency")
    else:
        print("No people found with the specified criteria.")

if __name__ == "__main__":
    people_data = generate_people_data(num_samples=100000)
    print("Random people data (first few rows):")
    print(people_data[:5])
    # Compare different numbers of clusters (using Age and Income for visualization)
    compare_kmeans(people_data, cluster_range=(2, 5))

    # Cluster and visualize high-value young adults
    print("\nColumns: Age, Income, Purchase History, Frequency of Purchase")
    cluster_high_value_young_adults(people_data, age_range=(25, 35), high_income_threshold=100000, high_frequency_threshold=80)
