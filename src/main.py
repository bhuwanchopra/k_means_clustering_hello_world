
import numpy as np
import matplotlib.pyplot as plt

def main():
    people_data = generate_people_data(num_samples=100000)
    print("Random people data (first few rows):")
    print(people_data[:5])
    print("\nColumns: Age, Income, Purchase History, Frequency of Purchase")
    # Compare different numbers of clusters (using Age and Income for visualization)
    run_compare_kmeans(people_data, cluster_range=(2, 5))
    # Cluster and visualize high-value young adults
    cluster_high_value_young_adults(people_data, age_range=(25, 35), high_income_threshold=100000, high_frequency_threshold=80)

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

def run_compare_kmeans(X, cluster_range=(2, 5), random_state=42):
    """
    Run k-means for a range of cluster counts and plot 2D clusters for each.
    """
    for k in range(cluster_range[0], cluster_range[1] + 1):
        labels, centers = kmeans(X, n_clusters=k, random_state=random_state)
        plot_2d_clusters(X, labels, centers, title=f'KMeans with {k} Clusters (Age vs Income)')

def plot_3d_clusters_plotly(X, labels, centers, title="3D Cluster Visualization"):
    import plotly.graph_objs as go
    import numpy as np
    n_clusters = len(np.unique(labels))
    traces = []
    for j in range(n_clusters):
        cluster_points = X[labels == j]
        traces.append(go.Scatter3d(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            z=cluster_points[:, 3],
            mode='markers',
            marker=dict(size=4),
            name=f'Cluster {j}',
            text=[f"Age: {age}<br>Income: {income}<br>Purchase: {purchase}<br>Freq: {freq}"
                  for age, income, purchase, freq in cluster_points],
            hoverinfo='text',
        ))
    # Add cluster centers
    traces.append(go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 3],
        mode='markers',
        marker=dict(size=10, color='black', symbol='x'),
        name='Centers',
        text=['Center']*len(centers),
        hoverinfo='text',
    ))
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='Age',
            yaxis_title='Income',
            zaxis_title='Frequency',
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def filter_people(people_data, age_range=(25, 35), high_income_threshold=100000, high_frequency_threshold=80):
    """
    Filter people with given age range, high income, and frequent purchases.
    """
    mask = (
        (people_data[:, 0] >= age_range[0]) & (people_data[:, 0] <= age_range[1]) &
        (people_data[:, 1] >= high_income_threshold) &
        (people_data[:, 3] >= high_frequency_threshold)
    )
    return people_data[mask]

def cluster_and_plot_people(X, n_clusters=3, random_state=42, plot3d=True):
    """
    Cluster people and plot both 2D and 3D clusters.
    """
    labels, centers = kmeans(X, n_clusters=n_clusters, random_state=random_state)
    plot_2d_clusters(X, labels, centers, title="2D Clusters: Age vs Income")
    if plot3d:
        plot_3d_clusters_plotly(X, labels, centers, title="3D Clusters: Age, Income, Frequency (Plotly)")

def cluster_high_value_young_adults(people_data, age_range=(25, 35), high_income_threshold=100000, high_frequency_threshold=80):
    """
    Filter people with given age range, high income, and frequent purchases, then cluster and visualize them (2D and 3D).
    """
    filtered_data = filter_people(people_data, age_range, high_income_threshold, high_frequency_threshold)
    print(f"\nFiltered people (age {age_range[0]}-{age_range[1]}, high income >= {high_income_threshold}, frequency >= {high_frequency_threshold}): {filtered_data.shape[0]} found")
    if filtered_data.shape[0] > 0:
        print(filtered_data[:5])
        cluster_and_plot_people(filtered_data, n_clusters=3, random_state=42, plot3d=True)
    else:
        print("No people found with the specified criteria.")

if __name__ == "__main__":
    main()
