import argparse
from data import generate_people_data, filter_people
from clustering import kmeans
from plotting import plot_2d_clusters, plot_3d_clusters_plotly

def run_compare_kmeans(X, cluster_range=(2, 5), random_state=42):
    for k in range(cluster_range[0], cluster_range[1] + 1):
        labels, centers = kmeans(X, n_clusters=k, random_state=random_state)
        plot_2d_clusters(X, labels, centers, title=f'KMeans with {k} Clusters (Age vs Income)')

def cluster_and_plot_people(X, n_clusters=3, random_state=42, plot3d=True):
    labels, centers = kmeans(X, n_clusters=n_clusters, random_state=random_state)
    plot_2d_clusters(X, labels, centers, title="2D Clusters: Age vs Income")
    if plot3d:
        plot_3d_clusters_plotly(X, labels, centers, title="3D Clusters: Age, Income, Frequency (Plotly)")

def cluster_high_value_young_adults(people_data, age_range=(25, 35), high_income_threshold=100000, high_frequency_threshold=80):
    filtered_data = filter_people(people_data, age_range, high_income_threshold, high_frequency_threshold)
    print(f"\nFiltered people (age {age_range[0]}-{age_range[1]}, high income >= {high_income_threshold}, frequency >= {high_frequency_threshold}): {filtered_data.shape[0]} found")
    if filtered_data.shape[0] > 0:
        print(filtered_data[:5])
        cluster_and_plot_people(filtered_data, n_clusters=3, random_state=42, plot3d=True)
    else:
        print("No people found with the specified criteria.")

def main():
    parser = argparse.ArgumentParser(description="K-means clustering and visualization for synthetic people data.")
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of people to generate')
    parser.add_argument('--plot', type=str, default='3d', choices=['2d', '3d', 'compare', 'filter'], help='Type of plot to show: 2d, 3d, compare, filter')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters for k-means')
    parser.add_argument('--age_min', type=int, default=25, help='Min age for filtering (used with --plot filter)')
    parser.add_argument('--age_max', type=int, default=35, help='Max age for filtering (used with --plot filter)')
    parser.add_argument('--income_min', type=int, default=100000, help='Min income for filtering (used with --plot filter)')
    parser.add_argument('--freq_min', type=int, default=80, help='Min frequency for filtering (used with --plot filter)')
    args = parser.parse_args()

    people_data = generate_people_data(num_samples=args.num_samples)
    print("Random people data (first few rows):")
    print(people_data[:5])
    print("\nColumns: Age, Income, Purchase History, Frequency of Purchase")

    if args.plot == 'compare':
        run_compare_kmeans(people_data, cluster_range=(2, 5))
    elif args.plot == 'filter':
        cluster_high_value_young_adults(
            people_data,
            age_range=(args.age_min, args.age_max),
            high_income_threshold=args.income_min,
            high_frequency_threshold=args.freq_min
        )
    else:
        labels, centers = kmeans(people_data, n_clusters=args.n_clusters, random_state=42)
        if args.plot == '2d':
            plot_2d_clusters(people_data, labels, centers, title=f"2D Clusters: Age vs Income (k={args.n_clusters})")
        elif args.plot == '3d':
            plot_3d_clusters_plotly(people_data, labels, centers, title=f"3D Clusters: Age, Income, Frequency (k={args.n_clusters})")
