# k_means_clustering_hello_world

A Python implementation of K-means clustering algorithm demonstrated on synthetic people data (age, income). This implementation showcases clustering analysis on demographic data, visualizing how people can be segmented based on their age and income characteristics.

## Features

- **Data Generation**: Creates synthetic demographic data with customizable parameters
  - Age range: 18-70 years
  - Income range: $20,000-$120,000
  - Configurable sample size (default: 100,000 samples)

- **Clustering Analysis**: 
  - Implementation of K-means clustering from scratch
  - Configurable number of clusters (k=2 to k=5 comparison)
  - Euclidean distance-based clustering
  - Iterative convergence with centroid updates

- **Visualization**:
  - Multi-plot comparison of different k values
  - Scatter plots with distinct colors for each cluster
  - Cluster centers marked with black X markers
  - Clear labeling of axes and clusters

## Structure

```
k_means_clustering_hello_world/
├── src/
│   └── main.py
├── tests/
│   └── test_main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### Setup

You need Python 3.x and pip. Install the required packages:

```sh
pip install -r requirements.txt
```

### Run

```sh
python src/main.py
```

### Test

```sh
pytest tests/
```

## What it does

- Generates random people data (age, income)
- Applies K-means clustering to segment people into groups
- Visualizes the clusters using matplotlib
- Prints cluster centers and assignments

## License

MIT