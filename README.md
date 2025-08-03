# k_means_clustering_hello_world

A Python implementation of K-means clustering algorithm demonstrated on synthetic people data (age, income). This implementation showcases clustering analysis on demographic data, visualizing how people can be segmented based on their age and income characteristics.

## Overview

The K-means clustering algorithm is an unsupervised learning method that partitions data into k distinct clusters. This implementation:
1. Generates synthetic demographic data
2. Applies K-means clustering to identify natural groupings
3. Visualizes the results with interactive plots
4. Provides a comparison of different cluster counts (k=2 to k=5)

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

### Prerequisites

- Python 3.x
- pip (Python package installer)
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```sh
git clone https://github.com/bhuwanchopra/k_means_clustering_hello_world.git
cd k_means_clustering_hello_world
```

2. Create and activate a virtual environment (recommended):
```sh
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate  # On Windows
```

3. Install the required packages:
```sh
pip install -r requirements.txt
```

### Run

Run the main script to see the clustering in action:
```sh
python src/main.py
```

This will:
1. Generate 100,000 random data points
2. Show a comparison of k=2 through k=5 clusters
3. Display a detailed view of the 3-cluster solution
4. Print cluster centers and point distribution

### Test

Run the test suite to verify the implementation:
```sh
# From the root directory of the project:
python -m pytest tests/test_main.py -v
```

If you encounter import errors, try one of these solutions:

1. Using PYTHONPATH:
```sh
PYTHONPATH=$PYTHONPATH:. python -m pytest tests/test_main.py -v  # On Unix/macOS
# OR
set PYTHONPATH=%PYTHONPATH%;. && python -m pytest tests/test_main.py -v  # On Windows
```

2. Or install the package in development mode:
```sh
pip install -e .
```

The test suite covers:
- Data generation functionality
- Euclidean distance calculations
- K-means clustering algorithm
- Cluster assignment and convergence
- Edge cases and reproducibility

## Implementation Details

### Data Generation
- Age range: 18-70 years (uniformly distributed)
- Income range: $20,000-$120,000 (uniformly distributed)
- Default sample size: 100,000 points

### K-means Algorithm
1. Random initialization of k cluster centers
2. Iterative process:
   - Assign each point to nearest center
   - Update centers to mean of assigned points
   - Check for convergence
3. Returns cluster labels and final centers

### Visualization
- Scatter plots with different colors per cluster
- Black X markers for cluster centers
- Age on x-axis, Income on y-axis
- Comprehensive legend and labels

## License

MIT