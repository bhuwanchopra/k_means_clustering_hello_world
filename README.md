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


## Usage

Run the main script with CLI arguments to control what is plotted:

```sh
# 2D cluster plot (default)
python src/main.py --plot 2d

# 3D interactive cluster plot (Plotly)
python src/main.py --plot 3d

# Compare clusters for k=2 to k=5
python src/main.py --plot compare

# Filter and plot only high-value young adults (customize thresholds)
python src/main.py --plot filter --age_min 25 --age_max 35 --income_min 120000 --freq_min 90

# Set number of clusters (for 2d/3d)
python src/main.py --plot 2d --n_clusters 4
```

### CLI Options

- `--plot` (`2d`, `3d`, `compare`, `filter`): What to plot (default: `2d`)
- `--num_samples`: Number of people to generate (default: 100000)
- `--n_clusters`: Number of clusters for k-means (default: 3)
- `--age_min`, `--age_max`, `--income_min`, `--freq_min`: Filtering thresholds (used with `--plot filter`)

All plots are interactive. 2D plots use Matplotlib (with hover tooltips), 3D uses Plotly for advanced interactivity.

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