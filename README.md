# k_means_clustering_hello_world

A Python implementation of K-means clustering algorithm demonstrated on synthetic people data (age, income).

## Structure

```
k_means_clustering_hello_world/
├── src/
│   └── main.py
├── tests/
│   └── test_main.py
├── .gitignore
└── README.md
```

## Getting Started

### Setup

You need Python 3.x and pip. Install the required packages:

```sh
pip install numpy matplotlib pytest
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