

from typing import Any
import numpy as np

__all__ = ["plot_3d_clusters_plotly"]

def plot_3d_clusters_plotly(
    X: np.ndarray,
    labels: Any,
    centers: np.ndarray,
    title: str = "3D Cluster Visualization"
) -> None:
    """
    Plot clusters in 3D using Plotly.

    Args:
        X: Data points, shape (n_samples, n_features)
        labels: Cluster labels for each point
        centers: Cluster centers, shape (n_clusters, n_features)
        title: Plot title
    """
    import plotly.graph_objs as go
    n_clusters = len(set(labels))
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
