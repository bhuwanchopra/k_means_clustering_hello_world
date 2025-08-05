import matplotlib.pyplot as plt
import numpy as np

def plot_2d_clusters(X, labels, centers, title="2D Cluster Visualization"):
    plt.figure(figsize=(8, 6))
    n_clusters = len(np.unique(labels))
    colors = plt.get_cmap('tab10', n_clusters)
    import mplcursors
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
    for sc, cluster_points in scatter_objs:
        cursor = mplcursors.cursor(sc, hover=True)
        @cursor.connect("add")
        def on_add(sel, points=cluster_points):
            idx = sel.index
            age, income, purchase, freq = points[idx]
            sel.annotation.set(text=f"Age: {age}\nIncome: {income}\nPurchase: {purchase}\nFreq: {freq}")
    plt.show()

def plot_3d_clusters_plotly(X, labels, centers, title="3D Cluster Visualization"):
    import plotly.graph_objs as go
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
