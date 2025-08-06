
from typing import Any
import argparse
from data import generate_people_data

__all__ = ["main"]

def _plot_customer_segments_3d(people_data: Any) -> None:
    """
    Visualize three customer segments in a 3D plot: Young Professionals, Budget Conscious, Loyal Customers.
    """
    import plotly.graph_objs as go
    # Define segment masks
    mask_young_prof = (
        (people_data[:, 0] >= 25) & (people_data[:, 0] <= 35) &
        (people_data[:, 1] > 90000) & (people_data[:, 3] > 50)
    )
    mask_budget = (
        (people_data[:, 1] < 40000) & (people_data[:, 3] > 30)
    )
    mask_loyal = (
        (people_data[:, 3] > 80) & (people_data[:, 2] > 20000)
    )
    segments = [
        (mask_young_prof, 'Young Professionals', 'blue'),
        (mask_budget, 'Budget Conscious', 'green'),
        (mask_loyal, 'Loyal Customers', 'red'),
    ]
    traces = []
    for mask, name, color in segments:
        seg_points = people_data[mask]
        if len(seg_points) > 0:
            traces.append(go.Scatter3d(
                x=seg_points[:, 0],
                y=seg_points[:, 1],
                z=seg_points[:, 3],
                mode='markers',
                marker=dict(size=4, color=color),
                name=name,
                text=[f"Age: {age}<br>Income: {income}<br>Purchase: {purchase}<br>Freq: {freq}"
                      for age, income, purchase, freq in seg_points],
                hoverinfo='text',
            ))
    layout = go.Layout(
        title="Customer Segments: 3D Visualization",
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

import argparse
from data import generate_people_data

def main() -> None:
    """
    CLI entry point for customer segment visualization.
    """

    parser = argparse.ArgumentParser(description="K-means clustering and visualization for synthetic people data.")
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of people to generate')
    # Young Professionals
    parser.add_argument('--young_age_min', type=int, default=25, help='Young Professionals: minimum age')
    parser.add_argument('--young_age_max', type=int, default=35, help='Young Professionals: maximum age')
    parser.add_argument('--young_income_min', type=int, default=90000, help='Young Professionals: minimum income')
    parser.add_argument('--young_freq_min', type=int, default=50, help='Young Professionals: minimum purchase frequency')
    # Budget Conscious
    parser.add_argument('--budget_income_max', type=int, default=40000, help='Budget Conscious: maximum income')
    parser.add_argument('--budget_freq_min', type=int, default=30, help='Budget Conscious: minimum purchase frequency')
    # Loyal Customers
    parser.add_argument('--loyal_freq_min', type=int, default=80, help='Loyal Customers: minimum purchase frequency')
    parser.add_argument('--loyal_purchase_min', type=int, default=20000, help='Loyal Customers: minimum purchase history')

    args = parser.parse_args()

    people_data = generate_people_data(num_samples=args.num_samples)
    print("Random people data (first few rows):")
    print(people_data[:5])
    print("\nColumns: Age, Income, Purchase History, Frequency of Purchase")

    def plot_segments(people_data):
        import plotly.graph_objs as go
        mask_young_prof = (
            (people_data[:, 0] >= args.young_age_min) & (people_data[:, 0] <= args.young_age_max) &
            (people_data[:, 1] >= args.young_income_min) & (people_data[:, 3] >= args.young_freq_min)
        )
        mask_budget = (
            (people_data[:, 1] <= args.budget_income_max) & (people_data[:, 3] >= args.budget_freq_min)
        )
        mask_loyal = (
            (people_data[:, 3] >= args.loyal_freq_min) & (people_data[:, 2] >= args.loyal_purchase_min)
        )
        segments = [
            (mask_young_prof, 'Young Professionals', 'blue'),
            (mask_budget, 'Budget Conscious', 'green'),
            (mask_loyal, 'Loyal Customers', 'red'),
        ]
        traces = []
        for mask, name, color in segments:
            seg_points = people_data[mask]
            if len(seg_points) > 0:
                traces.append(go.Scatter3d(
                    x=seg_points[:, 0],
                    y=seg_points[:, 1],
                    z=seg_points[:, 3],
                    mode='markers',
                    marker=dict(size=4, color=color),
                    name=name,
                    text=[f"Age: {age}<br>Income: {income}<br>Purchase: {purchase}<br>Freq: {freq}"
                          for age, income, purchase, freq in seg_points],
                    hoverinfo='text',
                ))
        layout = go.Layout(
            title="Customer Segments: 3D Visualization",
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

    plot_segments(people_data)
