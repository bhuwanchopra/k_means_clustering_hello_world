def plot_customer_segments_3d(people_data):
    """
    Visualize three customer segments in a 3D plot: Young Professionals, Budget Conscious, Loyal Customers.
    """
    import plotly.graph_objs as go
    # Define segment masks
    # Young Professionals: Age 25-35, Income > 90000, Frequency > 50
    mask_young_prof = (
        (people_data[:, 0] >= 25) & (people_data[:, 0] <= 35) &
        (people_data[:, 1] > 90000) & (people_data[:, 3] > 50)
    )
    # Budget Conscious: Income < 40000, Frequency > 30
    mask_budget = (
        (people_data[:, 1] < 40000) & (people_data[:, 3] > 30)
    )
    # Loyal Customers: Frequency > 80, Purchase History > 20000
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

def main():
    parser = argparse.ArgumentParser(description="K-means clustering and visualization for synthetic people data.")
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of people to generate')
    # Only --segments is needed for the new workflow
    parser.add_argument('--segments', action='store_true', help='Visualize three customer segments in a 3D plot')
    args = parser.parse_args()

    people_data = generate_people_data(num_samples=args.num_samples)
    print("Random people data (first few rows):")
    print(people_data[:5])
    print("\nColumns: Age, Income, Purchase History, Frequency of Purchase")

    if args.segments:
        plot_customer_segments_3d(people_data)
    else:
        print("Please use --segments to visualize customer segments in 3D.")
