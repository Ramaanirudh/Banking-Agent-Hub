import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.io as pio

# Set default renderer to browser-friendly format
pio.templates.default = "plotly_dark"

def plot_clusters_scatter(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create an interactive 3D scatter plot of Recency vs. MonetaryAvg vs. AvgAccountBalance.
    Downsamples data to ensure browser performance.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Downsample for Plotly visualization
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    
    # Clean cluster type for plotting
    df_sample['Cluster_Label'] = df_sample['Cluster'].astype(str)
    
    # Use log-scale for balance and monetary to handle outliers better
    df_sample['Log_AvgAccountBalance'] = np.log1p(df_sample['AvgAccountBalance'])
    df_sample['Log_MonetaryAvg'] = np.log1p(df_sample['MonetaryAvg'])
    
    fig = px.scatter_3d(
        df_sample,
        x='Recency',
        y='Log_MonetaryAvg',
        z='Log_AvgAccountBalance',
        color='Cluster_Label',
        hover_data=['CustomerID', 'AgeAtTx', 'Frequency'],
        labels={
            'Recency': 'Recency (Days)',
            'Log_MonetaryAvg': 'Log Avg Tx Amount (INR)',
            'Log_AvgAccountBalance': 'Log Avg Account Balance (INR)'
        },
        title=f"3D Cluster Scatter Plot (Sampled N={sample_size})",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(title="Segment / Cluster")
    )
    
    output_path = os.path.join(output_dir, 'cluster_scatter_3d.html')
    fig.write_html(output_path, include_plotlyjs='cdn')
    return output_path

def plot_segment_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create a bar chart showing the count and percentage of customers in each segment.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    counts = df['Cluster'].value_counts().reset_index()
    counts.columns = ['Segment', 'Customer_Count']
    counts['Percentage'] = (counts['Customer_Count'] / len(df)) * 100
    counts['Segment'] = counts['Segment'].astype(str)
    
    fig = px.bar(
        counts,
        x='Segment',
        y='Customer_Count',
        text=counts['Percentage'].apply(lambda x: f"{x:.2f}%"),
        labels={
            'Segment': 'Customer Segment / Cluster',
            'Customer_Count': 'Number of Customers'
        },
        title="Customer Distribution by Segment",
        color='Segment',
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        uniformtext_minsize=8, 
        uniformtext_mode='hide',
        showlegend=False
    )
    
    output_path = os.path.join(output_dir, 'segment_distribution.html')
    fig.write_html(output_path, include_plotlyjs='cdn')
    return output_path

def plot_feature_comparison(df: pd.DataFrame, output_dir: str) -> str:
    """
    Create comparative box plots for key metrics across segments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    df_sample['Cluster_Label'] = df_sample['Cluster'].astype(str)
    
    # We will plot AvgAccountBalance on log scale due to high skewness
    df_sample['Log_AvgAccountBalance'] = np.log1p(df_sample['AvgAccountBalance'])
    
    fig = px.box(
        df_sample,
        x='Cluster_Label',
        y='Log_AvgAccountBalance',
        color='Cluster_Label',
        labels={
            'Cluster_Label': 'Segment / Cluster',
            'Log_AvgAccountBalance': 'Log Avg Account Balance (INR)'
        },
        title=f"Comparative Account Balance (Log-Scale) across Clusters (Sampled N={sample_size})",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    fig.update_layout(showlegend=False)
    
    output_path = os.path.join(output_dir, 'feature_comparison_box.html')
    fig.write_html(output_path, include_plotlyjs='cdn')
    return output_path

if __name__ == "__main__":
    import sys
    from segmentation_tool import perform_segmentation
    
    csv_path = "data/customer_features.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run feature engineering first.")
        sys.exit(1)
        
    df_feats = pd.read_csv(csv_path)
    
    print("Performing segmentation...")
    df_seg, _ = perform_segmentation(df_feats, use_kmeans=True, k=3)
    
    report_dir = "reports"
    print(f"Generating charts in '{report_dir}'...")
    p1 = plot_clusters_scatter(df_seg, report_dir)
    p2 = plot_segment_distribution(df_seg, report_dir)
    p3 = plot_feature_comparison(df_seg, report_dir)
    
    print(f"Done! HTML reports generated:")
    print(f"  - Scatter plot: {p1}")
    print(f"  - Distribution chart: {p2}")
    print(f"  - Box plot: {p3}")
