import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_segmentation(features_df: pd.DataFrame, use_kmeans: bool = True, k: int = None, random_state: int = 42, balance_threshold: float = 100000.0, recency_threshold: int = 30) -> tuple[pd.DataFrame, dict]:
    """
    Perform customer segmentation using KMeans or a Rule-Based fallback.
    
    Parameters:
    features_df (pd.DataFrame): Customer-level features dataframe.
    use_kmeans (bool): Whether to run KMeans clustering. If False, runs rule-based fallback.
    k (int): Number of clusters. If None, performs elbow/silhouette evaluation to pick k.
    random_state (int): Random seed for reproducibility.
    balance_threshold (float): Balance threshold for priority rules.
    recency_threshold (int): Recency threshold in days for priority rules.
    
    Returns:
    tuple[pd.DataFrame, dict]: (df_with_labels, metrics_dict)
    """
    df_out = features_df.copy()
    metrics = {}
    
    # Identify numerical columns for scaling and clustering
    num_cols = [
        'AgeAtTx', 'Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg', 
        'MaxTxAmount', 'AvgMaxMonthlyBalance', 'MaxMonthlyBalance', 
        'AvgAccountBalance', 'TenureDays', 'TxFrequencyDaily'
    ]
    # Filter columns that actually exist in the dataframe
    num_cols = [c for c in num_cols if c in df_out.columns]
    
    # Handle NaNs in numerical columns (fill with median)
    for col in num_cols:
        if df_out[col].isnull().any():
            df_out[col] = df_out[col].fillna(df_out[col].median())
            
    if use_kmeans:
        # Preprocessing: Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_out[num_cols])
        
        inertias = []
        silhouette_scores = {}
        
        # If number of clusters k is not specified, run optimal k search (k=2 to 5)
        k_values = range(2, 6)
        
        # Sample for silhouette score to avoid OOM or huge execution lag
        sample_size = min(10000, len(df_out))
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(df_out), size=sample_size, replace=False)
        scaled_sample = scaled_data[sample_indices]
        
        # Run KMeans for range of K to evaluate
        eval_metrics = {}
        for temp_k in k_values:
            kmeans_eval = KMeans(n_clusters=temp_k, random_state=random_state, n_init='auto')
            kmeans_eval.fit(scaled_data)
            
            inertias.append(kmeans_eval.inertia_)
            
            # Sampled Silhouette Score
            labels_sample = kmeans_eval.predict(scaled_sample)
            sil_val = float(silhouette_score(scaled_sample, labels_sample))
            silhouette_scores[temp_k] = sil_val
            eval_metrics[temp_k] = {
                'inertia': float(kmeans_eval.inertia_),
                'silhouette_score': sil_val
            }
            
        metrics['optimal_k_metrics'] = eval_metrics
        
        # If k not provided, select k that maximizes silhouette score
        if k is None:
            k = max(silhouette_scores, key=silhouette_scores.get)
            metrics['selected_k'] = int(k)
        else:
            metrics['selected_k'] = int(k)
            
        # Fit final KMeans model
        final_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        df_out['Cluster'] = final_kmeans.fit_predict(scaled_data)
        
        metrics['final_inertia'] = float(final_kmeans.inertia_)
        metrics['final_silhouette'] = float(silhouette_score(scaled_sample, final_kmeans.predict(scaled_sample)))

        # FIX Issue 2: Compute and store the single source of truth for which cluster = Priority.
        # Priority cluster = the cluster with the highest mean AvgAccountBalance.
        cluster_balance_means = df_out.groupby('Cluster')['AvgAccountBalance'].mean()
        priority_cluster_id = int(cluster_balance_means.idxmax())
        metrics['priority_cluster_id'] = priority_cluster_id
        
    else:
        # Rule-based fallback classification
        # priority (high balance + active), dormant (inactive long time), regular (other active)
        conditions = [
            (df_out['AvgAccountBalance'] > balance_threshold) & (df_out['Recency'] <= recency_threshold), # Priority
            (df_out['Recency'] > 60), # Dormant
        ]
        choices = ['Priority', 'Dormant']
        df_out['Cluster'] = np.select(conditions, choices, default='Regular')
        
        metrics['selected_k'] = len(df_out['Cluster'].unique())
        metrics['segment_counts'] = df_out['Cluster'].value_counts().to_dict()
        # Rule-based segments are named strings, Priority cluster ID is string "Priority"
        metrics['priority_cluster_id'] = 'Priority'
        
    return df_out, metrics

if __name__ == "__main__":
    import os
    import sys
    
    csv_path = "data/customer_features.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run feature engineering first.")
        sys.exit(1)
        
    df_feats = pd.read_csv(csv_path)
    
    print("Testing segmentation tool (KMeans)...")
    res_kmeans, met_kmeans = perform_segmentation(df_feats, use_kmeans=True)
    print(f"Optimal K Selected: {met_kmeans['selected_k']}")
    print(f"Priority Cluster ID: {met_kmeans['priority_cluster_id']}")
    print("KMeans Evaluation Metrics:")
    for k_val, vals in met_kmeans['optimal_k_metrics'].items():
        print(f"  K={k_val}: Inertia = {vals['inertia']:.2f}, Silhouette Score = {vals['silhouette_score']:.4f}")
    print(f"Final Assigned Clusters Distribution:\n{res_kmeans['Cluster'].value_counts()}")
    
    print("\nTesting segmentation tool (Rule-based Fallback)...")
    res_rules, met_rules = perform_segmentation(df_feats, use_kmeans=False)
    print(f"Rule-based Segment Distribution:\n{res_rules['Cluster'].value_counts()}")
