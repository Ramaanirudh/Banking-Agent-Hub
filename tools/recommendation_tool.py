import pandas as pd
import numpy as np

def map_clusters_to_segments(profile_df: pd.DataFrame) -> dict:
    """
    Dynamically map numeric cluster IDs to semantic names (Priority, Regular, Dormant)
    based on centroid characteristics (Account Balance and Recency).
    
    Parameters:
    profile_df (pd.DataFrame): The cluster centroid profiles from profile_clusters().
    
    Returns:
    dict: Map of {Cluster_ID: Semantic_Name}
    """
    mapping = {}
    
    # Sort clusters by balance (descending) to identify high-value/priority
    sorted_by_balance = profile_df.sort_values(by='AvgAccountBalance', ascending=False)
    highest_balance_cluster = sorted_by_balance.index[0]
    
    # Sort clusters by recency (descending) to identify dormant/inactive
    sorted_by_recency = profile_df.sort_values(by='Recency', ascending=False)
    highest_recency_cluster = sorted_by_recency.index[0]
    
    # Map clusters
    for cluster_id in profile_df.index:
        if cluster_id == highest_balance_cluster:
            mapping[cluster_id] = "Priority"
        elif cluster_id == highest_recency_cluster:
            mapping[cluster_id] = "Dormant"
        else:
            mapping[cluster_id] = "Regular"
            
    # Handle overlap/edge case (e.g., if there are only 2 clusters or highest balance has highest recency)
    # Ensure unique names if possible
    if len(mapping.values()) != len(set(mapping.values())):
        # Fallback to simple rank mapping
        for idx, cluster_id in enumerate(sorted_by_balance.index):
            if idx == 0:
                mapping[cluster_id] = "Priority"
            elif idx == len(sorted_by_balance) - 1:
                mapping[cluster_id] = "Dormant"
            else:
                mapping[cluster_id] = f"Regular_C{cluster_id}"
                
    return mapping

def get_recommendation_for_row(row: pd.Series, segment_name: str = None) -> dict:
    """
    Get financial product recommendations for a single customer row.
    
    Parameters:
    row (pd.Series): A single customer data row.
    segment_name (str): Optional override for the semantic segment. 
                        If None, determined on-the-fly using thresholds.
    
    Returns:
    dict: Recommendations dictionary.
    """
    balance = row.get('AvgAccountBalance', 0)
    recency = row.get('Recency', 90)
    frequency = row.get('Frequency', 1)
    
    # Determine segment on-the-fly if not provided
    if segment_name is None:
        if balance > 100000 and recency <= 30:
            segment_name = "Priority"
        elif recency > 60:
            segment_name = "Dormant"
        else:
            segment_name = "Regular"
            
    recs = {
        'segment': segment_name,
        'recommended_products': [],
        'campaign_name': "",
        'actionable_message': ""
    }
    
    if segment_name == "Priority":
        recs['recommended_products'] = ["Wealth Management", "Premium Infinite Credit Card", "Fixed Deposit (High-Yield)"]
        recs['campaign_name'] = "Elite Wealth & Lifestyle Partnership"
        recs['actionable_message'] = (
            f"Dear Customer, with your current balance of INR {balance:,.2f}, you qualify for our Wealth Management services. "
            f"Contact your dedicated relationship manager to explore high-yield investment options."
        )
    elif segment_name == "Dormant":
        recs['recommended_products'] = ["Zero-Balance Savings Account", "1-Year Fixed Deposit Special", "E-Statements Enrollment"]
        recs['campaign_name'] = "Welcome Back - Re-engage & Save"
        recs['actionable_message'] = (
            f"We miss you! Reactivate your transactions today to lock in a special 7.5% p.a. interest rate on a 1-year Term Deposit. "
            f"Zero maintenance fees for the next 12 months."
        )
    else: # Regular
        recs['recommended_products'] = ["Personal Loan Pre-Approval", "Cashback Rewards Credit Card", "Mutual Fund SIP"]
        recs['campaign_name'] = "Grow & Transact Smartly"
        recs['actionable_message'] = (
            f"Start building your wealth today. Set up a Monthly Systematic Investment Plan (SIP) starting at just INR 1,000. "
            f"Also, check out your pre-approved credit card offers!"
        )
        
    return recs

def generate_recommendations_batch(df: pd.DataFrame, cluster_mapping: dict = None) -> list[dict]:
    """
    Generate recommendations for a batch of customers.
    """
    recommendations_list = []
    for _, row in df.iterrows():
        cluster_id = row.get('Cluster')
        seg_name = cluster_mapping.get(cluster_id) if cluster_mapping and cluster_id in cluster_mapping else None
        rec = get_recommendation_for_row(row, segment_name=seg_name)
        rec['CustomerID'] = row.get('CustomerID')
        recommendations_list.append(rec)
    return recommendations_list

def find_near_priority_customers(df: pd.DataFrame, threshold_pct: float = 0.1) -> pd.DataFrame:
    """
    Identify active 'Regular' customers who are within threshold_pct of the Priority balance threshold (100,000 INR).
    
    Parameters:
    df (pd.DataFrame): Customer features dataframe.
    threshold_pct (float): Distance threshold below 100,000 INR (default 0.1 for 10% i.e. balance >= 90k).
    
    Returns:
    pd.DataFrame: Dataframe of high-potential upsell candidates.
    """
    priority_threshold = 100000
    lower_bound = priority_threshold * (1.0 - threshold_pct)
    
    # Filter: Balance between lower_bound and 100k, and active (recency <= 30 days)
    # Exclude those already classified as Priority in rule-based terms
    candidates = df[
        (df['AvgAccountBalance'] >= lower_bound) & 
        (df['AvgAccountBalance'] < priority_threshold) & 
        (df['Recency'] <= 30)
    ].copy()
    
    # Calculate balance gap to cross priority tier
    candidates['BalanceGapToPriority'] = priority_threshold - candidates['AvgAccountBalance']
    
    # Sort by balance gap ascending (closest first)
    return candidates.sort_values(by='BalanceGapToPriority', ascending=True)

if __name__ == "__main__":
    import os
    import sys
    from segmentation_tool import perform_segmentation
    from explainability_tool import profile_clusters
    
    csv_path = "data/customer_features.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run feature engineering first.")
        sys.exit(1)
        
    df_feats = pd.read_csv(csv_path)
    
    # Run KMeans
    print("Running KMeans to demonstrate dynamic mapping...")
    df_seg, metrics = perform_segmentation(df_feats, use_kmeans=True, k=3)
    profile = profile_clusters(df_seg)
    
    cluster_map = map_clusters_to_segments(profile)
    print(f"Dynamic Cluster Mapping: {cluster_map}")
    
    # Get recommendation for a sample customer
    sample_cust = df_seg.iloc[0]
    rec = get_recommendation_for_row(sample_cust, segment_name=cluster_map.get(sample_cust['Cluster']))
    print(f"\nSample Recommendation for Customer {sample_cust['CustomerID']}:")
    print(f"  Segment: {rec['segment']}")
    print(f"  Campaign: {rec['campaign_name']}")
    print(f"  Products: {rec['recommended_products']}")
    print(f"  Message: {rec['actionable_message']}")
    
    # Find near priority customers
    print("\nFinding near-priority upsell candidates...")
    near_priority = find_near_priority_customers(df_feats, threshold_pct=0.1)
    print(f"Found {len(near_priority):,} candidates out of {len(df_feats):,} total customers.")
    print("Top 5 candidates:")
    print(near_priority[['CustomerID', 'AvgAccountBalance', 'Recency', 'BalanceGapToPriority']].head(5).to_string(index=False))
