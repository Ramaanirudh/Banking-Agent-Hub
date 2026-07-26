import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def profile_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute centroids (feature averages) for each cluster.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing numerical features and a 'Cluster' column.
    
    Returns:
    pd.DataFrame: A profiling dataframe with centroid averages and cluster sizes.
    """
    num_cols = [
        'AgeAtTx', 'Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg', 
        'MaxTxAmount', 'AvgMaxMonthlyBalance', 'MaxMonthlyBalance', 
        'AvgAccountBalance', 'TenureDays', 'TxFrequencyDaily'
    ]
    # Filter only columns present
    num_cols = [c for c in num_cols if c in df.columns]
    
    # Calculate group means
    centroids = df.groupby('Cluster')[num_cols].mean()
    
    # Calculate sizes of each cluster
    counts = df['Cluster'].value_counts()
    percentages = df['Cluster'].value_counts(normalize=True) * 100
    
    # Merge count stats into centroids dataframe
    profile = centroids.copy()
    profile['Cluster_Size'] = counts
    profile['Cluster_Percentage'] = percentages
    
    return profile

def explain_clusters_tree(df: pd.DataFrame, max_depth: int = 3, random_state: int = 42) -> list[dict]:
    """
    Train a shallow decision tree surrogate model on the clustering labels and 
    extract human-readable split rules.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing numerical features and a 'Cluster' column.
    max_depth (int): Max depth of decision tree surrogate.
    random_state (int): Random state for tree classifier.
    
    Returns:
    list[dict]: List of rules, each represented as a dictionary containing conditions, 
                predicted cluster, support, and purity.
    """
    num_cols = [
        'AgeAtTx', 'Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg', 
        'MaxTxAmount', 'AvgMaxMonthlyBalance', 'MaxMonthlyBalance', 
        'AvgAccountBalance', 'TenureDays', 'TxFrequencyDaily'
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    
    # Prepare X and y
    X = df[num_cols].copy()
    
    # Fill any NaNs
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
        
    y = df['Cluster'].astype(str)
    
    # Train the decision tree surrogate
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X, y)
    
    # Calculate accuracy vs cluster labels
    accuracy = float(dt.score(X, y))
    
    # Traverse tree to extract rules
    tree_ = dt.tree_
    feature_names = X.columns
    rules = []
    
    def recurse(node, path_conditions):
        # Check if it is a leaf node (no children)
        if tree_.children_left[node] != tree_.children_right[node]:
            # It's an internal node
            feat_idx = tree_.feature[node]
            feat_name = feature_names[feat_idx]
            threshold = tree_.threshold[node]
            
            # Left child path (feature <= threshold)
            left_cond = f"{feat_name} <= {threshold:.2f}"
            recurse(tree_.children_left[node], path_conditions + [left_cond])
            
            # Right child path (feature > threshold)
            right_cond = f"{feat_name} > {threshold:.2f}"
            recurse(tree_.children_right[node], path_conditions + [right_cond])
        else:
            # It's a leaf node
            value_counts = tree_.value[node][0]
            predicted_class_idx = np.argmax(value_counts)
            predicted_class = dt.classes_[predicted_class_idx]
            
            support = int(np.sum(value_counts))
            purity = float(value_counts[predicted_class_idx] / support) if support > 0 else 0.0
            
            rules.append({
                'predicted_cluster': predicted_class,
                'conditions': path_conditions,
                'support': support,
                'purity': purity
            })
            
    recurse(0, [])
    return rules, accuracy

def get_readable_rules(rules: list[dict]) -> list[str]:
    """
    Format extracted rules as human-readable text.
    """
    formatted_rules = []
    for idx, rule in enumerate(rules):
        conds_str = " AND ".join(rule['conditions']) if rule['conditions'] else "Always"
        purity_pct = rule['purity'] * 100
        readable_str = (
            f"Rule #{idx+1} [Predicts Cluster {rule['predicted_cluster']}]:\n"
            f"  If: {conds_str}\n"
            f"  Detail: Purity = {purity_pct:.1f}%, Support = {rule['support']:,} records"
        )
        formatted_rules.append(readable_str)
    return formatted_rules

if __name__ == "__main__":
    import os
    import sys
    from segmentation_tool import perform_segmentation
    
    csv_path = "data/customer_features.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run feature engineering first.")
        sys.exit(1)
        
    df_feats = pd.read_csv(csv_path)
    
    # Segment data using K=3 for standard explainability demo
    print("Running segmentation first...")
    df_segmented, _ = perform_segmentation(df_feats, use_kmeans=True, k=3)
    
    print("\nProfiling clusters...")
    profile = profile_clusters(df_segmented)
    print(profile.to_string())
    
    print("\nGenerating explainability rules via decision tree surrogate...")
    extracted_rules, accuracy = explain_clusters_tree(df_segmented, max_depth=3)
    print(f"Decision Tree Surrogate Accuracy: {accuracy*100:.2f}%")
    readable_rules = get_readable_rules(extracted_rules)
    for rule in readable_rules:
        print(rule)
        print()
