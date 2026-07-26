import pandas as pd
import sys
import os

# Ensure tools directory is in the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))

from segmentation_tool import perform_segmentation
from explainability_tool import profile_clusters, explain_clusters_tree, get_readable_rules
from recommendation_tool import map_clusters_to_segments, generate_recommendations_batch, find_near_priority_customers
from visualization_tool import plot_clusters_scatter, plot_segment_distribution, plot_feature_comparison

def main():
    features_path = "data/customer_features.csv"
    if not os.path.exists(features_path):
        print(f"Error: {features_path} not found. Please run Stage 1 (test_tools.py) first to generate features.")
        sys.exit(1)
        
    print(f"==================================================")
    print(f"STAGE 2 VERIFICATION RUN")
    print(f"==================================================")
    print(f"Loading customer features from {features_path}...")
    df_feats = pd.read_csv(features_path)
    print(f"Loaded {len(df_feats):,} customer profiles.")
    
    # --------------------------------------------------
    # 1. Test Segmentation
    # --------------------------------------------------
    print(f"\n[1/4] Running Segmentation Tool...")
    print("Executing KMeans optimal K selection (Silhouette scores computed on sampled size)...")
    df_seg, kmeans_metrics = perform_segmentation(df_feats, use_kmeans=True)
    
    selected_k = kmeans_metrics['selected_k']
    print(f"  Optimal K selected automatically: {selected_k}")
    print("  Clustering performance metrics per K:")
    for k_val, vals in kmeans_metrics['optimal_k_metrics'].items():
        print(f"    K={k_val}: Inertia = {vals['inertia']:.2f}, Silhouette Score (Sampled) = {vals['silhouette_score']:.4f}")
        
    print("\n  Assigned KMeans Segment Counts:")
    print(df_seg['Cluster'].value_counts().to_string())
    
    # Test Rule-based Fallback
    print("\n  Executing Rule-Based Fallback Segmentation...")
    df_rule, rule_metrics = perform_segmentation(df_feats, use_kmeans=False)
    print("  Rule-Based Segment Counts:")
    for seg_name, count in rule_metrics['segment_counts'].items():
        pct = (count / len(df_feats)) * 100
        print(f"    {seg_name}: {count:,} ({pct:.2f}%)")
        
    # --------------------------------------------------
    # 2. Test Explainability
    # --------------------------------------------------
    print(f"\n[2/4] Running Explainability Tool...")
    print("  Generating Cluster Centroid Profiles (KMeans clusters):")
    profiles = profile_clusters(df_seg)
    print(profiles.T.to_string())
    
    print("\n  Extracting split rules using Decision Tree surrogate...")
    rules, accuracy = explain_clusters_tree(df_seg, max_depth=3)
    print(f"  Decision Tree Surrogate Accuracy vs Cluster Labels: {accuracy*100:.2f}%")
    readable_rules = get_readable_rules(rules)
    for r in readable_rules:
        print(f"    {r}\n")
        
    # --------------------------------------------------
    # 3. Test Recommendations
    # --------------------------------------------------
    print(f"\n[3/4] Running Recommendation Tool...")
    print("  Mapping numeric clusters to semantic segments...")
    cluster_mapping = map_clusters_to_segments(profiles)
    print(f"  Cluster mapping: {cluster_mapping}")
    
    # Select a few sample rows to show recommendations
    sample_batch = df_seg.head(3)
    print("\n  Generating recommendations for sample customers:")
    recs = generate_recommendations_batch(sample_batch, cluster_mapping)
    for r in recs:
        print(f"    Customer ID: {r['CustomerID']}")
        print(f"      Mapped Segment: {r['segment']}")
        print(f"      Recommended Campaign: {r['campaign_name']}")
        print(f"      Products: {r['recommended_products']}")
        print(f"      Message: {r['actionable_message']}\n")
        
    # Identify near-priority customers
    print("  Checking for near-priority (upsell) candidates...")
    near_priority_df = find_near_priority_customers(df_feats, threshold_pct=0.10)
    print(f"    Total active candidates found within 10% below Priority threshold (90k-100k balance): {len(near_priority_df):,}")
    if len(near_priority_df) > 0:
        print("    Top 5 closest candidates for Priority Upgrade Campaigns:")
        print(near_priority_df[['CustomerID', 'AvgAccountBalance', 'Recency', 'BalanceGapToPriority']].head(5).to_string(index=False))
        
    # --------------------------------------------------
    # 4. Test Visualization
    # --------------------------------------------------
    print(f"\n[4/4] Running Visualization Tool...")
    report_dir = "reports"
    print(f"  Generating Plotly interactive charts in '{report_dir}' folder...")
    
    p1 = plot_clusters_scatter(df_seg, report_dir)
    p2 = plot_segment_distribution(df_seg, report_dir)
    p3 = plot_feature_comparison(df_seg, report_dir)
    
    print("  Verifying output file paths:")
    for p in [p1, p2, p3]:
        exists = os.path.exists(p)
        print(f"    File: {p} | Status: {'CREATED' if exists else 'FAILED'}")
        
    print(f"\n==================================================")
    print(f"Stage 2 Verification Complete!")
    print(f"==================================================")

if __name__ == "__main__":
    main()
