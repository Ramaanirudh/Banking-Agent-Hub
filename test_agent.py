import pandas as pd
import numpy as np
import sys
import os
import json
import re

# Ensure agent and tools directories are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agent')))

from router import route_query
from synthesizer import synthesize_answer
from segmentation_tool import perform_segmentation
from explainability_tool import profile_clusters, explain_clusters_tree, get_readable_rules
from recommendation_tool import map_clusters_to_segments, find_near_priority_customers, get_recommendation_for_row
from stats_tool import calculate_segment_aggregation
from eda_tool import run_eda

def execute_agent_loop(query: str, df_feats: pd.DataFrame, session_context: dict = None) -> str:
    """
    Executes the full agent loop: Router -> Tool Execution -> Synthesizer.

    Parameters:
    query       : The raw user query string.
    df_feats    : Precomputed customer features dataframe.
    session_context : Mutable dict shared across calls in a session.
                      Keys used:
                        "pending_query" – stored when clarification is issued;
                        cleared once the follow-up is processed.
    """
    if session_context is None:
        session_context = {}

    print(f"\n==================================================")
    print(f"USER QUERY: {query}")
    print(f"==================================================")

    # ── Issue 4: Context carry-forward ─────────────────────────────────────────
    # If there is a pending clarification query, combine it with the new input
    # before routing, then clear the pending entry.
    combined_query = query
    if session_context.get("pending_query"):
        combined_query = f"{session_context['pending_query']} based on {query}"
        print(f"[Session] Combining pending query: '{session_context['pending_query']}' + '{query}' => '{combined_query}'")
        session_context.pop("pending_query", None)

    # 1. Router Step
    plan = route_query(combined_query)
    intent = plan.get('intent')
    params = plan.get('parameters', {})
    
    # Map intents to planned tools for trace logs
    tools_map = {
        'segmentation': 'segmentation',
        'explain_rule': 'explainability',
        'aggregate_stat': 'stats_tool',
        'entity_lookup': 'lookup_tool',
        'entity_list': 'list_tool',
        'recommendation': 'recommendation',
        'conversion_candidates': 'recommendation',
        'eda': 'eda_tool',
        'clarification_needed': 'none'
    }
    planned_tool = tools_map.get(intent, 'none')
    print(f"[Router] Intent: {intent} | Tools planned: {planned_tool}")
    
    if intent == 'out_of_scope':
        return "This question is outside the scope of customer segmentation analysis. I can help with customer segments, transaction patterns, or product recommendations instead."
        
    if intent == 'clarification_needed':
        clarification = plan.get('clarification_question', "I need more details to answer your query. Could you please specify?")
        # ── Issue 4: Store original query for follow-up context ────────────────
        session_context["pending_query"] = query
        return f"[Agent Clarification]: {clarification}"
        
    # Persistent paths
    segmented_path = "data/customer_features_segmented.csv"
    metrics_path = "data/segmentation_metrics.json"
    
    # Tool Execution Step
    tool_output_str = ""
    
    if intent == 'segmentation':
        use_kmeans = params.get('use_kmeans', True)
        k = params.get('k', None)
        balance_threshold = params.get('balance_threshold', 100000.0)
        recency_threshold = params.get('recency_threshold', 30)
        
        # Run segmentation tool
        df_seg, seg_metrics = perform_segmentation(
            df_feats, 
            use_kmeans=use_kmeans, 
            k=k, 
            balance_threshold=balance_threshold, 
            recency_threshold=recency_threshold
        )
        print(f"[Tool: segmentation] done")
        
        # Cache results for consistency across session
        df_seg.to_csv(segmented_path, index=False)
        with open(metrics_path, 'w') as f:
            json.dump(seg_metrics, f)
            
        if use_kmeans:
            # ── Issue 2: Use priority_cluster_id from metrics (single source of truth)
            priority_cid = seg_metrics.get('priority_cluster_id', 0)
            cluster_counts = df_seg['Cluster'].value_counts()
            priority_count = cluster_counts.get(priority_cid, 0)
            other_counts = {f"Cluster {cid}": cnt for cid, cnt in cluster_counts.items() if cid != priority_cid}
            other_label = "Standard/Inactive" if len(cluster_counts) == 2 else "Regular/Other"

            tool_output_str = (
                f"KMeans Segmentation Results:\n"
                f"  Optimal K selected: {seg_metrics.get('selected_k')}\n"
                f"  Priority Cluster ID: {priority_cid}\n"
                f"  Cluster Distribution:\n{df_seg['Cluster'].value_counts().to_string()}\n"
                f"  Optimal K metrics (silhouette and inertia):\n"
                f"{json.dumps(seg_metrics.get('optimal_k_metrics'), indent=2)}"
            )
        else:
            tool_output_str = (
                f"Rule-Based Fallback Segmentation Results:\n"
                f"  Segment Counts:\n{json.dumps(seg_metrics.get('segment_counts'), indent=2)}\n"
                f"  Rule thresholds: \n"
                f"    - Priority: AvgAccountBalance > {balance_threshold:,.2f} INR and Recency <= {recency_threshold} days\n"
                f"    - Dormant: Recency > 60 days\n"
                f"    - Regular: All other active customers"
            )
            
    else:
        # Load cached segmented data if exists, otherwise generate default
        if os.path.exists(segmented_path):
            df_seg = pd.read_csv(segmented_path)
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        seg_metrics = json.load(f)
                except Exception:
                    seg_metrics = {}
            else:
                seg_metrics = {}
        else:
            # Default segment setup
            df_seg, seg_metrics = perform_segmentation(df_feats, use_kmeans=True)
            df_seg.to_csv(segmented_path, index=False)
            with open(metrics_path, 'w') as f:
                json.dump(seg_metrics, f)

        # ── Issue 2: Single source of truth for priority cluster ───────────────
        priority_cluster_id = seg_metrics.get('priority_cluster_id', None)
        if priority_cluster_id is None and 'Cluster' in df_seg.columns:
            # Compute it on the fly if not in metrics (backward compat)
            try:
                cb_means = df_seg.groupby('Cluster')['AvgAccountBalance'].mean()
                priority_cluster_id = cb_means.idxmax()
                seg_metrics['priority_cluster_id'] = priority_cluster_id
            except Exception:
                priority_cluster_id = 0  # safe fallback
                
        if intent == 'explain_rule':
            profile = profile_clusters(df_seg)
            rules, accuracy = explain_clusters_tree(df_seg, max_depth=3)
            readable_rules = get_readable_rules(rules)
            print(f"[Tool: explainability] done")
            
            tool_output_str = (
                f"Explainability Analysis:\n"
                f"  Priority Cluster ID: {priority_cluster_id}\n"
                f"  Surrogate Decision Tree Accuracy: {accuracy*100:.2f}%\n"
                f"  Extracted Segmentation Split Rules:\n" + "\n".join(readable_rules) + "\n\n"
                f"  Cluster Centroid Profile (Averages per Cluster):\n"
                f"{profile.T.to_string()}"
            )
            
        elif intent in ['recommendation', 'conversion_candidates']:
            # ── Issue 6: Hypothetical customer recommendation ─────────────────
            hyp_balance = params.get('hypothetical_balance')
            hyp_frequency = params.get('hypothetical_frequency')

            if hyp_balance is not None or hyp_frequency is not None:
                # Classify the hypothetical customer using rule-based logic
                bal = hyp_balance if hyp_balance is not None else 0.0
                freq = hyp_frequency if hyp_frequency is not None else 0.0
                # Recency unknown for hypothetical — assume active (recency=1) if freq > 0
                rec = 1 if freq > 0 else 999

                if bal > 100000 and rec <= 30:
                    hyp_segment = "Priority"
                elif rec > 60:
                    hyp_segment = "Dormant"
                else:
                    hyp_segment = "Regular"

                fake_row = pd.Series({
                    'AvgAccountBalance': bal,
                    'Frequency': freq,
                    'Recency': rec
                })
                rec_result = get_recommendation_for_row(fake_row, segment_name=hyp_segment)
                print(f"[Tool: recommendation] hypothetical customer done")

                tool_output_str = (
                    f"Hypothetical Customer Recommendation:\n"
                    f"  Input: balance={bal:,.2f} INR, frequency={freq}\n"
                    f"  Classified as: {hyp_segment}\n"
                    f"  Recommendation:\n{json.dumps(rec_result, indent=2)}"
                )
            else:
                threshold_pct = params.get('threshold_pct', 0.10)
                
                # 1. Near priority analysis
                near_priority = find_near_priority_customers(df_feats, threshold_pct=threshold_pct)
                top_candidates = near_priority[['CustomerID', 'AvgAccountBalance', 'Recency', 'BalanceGapToPriority']].head(5)
                
                # 2. Base sample recommendations — use priority_cluster_id for correct labelling
                profile = profile_clusters(df_seg)
                cluster_map = map_clusters_to_segments(profile)
                
                sample_recs = []
                for _, row in df_seg.head(3).iterrows():
                    cluster_id = row['Cluster']
                    seg_name = cluster_map.get(cluster_id) or cluster_map.get(str(cluster_id)) or f"Cluster {cluster_id}"
                    rec = get_recommendation_for_row(row, segment_name=seg_name)
                    sample_recs.append({
                        'CustomerID': row['CustomerID'],
                        'Segment': rec['segment'],
                        'Campaign': rec['campaign_name'],
                        'Products': rec['recommended_products'],
                        'Message': rec['actionable_message']
                    })
                print(f"[Tool: recommendation] done")
                    
                tool_output_str = (
                    f"Campaign and Product Recommendations:\n"
                    f"  Priority Balance Threshold: 100,000 INR\n"
                    f"  Upsell Search Threshold: within {threshold_pct*100:.1f}% below Priority (90k-100k balance) and active (Recency <= 30 days)\n"
                    f"  Count of Regular customers eligible for Priority conversion: {len(near_priority):,}\n"
                    f"  Top 5 candidates close to Priority upgrade:\n"
                    f"{top_candidates.to_string(index=False)}\n\n"
                    f"  Sample Customer Offers:\n"
                    f"{json.dumps(sample_recs, indent=2)}"
                )
            
        elif intent == 'aggregate_stat':
            group_col = params.get('group_col', 'Cluster')
            agg_col = params.get('agg_col', 'MonetaryAvg')
            agg_func = params.get('agg_func', 'mean')
            
            profile = profile_clusters(df_seg)
            
            # Get rule segment counts
            if 'segment_counts' in seg_metrics:
                rule_counts = seg_metrics['segment_counts']
            else:
                _, rule_metrics = perform_segmentation(df_feats, use_kmeans=False)
                rule_counts = rule_metrics.get('segment_counts')

            # ── Issue 5: Actually compute the requested aggregation ────────────
            # Determine grouping column — prefer RuleBasedSegment for named segments
            if agg_col in df_seg.columns:
                if group_col == 'RuleBasedSegment' or 'RuleBasedSegment' in df_seg.columns:
                    # Add rule-based segment column if missing
                    if 'RuleBasedSegment' not in df_seg.columns:
                        conditions = [
                            (df_seg['AvgAccountBalance'] > 100000) & (df_seg['Recency'] <= 30),
                            (df_seg['Recency'] > 60)
                        ]
                        df_seg = df_seg.copy()
                        df_seg['RuleBasedSegment'] = np.select(conditions, ['Priority', 'Dormant'], default='Regular')
                    grp_col_actual = 'RuleBasedSegment'
                else:
                    grp_col_actual = 'Cluster'

                if agg_func == 'sum':
                    agg_result = df_seg.groupby(grp_col_actual)[agg_col].sum().to_dict()
                elif agg_func == 'count':
                    agg_result = df_seg.groupby(grp_col_actual)[agg_col].count().to_dict()
                elif agg_func == 'percentage':
                    counts = df_seg.groupby(grp_col_actual)[agg_col].count()
                    agg_result = (counts / counts.sum() * 100).round(2).to_dict()
                else:  # mean
                    agg_result = df_seg.groupby(grp_col_actual)[agg_col].mean().round(2).to_dict()
            else:
                agg_result = {}

            stats_data = {
                'total_customers': len(df_seg),
                'kmeans_counts': df_seg['Cluster'].value_counts().to_dict(),
                'rule_counts': rule_counts,
                'kmeans_centroids': profile.to_dict(),
                'target_agg_col': agg_col,
                'target_agg_func': agg_func,
                'computed_result': agg_result,
                'priority_cluster_id': priority_cluster_id
            }
            print(f"[Tool: stats_tool] done")
            tool_output_str = f"Aggregate Statistics:\n{json.dumps(stats_data, indent=2)}"
            
        elif intent == 'entity_lookup':
            cust_id = params.get('customer_id', '')
            match = df_seg[df_seg['CustomerID'].str.upper() == cust_id.upper()]
            print(f"[Tool: lookup_tool] done")
            if match.empty:
                tool_output_str = f"Customer ID {cust_id} not found in database."
            else:
                cust_data = match.iloc[0].to_dict()
                balance = cust_data.get('AvgAccountBalance', 0)
                recency = cust_data.get('Recency', 99)
                # Rule-based segment
                if balance > 100000 and recency <= 30:
                    rule_segment = "Priority"
                elif recency > 60:
                    rule_segment = "Dormant"
                else:
                    rule_segment = "Regular"
                cust_data['RuleBasedSegment'] = rule_segment
                # ── Issue 2: Attach single source of truth for KMeans label ───
                cust_data['PriorityClusterID'] = priority_cluster_id
                cluster_val = cust_data.get('Cluster')
                cust_data['KMeansSegmentLabel'] = (
                    "Priority" if cluster_val == priority_cluster_id else "Standard/Inactive"
                )
                # Convert numpy types to native Python so json.dumps doesn't fail
                clean_data = {k: (v.item() if hasattr(v, 'item') else v) for k, v in cust_data.items()}
                tool_output_str = f"Entity Lookup for {cust_id}:\n{json.dumps(clean_data, indent=2)}"
                
        elif intent == 'entity_list':
            filter_col = params.get('filter_col', 'AvgAccountBalance')
            limit = params.get('limit', 10)
            min_val = params.get('min_val')
            max_val = params.get('max_val')
            sort_by = params.get('sort_by')
            # time_window_note may come from router (Gemini set it) OR detected below from query text
            router_time_note = params.get('time_window_note', False)

            # ── Bug 2: Detect time-window phrases and prepare disclaimer ────────────
            # The dataset's Frequency column is an ALL-TIME count, not a per-period count.
            # If the user asked for a time window, we must say so explicitly.
            # Priority: use flag from router params if Gemini already detected it;
            # otherwise fall back to local phrase scanning.
            time_window_phrases = [
                "last 90 days", "last 60 days", "last 30 days", "past 90 days",
                "past 60 days", "past 30 days", "in the last", "in the past",
                "over the past", "within the last", "recent 90", "recent 60"
            ]
            time_window_note = ""
            if router_time_note:
                matched_phrase = next(
                    (p for p in time_window_phrases if p in combined_query.lower()), "time window"
                )
                time_window_note = (
                    f"NOTE: This dataset does not contain per-transaction dates at the "
                    f"customer-feature level. The Frequency column reflects all-time transaction "
                    f"counts, not a '{matched_phrase}' window. Results below reflect all-time frequency."
                )
            else:
                for phrase in time_window_phrases:
                    if phrase in combined_query.lower():
                        time_window_note = (
                            f"NOTE: This dataset does not contain per-transaction dates at the "
                            f"customer-feature level. The Frequency column reflects all-time transaction "
                            f"counts, not a '{phrase}' window. Results below reflect all-time frequency."
                        )
                        break

            filtered_df = df_seg.copy()
            # Apply filter on the correct column
            if filter_col in filtered_df.columns:
                if min_val is not None:
                    filtered_df = filtered_df[filtered_df[filter_col] >= min_val]
                if max_val is not None:
                    filtered_df = filtered_df[filtered_df[filter_col] <= max_val]
            else:
                # Fall back to AvgAccountBalance if column not found
                if min_val is not None:
                    filtered_df = filtered_df[filtered_df['AvgAccountBalance'] >= min_val]
                if max_val is not None:
                    filtered_df = filtered_df[filtered_df['AvgAccountBalance'] <= max_val]

            if sort_by and sort_by in filtered_df.columns:
                filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)

            output_cols = [c for c in ['CustomerID', 'AvgAccountBalance', 'Recency', 'Frequency', 'Cluster'] if c in filtered_df.columns]
            results = filtered_df.head(limit)[output_cols].to_dict(orient='records')
            # Convert numpy types to native Python
            results = [{k: (v.item() if hasattr(v, 'item') else v) for k, v in r.items()} for r in results]
            print(f"[Tool: list_tool] done")
            tool_output_str = (
                f"Entity List Results:\n"
                f"  Filter: {filter_col} in [{min_val}, {max_val}]\n"
                f"  Total Matches: {len(filtered_df):,}\n"
                f"  Sorted by: {sort_by}\n"
                + (f"  Time Window Limitation: {time_window_note}\n" if time_window_note else "")
                + f"  List:\n{json.dumps(results, indent=2)}"
            )
            
        elif intent == 'eda':
            csv_path = "data/customers.csv"
            if not os.path.exists(csv_path):
                tool_output_str = "Error: raw customer transaction database not found under data/customers.csv."
            else:
                df_raw = pd.read_csv(csv_path)
                eda_results = run_eda(df_raw)
                print(f"[Tool: eda_tool] done")
                tool_output_str = f"EDA Summary Statistics:\n{json.dumps(eda_results, indent=2)}"
        
    # 3. Synthesis Step
    print("[Synthesizer] Drafting decisive response...")
    final_answer = synthesize_answer(combined_query, tool_output_str)
    return final_answer

def main():
    features_path = "data/customer_features.csv"
    if not os.path.exists(features_path):
        print(f"Error: {features_path} not found. Please run test_tools.py first.")
        sys.exit(1)
        
    # Load dataset
    df_feats = pd.read_csv(features_path)
    
    # Session context shared across all calls in this run
    session = {}
    
    queries = [
        # Original 4 example queries
        "Segment customers into priority, regular and dormant based on balance and transaction frequency",
        "On what basis were priority customers selected?",
        "What is the average size of transactions for priority and regular customers?",
        "Which regular customers can be converted to priority customers?",
        
        # Single-entity lookup query
        "Is customer C6013525 priority or regular?",
        
        # 6 New evaluation queries
        "How many customers are in the dormant segment?",
        "What's the balance of customer C1010011?",
        "Compare average balance between priority and dormant customers",
        "List the top 10 highest-balance customers overall",
        "What percentage of customers are considered priority?",
        "Show me customers with balance between 90,000 and 100,000",
        
        # EDA query
        "Run exploratory data analysis on the customer transaction base and show missing values"
    ]
    
    for q in queries:
        ans = execute_agent_loop(q, df_feats, session_context=session)
        print(f"\n[Agent Decisive Response]:\n{ans}\n")

if __name__ == "__main__":
    main()
