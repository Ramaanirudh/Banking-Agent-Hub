import os
import sys
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agent')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))

from test_agent import execute_agent_loop
from recommendation_tool import map_clusters_to_segments
# Define the ~20 queries we've tested throughout the project

# Define the ~20 queries we've tested throughout the project
TEST_CASES = [
    {
        "query": "Segment customers into priority, regular and dormant based on balance and transaction frequency",
        "expected_intent": "segmentation",
        "validators": [
            lambda p, t, a: p['intent'] == 'segmentation',
            lambda p, t, a: "KMeans" in a or "Rule-Based" in a,
            lambda p, t, a: p['parameters'].get('use_kmeans', True) is True
        ]
    },
    {
        "query": "On what basis were priority customers selected?",
        "expected_intent": "explain_rule",
        "validators": [
            lambda p, t, a: p['intent'] == 'explain_rule',
            lambda p, t, a: "Accuracy" in a or "99.88%" in a or "Surrogate" in t
        ]
    },
    {
        "query": "Which regular customers can be converted to priority customers?",
        "expected_intent": "conversion_candidates",
        "validators": [
            lambda p, t, a: p['intent'] == 'conversion_candidates',
            lambda p, t, a: "Conversion Candidates" in t or "conversion" in a.lower()
        ]
    },
    {
        "query": "Is customer C6013525 priority or regular?",
        "expected_intent": "entity_lookup",
        "validators": [
            lambda p, t, a: p['intent'] == 'entity_lookup',
            lambda p, t, a: p['parameters'].get('customer_id') == 'C6013525'
        ]
    },
    {
        "query": "What's the balance of customer C1010011?",
        "expected_intent": "entity_lookup",
        "validators": [
            lambda p, t, a: p['intent'] == 'entity_lookup',
            lambda p, t, a: p['parameters'].get('customer_id') == 'C1010011'
        ]
    },
    {
        "query": "Compare average balance between priority and dormant customers",
        "expected_intent": "aggregate_stat",
        "validators": [
            lambda p, t, a: p['intent'] == 'aggregate_stat',
            lambda p, t, a: p['parameters'].get('agg_func') == 'mean',
            lambda p, t, a: p['parameters'].get('agg_col') == 'AvgAccountBalance'
        ]
    },
    {
        "query": "List the top 10 highest-balance customers overall",
        "expected_intent": "entity_list",
        "validators": [
            lambda p, t, a: p['intent'] == 'entity_list',
            lambda p, t, a: p['parameters'].get('sort_by') == 'AvgAccountBalance' or 'sort_by' not in p['parameters']
        ]
    },
    {
        "query": "What is the weather today?",
        "expected_intent": "out_of_scope",
        "validators": [
            lambda p, t, a: p['intent'] == 'out_of_scope'
        ]
    },
    {
        "query": "Tell me a joke",
        "expected_intent": "out_of_scope",
        "validators": [
            lambda p, t, a: p['intent'] == 'out_of_scope'
        ]
    },
    {
        "query": "What's the best pizza topping?",
        "expected_intent": "out_of_scope",
        "validators": [
            lambda p, t, a: p['intent'] == 'out_of_scope'
        ]
    },
    {
        "query": "Which customers have never made a transaction?",
        "expected_intent": "entity_list",
        "validators": [
            lambda p, t, a: p['intent'] == 'entity_list',
            lambda p, t, a: p['parameters'].get('filter_col') == 'Frequency',
            lambda p, t, a: p['parameters'].get('max_val') == 0
        ]
    },
    {
        "query": "What's the total portfolio balance held by dormant customers?",
        "expected_intent": "aggregate_stat",
        "validators": [
            lambda p, t, a: p['intent'] == 'aggregate_stat',
            lambda p, t, a: p['parameters'].get('agg_func') == 'sum',
            lambda p, t, a: p['parameters'].get('agg_col') == 'AvgAccountBalance'
        ]
    },
    {
        "query": "Recommend a product for a customer with balance 45,000 and frequency 0.2",
        "expected_intent": "recommendation",
        "validators": [
            lambda p, t, a: p['intent'] == 'recommendation',
            lambda p, t, a: p['parameters'].get('hypothetical_balance') == 45000,
            lambda p, t, a: p['parameters'].get('hypothetical_frequency') == 0.2
        ]
    },
    {
        "query": "How many customers are in the dormant segment?",
        "expected_intent": "aggregate_stat",
        "validators": [
            lambda p, t, a: p['intent'] == 'aggregate_stat',
            lambda p, t, a: p['parameters'].get('agg_func') == 'count'
        ]
    },
    {
        "query": "What percentage of customers are considered priority?",
        "expected_intent": "aggregate_stat",
        "validators": [
            lambda p, t, a: p['intent'] == 'aggregate_stat',
            lambda p, t, a: p['parameters'].get('agg_func') == 'percentage'
        ]
    },
    {
        "query": "Show me customers with balance between 90,000 and 100,000",
        "expected_intent": "entity_list",
        "validators": [
            lambda p, t, a: p['intent'] == 'entity_list',
            lambda p, t, a: p['parameters'].get('filter_col') == 'AvgAccountBalance',
            lambda p, t, a: p['parameters'].get('min_val') == 90000,
            lambda p, t, a: p['parameters'].get('max_val') == 100000
        ]
    },
    {
        "query": "Run exploratory data analysis on the customer transaction base and show missing values",
        "expected_intent": "eda",
        "validators": [
            lambda p, t, a: p['intent'] == 'eda'
        ]
    },
    {
        "query": "segment customers",
        "expected_intent": "clarification_needed",
        "validators": [
            lambda p, t, a: p['intent'] == 'clarification_needed'
        ]
    },
    {
        "query": "balance",
        "session_carryover": "segment customers", # simulate clarification flow
        "expected_intent": "segmentation",
        "validators": [
            lambda p, t, a: p['intent'] == 'segmentation'
        ]
    },
    {
        "query": "List customers with zero transactions in the last 90 days",
        "expected_intent": "entity_list",
        "validators": [
            lambda p, t, a: p['intent'] == 'entity_list',
            lambda p, t, a: p['parameters'].get('filter_col') == 'Frequency',
            lambda p, t, a: p['parameters'].get('max_val') == 0,
            lambda p, t, a: p['parameters'].get('time_window_note') is True or "all-time" in a.lower() or "disclaimer" in t.lower()
        ]
    }
]

def main():
    features_path = "data/customer_features.csv"
    if not os.path.exists(features_path):
        print("Data not found. Exiting.")
        sys.exit(1)
        
    df_feats = pd.read_csv(features_path)
    
    # We will patch route_query slightly so we can extract 'plan' and 'tool_output'
    # Actually, execute_agent_loop doesn't return plan and tool_output, only final answer.
    # We will mock the test by calling router.route_query directly for plan validations.
    import agent.router as router
    import test_agent
    
    results = []
    
    print(f"Running {len(TEST_CASES)} tests...")
    
    for i, test in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] Testing: {test['query']}")
        
        # 1. Routing validation
        session = {}
        if "session_carryover" in test:
            session["pending_query"] = test["session_carryover"]
            
        combined_query = test["query"]
        if session.get("pending_query"):
            combined_query = f"{session['pending_query']} based on {test['query']}"
            
        plan = router.route_query(combined_query)
        
        # We also want to execute the agent to get the final answer and tool output.
        # Let's run execute_agent_loop but we'll need to capture the stdout to get tool_output or just 
        # test the final answer.
        # For full structural testing, let's capture the stdout or run it cleanly.
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            session_for_agent = {}
            if "session_carryover" in test:
                session_for_agent["pending_query"] = test["session_carryover"]
            answer = execute_agent_loop(test["query"], df_feats, session_context=session_for_agent)
            
        stdout_text = f.getvalue()
        
        # Extract tool output from stdout if possible
        # Alternatively, we just pass stdout_text as 'tool_output' for loose validation
        
        passes_all = True
        for v_idx, validator in enumerate(test['validators']):
            try:
                res = validator(plan, stdout_text, answer)
                if not res:
                    print(f"  ❌ Validator {v_idx+1} failed. Plan: {plan}")
                    passes_all = False
            except Exception as e:
                print(f"  ❌ Validator {v_idx+1} raised exception: {e}")
                passes_all = False
                
        results.append({
            "query": test["query"],
            "expected": test["expected_intent"],
            "actual": plan.get("intent"),
            "pass": passes_all
        })
        
    print("\n\n" + "="*80)
    print("TEST SUITE RESULTS")
    print("="*80)
    print(f"{'PASS/FAIL':<10} | {'EXPECTED INTENT':<22} | {'ACTUAL INTENT':<22} | QUERY")
    print("-" * 80)
    
    total_passed = 0
    for r in results:
        status = "[PASS]" if r["pass"] else "[FAIL]"
        if r["pass"]:
            total_passed += 1
        print(f"{status:<10} | {r['expected']:<22} | {r['actual']:<22} | {r['query'][:40]}...")
        
    print(f"\nFinal Score: {total_passed} / {len(TEST_CASES)} passed.")
    if total_passed < len(TEST_CASES):
        sys.exit(1)

if __name__ == "__main__":
    main()
