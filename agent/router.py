import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it in your environment or .env file.")

genai.configure(api_key=api_key)

# Keywords that indicate a query is banking-related (used by post-parse guard & heuristic fallback)
BANKING_KEYWORDS = [
    "segment", "cluster", "group", "basis", "why", "explain", "rules",
    "convert", "upgrade", "recommend", "offer", "campaign",
    "average", "mean", "sum", "total", "count", "percentage", "compare",
    "list", "show me", "top", "between", "balance", "transaction",
    "recency", "frequency", "customer", "eda", "missing", "null",
    "correlation", "distribution", "summary", "explore", "analysis",
    "dormant", "priority", "regular", "recompute", "recalculate",
    "never", "inactive", "portfolio", "how many", "which customers",
    "filter", "find customers"
]

# Keywords that conclusively indicate out-of-scope (checked AFTER banking_keywords)
OUT_OF_SCOPE_KEYWORDS = [
    "pizza", "weather", "joke", "trivia", "capital of", "president",
    "sky", "recipe", "javascript", "game", "football", "cricket",
    "who is", "what is the best", "tell me a", "tell me about",
    "movie", "song", "sport", "food", "animal", "geography"
]


def _is_out_of_scope(query: str) -> bool:
    """Heuristic check: returns True if query is definitively not banking-related."""
    q_lower = query.lower()
    has_banking = any(k in q_lower for k in BANKING_KEYWORDS) or bool(re.search(r'C\d+', query, re.IGNORECASE))
    has_oos = any(k in q_lower for k in OUT_OF_SCOPE_KEYWORDS)
    # Explicitly out-of-scope OR has no banking keywords at all AND is not a customer ID lookup
    return has_oos or not has_banking


def route_query(query: str) -> dict:
    """
    Use Gemini 2.5 Flash to parse a user query and extract intent, parameters, or clarification question.
    
    Parameters:
    query (str): User natural language query.
    
    Returns:
    dict: Parsed router plan containing 'intent', 'parameters', and 'clarification_question'.
    """
    prompt = f"""
You are a routing agent for a Bank Customer Segmentation and Personalization system.
Your job is to analyze the user's natural language query and decide which tool/intent it maps to.
Return your decision as a valid JSON object ONLY. Do not write markdown blocks (no ```json or similar tags), just the raw JSON text.

Available Intents and Tools:

1. intent: "segmentation"
   - Match this when the user wants to run or re-run the full clustering/segmentation pipeline to group the customer base.
   - Parameters:
     - "use_kmeans": boolean (default true). Set to false if they mention "rules", "rule-based", "fallback", "thresholds", or if they want to segment by explicit rules like "based on balance and transaction frequency thresholds".
     - "k": integer (optional). Number of clusters if specified.
     - "balance_threshold": float (optional). Custom balance threshold for priority rules (default: 100000.0).
     - "recency_threshold": integer (optional). Custom recency threshold in days for priority rules (default: 30).
   - EXCEPTION: If the query asks to segment/group customers but does not specify any concrete attributes to group them by (e.g., "segment customers"), you MUST classify this as clarification_needed instead of guessing.

2. intent: "explain_rule"
   - Match this when the user asks "why", "on what basis", "rules of selection", "how were they selected", or wants to know the rules/profiles behind clusters.
   - Parameters: None.

3. intent: "aggregate_stat"
   - Match this when the user asks for statistical calculations across segments or the full base.
   - Parameters:
     - "group_col": string (default "Cluster").
     - "agg_col": string. Choose one of: "MonetaryAvg" (for transaction sizes), "AvgAccountBalance" (for account balances), "Frequency" (for transaction counts).
     - "agg_func": string. Choose one of: "mean", "count", "percentage", "sum".
       - Use "sum" for: "total", "sum of", "combined", "aggregate balance", "total portfolio balance", "total held by".
       - Use "count" for: "how many", "number of", "count".
       - Use "percentage" for: "what percent", "what share", "proportion".
       - Use "mean" for: "average", "typical", "mean", "compare".

4. intent: "entity_lookup"
   - Match this when the user wants to look up a specific customer's segment or attributes by their ID (e.g. C1010011).
   - Parameters:
     - "customer_id": string.

5. intent: "entity_list"
   - Match this when the user wants to LIST, FILTER, or FIND multiple customers based on a condition on any column.
   - CRITICAL COLUMN DISAMBIGUATION RULES:
     - IF user says "zero transactions", "never transacted", "no transaction history", "no transactions", "never made a transaction", "zero frequency", "frequency is 0" -> filter_col MUST BE "Frequency", max_val: 0
     - IF user says "zero balance", "empty account", "no balance", "balance is 0", "zero account balance" -> filter_col MUST BE "AvgAccountBalance", max_val: 0
     - DO NOT conflate these! Using AvgAccountBalance for a zero-transaction query is a critical error.
   - Parameters:
     - "filter_col": string. Column to filter on ("Frequency", "Recency", "AvgAccountBalance").
     - "min_val": float (optional). Minimum value.
     - "max_val": float (optional). Maximum value.
     - "sort_by": string (optional). Column to sort results by.
     - "limit": integer (optional, default 10).
     - "time_window_note": boolean (optional, default false). MUST be true if the query mentions a specific time window limitation (e.g., "in the last 90 days", "past 30 days").

6. intent: "recommendation"
   - Match this when the user asks for recommendations, product offers, or marketing campaigns.
   - If the user describes a HYPOTHETICAL customer, extract their exact parameters.
   - Parameters:
     - "segment": string (optional).
     - "hypothetical_balance": float (optional). Exact numerical balance described (e.g. "balance 45000" -> 45000.0).
     - "hypothetical_frequency": float (optional). Exact numerical frequency described (e.g. "frequency 0.2" -> 0.2).

7. intent: "conversion_candidates"
   - Match this when the user asks for regular customers close to crossing a segment threshold who can be converted/upgraded.
   - Parameters:
     - "threshold_pct": float (default 0.10).

8. intent: "eda"
   - Match this when the user wants to run exploratory data analysis (EDA) or show missing values.
   - Parameters: None.

9. intent: "out_of_scope"
   - Match this if the query has absolutely NO relation to banking customer data, transaction patterns, segments, or campaigns.
   - E.g. weather, jokes, trivia, recipes, sports, unrelated coding.
   - Parameters: None.

10. intent: "clarification_needed"
    - Match this ONLY if the query is genuinely ambiguous about WHAT the user wants (e.g. "segment customers" with no attributes).
    - If a query has a specific ask, do NOT fallback to clarification.
    - Parameters: None.
    - "clarification_question": A polite question to clarify their need.

JSON Output Structure:
{{
  "intent": "segmentation" | "explain_rule" | "aggregate_stat" | "entity_lookup" | "entity_list" | "recommendation" | "conversion_candidates" | "eda" | "out_of_scope" | "clarification_needed",
  "parameters": {{ ... }},
  "clarification_question": "string or null"
}}

User Query: "{query}"
JSON Response:
"""

    model_name = 'models/gemini-2.5-flash'
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "response_mime_type": "application/json"
        }
    )
    
    try:
        t0 = time.time()
        print(f"[GEMINI CALL] model={model_name}, prompt_preview={prompt[:150].replace(chr(10), ' ')}")
        response = model.generate_content(prompt)
        elapsed = time.time() - t0
        print(f"[GEMINI CALL] response_preview={response.text[:150].replace(chr(10), ' ')}")
        print(f"[GEMINI CALL] latency={elapsed:.2f}s")
        text = response.text.strip()
        
        # Clean any accidental markdown fence output from Gemini if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                text = "\n".join(lines[1:-1])
        
        parsed_plan = json.loads(text)
        
        # Enforce basic schema presence
        if 'intent' not in parsed_plan:
            parsed_plan['intent'] = 'clarification_needed'
        if 'parameters' not in parsed_plan:
            parsed_plan['parameters'] = {}
        if 'clarification_question' not in parsed_plan:
            parsed_plan['clarification_question'] = None

        # ── Post-parse guard 1: Out-of-scope enforcement (Issue 1 fix) ──────────
        # Even if Gemini returned a valid banking intent, override if the query
        # is definitively out of scope. This catches cases where Gemini misroutes.
        if parsed_plan.get('intent') != 'out_of_scope' and _is_out_of_scope(query):
            parsed_plan['intent'] = 'out_of_scope'
            parsed_plan['parameters'] = {}
            parsed_plan['clarification_question'] = None

        # ── Post-parse guard 2: Vague segmentation → clarification (existing) ──
        if parsed_plan.get('intent') == 'segmentation':
            q_lower = query.lower()
            attributes = ["balance", "frequency", "transaction", "recency", "tenure", "age", "gender", "location", "amount", "monetary"]
            has_attribute = any(attr in q_lower for attr in attributes)
            if not has_attribute or q_lower.strip() in ["segment customers", "segment", "group customers", "run segmentation", "run clustering"]:
                parsed_plan['intent'] = 'clarification_needed'
                parsed_plan['clarification_question'] = "Which attributes should I segment on — balance, transaction frequency, tenure, or a combination?"
                
        return parsed_plan
        
    except Exception as e:
        # ── Heuristic fallback: safety net ONLY for API/parse failures ──────────
        # This is NOT the primary classification path. Do NOT add keyword-specific
        # triggers here to patch gaps — fix the Gemini prompt instead.
        print(f"[Router Error] {e}")
        q_lower = query.lower()
        
        # 0. Out-of-scope check
        if _is_out_of_scope(query):
            return {"intent": "out_of_scope", "parameters": {}, "clarification_question": None}

        # 1. Regex check for customer ID (entity_lookup)
        id_match = re.search(r'C\d+', query, re.IGNORECASE)
        if id_match:
            cust_id = id_match.group(0).upper()
            return {
                "intent": "entity_lookup",
                "parameters": {"customer_id": cust_id},
                "clarification_question": None
            }
            
        # 2. Check for EDA
        if any(k in q_lower for k in ["eda", "missing", "null", "correlation", "distribution", "summary of data", "explore", "data analysis"]):
            return {"intent": "eda", "parameters": {}, "clarification_question": None}

        # 3. Check for statistical aggregation
        if any(k in q_lower for k in ["average", "mean", "sum", "total", "how many", "percentage", "compare", "count"]):
            agg_func = "mean"
            if "how many" in q_lower or "count" in q_lower:
                agg_func = "count"
            elif "percentage" in q_lower or "percent" in q_lower:
                agg_func = "percentage"
            elif "total" in q_lower or "sum" in q_lower:
                agg_func = "sum"
                
            agg_col = "MonetaryAvg"
            if "balance" in q_lower:
                agg_col = "AvgAccountBalance"
            elif "frequency" in q_lower or "transaction" in q_lower:
                agg_col = "Frequency"
                
            return {
                "intent": "aggregate_stat",
                "parameters": {
                    "group_col": "RuleBasedSegment" if any(k in q_lower for k in ["dormant", "priority", "regular"]) else "Cluster",
                    "agg_col": agg_col,
                    "agg_func": agg_func
                },
                "clarification_question": None
            }

        # 4. Check for list filters
        if any(k in q_lower for k in ["list", "show me customers", "top", "between", "which customers", "find customers", "filter"]):
            limit = 10
            limit_match = re.search(r'top\s+(\d+)', q_lower)
            if limit_match:
                limit = int(limit_match.group(1))

            # Safety net for frequency-zero queries (when Gemini API is unavailable).
            # Note: uses substring matching; covers both "zero transaction" and "zero transactions".
            if any(k in q_lower for k in ["never", "zero transaction", "no transaction", "frequency 0", "frequency of 0", "no transactions", "never transacted", "not transacted"]):
                return {
                    "intent": "entity_list",
                    "parameters": {
                        "filter_col": "Frequency",
                        "min_val": None,
                        "max_val": 0,
                        "sort_by": "AvgAccountBalance",
                        "limit": limit,
                        "time_window_note": any(p in q_lower for p in ["last 90", "last 60", "last 30", "past 90", "past 60", "past 30", "in the last", "in the past"])
                    },
                    "clarification_question": None
                }

            # Safety net for balance-zero queries — distinct from frequency-zero.
            if any(k in q_lower for k in ["zero balance", "empty account", "balance is 0", "no balance"]):
                return {
                    "intent": "entity_list",
                    "parameters": {
                        "filter_col": "AvgAccountBalance",
                        "min_val": None,
                        "max_val": 0,
                        "sort_by": "AvgAccountBalance",
                        "limit": limit
                    },
                    "clarification_question": None
                }

            min_val = None
            max_val = None
            between_match = re.search(r'between\s+([\d,]+)\s+and\s+([\d,]+)', q_lower)
            if between_match:
                min_val = float(between_match.group(1).replace(',', ''))
                max_val = float(between_match.group(2).replace(',', ''))

            return {
                "intent": "entity_list",
                "parameters": {
                    "filter_col": "AvgAccountBalance",
                    "min_val": min_val,
                    "max_val": max_val,
                    "sort_by": "AvgAccountBalance",
                    "limit": limit
                },
                "clarification_question": None
            }
            
        # 5. Segmentation with vague check and parameter overrides
        if any(k in q_lower for k in ["segment", "cluster", "group"]):
            attributes = ["balance", "frequency", "transaction", "recency", "tenure", "age", "gender", "location", "amount", "monetary"]
            has_attribute = any(attr in q_lower for attr in attributes)
            if not has_attribute or q_lower.strip() in ["segment customers", "segment", "group customers", "run segmentation", "run clustering"]:
                return {
                    "intent": "clarification_needed",
                    "parameters": {},
                    "clarification_question": "Which attributes should I segment on — balance, transaction frequency, tenure, or a combination?"
                }
            
            use_kmeans = "rule" not in q_lower and "threshold" not in q_lower and "recompute" not in q_lower
            balance_val = None
            recency_val = None
            
            balance_match = re.search(r'balance\s*(?:>|greater\s+than|above)?\s*([\d,]+)', q_lower)
            if balance_match:
                try:
                    balance_val = float(balance_match.group(1).replace(',', ''))
                except ValueError:
                    pass
                    
            recency_match = re.search(r'recency\s*(?:<=|less\s+than|under|below)?\s*([\d,]+)', q_lower)
            if recency_match:
                try:
                    recency_val = int(recency_match.group(1).replace(',', ''))
                except ValueError:
                    pass
                    
            params = {"use_kmeans": use_kmeans}
            if balance_val is not None:
                params["balance_threshold"] = balance_val
                params["use_kmeans"] = False
            if recency_val is not None:
                params["recency_threshold"] = recency_val
                params["use_kmeans"] = False
                
            return {"intent": "segmentation", "parameters": params, "clarification_question": None}
            
        elif any(k in q_lower for k in ["basis", "why", "explain", "rules"]):
            return {"intent": "explain_rule", "parameters": {}, "clarification_question": None}
        elif any(k in q_lower for k in ["convert", "upgrade"]):
            return {"intent": "conversion_candidates", "parameters": {"threshold_pct": 0.10}, "clarification_question": None}
        elif any(k in q_lower for k in ["recommend", "offer", "campaign"]):
            # Try to extract hypothetical customer params
            params = {}
            bal_match = re.search(r'balance\s+(?:of\s+)?([0-9,]+)', q_lower)
            freq_match = re.search(r'frequency\s+(?:of\s+)?([\d.]+)', q_lower)
            if bal_match:
                try:
                    params["hypothetical_balance"] = float(bal_match.group(1).replace(',', ''))
                except ValueError:
                    pass
            if freq_match:
                try:
                    params["hypothetical_frequency"] = float(freq_match.group(1))
                except ValueError:
                    pass
            return {"intent": "recommendation", "parameters": params, "clarification_question": None}
        else:
            return {
                "intent": "clarification_needed",
                "parameters": {},
                "clarification_question": "I am not sure what you want. Would you like to segment customers, explain segment rules, get campaign recommendations, or look up specific customer balances?"
            }

if __name__ == "__main__":
    # Test cases
    queries = [
        "Segment customers into priority, regular and dormant based on balance and transaction frequency",
        "On what basis were priority customers selected?",
        "Which regular customers can be converted to priority customers?",
        "Is customer C6013525 priority or regular?",
        "What's the balance of customer C1010011?",
        "Compare average balance between priority and dormant customers",
        "List the top 10 highest-balance customers overall",
        "What is the weather today?",
        "Tell me a joke",
        "What's the best pizza topping?",
        "Which customers have never made a transaction?",
        "What's the total portfolio balance held by dormant customers?",
        "Recommend a product for a customer with balance 45,000 and frequency 0.2",
    ]
    
    print("Testing router...")
    for q in queries:
        print(f"\nQuery: {q}")
        plan = route_query(q)
        print(f"Plan: {json.dumps(plan, indent=2)}")
