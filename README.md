# Bank Customer Segmentation & Personalization Agent

An agentic AI and machine learning pipeline for retail banking customer segmentation, segment explainability, and personalized marketing recommendations. Built for a 48-hour hackathon.

## 🔗 Live App
[Streamlit App Link](https://your-app-name.streamlit.app) -- placeholder, to be updated

---

## 📋 Problem Statement

In retail banking, understanding customer profiles and behavior is key to driving targeted marketing campaigns, preventing churn, and identifying upsell opportunities. 

This project implements **Problem Statement 2: Customer Segmentation & Personalization Agent (bank customer data)**. It aggregates transaction-level bank data to customer-level metrics, applies machine learning clustering (KMeans) to classify customers, fits a surrogate explainability model (Decision Tree) to extract human-readable segment rules, maps segments to personalized campaign offers, and wraps the entire logic in an LLM-driven chat agent interface.

---

## 📊 Dataset Source & Citation

* **Dataset Name**: Bank Customer Segmentation (1 Million+ Transactions)
* **Author**: Shivam Bansal (Kaggle)
* **Dataset URL**: [https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation)
* **Details**: Contains transactional records from an Indian bank covering **1,048,567 transactions** for **884,265 unique customers** from August 2016 to October 2016.
* **Schema**:
  - `TransactionID` (Unique identifier for transaction)
  - `CustomerID` (Unique identifier for customer)
  - `CustomerDOB` (Date of birth, formatted DD/MM/YY)
  - `CustGender` (M/F)
  - `CustLocation` (City/State of register)
  - `CustAccountBalance` (Current account balance at transaction time)
  - `TransactionDate` (Date of transaction)
  - `TransactionTime` (HHMMSS formatted integer)
  - `TransactionAmount (INR)` (Transaction value in Indian Rupees)

---

## 🏗️ Architecture

The application is structured into modular pipeline layers:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           Streamlit Chat UI                               │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ User query
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      Agent Loop (Gemini Router)                           │
│ - Identifies intent: segmentation, explain_rule, aggregate_stat,          │
│   entity_lookup, entity_list, recommendation, conversion_candidates,      │
│   eda, out_of_scope, clarification_needed.                                │
│ - Creates a selective tool-execution plan (skips unnecessary tools).      │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Selective trigger
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           Execution Tools                                 │
│ 1. Segmentation Tool: Standardizes features, evaluates optimal K, runs    │
│    KMeans, and implements a deterministic fallback classifier.            │
│ 2. Explainability Tool: Compares group centroids and extracts human-      │
│    readable decision rules using a Decision Tree surrogate (depth=3).     │
│ 3. Recommendation Tool: Dynamic offerings mapping + near-priority finder. │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Tool outputs
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    Agent Loop (Gemini Synthesizer)                        │
│ - Merges query + tool data into a final, decisive response (no "it        │
│   depends" phrasing, clear list mapping, and exact numbers).              │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Decisive answer
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         Plots & Visualizations                            │
│ - Interactive 3D scatter plots and box distributions using Plotly.        │
└───────────────────────────────────────────────────────────────────────────┘
```

### Modular Components
1. **Feature Engineering**: Aggregates transaction data to customer-level metrics:
   - **Recency**: Days since last transaction relative to the maximum date in the dataset.
   - **Frequency**: Total transaction count per customer.
   - **Monetary (Total & Average)**: Sum and mean of `TransactionAmount (INR)`.
   - **Max Monthly Balance**: Groups transactions by customer and month, finds the max account balance, and averages it.
   - **TenureDays**: Duration of relationship from first to last transaction.
   - **TxFrequencyDaily**: Ratio of transactions to active tenure days.
   - **Cleaned Age**: Extracted from `CustomerDOB` and `TransactionDate`, correcting the 2-digit birth-year parser limits (adjusting future years like 2068 to 1968).
2. **KMeans Segmentation**: Standardizes the numerical profile and clusters customers. Evaluates $K$ using inertia elbow and sampled Silhouette analysis.
3. **Decision Tree Explainability**: Fits a shallow surrogate `DecisionTreeClassifier` (purity/support tracked) on the features using the cluster labels as target to output natural rules.
4. **Campaign Mapper**: Maps segments to specific products and extracts upsell targets who are active and hold balances within 10% below the Priority threshold (90k-100k INR).

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.13.3)
- Gemini API Key

### Installation

1. **Clone or navigate** to the project directory:
   ```bash
   git clone https://github.com/Ramaanirudh/Banking-Agent-Hub
   cd Banking-Agent-Hub
   ```

2. **Create and activate** a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and add your Gemini API Key:
     ```env
     GOOGLE_API_KEY=your_actual_gemini_api_key_here
     ```

5. **Place the dataset**:
   - Download the Kaggle dataset [here](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation).
   - Place the extracted CSV at `data/customers.csv`.

---

## 🚀 Usage Guide

### Step 1: Run Stage 1 Feature Engineering
Compile the transactional CSV into customer-level aggregated features:
```bash
python test_tools.py
```
This inspects the data, performs EDA statistics, and outputs the clean customer base to `data/customer_features.csv`.

### Step 2: Run Stage 2 Machine Learning Validation
Evaluate KMeans inertia, silhouette scores, centroid profiles, and decision tree surrogate rules:
```bash
python test_stage2.py
```
This verifies segmentation accuracy and saves interactive Plotly reports to the `reports/` folder.

### Step 3: Run the Regression Test Suite
Validate the LLM routing and synthesization loop across all 20 target queries (ensures intents, fallbacks, and validation layers work):
```bash
python test_suite.py
```

### Step 4: Launch the Streamlit Web Application
Run the chat interface in your browser:
```bash
streamlit run app.py
```
This opens the web interface at `http://localhost:8501`.

#### Sample Chat Queries to Try:
* *"Segment customers into priority, regular and dormant based on balance and transaction frequency"* (Triggers rule fallback/KMeans segmentation and displays Plotly distributions inline).
* *"On what basis were priority customers selected?"* (Triggers the explainability surrogate tree and centroid comparisons).
* *"Which regular customers can be converted to priority customers?"* (Triggers the upsell filter and displays a table of candidates with their balance gap details).

---

## 🔍 Note on Segmentation Method Divergence

During pipeline verification, a size discrepancy is observed between the two segmentation cohorts:
*   **KMeans Cluster 0 (Priority)**: 131,910 customers (14.92% of base)
*   **Rule-Based Priority**: 8,570 customers (0.97% of base)

### Why Do the Cohort Sizes Diverge?
The two approaches are designed to answer fundamentally different questions:
1.  **KMeans Clustering (Statistical Pattern Grouping)**: Groups customers based on multi-dimensional statistical patterns (similar RFM coordinates). It acts as a broad grouping identifying potential high-value clusters.
2.  **Rule-Based Filtering (Strict Business Policy)**: Applies hard constraints requiring both high value (`AvgAccountBalance > 100,000 INR`) **AND** high activity (`Recency <= 30 days`).

### Diagnostic Analysis (Why the Recency constraint dominates)
Our diagnostics show that the size gap is driven by a **recency bottleneck**:
*   **Balance Alone**: Filtering for `AvgAccountBalance > 100,000 INR` selects **157,161 customers** (17.77% of base). This represents a strong indicator since 100,000 INR sits between the 75th percentile (61,777.64 INR) and the 90th percentile (203,635.98 INR) of the customer base.
*   **The Inactivity Driver**: **94.55% (148,591) of these high-balance customers fail the recency condition** (having not transacted in the last 30 days).
*   **Overlap Validation**: Despite the size difference, the methods align logically: **32.24% (2,763)** of strict rule-based Priority customers are also grouped into KMeans Cluster 0. This is meaningfully above the expected random baseline of ~14.92% (KMeans Cluster 0's share of the full base, if the two splits were independent).

#### Balance Distribution Percentiles
*   **0th percentile (Min)**: 0.00 INR
*   **25th percentile**: 5,590.57 INR
*   **50th percentile (Median)**: 18,714.84 INR
*   **75th percentile**: 61,777.64 INR
*   **90th percentile**: 203,635.98 INR
*   **95th percentile**: 416,930.42 INR
*   **100th percentile (Max)**: 115,035,495.10 INR

---

## 🤖 AI Tools & Assistance Disclosed

- Antigravity (Google's agentic coding IDE) was used for code generation, debugging, and iterative development throughout this project.
- Google Gemini 2.5 Flash is used at runtime within the application for query intent routing and response synthesis (see `agent/router.py`, `agent/synthesizer.py`).

