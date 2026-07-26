import pandas as pd
import numpy as np

def run_eda(df: pd.DataFrame) -> dict:
    """
    Perform Exploratory Data Analysis (EDA) on the bank customer dataset.
    
    Parameters:
    df (pd.DataFrame): Input bank transactions dataframe.
    
    Returns:
    dict: A dictionary containing summary statistics, null counts, distributions, and correlations.
    """
    results = {}
    
    # 1. Basic Metadata
    results['shape'] = df.shape
    results['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # 2. Null Value Analysis
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    results['missing_data'] = pd.DataFrame({
        'null_count': null_counts,
        'null_percentage': null_percentages
    }).to_dict(orient='index')
    
    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 3. Numerical Summary Statistics
    summary_stats = df[num_cols].describe().to_dict()
    results['summary_stats'] = summary_stats
    
    # 4. Categorical Distributions (Gender & Top Locations)
    distributions = {}
    
    # Gender distribution
    if 'CustGender' in df.columns:
        gender_counts = df['CustGender'].value_counts(dropna=False)
        gender_pct = df['CustGender'].value_counts(dropna=False, normalize=True) * 100
        distributions['CustGender'] = pd.DataFrame({
            'count': gender_counts,
            'percentage': gender_pct
        }).to_dict(orient='index')
        
    # Location distribution (Top 10)
    if 'CustLocation' in df.columns:
        loc_counts = df['CustLocation'].value_counts(dropna=False).head(10)
        loc_pct = df['CustLocation'].value_counts(dropna=False, normalize=True).head(10) * 100
        distributions['CustLocation_top10'] = pd.DataFrame({
            'count': loc_counts,
            'percentage': loc_pct
        }).to_dict(orient='index')
        
    results['categorical_distributions'] = distributions
    
    # 5. Correlation Analysis
    # Let's extract numeric columns. We compute correlation for actual numeric cols.
    correlation_matrix = df[num_cols].corr().to_dict()
    results['correlations'] = correlation_matrix
    
    # 6. Date Range Analysis (if columns exist)
    date_info = {}
    for date_col in ['TransactionDate', 'CustomerDOB']:
        if date_col in df.columns:
            # Drop nulls for date parsing test
            non_null_dates = df[date_col].dropna()
            if len(non_null_dates) > 0:
                try:
                    # Let's parse with format='%d/%m/%y' (Indian format often used in this dataset)
                    parsed_dates = pd.to_datetime(non_null_dates, format='%d/%m/%y', errors='coerce')
                    valid_count = parsed_dates.notnull().sum()
                    date_info[date_col] = {
                        'total_non_null': len(non_null_dates),
                        'parsed_successfully': int(valid_count),
                        'min_date': str(parsed_dates.min()) if valid_count > 0 else None,
                        'max_date': str(parsed_dates.max()) if valid_count > 0 else None
                    }
                except Exception as e:
                    date_info[date_col] = {'error': str(e)}
    results['date_analysis'] = date_info
    
    return results

if __name__ == "__main__":
    import sys
    import json
    
    # Simple CLI check
    csv_path = "data/customers.csv"
    print(f"Loading dataset from {csv_path}...")
    try:
        # Load a sample or full depending on size
        df = pd.read_csv(csv_path)
        print("Running EDA...")
        eda_res = run_eda(df)
        
        # Pretty print results
        print("\n=== DATASET SHAPE ===")
        print(f"Rows: {eda_res['shape'][0]}, Columns: {eda_res['shape'][1]}")
        
        print("\n=== MISSING DATA ===")
        for col, val in eda_res['missing_data'].items():
            print(f"{col}: {val['null_count']} nulls ({val['null_percentage']:.4f}%)")
            
        print("\n=== NUMERICAL SUMMARY STATS ===")
        for col, stats in eda_res['summary_stats'].items():
            print(f"\nColumn: {col}")
            for stat_name, stat_val in stats.items():
                print(f"  {stat_name}: {stat_val:.2f}")
                
        print("\n=== GENDER DISTRIBUTION ===")
        if 'CustGender' in eda_res['categorical_distributions']:
            for val, counts in eda_res['categorical_distributions']['CustGender'].items():
                print(f"  {val}: {counts['count']} ({counts['percentage']:.2f}%)")
                
        print("\n=== TOP 5 LOCATIONS ===")
        if 'CustLocation_top10' in eda_res['categorical_distributions']:
            for i, (val, counts) in enumerate(eda_res['categorical_distributions']['CustLocation_top10'].items()):
                if i < 5:
                    print(f"  {val}: {counts['count']} ({counts['percentage']:.2f}%)")
                    
        print("\n=== DATE COLUMNS ANALYSIS ===")
        for col, info in eda_res['date_analysis'].items():
            print(f"Column: {col}")
            if 'error' in info:
                print(f"  Error: {info['error']}")
            else:
                print(f"  Total Non-Null: {info['total_non_null']}")
                print(f"  Successfully Parsed: {info['parsed_successfully']}")
                print(f"  Min Date: {info['min_date']}")
                print(f"  Max Date: {info['max_date']}")
                
        print("\n=== CORRELATION MATRIX ===")
        for col, corrs in eda_res['correlations'].items():
            print(f"  Correlations for {col}:")
            for other_col, val in corrs.items():
                print(f"    {other_col}: {val:.4f}")
                
    except Exception as e:
        print(f"Error during EDA run: {e}", file=sys.stderr)
        sys.exit(1)
