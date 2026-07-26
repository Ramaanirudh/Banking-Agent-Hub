import pandas as pd
import sys
import os

# Ensure tools directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))

from eda_tool import run_eda
from feature_engineering_tool import engineer_features

def main():
    csv_path = "data/customers.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please ensure the Kaggle bank customer segmentation dataset is downloaded and saved to that location.")
        sys.exit(1)
        
    print(f"==================================================")
    print(f"1. Loading Real Dataset from: {csv_path}")
    print(f"==================================================")
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded {len(df):,} transaction rows.")
    
    print(f"\n==================================================")
    print(f"2. Running EDA Tool")
    print(f"==================================================")
    eda_results = run_eda(df)
    
    # Print key insights from EDA
    print(f"Dataset Shape: {eda_results['shape'][0]} rows, {eda_results['shape'][1]} columns")
    print("\nMissing values per column:")
    for col, data in eda_results['missing_data'].items():
        print(f"  {col}: {data['null_count']} nulls ({data['null_percentage']:.2f}%)")
        
    print("\nSummary statistics of numerical columns:")
    for col, stats in eda_results['summary_stats'].items():
        print(f"  {col}: Mean = {stats['mean']:.2f}, Std = {stats['std']:.2f}, Min = {stats['min']:.2f}, Max = {stats['max']:.2f}")
        
    print("\nGender distribution:")
    if 'CustGender' in eda_results['categorical_distributions']:
        for val, counts in eda_results['categorical_distributions']['CustGender'].items():
            # handle NaN gender
            val_str = str(val) if pd.notnull(val) else "Missing"
            print(f"  {val_str}: {counts['count']} ({counts['percentage']:.2f}%)")
            
    print("\nDate Column Ranges:")
    for col, info in eda_results['date_analysis'].items():
        print(f"  {col}: Min = {info.get('min_date')}, Max = {info.get('max_date')}, Parsed = {info.get('parsed_successfully')}/{info.get('total_non_null')}")
        
    print(f"\n==================================================")
    print(f"3. Running Feature Engineering Tool")
    print(f"==================================================")
    print("Calculating RFM-style features, max monthly balances, transaction frequencies, and cleaning demographics...")
    features_df = engineer_features(df)
    
    print(f"Successfully engineered customer-level features.")
    print(f"Engineered features shape: {features_df.shape[0]} rows (unique customers), {features_df.shape[1]} columns")
    
    print("\nSample of engineered features:")
    print(features_df.head(5).to_string(index=False))
    
    print("\nSummary statistics of engineered customer features:")
    print(features_df.describe().T[['mean', 'std', 'min', '50%', 'max']].to_string())
    
    # Save the engineered features to a file
    output_path = "data/customer_features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"\nSaved engineered features to: {output_path}")
    print(f"==================================================")
    print("Verification Completed Successfully!")
    print(f"==================================================")

if __name__ == "__main__":
    main()
