import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer customer-level features from transaction-level data.
    
    Parameters:
    df (pd.DataFrame): Input bank transactions dataframe.
    
    Returns:
    pd.DataFrame: Customer-level aggregated dataframe with RFM and balance metrics.
    """
    # Create a copy to avoid SettingWithCopyWarning
    data = df.copy()
    
    # 1. Date Parsing
    # Parse TransactionDate: format is typically '%d/%m/%y'
    data['ParsedTxDate'] = pd.to_datetime(data['TransactionDate'], format='%d/%m/%y', errors='coerce')
    
    # Parse CustomerDOB: format is typically '%d/%m/%y'
    data['ParsedDOB'] = pd.to_datetime(data['CustomerDOB'], format='%d/%m/%y', errors='coerce')
    
    # Fix 2-digit DOB parsing issue where years in the future are parsed (e.g. 1968 parsed as 2068)
    # The transactions occur in 2016, so any DOB >= 2016 is likely a 20th century birth year.
    def fix_dob_year(dob):
        if pd.isnull(dob):
            return dob
        if dob.year >= 2016:
            try:
                return dob.replace(year=dob.year - 100)
            except ValueError:
                # Handle leap years (e.g., Feb 29)
                return dob.replace(year=dob.year - 100, day=dob.day - 1)
        return dob
        
    data['ParsedDOB'] = data['ParsedDOB'].apply(fix_dob_year)
    
    # Calculate customer age at the time of their first transaction (or maximum date)
    # We will average the age if there are multiple transactions, or calculate it based on first transaction
    data['AgeAtTx'] = (data['ParsedTxDate'] - data['ParsedDOB']).dt.days / 365.25
    # Remove unrealistic ages (e.g. age < 0 or age > 105)
    data.loc[(data['AgeAtTx'] < 0) | (data['AgeAtTx'] > 105), 'AgeAtTx'] = np.nan
    
    # Reference date for recency calculation (max transaction date in dataset + 1 day)
    ref_date = data['ParsedTxDate'].max() + pd.Timedelta(days=1)
    
    # 2. Extract Month-Year for Monthly Balance calculation
    data['TxYearMonth'] = data['ParsedTxDate'].dt.to_period('M')
    
    # 3. Aggregate Features at Customer Level
    # First, let's compute customer-level static/demographic attributes
    # We take the first occurrence or mode of demographics (Gender, Location, DOB-derived Age)
    customer_demographics = data.groupby('CustomerID').agg({
        'CustGender': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        'CustLocation': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        'AgeAtTx': 'mean' # average age across transactions
    })
    
    # Next, compute RFM and other transaction metrics
    # Note the column name: 'TransactionAmount (INR)'
    tx_amount_col = 'TransactionAmount (INR)'
    
    customer_tx_metrics = data.groupby('CustomerID').agg(
        # Recency: days since last transaction
        LastTxDate=('ParsedTxDate', 'max'),
        FirstTxDate=('ParsedTxDate', 'min'),
        # Frequency: count of transactions
        Frequency=('TransactionID', 'count'),
        # Monetary (Total): sum of transaction amounts
        MonetaryTotal=(tx_amount_col, 'sum'),
        # Monetary (Avg): average transaction amount
        MonetaryAvg=(tx_amount_col, 'mean'),
        # Max transaction amount
        MaxTxAmount=(tx_amount_col, 'max'),
        # Min transaction amount
        MinTxAmount=(tx_amount_col, 'min'),
        # Balance statistics across transactions
        AvgAccountBalance=('CustAccountBalance', 'mean'),
        MinAccountBalance=('CustAccountBalance', 'min'),
        MaxAccountBalance=('CustAccountBalance', 'max')
    )
    
    # Calculate Recency in days
    customer_tx_metrics['Recency'] = (ref_date - customer_tx_metrics['LastTxDate']).dt.days
    
    # Calculate Customer Tenure in days (first to last transaction)
    customer_tx_metrics['TenureDays'] = (customer_tx_metrics['LastTxDate'] - customer_tx_metrics['FirstTxDate']).dt.days + 1
    
    # Calculate Daily Transaction Frequency (Frequency / Tenure)
    customer_tx_metrics['TxFrequencyDaily'] = customer_tx_metrics['Frequency'] / customer_tx_metrics['TenureDays']
    
    # 4. Calculate Max Monthly Balance per Customer
    # Step 4a: Group by Customer and YearMonth to get the max balance in each month
    monthly_balances = data.groupby(['CustomerID', 'TxYearMonth'])['CustAccountBalance'].max().reset_index()
    # Step 4b: For each customer, get the average of their max monthly balances
    avg_max_monthly_balance = monthly_balances.groupby('CustomerID')['CustAccountBalance'].mean().rename('AvgMaxMonthlyBalance')
    # Step 4c: Also get the maximum of their max monthly balances (overall max monthly balance)
    overall_max_monthly_balance = monthly_balances.groupby('CustomerID')['CustAccountBalance'].max().rename('MaxMonthlyBalance')
    
    # Combine monthly balance features
    monthly_balance_metrics = pd.concat([avg_max_monthly_balance, overall_max_monthly_balance], axis=1)
    
    # 5. Merge all features together
    customer_features = customer_demographics.join(customer_tx_metrics).join(monthly_balance_metrics)
    
    # Reset index to make CustomerID a column
    customer_features = customer_features.reset_index()
    
    # Final column ordering for clarity
    col_order = [
        'CustomerID', 'CustGender', 'CustLocation', 'AgeAtTx', 
        'Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg', 'MaxTxAmount',
        'AvgMaxMonthlyBalance', 'MaxMonthlyBalance', 'AvgAccountBalance',
        'TenureDays', 'TxFrequencyDaily'
    ]
    
    # Keep only the ordered columns that exist
    final_cols = [c for c in col_order if c in customer_features.columns]
    return customer_features[final_cols]

if __name__ == "__main__":
    import sys
    csv_path = "data/customers.csv"
    print(f"Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        print("Engineering features...")
        features_df = engineer_features(df)
        
        print("\n=== FEATURE ENGINEERING SUMMARY ===")
        print(f"Original transaction rows: {len(df)}")
        print(f"Unique customers (features shape): {features_df.shape[0]} rows, {features_df.shape[1]} columns")
        
        print("\n=== SAMPLE CUSTOMER FEATURES ===")
        print(features_df.head(5).to_string())
        
        print("\n=== FEATURE DESCRIPTION ===")
        print(features_df.describe().T.to_string())
        
    except Exception as e:
        print(f"Error during feature engineering: {e}", file=sys.stderr)
        sys.exit(1)
