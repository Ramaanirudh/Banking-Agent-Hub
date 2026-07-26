import pandas as pd

def calculate_segment_aggregation(df: pd.DataFrame, group_col: str = 'Cluster', agg_col: str = 'MonetaryAvg', agg_func: str = 'mean') -> dict:
    """
    Perform group-by aggregation on the customer dataset.
    
    Parameters:
    df (pd.DataFrame): Customer features dataframe.
    group_col (str): Column to group by (e.g. 'Cluster').
    agg_col (str): Column to aggregate (e.g. 'MonetaryAvg', 'AvgAccountBalance').
    agg_func (str): Aggregation function name (e.g. 'mean', 'sum', 'max', 'min').
    
    Returns:
    dict: Dictionary containing the aggregation results per segment/group.
    """
    # Safeguard: if columns are missing, use defaults or fallback
    if group_col not in df.columns:
        # If Cluster is not present, use a default fallback or throw
        raise KeyError(f"Grouping column '{group_col}' not found in dataframe columns: {list(df.columns)}")
        
    if agg_col not in df.columns:
        # Fallback to a column that exists
        alternatives = ['MonetaryAvg', 'MonetaryTotal', 'AvgAccountBalance']
        found = False
        for alt in alternatives:
            if alt in df.columns:
                agg_col = alt
                found = True
                break
        if not found:
            raise KeyError(f"Aggregation column '{agg_col}' not found in dataframe.")
            
    # Perform aggregation
    try:
        grouped = df.groupby(group_col)[agg_col].agg(agg_func)
        return {
            'group_col': group_col,
            'agg_col': agg_col,
            'agg_func': agg_func,
            'results': grouped.to_dict()
        }
    except Exception as e:
        return {
            'error': str(e)
        }

if __name__ == "__main__":
    # Test stub
    df_test = pd.DataFrame({
        'Cluster': [0, 0, 1, 1],
        'MonetaryAvg': [100.0, 200.0, 50.0, 150.0]
    })
    res = calculate_segment_aggregation(df_test, 'Cluster', 'MonetaryAvg', 'mean')
    print(res)
