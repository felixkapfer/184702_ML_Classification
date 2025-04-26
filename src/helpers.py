import pandas as pd

def count_unique_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Returns a DataFrame with the counts of unique values for specified columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to count unique values for.

    Returns:
    - pd.DataFrame: A DataFrame with columns ['feature', unique values...]
      where each row corresponds to a feature and the count of each unique value.
    """
    # Initialize a dictionary to collect counts
    value_counts = {}
    
    # Iterate over each specified column
    for col in columns:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            value_counts[col] = counts
        else:
            raise KeyError(f"Column '{col}' does not exist in the DataFrame.")
    
    # Create a combined DataFrame, filling missing values with 0
    counts_df = pd.DataFrame(value_counts).fillna(0).astype(int).T
    counts_df.index.name = 'feature'
    
    return counts_df


def replace_values(df: pd.DataFrame, columns: list, replace_map: dict) -> pd.DataFrame:
    """
    Replaces specified values in given columns according to a replacement map.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names where replacements should occur.
    - replace_map (dict): Dictionary specifying which values to replace with what.
      Example: {'unknown': np.nan, 'yes': 1, 'no': 0}

    Returns:
    - pd.DataFrame: A new DataFrame with the values replaced.
    """
    # Work on a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    
    # Iterate over each specified column and apply the replacement
    for col in columns:
        if col in df_copy.columns:
            # Replace values
            df_copy[col] = df_copy[col].replace(replace_map)

            # Explicitly fix object types
            df_copy[col] = df_copy[col].infer_objects(copy=False)
        else:
            raise KeyError(f"Column '{col}' does not exist in the DataFrame.")
    
    return df_copy
