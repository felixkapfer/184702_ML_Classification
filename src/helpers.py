import pandas as pd
from sklearn.impute import SimpleImputer

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


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes the count and percentage of missing values in each column of the dataframe.

    Parameters:
        - df (pd.DataFrame): The input DataFrame.

    Returns:
        - pd.DataFrame: A DataFrame with two columns:
        - 'missing_count': The count of missing values per column.
        - 'missing_pct': The percentage of missing values per column.
    """
    total = len(df)  # Get the total number of rows in the dataframe
    missing_count = df.isna().sum()  # Count missing values per column
    missing_pct = missing_count / total * 100  # Calculate the percentage of missing values
    return pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct
    })

def drop_columns_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drops columns with a missing value percentage greater than the specified threshold.

    Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - threshold (float): The percentage threshold for missing values (0.0 - 1.0).
        Columns with more missing values than this threshold will be dropped.

    Returns:
        - pd.DataFrame: A copy of the DataFrame with columns dropped where missing values exceed the threshold.
    """
    df_copy = df.copy()  # Work with a copy to avoid modifying the original DataFrame
    miss = df_copy.isna().mean()  # Calculate the percentage of missing values per column
    cols_to_drop = miss[miss > threshold].index  # Find columns with more missing values than the threshold
    return df_copy.drop(columns=cols_to_drop)  # Drop those columns and return the cleaned DataFrame


def drop_rows_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drops rows with a missing value percentage greater than the specified threshold.

    Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - threshold (float): The percentage threshold for missing values (0.0 - 1.0).
        Rows with more missing values than this threshold will be dropped.

    Returns:
        - pd.DataFrame: A copy of the DataFrame with rows dropped where missing values exceed the threshold.
    """
    df_copy = df.copy()  # Work with a copy to avoid modifying the original DataFrame
    mask = df_copy.isna().mean(axis=1) <= threshold  # Create a mask to keep rows with less missing data
    return df_copy.loc[mask].copy()  # Apply the mask and return the cleaned DataFrame


def impute_mode(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Imputes missing values in the specified columns using the mode (most frequent value).

    Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - columns (list, optional): A list of column names where missing values should be imputed.
        If None, all columns with missing values will be considered.

    Returns:
        - pd.DataFrame: A copy of the DataFrame with missing values replaced by the mode.
    """
    df_copy = df.copy()  # Work with a copy to avoid modifying the original DataFrame
    if columns is None:
        columns = df_copy.columns.tolist()  # If no columns are specified, use all columns

    imputer = SimpleImputer(strategy="most_frequent")  # Create an imputer that uses the most frequent value
    df_copy[columns] = imputer.fit_transform(df_copy[columns])  # Impute missing values in the specified columns
    return df_copy  # Return the DataFrame with imputed values
