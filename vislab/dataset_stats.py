"""
Code for analyzing datasets.
"""
import pandas as pd
import numpy as np


def top_k_vals(df, column, top_k=10):
    """
    Note: df must have unique index.

    Parameters
    ----------
    df: pandas.DataFrame
    column: string
        Name of column to get the top k values of.
    top_k: int [10]
    """
    df = df.copy()
    assert(len(df.index.unique()) == len(df.index))
    df['index'] = df.index
    frequencies = df.groupby(column)['index'].nunique()
    frequencies.sort(ascending=False)
    top_frequencies = frequencies.iloc[:top_k]
    return top_frequencies


def get_joint_occurrence_df(df, row_column, col_column, top_k=10):
    """
    Form a DataFrame where:
    - index is composed of top_k top values in row_column.
    - columns are composed of top_k top values in col_column.
    - cell values are the number of times that the index and
    column values occur together in the given DataFrame.

    Note: Index of the DataFrame must be unique.
    """
    df_s = df[[row_column, col_column]].copy()

    # Get top row and column values.
    top_rows = top_k_vals(df, row_column, top_k)
    top_cols = top_k_vals(df, col_column, top_k)

    # Drop rows that don't have a genre and style in the top list.
    filter_lambda = lambda x: \
        x[row_column] in top_rows and x[col_column] in top_cols
    df_s = df_s[df_s.apply(filter_lambda, axis=1)]

    fname = 'get_joint_occurrence_df'
    print("{}: looking at co-occurrence of {} and {}".format(
        fname, row_column, col_column))
    print("  dropped {}/{} rows that don't have both vals in top-{}.".format(
        df_s.shape[0] - df_s.shape[0], df_s.shape[0], top_k))

    # Construct joint occurence matrix
    JM = np.zeros((top_k, top_k))
    for i, row in enumerate(top_rows.index):
        for j, col in enumerate(top_cols.index):
            JM[i, j] = (
                (df_s[row_column] == row) &
                (df_s[col_column] == col)
            ).sum()

    df_m = pd.DataFrame(JM, columns=top_cols.index, index=top_rows.index)
    return df_m


def condition_df_on_row(df):
    """
    Divide each cell by the sum of its row and return the resulting
    DataFrame.
    """
    df = df.copy()
    total = df.sum(1)
    df = (df.T / total).T
    df['prior'] = total / total.sum()
    return df
