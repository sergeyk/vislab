"""
Code for analyzing datasets.
"""
import pandas as pd
import numpy as np


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
    top_rows = df[row_column].value_counts().iloc[:top_k]
    top_cols = df[col_column].value_counts().iloc[:top_k]

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
