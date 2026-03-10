"""
Reporting Helpers
=================

Shared utility functions used by all report types:
- Pivot tables (long → wide)
- Reindex with missing subject IDs
- Save to Excel
- Combine duration columns (mean and std)
- Calculate additional derived columns
"""

import os
import numpy as np
import pandas as pd


def calculate_additional_columns(df):
    """Add derived columns: IA_abs, pathNorm, xTargetabs."""
    df["IA_abs"] = np.abs(df["IA_50RT"])
    df["pathNorm"] = df["pathlength"] / df["straightlength"]
    df["xTargetabs"] = np.abs(df["xTargetEnd"])
    return df


def save_grouped_data(df, group_cols, filename, results_dir):
    """Group by columns, compute means, drop duplicated columns, save CSV."""
    os.makedirs(results_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    all_df_numeric = df[group_cols + list(numeric_cols)].copy()
    all_df_numeric = all_df_numeric.loc[:, ~all_df_numeric.columns.duplicated()]
    grouped_df_means = all_df_numeric.groupby(group_cols).mean().reset_index()
    grouped_df_means.to_csv(os.path.join(results_dir, filename), index=False)
    return grouped_df_means


def save_grouped_data_std(df, group_cols, filename, results_dir):
    """Group by columns, compute std, drop duplicated columns, save CSV."""
    os.makedirs(results_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    all_df_numeric = df[group_cols + list(numeric_cols)].copy()
    all_df_numeric = all_df_numeric.loc[:, ~all_df_numeric.columns.duplicated()]
    grouped_df = all_df_numeric.groupby(group_cols).std().reset_index()
    grouped_df.to_csv(os.path.join(results_dir, filename), index=False)
    return grouped_df


def pivot_data_to_wide(df, index_cols, pivot_cols, varlist):
    """
    Pivot a long-format DataFrame to wide format.
    Columns are flattened to tuples and ordered by varlist order.
    """
    df_wide = df.pivot_table(index=index_cols, columns=pivot_cols, values=varlist)
    df_wide.columns = df_wide.columns.to_flat_index()
    df_wide = df_wide.reset_index()
    variable_order = {var: idx for idx, var in enumerate(varlist)}
    ordered_cols = index_cols + [
        col
        for col in sorted(
            df_wide.columns[len(index_cols) :],
            key=lambda x: (variable_order.get(x[0], len(varlist)), *x[1:]),
        )
    ]
    df_wwide = df_wide[ordered_cols]
    return df_wwide


def reindex_with_missing_ids(df_wide, all_ids, is_wide_dur):
    """
    Ensure all subject IDs appear in the DataFrame (fill missing with NaN).
    For duration-level data, drops metadata columns.
    For subject-level data, renames subject→KINARM_ID, day→Visit_day.
    """
    missing = list(set(all_ids) - set(df_wide["subject"].astype(str)))
    if missing:
        missing_df = pd.DataFrame({"subject": missing})
        df_wide = pd.concat([df_wide, missing_df], ignore_index=True, sort=False)
    df_wide.index.name = "NMSKL_ID"
    if is_wide_dur:
        df_wide = df_wide.drop(
            columns=["visit", "studyid", "group", "day"], errors="ignore"
        )
        return df_wide
    df_wide = df_wide.rename(columns={"subject": "KINARM_ID", "day": "Visit_day"})
    return df_wide


def save_excel(df, filepath, sheet_name, mode="w"):
    """Save a DataFrame to an Excel sheet."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with pd.ExcelWriter(filepath, engine="openpyxl", mode=mode) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def combine_mean_columns(df, col1, col2, new_col_name):
    """Combine two columns by averaging them."""
    df[new_col_name] = (df[col1] + df[col2]) / 2
    return df


def combine_std_columns(df, cols, new_col_name):
    """Combine standard deviation columns using root-mean-square formula."""
    combined_variance = df[cols].apply(lambda x: x**2).mean(axis=1)
    df[new_col_name] = np.sqrt(combined_variance)
    return df
