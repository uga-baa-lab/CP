"""
Report Generator for CHEAT-CP Kinematic Analysis
==================================================

Generates Excel mastersheets from the analysis CSV results.

Entry point:
    main(data_file_location, report_results_dir, pvar_results_dir, master_df)

This script produces 6 report types:
    - Means and Std for all kinematic variables
    - Means and Std for IDE only
    - Means and Std for PVar (path variability)

Each report pivots long-format subject data to wide format (one column per
Condition × Arm combination), broken down by day and also by duration.
"""

import os
import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Allow direct execution: fix imports and load config
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from reporting.helpers import (
        calculate_additional_columns,
        save_grouped_data,
        save_grouped_data_std,
        pivot_data_to_wide,
        reindex_with_missing_ids,
        save_excel,
        combine_std_columns,
    )
else:
    from .helpers import (
        calculate_additional_columns,
        save_grouped_data,
        save_grouped_data_std,
        pivot_data_to_wide,
        reindex_with_missing_ids,
        save_excel,
        combine_std_columns,
    )

# ---- Constants ----
TOTAL_IDS = 88
ALL_IDS = ["cpvib" + str(item).zfill(3) for item in range(1, TOTAL_IDS + 1)]
MAX_DAYS = 5
DURATIONS = [(500, 625), (750, 900)]
KINEMATIC_VARLIST = [
    "Accuracy",
    "MT",
    "RT",
    "pathlength",
    "velPeak",
    "EndPointError",
    "IDE",
    "PLR",
    "IE",
    "targetDist_Hit_Interception",
]
PVAR_VARLIST = ["Pvar"]


# =============================================================================
# SHARED DAY PROCESSING — used by means, std, and IDE reports
# =============================================================================


def _process_day(
    df, df_dur, day_num, varlist, exceltitle, sheet_suffix, all_ids, aggfunc="mean"
):
    """
    Pivot data for one day and save to one Excel sheet.

    Works for both mean and std reports — aggfunc controls how duration
    pairs are combined:
      - 'mean': simple average  (col1 + col2) / 2
      - 'std' : root-mean-square  sqrt(mean(col1², col2²))

    Parameters
    ----------
    df : DataFrame — subject-level grouped data (by condition/arm)
    df_dur : DataFrame — same but also grouped by duration
    day_num : int — day number (1–5)
    varlist : list — variables to include in the report
    exceltitle : str — output Excel file path
    sheet_suffix : str — appended to 'DayN_' for the sheet name (e.g. 'Master_Formatted')
    all_ids : list — full list of subject IDs to reindex against
    aggfunc : str — 'mean' or 'std' for duration combining
    """
    current_day = f"Day{day_num}"
    index_cols = ["subject", "visit", "studyid", "group", "day"]

    # Pivot by condition × arm
    df_day = df[df["day"] == current_day]
    df_wide = pivot_data_to_wide(df_day, index_cols, ["Condition", "Affected"], varlist)
    df_wide = reindex_with_missing_ids(df_wide, all_ids, False)

    # Pivot by condition × arm × duration
    df_day_dur = df_dur[df_dur["day"] == current_day]
    df_wide_dur = pivot_data_to_wide(
        df_day_dur, index_cols, ["Condition", "Affected", "Duration"], varlist
    )
    df_wide_dur = reindex_with_missing_ids(df_wide_dur, all_ids, True)

    # Combine duration pairs (500+625, 750+900)
    for var in varlist:
        for condition in ["Interception", "Reaching"]:
            for affected in ["Less Affected", "More Affected"]:
                for d1, d2 in DURATIONS:
                    col1 = (var, condition, affected, d1)
                    col2 = (var, condition, affected, d2)
                    if (
                        col1 not in df_wide_dur.columns
                        or col2 not in df_wide_dur.columns
                    ):
                        continue
                    if aggfunc == "std":
                        combined_name = (
                            var,
                            condition,
                            affected,
                            f"{d1}_{d2}_combined_std",
                        )
                        df_wide_dur = combine_std_columns(
                            df_wide_dur, [col1, col2], combined_name
                        )
                    else:
                        combined_name = (
                            var,
                            condition,
                            affected,
                            f"{d1}_{d2}_combined",
                        )
                        df_wide_dur[combined_name] = (
                            df_wide_dur[col1] + df_wide_dur[col2]
                        ) / 2

    df_combo = pd.concat(
        [df_wide, df_wide_dur.drop(columns=["studyid"], errors="ignore")],
        axis=1,
        join="inner",
    )
    mode = "w" if day_num == 1 else "a"
    save_excel(df_combo, exceltitle, f"{current_day}_{sheet_suffix}", mode=mode)


# =============================================================================
# KINEMATIC MEANS REPORT
# =============================================================================


def main_means(data_file_location, report_results_dir):
    """Generate subject-level means for all kinematic variables."""
    all_df = pd.read_csv(data_file_location)
    all_df.loc[all_df["IE"].abs() > 1000, "IE"] = np.nan  # Remove extreme IE outliers
    all_df = calculate_additional_columns(all_df)

    group_cols = [
        "group",
        "visit",
        "studyid",
        "subject",
        "day",
        "Condition",
        "Affected",
    ]
    df_means = save_grouped_data(
        all_df, group_cols, "means_bysubject.csv", report_results_dir
    )
    df_meansdur = save_grouped_data(
        all_df,
        group_cols + ["Duration"],
        "means_bysubjectandduration.csv",
        report_results_dir,
    )

    exceltitle = os.path.join(
        report_results_dir, "UL_KINARM_Mastersheet_Auto_Format_means.xlsx"
    )
    exceltitle2 = os.path.join(
        report_results_dir, "UL_KINARM_Mastersheet_Long_Format_means.xlsx"
    )

    for day_num in range(1, MAX_DAYS + 1):
        _process_day(
            df_means,
            df_meansdur,
            day_num,
            KINEMATIC_VARLIST,
            exceltitle,
            "Master_Formatted",
            ALL_IDS,
            aggfunc="mean",
        )

    save_excel(df_meansdur, exceltitle2, "AllDays_Master_Formatted_Means", mode="w")


# =============================================================================
# KINEMATIC STD REPORT
# =============================================================================


def main_std(data_file_location, report_results_dir):
    """Generate subject-level standard deviations for all kinematic variables."""
    all_df = pd.read_csv(data_file_location)
    all_df.loc[all_df["IE"].abs() > 1000, "IE"] = np.nan
    all_df = calculate_additional_columns(all_df)

    group_cols = [
        "group",
        "visit",
        "studyid",
        "subject",
        "day",
        "Condition",
        "Affected",
    ]
    df_stds = save_grouped_data_std(
        all_df, group_cols, "stds_bysubject.csv", report_results_dir
    )
    df_std_dur = save_grouped_data_std(
        all_df,
        group_cols + ["Duration"],
        "stds_bysubjectandduration.csv",
        report_results_dir,
    )

    exceltitle = os.path.join(
        report_results_dir, "UL_KINARM_Mastersheet_Auto_Format_stds.xlsx"
    )
    exceltitle2 = os.path.join(
        report_results_dir, "UL_KINARM_Mastersheet_Long_Format_stds.xlsx"
    )

    for day_num in range(1, MAX_DAYS + 1):
        _process_day(
            df_stds,
            df_std_dur,
            day_num,
            KINEMATIC_VARLIST,
            exceltitle,
            "Master_Formatted_STD",
            ALL_IDS,
            aggfunc="std",
        )

    save_excel(df_std_dur, exceltitle2, "AllDays_Master_Formatted_Stds", mode="w")


# =============================================================================
# IDE-ONLY MEANS REPORT
# =============================================================================


def main_means_ide(data_file_location, report_results_dir):
    """Generate IDE-only means report (IDE and ABS_IDE)."""
    all_df = pd.read_csv(data_file_location)
    all_df["ABS_IDE"] = all_df["IDE"].abs()
    all_df = calculate_additional_columns(all_df)

    group_cols = [
        "group",
        "visit",
        "studyid",
        "subject",
        "day",
        "Condition",
        "Affected",
    ]
    df_means = save_grouped_data(
        all_df, group_cols, "means_bysubject.csv", report_results_dir
    )
    df_meansdur = save_grouped_data(
        all_df,
        group_cols + ["Duration"],
        "means_bysubjectandduration.csv",
        report_results_dir,
    )

    exceltitle = os.path.join(
        report_results_dir, "UL_KINARM_Mastersheet_Only_IDE_means.xlsx"
    )

    for day_num in range(1, MAX_DAYS + 1):
        _process_day(
            df_means,
            df_meansdur,
            day_num,
            ["IDE", "ABS_IDE"],
            exceltitle,
            "IDE_Formatted",
            ALL_IDS,
            aggfunc="mean",
        )


# =============================================================================
# IDE-ONLY STD REPORT
# =============================================================================


def main_std_ide(data_file_location, report_results_dir):
    """Generate IDE-only standard deviation report (IDE and ABS_IDE)."""
    all_df = pd.read_csv(data_file_location)
    all_df["ABS_IDE"] = all_df["IDE"].abs()
    all_df = calculate_additional_columns(all_df)

    group_cols = [
        "group",
        "visit",
        "studyid",
        "subject",
        "day",
        "Condition",
        "Affected",
    ]
    df_stds = save_grouped_data_std(
        all_df, group_cols, "stds_bysubject.csv", report_results_dir
    )
    df_std_dur = save_grouped_data_std(
        all_df,
        group_cols + ["Duration"],
        "stds_bysubjectandduration.csv",
        report_results_dir,
    )

    exceltitle = os.path.join(
        report_results_dir, "UL_KINARM_Mastersheet_Only_IDE_STD.xlsx"
    )

    for day_num in range(1, MAX_DAYS + 1):
        _process_day(
            df_stds,
            df_std_dur,
            day_num,
            ["IDE", "ABS_IDE"],
            exceltitle,
            "IDE_STD_Formatted",
            ALL_IDS,
            aggfunc="std",
        )


# =============================================================================
# PVAR REPORTS — needs metadata merge + custom column ordering
# =============================================================================


def _merge_pvar_with_metadata(pvar_df, day, master_df):
    """
    Filter PVar data for a given day and enrich it with subject metadata
    (visit, studyid, group) from the master Excel sheet.
    """
    pvar_day_df = pvar_df[pvar_df["Day"] == day].rename(
        columns={"Subject": "subject", "Day": "day", "Arm": "Affected"}
    )

    metadata = master_df[
        ["KINARM ID", "Visit ID", "Subject ID", "Group", "Visit_Day"]
    ].copy()
    metadata = metadata.rename(
        columns={
            "KINARM ID": "subject",
            "Visit ID": "visit",
            "Subject ID": "studyid",
            "Visit_Day": "day",
        }
    )
    metadata["subject"] = metadata["subject"].apply(
        lambda x: x[-3:] if x.startswith("CHEAT") else x
    )
    metadata["group"] = metadata["Group"].map({0: "TDC", 1: "CP"})

    return pvar_day_df.merge(
        metadata[["subject", "day", "visit", "studyid", "group"]],
        on=["subject", "day"],
        how="left",
    )


def _sort_pvar_columns(df):
    """
    Apply the custom column ordering for PVar reports:
    group by condition → arm, then within each group: summary → individual durations.
    Combined duration columns go at the end.
    """
    base_cols = ["KINARM_ID", "visit", "studyid", "group", "Visit_day"]
    pivoted_cols = [col for col in df.columns if col not in base_cols]

    condition_order = {"Interception": 0, "Reaching": 1}
    arm_order = {"Less Affected": 0, "More Affected": 1}
    duration_order = {500: 0, 625: 1, 750: 2, 900: 3}
    combined_order = {
        "500_625_combined": 0,
        "750_900_combined": 1,
        "500_625_combined_std": 0,
        "750_900_combined_std": 1,
    }

    non_combined = [
        c
        for c in pivoted_cols
        if not (len(c) == 4 and not isinstance(c[3], (int, float)))
    ]
    combined = [
        c for c in pivoted_cols if len(c) == 4 and not isinstance(c[3], (int, float))
    ]

    # Group non-combined by (condition, arm), then sort within each group
    groups = {}
    for col in non_combined:
        key = (col[1], col[2])
        groups.setdefault(key, []).append(col)

    ordered = []
    for key in sorted(
        groups, key=lambda x: (condition_order.get(x[0], 999), arm_order.get(x[1], 999))
    ):
        cols = groups[key]
        summary = [c for c in cols if len(c) == 3]
        individual = sorted(
            [c for c in cols if len(c) == 4 and isinstance(c[3], (int, float))],
            key=lambda x: duration_order.get(x[3], 999),
        )
        ordered.extend(summary + individual)

    combined.sort(
        key=lambda x: (
            condition_order.get(x[1], 999),
            arm_order.get(x[2], 999),
            combined_order.get(x[3], 999),
        )
    )
    ordered.extend(combined)

    return df[base_cols + ordered]


def main_pvar_means(pvar_results_dir, report_results_dir, master_df):
    """Generate PVar means report."""
    pvar_file = os.path.join(pvar_results_dir, "pvar_results.csv")
    if not os.path.exists(pvar_file):
        print(
            f"PVar results file not found at {pvar_file}. Please run NewPVar.py first."
        )
        return

    pvar_df = pd.read_csv(pvar_file)
    exceltitle = os.path.join(
        report_results_dir, "UL_KINARM_Mastersheet_PVar_Auto_Format_means.xlsx"
    )
    index_cols = ["subject", "visit", "studyid", "group", "day"]

    for day_num in range(1, MAX_DAYS + 1):
        current_day = f"Day{day_num}"
        pvar_day_df = _merge_pvar_with_metadata(pvar_df, current_day, master_df)

        pvar_wide = pivot_data_to_wide(
            pvar_day_df, index_cols, ["Condition", "Affected"], PVAR_VARLIST
        )
        pvar_wide = reindex_with_missing_ids(pvar_wide, ALL_IDS, False)

        pvar_wide_dur = pivot_data_to_wide(
            pvar_day_df, index_cols, ["Condition", "Affected", "Duration"], PVAR_VARLIST
        )
        pvar_wide_dur = reindex_with_missing_ids(pvar_wide_dur, ALL_IDS, True)

        for var in PVAR_VARLIST:
            for condition in ["Interception", "Reaching"]:
                for affected in ["Less Affected", "More Affected"]:
                    for d1, d2 in DURATIONS:
                        col1, col2 = (var, condition, affected, d1), (
                            var,
                            condition,
                            affected,
                            d2,
                        )
                        if (
                            col1 in pvar_wide_dur.columns
                            and col2 in pvar_wide_dur.columns
                        ):
                            pvar_wide_dur[
                                (var, condition, affected, f"{d1}_{d2}_combined")
                            ] = (pvar_wide_dur[col1] + pvar_wide_dur[col2]) / 2

        combo = pd.concat([pvar_wide, pvar_wide_dur], axis=1, join="inner")
        combo = combo.drop(columns=["subject"], errors="ignore")
        combo = _sort_pvar_columns(combo)

        mode = "w" if day_num == 1 else "a"
        save_excel(combo, exceltitle, f"{current_day}_Auto_Format", mode=mode)

    print("PVar means pivot table report generated successfully.")


def main_pvar_std(pvar_results_dir, report_results_dir, master_df):
    """Generate PVar standard deviation report."""
    pvar_file = os.path.join(pvar_results_dir, "pvar_results.csv")
    if not os.path.exists(pvar_file):
        print(
            f"PVar results file not found at {pvar_file}. Please run NewPVar.py first."
        )
        return

    pvar_df = pd.read_csv(pvar_file)
    exceltitle = os.path.join(
        report_results_dir, "UL_KINARM_Mastersheet_PVar_Auto_Format_stds.xlsx"
    )
    index_cols = ["subject", "visit", "studyid", "group", "day"]

    for day_num in range(1, MAX_DAYS + 1):
        current_day = f"Day{day_num}"
        pvar_day_df = _merge_pvar_with_metadata(pvar_df, current_day, master_df)

        # PVar std uses aggfunc='std' directly (pivot_data_to_wide only does mean)
        for pivot_cols, is_dur in [
            (["Condition", "Affected"], False),
            (["Condition", "Affected", "Duration"], True),
        ]:
            wide = pvar_day_df.pivot_table(
                index=index_cols, columns=pivot_cols, values=PVAR_VARLIST, aggfunc="std"
            )
            wide.columns = wide.columns.to_flat_index()
            wide = wide.reset_index()
            wide = reindex_with_missing_ids(wide, ALL_IDS, is_dur)
            if not is_dur:
                pvar_wide = wide
            else:
                pvar_wide_dur = wide

        for var in PVAR_VARLIST:
            for condition in ["Interception", "Reaching"]:
                for affected in ["Less Affected", "More Affected"]:
                    for d1, d2 in DURATIONS:
                        col1, col2 = (var, condition, affected, d1), (
                            var,
                            condition,
                            affected,
                            d2,
                        )
                        if (
                            col1 in pvar_wide_dur.columns
                            and col2 in pvar_wide_dur.columns
                        ):
                            pvar_wide_dur = combine_std_columns(
                                pvar_wide_dur,
                                [col1, col2],
                                (var, condition, affected, f"{d1}_{d2}_combined_std"),
                            )

        combo = pd.concat([pvar_wide, pvar_wide_dur], axis=1, join="inner")
        combo = combo.drop(columns=["subject"], errors="ignore")
        combo = _sort_pvar_columns(combo)

        mode = "w" if day_num == 1 else "a"
        save_excel(combo, exceltitle, f"{current_day}_Auto_Format", mode=mode)

    print("PVar standard deviations pivot table report generated successfully.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main(data_file_location, report_results_dir, pvar_results_dir, master_df):
    """
    Run all 6 reports.

    Parameters
    ----------
    data_file_location : str — Path to the main analysis CSV
    report_results_dir : str — Where to save all report output files
    pvar_results_dir : str — Directory containing pvar_results.csv
    master_df : DataFrame — Master subject list (from load_master_excel)
    """
    main_means(data_file_location, report_results_dir)
    print("Done: means")

    main_std(data_file_location, report_results_dir)
    print("Done: stds")

    main_means_ide(data_file_location, report_results_dir)
    print("Done: IDE means")

    main_std_ide(data_file_location, report_results_dir)
    print("Done: IDE stds")

    main_pvar_means(pvar_results_dir, report_results_dir, master_df)
    main_pvar_std(pvar_results_dir, report_results_dir, master_df)


if __name__ == "__main__":
    from config import (
        data_file_location,
        REPORT_RESULTS_DIR,
        PVAR_RESULTS_DIR,
        MASTER_FILE,
    )
    import pandas as pd

    master_df = pd.read_excel(MASTER_FILE)
    main(data_file_location, REPORT_RESULTS_DIR, PVAR_RESULTS_DIR, master_df)
