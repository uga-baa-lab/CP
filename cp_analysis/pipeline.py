"""
Data Processing Pipeline for CHEAT-CP Kinematic Analysis
==========================================================

This module contains TWO versions of getDataCP():

1. getDataCP()         — Uses SIMPLE Linear Regression
2. getDataCP_ransac()  — Uses RANSAC Robust Regression
                          Includes outlier tracking, R² comparisons, outlier plots

HOW TO USE:

# As a package (from project root):
from cp_analysis.pipeline import run_pipeline

# Directly (from inside cp_analysis/):
python3 pipeline.py
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.io

# Ensure the cp_analysis/ directory is on sys.path so sibling modules
# (kinematics, regression) can be found whether pipeline.py is run
# directly or imported as part of the cp_analysis package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kinematics as kin
import regression as reg
import pvar as pv

# =============================================================================
# LOADING DATA
# =============================================================================


def load_master_excel(master_file_path):
    """
    Load the Master Excel file containing the subject list.
    """
    mdf = pd.read_excel(
        open(master_file_path, "rb"), sheet_name="KINARM_AllVisitsMaster"
    )
    return mdf


# =============================================================================
# VERSION 1: SIMPLE LINEAR REGRESSION (from CHEATCP_Final_With_IE.py)
# =============================================================================


def getDataCP(mdf, matfiles, defaults, RESULTS_DIR, subject_filter=None):
    """
    Process all subjects using SIMPLE Linear Regression for IE.
    This is the original getDataCP() from CHEATCP_Final_With_IE.py.

    Parameters
    ----------
    mdf : DataFrame — Master subject list
    matfiles : str — Path to .mat files directory
    defaults : dict — DEFAULTS from config
    RESULTS_DIR : str — Where to save results
    subject_filter : list, optional — Only process these subjects

    Returns
    -------
    all_df : DataFrame — All trials, all subjects
    allTrajs : dict — Trajectory data keyed by subject+visit_day
    """
    all_df = pd.DataFrame()
    allTrajs = {}
    all_trials_ie_by_subject = []

    for index, row in mdf.iterrows():
        if subject_filter is not None:
            if row["KINARM ID"] not in subject_filter:
                continue

        if row["KINARM ID"].startswith("CHEAT"):
            subject = row["KINARM ID"][-3:]
        else:
            subject = row["KINARM ID"]

        print(f"evaluating the subject : {subject}")
        subjectmat = "CHEAT-CP" + subject + row["Visit_Day"] + ".mat"
        mat = os.path.join(matfiles, subjectmat)

        if not os.path.exists(mat):
            print("skipping", mat)
            continue

        loadmat = scipy.io.loadmat(mat)
        data = loadmat["subjDataMatrix"][0][0]

        allTrials = []
        all_trials_df = []
        subjectTrajs = []

        for i in range(len(data)):
            thisData = data[i]
            trajData = kin.get_hand_trajectories(thisData, defaults)
            kinData = kin.compute_trial_kinematics(thisData, defaults, i, subject)

            row_values = [
                kinData["Condition"],
                thisData.T[16][0],
                thisData.T[11][0],
                thisData.T[13][0],
                thisData.T[14][0],
                thisData.T[15][0],
                kinData["RT"],
                kinData["CT"],
                kinData["velPeak"],
                kinData["xPosError"],
                kinData["minDist"],
                kinData["targetDist"],
                kinData["handDist"],
                kinData["straightlength"],
                kinData["pathlength"],
                kinData["targetlength"],
                kinData["CursorX"],
                kinData["CursorY"],
                kinData["IA_RT"],
                kinData["IA_50RT"],
                kinData["RTalt"],
                kinData["IA_RTalt"],
                kinData["maxpathoffset"],
                kinData["meanpathoffset"],
                kinData["xTargetEnd"],
                kinData["yTargetEnd"],
                kinData["EndPointError"],
                kinData["IDE"],
                kinData["PLR"],
                kinData["isCurveAround"],
                i,
            ]

            allTrials.append(row_values)
            all_trials_df.append(kinData)
            subjectTrajs.append(trajData)

        print(f"evaluated the kinData for sub : {subject}")

        regression_coeffs, data_by_arm = reg.perform_subject_level_regression(
            all_trials_df
        )
        reg.plot_regression_points(
            subject, data_by_arm, regression_coeffs, row["Visit_Day"]
        )
        subject_wise_IE = reg.calculate_ie_for_interception_trials(
            all_trials_df, regression_coeffs, subject
        )
        all_trials_ie_by_subject.append(subject_wise_IE)

        df = pd.DataFrame(
            allTrials,
            columns=[
                "Condition",
                "Affected",
                "TP",
                "Duration",
                "Accuracy",
                "FeedbackTime",
                "RT",
                "CT",
                "velPeak",
                "xPosError",
                "minDist",
                "targetDist",
                "handDist",
                "straightlength",
                "pathlength",
                "targetlength",
                "cursorX",
                "cursorY",
                "IA_RT",
                "IA_50RT",
                "RTalt",
                "IA_RTalt",
                "maxpathoffset",
                "meanpathoffset",
                "xTargetEnd",
                "yTargetEnd",
                "EndPointError",
                "IDE",
                "PLR",
                "isCurveAround",
                "trial_number",
            ],
        )

        # --- Data cleaning — same mappings as original ---
        df["Affected"] = df["Affected"].map({1: "More Affected", 0: "Less Affected"})
        df["Condition"] = df["Condition"].map({1: "Reaching", 2: "Interception"})
        df["Duration"] = df["TP"].map(
            {1: 625, 2: 500, 3: 750, 4: 900, 5: 625, 6: 500, 7: 750, 8: 900}
        )
        df["MT"] = df["CT"] - df["RT"]
        df["subject"] = subject
        df["age"] = row["Age at Visit (yr)"]
        df["visit"] = row["Visit ID"]
        df["day"] = row["Visit_Day"]
        df["studyid"] = row["Subject ID"]

        if row["Group"] == 0:
            df["group"] = "TDC"
        else:
            df["group"] = "CP"

        df["pathratio"] = df["pathlength"] / df["targetlength"]
        print(f'Df of condition after mapping : {df["Condition"].unique()}')
        all_df = pd.concat([all_df, df])
        allTrajs[subject + row["Visit_Day"]] = subjectTrajs

    # Save IE results
    output_path = os.path.join(RESULTS_DIR, "IE_CSV_Results")
    os.makedirs(output_path, exist_ok=True)

    if len(all_trials_ie_by_subject) > 0:
        all_trials_df = pd.concat(all_trials_ie_by_subject, ignore_index=True)
        print("Columns in the DataFrame:", all_trials_df.columns)
        grouped_data = all_trials_df.sort_values(by=["Subject", "Condition"])
        file_name = os.path.join(output_path, "All_Trials_IE.csv")
        grouped_data_file_name = os.path.join(output_path, "All_Trials_IE_Grouped.csv")
        grouped_data.to_csv(grouped_data_file_name, index=False)
        pd.concat(all_trials_ie_by_subject, ignore_index=True).to_csv(
            file_name, index=True
        )

    return all_df, allTrajs


def getDataCP_ransac(mdf, matfiles, defaults, RESULTS_DIR, subject_filter=None):
    """
    Process all subjects using RANSAC Robust Regression for IE.
    This is the getDataCP() from InitialEstimateFinalCode.py.

    Includes:
    - RANSAC regression with outlier detection
    - TWO regression plots per arm (all data + inliers only)
    - R² comparison between Linear and Robust regression
    - Per-subject IE CSVs
    - Outlier trial trajectory plots
    - Outlier summary CSV
    - regression_results.csv with R² values

    Parameters and Returns are the same as getDataCP().
    """
    all_df = pd.DataFrame()
    outlier_summaries = []
    rsquared_summaries = []
    allTrajs = {}
    all_trials_ie_by_subject = []
    rsquared_results_df = pd.DataFrame(
        columns=["Subject", "Visit Day", "Arm", "Data Type", "R^2_Linear", "R^2_Robust"]
    )

    for index, row in mdf.iterrows():
        if subject_filter is not None:
            if row["KINARM ID"] not in subject_filter:
                continue

        if row["KINARM ID"].startswith("CHEAT"):
            subject = row["KINARM ID"][-3:]
        else:
            subject = row["KINARM ID"]

        print(f'Evaluating the subject: {subject} for visit day : {row["Visit_Day"]}')
        subjectmat = "CHEAT-CP" + subject + row["Visit_Day"] + ".mat"
        mat = os.path.join(matfiles, subjectmat)

        if not os.path.exists(mat):
            print("Skipping", mat)
            continue

        loadmat = scipy.io.loadmat(mat)
        data = loadmat["subjDataMatrix"][0][0]

        allTrials = []
        subjectTrajs = []

        for i in range(len(data)):
            thisData = data[i]
            trajData = kin.get_hand_trajectories(thisData, defaults)
            kinData = kin.compute_trial_kinematics(thisData, defaults, i, subject)

            row_values = [
                kinData["Condition"],
                thisData.T[16][0],
                thisData.T[11][0],
                thisData.T[13][0],
                thisData.T[14][0],
                thisData.T[15][0],
                kinData["RT"],
                kinData["CT"],
                kinData["velPeak"],
                kinData["xPosError"],
                kinData["minDist"],
                kinData["targetDist"],
                kinData["handDist"],
                kinData["straightlength"],
                kinData["pathlength"],
                kinData["targetlength"],
                kinData["CursorX"],
                kinData["CursorY"],
                kinData["IA_RT"],
                kinData["IA_50RT"],
                kinData["RTalt"],
                kinData["IA_RTalt"],
                kinData["maxpathoffset"],
                kinData["meanpathoffset"],
                kinData["xTargetEnd"],
                kinData["yTargetEnd"],
                kinData["EndPointError"],
                kinData["IDE"],
                kinData["PLR"],
                kinData["isCurveAround"],
                i,
                kinData["x_intersect"],
                kinData["x_target_at_RT"],
                kinData["Delta_T_Used"],
                kinData["targetDist_Hit_Interception"],
            ]

            allTrials.append(row_values)
            subjectTrajs.append(trajData)

        # Create DataFrame for the current subject
        df = pd.DataFrame(
            allTrials,
            columns=[
                "Condition",
                "Affected",
                "TP",
                "Duration",
                "Accuracy",
                "FeedbackTime",
                "RT",
                "CT",
                "velPeak",
                "xPosError",
                "minDist",
                "targetDist",
                "handDist",
                "straightlength",
                "pathlength",
                "targetlength",
                "cursorX",
                "cursorY",
                "IA_RT",
                "IA_50RT",
                "RTalt",
                "IA_RTalt",
                "maxpathoffset",
                "meanpathoffset",
                "xTargetEnd",
                "yTargetEnd",
                "EndPointError",
                "IDE",
                "PLR",
                "isCurveAround",
                "trial_number",
                "x_intersect",
                "x_target_at_RT",
                "Delta_T_Used",
                "targetDist_Hit_Interception",
            ],
        )

        # Data cleaning
        df["Affected"] = df["Affected"].map({1: "More Affected", 0: "Less Affected"})
        df["Duration"] = df["TP"].map(
            {1: 625, 2: 500, 3: 750, 4: 900, 5: 625, 6: 500, 7: 750, 8: 900}
        )
        df["MT"] = df["CT"] - df["RT"]
        df["subject"] = subject
        df["age"] = row["Age at Visit (yr)"]
        df["visit"] = row["Visit ID"]
        df["day"] = row["Visit_Day"]
        df["studyid"] = row["Subject ID"]
        df["group"] = "TDC" if row["Group"] == 0 else "CP"
        df["pathratio"] = df["pathlength"] / df["targetlength"]

        # RANSAC regression with outlier tracking
        regression_models, data_by_subject_arm, subject_df, subject_outlier_summary = (
            reg.perform_subject_level_regression_ransac(df)
        )
        outlier_summaries.append(subject_outlier_summary)

        # Plot regression points (both Linear and RANSAC) and collect R² values
        rsquared_results_df = reg.plot_regression_points_ransac(
            subject,
            data_by_subject_arm,
            regression_models,
            row["Visit_Day"],
            row["Visit ID"],
            row["Subject ID"],
            RESULTS_DIR,
            rsquared_results_df,
        )
        rsquared_summaries.append(rsquared_results_df)

        # Calculate IE for Interception trials (RANSAC version)
        subject_wise_IE = reg.calculate_ie_for_interception_trials_ransac(
            subject_df, regression_models, subject, subjectTrajs, RESULTS_DIR
        )
        all_trials_ie_by_subject.append(subject_wise_IE)

        # Concatenate into main DataFrame
        all_df = pd.concat([all_df, df])
        allTrajs[subject + row["Visit_Day"]] = subjectTrajs

    # Save outlier summary
    if outlier_summaries:
        output_summaries_file = os.path.join(
            RESULTS_DIR, "Subject_Arm_Block_Level_Outliers.csv"
        )
        outlier_summary = pd.concat(outlier_summaries, ignore_index=True)
        outlier_summary.to_csv(output_summaries_file, index=False)

    # Save R² comparison (Linear vs Robust)
    if rsquared_summaries:
        regression_summary_file = os.path.join(RESULTS_DIR, "regression_results.csv")
        rsquared_summary = pd.concat(rsquared_summaries, ignore_index=True)
        rsquared_summary.to_csv(regression_summary_file, index=False)

    # Save IE results
    output_path = os.path.join(RESULTS_DIR, "IE_CSV_Results")
    os.makedirs(output_path, exist_ok=True)

    if len(all_trials_ie_by_subject) > 0:
        print(f"All Trials Data : {all_trials_ie_by_subject}")
        all_trials_ie_df = pd.concat(all_trials_ie_by_subject, ignore_index=True)
        grouped_data = all_trials_ie_df.sort_values(by=["Subject", "Condition"])
        file_name = os.path.join(output_path, "All_Trials_IE.csv")
        grouped_data_file_name = os.path.join(output_path, "Grouped_IE.csv")
        grouped_data.to_csv(grouped_data_file_name, index=False)
        all_trials_ie_df.to_csv(file_name, index=False)

    return all_df, allTrajs


# =============================================================================
# HELPERS
# =============================================================================


def convert_to_serializable(obj):
    """Recursively convert numpy arrays/dicts/lists to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


# =============================================================================
# FULL PIPELINE — single entry point
# =============================================================================


def run_pipeline(
    master_file,
    matfiles,
    defaults,
    results_dir,
    report_results_dir,
    pvar_results_dir,
    subject_filter=None,
):
    """
    Run the complete CHEAT-CP analysis from start to finish.

    Steps:
      1. Load master subject list from Excel
      2. Process all subjects (kinematics + RANSAC regression + IE)
      3. Save trial-level CSV to results_dir
      4. Generate all 6 Excel reports (means, stds, IDE, PVar)

    Parameters
    ----------
    master_file : str — Path to the master Excel file
    matfiles : str — Path to directory containing .mat files
    defaults : dict — Analysis defaults (fs, fc, fdfwd, etc.)
    results_dir : str — Where to save trial CSV, IE CSVs, outlier summaries
    report_results_dir : str — Where to save Excel report files
    pvar_results_dir : str — Directory containing pvar_results.csv
    subject_filter : list, optional — Only process these subject IDs
    """
    # Import reporting as a package so relative imports work
    from reporting.report_generator import main as run_reports

    # Step 1: Load master subject list
    mdf = load_master_excel(master_file)

    # Step 2: Run the RANSAC pipeline
    all_df, allTrajs = getDataCP_ransac(
        mdf, matfiles, defaults, results_dir, subject_filter=subject_filter
    )

    # Step 3: Save trial-level CSV
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "all_processed_trials_final.csv")
    all_df.to_csv(csv_path, index=False)
    print(f"Saved trial-level CSV to: {csv_path}")

    # Step 4: Compute PVar (uses allTrajs from Step 2 — no re-processing needed)
    print("\nComputing Path Variability (PVar)...")
    pv.compute_pvar(all_df, allTrajs, pvar_results_dir)

    # Step 5: Generate Excel reports (kinematic means/stds + PVar)
    run_reports(
        data_file_location=csv_path,
        report_results_dir=report_results_dir,
        pvar_results_dir=pvar_results_dir,
        master_df=mdf,
    )
    print("All reports generated successfully.")


if __name__ == "__main__":
    # When run directly from cp_analysis/, load Config from the parent directory
    from config import (
        MASTER_FILE,
        MATFILES_DIR,
        DEFAULTS,
        RESULTS_DIR,
        REPORT_RESULTS_DIR,
        PVAR_RESULTS_DIR,
    )

    run_pipeline(
        master_file=MASTER_FILE,
        matfiles=MATFILES_DIR,
        defaults=DEFAULTS,
        results_dir=RESULTS_DIR,
        report_results_dir=REPORT_RESULTS_DIR,
        pvar_results_dir=PVAR_RESULTS_DIR,
    )
