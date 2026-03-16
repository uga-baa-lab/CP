"""
Regression & Initial Estimate (IE) Module for CHEAT-CP Kinematic Analysis

This file contains ALL the regression-related code from both original files:

1. Simple Linear Regression approach):
   - perform_subject_level_regression() — uses LinearRegression with ±3 std cleaning
   - calculate_ie_for_interception_trials() — computes IE using a*x + b formula
   - plot_regression_points() — simple scatter plot with R² display

2. RANSAC / Robust Regression approach:
   - collect_data_by_arm() — collects Reaching data with IQR cleaning
   - perform_regression() — fits RANSAC model
   - perform_subject_level_regression_ransac() — full RANSAC pipeline with outlier tracking
   - calculate_ie_for_interception_trials_ransac() — IE with model.predict() and outlier plots
   - plot_regression_points_ransac() — TWO plots per arm (all data + inliers only), R² for both
   - update_outliers() — marks outlier trials in the DataFrame
   - count_outliers_by_subject_arm_duration() — outlier summary
   - plot_outlier_trials() — visual inspection of outlier trial trajectories
"""

import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score

import signal_processing as sp

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED HELPER FUNCTIONS
def initialize_arms_data_structure_for_regression():
    """Initialize empty data structure for collecting regression data by arm."""
    return {
        "Less Affected": {"x_intersect": [], "x_target": [], "indices": []},
        "More Affected": {"x_intersect": [], "x_target": [], "indices": []},
    }


def fix_data_types(subject_trials):
    """
    This method converts subject_trials to a DataFrame if it's a list of dicts.
    The RANSAC version expects a DataFrame, the simple version uses a list.
    """
    if isinstance(subject_trials, list):
        subject_trials = pd.DataFrame(subject_trials)
    elif not isinstance(subject_trials, pd.DataFrame):
        raise TypeError("subject_trials must be a DataFrame or a list of dictionaries.")
    return subject_trials


# ======================= SIMPLE LINEAR REGRESSION ===========================


def perform_subject_level_regression(subject_trials):
    """
    Performs SIMPLE linear regression for each arm using Reaching trials.
    Uses ±3 standard deviations for outlier removal.

    Parameters
    ----------
    subject_trials : list of dict
        List of kinData dictionaries for all trials of one subject.

    Returns
    -------
    regression_coeffs : dict
        {'More Affected': {'a': slope, 'b': intercept, 'model': LinearRegression},
         'Less Affected': {'a': slope, 'b': intercept, 'model': LinearRegression}}

    data_by_arm : dict
        Raw data used for regression, separated by arm.
    """
    # Initialize dictionaries to hold data for each arm
    data_by_arm = {
        "Less Affected": {"x_intersect": [], "x_target": []},
        "More Affected": {"x_intersect": [], "x_target": []},
    }

    # Collect data from Reaching trials
    for kinData in subject_trials:
        condition = kinData["Condition"]
        arm = kinData["Affected"]  # 'Less Affected' or 'More Affected'
        if (
            condition == "Reaching"
            and not np.isnan(kinData["x_intersect"])
            and not np.isnan(kinData["x_target_at_RT"])
        ):
            data_by_arm[arm]["x_intersect"].append(kinData["x_intersect"])
            data_by_arm[arm]["x_target"].append(kinData["x_target_at_RT"])

    # Perform regression for each arm
    regression_coeffs = {}
    for arm, data in data_by_arm.items():
        x_intersects = np.array(data["x_intersect"])
        x_targets = np.array(data["x_target"])

        # Clean using ±3 standard deviations
        mean = np.mean(x_intersects)
        std_dev = np.std(x_intersects)
        threshold = 3 * std_dev
        lower_bound = mean - threshold
        upper_bound = mean + threshold

        # Filter out outliers
        non_outlier_indices = (x_intersects >= lower_bound) & (
            x_intersects <= upper_bound
        )
        x_intersects_filtered = x_intersects[non_outlier_indices]
        x_targets_filtered = x_targets[non_outlier_indices]

        # Fit simple linear regression
        X = x_intersects_filtered.reshape(-1, 1)
        y = x_targets_filtered
        model = LinearRegression()
        model.fit(X, y)
        a = model.coef_[0]
        b = model.intercept_
        regression_coeffs[arm] = {"a": a, "b": b, "model": model}

    return regression_coeffs, data_by_arm


def calculate_ie_for_interception_trials(subject_trials, regression_coeffs, subject):
    """
    Applies the regression model to Interception trials to compute IE.
    Uses a * x_intersect + b formula directly.

    This is the ORIGINAL method from CHEATCP_Final_With_IE.py.

    Parameters
    ----------
    subject_trials : list of dict
        List of kinData dicts for all trials of one subject.

    regression_coeffs : dict
        {'arm': {'a': slope, 'b': intercept, 'model': model}}

    subject : str
        Subject ID.

    Returns
    -------
    result_df : pandas.DataFrame
        Columns: ['IE', 'Duration', 'Condition', 'Subject']
    """
    trial_list = []
    for kinData in subject_trials:
        condition = kinData["Condition"]
        arm = kinData["Affected"]
        duration = kinData["Duration"]
        a = regression_coeffs[arm]["a"]
        b = regression_coeffs[arm]["b"]

        if (
            condition == "Interception"
            and not np.isnan(kinData["x_intersect"])
            and not np.isnan(a)
            and not np.isnan(b)
        ):
            x_predicted = a * kinData["x_intersect"] + b
            IE = x_predicted - kinData["x_target_at_RT"]
            kinData["IE"] = IE
        else:
            kinData["IE"] = np.nan
        trial_list.append([kinData["IE"], duration, condition, subject])
        result_df = pd.DataFrame(
            trial_list, columns=["IE", "Duration", "Condition", "Subject"]
        )
    return result_df


# These plots are for more unerstanding on how regression working for us
def plot_regression_points(subject, data_by_arm, regression_coeffs, visit_day):
    """
    Plot regression scatter plots — One plot per arm and it shows all data points with regression line and R².
    """
    for arm in data_by_arm.keys():
        x_positions_list = np.array(data_by_arm[arm]["x_intersect"])
        x_target_rt_list = np.array(data_by_arm[arm]["x_target"])
        print(f"Size of X_intersect points for arm {arm}: {len(x_positions_list)}")

        if len(x_positions_list) == 0 or len(x_target_rt_list) == 0:
            print(f"No valid data for arm {arm} of subject {subject}")
            continue

        plt.figure(figsize=(8, 6))
        plt.scatter(
            x_positions_list, x_target_rt_list, color="blue", label="Data Points"
        )

        # Get regression coefficients
        a = regression_coeffs[arm]["a"]
        b = regression_coeffs[arm]["b"]

        if not np.isnan(a) and not np.isnan(b):
            y_pred = a * x_positions_list + b
            sorted_indices = np.argsort(x_positions_list)
            x_sorted = x_positions_list[sorted_indices]
            y_pred_sorted = y_pred[sorted_indices]
            plt.plot(x_sorted, y_pred_sorted, color="red", label="Regression Line")

            # Calculate R²
            ss_res = np.sum((x_target_rt_list - y_pred) ** 2)
            ss_tot = np.sum((x_target_rt_list - np.mean(x_target_rt_list)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            equation_text = f"y = {a:.2f}x + {b:.2f}\n$R^2$ = {r_squared:.2f}"
            plt.text(
                0.05,
                0.95,
                equation_text,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
            )
        else:
            plt.text(
                0.05,
                0.95,
                "Insufficient data for regression",
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
                color="red",
            )

        plt.xlabel("Intersection Point (x_intersect)")
        plt.ylabel("Target Position at RT (x_target_at_RT)")
        plt.title(f"Subject: {subject} - Day {visit_day}, Arm: {arm} - Reaching Trials")
        plt.legend()
        plt.grid(True)
        subject_folder = os.path.join("path_to_results", subject)
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)
        plot_filename = f"{subject}_{visit_day}_{arm}.png"
        plt.savefig(os.path.join(subject_folder, plot_filename), bbox_inches="tight")
        plt.close()


# ======================= RANSAC / ROBUST REGRESSION =========================


def collect_data_by_arm(subject_trials):
    """
    Collect Reaching trial data by arm, with IQR cleaning.
    Only considers x_intersect values within ±1000 range.

    Parameters
    ----------
    subject_trials : pandas.DataFrame
        DataFrame with all trials for one subject.

    Returns
    -------
    data_by_arm : dict
        Cleaned data separated by arm, ready for regression.
    """

    """ We're considering only those trials where x_intersect values are less than 1000 - Just hardcoding it! 
    And later anyhow we're removing the outliers by IQR but just for the sake of it we're doing it explicitly. This was based on Discussion with owais"""

    subject_trials = subject_trials[
        (subject_trials["x_intersect"] <= 1000)
        & (subject_trials["x_intersect"] >= -1000)
    ]

    data_by_arm = initialize_arms_data_structure_for_regression()
    for idx, kinData in subject_trials.iterrows():
        condition = kinData["Condition"]
        arm = kinData["Affected"]
        if (
            condition == "Reaching"
            and not np.isnan(kinData["x_intersect"])
            and not np.isnan(kinData["x_target_at_RT"])
        ):
            data_by_arm[arm]["x_intersect"].append(kinData["x_intersect"])
            data_by_arm[arm]["x_target"].append(kinData["x_target_at_RT"])
            data_by_arm[arm]["indices"].append(int(idx))

    # Apply IQR cleaning
    for arm in data_by_arm.keys():
        x = np.array(data_by_arm[arm]["x_intersect"])
        y = np.array(data_by_arm[arm]["x_target"])
        indices = np.array(data_by_arm[arm]["indices"])

        if len(x) > 0 and len(y) > 0:
            x_clean, y_clean, indices_clean, filtered_indices = sp.clean_data_iqr(
                x, y, indices
            )
            data_by_arm[arm]["x_intersect"] = x_clean.tolist()
            data_by_arm[arm]["x_target"] = y_clean.tolist()
            data_by_arm[arm]["indices"] = indices_clean.tolist()

    return data_by_arm


def perform_regression(x_clean, y_clean):
    """
    Fit a RANSAC regression model.

    Parameters
    ----------
    x_clean : numpy.ndarray
        Cleaned x_intersect values.
    y_clean : numpy.ndarray
        Cleaned x_target values.

    Returns
    -------
    ransac : RANSACRegressor or None
        Fitted model, or None if not enough data.
    inlier_mask : numpy.ndarray or None
        Boolean mask (True = inlier, False = outlier).
    """
    if len(x_clean) >= 2:
        X = x_clean.reshape(-1, 1)
        ransac = RANSACRegressor(LinearRegression(), random_state=42)
        ransac.fit(X, y_clean)
        return ransac, ransac.inlier_mask_
    else:
        return None, None


def update_outliers(subject_trials, clean_indices, inlier_mask):
    """
    Mark outlier trials in the DataFrame based on RANSAC inlier_mask.
    From InitialEstimateFinalCode.py.
    """
    if inlier_mask is not None:
        outlier_mask = np.logical_not(inlier_mask)
        outlier_indices = clean_indices[outlier_mask]
        subject_trials.loc[outlier_indices, "Outlier_Trial"] = True


def count_outliers_by_subject_arm_duration(subject_trials):
    """
    Count outliers broken down by subject, arm, and duration.
    From InitialEstimateFinalCode.py.

    Returns a DataFrame with outlier counts per Subject_Visit × Arm × Duration.
    """
    arms = subject_trials["Affected"].unique()
    durations = (
        subject_trials["Duration"].unique()
        if "Duration" in subject_trials.columns
        else [500, 625, 750, 900]
    )
    subject_ids = (
        subject_trials["studyid"].unique()
        if "studyid" in subject_trials.columns
        else ["Subject"]
    )
    visit_days = (
        subject_trials["visit"].unique()
        if "visit" in subject_trials.columns
        else ["visit"]
    )

    records = []
    for subject_id in subject_ids:
        for visit_day in visit_days:
            combined_id = f"{subject_id}{visit_day}"
            record = {"Subject_Visit": combined_id}
            total_outliers = 0
            for arm in arms:
                for duration in durations:
                    key = f"{arm}_{duration}"
                    subject_arm_duration_data = subject_trials[
                        (subject_trials["Affected"] == arm)
                        & (subject_trials["studyid"] == subject_id)
                        & (subject_trials["Duration"] == duration)
                        & (subject_trials["visit"] == visit_day)
                    ]
                    outlier_count = subject_arm_duration_data["Outlier_Trial"].sum()
                    total_outliers += outlier_count
                    record[key] = outlier_count
            record["Total Outlier Count"] = total_outliers
            records.append(record)

    df = pd.DataFrame(records)
    sorted_columns = sorted([col for col in df.columns if col not in ["Subject_Visit"]])
    return df[["Subject_Visit"] + sorted_columns]


def perform_subject_level_regression_ransac(subject_trials):
    """
    Performs RANSAC regression for each arm, with outlier tracking.
    Make sure that the trials received for subjects are data frames. If it's a list type convert them into Dataframes, It's basically the all the KinData of trials for a particular suject

    Step 1 : Add a new column Outlier_Trial to the dataframe
    Step 2 : Now create a dict with 2 arms - Less Affected and More Affected and then add the relevant data you need to for performing regression
    Step 3 : Clean the data before passing it to the regression - Apply this +=3 stds for cleaning the data
    Step 4 : Figure out the outlier indices and mark the Outlier_Trial flag as true for the coressponding indexes.



    Parameters
    ----------
    subject_trials : list or DataFrame
        All trial data for one subject. Converted to DataFrame internally.

    Returns
    -------
    regression_models : dict
        {'arm': {'model': RANSACRegressor or None}}

    data_by_arm : dict
        Cleaned data by arm (post-IQR).

    subject_trials : DataFrame
        Updated DataFrame with 'Outlier_Trial' column.

    outlier_summary : DataFrame
        Outlier counts by subject × arm × duration.
    """
    subject_trials = fix_data_types(subject_trials)
    subject_trials["Outlier_Trial"] = (
        False  # Add 'Outlier_Trial' column to the DataFrame, initialize to False
    )

    print(f"Total Subject trials received for subject : {len(subject_trials)}")

    regression_models = {}
    outlier_summary = pd.DataFrame()
    data_by_arm = collect_data_by_arm(
        subject_trials
    )  # Collect data by arm [less affected and more affected]

    for arm, data in data_by_arm.items():
        x = np.array(data["x_intersect"])
        print(f"Len of x_intersect values after dropping NA records : {len(x)}")
        y = np.array(data["x_target"])
        print(f"Length of x_target records after dropping NA records : {len(y)}")
        clean_indices = np.array(data["indices"], dtype=int)
        # now clean the data by removing the values which are greater than +- 3 stds from the mean
        # basically I need a method to clean the data of X and y- remove values > +=3 stds
        if len(x) >= 2:
            model, inlier_mask = perform_regression(x, y)
            if model:
                regression_models[arm] = {"model": model}
                update_outliers(subject_trials, clean_indices, inlier_mask)
                outlier_summary = count_outliers_by_subject_arm_duration(subject_trials)
            else:
                print(
                    f"Not enough cleaned data points for arm {arm} to perform regression."
                )
                regression_models[arm] = {"model": None}
        else:
            print(f"Not enough data points for arm {arm} to perform regression.")
            regression_models[arm] = {"model": None}

    return regression_models, data_by_arm, subject_trials, outlier_summary


def plot_regression_points_ransac(
    subject,
    data_by_arm,
    regression_models,
    visit_day,
    visit_id,
    study_id,
    RESULTS_DIR,
    results_df,
):
    """
    Plot regression scatter plots — with RANSAC version we're generating TWO plots per arm for more detailed understanding:
      1. All data points with simple Linear Regression line + R²
      2. Inliers only with RANSAC regression line + R²

    Also this returns a DataFrame with R² values for both Linear and Robust fits.

    Parameters
    ----------
    subject : str
        Subject ID.
    data_by_arm : dict
        Data from collect_data_by_arm().
    regression_models : dict
        From perform_subject_level_regression_ransac().
    visit_day : str
        Visit day string (e.g., 'Day1').
    visit_id : str
        Visit identifier.
    study_id : str
        Study-level subject identifier.
    RESULTS_DIR : str
        Where to save plots.
    results_df : DataFrame
        Existing results DataFrame to append to.

    Returns
    -------
    DataFrame with new R² rows appended.
    """
    new_rows = []
    for arm, data in data_by_arm.items():
        print(
            f'for arm : {arm} length of data is - plot func: {len(data["x_intersect"])}'
        )

    for arm in data_by_arm.keys():
        x_positions = np.array(data_by_arm[arm]["x_intersect"])
        print(f"Length of x_positions in plot regression points : {len(x_positions)}")
        x_targets = np.array(data_by_arm[arm]["x_target"])
        indices = data_by_arm[arm]["indices"]

        if len(x_positions) == 0 or len(x_targets) == 0:
            print(f"No valid data for arm {arm} of subject {subject}")
            continue

        r_squared_all = np.nan
        r_squared_inliers = np.nan

        x_positions = x_positions.flatten()
        x_targets = x_targets.flatten()
        X = x_positions.reshape(-1, 1)
        print(f"Length of reshaped data : {len(X)}")
        y = x_targets
        print(f"Length of y variable : {len(y)}")

        # --- Plot 1: All data points with Linear Regression line ---
        if len(x_positions) >= 2:
            model_all = LinearRegression()
            model_all.fit(X, y)

            a_all = model_all.coef_[0]
            b_all = model_all.intercept_

            y_pred_all = model_all.predict(X)
            r_squared_all = r2_score(y, y_pred_all)

            # Generate x values for plotting the regression line
            x_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_fit_all = model_all.predict(x_fit)

            plt.figure(figsize=(8, 6))
            plt.scatter(X, y, color="blue", label="Data Points")
            plt.plot(
                x_fit, y_fit_all, color="green", label="Regression Line (All Data)"
            )

            equation_text = (
                f"y = {a_all:.2f}x + {b_all:.2f}\n$R^2$ = {r_squared_all:.2f}"
            )
            plt.text(
                0.05,
                0.95,
                equation_text,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
            )

            plt.xlabel("Intersection Point (x_intersect)")
            plt.ylabel("Target Position at RT (x_target_at_RT)")
            plt.title(f"Subject: {subject} - {visit_day}, Arm: {arm} - All Data")
            plt.legend()
            plt.grid(True)

            subject_folder = os.path.join(RESULTS_DIR, f"{subject}")
            subject_folder = os.path.join(subject_folder, "Robust_Reg")
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder, exist_ok=True)

            plot_filename = f"{subject}_{visit_day}_{arm}_all_data.png"
            plt.savefig(
                os.path.join(subject_folder, plot_filename), bbox_inches="tight"
            )
            plt.close()
        else:
            print(
                f"Not enough data points for regression for arm {arm} of subject {subject}"
            )

        # --- Plot 2: Inliers only with RANSAC regression line ---
        model_info = regression_models.get(arm)
        if model_info and model_info.get("model"):
            ransac = model_info["model"]

            # Get inlier mask
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            # Get coefficients from RANSAC model
            a_inliers = ransac.estimator_.coef_[0]
            b_inliers = ransac.estimator_.intercept_

            # R² using only inliers
            X_inliers = X[inlier_mask]
            y_inliers = y[inlier_mask]
            y_pred_inliers = ransac.predict(X_inliers)
            r_squared_inliers = r2_score(y_inliers, y_pred_inliers)

            # Generate x values for plotting the regression line
            x_fit_inliers = np.linspace(X_inliers.min(), X_inliers.max(), 100).reshape(
                -1, 1
            )
            y_fit_inliers = ransac.predict(x_fit_inliers)

            # Plot inliers and outliers
            plt.figure(figsize=(8, 6))
            plt.scatter(X_inliers, y_inliers, color="blue", label="Inliers")
            if np.any(outlier_mask):
                plt.scatter(
                    X[outlier_mask], y[outlier_mask], color="red", label="Outliers"
                )

            plt.plot(
                x_fit_inliers,
                y_fit_inliers,
                color="green",
                label="Regression Line (Inliers Only)",
            )

            equation_text = f"y = {a_inliers:.2f}x + {b_inliers:.2f}\n$R^2$ = {r_squared_inliers:.2f}"
            plt.text(
                0.05,
                0.95,
                equation_text,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
            )

            plt.xlabel("Intersection Point (x_intersect)")
            plt.ylabel("Target Position at RT (x_target_at_RT)")
            plt.title(f"Subject: {subject} - {visit_day}, Arm: {arm} - Inliers Only")
            plt.legend()
            plt.grid(True)

            plot_filename = f"{subject}_{visit_day}_{arm}_inliers_only.png"
            plt.savefig(
                os.path.join(subject_folder, plot_filename), bbox_inches="tight"
            )
            plt.close()
        else:
            print(
                f"Not enough data points for RANSAC regression for arm {arm} of subject {subject}"
            )

        # Collect R² values for both methods
        new_rows.append(
            {
                "Subject": f"{study_id}{visit_id}",
                "Visit Day": visit_day,
                "Arm": arm,
                "R^2_Linear": r_squared_all,
                "R^2_Robust": r_squared_inliers,
            }
        )

    # Append new R² rows to the results DataFrame
    if new_rows:
        new_rows_df = pd.DataFrame(new_rows)
        return new_rows_df

    return results_df


def plot_outlier_trials(
    subject, outlier_trials, subjectTrajs, all_trials_df, RESULTS_DIR
):
    """
    Plot outlier trial trajectories for visual inspection.
    From InitialEstimateFinalCode.py.

    Shows the hand path, initial movement vector, extended projection line,
    x_intersect point, target path, and target position at RT.
    """
    for idx, trial_info in outlier_trials.iterrows():
        trial_number = trial_info["Trial_Index"]
        kinData = all_trials_df.iloc[idx]
        trajData = subjectTrajs[trial_number]
        HandX_filt = trajData["HandX_filt"]
        HandY_filt = trajData["HandY_filt"]
        xTargetPos = trajData["xTargetPos"]
        yTargetPos = trajData["yTargetPos"]
        CursorX = trajData["CursorX"]
        CursorY = trajData["CursorY"]
        delta_t_used = kinData["Delta_T_Used"]

        RT = int(kinData["RT"]) if not np.isnan(kinData["RT"]) else None
        x_intersect = kinData["x_intersect"]
        y_target_at_RT = yTargetPos[RT] if RT is not None else np.nan

        plt.figure(figsize=(8, 6))
        plt.title(f"Outlier Trial: Subject {subject}, Trial {trial_number}")

        # Plot hand path
        plt.plot(HandX_filt, HandY_filt, label="Hand Path", color="blue")

        # Plot initial movement vector
        if RT is not None:
            delta_t = 50
            if not np.isnan(delta_t_used):
                delta_t = int(delta_t_used)

            RT_plus_delta = min(RT + delta_t, len(HandX_filt) - 1)
            if RT >= len(HandX_filt) - 1:
                print(
                    f"RT index {RT} is too close to the end of the array for reliable plotting."
                )
                return

            plt.plot(
                [HandX_filt[RT], HandX_filt[RT_plus_delta]],
                [HandY_filt[RT], HandY_filt[RT_plus_delta]],
                label="Initial Movement",
                color="green",
                linewidth=2,
            )

            # Extend initial movement vector to intersect with y = y_target_at_RT
            vx = HandX_filt[RT_plus_delta] - HandX_filt[RT]
            vy = HandY_filt[RT_plus_delta] - HandY_filt[RT]
            if vx != 0:
                m = vy / vx
                c = HandY_filt[RT] - m * HandX_filt[RT]
                x_vals = np.array([HandX_filt[RT], x_intersect])
                y_vals = m * x_vals + c
                plt.plot(
                    x_vals,
                    y_vals,
                    linestyle="--",
                    color="green",
                    label="Extended Initial Movement",
                )
                plt.scatter(
                    x_intersect, y_target_at_RT, color="red", label="x_intersect"
                )

        # Plot target path
        plt.plot(xTargetPos, yTargetPos, label="Target Path", color="orange")

        # Plot target position at RT
        if RT is not None:
            x_target_at_RT = xTargetPos[RT]
            plt.scatter(
                x_target_at_RT, y_target_at_RT, color="purple", label="Target at RT"
            )

        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        subject_folder = os.path.join(RESULTS_DIR, subject, "Outlier_Trials")
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder, exist_ok=True)
        plot_filename = f"Outlier_Trial_{subject}_Trial_{trial_number}.png"
        plt.savefig(os.path.join(subject_folder, plot_filename), bbox_inches="tight")
        plt.close()


def calculate_ie_for_interception_trials_ransac(
    subject_trials, regression_models, subject, subjectTrajs, RESULTS_DIR
):
    """
    Compute IE for Interception trials using RANSAC regression model.

    Uses model.predict() instead of a*x + b formula.
    Also saves per-subject IE CSV and plots outlier trials.

    Parameters
    ----------
    subject_trials : DataFrame
        All trial data (with 'Outlier_Trial' column already set).
    regression_models : dict
        From perform_subject_level_regression_ransac().
    subject : str
        Subject ID.
    subjectTrajs : list
        Trajectory data for outlier plotting.
    RESULTS_DIR : str
        Where to save per-subject IE CSVs and outlier plots.

    Returns
    -------
    result_df : DataFrame
        Columns: Trial_Index, IE, Duration, Condition, Subject,
        x_intersect, x_target_at_RT, Outlier_Trial, Arm, Is_Above_Threshold
    """
    trial_list = []

    for index, kinData in subject_trials.iterrows():
        condition = kinData["Condition"]
        arm = kinData["Affected"]
        duration = kinData["Duration"]
        x_intersect = kinData["x_intersect"]
        x_target_at_RT = kinData["x_target_at_RT"]
        IE = np.nan
        x_predicted = np.nan
        is_above_threshold = False

        if (
            condition == "Interception"
            and not np.isnan(x_intersect)
            and not np.isnan(x_target_at_RT)
        ):
            model_info = regression_models.get(arm)
            if model_info and model_info.get("model"):
                model = model_info["model"]
                try:
                    X_new = np.array([[x_intersect]])
                    x_predicted = model.predict(X_new)[0]
                    IE = x_predicted - x_target_at_RT
                except Exception as e:
                    print(f"Prediction failed for subject {subject}, arm {arm}: {e}")

        subject_trials.at[index, "IE"] = IE
        if IE > 600:
            is_above_threshold = True
            subject_trials.at[index, "is_abnormal_IE"] = True
            print(
                f"IE value is greater than 600 for subject {subject} and arm {arm} and trial {index}"
            )

        trial_list.append(
            [
                index,
                IE,
                duration,
                condition,
                subject,
                x_intersect,
                x_target_at_RT,
                kinData["Outlier_Trial"],
                arm,
                is_above_threshold,
            ]
        )

    result_df = pd.DataFrame(
        trial_list,
        columns=[
            "Trial_Index",
            "IE",
            "Duration",
            "Condition",
            "Subject",
            "x_intersect",
            "x_target_at_RT",
            "Outlier_Trial",
            "Arm",
            "Is_Above_Threshold",
        ],
    )

    # Save per-subject IE CSV
    csv_filename = f"IE_values_{subject}.csv"
    results_dir = os.path.join(RESULTS_DIR, subject)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_df_save_loc = os.path.join(results_dir, csv_filename)
    result_df.to_csv(result_df_save_loc, index=False)

    # Plot outlier trials
    outlier_trials = result_df[result_df["Outlier_Trial"] == True]
    if not outlier_trials.empty:
        print(f"Plotting {len(outlier_trials)} outlier trials for subject {subject}.")
        plot_outlier_trials(
            subject, outlier_trials, subjectTrajs, subject_trials, RESULTS_DIR
        )
    else:
        print(f"No outlier trials to plot for subject {subject}.")

    return result_df
