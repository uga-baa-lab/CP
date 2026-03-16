"""
Path Variability (PVar) Computation
=====================================

Implements the documented PVar pipeline:

  1. Time-Normalization    : Normalize each trajectory to 150 time points
  2. Mean Trajectory       : Compute ONE mean trajectory per subject across
                             ALL their Reaching trials (all arms, all durations)
  3. Deviation Computation : Subtract the subject-level mean from each trial
  4. Grouping              : Group deviations by (Arm, Duration)
  5. Pvar                  : Sum of std of deviations at each time point

HOW TO USE:
    from cp_analysis.pvar import compute_pvar

    pvar_df = compute_pvar(all_df, allTrajs, pvar_results_dir)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

NUM_POINTS = 150


# =============================================================================
# STEP 1: TIME NORMALIZATION
# =============================================================================


def time_normalize_trial(HandX, HandY, RT, CT, num_points=NUM_POINTS):
    """
    Slice RT→CT and resample to num_points using spline interpolation.
    Returns (normX, normY) or (None, None) if the trial is invalid.
    """
    if pd.isna(RT) or pd.isna(CT):
        return None, None

    RT, CT = int(RT), int(CT)

    if RT < 0 or CT > len(HandX) or RT >= CT:
        return None, None

    segX = HandX[RT:CT]
    segY = HandY[RT:CT]

    if len(segX) < 2:
        return None, None

    t_orig = np.linspace(0, 1, len(segX))
    t_new = np.linspace(0, 1, num_points)

    try:
        normX = splev(t_new, splrep(t_orig, segX, s=0))
        normY = splev(t_new, splrep(t_orig, segY, s=0))
        return np.array(normX), np.array(normY)
    except Exception as e:
        print(f"    Interpolation error: {e}")
        return None, None


# =============================================================================
# STEP 2: SUBJECT-LEVEL MEAN TRAJECTORY (across all Reaching trials)
# =============================================================================


def compute_subject_mean(norm_trials_x, norm_trials_y):
    """
    Compute the subject-level mean trajectory across ALL Reaching trials
    (all arms, all durations combined).

    Parameters
    ----------
    norm_trials_x, norm_trials_y : list of np.ndarray (each length NUM_POINTS)

    Returns
    -------
    mean_x, mean_y : np.ndarray of shape (NUM_POINTS,), or (None, None)
    """
    if len(norm_trials_x) < 2:
        return None, None
    return np.mean(norm_trials_x, axis=0), np.mean(norm_trials_y, axis=0)


# =============================================================================
# STEP 3–5: PVar per block (Arm × Duration)
# =============================================================================


def compute_pvar_for_block(deviations_x, deviations_y):
    """
    Given deviations (trial - subject_mean) for a block of trials,
    compute Pvar = sum of std of deviations at each time point.

    Parameters
    ----------
    deviations_x, deviations_y : np.ndarray, shape (n_trials, NUM_POINTS)

    Returns
    -------
    Pvar : float
    """
    std_x = np.std(deviations_x, axis=0, ddof=1)
    std_y = np.std(deviations_y, axis=0, ddof=1)
    return np.sum(std_x + std_y)


# =============================================================================
# PLOTTING
# =============================================================================


def plot_group_trajectories(norm_x, norm_y, mean_x, mean_y, group_key, save_dir):
    """
    Plot individual normalized trials (blue, faint) and subject mean (red).
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for i in range(len(norm_x)):
        plt.plot(norm_x[i], norm_y[i], color="blue", alpha=0.2)

    plt.plot(mean_x, mean_y, color="red", linewidth=2, label="Subject Mean")

    plt.title(f"Group: {group_key}\n{len(norm_x)} trials")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    filename = "_".join([str(x) for x in group_key]) + ".png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


# =============================================================================
# MAIN  Function
# =============================================================================


def compute_pvar(all_df, allTrajs, pvar_results_dir):
    """
    Compute Path Variability (Pvar) following the documented pipeline.

    Parameters
    ----------
    all_df : DataFrame
        Trial-level data from getDataCP_ransac(). Must have columns:
        subject, day, Condition, Affected, Duration, RT, CT.

    allTrajs : dict
        Keyed by '{subject}{day}'. Each value is a list of trajData dicts
        with keys 'HandX_filt', 'HandY_filt'.

    pvar_results_dir : str
        Where to save pvar_results.csv and trajectory plots.

    Returns
    -------
    pvar_df : DataFrame
        Columns: Subject, Day, Arm, Duration, Pvar, nValidTrials
    """
    os.makedirs(pvar_results_dir, exist_ok=True)
    plots_dir = os.path.join(pvar_results_dir, "PVAR_Plots")

    all_df = all_df.reset_index(drop=True)
    pvar_results = []

    for subject in all_df["subject"].unique():
        subject_df = all_df[all_df["subject"] == subject]

        for day in subject_df["day"].unique():
            day_df = subject_df[subject_df["day"] == day].reset_index(drop=True)
            traj_key = f"{subject}{day}"

            if traj_key not in allTrajs:
                print(f"  No trajectory data for {traj_key}, skipping.")
                continue

            traj_list = allTrajs[traj_key]

            if len(traj_list) != len(day_df):
                print(
                    f"  Trajectory count mismatch for {traj_key} "
                    f"(expected {len(day_df)}, got {len(traj_list)}), skipping."
                )
                continue

            # ------------------------------------------------------------------
            # STEP 1 + 2: Time-normalize ALL Reaching trials and compute
            #             the single subject-level mean trajectory
            # ------------------------------------------------------------------
            reaching_mask = day_df["Condition"] == "Reaching"
            all_norm_x, all_norm_y = [], []
            # Store per-trial normalized data alongside row metadata
            trial_records = []  # list of (row_idx, normX, normY, arm, duration)

            for local_idx, row in day_df[reaching_mask].iterrows():
                traj = traj_list[local_idx]
                HandX = np.array(traj.get("HandX_filt", []))
                HandY = np.array(traj.get("HandY_filt", []))

                if len(HandX) == 0 or np.isnan(HandX).any():
                    continue

                normX, normY = time_normalize_trial(HandX, HandY, row["RT"], row["CT"])
                if normX is None:
                    continue

                all_norm_x.append(normX)
                all_norm_y.append(normY)
                trial_records.append((normX, normY, row["Affected"], row["Duration"]))

            if len(all_norm_x) < 2:
                print(f"  {traj_key}: fewer than 2 valid Reaching trials, skipping.")
                continue

            # Subject-level mean across ALL Reaching trials
            mean_x, mean_y = compute_subject_mean(
                np.array(all_norm_x), np.array(all_norm_y)
            )

            # ------------------------------------------------------------------
            # STEP 3: Compute per-trial deviations from subject mean
            # ------------------------------------------------------------------
            for normX, normY, arm, duration in trial_records:
                # attach deviation alongside trial metadata
                pass  # deviations computed per-block below

            # ------------------------------------------------------------------
            # STEP 4 + 5: Group deviations by (Arm, Duration) → Pvar
            # ------------------------------------------------------------------
            # Build lookup: (arm, duration) → list of (devX, devY)
            block_deviations = {}
            block_trials = {}  # for plotting

            for normX, normY, arm, duration in trial_records:
                bkey = (arm, duration)
                dev_x = normX - mean_x
                dev_y = normY - mean_y
                block_deviations.setdefault(bkey, ([], []))
                block_deviations[bkey][0].append(dev_x)
                block_deviations[bkey][1].append(dev_y)
                block_trials.setdefault(bkey, ([], []))
                block_trials[bkey][0].append(normX)
                block_trials[bkey][1].append(normY)

            for (arm, duration), (dev_x_list, dev_y_list) in block_deviations.items():
                n_valid = len(dev_x_list)
                if n_valid < 2:
                    print(
                        f"    {traj_key} {arm} {duration}: only {n_valid} trial(s), skipping."
                    )
                    continue

                dev_x_arr = np.array(dev_x_list)  # (n_trials, NUM_POINTS)
                dev_y_arr = np.array(dev_y_list)

                Pvar = compute_pvar_for_block(dev_x_arr, dev_y_arr)

                pvar_results.append(
                    {
                        "Subject": subject,
                        "Day": day,
                        "Arm": arm,
                        "Duration": duration,
                        "Pvar": Pvar,
                        "nValidTrials": n_valid,
                    }
                )

                print(
                    f"    {subject} {day} {arm} {duration}: "
                    f"Pvar={Pvar:.4f} ({n_valid} trials)"
                )

                group_key = (subject, day, arm, duration)
                norm_x_arr, norm_y_arr = block_trials[(arm, duration)]
                plot_group_trajectories(
                    norm_x_arr, norm_y_arr, mean_x, mean_y, group_key, plots_dir
                )

    pvar_df = pd.DataFrame(pvar_results)

    if pvar_df.empty:
        print("No PVar results computed — check your data.")
        return pvar_df

    csv_path = os.path.join(pvar_results_dir, "pvar_results.csv")
    pvar_df.to_csv(csv_path, index=False)
    print(f"\nPVar results saved to: {csv_path}")
    print(pvar_df)

    return pvar_df
