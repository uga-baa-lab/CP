"""
Trial Plotting Functions for CHEAT-CP Kinematic Analysis
=========================================================

This module provides visualization functions for inspecting individual trials,
trajectory paths, and group-level kinematic summaries from the CHEAT-CP task.

Functions fall into three categories:

1. SINGLE TRIAL INSPECTION
   - plot_trial_IDE()    : Detailed single-trial plot showing hand/cursor paths,
                           ideal vs participant movement vectors, and the IDE angle.
                           Useful for debugging kinematic calculations or verifying
                           that IDE, accuracy, and endpoint error make sense visually.

   - plot_singletraj()   : Minimal single-trial plot — just the cursor path with
                           RT and RTalt markers. Quick visual check of one trial.

2. TRAJECTORY OVERVIEW
   - plot_trajectories_range() : Plot a range of trials (e.g., trials 10-20) for
                                  one subject on one day. Useful for spotting movement
                                  pattern changes across a block of trials.

   - plot_filtered_hand_trajectories() : Arc-length normalized hand paths with
                                          spline resampling. Separates left/right
                                          reaching trajectories and computes average
                                          paths per side. Used for comparing movement
                                          symmetry between affected/less-affected arms.

3. GROUP SUMMARIES (Seaborn faceted plots)
   - plot_groupmeans()      : CP vs TDC group means, faceted by Condition
                              (Reaching/Interception), colored by arm (Affected).
   - plot_byduration()      : Same but grouped by target duration (500/625/750/900ms).
   - plot_twocolumnbar()    : Generic two-column bar+strip plot for any variable.

ORIGINAL FILE: TrialPlots.py
"""

import os
import uuid
import logging as logger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.interpolate import splprep, splev

# =============================================================================
# 1. SINGLE TRIAL INSPECTION
# =============================================================================


def plot_trial_IDE(
    kinData,
    HandX_filt,
    HandY_filt,
    xTargetPos,
    yTargetPos,
    theta_deg,
    i,
    subject,
    CursorX,
    CursorY,
    velX_filt,
    velY_filt,
    results_dir,
):
    """
    Plot a detailed single-trial visualization for IDE inspection.

    Shows:
    - Hand path (blue) and cursor path (light blue)
    - Start/end positions
    - Target position at RT+50ms and at completion time
    - Participant movement vector vs ideal movement vector
    - Annotations: IDE angle, condition, endpoint error, PLR, accuracy,
      cursor/hand/target coordinates at completion

    The plot is saved to: {results_dir}/trial_plots/{subject}/{subject}_{trial}_{uuid}.png

    Parameters
    ----------
    kinData : dict
        Kinematic data for this trial (from compute_trial_kinematics).
    HandX_filt, HandY_filt : array-like
        Filtered hand position time series (mm).
    xTargetPos, yTargetPos : array-like
        Target position time series (mm).
    theta_deg : float
        IDE angle in degrees.
    i : int
        Trial index number.
    subject : str
        Subject ID (e.g., '001').
    CursorX, CursorY : array-like
        Cursor position time series (mm).
    velX_filt, velY_filt : array-like
        Filtered velocity time series (currently unused, reserved for future).
    results_dir : str
        Base results directory path.
    """
    completion_time = int(kinData["CT"])
    reaction_time_50 = int(kinData["RT"] + 50)
    reaction_time_100 = int(kinData["RT"] + 100)
    target_x = xTargetPos[reaction_time_50]
    target_y = yTargetPos[reaction_time_50]
    target_x_at_ct = xTargetPos[completion_time]

    accuracy_value = kinData["Accuracy"]
    end_point_error = kinData["EndPointError"]
    path_length_ratio = kinData["PLR"]
    condition = kinData["Condition"]

    # Create folder for subject if it doesn't exist
    folder_for_trial_plots = os.path.join(results_dir, "trial_plots")
    subject_folder = os.path.join(folder_for_trial_plots, subject)
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)

    plt.figure(figsize=(12, 10))
    plt.plot(HandX_filt[0:], HandY_filt[0:], label="Participant Path", color="blue")
    plt.plot(CursorX[0:], CursorY[0:], label="Cursor Path", color="deepskyblue")
    plt.scatter(HandX_filt[0], HandY_filt[0], color="yellow", label="Start Position")
    plt.scatter(HandX_filt[-1], HandY_filt[-1], color="pink", label="End Position")
    plt.scatter(target_x, target_y, color="orange", label="Target Position", zorder=5)
    plt.scatter(
        target_x_at_ct, target_y, color="darkolivegreen", label="Target Position at CT"
    )
    plt.scatter(
        HandX_filt[completion_time],
        HandY_filt[completion_time],
        color="brown",
        label="Hand Coordinates at Completion Time",
    )
    plt.scatter(
        HandX_filt[reaction_time_50],
        HandY_filt[reaction_time_50],
        color="olive",
        label="Point of HandPath at RT50",
    )
    plt.scatter(
        CursorX[completion_time],
        CursorY[completion_time],
        color="chocolate",
        label="Cursor at CT",
    )

    # Participant movement vector: from start to position at RT+50ms
    if reaction_time_50 < len(HandX_filt):
        participant_vector = np.array(
            [
                HandX_filt[reaction_time_50] - HandX_filt[0],
                HandY_filt[reaction_time_50] - HandY_filt[0],
            ]
        )
    else:
        participant_vector = np.array([np.nan, np.nan])

    # Ideal movement vector: from start directly to target at RT+50ms
    ideal_vector = np.array(
        [
            xTargetPos[reaction_time_50] - HandX_filt[0],
            yTargetPos[reaction_time_50] - HandY_filt[0],
        ]
    )

    plt.quiver(
        HandX_filt[0],
        HandY_filt[0],
        ideal_vector[0],
        ideal_vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="orange",
        label="Ideal Vector",
        headwidth=2,
        headlength=3,
    )

    plt.quiver(
        HandX_filt[0],
        HandY_filt[0],
        participant_vector[0],
        participant_vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        label="Participant Vector",
        headwidth=2,
        headlength=3,
    )

    # Annotations
    annotation_x = plt.xlim()[0] + 10
    annotation_y = plt.ylim()[1] - 10
    line_spacing = 20

    plt.text(
        annotation_x,
        annotation_y,
        f"IDE = {theta_deg:.2f}°\nCondition: {condition}",
        fontsize=10,
        color="black",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.text(
        annotation_x,
        annotation_y - line_spacing,
        f"End Point Error: {end_point_error:.2f} mm\nPath Length Ratio: {path_length_ratio:.2f}",
        fontsize=8,
        color="black",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.text(
        annotation_x,
        annotation_y - 2 * line_spacing,
        f'Accuracy: {"Miss" if accuracy_value == 0 else "Hit"}',
        fontsize=8,
        color="black",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.text(
        annotation_x,
        annotation_y - 3 * line_spacing,
        f"Cursor X: {CursorX[completion_time]}\nCursor Y: {CursorY[completion_time]}",
        fontsize=8,
        color="black",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.text(
        annotation_x,
        annotation_y - 4 * line_spacing,
        f"HandX: {HandX_filt[completion_time]}\nHandY: {HandY_filt[completion_time]}",
        fontsize=8,
        color="black",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.text(
        annotation_x,
        annotation_y - 5 * line_spacing,
        f"Target X: {target_x_at_ct}\nTarget Y: {target_y}",
        fontsize=8,
        color="blue",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.title("Participant Movement and Ideal Movement Vectors")
    plt.legend(loc="upper right", fontsize=8, framealpha=0.9, bbox_to_anchor=(1.15, 1))
    plt.axis("equal")
    plt.grid(True)

    plot_filename = f"{subject}_{i}_{str(uuid.uuid4())}.png"
    plt.savefig(os.path.join(subject_folder, plot_filename), bbox_inches="tight")
    plt.close()


def plot_singletraj(plotsubject, plotday, trajx, allTrajs, all_df):
    """
    Plot a minimal single trajectory — cursor path with RT markers.

    Shows cursor path up to feedback time, with:
    - Blue dot at RT (reaction time)
    - Green dot at RTalt (alternative reaction time)
    - Red circle at target position

    Parameters
    ----------
    plotsubject : str
        Subject ID (e.g., '001').
    plotday : str
        Visit day (e.g., 'Day1').
    trajx : int
        Trial index.
    allTrajs : dict
        Trajectory data keyed by subject+day.
    all_df : DataFrame
        All trials DataFrame.
    """
    subject_df = all_df.loc[
        (all_df["subject"] == plotsubject) & (all_df["day"] == plotday)
    ]
    traj = allTrajs[plotsubject + plotday][trajx]
    trajinfo = subject_df.iloc[trajx]

    fig, ax = plt.subplots()
    ft = int(trajinfo["FeedbackTime"])
    plt.plot(traj["CursorX"][0:ft], traj["CursorY"][0:ft])
    plt.plot(
        traj["CursorX"][int(trajinfo["RT"])], traj["CursorY"][int(trajinfo["RT"])], "bo"
    )
    plt.plot(
        traj["CursorX"][int(trajinfo["RTalt"])],
        traj["CursorY"][int(trajinfo["RTalt"])],
        "go",
    )
    circle1 = plt.Circle(
        (traj["xTargetPos"][ft], traj["yTargetPos"][ft]), 10, color="r"
    )
    ax.add_patch(circle1)
    ax.axis("equal")
    ax.set(xlim=(-150, 150), ylim=(40, 200))


# =============================================================================
# 2. TRAJECTORY OVERVIEW
# =============================================================================


def plot_trajectories_range(plotsubject, plotday, tstart, tend, all_df, allTrajs):
    """
    Plot a range of consecutive trial trajectories for one subject/day.

    Useful for visually inspecting how movement patterns change across
    a block of trials. Each trial's cursor path is drawn up to completion
    time, with a red circle at the target position.

    Saves as: {subject}_ExampleTraj_{condition}_{duration}_{arm}.pdf

    Parameters
    ----------
    plotsubject : str
        Subject ID.
    plotday : str
        Visit day (e.g., 'Day1').
    tstart, tend : int
        Start and end trial indices (inclusive range).
    all_df : DataFrame
        All trials DataFrame.
    allTrajs : dict
        Trajectory data keyed by subject+day.
    """
    palette = sns.color_palette(["#7fc97f", "#998ec3"])
    fig, ax = plt.subplots()

    for trajx in range(tstart, tend):
        subject_df = all_df[
            (all_df["subject"] == plotsubject) & (all_df["day"] == plotday)
        ]
        traj = allTrajs[plotsubject + plotday][trajx]
        trajinfo = subject_df.iloc[trajx]

        if np.isnan(trajinfo.RT):
            logger.warning("Missing RT for trajectory %d", trajx)
            continue

        ft = int(trajinfo["CT"])
        style = "--" if tstart <= trajx <= tend else "-"
        ax.plot(traj["CursorX"][0:ft], traj["CursorY"][0:ft], style, color=palette[0])
        circle1 = plt.Circle(
            (traj["xTargetPos"][ft], traj["yTargetPos"][ft]), 10, color="r"
        )
        ax.add_patch(circle1)
        ax.axis("equal")
        ax.set(xlim=(-150, 150), ylim=(40, 150))

    plt.savefig(
        f"{plotsubject}_ExampleTraj_{trajinfo.Condition}_{trajinfo.Duration}_{trajinfo.Affected}.pdf",
        dpi=100,
        bbox_inches="tight",
    )


def plot_filtered_hand_trajectories(
    plotsubject, plotday, all_df, allTrajs, numPoints=75, max_cols=4
):
    """
    Plot arc-length normalized hand trajectories for one subject/day.

    This function:
    1. Filters for 625ms Interception trials only
    2. For each trial, resamples the hand path using cubic spline interpolation
       to normalize by arc-length (so all paths have the same number of points)
    3. Separates trajectories into left-side and right-side based on mean X position
    4. Plots individual trial subplots (hand path, cursor path, normalized path)
    5. Computes and plots average trajectories per side

    This is useful for comparing movement symmetry and consistency between
    the affected and less-affected arms.

    Parameters
    ----------
    plotsubject : str
        Subject ID.
    plotday : str
        Visit day.
    all_df : DataFrame
        All trials DataFrame.
    allTrajs : dict
        Trajectory data keyed by subject+day.
    numPoints : int
        Number of resampled points per trajectory (default: 75).
    max_cols : int
        Max columns in the subplot grid (default: 4).
    """
    subject_df = all_df[
        (all_df["subject"] == plotsubject)
        & (all_df["day"] == plotday)
        & (all_df["Duration"] == 625)
        & (all_df["Condition"] == "Interception")
    ]
    trials_to_plot = list(subject_df.index)

    if not trials_to_plot:
        logger.warning(f"No trials to plot for subject {plotsubject} on {plotday}")
        return

    logger.info(
        f"Plotting {len(trials_to_plot)} trial(s) for subject {plotsubject} on {plotday}"
    )

    num_trials = len(trials_to_plot)
    num_cols = min(num_trials, max_cols)
    num_rows = (num_trials + num_cols - 1) // num_cols

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5), sharey=True)
    ax = ax.flatten()

    right_side_resampled_x, right_side_resampled_y = [], []
    left_side_resampled_x, left_side_resampled_y = [], []

    for idx, trajx in enumerate(trials_to_plot):
        try:
            traj = allTrajs[plotsubject + plotday][trajx]
        except KeyError as e:
            logger.error(
                f"Trajectory data for {plotsubject} on {plotday} trial {trajx} not found: {e}"
            )
            continue

        logger.info(f"Plotting trajectory for trial {trajx}")

        # Plot original hand and cursor paths
        ax[idx].plot(
            traj["HandX_filt"], traj["HandY_filt"], label="Hand Path", color="orange"
        )
        ax[idx].plot(
            traj["CursorX"], traj["CursorY"], label="Cursor Path", color="blue"
        )
        ax[idx].plot(
            traj["CursorX"][499],
            traj["CursorY"][499],
            "bo",
            label="Cursor Position at 499",
        )

        # Arc-length normalization: fit a cubic spline, then resample at equal arc-length intervals
        tck, u = splprep([traj["HandX_filt"], traj["HandY_filt"]], s=0, k=3)
        alpha = np.linspace(0, 1, numPoints)
        resampled_x, resampled_y = splev(alpha, tck)

        # Separate left vs right based on mean X position
        mean_resampled_x = np.mean(resampled_x)
        if mean_resampled_x >= 0:
            right_side_resampled_x.append(resampled_x)
            right_side_resampled_y.append(resampled_y)
        else:
            left_side_resampled_x.append(resampled_x)
            left_side_resampled_y.append(resampled_y)

        ax[idx].plot(
            resampled_x,
            resampled_y,
            label="Arc-Length Normalized Path",
            linestyle="--",
            color="red",
        )
        ax[idx].axis("equal")
        ax[idx].set_xlabel("X Position")
        ax[idx].set_ylabel("Y Position")
        ax[idx].set_title(f"Trial {trajx} for Subject {plotsubject} on {plotday}")

    # Hide empty subplots
    for idx in range(num_trials, num_rows * num_cols):
        fig.delaxes(ax[idx])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.05),
        fontsize="large",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Plot average trajectories for right and left side groups
    plt.figure(figsize=(7, 7))
    for resampled_x, resampled_y in zip(right_side_resampled_x, right_side_resampled_y):
        plt.plot(resampled_x, resampled_y, color="red", alpha=0.3)
    for resampled_x, resampled_y in zip(left_side_resampled_x, left_side_resampled_y):
        plt.plot(resampled_x, resampled_y, color="blue", alpha=0.3)

    if right_side_resampled_x:
        avg_right_x = np.mean(right_side_resampled_x, axis=0)
        avg_right_y = np.mean(right_side_resampled_y, axis=0)
        plt.plot(
            avg_right_x,
            avg_right_y,
            "r-",
            label="Average Right Side Trajectory",
            linewidth=3,
        )

    if left_side_resampled_x:
        avg_left_x = np.mean(left_side_resampled_x, axis=0)
        avg_left_y = np.mean(left_side_resampled_y, axis=0)
        plt.plot(
            avg_left_x,
            avg_left_y,
            "b-",
            label="Average Left Side Trajectory",
            linewidth=3,
        )

    plt.axis("equal")
    plt.legend()
    plt.title(f"Average Trajectories for Subject {plotsubject} on {plotday}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()

    logger.info("Finished plotting all trials and the average trajectories")


# =============================================================================
# 3. GROUP SUMMARIES (Seaborn faceted plots)
# =============================================================================


def plot_groupmeans(varlist, df):
    """
    Plot CP vs TDC group means for each kinematic variable.

    Creates a FacetGrid with columns for Reaching and Interception conditions.
    Each panel shows pointplot (means+CI) and stripplot (individual subjects)
    colored by arm (Less Affected vs More Affected).

    Parameters
    ----------
    varlist : list of str
        Kinematic variables to plot (e.g., ['RT', 'MT', 'velPeak']).
    df : DataFrame
        Subject-level means DataFrame.
    """
    for vartotest in varlist:
        g = sns.FacetGrid(
            df, col="Condition", legend_out=True, col_order=["Reaching", "Interception"]
        )
        g = g.map(
            sns.pointplot,
            "group",
            vartotest,
            "Affected",
            order=["CP", "TDC"],
            hue_order=["Less Affected", "More Affected"],
            palette=sns.color_palette("muted"),
        )
        g = g.map(
            sns.stripplot,
            "group",
            vartotest,
            "Affected",
            order=["CP", "TDC"],
            hue_order=["Less Affected", "More Affected"],
            palette=sns.color_palette("muted"),
        )


def plot_byduration(varlist, df):
    """
    Plot kinematic variables by target duration, pooling CP and TDC groups.

    Creates a FacetGrid with Reaching/Interception columns, showing how
    each variable changes across the four target durations (500-900ms).

    Parameters
    ----------
    varlist : list of str
        Kinematic variables to plot.
    df : DataFrame
        Subject-level means DataFrame (with Duration column).
    """
    for vartotest in varlist:
        g = sns.FacetGrid(df, col="Condition", legend_out=True)
        g = g.map(
            sns.pointplot,
            "Duration",
            vartotest,
            "Affected",
            order=[500, 625, 750, 900],
            hue_order=["Less Affected", "More Affected"],
            palette=sns.color_palette("muted"),
        )
        g.add_legend()


def plot_twocolumnbar(vartotest, col, col_order, x, hue, df, order, hue_order):
    """
    Generic two-column bar + strip plot for any kinematic variable.

    Saves output as: {vartotest}{col}{x}.jpg

    Parameters
    ----------
    vartotest : str
        Variable name for the y-axis.
    col : str
        Column to facet by (e.g., 'Condition').
    col_order : list
        Order of facet columns.
    x : str
        Variable for the x-axis (e.g., 'group').
    hue : str
        Variable for color coding (e.g., 'Affected').
    df : DataFrame
        Data to plot.
    order : list
        Order of x-axis categories.
    hue_order : list
        Order of hue categories.
    """
    g = sns.FacetGrid(df, col=col, col_order=col_order, legend_out=True)
    g = g.map(
        sns.barplot,
        x,
        vartotest,
        hue,
        order=order,
        hue_order=hue_order,
        palette=sns.color_palette("muted"),
        alpha=0.6,
    )
    g = g.map(
        sns.stripplot,
        x,
        vartotest,
        hue,
        order=order,
        hue_order=hue_order,
        palette=sns.color_palette("muted"),
        split=True,
    )
    g.savefig(vartotest + col + x + ".jpg", format="jpeg", dpi=300)
