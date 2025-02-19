import numpy as np
import pandas as pd
import math
import os
import matplotlib
import seaborn as sns
from scipy import signal

def clean_data_iqr(x, y, indices):
    # Calculate IQR for x
    q1_x, q3_x = np.percentile(x[~np.isnan(x)], [25, 75])
    iqr_x = q3_x - q1_x
    lower_bound_x = q1_x - 1.5 * iqr_x
    upper_bound_x = q3_x + 1.5 * iqr_x

    # Calculate IQR for y
    q1_y, q3_y = np.percentile(y[~np.isnan(y)], [25, 75])
    iqr_y = q3_y - q1_y
    lower_bound_y = q1_y - 1.5 * iqr_y
    upper_bound_y = q3_y + 1.5 * iqr_y

    mask_x = (x >= lower_bound_x) & (x <= upper_bound_x)
    mask_y = (y >= lower_bound_y) & (y <= upper_bound_y)

    # Mask to filter values within the IQR range
    combined_mask = mask_x & mask_y
    filtered_indices = indices[~combined_mask]
    print(f"Filtered out indices due to being outliers: {filtered_indices}")
    return x[combined_mask], y[combined_mask], indices[combined_mask], filtered_indices


def initialize_arms_data_structure_for_regression():
    return {'Less Affected': {'x_intersect': [], 'x_target': [], 'indices': []},
            'More Affected': {'x_intersect': [], 'x_target': [], 'indices': []}}

def set_seaborn_preference():
    """
    Set the seaborn and matplotlib preferences for plotting.
    """
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 10
    rcstyle = {'axes.linewidth': 1.0,
               'axes.edgecolor': 'black', 'ytick.minor.size': 5.0}
    sns.set(font_scale=1.0, rc={'figure.figsize': (20, 10)})
    sns.set_style('ticks', rcstyle)
    sns.set_context("paper", rc={"lines.linewidth": 1,
                    "xtick.labelsize": 10, "ytick.labelsize": 10})
    
def lowPassFilter(data, fc, fs, filter_order=4):
    """ fc is cut-off frequency of filter
    fs is sampling rate
    """
    w = fc/(fs/2)  # Normalize the frequency
    # divide filter order by 2
    [b, a] = signal.butter(filter_order/2, w, 'low')
    dataLPfiltfilt = signal.filtfilt(b, a, data)  # apply filtfilt to data
    return dataLPfiltfilt

def dist(x1, y1, x2, y2):
    distance = np.sqrt(np.square(x1-x2)+np.square(y1-y2))
    return distance

def filter_hand_data(HandX, HandY, velX, velY, defaults):
    HandX_filt = lowPassFilter(HandX, defaults['fc'], defaults['fs'])
    HandY_filt = lowPassFilter(HandY, defaults['fc'], defaults['fs'])
    velX_filt = lowPassFilter(velX, defaults['fc'], defaults['fs'])
    velY_filt = lowPassFilter(velY, defaults['fc'], defaults['fs'])
    return HandX_filt, HandY_filt, velX_filt, velY_filt

def calculate_handspeed(velX_filt, velY_filt):
    return np.sqrt(velX_filt**2 + velY_filt**2)

def fix_data_types(subject_trials):
    if isinstance(subject_trials, list):
        subject_trials = pd.DataFrame(subject_trials)
    elif not isinstance(subject_trials, pd.DataFrame):
        raise TypeError("subject_trials must be a DataFrame or a list of dictionaries.")
    return subject_trials


def reset_kinematic_data(kinData):
    """
    Reset the kinematic data to NaN if the movement is invalid.
    """
    kinData.update({
        'xPosError': np.nan,
        'targetDist': np.nan,
        'handDist': np.nan,
        'straightlength': np.nan,
        'pathlength': np.nan,
        'targetlength': np.nan,
        'CursorX': np.nan,
        'CursorY': np.nan,
        'maxpathoffset': np.nan,
        'meanpathoffset': np.nan,
        'xTargetEnd': np.nan,
        'yTargetEnd': np.nan,
        'EndPointError': np.nan,
        'IDE': np.nan,
        'ABS_IDE':np.nan,
        'PLR': np.nan,
        'PLR_2': np.nan,
        'isCurveAround': np.nan,
        'idealPathlength': np.nan,
        'x_intersect' : np.nan,
        'x_target_at_RT' : np.nan,
        'Delta_T_Used' : np.nan,
        'targetDist_Hit_Interception': np.nan,
    })
    return kinData


def compute_pvar_for_group(trial_list, num_points=150):
    """
    Given a list of trial dicts (all from the same group), time-normalize
    each trial and compute path variability (Pvar).

    Returns a tuple: (Pvar, mean_traj_x, mean_traj_y, n_valid)
     - Pvar: float or None (if fewer than 2 valid trials)
     - mean_traj_x: array of shape (num_points,) or None
     - mean_traj_y: same shape as mean_traj_x
     - n_valid: how many trials were successfully time-normalized
    """
    all_norm_x = []
    all_norm_y = []

    for trial in trial_list:
        normX, normY = time_normalize_trial(trial, num_points=num_points)
        # Only keep if not None
        if normX is not None and normY is not None:
            all_norm_x.append(normX)
            all_norm_y.append(normY)

    if len(all_norm_x) < 2:
        # Not enough data to compute a standard deviation across trials
        return None, None, None, len(all_norm_x)

    # Convert to arrays: shape (nTrials, num_points)
    norm_x = np.array(all_norm_x)
    norm_y = np.array(all_norm_y)

    # Compute mean trajectory
    mean_traj_x = np.mean(norm_x, axis=0)
    mean_traj_y = np.mean(norm_y, axis=0)

    # Deviations
    dev_x = norm_x - mean_traj_x
    dev_y = norm_y - mean_traj_y

    # Standard deviations at each time point
    std_x = np.std(dev_x, axis=0)
    std_y = np.std(dev_y, axis=0)

    # Sum them up to get Pvar
    # You can also do radial: sum of sqrt(std_x^2 + std_y^2) if you prefer
    Pvar = np.sum(std_x + std_y)

    return Pvar, mean_traj_x, mean_traj_y, len(all_norm_x)
