"""
Signal Processing Module for CHEAT-CP Kinematic Analysis
=========================================================

PURPOSE:
    This module handles all the signal processing steps that transform raw
    KINARM sensor data into clean, usable signals for kinematic analysis.

WHY DO WE NEED THIS?
    The KINARM robot records hand position and velocity at 1000 Hz (1000 times
    per second). The raw signals contain high-frequency noise from the sensors
    and minor hand tremors. Before we can accurately measure things like
    reaction time or movement direction, we need to smooth out this noise
    while preserving the real movement information.

WHAT THIS MODULE DOES:
    1. Low-pass filtering (Butterworth filter at 5 Hz)
    2. Computing hand speed from filtered velocities
    3. Computing cursor positions (feedforward model)
    4. Distance calculations (Euclidean distance between points)

KEY CONCEPTS:
    - Butterworth Filter: A type of filter that smoothly removes frequencies
      above a cutoff. We use 5 Hz because real hand movements happen below
      5-10 Hz. Higher frequencies are noise.

    - Zero-Phase Filtering (filtfilt): We apply the filter twice — once forward
      and once backward. This ensures no time delay is introduced. Critical
      because we need accurate timing for RT/CT measurements.

    - Feedforward Cursor Model: The KINARM shows a cursor that's slightly
      ahead of the actual hand position, based on current velocity:
        CursorX = HandX + 0.06 * velX
      This makes the cursor feel responsive to the participant.

ORIGINAL FILES: Utils.py (most functions), CHEATCP_Final_With_IE.py (cursor)
"""

import numpy as np
from scipy import signal

# =============================================================================
# LOW-PASS FILTERING
# =============================================================================


def low_pass_filter(data, fc, fs):
    """
    Apply a zero-phase Butterworth low-pass filter to a 1D signal.

    This removes high-frequency noise while keeping the meaningful movement
    information intact. The zero-phase approach (filtfilt) ensures NO time
    delay is introduced — critical for accurate timing measurements.

    How it works:
    1. Design a 2nd-order Butterworth filter with cutoff at fc Hz
    2. Apply the filter forward through the data
    3. Apply the filter backward through the result
    4. This double-pass cancels out any phase shift

    Parameters
    ----------
    data : numpy.ndarray
        The raw 1D signal to filter. Can be hand position (X or Y)
        or velocity (velX or velY). Each element is one sample at 1 ms.

    fc : float
        Cutoff frequency in Hz. Frequencies ABOVE this are removed.
        Default in our analysis: 5 Hz.
        - Hand movements are typically 1-5 Hz
        - Sensor noise and tremors are typically > 10 Hz
        - So 5 Hz keeps the real movement and removes noise

    fs : float
        Sampling frequency in Hz. How many samples per second.
        Default in our analysis: 1000 Hz (= 1 sample per millisecond)

    Returns
    -------
    numpy.ndarray
        The filtered signal, same length as input. Temporally aligned
        (no phase shift) with the original signal.

    Example
    -------
    >>> HandX_raw = np.array([0.1, 0.12, 0.15, ...])  # 1000 samples
    >>> HandX_filtered = low_pass_filter(HandX_raw, fc=5, fs=1000)
    >>> # HandX_filtered is smoother but same length

    Notes
    -----
    - Uses a 2nd-order Butterworth design (N=2)
    - The Nyquist frequency is fs/2 = 500 Hz
    - Wn = fc / (fs/2) normalizes the cutoff to [0, 1] range
    - filtfilt effectively doubles the filter order (becomes 4th-order)
    """
    # Design the Butterworth filter coefficients
    # N=2: 2nd-order filter (smooth rolloff, no ringing)
    # Wn: normalized cutoff frequency (0 to 1, where 1 = Nyquist = fs/2)
    # btype='low': low-pass (keep frequencies below fc, remove above)
    w = fc / (fs / 2)  # Normalize the frequency
    # divide filter order by 2
    [b, a] = signal.butter(2, w, "low")
    # Apply zero-phase filtering (forward + backward pass)
    # This is mathematically equivalent to: filter(reverse(filter(data)))
    dataLPfiltfilt = signal.filtfilt(b, a, data)  # apply filtfilt to data
    return dataLPfiltfilt


def filter_hand_data(HandX, HandY, velX, velY, defaults):
    """
    Apply low-pass filtering to ALL four hand signals at once.

    This is a convenience function that filters the X/Y positions and
    X/Y velocities in one call. All four signals need the same filtering
    treatment for consistency.

    Parameters
    ----------
    HandX : numpy.ndarray
        Raw hand X-position over time (in mm). One value per millisecond.

    HandY : numpy.ndarray
        Raw hand Y-position over time (in mm).

    velX : numpy.ndarray
        Raw hand X-velocity over time (in mm/s).

    velY : numpy.ndarray
        Raw hand Y-velocity over time (in mm/s).

    defaults : dict
        Configuration defaults containing:
        - 'fc': Cutoff frequency (5 Hz)
        - 'fs': Sampling frequency (1000 Hz)

    Returns
    -------
    tuple of 4 numpy.ndarray
        (HandX_filt, HandY_filt, velX_filt, velY_filt)
        All filtered versions of the input signals.

    Example
    -------
    >>> HandX_f, HandY_f, velX_f, velY_f = filter_hand_data(
    ...     HandX, HandY, velX, velY, config.defaults
    ... )
    """
    fc = defaults["fc"]  # Cutoff frequency (5 Hz)
    fs = defaults["fs"]  # Sampling frequency (1000 Hz)

    HandX_filt = low_pass_filter(HandX, fc, fs)
    HandY_filt = low_pass_filter(HandY, fc, fs)
    velX_filt = low_pass_filter(velX, fc, fs)
    velY_filt = low_pass_filter(velY, fc, fs)

    return HandX_filt, HandY_filt, velX_filt, velY_filt


# =============================================================================
# DERIVED SIGNALS
# =============================================================================


def compute_hand_speed(velX_filt, velY_filt):
    """
    Compute the resultant hand speed from filtered X and Y velocities.

    Hand speed is the magnitude of the 2D velocity vector at each time point:
        speed(t) = sqrt(velX(t)² + velY(t)²)

    This gives us a single "how fast the hand is moving" value at each
    millisecond, regardless of direction. We use this to:
    - Find velocity peaks (for determining reaction time)
    - Detect movement onset (when speed exceeds a threshold)
    - Measure peak velocity (velPeak)

    Parameters
    ----------
    velX_filt : numpy.ndarray
        Filtered hand X-velocity over time (mm/s).

    velY_filt : numpy.ndarray
        Filtered hand Y-velocity over time (mm/s).

    Returns
    -------
    numpy.ndarray
        Hand speed at each time point (mm/s). Always positive.
        Same length as input arrays.

    Example
    -------
    >>> speed = compute_hand_speed(velX_filt, velY_filt)
    >>> peak_speed = np.max(speed)  # Maximum speed during the trial
    """
    handspeed = np.sqrt(velX_filt**2 + velY_filt**2)
    return handspeed


def compute_cursor_positions(HandX, HandY, velX, velY, defaults):
    """
    Compute the cursor position using the KINARM feedforward model.

    The KINARM doesn't show the cursor exactly at the hand position.
    Instead, it shows it slightly AHEAD of the hand, based on the
    current velocity. This makes the cursor feel more responsive
    to the participant.

    The formula is:
        CursorX = HandX + fdfwd * velX
        CursorY = HandY + fdfwd * velY

    Where fdfwd = 0.06 is the KINARM's built-in constant.

    WHY this matters for analysis:
        When we calculate Completion Time (CT), we check when the CURSOR
        (not the hand) crosses the target's Y-position. So we need the
        cursor position, not just the hand position.

    IMPORTANT:
        This uses the RAW (unfiltered) positions and velocities, because
        the KINARM itself uses raw data for the feedforward computation.

    Parameters
    ----------
    HandX : numpy.ndarray
        Raw hand X-position (mm). NOT filtered.

    HandY : numpy.ndarray
        Raw hand Y-position (mm). NOT filtered.

    velX : numpy.ndarray
        Raw hand X-velocity (mm/s). NOT filtered.

    velY : numpy.ndarray
        Raw hand Y-velocity (mm/s). NOT filtered.

    defaults : dict
        Configuration defaults containing:
        - 'fdfwd': Feedforward scaling factor (0.06)

    Returns
    -------
    tuple of 2 numpy.ndarray
        (CursorX, CursorY) — Cursor positions at each time point.

    Example
    -------
    >>> CursorX, CursorY = compute_cursor_positions(
    ...     HandX, HandY, velX, velY, config.defaults
    ... )
    """
    fdfwd = defaults["fdfwd"]  # 0.06 — KINARM's feedforward constant

    CursorX = HandX + fdfwd * velX
    CursorY = HandY + fdfwd * velY

    return CursorX, CursorY


# =============================================================================
# DISTANCE CALCULATIONS
# =============================================================================


def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two 2D points (or arrays of points).

    This is the standard "straight line" distance formula:
        d = sqrt((x2 - x1)² + (y2 - y1)²)

    Used throughout the analysis for:
    - End-Point Error (cursor to target distance at CT)
    - Minimum distance (closest approach of cursor to target)
    - Target distance (how far the target moved)
    - Hand distance (how far the hand is from the target)

    Parameters
    ----------
    x1, y1 : float or numpy.ndarray
        X and Y coordinates of the first point(s).

    x2, y2 : float or numpy.ndarray
        X and Y coordinates of the second point(s).

    Returns
    -------
    float or numpy.ndarray
        Distance between the points. If inputs are arrays,
        returns element-wise distances.

    Example
    -------
    >>> # Distance between cursor and target at CT
    >>> epe = euclidean_distance(CursorX[CT], CursorY[CT],
    ...                          xTarget[CT], yTarget[CT])
    >>> print(f"End-Point Error: {epe:.2f} mm")
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# =============================================================================
# DATA CLEANING UTILITIES
# =============================================================================


def clean_data_iqr(x, y, indices):
    """
    Remove outliers from paired (x, y) data using the IQR method.

    The Interquartile Range (IQR) method identifies outliers as values
    that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR, where:
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1

    This is a robust statistical method — it's not affected by extreme
    outliers (unlike mean ± 3*std). We apply it to BOTH x and y values,
    and only keep data points that are within bounds for BOTH dimensions.

    WHERE this is used:
        In regression.py, before fitting the RANSAC regression model.
        We clean the x_intersect and x_target_at_RT data to remove
        extreme values that would distort the regression.

    Parameters
    ----------
    x : numpy.ndarray
        First variable values (e.g., x_intersect positions).

    y : numpy.ndarray
        Second variable values (e.g., x_target_at_RT positions).

    indices : numpy.ndarray
        Original indices of the data points (so we can track which
        trials survived the cleaning).

    Returns
    -------
    tuple of 3 numpy.ndarray
        (x_clean, y_clean, indices_clean) — Cleaned arrays with
        outliers removed.

    Example
    -------
    >>> x_clean, y_clean, idx_clean = clean_data_iqr(
    ...     x_intersects, x_targets, trial_indices
    ... )
    >>> print(f"Removed {len(x) - len(x_clean)} outlier trials")
    """
    # --- Clean X dimension ---
    # Calculate Q1, Q3, and IQR for x (ignoring NaN values)
    q1_x, q3_x = np.percentile(x[~np.isnan(x)], [25, 75])
    iqr_x = q3_x - q1_x
    lower_bound_x = q1_x - 1.5 * iqr_x
    upper_bound_x = q3_x + 1.5 * iqr_x

    # --- Clean Y dimension ---
    # Calculate Q1, Q3, and IQR for y (ignoring NaN values)
    q1_y, q3_y = np.percentile(y[~np.isnan(y)], [25, 75])
    iqr_y = q3_y - q1_y
    lower_bound_y = q1_y - 1.5 * iqr_y
    upper_bound_y = q3_y + 1.5 * iqr_y

    # --- Create combined mask ---
    # A point is an inlier only if BOTH x and y are within their bounds
    mask = (
        (x >= lower_bound_x)
        & (x <= upper_bound_x)
        & (y >= lower_bound_y)
        & (y <= upper_bound_y)
    )

    filtered_indices = indices[~mask]
    print(f"Filtered out indices due to being outliers: {filtered_indices}")
    return x[mask], y[mask], indices[mask], filtered_indices


# =============================================================================
# VISUALIZATION PREFERENCES
# =============================================================================


def set_seaborn_preference():
    """
    Configure Seaborn and Matplotlib settings for publication-quality plots.

    Sets consistent font styles, plot dimensions, and line properties
    so all plots in the analysis look clean and professional.

    This function should be called ONCE at the beginning of the pipeline
    (not at import time). It modifies global matplotlib/seaborn state.

    Settings applied:
    - White background with ticks (clean look)
    - Font: Times New Roman, size 16
    - Figure size: 8 x 5 inches
    - Thin lines (linewidth 0.75)
    - Dark edge colors for bars/patches
    """
    import seaborn as sns
    import matplotlib

    sns.set_style("ticks")
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman"]
    matplotlib.rcParams["font.size"] = 16
    matplotlib.rcParams["figure.figsize"] = (8, 5)
    matplotlib.rcParams["lines.linewidth"] = 0.75
    matplotlib.rcParams["patch.edgecolor"] = "0.15"
