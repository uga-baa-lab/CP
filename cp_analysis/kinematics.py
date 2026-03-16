"""
Kinematics Module for CHEAT-CP Kinematic Analysis
===================================================

PURPOSE:
    This is the CORE module of the analysis pipeline. It computes ALL
    kinematic measures for each individual trial of the KINARM experiment.

WHAT IS A "TRIAL"?
    In the experiment, the participant sits in front of the KINARM robot and
    performs a series of movements. Each movement is one "trial":
    - A target appears on screen
    - The participant moves their hand to reach (or intercept) the target
    - The KINARM records hand position and velocity at 1000 Hz

    Each trial produces a data matrix where: Need to recheck this part
    - Rows = time points (1 per millisecond)
    - Columns = different variables:
        Column 1-2: Target X, Y positions
        Column 4-5: Hand X, Y positions
        Column 6-7: Hand X, Y velocities
        Column 11: Trial duration code
        Column 12: Condition (1=Reaching, 2=Interception)
        Column 14: Accuracy (0=Miss, 1=Hit)
        Column 15: Feedback onset time
        Column 16: Arm type (1=More Affected, 0=Less Affected)

WHAT THIS MODULE COMPUTES (per trial):
    Timing:
    - RT (Reaction Time): When the hand starts moving
    - CT (Completion Time): When the cursor crosses the target Y-position
    - MT (Movement Time): CT - RT

    Velocity:
    - velPeak: Peak hand speed (first real peak)
    - initVel: Initial hand velocity

    Angles:
    - IA_RT, IA_50RT: Initial angles at RT and RT+50ms. - Looks like I implemented there is a mistake in this @TODO check with Dr.Barany
    - IA_RTalt, IA_50RTalt: Angles using alternative RT

    Errors:
    - IDE: Initial Direction Error (angle between actual and ideal path)
    - EndPointError: Distance between cursor and target at CT
    - xPosError: Absolute X-position error at CT

    Path Measures:
    - pathlength, idealPathLength, PLR (Path Length Ratio)
    - maxpathoffset, meanpathoffset (perpendicular deviations)

    Distances:
    - targetDist, handDist, minDist

    Movement Quality:
    - x_intersect: Where initial movement projects onto target Y-line
    - isCurveAround: Whether hand curved back after CT

FLOW:
    The main function is `compute_trial_kinematics()` which calls all the
    sub-functions in order. You pass in the raw trial data and get back a
    dictionary with ALL computed measures.
"""

import numpy as np
import logging
from scipy import signal as scipy_signal

import signal_processing as sp

# Set up logging for this module
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: RAW DATA EXTRACTION
# =============================================================================


def extract_raw_positions_and_velocities(trial_data):
    """
    Extract the 4 raw hand signals from the trial data matrix.

    The KINARM stores data in a matrix where each column is a different
    variable. The hand position and velocity are in columns 4-7
    (0-indexed).

    Parameters
    ----------
    trial_data : numpy.ndarray
        Raw trial data matrix from the .mat file.
        Shape: (n_timepoints, n_columns)
        Column 4 = Hand X position (mm)
        Column 5 = Hand Y position (mm)
        Column 6 = Hand X velocity (mm/s)
        Column 7 = Hand Y velocity (mm/s)

    """
    # .T transposes the matrix so we can index by column number
    HandX = trial_data.T[4]  # Hand X position (mm)
    HandY = trial_data.T[5]  # Hand Y position (mm)
    velX = trial_data.T[6]  # Hand X velocity (mm/s)
    velY = trial_data.T[7]  # Hand Y velocity (mm/s)

    return HandX, HandY, velX, velY


def extract_trial_metadata(trial_data):
    """
    Extract trial-level metadata from the data matrix.

    Each trial has metadata stored in specific columns of the data matrix.
    Some values are constant across all time points (like condition), so
    we take the value from the first time point [0].

    Parameters
    ----------
    trial_data : numpy.ndarray
        Raw trial data matrix from the .mat file.

    Returns
    -------
    dict with keys:
        'feedbackOn' : int
            Time point (ms) when visual feedback was provided.
            After this point, the participant can see the result.

        'xTargetPos' : numpy.ndarray
            Target X-position at each time point (mm).
            For Reaching: constant. For Interception: moves horizontally.

        'yTargetPos' : numpy.ndarray
            Target Y-position at each time point (mm).
            Always constant (target moves horizontally only).

        'condition_code' : int
            Raw condition code. 1 = Reaching, 2 = Interception.

        'accuracy' : float
            0 = Miss, 1 = Hit. Whether participant reached the target.

        'arm_type_value' : int
            1 = More Affected arm, 0 = Less Affected arm.

        'duration_code' : int
            Duration code (1-8). Maps to actual duration in ms:
            {1: 625, 2: 500, 3: 750, 4: 900,
             5: 625, 6: 500, 7: 750, 8: 900}
            Codes 1-4 and 5-8 represent the same durations for
            different experimental blocks.
    """
    metadata = {
        "feedbackOn": int(trial_data.T[15][0]),
        "xTargetPos": trial_data.T[1],
        "yTargetPos": trial_data.T[2],
        "condition_code": int(trial_data[0, 12]),
        "accuracy": trial_data.T[14][0],
        "arm_type_value": int(trial_data.T[16][0]),
        "duration_code": int(trial_data.T[11][0]),
    }
    return metadata


# =============================================================================
# SECTION 2: TRAJECTORY EXTRACTION
# =============================================================================


def get_hand_trajectories(trial_data, defaults):
    """
    Extract and filter hand trajectories for a single trial.

    This function does three things:
    1. Extracts raw hand positions and velocities
    2. Applies low-pass filtering to remove noise
    3. Computes derived signals (hand speed and cursor positions)

    The resulting trajectory data is used for plotting (TrialPlots) and
    for storing the full trajectory information per subject.

    Parameters
    ----------
    trial_data : numpy.ndarray
        Raw trial data matrix from the .mat file.

    defaults : dict
        Configuration defaults with 'fc', 'fs', 'fdfwd'.

    Returns
    -------
    dict with keys:
        'xTargetPos' : numpy.ndarray — Target X positions
        'yTargetPos' : numpy.ndarray — Target Y positions
        'HandX_filt' : numpy.ndarray — Filtered hand X positions
        'HandY_filt' : numpy.ndarray — Filtered hand Y positions
        'velX_filt'  : numpy.ndarray — Filtered hand X velocities
        'velY_filt'  : numpy.ndarray — Filtered hand Y velocities
        'handspeed'  : numpy.ndarray — Resultant hand speed
        'CursorX'    : numpy.ndarray — Cursor X positions
        'CursorY'    : numpy.ndarray — Cursor Y positions
    """
    # Step 1: Get raw signals
    HandX, HandY, velX, velY = extract_raw_positions_and_velocities(trial_data)

    # Step 2: Filter all signals
    HandX_filt, HandY_filt, velX_filt, velY_filt = sp.filter_hand_data(
        HandX, HandY, velX, velY, defaults
    )

    # Step 3: Compute derived signals
    handspeed = sp.compute_hand_speed(velX_filt, velY_filt)
    CursorX, CursorY = sp.compute_cursor_positions(HandX, HandY, velX, velY, defaults)

    traj_data = {
        "xTargetPos": trial_data.T[1],
        "yTargetPos": trial_data.T[2],
        "HandX_filt": HandX_filt,
        "HandY_filt": HandY_filt,
        "velX_filt": velX_filt,
        "velY_filt": velY_filt,
        "handspeed": handspeed,
        "CursorX": CursorX,
        "CursorY": CursorY,
    }
    return traj_data


# =============================================================================
# SECTION 3: PEAK DETECTION
# =============================================================================


def find_velocity_peaks(handspeed):
    """
    Detect peaks (local maxima) in the hand speed signal.

    We use scipy's find_peaks to locate moments where the hand is moving
    fastest. The first real peak is used to determine reaction time.

    Peak Detection Criteria:
    - height: Must be at least 25% of the maximum hand speed (max/4).
      This filters out tiny fluctuations that aren't real movements.
    - distance: Peaks must be at least 150 ms apart.
      This prevents detecting the same movement as multiple peaks.

    Parameters
    ----------
    handspeed : numpy.ndarray
        Hand speed at each time point (mm/s).

    Returns
    -------
    peaks : numpy.ndarray
        Indices (time points) where peaks occur.

    properties : dict
        Peak properties including 'peak_heights' array.

    """
    peaks, properties = scipy_signal.find_peaks(
        handspeed,
        height=max(handspeed) / 4,  # At least 25% of max speed
        distance=150,  # At least 150 ms between peaks
    )
    return peaks, properties


def check_first_peak(peaks, properties, kin_data):
    """
    Validate the first detected peak.

    Sometimes the first peak occurs before 100 ms, which is too early
    to be a real voluntary movement (it takes at least ~100 ms to process
    a visual stimulus and initiate a movement). If the first peak is
    before 100 ms AND there are other peaks, we remove it.

    Parameters
    ----------
    peaks : numpy.ndarray
        Peak indices from find_velocity_peaks().

    properties : dict
        Peak properties from find_velocity_peaks().

    kin_data : dict
        Kinematic data dictionary (modified in place).

    Returns
    -------
    tuple
        (peaks, properties, kin_data) — Updated with bad peak removed if needed.
        kin_data['badfirstpeak'] is set to True if removed, False otherwise.
    """
    if len(properties["peak_heights"]) > 1 and peaks[0] < 100:
        # First peak is before 100 ms AND there are other peaks → remove it
        logger.info("Removing bad first peak (before 100 ms)")
        peaks = np.delete(peaks, 0)
        properties["peak_heights"] = np.delete(properties["peak_heights"], 0)
        kin_data["badfirstpeak"] = True
    else:
        kin_data["badfirstpeak"] = False

    return peaks, properties, kin_data


# =============================================================================
# SECTION 4: REACTION TIME (RT)
# =============================================================================


def find_reaction_time(peaks, properties, handspeed, kin_data):
    """
    Find the Reaction Time (RT) — when the hand starts moving.

    ALGORITHM (from the project documentation):
    1. Take the first real peak in the hand speed signal
    2. Calculate 5% of that peak's height (the threshold)
    3. Move BACKWARDS in time from the peak
    4. Find the last time point where speed drops below the 5% threshold
    5. RT = that time point + 1

    WHY 5%?
        At 5% of peak velocity, the hand is essentially still. This gives
        us the moment just before the hand starts accelerating.

    WHY move backwards?
        Moving backwards from the peak guarantees we find the onset of
        the movement that produced that peak, not some earlier noise.

    Parameters
    ----------
    peaks : numpy.ndarray
        Peak indices from find_velocity_peaks().

    properties : dict
        Peak properties, including 'peak_heights'.

    handspeed : numpy.ndarray
        Hand speed at each time point (mm/s).

    kin_data : dict
        Kinematic data dictionary (modified in place).

    Returns
    -------
    dict
        Updated kin_data with:
        - 'RT': Reaction time (ms), or NaN if not found
        - 'velPeak': Peak velocity (mm/s), or NaN
        - 'velLoc': Time of peak velocity (ms), or NaN
        - 'RTexclusion': Reason if RT is invalid
    """
    if len(peaks) > 0:
        # Use the FIRST real peak (not necessarily the highest)
        kin_data["velPeak"] = properties["peak_heights"][0]
        kin_data["velLoc"] = peaks[0]

        # Calculate 5% threshold
        threshold = kin_data["velPeak"] * 0.05

        # Get all time points BEFORE the first peak where speed < threshold
        before_peak = handspeed[0 : peaks[0]]
        below_threshold = before_peak < threshold
        onset_indices = np.where(below_threshold == True)[0]

        if len(onset_indices) > 0:
            # RT = last time point below threshold, + 1
            # (the +1 is because the NEXT time point is where movement starts)
            kin_data["RT"] = onset_indices[-1] + 1
        else:
            # Speed never dropped below threshold before the peak
            kin_data["RT"] = np.nan
            kin_data["velPeak"] = np.nan
            kin_data["velLoc"] = np.nan
            kin_data["RTexclusion"] = "could not find onset"
    else:
        # No peaks found at all — trial has no detectable movement
        logger.debug("No peaks found in handspeed")
        kin_data["RT"] = np.nan
        kin_data["velPeak"] = np.nan
        kin_data["velLoc"] = np.nan
        kin_data["RTexclusion"] = "no peaks"

    return kin_data


def check_rt_contingencies(kin_data, feedbackOn, CursorY, yTargetPos):
    """
    Validate the Reaction Time (RT) based on physiological constraints.

    Even after finding RT, we need to check if it makes sense:

    1. RT < 100 ms → TOO EARLY
       Human visual processing + motor initiation takes at least ~100 ms.
       If RT is earlier, it's likely noise or an anticipatory movement.

    2. RT > feedbackOn → TOO LATE
       If RT is after the feedback time, the movement started after
       the trial was already over. Invalid.

    3. CursorY[0] > yTargetPos[0] → ALREADY PAST TARGET
       If the cursor starts beyond the target's Y-position, the "movement"
       is not a real reaching movement.

    If ANY of these fail, RT is set to NaN (invalid).

    Parameters
    ----------
    kin_data : dict
        Must contain 'RT' from find_reaction_time().

    feedbackOn : int
        Time point when feedback was provided.

    CursorY : numpy.ndarray
        Cursor Y-position over time.

    yTargetPos : numpy.ndarray
        Target Y-position over time.

    """
    if (
        kin_data["RT"] < 100
        or kin_data["RT"] > feedbackOn
        or CursorY[0] > yTargetPos[0]
    ):
        # Failed validation → invalidate RT
        kin_data["RT"] = np.nan
        kin_data["velPeak"] = np.nan
        kin_data["velLoc"] = np.nan
        kin_data["RTexclusion"] = "outlier value"
    else:
        kin_data["RTexclusion"] = "good RT"

    return kin_data


def calculate_alternative_rt(kin_data, handspeed, threshold=100):
    """
    Calculate an alternative Reaction Time (RTalt) using a simple speed threshold.

    Unlike the primary RT (which uses 5% of peak velocity), this method
    simply finds when the hand speed FIRST exceeds a fixed threshold
    (default: 100 mm/s).

    This provides a backup measure in case the primary RT method has
    issues (e.g., noisy baseline causing false onset detection).

    Parameters
    ----------
    kin_data : dict
        Kinematic data dictionary (modified in place).

    handspeed : numpy.ndarray
        Hand speed at each time point (mm/s).

    threshold : float, optional
        Speed threshold in mm/s. Default: 100 mm/s.

    Returns
    -------
    dict
        Updated kin_data with 'RTalt' (ms or NaN).
    """
    above_threshold = handspeed > threshold
    onset = np.where(above_threshold == True)[0]

    if len(onset) > 0:
        kin_data["RTalt"] = onset[0] + 1
    else:
        kin_data["RTalt"] = np.nan

    return kin_data


# =============================================================================
# SECTION 5: COMPLETION TIME (CT)
# =============================================================================


def calculate_completion_time(kin_data, feedbackOn, CursorY, yTargetPos):
    """
    Calculate the Completion Time (CT) — when the cursor crosses the target Y-position.

    CT is the time point when the cursor's Y-position first exceeds the
    target's Y-position. This represents the moment the participant's
    cursor "reaches" the target level.

    ALGORITHM (from documentation):
    1. Check if RT is valid (if not, CT = NaN)
    2. Add a small tolerance to the cursor Y-position (+5 mm)
    3. Subtract a small tolerance from the target Y-position (-10 mm)
    4. Find when: (CursorY + 5) > (yTargetPos[0] - 10)
    5. Apply boundary checks:
       - If crossing happens too late (> feedbackOn + 200 ms) → cap at feedbackOn + 200
       - If cursor never crosses → CT = NaN

    WHY the tolerances (+5, -10)?
        These account for the finite size of the cursor and target on screen.
        The cursor doesn't need to reach the exact center of the target;
        when they visually overlap, that counts as reaching.

    Parameters
    ----------
    kin_data : dict
        Must contain 'RT'. Modified in place.

    feedbackOn : int
        Time point when feedback was provided.

    CursorY : numpy.ndarray
        Cursor Y-position over time.

    yTargetPos : numpy.ndarray
        Target Y-position over time.

    Returns
    -------
    dict
        Updated kin_data with:
        - 'CT': Completion time (ms), or NaN
        - 'CTexclusion': Reason/status string
    """
    if np.isnan(kin_data["RT"]):
        # No valid RT → can't compute CT
        kin_data["CT"] = np.nan
        kin_data["CTexclusion"] = "no RT"
        return kin_data

    # Find when cursor Y (+tolerance) exceeds target Y (-tolerance)
    crossing_points = np.where((CursorY + 5) > (yTargetPos[0] - 10))[0]

    if len(crossing_points) > 0:
        if crossing_points[0] > feedbackOn + 200:
            # Crossed too late → likely missed the target
            kin_data["CT"] = feedbackOn + 200
            kin_data["CTexclusion"] = "Likely Missed target"
        else:
            # Normal crossing
            kin_data["CT"] = crossing_points[0]
            kin_data["CTexclusion"] = "Crossed Y pos at CT"
    else:
        # Cursor never crossed the target
        kin_data["CT"] = np.nan
        kin_data["CTexclusion"] = "Cursor center did not cross target"

    return kin_data


# =============================================================================
# SECTION 6: INITIAL ANGLES
# =============================================================================


def calculate_initial_angles(kin_data, HandX_filt, HandY_filt):
    """
    Calculate initial movement angles at RT and RT+50ms.

    The initial angle tells us which DIRECTION the hand was moving at
    the very start of the movement. We compute this for both:
    - RT: The exact moment of movement onset
    - RT + 50ms: 50 milliseconds after onset (more stable direction)

    The angle is computed as:
        angle = arctan2(Y_displacement, X_displacement)
    Converted to degrees. 0° = rightward, 90° = upward.

    We compute these for both the primary RT and the alternative RTalt,
    giving us 4 angle values: IA_RT, IA_50RT, IA_RTalt, IA_50RTalt.

    Parameters
    ----------
    kin_data : dict
        Must contain 'RT' and 'RTalt'. Modified in place.

    HandX_filt : numpy.ndarray
        Filtered hand X-position.

    HandY_filt : numpy.ndarray
        Filtered hand Y-position.

    Returns
    -------
    dict
        Updated kin_data with:
        - 'IA_RT': Angle at primary RT
        - 'IA_50RT': Angle at RT + 50ms
        - 'IA_RTalt': Angle at alternative RT
        - 'IA_50RTalt': Angle at RTalt + 50ms
    """
    for rt_name in ["RT", "RTalt"]:
        rt_value = kin_data[rt_name]

        if not np.isnan(rt_value) and (rt_value + 50 < len(HandX_filt)):
            # --- Angle at RT ---
            # Displacement from start (0,0) to position at RT
            xdiff = HandX_filt[int(rt_value)] - HandX_filt[0]
            ydiff = HandY_filt[int(rt_value)] - HandY_filt[0]
            kin_data["IA_" + rt_name] = np.arctan2(ydiff, xdiff) * 180 / np.pi

            # --- Angle at RT + 50ms ---
            # Displacement from start to position 50ms after RT
            xdiff_50 = HandX_filt[int(rt_value + 50)] - HandX_filt[0]
            ydiff_50 = HandY_filt[int(rt_value + 50)] - HandY_filt[0]
            kin_data["IA_50" + rt_name] = np.arctan2(ydiff_50, xdiff_50) * 180 / np.pi
        else:
            # RT invalid or not enough data after RT for the +50ms window
            kin_data["IA_" + rt_name] = np.nan
            kin_data["IA_50" + rt_name] = np.nan

    return kin_data


# =============================================================================
# SECTION 7: INITIAL DIRECTION ERROR (IDE)
# =============================================================================


def calculate_ide(kin_data, HandX_filt, HandY_filt, xTargetPos, yTargetPos):
    """
    Calculate the Initial Direction Error (IDE).

    IDE measures how accurately the participant aimed their initial movement
    toward the target. It's the ANGLE between two vectors at time RT+50ms:

    1. IDEAL VECTOR: From start position -> target position at RT+50ms
       (where you SHOULD be heading)

    2. ACTUAL VECTOR: From start position -> hand position at RT+50ms
       (where you ARE actually heading)

    Sign Convention:
    - POSITIVE IDE: The actual movement leads (is ahead of) the ideal path
    - NEGATIVE IDE: The actual movement lags (is behind) the ideal path

    IMPORTANT: If the target is on the LEFT side (negative X), both vectors
    are mirrored so the sign convention remains consistent regardless of
    target direction.

    Parameters
    ----------
    kin_data : dict
        Must contain 'RT'. Modified in place.

    HandX_filt : numpy.ndarray
        Filtered hand X-position.

    HandY_filt : numpy.ndarray
        Filtered hand Y-position.

    xTargetPos : numpy.ndarray
        Target X-position at each time point.

    yTargetPos : numpy.ndarray
        Target Y-position at each time point.

    Returns
    -------
    dict
        Updated kin_data with:
        - 'IDE': Signed angle in degrees (+ = leading, - = lagging)
        - 'ABS_IDE': Absolute value of IDE

    Notes
    -----
    The angle is computed using the cross product (determinant) and
    dot product of the two vectors:
        θ = arctan2(cross_product, dot_product)

    This gives a signed angle in [-180°, 180°].
    """
    if np.isnan(kin_data["RT"]):
        kin_data["IDE"] = np.nan
        kin_data["ABS_IDE"] = np.nan
        return kin_data

    # Time point 50ms after reaction time
    rt_plus_50 = int(kin_data["RT"] + 50)

    max_length = min(len(HandX_filt), len(xTargetPos))
    if rt_plus_50 >= max_length:
        rt_plus_50 = max_length - 1

    # Starting position (always the first point)
    start_x = HandX_filt[0]
    start_y = HandY_filt[0]

    # Target position at RT+50ms
    target_x = xTargetPos[rt_plus_50]
    target_y = yTargetPos[rt_plus_50]

    # Compute the two vectors
    # Ideal vector: start → target (where you SHOULD go)
    ideal_vector = np.array([target_x - start_x, target_y - start_y])

    # Actual vector: start → hand at RT+50ms (where you ARE going)
    actual_vector = np.array(
        [HandX_filt[rt_plus_50] - start_x, HandY_filt[rt_plus_50] - start_y]
    )

    # Mirror for left-side targets
    # If target is on the left (negative X), flip X components so
    # the sign convention (+/- for leading/lagging) stays consistent
    if target_x < 0:
        ideal_vector = np.array([-ideal_vector[0], ideal_vector[1]])
        actual_vector = np.array([-actual_vector[0], actual_vector[1]])

    # Calculate the signed angle between vectors
    # Cross product (determinant) gives the sine of the angle × magnitudes
    determinant = (
        actual_vector[0] * ideal_vector[1] - actual_vector[1] * ideal_vector[0]
    )

    # Dot product gives the cosine of the angle × magnitudes
    dot_product = (
        actual_vector[0] * ideal_vector[0] + actual_vector[1] * ideal_vector[1]
    )

    # arctan2(sin, cos) gives the signed angle in radians
    theta_rad = np.arctan2(determinant, dot_product)
    theta_deg = np.degrees(theta_rad)

    kin_data["IDE"] = theta_deg if not np.isnan(theta_deg) else np.nan
    kin_data["ABS_IDE"] = np.abs(theta_deg) if not np.isnan(theta_deg) else np.nan

    return kin_data


# =============================================================================
# SECTION 8: POSITIONS, DISTANCES, AND PATH LENGTHS
# =============================================================================


def calculate_endpoint_error(kin_data, CursorX, CursorY, xTargetPos, yTargetPos):
    """
    Calculate the End-Point Error (EPE) at Completion Time.

    EPE is the Euclidean distance between the CURSOR position and the
    TARGET position at the moment of completion (CT). It measures how
    accurately the participant reached the target.

    Parameters
    ----------
    kin_data : dict
        Must contain valid 'CT'. Modified in place.

    CursorX, CursorY : numpy.ndarray
        Cursor positions over time.

    xTargetPos, yTargetPos : numpy.ndarray
        Target positions over time.

    Returns
    -------
    dict
        Updated kin_data with 'EndPointError' (mm).
    """
    ct = int(kin_data["CT"])
    kin_data["EndPointError"] = sp.euclidean_distance(
        CursorX[ct], CursorY[ct], xTargetPos[ct], yTargetPos[ct]
    )
    return kin_data


def calculate_positions_and_distances(
    kin_data, xTargetPos, yTargetPos, CursorX, HandX_filt, HandY_filt
):
    """
    Calculate position errors and distances at trial level.

    Computes three things:
    1. xPosError: How far off the cursor X is from target X at CT
    2. targetDist: How far the TARGET moved from start to CT
    3. handDist: How far the hand start is from the target at CT

    Parameters
    ----------
    kin_data : dict
        Must contain 'CT'. Modified in place.

    xTargetPos, yTargetPos : numpy.ndarray
        Target positions.

    CursorX : numpy.ndarray
        Cursor X positions.

    HandX_filt, HandY_filt : numpy.ndarray
        Filtered hand positions.

    Returns
    -------
    dict
        Updated kin_data with:
        - 'xTargetEnd': Target X at CT
        - 'yTargetEnd': Target Y at CT
        - 'xPosError': |CursorX[CT] - xTarget[CT]|
        - 'targetDist': Distance target moved (start -> CT)
        - 'handDist': Distance from hand start to target at CT
    """
    ct = kin_data["CT"]

    # Target position at completion time
    kin_data["xTargetEnd"] = xTargetPos[ct]
    kin_data["yTargetEnd"] = yTargetPos[ct]

    # X-position error (absolute difference in X only)
    kin_data["xPosError"] = np.abs(CursorX[ct] - xTargetPos[ct])

    # How far the target moved from its start to CT
    # Can be named as targetLength : Total distance travelled by the target
    kin_data["targetDist"] = sp.euclidean_distance(
        xTargetPos[0], yTargetPos[0], xTargetPos[ct], yTargetPos[ct]
    )

    # Distance from hand's starting position to the target at CT
    kin_data["handDist"] = sp.euclidean_distance(
        HandX_filt[0], HandY_filt[0], xTargetPos[ct], yTargetPos[ct]
    )

    kin_data["targetDist_Hit_Interception"] = np.nan
    # Filter and store targetDist for Hit and Interception trials
    if kin_data["Accuracy"] == 1 and kin_data["Condition"] == "Interception":
        kin_data["targetDist_Hit_Interception"] = kin_data["targetDist"]

    return kin_data


def calculate_path_lengths(
    kin_data,
    HandX_filt,
    HandY_filt,
    xTargetPos,
    yTargetPos,
    feedbackOn,
    CursorX,
    CursorY,
):
    """
    Calculate all path length measures for a trial.

    Computes four different path length measures:

    1. pathlength: ACTUAL distance the hand traveled from start to CT.
       Calculated by summing up tiny segments:
       sum of sqrt((dx)² + (dy)²) for consecutive points.

    2. idealPathLength: STRAIGHT-LINE distance from hand start to hand at CT.
       This is the shortest possible path — a perfectly straight movement.

    3. targetlength: How far the TARGET moved (its total path length).
       For Reaching trials, this is ~0 (stationary target).
       For Interception trials, this is the horizontal distance.

    4. straightlength: Length of the straight line from start to end,
       divided into as many segments as CT. Used as a reference.

    Also stores the cursor positions truncated to the movement period
    (start to min(CT, feedbackOn)) for later analysis.

    Parameters
    ----------
    kin_data : dict
        Must contain 'CT' and 'RT'. Modified in place.

    HandX_filt, HandY_filt : numpy.ndarray
        Filtered hand positions.

    xTargetPos, yTargetPos : numpy.ndarray
        Target positions.

    feedbackOn : int
        Time point of feedback onset.

    CursorX, CursorY : numpy.ndarray
        Cursor positions.

    Returns
    -------
    dict
        Updated kin_data with:
        - 'pathlength': Total hand path length (mm)
        - 'idealPathLength': Straight-line distance (mm)
        - 'targetlength': Total target path length (mm)
        - 'straightlength': Interpolated straight path length (mm)
        - 'CursorX', 'CursorY': Truncated cursor arrays
    """
    ct = int(kin_data["CT"])

    # --- 1. Actual path length (sum of tiny segments from start to CT) ---
    hand_path = np.sqrt(
        np.sum(
            np.diff(list(zip(HandX_filt[0:ct], HandY_filt[0:ct])), axis=0) ** 2, axis=1
        )
    )
    kin_data["pathlength"] = np.sum(hand_path)

    # --- 2. Ideal (straight-line) path length ---
    kin_data["idealPathLength"] = np.sqrt(
        (HandX_filt[ct] - HandX_filt[0]) ** 2 + (HandY_filt[ct] - HandY_filt[0]) ** 2
    )

    # --- 3. Target path length ---
    target_path = np.sqrt(
        np.sum(
            np.diff(list(zip(xTargetPos[0:ct], yTargetPos[0:ct])), axis=0) ** 2, axis=1
        )
    )
    kin_data["targetlength"] = np.sum(target_path)

    # --- 4. Straight path (interpolated line from start to end) ---
    pathx = [HandX_filt[0], HandX_filt[ct]]
    pathy = [HandY_filt[0], HandY_filt[ct]]
    x_new = np.linspace(start=pathx[0], stop=pathx[-1], num=ct)
    y_new = np.linspace(start=pathy[0], stop=pathy[-1], num=ct)
    s_length = np.sqrt(np.sum(np.diff(list(zip(x_new, y_new)), axis=0) ** 2, axis=1))
    kin_data["straightlength"] = np.sum(s_length)

    # --- Store truncated cursor data ---
    end_of_move = int(np.min([ct, feedbackOn]))
    kin_data["CursorX"] = CursorX[0:end_of_move]
    kin_data["CursorY"] = CursorY[0:end_of_move]

    return kin_data


def calculate_path_offset(kin_data, HandX_filt, HandY_filt):
    """
    Calculate the perpendicular deviation of the hand path from a straight line.

    For each point along the hand's path between RT and CT, we compute
    how far it deviates perpendicularly from the straight line connecting
    the start (RT) and end (CT) points.

    This tells us how "straight" the movement was:
    - Low offset → straight, efficient movement
    - High offset → curved, less efficient movement

    The formula uses the cross product to compute perpendicular distance:
        perp_distance = |start_end × point_to_start| / |start_end|

    We then normalize by the straight-line distance to get a ratio.

    Parameters
    ----------
    kin_data : dict
        Must contain 'RT' and 'CT'. Modified in place.

    HandX_filt, HandY_filt : numpy.ndarray
        Filtered hand positions.

    Returns
    -------
    dict
        Updated kin_data with:
        - 'maxpathoffset': Maximum normalized perpendicular deviation
        - 'meanpathoffset': Mean normalized perpendicular deviation
    """
    rt = int(kin_data["RT"])
    ct = int(kin_data["CT"])

    # Vector from start (RT) to end (CT) — the "straight line"
    start_end = [HandX_filt[ct] - HandX_filt[rt], HandY_filt[ct] - HandY_filt[rt]]
    start_end_distance = np.sqrt(np.sum(np.square(start_end)))
    start_end.append(0)  # Add z=0 for 3D cross product

    # For each point along the path, compute perpendicular distance
    perp_distances = []
    for m, handpos in enumerate(HandX_filt[rt:ct]):
        # Vector from start to this point
        point_to_start = [
            HandX_filt[m] - HandX_filt[rt],
            HandY_filt[m] - HandY_filt[rt],
        ]
        point_to_start.append(0)  # z=0 for 3D cross product

        # Perpendicular distance = |cross product| / |start_end|
        p = np.divide(
            np.sqrt(np.square(np.sum(np.cross(start_end, point_to_start)))),
            np.sqrt(np.sum(np.square(start_end))),
        )
        perp_distances.append(p)

    # Normalize by the total straight-line distance
    path_offset = np.divide(perp_distances, start_end_distance)

    kin_data["maxpathoffset"] = np.max(path_offset)
    kin_data["meanpathoffset"] = np.mean(path_offset)

    return kin_data


# =============================================================================
# SECTION 9: X-INTERSECT (Initial Movement Projection)
# =============================================================================


def calculate_x_intersect(
    kin_data,
    HandX_filt,
    HandY_filt,
    xTargetPos,
    yTargetPos,
    consider_window_for_intial_plan,
):
    """
    Calculate where the initial movement direction projects onto the target Y-line.

    This is a KEY measure for the Initial Estimate (IE) computation.
    We project the initial movement direction (at RT+50ms or up to RT+100ms)
    forward to where it would intersect the horizontal line at the target's
    Y-position.

    ALGORITHM:
    1. Get hand position at RT (starting point)
    2. Get hand position at RT+Δt (direction point, Δt = 50 to 100 ms)
    3. Compute the slope of the movement line
    4. Project that line to y = yTargetPos → find x_intersect
    5. If using window mode (consider_window=True):
       Try Δt = 50, 60, 70, ..., 100 ms
       Choose the Δt that gives x_intersect closest to the actual target

    WHY x_intersect matters:
    - For REACHING trials: x_intersect should be close to the target X
      (because the hand aims directly at the stationary target)
    - For INTERCEPTION trials: x_intersect predicts where the hand THINKS
      the target will be when it arrives
    - By building a regression (x_intersect → actual target) on Reaching
      trials, we can predict what the hand was aiming for in Interception

    Parameters
    ----------
    kin_data : dict
        Must contain 'RT'. Modified in place.

    HandX_filt, HandY_filt : numpy.ndarray
        Filtered hand positions.

    xTargetPos, yTargetPos : numpy.ndarray
        Target positions.

    consider_window : bool, optional
        If True, try RT+50 to RT+100 and pick the best projection.
        If False, only use RT+50. Default: True.

    Returns
    -------
    dict
        Updated kin_data with:
        - 'x_intersect': Projected X at target Y-line (mm), or NaN
        - 'x_target_at_RT': Target X-position at RT (mm), or NaN
        - 'Delta_T_Used': Which time offset was used (50-100 ms), or NaN
    """
    if np.isnan(kin_data["RT"]):
        kin_data["x_intersect"] = np.nan
        kin_data["x_target_at_RT"] = np.nan
        kin_data["Delta_T_Used"] = np.nan
        return kin_data

    rt = int(kin_data["RT"])

    # Starting point of the movement
    x0 = HandX_filt[rt]
    y0 = HandY_filt[rt]

    # Target position at RT
    x_target_at_RT = xTargetPos[rt]
    y_target_at_RT = yTargetPos[rt]
    initial_distance = np.sqrt((x_target_at_RT - x0) ** 2 + (y_target_at_RT - y0) ** 2)

    # --- Dynamic time window ---
    delta_t = 50  # Start with 50 ms
    max_delta_t = 100 if consider_window_for_intial_plan else 50
    movement_threshold = 1  # Minimum displacement in mm to be valid
    found_valid_movement = False
    x_intersect = np.nan
    delta_t_used = np.nan

    while delta_t <= max_delta_t:
        rt_plus_delta = rt + delta_t

        # Safety: don't exceed the data length
        if rt_plus_delta >= len(HandX_filt):
            rt_plus_delta = len(HandX_filt) - 1

        # Direction point
        x1 = HandX_filt[rt_plus_delta]
        y1 = HandY_filt[rt_plus_delta]

        # Movement vector
        vx = x1 - x0
        vy = y1 - y0
        displacement = np.sqrt(vx**2 + vy**2)

        if displacement >= movement_threshold and vx != 0:
            # Compute slope (m) and y-intercept (c) of the movement line
            m = vy / vx
            c = y0 - m * x0

            # Project to target Y-line: y_target = m * x + c → x = (y_target - c) / m
            x_intersect_computed = (y_target_at_RT - c) / m

            # Distance from this projection to the actual target
            new_distance = np.sqrt(
                (x_intersect_computed - x_target_at_RT) ** 2
                + (y_target_at_RT - (m * x_intersect_computed + c)) ** 2
            )

            # Keep this if it's better than previous (or first valid one)
            if np.isnan(x_intersect) or new_distance < initial_distance:
                x_intersect = x_intersect_computed
                found_valid_movement = True
                delta_t_used = delta_t
                initial_distance = new_distance

        # If not using window mode, break after first attempt
        if not consider_window_for_intial_plan:
            break

        delta_t += 10  # Try next window: 60, 70, 80, 90, 100 ms

    if not found_valid_movement:
        x_intersect = np.nan
        delta_t_used = np.nan

    kin_data["x_intersect"] = x_intersect
    kin_data["x_target_at_RT"] = x_target_at_RT
    kin_data["Delta_T_Used"] = delta_t_used

    return kin_data


# =============================================================================
# SECTION 10: CURVE-AROUND DETECTION
# =============================================================================


def check_curve_around(kin_data, CursorY, yTargetPos):
    """
    Check if the hand curves back below the target after completing the movement.

    After the cursor crosses the target Y-position (CT), does it then
    come back below the target? This would indicate an overshooting
    movement that curves around.

    Parameters
    ----------
    kin_data : dict
        Must contain 'CT'. Modified in place.

    CursorY : numpy.ndarray
        Cursor Y-position over time.

    yTargetPos : numpy.ndarray
        Target Y-position over time.

    Returns
    -------
    dict
        Updated kin_data with 'isCurveAround' (True/False/NaN).
    """
    ct = int(kin_data["CT"])

    # All time points after CT
    post_ct_indices = np.arange(ct, len(CursorY))

    if len(post_ct_indices) > 0:
        post_ct_y = CursorY[post_ct_indices]
        # Check if any Y-position after CT is below the initial target Y
        kin_data["isCurveAround"] = np.any(post_ct_y < yTargetPos[0])
    else:
        kin_data["isCurveAround"] = np.nan

    return kin_data


# =============================================================================
# SECTION 11: RESET / NaN FILL
# =============================================================================


def reset_kinematic_data(kin_data):
    """
    Set all derived kinematic measures to NaN for an invalid trial.

    When a trial has invalid RT or CT (e.g., RT=NaN, or RT >= CT),
    none of the downstream measures can be computed. This function
    sets them all to NaN to indicate "not available".

    Parameters
    ----------
    kin_data : dict
        Kinematic data dictionary. Modified in place.

    Returns
    -------
    dict
        Updated kin_data with all derived measures set to NaN.
    """
    keys_to_nan = [
        "xPosError",
        "targetDist",
        "handDist",
        "straightlength",
        "pathlength",
        "targetlength",
        "CursorX",
        "CursorY",
        "maxpathoffset",
        "meanpathoffset",
        "xTargetEnd",
        "yTargetEnd",
        "EndPointError",
        "PLR",
        "isCurveAround",
        "IDE",
        "ABS_IDE",
        "idealPathLength",
        "x_intersect",
        "x_target_at_RT",
        "Delta_T_Used",
        "isAbnormal",
        "targetDist_Hit_Interception",
    ]
    for key in keys_to_nan:
        kin_data[key] = np.nan

    return kin_data


# =============================================================================
# SECTION 12: MAIN FUNCTION — Compute Everything for One Trial
# =============================================================================


def compute_trial_kinematics(
    trial_data, defaults, trial_index, subject, consider_window=True
):
    """
    Compute ALL kinematic measures for a single trial.

    This is the MAIN FUNCTION of this module. It takes raw trial data and
    produces a dictionary with every kinematic measure. All the individual
    functions above are called in the correct order.

    FLOW:
    1. Extract metadata (condition, accuracy, arm, duration)
    2. Extract and filter hand data
    3. Compute cursor positions and hand speed
    4. Detect velocity peaks → find RT
    5. Validate RT → calculate CT
    6. If valid RT and CT:
       - Calculate all path measures (lengths, offsets, PLR)
       - Calculate all error measures (IDE, EndPointError, x_intersect)
       - Calculate angles (IA_RT, IA_50RT, etc.)
    7. If invalid: fill everything with NaN

    Parameters
    ----------
    trial_data : numpy.ndarray
        Raw trial data matrix from the .mat file.

    defaults : dict
        Configuration defaults with 'fc', 'fs', 'fdfwd'.

    trial_index : int
        Index of this trial in the experiment (0-based).

    subject : str
        Subject ID (e.g., 'cpvib001').

    consider_window : bool, optional
        Whether to use the dynamic time window for x_intersect.
        Default: True.

    Returns
    -------
    dict
        Complete kinematic data dictionary with ALL measures.
        See the module docstring for the full list.
    """
    kin_data = {}

    # -----------------------------------------------------------------
    # STEP 1: Extract metadata
    # -----------------------------------------------------------------
    metadata = extract_trial_metadata(trial_data)
    feedbackOn = metadata["feedbackOn"]
    xTargetPos = metadata["xTargetPos"]
    yTargetPos = metadata["yTargetPos"]

    kin_data["feedbackOn"] = feedbackOn
    kin_data["Accuracy"] = metadata["accuracy"]

    # Map condition code to human-readable string
    condition_mapping = {1: "Reaching", 2: "Interception"}
    kin_data["Condition"] = condition_mapping.get(metadata["condition_code"], "Unknown")

    # Map arm type to human-readable string
    arm_mapping = {1: "More Affected", 0: "Less Affected"}
    kin_data["Affected"] = arm_mapping.get(metadata["arm_type_value"], "Unknown")

    # Map duration code to actual duration in milliseconds
    duration_mapping = {1: 625, 2: 500, 3: 750, 4: 900, 5: 625, 6: 500, 7: 750, 8: 900}
    kin_data["Duration"] = duration_mapping.get(metadata["duration_code"], np.nan)

    # -----------------------------------------------------------------
    # STEP 2: Extract and filter hand data
    # -----------------------------------------------------------------
    HandX, HandY, velX, velY = extract_raw_positions_and_velocities(trial_data)

    HandX_filt, HandY_filt, velX_filt, velY_filt = sp.filter_hand_data(
        HandX, HandY, velX, velY, defaults
    )

    handspeed = sp.compute_hand_speed(velX_filt, velY_filt)
    CursorX, CursorY = sp.compute_cursor_positions(HandX, HandY, velX, velY, defaults)

    kin_data["initVel"] = handspeed[0]

    # -----------------------------------------------------------------
    # STEP 3: Find peaks and Reaction Time
    # -----------------------------------------------------------------
    peaks, properties = find_velocity_peaks(handspeed)
    kin_data["peaks"] = peaks
    kin_data["props"] = properties

    peaks, properties, kin_data = check_first_peak(peaks, properties, kin_data)
    kin_data = find_reaction_time(peaks, properties, handspeed, kin_data)
    kin_data = check_rt_contingencies(kin_data, feedbackOn, CursorY, yTargetPos)
    kin_data = calculate_alternative_rt(kin_data, handspeed)

    # -----------------------------------------------------------------
    # STEP 4: Calculate Completion Time
    # -----------------------------------------------------------------
    kin_data = calculate_completion_time(kin_data, feedbackOn, CursorY, yTargetPos)

    # -----------------------------------------------------------------
    # STEP 5: Minimum distance (cursor to target, from onset to feedback)
    # -----------------------------------------------------------------
    kin_data["minDist"] = np.min(
        sp.euclidean_distance(
            CursorX[0 : feedbackOn + 10],
            CursorY[0 : feedbackOn + 10],
            xTargetPos[0 : feedbackOn + 10],
            yTargetPos[0 : feedbackOn + 10],
        )
    )

    # -----------------------------------------------------------------
    # STEP 6: Initial angles
    # -----------------------------------------------------------------
    kin_data = calculate_initial_angles(kin_data, HandX_filt, HandY_filt)

    # -----------------------------------------------------------------
    # STEP 7: Compute all path/error measures if RT and CT are valid
    # -----------------------------------------------------------------
    if not np.isnan(kin_data["CT"]) and kin_data["RT"] < kin_data["CT"]:
        # Valid trial — compute everything
        kin_data = calculate_endpoint_error(
            kin_data, CursorX, CursorY, xTargetPos, yTargetPos
        )
        kin_data = calculate_positions_and_distances(
            kin_data, xTargetPos, yTargetPos, CursorX, HandX_filt, HandY_filt
        )
        kin_data = calculate_path_lengths(
            kin_data,
            HandX_filt,
            HandY_filt,
            xTargetPos,
            yTargetPos,
            feedbackOn,
            CursorX,
            CursorY,
        )
        kin_data = calculate_path_offset(kin_data, HandX_filt, HandY_filt)
        kin_data = check_curve_around(kin_data, CursorY, yTargetPos)

        # Path Length Ratio
        if kin_data["idealPathLength"] != 0:
            kin_data["PLR"] = kin_data["pathlength"] / kin_data["idealPathLength"]
        else:
            kin_data["PLR"] = np.nan

        # IDE (Initial Direction Error)
        kin_data = calculate_ide(
            kin_data, HandX_filt, HandY_filt, xTargetPos, yTargetPos
        )

        # x_intersect (Initial movement projection)
        kin_data = calculate_x_intersect(
            kin_data, HandX_filt, HandY_filt, xTargetPos, yTargetPos, consider_window
        )

        # Flag abnormal IDE (> 90 degrees)
        kin_data["isAbnormal"] = (
            abs(kin_data["IDE"]) > 90 if not np.isnan(kin_data["IDE"]) else np.nan
        )

        # Store filtered hand data for PVar computation later
        kin_data["HandX_filt"] = HandX_filt.tolist()
        kin_data["HandY_filt"] = HandY_filt.tolist()
    else:
        # Invalid trial — set everything to NaN
        kin_data = reset_kinematic_data(kin_data)

    return kin_data
