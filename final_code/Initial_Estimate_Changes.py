
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:00:35 2020

@author: dab32176

Functions for CHEATCP
"""

from scipy import signal
import scipy.io
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import TrialPlots as plots


consider_window_for_intial_plan = False

def define_defaults():
    defaults = dict()
    defaults['fs'] = 1e3  # sampling frequency
    defaults['fc'] = 5  # low pass cut-off (Hz)
    # feedforward estimate of hand position used by KINARM (constant)
    defaults['fdfwd'] = 0.06
    # Define condition order for plotting
    defaults['reachorder'] = ['Reaching', 'Interception']
    defaults['grouporder'] = ['CP', 'TDC']
    defaults['armorder'] = ['More Affected', 'Less Affected']
    return defaults


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
    dist = np.sqrt(np.square(x1-x2)+np.square(y1-y2))
    return dist


def getHandTrajectories(thisData, defaults):
    # init dictionary
    trajData = dict()

    trajData['xTargetPos'] = thisData.T[1]
    trajData['yTargetPos'] = thisData.T[2]  # constant y postion of the target

    # Get non-filtered and filtered hand positions and velocities
    HandX = thisData.T[4]  # X position of hand
    HandY = thisData.T[5]  # Y position of hand
    velX = thisData.T[6]  # X velocity
    velY = thisData.T[7]  # Y velocity

    # filtered data, speed and acceleration
    trajData['HandX_filt'] = lowPassFilter(
        HandX, defaults['fc'], defaults['fs'])
    trajData['HandY_filt'] = lowPassFilter(
        HandY, defaults['fc'], defaults['fs'])
    trajData['velX_filt'] = lowPassFilter(velX, defaults['fc'], defaults['fs'])
    trajData['velY_filt'] = lowPassFilter(velY, defaults['fc'], defaults['fs'])

    trajData['handspeed'] = np.sqrt(
        trajData['velX_filt']**2 + trajData['velY_filt']**2)
   # accel = np.append(0,np.diff(handspeed/defaults['fs'])) #Acceleration

    # Cursor position Cursor position is based on hand position + velocity *feedforward estimate (velocity-dependent!)
    trajData['CursorX'] = HandX + defaults['fdfwd'] * velX
    trajData['CursorY'] = HandY + defaults['fdfwd'] * velY

    return trajData

# In this scenario thisData contains all the x,y cordinates, velocity profile for a particular trial.


def getHandKinematics(thisData, defaults, i, subject):
    """
    Parameters
    ----------
    thisData : raw hand kinematic data
    defaults : default parameter settings

    Returns
    -------
    kinData : processed hand kinematic data, including:
        - Reaction Time (RT)
        - Completion Time (CT)
        - Peak Velocity (velPeak)
        - Initial Angles (IA_RT, IA_50RT, etc.)
        - Initial Movement Direction Error (IDE)
        - End-Point Error (EPE)
        - Path Length Ratio (PLR)
        - Other existing measures from original code
    """

    kinData = initialize_kinData(thisData)
    xTargetPos = thisData.T[1]
    yTargetPos = thisData.T[2]  # constant y postion of the target
    feedbackOn = int(thisData.T[15][0])
    kinData['feedbackOn'] = feedbackOn
    kinData['Condtion'] = thisData.T[12][0]
    target_type_mapping = {1: 'Reaching', 2: 'Interception'}
    target_type_value = int(kinData['Condtion'])
    kinData['Condtion'] = target_type_mapping.get(target_type_value, 'Unknown')
    kinData['Accuracy']  =  thisData.T[14][0]
    
    HandX, HandY, velX, velY = get_hand_positions_and_velocities(thisData)
    HandX_filt, HandY_filt, velX_filt, velY_filt = filter_hand_data(
        HandX, HandY, velX, velY, defaults)
    handspeed = np.sqrt(velX_filt**2 + velY_filt**2)

    kinData['initVel'] = handspeed[0]
    CursorX, CursorY = compute_cursor_positions(
        HandX, HandY, velX, velY, defaults)

        
    condition_mapping = {1: 'Reaching', 2: 'Interception'}
    condition_code = int(thisData[0, 12])
    kinData['Condition'] = condition_mapping.get(condition_code, 'Unknown')

    arm_type_value = int(thisData.T[16][0])  # Get the value from the data to use as a key
    arm_type_mapping = {1: 'More Affected', 0: 'Less Affected'}
    kinData['Affected'] = arm_type_mapping.get(arm_type_value, 'Unknown')  # Pass the key correctly

    duration = int(thisData.T[11][0])
    duration_mapping = {1: 625, 2: 500, 3: 750, 4: 900, 5: 625, 6: 500, 7: 750, 8: 900}
    kinData['Duration'] = duration_mapping.get(duration, np.nan)

    # Determine the peak movements in the trial

    [peaks, props] = signal.find_peaks(
        handspeed, height=max(handspeed)/4, distance=150)
    kinData['peaks'] = peaks
    kinData['props'] = props

    peaks, props, kinData = check_first_peak(peaks, props, kinData)
    kinData = find_reaction_time(peaks, props, handspeed, kinData)
    kinData = check_rt_contingencies(kinData, feedbackOn, CursorY, yTargetPos)
    kinData = calculate_alternative_rt(kinData, handspeed)
    kinData = calculate_movement_time(kinData, feedbackOn, CursorY, yTargetPos)
    # more contingencies for RT: connect be <100 or greater than when feedback comes on
    # also the Y pos at RT cannot exceed the Y position of the target
    kinData = calculate_initial_angles(kinData, HandX_filt, HandY_filt)
    # plot_vectors(HandX_filt, HandY_filt, kinData, xTargetPos, yTargetPos)
    # minimum distance between target and cursor (from onset to feedback)
    kinData['minDist'] = np.min(dist(CursorX[0:feedbackOn+10], CursorY[0:feedbackOn+10],
                                xTargetPos[0:feedbackOn+10], yTargetPos[0:feedbackOn+10]))

    if not np.isnan(kinData['CT']) and kinData['RT'] < kinData['CT']:
        if(i > 90 and subject == 'cpvib40'):
            print("I'm here")
        kinData['EndPointError'] = dist(CursorX[int(kinData['CT'])], CursorY[int(kinData['CT'])],
                                        xTargetPos[int(kinData['CT'])], yTargetPos[int(kinData['CT'])])
        kinData = calculate_position_and_distances(
            kinData, xTargetPos, yTargetPos, CursorX, HandX_filt, HandY_filt)
        kinData = calculate_path_lengths(
            kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, feedbackOn, CursorX, CursorY)
        kinData = calculate_path_offset(kinData, HandX_filt, HandY_filt)
        kinData = check_for_curve_around(kinData, CursorY, yTargetPos)
        kinData['PLR'] = (kinData['pathlength']/kinData['idealPathLength']
                            if kinData['idealPathLength'] != 0 else np.nan)
        kinData = calculate_initial_movement_direction_error(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, i,subject,CursorX, CursorY, velX_filt, velY_filt)
        kinData = calculate_x_intersect(
            kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos)
        kinData['isAbnormal'] = abs(kinData['IDE']) > abs(90)
        
        #Ideal Path Length is the St. line distance between Hand(0,0) and Hand at CT.
        #Path length is sum of total distance travelled from (0,0) to (CT)
    else:
        kinData = reset_kinematic_data(kinData)

    return kinData


# def calculate_initial_movement_direction_error(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos):
#     """
#     Calculate Initial Movement Direction Error (IDE)
#     Calculate the ideal movement angle from start to target

#     """
#     if(np.isnan(kinData['RT'])):
#         kinData['IDE'] = np.nan
#         return kinData
    
#     reaction_time_50 = kinData['RT'] + 50
#     start_x = HandX_filt[0] #Here I'm taking the start time as 0,0 but we can take it as RT and the endtime as RT+50 - may be an option to consider (Trial Specific)
#     start_y = HandY_filt[0]
#     target_x = xTargetPos[reaction_time_50]
#     target_y = yTargetPos[reaction_time_50]
#     # max_length = min(len(HandX_filt), len(xTargetPos))
#     ideal_vector = np.array([target_x-start_x, target_y - start_y])
#     actual_vector = np.array([HandX_filt[reaction_time_50] - start_x, HandY_filt[reaction_time_50]-start_y])
   
#     # dot_product = np.dot(actual_vector,ideal_vector)
#     # #TODO check the norm/magnitude of participant vector and if the value turned out to be 0, then set the IDE as na
#     # magnitude_product = np.linalg.norm(actual_vector) * np.linalg.norm(ideal_vector)
#     # if magnitude_product == 0:
#     #     return 0
#     # cos_theta = dot_product/magnitude_product
#     # cos_theta = np.clip(cos_theta,-1.0,1.0)
#     # theta_rad = np.arccos(cos_theta)
#     # theta_deg = np.degrees(theta_rad)
#     # theta_deg =  theta_deg if dot_product > 0 else -theta_deg

#     determinant = actual_vector[0] * ideal_vector[1] - actual_vector[1] * ideal_vector[0]
#     dot_product = actual_vector[0] * ideal_vector[0] + actual_vector[1] * ideal_vector[1]
#     theta_rad = np.arctan2(determinant, dot_product)
#     theta_deg = np.degrees(theta_rad)
#     kinData['IDE'] = theta_deg if theta_deg else np.nan
#     plots.plot_trial_IDE(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, theta_deg)

#     return kinData


# def calculate_initial_movement_direction_error(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos,i,subject,CursorX, CursorY, velX_filt, velY_filt):
#     """
#     Calculate Initial Movement Direction Error (IDE).

#     Parameters:
#     - kinData: Dictionary containing trial information, including 'RT'.
#     - HandX_filt: Numpy array of filtered hand X positions.
#     - HandY_filt: Numpy array of filtered hand Y positions.
#     - xTargetPos: Numpy array of target X positions.
#     - yTargetPos: Numpy array of target Y positions.

#     Returns:
#     - kinData: Updated dictionary with 'IDE' value.
#     """
#     if np.isnan(kinData['RT']):
#         kinData['IDE'] = np.nan
#         return kinData
#     reaction_time = int(kinData['RT'])
#     reaction_time_50 = int(kinData['RT'] + 50)
#     idx_RT = reaction_time_50
#     delta_x = HandX_filt[reaction_time_50] - HandX_filt[reaction_time]
#     delta_y = HandY_filt[reaction_time_50] - HandY_filt[reaction_time]

#     if delta_x == 0:
#         slope = np.inf
#     else:
#         slope = delta_y / delta_x
#     y_target_plane = yTargetPos[reaction_time_50]
#     if np.isfinite(slope):
#         # x_intersect = HandX_filt[reaction_time] + (y_target_plane - HandY_filt[idx_RT]) / slope
#         x_intersect = HandX_filt[reaction_time] + (y_target_plane - HandY_filt[reaction_time]) * (delta_x / delta_y)

#     else:
#         x_intersect = HandX_filt[reaction_time]

#     kinData['x_intersect'] = x_intersect
#     kinData['x_target_at_RT'] = xTargetPos[reaction_time]



#     max_length = min(len(HandX_filt), len(xTargetPos))
#     if reaction_time_50 >= max_length:
#         reaction_time_50 = max_length - 1
#     start_x = HandX_filt[0]
#     start_y = HandY_filt[0]
#     target_x = xTargetPos[reaction_time_50]
#     target_y = yTargetPos[reaction_time_50]
#     ideal_vector = np.array([target_x - start_x, target_y - start_y])
#     actual_vector = np.array([HandX_filt[reaction_time_50] - start_x, HandY_filt[reaction_time_50] - start_y])
#     if target_x < 0:
#         ideal_vector = np.array([-ideal_vector[0], ideal_vector[1]])
#         actual_vector = np.array([-actual_vector[0], actual_vector[1]])

#     determinant = actual_vector[0] * ideal_vector[1] - actual_vector[1] * ideal_vector[0]
#     dot_product = actual_vector[0] * ideal_vector[0] + actual_vector[1] * ideal_vector[1]
#     theta_rad = np.arctan2(determinant, dot_product)
#     theta_deg = np.degrees(theta_rad)
#     kinData['IDE'] = theta_deg if not np.isnan(theta_deg) else np.nan
#     plots.plot_trial_IDE(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, theta_deg, i,subject,CursorX, CursorY, velX_filt, velY_filt)

#     return kinData

def calculate_initial_movement_direction_error(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos,i,subject,CursorX, CursorY, velX_filt, velY_filt):
    """
    Calculate Initial Movement Direction Error (IDE).

    Parameters:
    - kinData: Dictionary containing trial information, including 'RT'.
    - HandX_filt: Numpy array of filtered hand X positions.
    - HandY_filt: Numpy array of filtered hand Y positions.
    - xTargetPos: Numpy array of target X positions.
    - yTargetPos: Numpy array of target Y positions.

    Returns:
    - kinData: Updated dictionary with 'IDE' value.
    """
    if np.isnan(kinData['RT']):
        kinData['IDE'] = np.nan
        kinData['ABS_IDE'] = np.nan
        return kinData

    reaction_time_50 = int(kinData['RT'] + 50)
    max_length = min(len(HandX_filt), len(xTargetPos))
    if reaction_time_50 >= max_length:
        reaction_time_50 = max_length - 1
    start_x = HandX_filt[0]
    start_y = HandY_filt[0]
    target_x = xTargetPos[reaction_time_50]
    target_y = yTargetPos[reaction_time_50]
    ideal_vector = np.array([target_x - start_x, target_y - start_y])
    actual_vector = np.array([HandX_filt[reaction_time_50] - start_x, HandY_filt[reaction_time_50] - start_y])
    if target_x < 0:
        ideal_vector = np.array([-ideal_vector[0], ideal_vector[1]])
        actual_vector = np.array([-actual_vector[0], actual_vector[1]])
    
    determinant = actual_vector[0] * ideal_vector[1] - actual_vector[1] * ideal_vector[0]
    dot_product = actual_vector[0] * ideal_vector[0] + actual_vector[1] * ideal_vector[1]
    theta_rad = np.arctan2(determinant, dot_product)
    theta_deg = np.degrees(theta_rad)
    kinData['IDE'] = theta_deg if not np.isnan(theta_deg) else np.nan
    kinData['ABS_IDE'] = np.abs(theta_deg) if not np.isnan(theta_deg) else np.nan
    # print(f'Initial Direction Error Value : {kinData['IDE']}')
    # plots.plot_trial_IDE(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, theta_deg, i,subject,CursorX, CursorY, velX_filt, velY_filt)

    return kinData

def calculate_x_intersect(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos):
    if np.isnan(kinData['RT']):
        kinData['x_intersect'] = np.nan
        kinData['x_target_at_RT'] = np.nan
        return kinData

    RT = int(kinData['RT'])
    x0 = HandX_filt[RT]
    y0 = HandY_filt[RT]
    x_target_at_RT = xTargetPos[RT]
    y_target_at_RT = yTargetPos[RT]
    initial_distance = np.sqrt((x_target_at_RT - x0)**2 + (y_target_at_RT - y0)**2)
    # Initialize variables for dynamic time window
    delta_t = 50  # Start with 50 ms
    max_delta_t = 100 if consider_window_for_intial_plan else 50 # Maximum window size
    # print(f'Max_Delta size is : {max_delta_t}')
    movement_threshold = 1  # expecting Minimum displacement in mm
    found_valid_movement = False
    x_intersect = np.nan
    while delta_t <= max_delta_t:
        RT_plus_delta = RT + delta_t
        if RT_plus_delta >= len(HandX_filt):
            RT_plus_delta = len(HandX_filt) - 1 #making sure we're not having RT+delta value beyond the hand movement points
        x1 = HandX_filt[RT_plus_delta]
        y1 = HandY_filt[RT_plus_delta]
        vx = x1 - x0  
        vy = y1 - y0
        displacement = np.sqrt(vx**2 + vy**2)
        if displacement >= movement_threshold and vx != 0:
            # Compute slope (m) and intercept (c)
            m = vy / vx
            c = y0 - m * x0
            # Calculate x_intersect
            x_intersect_computed = (y_target_at_RT - c) / m
            new_distance = np.sqrt((x_intersect_computed - x_target_at_RT)**2 + (y_target_at_RT - (m * x_intersect_computed + c))**2)
            # print(f'New Distance evaluated between target and the x_intersect at RT+ {delta_t} is {new_distance}')
            if np.isnan(x_intersect) or new_distance < initial_distance:
                x_intersect = x_intersect_computed
                found_valid_movement = True
                delta_t_used = delta_t
                initial_distance = new_distance
            # break let's iterate till RT+100
        if not consider_window_for_intial_plan:
            break   
        delta_t += 10
    if not found_valid_movement:
        x_intersect = np.nan
        delta_t_used = np.nan

    kinData['x_intersect'] = x_intersect
    kinData['x_target_at_RT'] = x_target_at_RT
    kinData['Delta_T_Used'] = delta_t_used
    return kinData

# def calculate_initial_movement_direction_error(
#     kinData,
#     HandX_filt,
#     HandY_filt,
#     xTargetPos,
#     yTargetPos,
#     i,
#     subject,
#     CursorX,
#     CursorY,
#     velX_filt,
#     velY_filt,
#     defaults  # Pass the defaults to access sampling frequency
# ):
#     # Check if RT is NaN
#     if np.isnan(kinData['RT']):
#         kinData['IDE'] = np.nan
#         kinData['x_intersect'] = np.nan
#         kinData['x_target_at_RT'] = np.nan
#         return kinData
#     # print(1/0)
#     # Reaction time index
#     RT = int(kinData['RT'])

#     # Get hand position and velocity at RT
#     x_rt = HandX_filt[RT]
#     y_rt = HandY_filt[RT]
#     v_x_rt = velX_filt[RT]
#     v_y_rt = velY_filt[RT]

#     # Get target y position at RT
#     y_target_at_RT = yTargetPos[RT]

#     # Store x_target_at_RT for regression
#     kinData['x_target_at_RT'] = xTargetPos[RT]

#     # Compute t_intersect
#     if v_y_rt != 0:
#         t_intersect = (y_target_at_RT - y_rt) / v_y_rt

#         # Calculate x_intersect
#         x_intersect = x_rt + v_x_rt * t_intersect

#         kinData['x_intersect'] = x_intersect
#     else:
#         # If v_y_rt is zero, cannot compute intersection
#         kinData['x_intersect'] = np.nan

#     # Now, compute IDE (Initial Direction Error)

#     # The ideal movement vector is from start position to target position at RT
#     start_x = HandX_filt[0]
#     start_y = HandY_filt[0]
#     target_x = xTargetPos[RT]
#     target_y = yTargetPos[RT]
#     ideal_vector = np.array([target_x - start_x, target_y - start_y])

#     # The actual movement vector is from start position to hand position at RT + 50 ms
#     RT_50 = RT + int(50 / (1000 / defaults['fs']))  # Convert 50 ms to samples

#     if RT_50 >= len(HandX_filt):
#         RT_50 = len(HandX_filt) - 1

#     actual_vector = np.array([HandX_filt[RT_50] - start_x, HandY_filt[RT_50] - start_y])

#     # Calculate angle between vectors
#     determinant = actual_vector[0] * ideal_vector[1] - actual_vector[1] * ideal_vector[0]
#     dot_product = actual_vector[0] * ideal_vector[0] + actual_vector[1] * ideal_vector[1]
#     theta_rad = np.arctan2(determinant, dot_product)
#     theta_deg = np.degrees(theta_rad)

#     kinData['IDE'] = theta_deg if not np.isnan(theta_deg) else np.nan

#     return kinData

# def calculate_initial_movement_direction_error(
#     kinData,
#     HandX_filt,
#     HandY_filt,
#     xTargetPos,
#     yTargetPos,
#     i,
#     subject,
#     CursorX,
#     CursorY,
#     velX_filt,
#     velY_filt,
#     window_size=10  # Define the window size around RT+50
# ):
#        # Check if RT is NaN
#     if np.isnan(kinData['RT']):
#         kinData['IDE'] = np.nan
#         kinData['x_intersect'] = np.nan
#         kinData['x_target_at_RT'] = np.nan
#         return kinData

#     # Define the primary reaction time index
#     reaction_time = int(kinData['RT'])
#     reaction_time_50 = int(kinData['RT'] + 50)

#     # Define the window around RT+50
#     window_start = reaction_time_50 - window_size
#     window_end = reaction_time_50 + window_size

#     # Ensure window boundaries are within data limits
#     max_length = min(len(HandX_filt), len(xTargetPos))
#     window_start = max(reaction_time, window_start)
#     window_end = min(window_end, max_length - 1)

#     # Initialize lists to store x_intersect candidates
#     x_intersect_candidates = []
#     valid_indices = []

#     for t in range(window_start, window_end + 1):
#         # Calculate deltas
#         delta_x = HandX_filt[t] - HandX_filt[reaction_time]
#         delta_y = HandY_filt[t] - HandY_filt[reaction_time]

#         # Avoid division by zero
#         if delta_x == 0:
#             slope = np.inf
#         else:
#             slope = delta_y / delta_x

#         # Get the target Y position at current time point
#         y_target_plane = yTargetPos[t]

#         # Calculate x_intersect based on slope
#         if np.isfinite(slope) and slope != 0:
#             # Compute x_intersect where the movement line intersects y_target_plane
#             x_intersect = HandX_filt[reaction_time] + (y_target_plane - HandY_filt[t]) / slope
#         else:
#             # If slope is infinite or zero, set x_intersect to HandX at reaction time
#             x_intersect = HandX_filt[reaction_time]

#         x_intersect_candidates.append(x_intersect)
#         valid_indices.append(t)

#     # Select the most meaningful x_intersect from candidates
#     if x_intersect_candidates:
#         # Criteria for selection:
#         # Here, we choose the median x_intersect to reduce the influence of any remaining outliers
#         selected_x_intersect = np.median(x_intersect_candidates)
        
#         # Optionally, you can choose the x_intersect corresponding to the median time point
#         # median_index = valid_indices[np.argsort(x_intersect_candidates)[len(x_intersect_candidates) // 2]]
#         # selected_x_intersect = x_intersect_candidates[len(x_intersect_candidates) // 2]

#         # Assign selected x_intersect
#         kinData['x_intersect'] = selected_x_intersect

#         # Assign x_target_at_RT using the selected time point
#         # Find the closest time index to the window's center
#         center_time = reaction_time_50
#         closest_time = min(valid_indices, key=lambda x: abs(x - center_time))
#         kinData['x_target_at_RT'] = xTargetPos[closest_time]
#     else:
#         # If no valid x_intersect found within the window, assign NaN
#         kinData['x_intersect'] = np.nan
#         kinData['x_target_at_RT'] = np.nan

#     # After selecting x_intersect, calculate IDE

#     # Define start and target positions
#     start_x = HandX_filt[0]
#     start_y = HandY_filt[0]

#     if not np.isnan(kinData['x_intersect']) and not np.isnan(kinData['x_target_at_RT']):
#         target_x = kinData['x_intersect']
#         target_y = yTargetPos[closest_time]

#         # Compute ideal and actual movement vectors
#         ideal_vector = np.array([target_x - start_x, target_y - start_y])
#         actual_vector = np.array([HandX_filt[closest_time] - start_x, HandY_filt[closest_time] - start_y])

#         # Adjust for target direction if necessary
#         if target_x < 0:
#             ideal_vector = np.array([-ideal_vector[0], ideal_vector[1]])
#             actual_vector = np.array([-actual_vector[0], actual_vector[1]])

#         # Calculate angle between vectors using arctan2 of determinant and dot product
#         determinant = actual_vector[0] * ideal_vector[1] - actual_vector[1] * ideal_vector[0]
#         dot_product = actual_vector[0] * ideal_vector[0] + actual_vector[1] * ideal_vector[1]
#         theta_rad = np.arctan2(determinant, dot_product)
#         theta_deg = np.degrees(theta_rad)

#         kinData['IDE'] = theta_deg if not np.isnan(theta_deg) else np.nan
#     else:
#         kinData['IDE'] = np.nan

#     return kinData

from sklearn.linear_model import LinearRegression
def perform_subject_level_regression(subject_trials):
    """
    Performs linear regression for each subject and arm using Reaching trials.

    Parameters:
        subject_trials: list of dictionaries containing kinData for each trial.

    Returns:
        regression_coeffs: dictionary with regression coefficients for each arm.
    """
    # Initialize dictionaries to hold data for each arm
    data_by_arm = {'Less Affected': {'x_intersect': [], 'x_target': []},
                   'More Affected': {'x_intersect': [], 'x_target': []}}

    # Collect data from Reaching trials
    for kinData in subject_trials:
        condition = kinData['Condition']
        arm = kinData['Affected']  # 'Less Affected' or 'More Affected'
        if condition == 'Reaching' and not np.isnan(kinData['x_intersect']) and not np.isnan(kinData['x_target_at_RT']):
            data_by_arm[arm]['x_intersect'].append(kinData['x_intersect'])
            data_by_arm[arm]['x_target'].append(kinData['x_target_at_RT'])

    # Perform regression for each arm
    regression_coeffs = {}
    for arm, data in data_by_arm.items():
        x_intersects = np.array(data['x_intersect'])
        x_targets = np.array(data['x_target'])

        mean = np.mean(x_intersects)
        std_dev = np.std(x_intersects)
        threshold = 3 * std_dev
        lower_bound = mean - threshold
        upper_bound = mean + threshold

        # Filter out NaN values
        non_outlier_indices = (x_intersects >= lower_bound) & (x_intersects <= upper_bound)
        x_intersects_filtered = x_intersects[non_outlier_indices]
        x_targets_filtered = x_targets[non_outlier_indices]

        # Check for minimum data points
        X = x_intersects_filtered.reshape(-1, 1)
        y = x_targets_filtered
        model = LinearRegression()
        model.fit_predict(X, y)
        a = model.coef_[0]
        b = model.intercept_
        regression_coeffs[arm] = {'a': a, 'b': b, 'model': model}
    return regression_coeffs, data_by_arm
'''
# import matplotlib.pyplot as plt
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# def plot_regression_for_subject_arm(data_by_subject_arm, regression_models):
#     for arm, data in data_by_subject_arm.items():
#         x = data['x_intersect']
#         y = data['x_target']
#         x_clean, y_clean = remove_outliers_iqr(x, y)
#         if len(x_clean) >= 2:
#             X = np.array(x_clean).reshape(-1, 1)
#             y = np.array(y_clean)
#             model = regression_models[arm]['model']
#             # Plotting
#             plt.figure(figsize=(8, 6))
#             plt.scatter(X, y, color='blue', label='Data Points')
#             x_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#             y_fit = model.predict(x_fit)
#             plt.plot(x_fit, y_fit, color='red', label='Regression Line')
#             # Calculate R^2
#             r_squared = model.score(X, y)
#             # Display equation and R^2
#             a = model.coef_[0]
#             b = model.intercept_
#             equation_text = f'y = {a:.2f}x + {b:.2f}\n$R^2$ = {r_squared:.2f}'
#             plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
#                      fontsize=12, verticalalignment='top')
#             plt.xlabel('x_intersect')
#             plt.ylabel('x_target_at_RT')
#             plt.title(f' Arm: {arm} - Reaching Trials')
#             plt.legend()
#             plt.grid(True)
#             plt.show()
#         else:
#             print(f"Not enough data to plot for subject {subject}, arm {arm}.")
'''
def calculate_ie_for_interception_trials(subject_trials, regression_coeffs,subject):
    """
    Applies the regression model to Interception trials to compute IE.

    Parameters:
        subject_trials: list of dictionaries containing kinData for each trial.
        regression_coeffs: dictionary with regression coefficients for each arm.

    Returns:
        Updated subject_trials with IE calculated for Interception trials.
    """
    trial_list=[]
    for kinData in subject_trials:
        condition = kinData['Condition']
        arm = kinData['Affected']
        duration = kinData['Duration']
        a = regression_coeffs[arm]['a']
        b = regression_coeffs[arm]['b']

        if condition == 'Interception' and not np.isnan(kinData['x_intersect']) and not np.isnan(a) and not np.isnan(b):
            x_predicted = a * kinData['x_intersect'] + b
            IE = x_predicted - kinData['x_target_at_RT']
            kinData['IE'] = IE
        else:
            kinData['IE'] = np.nan
        trial_list.append([kinData['IE'], duration, condition, subject])
        result_df = pd.DataFrame(trial_list, columns=['IE', 'Duration', 'Condition', 'Subject'])
    return result_df


# def plot_regression_points(subject, data_by_arm, regression_coeffs,visit_day):
#     """
#     Plots the regression line and data points for each arm of a subject.

#     Parameters:
#         subject: Subject ID (for labeling the plot)
#         data_by_arm: Dictionary containing data per arm used for regression
#         regression_coeffs: Dictionary with regression coefficients per arm, i.e., regression_coeffs[arm] = {'a': a, 'b': b}
#     """
#     for arm in data_by_arm.keys():
#         # Ensure data is converted to NumPy arrays
#         x_positions_list = np.array(data_by_arm[arm]['x_intersect'])
#         x_target_rt_list = np.array(data_by_arm[arm]['x_target'])
#         print(f'Size of X_intersect points for Subject : {subject}_{visit_day} - arm {arm} : are : {len(x_positions_list)} and target size : {len(x_target_rt_list)}')


#         if len(x_positions_list) == 0 or len(x_target_rt_list) == 0:
#             print(f"No valid data for arm {arm} of subject {subject}")
#             continue

#         plt.figure(figsize=(8, 6))
#         plt.scatter(x_positions_list, x_target_rt_list, color='blue', label='Data Points')

#         # Get regression coefficients
#         a = regression_coeffs[arm]['a']
#         b = regression_coeffs[arm]['b']

#         if not np.isnan(a) and not np.isnan(b):
#             x_fit = np.linspace(np.min(x_positions_list), np.max(x_positions_list), 100)
#             y_fit = a * x_fit + b
#             plt.plot(x_fit, y_fit, color='red', label='Regression Line')

#             # Calculate y_pred for R^2 calculation
#             y_pred = a * x_positions_list + b

#             # Calculate R^2
#             ss_res = np.sum((x_target_rt_list - y_pred) ** 2)
#             ss_tot = np.sum((x_target_rt_list - np.mean(x_target_rt_list)) ** 2)
#             r_squared = 1 - (ss_res / ss_tot)

#             # Display regression equation and R^2
#             equation_text = f'y = {a:.2f}x + {b:.2f}\n$R^2$ = {r_squared:.2f}'
#             plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
#                      fontsize=12, verticalalignment='top')
#         else:
#             plt.text(0.05, 0.95, 'Insufficient data for regression', transform=plt.gca().transAxes,
#                      fontsize=12, verticalalignment='top', color='red')

#         plt.xlabel('Intersection Point (x_intersect)')
#         plt.ylabel('Target Position at RT (x_target_at_RT)')
#         plt.title(f'Subject: {subject} - Day {visit_day}, Arm: {arm} - Reaching Trials')
#         plt.legend()
#         plt.grid(True)
#         subject_folder = os.path.join(r'C:\Users\LibraryUser\Downloads\Fall2024\BrainAndAction\CP\CP\results\IE_plots', subject)
#         if not os.path.exists(subject_folder):
#             os.makedirs(subject_folder)
#         plot_filename = f'{subject}_{visit_day}_{arm}.png'
#         plt.savefig(os.path.join(subject_folder, plot_filename), bbox_inches='tight')
#         plt.close()

def plot_regression_points(subject, data_by_arm, regression_coeffs, visit_day):
    for arm in data_by_arm.keys():
        # Ensure data is converted to NumPy arrays
        x_positions_list = np.array(data_by_arm[arm]['x_intersect'])
        x_target_rt_list = np.array(data_by_arm[arm]['x_target'])
        print(f'Size of X_intersect points for arm {arm}: {len(x_positions_list)}')

        if len(x_positions_list) == 0 or len(x_target_rt_list) == 0:
            print(f"No valid data for arm {arm} of subject {subject}")
            continue

        plt.figure(figsize=(8, 6))
        plt.scatter(x_positions_list, x_target_rt_list, color='blue', label='Data Points')

        # Get regression coefficients
        a = regression_coeffs[arm]['a']
        b = regression_coeffs[arm]['b']

        if not np.isnan(a) and not np.isnan(b):
            # Compute predicted y values for actual x positions
            y_pred = a * x_positions_list + b

            # Sort x and y for plotting
            sorted_indices = np.argsort(x_positions_list)
            x_sorted = x_positions_list[sorted_indices]
            y_pred_sorted = y_pred[sorted_indices]

            # Plot the regression line using the predicted values
            plt.plot(x_sorted, y_pred_sorted, color='red', label='Regression Line')

            # Calculate R^2
            ss_res = np.sum((x_target_rt_list - y_pred) ** 2)
            ss_tot = np.sum((x_target_rt_list - np.mean(x_target_rt_list)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Display regression equation and R^2
            equation_text = f'y = {a:.2f}x + {b:.2f}\n$R^2$ = {r_squared:.2f}'
            plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top')
        else:
            plt.text(0.05, 0.95, 'Insufficient data for regression',
                     transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top', color='red')

        plt.xlabel('Intersection Point (x_intersect)')
        plt.ylabel('Target Position at RT (x_target_at_RT)')
        plt.title(f'Subject: {subject} - Day {visit_day}, Arm: {arm} - Reaching Trials')
        plt.legend()
        plt.grid(True)
        subject_folder = os.path.join('path_to_results', subject)
        print(f'Sub Folder : {subject_folder}')
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)
        plot_filename = f'{subject}_{visit_day}_{arm}.png'
        plt.savefig(os.path.join(subject_folder, plot_filename), bbox_inches='tight')
        plt.close()




def check_first_peak(peaks, props, kinData):
    """
    Checks if the first peak is a bad peak (before 100 ms) and removes it if necessary.
    """
    if len(props['peak_heights']) > 1 and peaks[0] < 100:  # First peak is not real if before 100 ms
        print('bad first peak')
        peaks = np.delete(peaks, 0)
        props['peak_heights'] = np.delete(props['peak_heights'], 0)
        kinData['badfirstpeak'] = True
    else:
        kinData['badfirstpeak'] = False
    return peaks, props, kinData


def find_reaction_time(peaks, props, handspeed, kinData):
    """
    Finds the reaction time by moving backwards from the first peak to the onset where 
    hand speed drops below 5% of the peak height.
    """
    if len(peaks) > 0:
        # Taking the first real peak, not the overall peak velocity
        kinData['velPeak'] = props['peak_heights'][0]
        kinData['velLoc'] = peaks[0]

        # Move backwards from the first peak to determine RT
        findonset = handspeed[0:peaks[0]] < props['peak_heights'][0] * 0.05
        onset = np.where(findonset == True)[0]

        if len(onset) > 0:
            kinData['RT'] = onset[-1] + 1
        else:
            kinData['RT'] = np.nan
            kinData['velPeak'] = np.nan
            kinData['velLoc'] = np.nan
            kinData['RTexclusion'] = 'could not find onset'
    else:
        print('no peaks found')
        kinData['RT'] = np.nan
        kinData['velPeak'] = np.nan
        kinData['velLoc'] = np.nan
        kinData['RTexclusion'] = 'no peaks'

    return kinData


def check_rt_contingencies(kinData, feedbackOn, CursorY, yTargetPos):
    """
    Checks for contingencies on reaction time (RT), ensuring it is valid.
    """
    if kinData['RT'] < 100 or kinData['RT'] > feedbackOn or CursorY[0] > yTargetPos[0]:
        kinData['RT'] = np.nan
        kinData['velPeak'] = np.nan
        kinData['velLoc'] = np.nan
        kinData['RTexclusion'] = 'outlier value'
    else:
        kinData['RTexclusion'] = 'good RT'

    return kinData


def calculate_alternative_rt(kinData, handspeed, threshold=100):
    """
    Calculates an alternative reaction time (RTalt) based on when the hand speed exceeds a threshold.
    """
    findonset = handspeed > threshold
    onset = np.where(findonset == True)[0]

    if len(onset) > 0:
        kinData['RTalt'] = onset[0] + 1
    else:
        kinData['RTalt'] = np.nan

    return kinData


def calculate_movement_time(kinData, feedbackOn, CursorY, yTargetPos):
    """
    Calculates the movement time (CT), the time when the cursor crosses the y-position of the target.
    """
    if not np.isnan(kinData['RT']):
        # Placeholder adjustment values (+5 and -10)
        # findCT = np.where((CursorY + 5) > (yTargetPos[0] - 10))[0]
        findCT= np.where((CursorY+5)>(yTargetPos[0]-10))[0]
        if len(findCT) > 0:
            if findCT[0] > feedbackOn + 200:
                kinData['CT'] = feedbackOn + 200
                kinData['CTexclusion'] = 'Likely Missed target'
            else:
                kinData['CT'] = findCT[0]
                kinData['CTexclusion'] = 'Crossed Y pos at CT'
        else:
            kinData['CT'] = np.nan
            kinData['CTexclusion'] = 'Cursor center did not cross target'

        return kinData
    else:
        kinData['CT'] = np.nan
        kinData['CTexclusion'] = 'no RT'
        return kinData


def calculate_initial_angles(kinData, HandX_filt, HandY_filt):
    for v in ['RT', 'RTalt']:

        if not np.isnan(kinData[v]) and ( kinData[v] + 50 < len(HandX_filt)):
            xdiff = HandX_filt[int(kinData[v])] - HandX_filt[0]
            ydiff = HandY_filt[int(kinData[v])] - HandY_filt[0]
            kinData['IA_' + v] = np.arctan2(ydiff, xdiff) * 180 / np.pi
            xdiff_50 = HandX_filt[int(kinData[v] + 50)] - HandX_filt[0]
            ydiff_50 = HandY_filt[int(kinData[v] + 50)] - HandY_filt[0]
            kinData['IA_50' + v] = np.arctan2(ydiff_50, xdiff_50) * 180 / np.pi
        else:
            kinData['IA_' + v] = np.nan
            kinData['IA_50' + v] = np.nan
    return kinData


def calculate_position_and_distances(kinData, xTargetPos, yTargetPos, CursorX, HandX_filt, HandY_filt):
    kinData['xTargetEnd'] = xTargetPos[kinData['CT']]
    kinData['yTargetEnd'] = yTargetPos[kinData['CT']]
    kinData['xPosError'] = np.abs(
        CursorX[kinData['CT']] - xTargetPos[kinData['CT']])
    # distance from start position to target position at time y position crossed
    # Can be named as targetLength : Total distance travelled by the target
    kinData['targetDist'] = dist(
        xTargetPos[0], yTargetPos[0], xTargetPos[kinData['CT']], yTargetPos[kinData['CT']])
    # Total distance travelled by the hand from initial position to the target
    kinData['handDist'] = dist(
        HandX_filt[0], HandY_filt[0], xTargetPos[kinData['CT']], yTargetPos[kinData['CT']])
    return kinData


def calculate_path_lengths(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, feedbackOn, CursorX, CursorY):
    hand_path_lengths = np.sqrt(np.sum(np.diff(list(zip(
        HandX_filt[0:kinData['CT']], HandY_filt[0:kinData['CT']])), axis=0)**2, axis=1))
    kinData['pathlength'] = np.sum(hand_path_lengths)

    ideal_distance = np.sqrt((HandX_filt[kinData['CT']] - HandX_filt[0])**2 +
                             (HandY_filt[kinData['CT']] - HandY_filt[0])**2)
    kinData['idealPathLength'] = ideal_distance

    target_lengths = np.sqrt(np.sum(np.diff(list(zip(
        xTargetPos[0:kinData['CT']], yTargetPos[0:kinData['CT']])), axis=0)**2, axis=1))
    kinData['targetlength'] = np.sum(target_lengths)

    pathx = [HandX_filt[0], HandX_filt[kinData['CT']]]
    pathy = [HandY_filt[0], HandY_filt[kinData['CT']]]
    x_new = np.linspace(start=pathx[0], stop=pathx[-1], num=int(kinData['CT']))
    y_new = np.linspace(start=pathy[0], stop=pathy[-1], num=int(kinData['CT']))
    slength = np.sqrt(
        np.sum(np.diff(list(zip(x_new, y_new)), axis=0)**2, axis=1))
    kinData['straightlength'] = np.sum(slength)
    endofMove = np.min([kinData['CT'], feedbackOn])
    # Truncate CursorX and CursorY data based on end of movement (feedbackOn)
    kinData['CursorX'] = CursorX[0:endofMove]
    kinData['CursorY'] = CursorY[0:endofMove]
    return kinData

    # timepoints = kinData['CT'] - kinData['RT'] + 1; #i.e., MT
    # xPath = np.linspace(HandX_filt[kinData['RT']],HandX_filt[kinData['CT']],timepoints)
    # yPath = np.linspace(HandY_filt[kinData['RT']],HandY_filt[kinData['CT']],timepoints)


def calculate_path_offset(kinData, HandX_filt, HandY_filt):
    """
    Calculate the perpendicular distance (path offset) from the straight-line path.
    """
    start_end = [HandX_filt[kinData['CT']] - HandX_filt[kinData['RT']],
                 HandY_filt[kinData['CT']] - HandY_filt[kinData['RT']]]
    start_end_distance = np.sqrt(np.sum(np.square(start_end)))
    start_end.append(0)  # Append a 0 to represent the z-axis

    perp_distance = []
    for m, handpos in enumerate(HandX_filt[kinData['RT']:kinData['CT']]):
        thispointstart = [[HandX_filt[m]-HandX_filt[kinData['RT']]],
                          [HandY_filt[m]-HandY_filt[kinData['RT']]]]
        thispointstart.append([0])

        thispointstart = [HandX_filt[m]-HandX_filt[kinData['RT']],
                          HandY_filt[m]-HandY_filt[kinData['RT']]]
        thispointstart.append(0)

        p = np.divide(np.sqrt(np.square(np.sum(np.cross(start_end, thispointstart)))),
                      np.sqrt(np.sum(np.square(start_end))))

        perp_distance.append(p)

    pathoffset = np.divide(perp_distance, start_end_distance)
    kinData['maxpathoffset'] = np.max(pathoffset)
    kinData['meanpathoffset'] = np.mean(pathoffset)

    return kinData


def check_for_curve_around(kinData, CursorY, yTargetPos):
    """
    Check if the movement curves back below the target y-position after CT
    """
    post_CT_indices = np.arange(int(kinData['CT']), len(CursorY))
    if len(post_CT_indices) > 0:
        post_CT_YPoints = CursorY[post_CT_indices]
        # Check if any y-position after CT is less than the initial target y-position
        kinData['isCurveAround'] = np.any(post_CT_YPoints < yTargetPos[0])
    else:
        kinData['isCurveAround'] = np.nan
    return kinData


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
        'Delta_T_Used' : np.nan
    })
    return kinData


def plot_path(kinData, HandX_filt, HandY_filt, RT, CT):
    plt.plot(HandX_filt[RT:], HandY_filt[RT:],
             color='b', label="Hand Positions")
    plt.plot([HandX_filt[RT], HandX_filt[CT]], [HandY_filt[RT],
             HandY_filt[CT]], color='r', linestyle='--', label="Straight Path")
    plt.show()
    return kinData


def check_for_curve_around(kinData, CursorY, yTargetPos):
    post_CT_indices = np.arange(int(kinData['CT']), len(CursorY))
    if len(post_CT_indices) > 0:
        post_CT_YPoints = CursorY[post_CT_indices]
        kinData['isCurveAround'] = np.any(post_CT_YPoints < yTargetPos[0])
    else:
        kinData['isCurveAround'] = np.nan
    return kinData


def fill_nan_values(kinData):
    keys_to_nan = ['xPosError', 'targetDist', 'handDist', 'straightlength', 'pathlength', 'targetlength',
                   'CursorX', 'CursorY', 'maxpathoffset', 'meanpathoffset', 'xTargetEnd', 'yTargetEnd',
                   'EndPointError', 'PLR', 'isCurveAround', 'IDE']
    for key in keys_to_nan:
        kinData[key] = np.nan
    return kinData


def compute_cursor_positions(HandX, HandY, velX, velY, defaults):
    CursorX = HandX + defaults['fdfwd'] * velX
    CursorY = HandY + defaults['fdfwd'] * velY
    return CursorX, CursorY


def get_hand_positions_and_velocities(thisData):
    HandX = thisData.T[4]
    HandY = thisData.T[5]
    velX = thisData.T[6]
    velY = thisData.T[7]
    return HandX, HandY, velX, velY


def filter_hand_data(HandX, HandY, velX, velY, defaults):
    HandX_filt = lowPassFilter(HandX, defaults['fc'], defaults['fs'])
    HandY_filt = lowPassFilter(HandY, defaults['fc'], defaults['fs'])
    velX_filt = lowPassFilter(velX, defaults['fc'], defaults['fs'])
    velY_filt = lowPassFilter(velY, defaults['fc'], defaults['fs'])
    return HandX_filt, HandY_filt, velX_filt, velY_filt


def initialize_kinData(thisData):
    kinData = dict()
    kinData['feedbackOn'] = int(thisData.T[15][0])
    return kinData


def getDataCP(mdf, matfiles, defaults):
    # initialize dataframe
    all_df = pd.DataFrame()

    # initialize trajectory data cell
    allTrajs = {}
    row_name_str=  ["cpvib078"]
    for index, row in mdf.iterrows():
        if row['KINARM ID'] not in row_name_str:
            continue
        elif row['KINARM ID'].startswith('CHEAT'):
            subject = row['KINARM ID'][-3:]
        else:
            subject = row['KINARM ID']
        # if row['KINARM ID'].startswith('CHEAT'):
        print(f'evaluating the subject : {subject}')
        subjectmat = 'CHEAT-CP'+subject+row['Visit_Day']+'.mat'

        mat = os.path.join(matfiles, subjectmat)

        if not os.path.exists(mat):
            print('skipping', mat)
            # if row['Visit_Day'] =='Day1':
            #    assert 0
            continue

        loadmat = scipy.io.loadmat(mat)

        data = loadmat['subjDataMatrix'][0][0]
        # print(data.keys())
        # assert 0
        # data[23] is all data for trial 24
        # data[23].T[4] is all data for the 5th column of that trial

        # simple, for each trial, collect (a) condition--12, (b) TP--11 (c) hitormiss-->14 (d) feedback time from 15
        # (e) hitormis and feedback time from 10 SKIP (f) trial duration 13 ; affectedhand 16 (1 if using affected)

        allTrials = []
        all_trials_df = []
        subjectTrajs = []
        all_trials_ie_by_subject = []
        for i in range(len(data)):
            thisData = data[i]
            trajData = getHandTrajectories(thisData, defaults)
            kinData = getHandKinematics(thisData, defaults,i,subject)
            row_values = [kinData['Condition'], thisData.T[16][0], thisData.T[11][0],
                          thisData.T[13][0], thisData.T[14][0], thisData.T[15][0],
                          kinData['RT'], kinData['CT'], kinData['velPeak'],
                          kinData['xPosError'], kinData['minDist'], kinData['targetDist'], kinData['handDist'], kinData['straightlength'],
                          kinData['pathlength'], kinData['targetlength'], kinData['CursorX'], kinData['CursorY'],
                          kinData['IA_RT'], kinData['IA_50RT'], kinData['RTalt'], kinData['IA_RTalt'],
                          kinData['maxpathoffset'], kinData['meanpathoffset'], kinData['xTargetEnd'], kinData['yTargetEnd'], kinData['EndPointError'], kinData['IDE'], kinData['PLR'], kinData['isCurveAround'],i]
            
            
            allTrials.append(row_values)
            all_trials_df.append(kinData)
            subjectTrajs.append(trajData)
        print(f'evaluated the kindData for sub : {subject}')
        regression_coeffs,data_by_arm = perform_subject_level_regression(all_trials_df)
        plot_regression_points(subject,data_by_arm,regression_coeffs,row['Visit_Day'])
        subject_wise_IE = calculate_ie_for_interception_trials(all_trials_df, regression_coeffs,subject)
        all_trials_ie_by_subject.append(subject_wise_IE)
        
        df = pd.DataFrame(allTrials, columns=['Condition', 'Affected', 'TP', 'Duration', 'Accuracy', 'FeedbackTime',
                                              'RT', 'CT', 'velPeak', 'xPosError', 'minDist', 'targetDist', 'handDist', 'straightlength',
                                              'pathlength', 'targetlength', 'cursorX', 'cursorY', 'IA_RT', 'IA_50RT',
                                              'RTalt', 'IA_RTalt', 'maxpathoffset', 'meanpathoffset', 'xTargetEnd', 'yTargetEnd', 'EndPointError', 'IDE', 'PLR', 'isCurveAround','trial_number'])
        # data cleaning
        df['Affected'] = df['Affected'].map(
            {1: 'More Affected', 0: 'Less Affected'})
        df['Condition'] = df['Condition'].map(
            {1: 'Reaching', 2: 'Interception'})
        df['Duration'] = df['TP'].map(
            {1: 625, 2: 500, 3: 750, 4: 900, 5: 625, 6: 500, 7: 750, 8: 900})
        df['MT'] = df['CT'] - df['RT']
        df['subject'] = subject
        df['age'] = row['Age at Visit (yr)']
        df['visit'] = row['Visit ID']
        df['day'] = row['Visit_Day']
        df['studyid'] = row['Subject ID']

        if row['Group'] == 0:
            df['group'] = 'TDC'
        else:
            df['group'] = 'CP'

        df['pathratio'] = df['pathlength'] / df['targetlength']
        all_df = pd.concat([all_df, df])

        # combine all trajectories
        allTrajs[subject+row['Visit_Day']] = subjectTrajs
    output_path = os.path.join(r'C:\Users\LibraryUser\Downloads\Fall2024\BrainAndAction\CP\CP\results')
    all_trials_df = pd.concat(all_trials_ie_by_subject, ignore_index=True)
    print("Columns in the DataFrame:", all_trials_df.columns)

    grouped_data = all_trials_df.sort_values(by=['Subject', 'Condition'])

    file_name = os.path.join(output_path,'All_Trials_IE.csv')
    grouped_data_file_name = os.path.join(output_path,'Grouped_IE.csv')
    grouped_data_file_name = os.path.join(output_path, 'All_Trials_IE_Grouped.csv')
    grouped_data.to_csv(grouped_data_file_name, index=False)

    pd.concat(all_trials_ie_by_subject,ignore_index=True).to_csv(file_name,index=True)
    return all_df, allTrajs

# plotting functions


def plot_groupmeans(varlist, df):
    for vartotest in varlist:
        g = sns.FacetGrid(df, col="Condition", legend_out=True,
                          col_order=['Reaching', 'Interception'])
        g = g.map(sns.pointplot, 'group', vartotest, 'Affected', order=['CP', 'TDC'], hue_order=['Less Affected', 'More Affected'],
                  palette=sns.color_palette("muted"))
        g = g.map(sns.stripplot, 'group', vartotest, 'Affected', order=['CP', 'TDC'], hue_order=['Less Affected', 'More Affected'],
                  palette=sns.color_palette("muted"))


def plot_byduration(varlist, df):
    """Note this pools TDC and CP"""
    for vartotest in varlist:
        g = sns.FacetGrid(df, col="Condition", legend_out=True)
        g = g.map(sns.pointplot, "Duration", vartotest, "Affected", order=[500, 625, 750, 900], hue_order=['Less Affected', 'More Affected'],
                  # Blue is Non-Affected (0), Orange is Affected (1) (need to verify)
                  palette=sns.color_palette("muted"))
        g.add_legend()


def plot_twocolumnbar(vartotest, col, col_order, x, hue, df, order, hue_order):
    g = sns.FacetGrid(df, col=col, col_order=col_order, legend_out=True)
    g = g.map(sns.barplot, x, vartotest, hue, order=order, hue_order=hue_order,
              palette=sns.color_palette("muted"), alpha=.6)
    g = g.map(sns.stripplot, x, vartotest, hue, order=order, hue_order=hue_order,
              palette=sns.color_palette("muted"), split=True)
    g.savefig(vartotest+col+x+'.jpg', format='jpeg', dpi=300)


def plot_singletraj(plotsubject, plotday, trajx, allTrajs, all_df):
    subject_df = all_df.loc[(all_df['subject'] == plotsubject) & (
        all_df['day'] == plotday)]
    traj = allTrajs[plotsubject+plotday][trajx]
    trajinfo = subject_df.iloc[trajx]

    # plot it
    fig, ax = plt.subplots()
    ft = int(trajinfo['FeedbackTime'])
    plt.plot(traj['CursorX'][0:ft], traj['CursorY'][0:ft])
    plt.plot(traj['CursorX'][int(trajinfo['RT'])],
             traj['CursorY'][int(trajinfo['RT'])], 'bo')
    plt.plot(traj['CursorX'][int(trajinfo['RTalt'])],
             traj['CursorY'][int(trajinfo['RTalt'])], 'go')
    circle1 = plt.Circle(
        (traj['xTargetPos'][ft], traj['yTargetPos'][ft]), 10, color='r')

    ax.add_patch(circle1)
    ax.axis('equal')
    ax.set(xlim=(-150, 150), ylim=(40, 200))


"""
# publication plot
y = 'Accuracy'
x = 'Duration'
hue = 'Affected'
col = 'Condition'
two_color_palette = sns.color_palette(["#f1a340","#998ec3"])
data = all_df
savename = (y+'CP2.pdf')
                                       
groupcols = ['subject',x]
for param in [hue,col]:
    if param is not None:
        groupcols.append(param)
data = data.groupby(groupcols).mean().reset_index()
background = data.assign(idhue = lambda df: df['subject'] \
                            .astype(str)+df[hue].astype(str))

#define orderings
x_order =list(background[x].unique())
hue_order = list(background["idhue"].unique()) #purple = more affected, orange = 'less affected'
point_hue_order = list(background[hue].unique())
col_order =  ['Reaching','Interception']  #list(background[col].unique())

g = sns.FacetGrid(background,col=col,col_order=col_order,legend_out=False) 

g.map(sns.pointplot,x,y,"idhue",palette=two_color_palette,scale=.7,markers='',
                    order = x_order,hue_order=hue_order)

backgroundartists = []
for ax in g.axes.flat:
    for l in ax.lines + ax.collections:
        l.set_zorder(1)
        l.set_alpha(0.4)
        backgroundartists.append(l)
#ci=95 is "bootstrap 95% CIs"--> could do bootstrap 68% CI as alternative
g.map(sns.pointplot, x, y, 'hue',unit="subject",palette=two_color_palette,alpha=0,scale=1.5,markers='',errwidth=2,capsize=.1,
      order = list(background[x].unique()),hue_order=point_hue_order)

#create own legend
h=2
w=1
g.fig.set_size_inches(h,w)
#    g.set(xlabel='',ylabel=ylabel)  

for ax in g.axes.flat:
    for l in ax.lines + ax.collections:
        if l not in backgroundartists:
            l.set_zorder(2)
            l.set_alpha(1)
g.savefig(savename)
"""

# Plotting trajectories
#    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize = (5,10), dpi=100)
#    axs = axs.flatten()
#    #plt.subplots_adjust(bottom=0.1, right=9, top=0.9)
#    # Create a continuous norm to map from data points to colors
#
#    trials=[]
#   # trialends=[]
#    for t,thistrial in enumerate(trialconds):
#        trialcond=thistrial
#           # incorrecttry = 1
#       # trialdir = -1 #left to right
#       # trialvel = 'Fast'
#       # trialacc = 1 #Accurate
#
#        #locate trial options
#        trials = df.loc[(df['Condition']==trialcond)
#                             & (df['subject']==subject)]
#        trials = trials.sample(frac=1)
#        for index, trial in trials.iterrows():
#           # trial = trials.loc[trials.index[i]]
#          #  trialend = int(trial['kinEnd'])
#            if trial['Affected']=='More Affected':
#                color = "#998ec3"
#            else:
#                color = "#f1a340" #orange
#          #      if trial['Decision']=='Decision':
#           #         color = "#2c7fb8" #"#0571b0" #blue
#           #     else:
#           #         if np.random.randint(2)==0: #only plot half the trials
#           #             continue
#           #         color = "#7fcdbb" #"#b2abd2"
#           #
#           # else:
#            #    color = "#ca0020" #red
#
#            axs[t].plot(trial['cursorX'],trial['cursorY'],linewidth = .7,c=color) #correct go
#
#    #set up each subplot the same way
#    for ax in axs:
#        ax.set_xlim(-170, 170)
#        ax.set_ylim(0, 250)
#        ax.set(adjustable='box', aspect='equal')
#        ax.set_xlabel('X position (mm)')
#        ax.set_ylabel('Y position (mm)')
#       # rect = patches.Rectangle((-170,140),340,340,linewidth=1,edgecolor='k',facecolor='none')
#        rect = patches.Rectangle((-170,140),340,340,linewidth=1,edgecolor='k',facecolor='none')
#
#        lcbar = LineCollection([[(-50,50),(50,50)]], colors=np.array([(0, 0, 0, 1)]), linewidths=1.5)
#       # ax.add_patch(rect)
#       # ax.add_collection(lcbar)
#    plt.show()
#    fig.tight_layout(pad=0)
#    fig.savefig('CP_trajectories.pdf')
#    assert 0
