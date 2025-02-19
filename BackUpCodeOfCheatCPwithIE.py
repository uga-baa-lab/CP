
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
import os
import uuid
import logging
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score
import TrialPlots as plots
from Config import RESULTS_DIR, consider_window_for_intial_plan
import logging
logging.basicConfig(level=logging.INFO)
from Utils import clean_data_iqr, initialize_arms_data_structure_for_regression, set_seaborn_preference, lowPassFilter
import Utils as utils



set_seaborn_preference()

def get_hand_positions_and_velocities(thisData):
    # Get non-filtered and filtered hand positions and velocities
    HandX = thisData.T[4]  # X position of hand
    HandY = thisData.T[5]  # Y position of hand
    velX = thisData.T[6]  # X velocity
    velY = thisData.T[7]  # Y velocity
    return HandX, HandY, velX, velY



def getHandTrajectories(thisData, defaults):
    HandX, HandY, velX, velY = get_hand_positions_and_velocities(thisData)
    HandX_filt, HandY_filt, velX_filt, velY_filt = utils.filter_hand_data(HandX, HandY, velX, velY, defaults)
    handspeed = utils.calculate_handspeed(velX_filt, velY_filt)
    CursorX, CursorY = compute_cursor_positions(HandX, HandY, velX, velY, defaults)

    trajData = {
        'xTargetPos': thisData.T[1],
        'yTargetPos': thisData.T[2],
        'HandX_filt': HandX_filt,
        'HandY_filt': HandY_filt,
        'velX_filt': velX_filt,
        'velY_filt': velY_filt,
        'handspeed': handspeed,
        'CursorX': CursorX,
        'CursorY': CursorY
    }
    return trajData
# In this scenario thisData contains all the x,y cordinates, velocity profile for a particular trial.
def extract_values_from_data(thisData):
    feedbackOn = int(thisData.T[15][0])
    xTargetPos = thisData.T[1]
    yTargetPos = thisData.T[2]  # constant y postion of the target
    condition = thisData.T[12][0]
    accuracy  =  thisData.T[14][0]
    condition_code = int(thisData[0, 12])
    arm_type_value = int(thisData.T[16][0])  # Get the value from the data to use as a key
    duration = int(thisData.T[11][0])
    return feedbackOn, xTargetPos, yTargetPos, condition, accuracy, condition_code, arm_type_value, duration

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

    kinData = dict()
    feedbackOn, xTargetPos, yTargetPos, condition, accuracy, condition_code, arm_type_value, duration = extract_values_from_data(thisData)
    kinData['feedbackOn'] = feedbackOn
    
    kinData['Condtion'] = condition # 1 for Reaching, 2 for Interception
    kinData['Accuracy']  =  accuracy #0 for Miss, 1 for Hit
   
    HandX, HandY, velX, velY = get_hand_positions_and_velocities(thisData)
    HandX_filt, HandY_filt, velX_filt, velY_filt = utils.filter_hand_data(HandX, HandY, velX, velY, defaults)
    handspeed = utils.calculate_handspeed(velX_filt, velY_filt)

    target_type_mapping = {1: 'Reaching', 2: 'Interception'}
    target_type_value = int(kinData['Condtion'])
    kinData['Condtion'] = target_type_mapping.get(target_type_value, 'Unknown')
    kinData['initVel'] = handspeed[0]
    CursorX, CursorY = compute_cursor_positions(HandX, HandY, velX, velY, defaults)
    condition_mapping = {1: 'Reaching', 2: 'Interception'}
    kinData['Condition'] = condition_mapping.get(condition_code, 'Unknown')


    arm_type_mapping = {1: 'More Affected', 0: 'Less Affected'}
    kinData['Affected'] = arm_type_mapping.get(arm_type_value, 'Unknown')  # Pass the key correctly

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
    kinData['minDist'] = np.min(utils.dist(CursorX[0:feedbackOn+10], CursorY[0:feedbackOn+10],
                                xTargetPos[0:feedbackOn+10], yTargetPos[0:feedbackOn+10]))

    if not np.isnan(kinData['CT']) and kinData['RT'] < kinData['CT']:
        if(i > 90 and subject == 'cpvib40'):
            logging.debug("Reached condition for subject cpvib40 with i > 90")
        kinData['EndPointError'] = utils.dist(CursorX[int(kinData['CT'])], CursorY[int(kinData['CT'])],
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
        kinData = utils.reset_kinematic_data(kinData)

    return kinData



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
    # logging.info(f'Initial Direction Error Value : {kinData["IDE"]}')
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
        model.fit(X, y)
        a = model.coef_[0]
        b = model.intercept_
        regression_coeffs[arm] = {'a': a, 'b': b, 'model': model}
    return regression_coeffs, data_by_arm


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
        findonset = handspeed[0:peaks[0]] < kinData['velPeak'] * 0.05
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

        if not np.isnan(kinData[v]) and ( kinData[v] + 50 < len(HandX_filt)): #Taking 50 ms more than RT for Calculating IA_50›
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
    kinData['targetDist'] = utils.dist(
        xTargetPos[0], yTargetPos[0], xTargetPos[kinData['CT']], yTargetPos[kinData['CT']])
    # Total distance travelled by the hand from initial position to the target
    kinData['handDist'] = utils.dist(
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




def getDataCP(mdf, matfiles, defaults):
    # initialize dataframe
    all_df = pd.DataFrame()

    # initialize trajectory data cell
    allTrajs = {}
    row_name_str=  ["cpvib066"]
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

    output_path = os.path.join(RESULTS_DIR, 'IE_CSV_Results')
    all_trials_df = pd.concat(all_trials_ie_by_subject, ignore_index=True)
    print("Columns in the DataFrame:", all_trials_df.columns)

    grouped_data = all_trials_df.sort_values(by=['Subject', 'Condition'])

    file_name = os.path.join(output_path,'All_Trials_IE.csv')
    grouped_data_file_name = os.path.join(output_path,'Grouped_IE.csv')
    grouped_data_file_name = os.path.join(output_path, 'All_Trials_IE_Grouped.csv')
    grouped_data.to_csv(grouped_data_file_name, index=False)

    pd.concat(all_trials_ie_by_subject,ignore_index=True).to_csv(file_name,index=True)
    return all_df, allTrajs







