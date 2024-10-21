
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

def set_seaborn_preference():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 10
    rcstyle = {'axes.linewidth': 1.0,
               'axes.edgecolor': 'black', 'ytick.minor.size': 5.0}
    sns.set(font_scale=1.0, rc={'figure.figsize': (20, 10)})
    sns.set_style('ticks', rcstyle)
    sns.set_context("paper", rc={"lines.linewidth": 1,
                    "xtick.labelsize": 10, "ytick.labelsize": 10})


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
    kinData['Target_Type'] = thisData.T[12][0]
    target_type_mapping = {1: 'Reaching', 2: 'Interception'}
    target_type_value = int(kinData['Target_Type'])
    kinData['Target_Type'] = target_type_mapping.get(target_type_value, 'Unknown')
    kinData['Accuracy']  =  thisData.T[14][0]

    HandX, HandY, velX, velY = get_hand_positions_and_velocities(thisData)
    HandX_filt, HandY_filt, velX_filt, velY_filt = filter_hand_data(
        HandX, HandY, velX, velY, defaults)
    handspeed = np.sqrt(velX_filt**2 + velY_filt**2)

    kinData['initVel'] = handspeed[0]
    CursorX, CursorY = compute_cursor_positions(
        HandX, HandY, velX, velY, defaults)

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
        calculate_initial_movement_direction_error(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, i,subject,CursorX, CursorY, velX_filt, velY_filt)

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
    # plots.plot_trial_IDE(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, theta_deg, i,subject,CursorX, CursorY, velX_filt, velY_filt)

    return kinData




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
        'PLR': np.nan,
        'PLR_2': np.nan,
        'isCurveAround': np.nan,
        'idealPathlength': np.nan
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
    row_name_str=  ["cpvib040"]
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
        subjectTrajs = []
        for i in range(len(data)):
            thisData = data[i]
            trajData = getHandTrajectories(thisData, defaults)
            # print(f"Calling the Kinematics function for the sub : {subjectmat}")
            kinData = getHandKinematics(thisData, defaults,i,subject)

            row_values = [thisData.T[12][0], thisData.T[16][0], thisData.T[11][0],
                          thisData.T[13][0], thisData.T[14][0], thisData.T[15][0],
                          kinData['RT'], kinData['CT'], kinData['velPeak'],
                          kinData['xPosError'], kinData['minDist'], kinData['targetDist'], kinData['handDist'], kinData['straightlength'],
                          kinData['pathlength'], kinData['targetlength'], kinData['CursorX'], kinData['CursorY'],
                          kinData['IA_RT'], kinData['IA_50RT'], kinData['RTalt'], kinData['IA_RTalt'],
                          kinData['maxpathoffset'], kinData['meanpathoffset'], kinData['xTargetEnd'], kinData['yTargetEnd'], kinData['EndPointError'], kinData['IDE'], kinData['PLR'], kinData['isCurveAround'],i]

            allTrials.append(row_values)
            subjectTrajs.append(trajData)
        print(f'evaluated the kindData for sub : {subject}')

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
