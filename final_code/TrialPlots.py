import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import numpy as np
import uuid
import logging as logger
from scipy.interpolate import interp1d, splprep, splev
from scipy import stats

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



# def plot_trial_IDE(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, theta_deg):
   
#     completion_time = int(kinData['CT'])
#     reaction_time_50 = int(kinData['RT']+50)
#     reaction_time_100 = int(kinData['RT']+100)
#     target_x = xTargetPos[reaction_time_50]
#     target_y = yTargetPos[reaction_time_50]

#     accuracy_value = kinData['Accuracy']
#     end_point_error = kinData['EndPointError']
#     path_length_ratio = kinData['PLR']

#     plt.plot(HandX_filt[0:], HandY_filt[0:], label='Participant Path', color='blue')
#     plt.scatter(HandX_filt[0], HandY_filt[0], color='yellow', label='Start Position')
#     plt.scatter(HandX_filt[-1], HandY_filt[-1], color='pink', label='End Position')
#     plt.scatter(target_x, target_y, color='orange', label="Target Position", zorder=5)
#     plt.scatter(HandX_filt[completion_time], HandY_filt[completion_time], color = 'brown', label = 'Hand Co-ordinates at Completion Time' )
#     plt.scatter(HandX_filt[reaction_time_50],HandY_filt[reaction_time_50],color='olive', label='Point of HandPath at RT50')
#     if int(kinData['RT'] + 50) < len(HandX_filt):
#         participant_vector = np.array([
#             HandX_filt[int(kinData['RT'] + 50)] - HandX_filt[0],
#             HandY_filt[int(kinData['RT'] + 50)] - HandY_filt[0]
#         ])
#     else:
#         participant_vector = np.array([np.nan, np.nan])
    
#     ideal_vector = np.array([
#         xTargetPos[int(kinData['RT'] + 50)] - HandX_filt[0],
#         yTargetPos[int(kinData['RT'] + 50)] - HandY_filt[0]
#     ])

#     plt.quiver(
#             HandX_filt[0], HandY_filt[0],
#             ideal_vector[0], ideal_vector[1],
#             angles='xy', scale_units='xy', scale=1, color='orange', label='Ideal Vector',headwidth=2,headlength = 3
#         )

#     plt.quiver(
#             HandX_filt[0], HandY_filt[0],
#             participant_vector[0], participant_vector[1],
#             angles='xy', scale_units='xy', scale=1, color='black', label='Participant Vector',headwidth=2,headlength = 3
#         )

#     mid_x = HandX_filt[0] + (participant_vector[0] + ideal_vector[0]) / 4
#     mid_y = HandY_filt[0] + (participant_vector[1] + ideal_vector[1]) / 4

#         # Annotate the angle
#     plt.text(mid_x, mid_y, f'IDE = {theta_deg:.2f}° and Condition Is : {kinData["Target_Type"]}', fontsize=12, color='black')
#     plt.text(mid_x- 10,mid_y -10, f'Subject Initial vector : {participant_vector} \n Ideal Vector : {ideal_vector}', fontsize = 8, color='black')
#     plt.xlabel('X Position (mm)')
#     plt.ylabel('Y Position (mm)')
#     plt.title('Participant Movement and Ideal Movement Vectors')
#     plt.legend()
#     plt.axis('equal')
#     plt.grid(True)
#     plt.savefig(os.path.join(r'C:\Users\LibraryUser\Downloads\Fall2024/BrainAndAction\CP\CP\results\plots_v2', str(uuid.uuid4()) + '.png'),bbox_inches='tight')
#     plt.close()



# def plot_trial_IDE(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, theta_deg, i, subject):
    
#     completion_time = int(kinData['CT'])
#     reaction_time_50 = int(kinData['RT']+50)
#     reaction_time_100 = int(kinData['RT']+100)
#     target_x = xTargetPos[reaction_time_50]
#     target_y = yTargetPos[reaction_time_50]
#     target_x_at_ct = xTargetPos[completion_time]
#     accuracy_value = kinData['Accuracy']
#     end_point_error = kinData['EndPointError']
#     path_length_ratio = kinData['PLR']
#     condition = kinData['Target_Type']
    
#     # Create folder for subject if it doesn't exist
#     subject_folder = os.path.join(r'C:\Users\LibraryUser\Downloads\Fall2024\BrainAndAction\CP\CP\results\plots_v2', subject)
#     if not os.path.exists(subject_folder):
#         os.makedirs(subject_folder)

    
#     plt.figure(figsize=(10, 8))
#     plt.plot(HandX_filt[0:], HandY_filt[0:], label='Participant Path', color='blue')
#     plt.scatter(HandX_filt[0], HandY_filt[0], color='yellow', label='Start Position')
#     plt.scatter(HandX_filt[-1], HandY_filt[-1], color='pink', label='End Position')
#     plt.scatter(target_x_at_ct,target_y, color='maroon', label='End Position')
#     plt.scatter(target_x, target_y, color='orange', label="Target Position", zorder=5)
#     plt.scatter(HandX_filt[completion_time], HandY_filt[completion_time], color='brown', label='Hand Coordinates at Completion Time')
#     plt.scatter(HandX_filt[reaction_time_50], HandY_filt[reaction_time_50], color='olive', label='Point of HandPath at RT50')

#     if reaction_time_50 < len(HandX_filt):
#         participant_vector = np.array([
#             HandX_filt[reaction_time_50] - HandX_filt[0],
#             HandY_filt[reaction_time_50] - HandY_filt[0]
#         ])
#     else:
#         participant_vector = np.array([np.nan, np.nan])
    
#     ideal_vector = np.array([
#         xTargetPos[reaction_time_50] - HandX_filt[0],
#         yTargetPos[reaction_time_50] - HandY_filt[0]
#     ])

#     plt.quiver(
#             HandX_filt[0], HandY_filt[0],
#             ideal_vector[0], ideal_vector[1],
#             angles='xy', scale_units='xy', scale=1, color='orange', label='Ideal Vector', headwidth=2, headlength=3
#         )

#     plt.quiver(
#             HandX_filt[0], HandY_filt[0],
#             participant_vector[0], participant_vector[1],
#             angles='xy', scale_units='xy', scale=1, color='black', label='Participant Vector', headwidth=2, headlength=3
#         )

#     mid_x = HandX_filt[0] + (participant_vector[0] + ideal_vector[0]) / 4
#     mid_y = HandY_filt[0] + (participant_vector[1] + ideal_vector[1]) / 4

#     # Annotate the angle and additional information
#     plt.text(mid_x, mid_y, f'IDE = {theta_deg:.2f}°\nCondition: {condition}', fontsize=10, color='black', ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
#     plt.text(mid_x, mid_y - 20, f'End Point Error: {end_point_error:.2f} mm\nPath Length Ratio: {path_length_ratio:.2f}', fontsize=8, color='black', ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
#     plt.text(mid_x, mid_y - 40, f'Accuracy: {"Hit" if accuracy_value == 0 else "Miss"}', fontsize=8, color='black', ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
#     plt.text(mid_x - 10, mid_y - 60, f'Subject Initial Vector: {participant_vector}\nIdeal Vector: {ideal_vector}', fontsize=8, color='black', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
    
#     plt.xlabel('X Position (mm)')
#     plt.ylabel('Y Position (mm)')
#     plt.title('Participant Movement and Ideal Movement Vectors')
#     plt.legend(loc='upper right', fontsize=8, framealpha=0.9)
#     plt.axis('equal')
#     plt.grid(True)
    
#     # Save plot with subject name and index
#     plot_filename = f'{subject}_{i}_{str(uuid.uuid4())}.png'
#     plt.savefig(os.path.join(subject_folder, plot_filename), bbox_inches='tight')
#     plt.close()


def plot_trial_IDE(kinData, HandX_filt, HandY_filt, xTargetPos, yTargetPos, theta_deg, i,subject,CursorX, CursorY,velX_filt, velY_filt ):
    
    completion_time = int(kinData['CT'])
    reaction_time_50 = int(kinData['RT']+50)
    reaction_time_100 = int(kinData['RT']+100)
    target_x = xTargetPos[reaction_time_50]
    target_y = yTargetPos[reaction_time_50]
    target_x_at_ct = xTargetPos[completion_time]

    accuracy_value = kinData['Accuracy']
    end_point_error = kinData['EndPointError']
    path_length_ratio = kinData['PLR']
    condition = kinData['Condtion']
    
    # Create folder for subject if it doesn't exist
    subject_folder = os.path.join(r'C:\Users\LibraryUser\Downloads\Fall2024\BrainAndAction\CP\CP\results\plots_v2', subject)
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)

    
    plt.figure(figsize=(12, 10))
    plt.plot(HandX_filt[0:], HandY_filt[0:], label='Participant Path', color='blue')
    plt.plot(CursorX[0:], CursorY[0:],label='Cursor Path', color = 'deepskyblue' )
    plt.scatter(HandX_filt[0], HandY_filt[0], color='yellow', label='Start Position')
    plt.scatter(HandX_filt[-1], HandY_filt[-1], color='pink', label='End Position')
    plt.scatter(target_x, target_y, color='orange', label="Target Position", zorder=5)
    plt.scatter(target_x_at_ct, target_y, color='darkolivegreen', label="Target Position at CT")
    plt.scatter(HandX_filt[completion_time], HandY_filt[completion_time], color='brown', label='Hand Coordinates at Completion Time')
    plt.scatter(HandX_filt[reaction_time_50], HandY_filt[reaction_time_50], color='olive', label='Point of HandPath at RT50')
    plt.scatter(CursorX[completion_time], CursorY[completion_time],color='chocolate', label= 'Cursor at CT')

    if reaction_time_50 < len(HandX_filt):
        participant_vector = np.array([
            HandX_filt[reaction_time_50] - HandX_filt[0],
            HandY_filt[reaction_time_50] - HandY_filt[0]
        ])
    else:
        participant_vector = np.array([np.nan, np.nan])
    
    ideal_vector = np.array([
        xTargetPos[reaction_time_50] - HandX_filt[0],
        yTargetPos[reaction_time_50] - HandY_filt[0]
    ])

    plt.quiver(
            HandX_filt[0], HandY_filt[0],
            ideal_vector[0], ideal_vector[1],
            angles='xy', scale_units='xy', scale=1, color='orange', label='Ideal Vector', headwidth=2, headlength=3
        )

    plt.quiver(
            HandX_filt[0], HandY_filt[0],
            participant_vector[0], participant_vector[1],
            angles='xy', scale_units='xy', scale=1, color='black', label='Participant Vector', headwidth=2, headlength=3
        )

    # Ensure annotations are within plot bounds
    mid_x = min(max(HandX_filt[0] + (participant_vector[0] + ideal_vector[0]) / 4, plt.xlim()[0]), plt.xlim()[1])
    mid_y = min(max(HandY_filt[0] + (participant_vector[1] + ideal_vector[1]) / 4, plt.ylim()[0]), plt.ylim()[1])

    # Annotate the angle and additional information
    # plt.text(mid_x, mid_y, f'IDE = {theta_deg:.2f}°\nCondition: {condition}', fontsize=10, color='black', ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
    # plt.text(mid_x, mid_y - 30, f'End Point Error: {end_point_error:.2f} mm\nPath Length Ratio: {path_length_ratio:.2f}', fontsize=8, color='black', ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
    # plt.text(mid_x, mid_y - 60, f'Accuracy: {"Hit" if accuracy_value == 0 else "Miss"}', fontsize=8, color='black', ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
    # plt.text(mid_x - 20, mid_y - 90, f'Subject Initial Vector: {participant_vector}\nIdeal Vector: {ideal_vector}', fontsize=8, color='black', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
    
    # plt.text(mid_x, mid_y, f'IDE = {theta_deg:.2f}°\nCondition: {condition}', fontsize=10)
    # plt.text(mid_x, mid_y - 30, f'End Point Error: {end_point_error:.2f} mm\nPath Length Ratio: {path_length_ratio:.2f}', fontsize=8, color='black')
    # plt.text(mid_x, mid_y - 60, f'Accuracy: {"Hit" if accuracy_value == 0 else "Miss"}', fontsize=8, color='black')
    # plt.text(mid_x - 20, mid_y - 90, f'Subject Initial Vector: {participant_vector}\nIdeal Vector: {ideal_vector}', fontsize=8, color='black')

    annotation_x = plt.xlim()[0] + 10
    annotation_y = plt.ylim()[1] - 10
    line_spacing = 20

    plt.text(annotation_x, annotation_y, f'IDE = {theta_deg:.2f}°\nCondition: {condition}', fontsize=10, color='black', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(annotation_x, annotation_y - line_spacing, f'End Point Error: {end_point_error:.2f} mm\nPath Length Ratio: {path_length_ratio:.2f}', fontsize=8, color='black', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(annotation_x, annotation_y - 2 * line_spacing, f'Accuracy: {"Miss" if accuracy_value == 0 else "Hit"}', fontsize=8, color='black', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(annotation_x, annotation_y - 3 * line_spacing, f'Cursor X: {CursorX[completion_time]}\nCursor Y: {CursorY[completion_time]}', fontsize=8, color='black', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(annotation_x, annotation_y - 4 * line_spacing, f'HandX: {HandX_filt[completion_time]}\nHandY: {HandY_filt[completion_time]}', fontsize=8, color='black', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(annotation_x, annotation_y - 5 * line_spacing, f'Target X: {target_x_at_ct}\nTarget Y: {target_y}', fontsize=8, color='blue', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title('Participant Movement and Ideal Movement Vectors')
    plt.legend(loc='upper right', fontsize=8, framealpha=0.9,bbox_to_anchor=(1.15, 1))
    plt.axis('equal')
    plt.grid(True)
    
    plot_filename = f'{subject}_{i}_{str(uuid.uuid4())}.png'
    plt.savefig(os.path.join(subject_folder, plot_filename), bbox_inches='tight')
    # plt.close()
    plt.show()



def plot_trajectories_range(plotsubject, plotday, tstart, tend, all_df, allTrajs):
    """Plot a range of trajectories for a given subject and day."""
    palette = sns.color_palette(["#7fc97f", "#998ec3"])
    fig, ax = plt.subplots()
    
    for trajx in range(tstart, tend):
        subject_df = all_df[(all_df['subject'] == plotsubject) & (all_df['day'] == plotday)]
        traj = allTrajs[plotsubject + plotday][trajx]
        trajinfo = subject_df.iloc[trajx]

        if np.isnan(trajinfo.RT):
            logger.warning('Missing RT for trajectory %d', trajx)
            continue

        ft = int(trajinfo['CT'])
        style = '--' if tstart <= trajx <= tend else '-'
        ax.plot(traj['CursorX'][0:ft], traj['CursorY'][0:ft], style, color=palette[0])
        circle1 = plt.Circle((traj['xTargetPos'][ft], traj['yTargetPos'][ft]), 10, color='r')
        ax.add_patch(circle1)
        ax.axis('equal')
        ax.set(xlim=(-150, 150), ylim=(40, 150))

    plt.savefig(f'{plotsubject}_ExampleTraj_{trajinfo.Condition}_{trajinfo.Duration}_{trajinfo.Affected}.pdf', dpi=100, bbox_inches="tight")



    


def plot_filtered_hand_trajectories(plotsubject, plotday, all_df, allTrajs, numPoints=75, max_cols=4):
    """Plot filtered hand trajectories for a given subject and day."""
    subject_df = all_df[(all_df['subject'] == plotsubject) & (all_df['day'] == plotday) &
                        (all_df['Duration'] == 625) & (all_df['Condition'] == 'Interception')]
    trials_to_plot = list(subject_df.index)

    if not trials_to_plot:
        logger.warning(f'No trials to plot for subject {plotsubject} on {plotday}')
        return

    logger.info(f'Plotting {len(trials_to_plot)} trial(s) for subject {plotsubject} on {plotday}')

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
            logger.error(f'Trajectory data for {plotsubject} on {plotday} trial {trajx} not found: {e}')
            continue

        logger.info(f'Plotting trajectory for trial {trajx}')
        
        # Plot original Hand and Cursor paths
        ax[idx].plot(traj['HandX_filt'], traj['HandY_filt'], label='Hand Path', color='orange')
        ax[idx].plot(traj['CursorX'], traj['CursorY'], label='Cursor Path', color='blue')
        ax[idx].plot(traj['CursorX'][499], traj['CursorY'][499], 'bo', label='Cursor Position at 499')

        # Arc-length normalization using spline
        tck, u = splprep([traj['HandX_filt'], traj['HandY_filt']], s=0, k=3)
        alpha = np.linspace(0, 1, numPoints)
        resampled_x, resampled_y = splev(alpha, tck)

        mean_resampled_x = np.mean(resampled_x)
        if mean_resampled_x >= 0:
            right_side_resampled_x.append(resampled_x)
            right_side_resampled_y.append(resampled_y)
        else:
            left_side_resampled_x.append(resampled_x)
            left_side_resampled_y.append(resampled_y)

        ax[idx].plot(resampled_x, resampled_y, label='Arc-Length Normalized Path', linestyle='--', color='red')
        ax[idx].axis('equal')
        ax[idx].set_xlabel('X Position')
        ax[idx].set_ylabel('Y Position')
        ax[idx].set_title(f'Trial {trajx} for Subject {plotsubject} on {plotday}')

    # Hide empty subplots
    for idx in range(num_trials, num_rows * num_cols):
        fig.delaxes(ax[idx])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05), fontsize='large')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Plot average trajectories for right and left side groups
    plt.figure(figsize=(7, 7))
    for resampled_x, resampled_y in zip(right_side_resampled_x, right_side_resampled_y):
        plt.plot(resampled_x, resampled_y, color='red', alpha=0.3)
    for resampled_x, resampled_y in zip(left_side_resampled_x, left_side_resampled_y):
        plt.plot(resampled_x, resampled_y, color='blue', alpha=0.3)

    if right_side_resampled_x:
        avg_right_x = np.mean(right_side_resampled_x, axis=0)
        avg_right_y = np.mean(right_side_resampled_y, axis=0)
        plt.plot(avg_right_x, avg_right_y, 'r-', label='Average Right Side Trajectory', linewidth=3)

    if left_side_resampled_x:
        avg_left_x = np.mean(left_side_resampled_x, axis=0)
        avg_left_y = np.mean(left_side_resampled_y, axis=0)
        plt.plot(avg_left_x, avg_left_y, 'b-', label='Average Left Side Trajectory', linewidth=3)

    plt.axis('equal')
    plt.legend()
    plt.title(f'Average Trajectories for Subject {plotsubject} on {plotday}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()
    
    logger.info('Finished plotting all trials and the average trajectories')