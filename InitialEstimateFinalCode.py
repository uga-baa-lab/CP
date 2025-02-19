import numpy as np
import pandas as pd
import seaborn as sns
import math
import os
import TrialPlots as plots
import scipy.io
from scipy import signal
import CHEATCP_Final_With_IE as cf
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import uuid
from sklearn.linear_model import RANSACRegressor
import Utils as utils
from Config import BASE_DIR,mdf,MASTER_FILE,MATFILES_DIR,DEFAULTS,RESULTS_DIR,consider_window_for_intial_plan

def collect_data_by_arm(subject_trials):
    ''' We're considering only those trials where x_intersect values are less than 1000 - Just hardcoding it! And later anyhow we're removing the outliers by IQR but just for the sake of it we're doing it explicitly'''
    subject_trials = subject_trials[(subject_trials['x_intersect'] <= 1000) & (subject_trials['x_intersect'] >= -1000)]

    data_by_arm = utils.initialize_arms_data_structure_for_regression()
    for idx, kinData in subject_trials.iterrows():
        condition = kinData['Condition']
        arm = kinData['Affected']
        if condition == 'Reaching' and not np.isnan(kinData['x_intersect']) and not np.isnan(kinData['x_target_at_RT']):
            data_by_arm[arm]['x_intersect'].append(kinData['x_intersect'])
            data_by_arm[arm]['x_target'].append(kinData['x_target_at_RT'])
            data_by_arm[arm]['indices'].append(int(idx))  # Ensure idx is an integer
    
    for arm in data_by_arm.keys():
        x = np.array(data_by_arm[arm]['x_intersect'])
        y = np.array(data_by_arm[arm]['x_target'])
        indices = np.array(data_by_arm[arm]['indices'])
        
        # Clean using IQR
        if len(x) > 0 and len(y) > 0:
            x_clean, y_clean, indices_clean, filtered_indexs = utils.clean_data_iqr(x, y, indices)
            data_by_arm[arm]['x_intersect'] = x_clean.tolist()
            data_by_arm[arm]['x_target'] = y_clean.tolist()
            data_by_arm[arm]['indices'] = indices_clean.tolist()
    
    return data_by_arm

def perform_regression(x_clean, y_clean):
    if len(x_clean) >= 2:
        X = x_clean.reshape(-1, 1)
        # print(f'Length of X Dataset when passing it to RANSAC : {len(X)}')
        ransac = RANSACRegressor(LinearRegression(), random_state=42)
        ransac.fit(X, y_clean)
        return ransac, ransac.inlier_mask_
    else:
        return None, None
    

def update_outliers(subject_trials, clean_indices, inlier_mask):
    if inlier_mask is not None:
        outlier_mask = np.logical_not(inlier_mask)
        outlier_indices = clean_indices[outlier_mask]
        subject_trials.loc[outlier_indices, 'Outlier_Trial'] = True


def count_outliers_by_subject_arm_duration(subject_trials):

    arms = subject_trials['Affected'].unique()
    durations = subject_trials['Duration'].unique() if 'Duration' in subject_trials.columns else [500, 625, 750, 900]
    subject_ids = subject_trials['studyid'].unique() if 'studyid' in subject_trials.columns else ['Subject']
    visit_days = subject_trials['visit'].unique() if 'visit' in subject_trials.columns else ['visit']
    # print(f'Visit Days : {visit_days}')
    records = []

    for subject_id in subject_ids:
        for visit_day in visit_days:
            combined_id = f"{subject_id}{visit_day}"
            record = {'Subject_Visit': combined_id}
            total_outliers = 0
            for arm in arms:
                for duration in durations:
                    key = f"{arm}_{duration}"
                    subject_arm_duration_data = subject_trials[
                        (subject_trials['Affected'] == arm) &
                        (subject_trials['studyid'] == subject_id) &
                        (subject_trials['Duration'] == duration) &
                        (subject_trials['visit'] == visit_day)
                    ]
                    outlier_count = subject_arm_duration_data['Outlier_Trial'].sum()
                    total_outliers+=outlier_count
                    record[key] = outlier_count
            record['Total Outlier Count'] = total_outliers
            records.append(record)

    df = pd.DataFrame(records)
    sorted_columns = sorted([col for col in df.columns if col not in ['Subject_Visit']])
    return df[['Subject_Visit'] + sorted_columns]


def perform_subject_level_regression(subject_trials):
    '''
    Make sure that the trials received for subjects are data frames. If it's a list type convert them into Dataframes, It's basically the all the KinData of trials for a particular suject

    Step 1 : Add a new column Outlier_Trial to the dataframe
    Step 2 : Now create a dict with 2 arms - Less Affected and More Affected and then add the relevant data you need to for performing regression
    Step 3 : Clean the data before passing it to the regression - Apply this +=3 stds for cleaning the data
    Step 4 : Figure out the outlier indices and mark the Outlier_Trial flag as true for the coressponding indexes.

    '''
    subject_trials = utils.fix_data_types(subject_trials)
    subject_trials['Outlier_Trial'] = False     # Add 'Outlier_Trial' column to the DataFrame, initialize to False

    print(f'Total Subject trials received for subject : {len(subject_trials)}')    # Initialize data structure to hold data for each arm

    regression_models = {}
    data_by_arm= collect_data_by_arm(subject_trials) # Collect data by arm [less affected and more affected]
    for arm, data in data_by_arm.items():
        x = np.array(data['x_intersect'])
        print(f'Len of x_intersect values after dropping NA records : {len(x)}')
        y = np.array(data['x_target'])
        print(f'Length of x_target records after dropping NA records : {len(y)}')
        clean_indices = np.array(data['indices'], dtype=int)
        #now clean the data by removing the values which are greater than +- 3 stds from the mean
        #basically I need a method to clean the data of X and y- remove values > +=3 stds
        if len(x) >= 2:
            model, inlier_mask = perform_regression(x, y)
            if model:
                regression_models[arm] = {'model': model}
                update_outliers(subject_trials, clean_indices, inlier_mask)
                outlier_summary = count_outliers_by_subject_arm_duration(subject_trials)
            else:
                print(f"Not enough cleaned data points for arm {arm} to perform regression.")
                regression_models[arm] = {'model': None}
        else:
            print(f"Not enough data points for arm {arm} to perform regression.")
            regression_models[arm] = {'model': None}

    return regression_models, data_by_arm, subject_trials, outlier_summary


def plot_regression_points(subject, data_by_arm, regression_models, visit_day,visit_id,study_id, RESULTS_DIR, results_df):
    new_rows = []  
    for arm, data in data_by_arm.items():
        print(f'for arm : {arm} length of data is - plot func: {len(data["x_intersect"])}')

    for arm in data_by_arm.keys():
        # Convert data to numpy arrays
        x_positions = np.array(data_by_arm[arm]['x_intersect'])
        print(f'Length of x_positions in plot regression points : {len(x_positions)}')
        x_targets = np.array(data_by_arm[arm]['x_target'])
        indices = data_by_arm[arm]['indices']

        if len(x_positions) == 0 or len(x_targets) == 0:
            print(f"No valid data for arm {arm} of subject {subject}")
            continue
        r_squared_all = np.nan
        r_squared_inliers = np.nan
        
        # Ensure x_positions and x_targets are 1D arrays
        x_positions = x_positions.flatten()
        x_targets = x_targets.flatten()
        
        # Reshape data for sklearn
        X = x_positions.reshape(-1, 1)
        print(f'Length of reshaped data : {len(X)}')
        y = x_targets
        print(f'Length of y variable : {len(y)}')
        # --- Plot 1: All data points with regression line fit to all data ---
        if len(x_positions) >= 2:
            # Fit Linear Regression model to all data
            model_all = LinearRegression()
            model_all.fit(X, y)

            # Get coefficients
            a_all = model_all.coef_[0]
            b_all = model_all.intercept_

            # Calculate R² score
            y_pred_all = model_all.predict(X)
            r_squared_all = r2_score(y, y_pred_all)

            # Store R² value for "All Data" to the new_rows list
            

            # Generate x values for plotting the regression line
            x_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_fit_all = model_all.predict(x_fit)

            # Plot all data points
            plt.figure(figsize=(8, 6))
            plt.scatter(X, y, color='blue', label='Data Points')

            # Plot regression line
            plt.plot(x_fit, y_fit_all, color='green', label='Regression Line (All Data)')

            # Display regression equation and R²
            equation_text = f'y = {a_all:.2f}x + {b_all:.2f}\n$R^2$ = {r_squared_all:.2f}'
            plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top')

            plt.xlabel('Intersection Point (x_intersect)')
            plt.ylabel('Target Position at RT (x_target_at_RT)')
            plt.title(f'Subject: {subject} - {visit_day}, Arm: {arm} - All Data')
            plt.legend()
            plt.grid(True)

            # Save the plot
            subject_folder = os.path.join(RESULTS_DIR, f'{subject}')
            subject_folder = os.path.join(subject_folder, 'Robust_Reg')
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder, exist_ok=True)  # Create directory if it doesn't exist

            plot_filename = f'{subject}_{visit_day}_{arm}_all_data.png'
            plt.savefig(os.path.join(subject_folder, plot_filename), bbox_inches='tight')
            plt.close()

        else:
            print(f"Not enough data points for regression for arm {arm} of subject {subject}")

        # --- Plot 2: Data excluding outliers (inliers only) with regression line ---
        model_info = regression_models.get(arm)
        if model_info and model_info.get('model'):
            ransac = model_info['model']

            # Get inlier mask
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            # Get coefficients from RANSAC model
            a_inliers = ransac.estimator_.coef_[0]
            b_inliers = ransac.estimator_.intercept_
            
            # Calculate R² score using only inliers
            X_inliers = X[inlier_mask]
            y_inliers = y[inlier_mask]
            y_pred_inliers = ransac.predict(X_inliers)
            r_squared_inliers = r2_score(y_inliers, y_pred_inliers)



            # Generate x values for plotting the regression line
            x_fit_inliers = np.linspace(X_inliers.min(), X_inliers.max(), 100).reshape(-1, 1)
            y_fit_inliers = ransac.predict(x_fit_inliers)

            # Plot inliers and outliers
            plt.figure(figsize=(8, 6))
            plt.scatter(X_inliers, y_inliers, color='blue', label='Inliers')
            if np.any(outlier_mask):
                plt.scatter(X[outlier_mask], y[outlier_mask], color='red', label='Outliers')

            # Plot regression line
            plt.plot(x_fit_inliers, y_fit_inliers, color='green', label='Regression Line (Inliers Only)')

            # Display regression equation and R²
            equation_text = f'y = {a_inliers:.2f}x + {b_inliers:.2f}\n$R^2$ = {r_squared_inliers:.2f}'
            plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top')

            plt.xlabel('Intersection Point (x_intersect)')
            plt.ylabel('Target Position at RT (x_target_at_RT)')
            plt.title(f'Subject: {subject} - {visit_day}, Arm: {arm} - Inliers Only')
            plt.legend()
            plt.grid(True)

            # Save the plot
            plot_filename = f'{subject}_{visit_day}_{arm}_inliers_only.png'
            plt.savefig(os.path.join(subject_folder, plot_filename), bbox_inches='tight')
            plt.close()
        else:
            print(f"Not enough data points for RANSAC regression for arm {arm} of subject {subject}")
        new_rows.append({
            'Subject': f"{study_id}{visit_id}",
            'Visit Day': visit_day,
            'Arm': arm,
            'R^2_Linear': r_squared_all,
            'R^2_Robust': r_squared_inliers
        })
    # Concatenate the new rows to the original DataFrame
    if new_rows:
        new_rows_df = pd.DataFrame(new_rows)
        return new_rows_df

    return results_df


def plot_outlier_trials(subject, outlier_trials, subjectTrajs,all_trials_df, RESULTS_DIR):
    """
    Plots the outlier trials for a subject.

    Parameters:
        subject (str): Identifier for the subject.
        outlier_trials (pd.DataFrame): DataFrame containing outlier trial information.
        subjectTrajs (list): List of trajectory data for all trials.
        RESULTS_DIR (str): Directory to save the plots.
    """
    for idx, trial_info in outlier_trials.iterrows():
        trial_number = trial_info['Trial_Index']
        kinData = all_trials_df.iloc[idx]
        # Retrieve trajectory data
        trajData = subjectTrajs[trial_number]
        HandX_filt = trajData['HandX_filt']
        HandY_filt = trajData['HandY_filt']
        xTargetPos = trajData['xTargetPos']
        yTargetPos = trajData['yTargetPos']
        CursorX = trajData['CursorX']
        CursorY = trajData['CursorY']
        delta_t_used = kinData['Delta_T_Used']
        # Extract necessary kinematic data
        RT = int(kinData['RT']) if not np.isnan(kinData['RT']) else None
        x_intersect = kinData['x_intersect']
        y_target_at_RT = yTargetPos[RT] if RT is not None else np.nan

        # Plot the trial
        plt.figure(figsize=(8, 6))
        plt.title(f"Outlier Trial: Subject {subject}, Trial {trial_number}")

        # Plot hand path
        plt.plot(HandX_filt, HandY_filt, label='Hand Path', color='blue')

        # Plot initial movement vector
        if RT is not None:
            delta_t = 50  # Or the actual delta_t used
            if not np.isnan(delta_t_used):
                delta_t = int(delta_t_used)
                
            RT_plus_delta = min(RT + delta_t, len(HandX_filt) - 1)
            if RT >= len(HandX_filt) - 1:
                print(f"RT index {RT} is too close to the end of the array for reliable plotting.")
                return  # or handle this case accordingly
            # print(f"RT: {RT}, delta_t: {delta_t}, HandX_filt length: {len(HandX_filt)}")

            plt.plot([HandX_filt[RT], HandX_filt[RT_plus_delta]],
                     [HandY_filt[RT], HandY_filt[RT_plus_delta]],
                     label='Initial Movement', color='green', linewidth=2)

            # Extend initial movement vector to intersect with y = y_target_at_RT
            vx = HandX_filt[RT_plus_delta] - HandX_filt[RT]
            vy = HandY_filt[RT_plus_delta] - HandY_filt[RT]
            if vx != 0:
                m = vy / vx
                c = HandY_filt[RT] - m * HandX_filt[RT]
                x_vals = np.array([HandX_filt[RT], x_intersect])
                y_vals = m * x_vals + c
                plt.plot(x_vals, y_vals, linestyle='--', color='green', label='Extended Initial Movement')
                plt.scatter(x_intersect, y_target_at_RT, color='red', label='x_intersect')

        # Plot target path
        plt.plot(xTargetPos, yTargetPos, label='Target Path', color='orange')

        # Plot target position at RT
        if RT is not None:
            x_target_at_RT = xTargetPos[RT]
            plt.scatter(x_target_at_RT, y_target_at_RT, color='purple', label='Target at RT')

        # Add labels and legend
        plt.xlabel('X Position (mm)')
        plt.ylabel('Y Position (mm)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        # Save the plot
        subject_folder = os.path.join(RESULTS_DIR, subject, 'Outlier_Trials')
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder,exist_ok=True)
        plot_filename = f'Outlier_Trial_{subject}_Trial_{trial_number}.png'
        plt.savefig(os.path.join(subject_folder, plot_filename), bbox_inches='tight')
        plt.close()

def calculate_ie_for_interception_trials(subject_trials, regression_models, subject, subjectTrajs,RESULTS_DIR):
    """
    Applies the regression model to Interception trials to compute IE.

    Parameters:
        subject_trials (pd.DataFrame): DataFrame containing trial data.
        regression_models (dict): Dictionary with regression models for each arm.
        subject (str): Identifier for the subject.

    Returns:
        pd.DataFrame: DataFrame containing IE values for Interception trials.
    """
    trial_list = []

    # Iterate over DataFrame rows
    for index, kinData in subject_trials.iterrows():
        condition = kinData['Condition']
        arm = kinData['Affected']
        duration = kinData['Duration']
        x_intersect = kinData['x_intersect']
        x_target_at_RT = kinData['x_target_at_RT']
        IE = np.nan
        x_predicted = np.nan
        is_above_threshold = False

        if condition == 'Interception' and not np.isnan(x_intersect) and not np.isnan(x_target_at_RT):
            model_info = regression_models.get(arm)
            if model_info and model_info.get('model'):
                model = model_info['model']
                try:
                    X_new = np.array([[x_intersect]])
                    x_predicted = model.predict(X_new)[0]
                    IE = x_predicted - x_target_at_RT
                except Exception as e:
                    print(f"Prediction failed for subject {subject}, arm {arm}: {e}")
        subject_trials.at[index, 'IE'] = IE
        if(IE > 600):
            is_above_threshold = True
            subject_trials.at[index, 'is_abnormal_IE'] = True
            print(f'IE value is greater than 1000 for subject {subject} and arm {arm} and trial {index}')

        trial_list.append([index, IE, duration, condition, subject, x_intersect, x_target_at_RT, kinData["Outlier_Trial"], arm,is_above_threshold])

    result_df = pd.DataFrame(trial_list, columns=['Trial_Index', 'IE', 'Duration', 'Condition', 'Subject',
                                                  'x_intersect', 'x_target_at_RT', 'Outlier_Trial', 'Arm','Is_Above_Threshold'])
    csv_filename = f'IE_values_{subject}.csv'
    results_dir = os.path.join(RESULTS_DIR, subject)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_df_save_loc = os.path.join(results_dir, csv_filename)
    result_df.to_csv(result_df_save_loc, index=False)

    # Extract outlier trials
    outlier_trials = result_df[result_df['Outlier_Trial'] == True]

    if not outlier_trials.empty:
        print(f'Plotting {len(outlier_trials)} outlier trials for subject {subject}.')
        # You can implement plot_outlier_trials function to plot these trials
        plot_outlier_trials(subject, outlier_trials, subjectTrajs,subject_trials, RESULTS_DIR)
    else:
        print(f'No outlier trials to plot for subject {subject}.')

    return result_df


def getDataCP(mdf,matfiles,defaults,RESULTS_DIR):
    # initialize dataframe
    all_df = pd.DataFrame()
    outlier_summaries = []
    rsqured_summaries = []
    allTrajs = {}
    # row_name_str = ["cpvib070"]

    all_trials_ie_by_subject = []  # Initialize to collect all IE by subject
    rsquared_results_df = pd.DataFrame(columns=['Subject', 'Visit Day', 'Arm', 'Data Type', 'R^2_Linear','R^2_Robust'])

    for index, row in mdf.iterrows():
        # if row['KINARM ID'] not in row_name_str:
        #     continue
        # el
        if row['KINARM ID'].startswith('CHEAT'):
            subject = row['KINARM ID'][-3:]
        else:
            subject = row['KINARM ID']
        
        print(f'Evaluating the subject: {subject} for visit day : {row["Visit_Day"]}')
        subjectmat = 'CHEAT-CP' + subject + row['Visit_Day'] + '.mat'
        mat = os.path.join(matfiles, subjectmat)

        if not os.path.exists(mat):
            print('Skipping', mat)
            continue

        loadmat = scipy.io.loadmat(mat)
        # if row['Visit_Day'] !='Day1':
        #     # assert 0
        #     continue
        data = loadmat['subjDataMatrix'][0][0]

        allTrials = []
        subjectTrajs = []

        for i in range(len(data)):
            thisData = data[i]
            trajData = cf.getHandTrajectories(thisData, defaults)
            kinData = cf.getHandKinematics(thisData, defaults, i, subject)
            row_values = [
                kinData['Condition'], thisData.T[16][0], thisData.T[11][0],
                thisData.T[13][0], thisData.T[14][0], thisData.T[15][0],
                kinData['RT'], kinData['CT'], kinData['velPeak'], kinData['xPosError'],
                kinData['minDist'], kinData['targetDist'], kinData['handDist'],
                kinData['straightlength'], kinData['pathlength'], kinData['targetlength'],
                kinData['CursorX'], kinData['CursorY'], kinData['IA_RT'], kinData['IA_50RT'],
                kinData['RTalt'], kinData['IA_RTalt'], kinData['maxpathoffset'],
                kinData['meanpathoffset'], kinData['xTargetEnd'], kinData['yTargetEnd'],
                kinData['EndPointError'], kinData['IDE'], kinData['PLR'],
                kinData['isCurveAround'], i, kinData['x_intersect'],kinData['x_target_at_RT'],kinData['Delta_T_Used'],kinData['targetDist_Hit_Interception']
            ]

            allTrials.append(row_values)
            subjectTrajs.append(trajData)


        # Create DataFrame for the current subject
        df = pd.DataFrame(allTrials, columns=[
            'Condition', 'Affected', 'TP', 'Duration', 'Accuracy', 'FeedbackTime',
            'RT', 'CT', 'velPeak', 'xPosError', 'minDist', 'targetDist', 'handDist',
            'straightlength', 'pathlength', 'targetlength', 'cursorX', 'cursorY',
            'IA_RT', 'IA_50RT', 'RTalt', 'IA_RTalt', 'maxpathoffset', 'meanpathoffset',
            'xTargetEnd', 'yTargetEnd', 'EndPointError', 'IDE', 'PLR', 'isCurveAround',
            'trial_number','x_intersect','x_target_at_RT','Delta_T_Used','targetDist_Hit_Interception'
        ])

        # Data cleaning and feature engineering
        df['Affected'] = df['Affected'].map({1: 'More Affected', 0: 'Less Affected'})
        # df['Condition'] = df['Condition'].map({1: 'Reaching', 2: 'Interception'})
        df['Duration'] = df['TP'].map({1: 625, 2: 500, 3: 750, 4: 900, 5: 625, 6: 500, 7: 750, 8: 900})
        df['MT'] = df['CT'] - df['RT']
        df['subject'] = subject
        df['age'] = row['Age at Visit (yr)']
        df['visit'] = row['Visit ID']
        df['day'] = row['Visit_Day']
        df['studyid'] = row['Subject ID']
        df['group'] = 'TDC' if row['Group'] == 0 else 'CP'
        df['pathratio'] = df['pathlength'] / df['targetlength']
        regression_models, data_by_subject_arm, subject_df, subject_outlier_summary = perform_subject_level_regression(df)
        outlier_summaries.append(subject_outlier_summary)
        # print(f'Successfully executed regression')
        # df_cleaned = subject_df[subject_df['Outlier_Trial'] == False]  #removing outlier trials before preparing the final dataframe of all_trials - it includes legit trials for all the subjects
        # Plot regression points
        
        rsquared_results_df = plot_regression_points(subject, data_by_subject_arm, regression_models, row['Visit_Day'],row['Visit ID'], row['Subject ID'],RESULTS_DIR,rsquared_results_df)
        rsqured_summaries.append(rsquared_results_df)
        
       
        # Calculate IE for Interception trials
        subject_wise_IE = calculate_ie_for_interception_trials(subject_df, regression_models, subject,subjectTrajs, RESULTS_DIR)
        all_trials_ie_by_subject.append(subject_wise_IE)

        # Concatenate cleaned subject data into the main DataFrame
        #by this time the subject_df will have the IE values for the subject i.e we're editing in the df object, so nothing to worry
        all_df = pd.concat([all_df, df])
        allTrajs[subject + row['Visit_Day']] = subjectTrajs         # Add cleaned trajectories to allTrajs

        
    if outlier_summaries:
    # Finalize IE DataFrames and save
        output_summaries_file = os.path.join(RESULTS_DIR, 'Subject_Arm_Block_Level_Outliers.csv')
        outlier_summary = pd.concat(outlier_summaries, ignore_index=True)
        outlier_summary.to_csv(output_summaries_file,index=False)

    if rsqured_summaries:
        regression_summary_file = os.path.join(RESULTS_DIR, 'regression_results.csv')
        rsquared_summary = pd.concat(rsqured_summaries, ignore_index= True)
        rsquared_summary.to_csv(regression_summary_file, index = False)

    output_path = os.path.join(RESULTS_DIR, 'IE_CSV_Results')
    os.makedirs(output_path, exist_ok=True)  # Ensure directory exists

    print(f'All Trials Data : {all_trials_ie_by_subject}')
    all_trials_ie_df = pd.concat(all_trials_ie_by_subject, ignore_index=True)
    

    grouped_data = all_trials_ie_df.sort_values(by=['Subject', 'Condition'])
    file_name = os.path.join(output_path, 'All_Trials_IE.csv')
    grouped_data_file_name = os.path.join(output_path, 'Grouped_IE.csv') 
    grouped_data.to_csv(grouped_data_file_name, index=False)
    all_trials_ie_df.to_csv(file_name, index=False)
    return all_df, allTrajs


if consider_window_for_intial_plan:
    RESULTS_DIR = os.path.join(RESULTS_DIR, 'IE_plots_window_based')
else:
    print(f'Running without_window based approach')
    RESULTS_DIR = os.path.join(RESULTS_DIR, 'IE_plots_without_window')

defaults = DEFAULTS
all_df, allTrajs = getDataCP(mdf,MATFILES_DIR,defaults,RESULTS_DIR)
all_df.to_csv(os.path.join(RESULTS_DIR,'all_processed_trials_final.csv'), index=False)

