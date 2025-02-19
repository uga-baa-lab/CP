import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
import CHEATCP_Final_With_IE as cf
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from Config import define_defaults
from Config import BASE_DIR,mdf,MASTER_FILE,MATFILES_DIR,DEFAULTS,RESULTS_DIR,consider_window_for_intial_plan

print(f"Results Directory: {RESULTS_DIR}")

excel_output_path = os.path.join(RESULTS_DIR, 'Pvar_Values_BlockWise_oct14.xlsx')

try:
    mdf = pd.read_excel(open(MASTER_FILE, 'rb'), sheet_name='KINARM_AllVisitsMaster')
    print("Master Excel file loaded successfully.")
except FileNotFoundError:
    print(f"Master file not found at {MASTER_FILE}. Please check the path.")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading the master file: {e}")
    exit(1)

try:
    all_df, allTrajs = cf.getDataCP(mdf, MATFILES_DIR, DEFAULTS)
    print("Successfully loaded trials data.")
except Exception as e:
    print(f"An error occurred while loading data with getDataCP: {e}")
    exit(1)




def plot_trials_and_mean(subject, condition, arm, duration, norm_traj_x, norm_traj_y, mean_traj_x, mean_traj_y, results_dir):
    plt.figure(figsize=(10, 8))
    
    for i in range(norm_traj_x.shape[0]):
        plt.plot(norm_traj_x[i], norm_traj_y[i], alpha=0.3, color='blue')
    
    plt.plot(mean_traj_x, mean_traj_y, color='red', linewidth=2, label='Mean Trajectory')
    
    plt.title(f"Subject: {subject} - Condition: {condition}, Arm: {arm}, Duration: {duration}")
    plt.xlabel("Hand X Position")
    plt.ylabel("Hand Y Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f"{subject}_{condition}_{arm}_{duration}_trials_mean.png"
    plot_path = os.path.join(results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved for Subject: {subject}, Condition: {condition}, Arm: {arm}, Duration: {duration} at {plot_path}")


def generate_pvar(norm_traj_x, norm_traj_y):
    """
    Computing Pvar as the sum of standard deviations across all data points as per vanthiels
    """
    mean_traj_x = np.mean(norm_traj_x, axis=0)
    mean_traj_y = np.mean(norm_traj_y, axis=0)
    
    deviations_x = norm_traj_x - mean_traj_x
    deviations_y = norm_traj_y - mean_traj_y
    
    std_dev_x = np.std(deviations_x, axis=0)
    std_dev_y = np.std(deviations_y, axis=0)
    
    Pvar = np.sum(std_dev_x + std_dev_y)
    
    return Pvar,mean_traj_x, mean_traj_y

all_df = all_df.reset_index(drop=True)
num_points = 500  # Number of points for time normalization

conditions = ['Reaching'] #only reaching trials
durations = all_df['Duration'].unique() #should be 4 durations
affected_type_arms = all_df['Affected'].unique()

pvar_results = []
subjects = all_df['subject'].unique()

for subject in subjects:
    subject_df = all_df[all_df['subject'] == subject]
    days = subject_df['day'].unique()
    
    for day in days:
        day_df = subject_df[subject_df['day'] == day].reset_index(drop=True)
        key = f"{subject}{day}"
        
        if key not in allTrajs:
            print(f"Trajectories for {key} not found. Skipping.")
            continue 
        
        traj_list = allTrajs[key]
        
        if len(traj_list) != len(day_df):
            print(f"Mismatch in number of trajectories for {key}. Expected {len(day_df)}, got {len(traj_list)}. Skipping.")
            continue
        
        day_df['trajData'] = traj_list
        
        for condition in conditions:
            for arm in affected_type_arms:
                for duration in durations:
                    block_df = day_df[
                        (day_df['Condition'] == condition) &
                        (day_df['Affected'] == arm) &
                        (day_df['Duration'] == duration)
                    ]
                    
                    # if len(block_df) < 2:
                    #     print(f"Not enough trials for Subject: {subject}, Day: {day}, "
                    #           f"Condition: {condition}, Arm: {arm}, Duration: {duration}. Skipping.")
                    #     continue 
                    
                    norm_traj_x = []
                    norm_traj_y = []
                    
                    for idx, trial in block_df.iterrows():
                        trajData = trial['trajData']
                        HandX = np.array(trajData.get('HandX_filt', []))
                        HandY = np.array(trajData.get('HandY_filt', []))
                        
                        if len(HandX) == 0 or len(HandY) == 0:
                            print(f"Empty trajectory for trial index {idx}. Skipping.")
                            continue
                        
                        if np.isnan(HandX).any() or np.isnan(HandY).any():
                            print(f"NaN values found in trajectory for trial index {idx}. Skipping.")
                            continue
                        
                        RT = trial['RT']
                        CT = trial['CT']
                        
                        if pd.isna(RT) or pd.isna(CT):
                            print(f"NaN RT or CT for trial index {idx}. Skipping.")
                            continue 
                        
                        RT = int(RT)
                        CT = int(CT)
                        
                        if RT < 0 or CT > len(HandX) or RT >= CT:
                            print(f"Invalid RT ({RT}) or CT ({CT}) for trial index {idx}. Skipping.")
                            continue 
                        # Align the trajectory from movement onset (RT) to movement end (CT)
                        HandX_aligned = HandX[RT:CT]
                        HandY_aligned = HandY[RT:CT]
                        
                        if len(HandX_aligned) < 10:
                            print(f"Short trajectory for trial index {idx}. Skipping.")
                            continue 
                        
                        time = np.linspace(0, 1, len(HandX_aligned))
                        new_time = np.linspace(0, 1, num_points)
                        
                        try:
                            tck_x = splrep(time, HandX_aligned, s=0)
                            tck_y = splrep(time, HandY_aligned, s=0)
                            HandX_norm = splev(new_time, tck_x)
                            HandY_norm = splev(new_time, tck_y)
                            # HandX_norm = np.interp(new_time, time, HandX_aligned)
                            # HandY_norm = np.interp(new_time, time, HandY_aligned)
                        except Exception as e:
                            print(f"Interpolation error for trial index {idx}: {e}. Skipping.")
                            continue  # Skip if interpolation fails
                        
                        norm_traj_x.append(HandX_norm)
                        norm_traj_y.append(HandY_norm)
                    
                    norm_traj_x = np.array(norm_traj_x)  
                    norm_traj_y = np.array(norm_traj_y)
                    
                    if norm_traj_x.shape[0] < 2:
                        print(f"Not enough valid trials for Pvar calculation in Subject: {subject}, Day: {day}, "
                              f"Condition: {condition}, Arm: {arm}, Duration: {duration}. Skipping.")
                        continue
                    
                    Pvar, mean_traj_x, mean_traj_y = generate_pvar(norm_traj_x, norm_traj_y)
                    pvar_results.append({
                        'Subject': subject,
                        'Day': day,
                        'Condition': condition,
                        'Arm': arm,
                        'Duration': duration,
                        'Pvar': Pvar
                    })
                    plot_trials_and_mean(
    subject=subject,
    condition=condition,
    arm=arm,
    duration=duration,
    norm_traj_x=norm_traj_x,
    norm_traj_y=norm_traj_y,
    mean_traj_x=mean_traj_x,
    mean_traj_y=mean_traj_y,
    results_dir=RESULTS_DIR
)

pvar_df = pd.DataFrame(pvar_results)

if pvar_df.empty:
    print("No Pvar results were computed. Please check your data and processing steps.")
    exit(1)
pvar_df['Block'] = pvar_df.apply(lambda row: f"{row['Condition']}_{row['Arm']}_{row['Duration']}", axis=1)

pivot_df = pvar_df.pivot_table(index=['Subject', 'Day'], columns='Block', values='Pvar')
pivot_df = pivot_df.reset_index()
pivot_df.columns.name = None  # Remove the aggregation name
pivot_df.columns = ['Subject', 'Day'] + list(pivot_df.columns[2:])

try:
    pivot_df.to_excel(excel_output_path, index=False, engine='openpyxl')
    print(f"Pvar values successfully saved to {excel_output_path}")
except Exception as e:
    print(f"An error occurred while saving to Excel: {e}")
    exit(1)

print("\nPvar computation and export completed successfully.")
