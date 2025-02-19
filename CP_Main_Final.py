import numpy as np
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.interpolate import splprep, splev
# import CHEATCP_fxns_V3 as cf
# import Initial_Estimate_Changes as cf
import CHEATCP_Final_With_IE as cf

import TrialPlots as cp_plots
import os
from scipy import stats
import ReportGenerator as reports
import json
from Config import BASE_DIR,mdf,MASTER_FILE,MATFILES_DIR,DEFAULTS,RESULTS_DIR

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to a list
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj  # Handle basic data types directly
    else:
        # Return a string representation if the object cannot be serialized
        return str(obj)




all_df, allTrajs = cf.getDataCP(mdf, MATFILES_DIR, DEFAULTS)
print(f'Results Directory is : {RESULTS_DIR}')
all_df.to_csv(os.path.join(RESULTS_DIR,'all_processed_trials_final.csv'), index=False)
print("Successfully loaded trials data")
serializable_allTrajs = convert_to_serializable(allTrajs)

# serializable_dict = {key: value.to_dict(orient='records') for key, value in allTrajs.items()}
output_file = "allTrajs.json"

# Save the serializable dictionary to a JSON file
# with open(output_file, 'w') as f:
#     json.dump(allTrajs, f, indent=4)

print("Successfully saved allTrajs to JSON")



    
def main():
#    cp_plots.plot_singletraj('011', 'Day1', 1, all_df, allTrajs)
#    cp_plots.plot_trajectories_range('011', 'Day1', 90, 100, all_df, allTrajs)
#    cp_plots.plot_filtered_hand_trajectories('011', 'Day1', all_df, allTrajs)
   reports.main() #generates excel for both means and stds


if __name__ == "__main__":
     main()