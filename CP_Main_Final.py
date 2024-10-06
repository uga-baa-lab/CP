import numpy as np
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.interpolate import splprep, splev
import CHEATCP_fxns_V3 as cf
import TrialPlots as cp_plots
import os
from scipy import stats
import copy

BASE_DIR = r'C:\Users\LibraryUser\Downloads\Fall2024/BrainAndAction\CP\CP'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_final_run')
print(f"Results Dir : {RESULTS_DIR}")
MATFILES_DIR = os.path.join(DATA_DIR, 'matfiles')
MASTER_FILE = os.path.join(
    DATA_DIR, 'KINARMdataset_SubjectSummary_All Visits_OK_12-20-23.xlsx')
MASTER_FILE=os.path.join(DATA_DIR,'KINARM_Test.xlsx')
DEFAULTS = cf.define_defaults()
mdf = pd.read_excel(open(MASTER_FILE, 'rb'),
                    sheet_name='KINARM_AllVisitsMaster')

all_df, allTrajs = cf.getDataCP(mdf, MATFILES_DIR, DEFAULTS)
all_df.to_csv(os.path.join(RESULTS_DIR,'all_processed_trials_final.csv'), index=False)
print("Successfully loaded trials data")


def corrfunc(x, y, **kws):
        if len(x) > 10:
            height = .9
        else:
            height = .1
        r, p = stats.pearsonr(x, y)
        print(f'r= {r}, p= {p}')
        

def r2(x, y):
        return stats.pearsonr(x, y)[0] ** 2

def test():
     print("Hello")

def main():
   print("Hello")




if __name__ == "__main__":
     main()