"""
Configuration for CHEAT-CP Kinematic Analysis
===============================================

This file sets up all the paths and default parameters needed
for the analysis. 

HOW TO USE:
  example:  from cp_analysis.config import DEFAULTS, MATFILES_DIR, RESULTS_DIR
"""

import os

# BASE DIRECTORY — Change this to point to your data folder
BASE_DIR = r"/Users/YourUserName/Path-To-Your-DataFolder"


# DATA PATHS — All paths are relative to BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MATFILES_DIR = os.path.join(DATA_DIR, "matfiles")

# Master Excel file — contains the list of all subjects + metadata
MASTER_FILE = os.path.join(DATA_DIR, "KINARM_Test.xlsx")


# ANALYSIS SETTINGS

# If True, use a time window (RT+50 to RT+100 ms) to find the best
# initial movement projection. If False, only use RT+50 ms.
consider_window_for_intial_plan = True


# OUTPUT DIRECTORIES — Where results get saved
REPORT_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    (
        "IE_plots_window_based"
        if consider_window_for_intial_plan
        else "IE_plots_without_window"
    ),
    "report_results",
)

PVAR_RESULTS_DIR = os.path.join(RESULTS_DIR, "PVAR_RESULTS")

# Location of the processed data CSV (output of the main pipeline)
data_file_location = os.path.join(RESULTS_DIR, "all_processed_trials_final.csv")


# DEFAULT ANALYSIS PARAMETERS

def define_defaults():
    """
    Define the default parameters used throughout the analysis.

    Returns a dictionary with:
        'fs'    : Sampling frequency in Hz (1000 Hz — KINARM records 1000 samples/sec)
        'fc'    : Low-pass filter cutoff in Hz (5 Hz — removes noise above 5 Hz)
        'fdfwd' : Feedforward constant (0.06 — KINARM's built-in cursor prediction)
        'reachorder' : Order of conditions for plotting
        'grouporder' : Order of groups for plotting
        'armorder'   : Order of arms for plotting
    """
    defaults = dict()
    defaults["fs"] = 1e3  # sampling frequency (1000 Hz)
    defaults["fc"] = 5  # low pass cut-off (Hz)
    defaults["fdfwd"] = 0.06  # feedforward estimate (KINARM constant)
    defaults["reachorder"] = ["Reaching", "Interception"]
    defaults["grouporder"] = ["CP", "TDC"]
    defaults["armorder"] = ["More Affected", "Less Affected"]
    return defaults


# Create the defaults dictionary (ready to use immediately)
DEFAULTS = define_defaults()


# REPORT GENERATION SETTINGS

TOTAL_IDS = 88
MAX_DAYS = 5

# Variables included in the main Excel reports
VARLIST = [
    "Accuracy",
    "MT",
    "RT",
    "pathlength",
    "velPeak",
    "EndPointError",
    "IDE",
    "PLR",
]
