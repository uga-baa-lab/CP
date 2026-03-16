
import os
import pandas as pd


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


BASE_DIR = r'/Users/santoshkp/BAA/CP-Owais/'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_macos')
MATFILES_DIR = os.path.join(DATA_DIR, 'matfiles')
MASTER_FILE = os.path.join(
    DATA_DIR, 'KINARMdataset_SubjectSummary_All Visits_OK_12-20-23.xlsx')
MASTER_FILE=os.path.join(DATA_DIR,'KINARM_Test.xlsx')
DEFAULTS = define_defaults()
mdf = pd.read_excel(open(MASTER_FILE, 'rb'),
                    sheet_name='KINARM_AllVisitsMaster')
data_file_location = r"/Users/santoshkp/BAA/CP-Owais/results_macos/IE_plots_window_based/all_processed_trials_final.csv"

consider_window_for_intial_plan = False
REPORT_RESULTS_DIR = os.path.join(
    RESULTS_DIR, 
    'IE_plots_window_based' if consider_window_for_intial_plan else 'IE_plots_without_window',
    'report_results'
)
