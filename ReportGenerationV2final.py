import os
import numpy as np
import pandas as pd


# Define constants
RESULTS_DIR = r'C:\Users\LibraryUser\Downloads\Fall2024\BrainAndAction\CP\CP\results_final_run\reports'  # Ensure this directory is defined
TOTAL_IDS = 88
ALL_IDS = ['cpvib' + str(item).zfill(3) for item in range(1, TOTAL_IDS + 1)]
MAX_DAYS = 5
VARLIST = ['Accuracy','MT','RT','pathlength','velPeak','EndPointError','IDE','PLR']
# Helper functions
def calculate_additional_columns(df):
    df['pathNorm'] = df['pathlength'] / df['straightlength']
    df['xTargetabs'] = np.abs(df['xTargetEnd'])
    return df

def save_grouped_data(df, group_cols, filename):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    all_df_numeric = df[group_cols + list(numeric_cols)].copy()
    all_df_numeric = all_df_numeric.loc[:, ~all_df_numeric.columns.duplicated()]

    grouped_df_means = all_df_numeric.groupby(group_cols).mean().reset_index()
    grouped_df_means.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
    return grouped_df_means

# def pivot_data_to_wide(df, index_cols, pivot_cols, varlist):
#     # wide_df = df.pivot_table(index=index_cols, columns=pivot_cols, values=values_cols)
#     # wide_df.columns = wide_df.columns.to_flat_index()
#     # wide_df = wide_df.reset_index()
#     df_wide = df[['subject', 'visit', 'studyid', 'group', 'day', 'Condition', 'Affected'] + varlist].pivot_table(
#                 index=["subject", "visit", "studyid", "group", "day"],
#                 columns=['Condition', 'Affected'],
#                 values=varlist
#             )

#     df_wide = df_wide.columns.to_flat_index()
#     df_wide = df_wide.reset_index()
#     variable_order = {var: idx for idx, var in enumerate(VARLIST)}
#     ordered_cols = index_cols + [
#         col for col in sorted(df_wide.columns[len(index_cols):], key=lambda x: (variable_order.get(x[0], len(VARLIST)), *x[1:]))
#     ]
#     df_wide = df_wide[ordered_cols]

#     return df_wide

def pivot_data_to_wide(df, index_cols, pivot_cols, varlist):
    # df_wide = df[['subject', 'visit', 'studyid', 'group', 'day', 'Condition', 'Affected'] + varlist].pivot_table(
    #     index=["subject", "visit", "studyid", "group", "day"],
    #     columns=['Condition', 'Affected'],
    #     values=varlist
    # )
    df_wide = df.pivot_table(index=index_cols, columns=pivot_cols, values=varlist)
    df_wide.columns = df_wide.columns.to_flat_index()
    df_wide = df_wide.reset_index()
    variable_order = {var: idx for idx, var in enumerate(varlist)}
    ordered_cols = index_cols + [
        col for col in sorted(df_wide.columns[len(index_cols):], key=lambda x: (variable_order.get(x[0], len(varlist)), *x[1:]))
    ]
    df_wwide = df_wide[ordered_cols]

    return df_wwide


def reindex_with_missing_ids(df_wide, all_ids, is_wide_dur):

    missing = list(set(all_ids) - set(df_wide['subject'].astype(str)))
    if missing:
        # Create a DataFrame for missing IDs with NaN values
        missing_df = pd.DataFrame({'subject': missing})
        df_wide = pd.concat(
            [df_wide, missing_df], ignore_index=True, sort=False)
    df_wide.index.name = 'NMSKL_ID'
    if is_wide_dur:
        df_wide = df_wide.drop(columns=["visit", "studyid", "group", "day"], errors='ignore')
        return df_wide
    df_wide = df_wide.rename(
        columns={'subject': 'KINARM_ID', 'day': 'Visit_day'})
    return df_wide


def save_excel(df, filepath, sheet_name, mode='w'):
    with pd.ExcelWriter(filepath, engine='openpyxl', mode=mode) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

def combine_mean_columns(df, col1, col2, new_col_name):
    df[new_col_name] = (df[col1] + df[col2]) / 2
    return df

def process_and_save_day_data_std(df_stds, df_std_dur, day_num, exceltitle, all_ids):
    current_day = f'Day{day_num}'
    VARLIST = ['Accuracy', 'MT', 'RT', 'pathlength', 'velPeak', 'EndPointError', 'IDE', 'PLR']
    DURATIONS = [(500, 625), (750, 900)]  # Defined pairs of durations to combine

    # Process Day-Wise Wide Format Data for Standard Deviation
    df_day_stds = df_stds[df_stds['day'] == current_day]
    df_day_wide_std = pivot_data_to_wide(df_day_stds, ['subject', 'visit', 'studyid', 'group', 'day'], ['Condition', 'Affected'], VARLIST)
    df_day_wide_std = reindex_with_missing_ids(df_day_wide_std, all_ids,False)
    # df_day_wide_std.rename(columns={'subject': 'KINARM_ID', 'day': 'Visit_day'}, inplace=True)

    # Process Day-Wise Data by Duration for Standard Deviation
    df_day_std_dur = df_std_dur[df_std_dur['day'] == current_day]
    df_day_wide_dur_std = pivot_data_to_wide(df_day_std_dur, ['subject', 'visit', 'studyid', 'group', 'day'], ['Condition', 'Affected', 'Duration'], VARLIST)
    df_day_wide_dur_std = reindex_with_missing_ids(df_day_wide_dur_std, all_ids,True)
    # df_day_wide_dur_std.drop(columns=['subject', 'visit', 'day', 'group'], inplace=True)

    # Combine standard deviations for specific durations (`500_625` and `750_900`) for each variable
    for var in VARLIST:
        for condition in ['Interception', 'Reaching']:
            for affected in ['Less Affected', 'More Affected']:
                for duration_pair in DURATIONS:
                    col1 = (var, condition, affected, duration_pair[0])
                    col2 = (var, condition, affected, duration_pair[1])
                    combined_col_name = (var, condition, affected, f"{duration_pair[0]}_{duration_pair[1]}_combined_std")

                    # Ensure the columns exist before combining
                    if col1 in df_day_wide_dur_std.columns and col2 in df_day_wide_dur_std.columns:
                        # Corrected function call to pass the list of columns
                        df_day_wide_dur_std = combine_std_columns(
                            df_day_wide_dur_std,
                            [col1, col2],
                            combined_col_name
                        )

    # Combine the data and write to Excel
    df_day_combo = pd.concat([df_day_wide_std, df_day_wide_dur_std.drop(columns=['studyid'], errors='ignore')], axis=1, join="inner")

    if day_num == 1:
        save_excel(df_day_combo, exceltitle, f'{current_day}_Master_Formatted_STD', mode='w')
    else:
        save_excel(df_day_combo, exceltitle, f'{current_day}_Master_Formatted_STD', mode='a')


def process_and_save_day_data(df_means, df_meansdur, day_num, exceltitle, all_ids):
    current_day = f'Day{day_num}'
    VARLIST = ['Accuracy', 'MT', 'RT', 'pathlength', 'velPeak', 'EndPointError', 'IDE', 'PLR']
    DURATIONS = [(500, 625), (750, 900)]  
    df_day_means = df_means[df_means['day'] == current_day]
    df_day_wide = pivot_data_to_wide(df_day_means, ['subject', 'visit', 'studyid', 'group', 'day'], ['Condition', 'Affected'], VARLIST)
    df_day_wide = reindex_with_missing_ids(df_day_wide, all_ids,False)
    # df_day_wide.rename(columns={'subject': 'KINARM_ID', 'day': 'Visit_day'}, inplace=True)

    df_day_meansdur = df_meansdur[df_meansdur['day'] == current_day]
    df_day_widedur = pivot_data_to_wide(df_day_meansdur, ['subject', 'visit', 'studyid', 'group', 'day'], ['Condition', 'Affected', 'Duration'], VARLIST)
    df_day_widedur = reindex_with_missing_ids(df_day_widedur, all_ids,True)
    for var in VARLIST:
        for condition in ['Interception', 'Reaching']:
            for affected in ['Less Affected', 'More Affected']:
                for duration_pair in DURATIONS:
                    col1 = (var, condition, affected, duration_pair[0])
                    col2 = (var, condition, affected, duration_pair[1])
                    combined_col_name = (var, condition, affected, f"{duration_pair[0]}_{duration_pair[1]}_combined")

                    # Ensure the columns exist before combining
                    if col1 in df_day_widedur.columns and col2 in df_day_widedur.columns:
                        df_day_widedur[combined_col_name] = (df_day_widedur[col1] + df_day_widedur[col2]) / 2
    
    df_day_combo = pd.concat([df_day_wide, df_day_widedur.drop(columns=['studyid'], errors='ignore')], axis=1, join="inner")
    # df_day_combo = pd.concat([df_day_wide,df_day_widedur],axis=1)
    if day_num == 1:
        save_excel(df_day_combo, exceltitle, f'{current_day}_Master_Formatted', mode='w')
    else:
        save_excel(df_day_combo, exceltitle, f'{current_day}_Master_Formatted', mode='a')




def save_grouped_data_std(df, group_cols, filename):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    all_df_numeric = df[group_cols + list(numeric_cols)].copy()
    all_df_numeric = all_df_numeric.loc[:, ~all_df_numeric.columns.duplicated()]
    grouped_df = all_df_numeric.groupby(group_cols).std().reset_index()
    grouped_df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)  # Flatten MultiIndex columns
    return grouped_df


def combine_std_columns(df, cols, new_col_name):
    combined_variance = df[cols].apply(lambda x: x**2).mean(axis=1)
    df[new_col_name] = np.sqrt(combined_variance)
    return df


# Main processing
def main_std():
    # Load and calculate additional columns
    all_df = pd.read_csv(r"C:\Users\LibraryUser\Downloads\Fall2024\BrainAndAction\CP\CP\results_final_run\all_processed_trials_final.csv")  # Assuming input data is loaded from a CSV
    all_df = calculate_additional_columns(all_df)

    df_stds = save_grouped_data_std(all_df, ['group', 'visit', 'studyid', 'subject', 'day', 'Condition', 'Affected'], 'stds_bysubject.csv')
    df_std_dur = save_grouped_data_std(all_df, ['group', 'visit', 'studyid', 'subject', 'day', 'Condition', 'Affected', 'Duration'], 'stds_bysubjectandduration.csv')

    # Filter and save Day 1 means
    df1_stds = df_stds[df_stds['day'] == 'Day1']
    df1_stds[['subject', 'group', 'day', 'Condition', 'Affected'] + VARLIST].to_csv(os.path.join(RESULTS_DIR, 'Day1_stds_bysubject.csv'), index=False)

    # Convert long data to wide format for Day 1 and save
    df1_wide = pivot_data_to_wide(df1_stds, ['subject', 'studyid', 'group', 'day'], ['Condition', 'Affected'], VARLIST)
    df1_wide = reindex_with_missing_ids(df1_wide, ALL_IDS,False)
    df1_wide.rename(columns={'subject': 'KINARM_ID', 'day': 'Visit_day'}, inplace=True)
    df1_wide.sort_values(['group', 'KINARM_ID'], ascending=True).to_csv(os.path.join(RESULTS_DIR, 'Day1_stds_bysubject_wide.csv'), index=False)

    # Excel Sheet Creation for Multiple Days
    exceltitle = os.path.join(RESULTS_DIR, 'UL_KINARM_Mastersheet_Auto_Format_stds.xlsx')
    exceltitle2 = os.path.join(RESULTS_DIR, 'UL_KINARM_Mastersheet_Long_Format_stds.xlsx')
    for day_num in range(1, MAX_DAYS + 1):
        process_and_save_day_data_std(df_stds, df_std_dur, day_num, exceltitle, ALL_IDS)

    # Save long format Excel file for all days
    save_excel(df_std_dur, exceltitle2, 'AllDays_Master_Formatted_Stds', mode='w')

def main():
    # Load and calculate additional columns
    all_df = pd.read_csv(r"C:\Users\LibraryUser\Downloads\Fall2024\BrainAndAction\CP\CP\results_final_run\all_processed_trials_final.csv")  # Assuming input data is loaded from a CSV
    all_df = calculate_additional_columns(all_df)

    # Save grouped data to CSV files
    df_means = save_grouped_data(all_df, ['group', 'visit', 'studyid', 'subject', 'day', 'Condition', 'Affected'], 'means_bysubject.csv')
    df_meansdur = save_grouped_data(all_df, ['group', 'visit', 'studyid', 'subject', 'day', 'Condition', 'Affected', 'Duration'], 'means_bysubjectandduration.csv')
    
    df1_means = df_means[df_means['day'] == 'Day1']
    df1_means[['subject', 'group', 'day', 'Condition', 'Affected'] + VARLIST].to_csv(os.path.join(RESULTS_DIR, 'Day1_means_bysubject.csv'), index=False)

    
    df1_wide = pivot_data_to_wide(df1_means, ['subject', 'studyid', 'group', 'day'], ['Condition', 'Affected'], VARLIST)
    # df1_wide.sort_values(['group', 'KINARM_ID'], ascending=True).to_csv(os.path.join(RESULTS_DIR, 'Day1_means_bysubject_wide.csv'), index=False)

    # Excel Sheet Creation for Multiple Days
    exceltitle = os.path.join(RESULTS_DIR, 'UL_KINARM_Mastersheet_Auto_Format_means.xlsx')
    exceltitle2 = os.path.join(RESULTS_DIR, 'UL_KINARM_Mastersheet_Long_Format_means.xlsx')

    for day_num in range(1, MAX_DAYS + 1):
        process_and_save_day_data(df_means, df_meansdur, day_num, exceltitle, ALL_IDS)

    # Save long format Excel file for all days
    save_excel(df_meansdur, exceltitle2, 'AllDays_Master_Formatted_Means', mode='w')
    print("Done successfully for means")
    main_std()
    print("Done Successfully for STDS")

if __name__ == "__main__":
    main()
