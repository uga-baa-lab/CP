# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 07:28:58 2024

@author: Santosh
"""

import scipy.io
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import CHEATCP_fxns as cf
import numpy as np
from scipy.interpolate import interp1d
from scipy import stats
import logging

# Setting up the Path for the required directories
BASE_DIR = '/Users/barany/OneDrive - University of Georgia/Research/Projects/CP'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MATFILES_DIR = os.path.join(DATA_DIR, 'matfiles')
MASTER_FILE = os.path.join(RESULTS_DIR, 'KINARMdataset_SubjectSummary_All Visits_OK_12-20-23.xlsx')
DEFAULTS = cf.define_defaults()
print('change gui')

print("Hello world")
print("Hello World by Santosh")

def load_data(master_file):
    print('Starting the process to load data from the excel sheet')
    mdf = pd.read_excel(open(master_file, 'rb'), sheet_name='KINARM_AllVisitsMaster')
    all_df, allTrajs = cf.getDataCP(mdf, MATFILES_DIR, DEFAULTS)
    print('Finished loading the data from the Excel sheet')
    return all_df, allTrajs,hello

def plot_single_trajectory(plotsubject, plotday, trajx, all_df, allTrajs):
    subject_df = all_df.loc[(all_df['subject'] == plotsubject) & (all_df['day'] == plotday)]
    cf.plot_singletraj(plotsubject, plotday, trajx, allTrajs, all_df)

def plot_trajectories_range(plotsubject, plotday, tstart, tend, all_df, allTrajs):
    color = 0
    palette = sns.color_palette(["#7fc97f", "#998ec3"])
    fig, ax = plt.subplots()
    
    for trajx in range(tstart, tend):
        subject_df = all_df.loc[(all_df['subject'] == plotsubject) & (all_df['day'] == plotday)]
        traj = allTrajs[plotsubject + plotday][trajx]
        trajinfo = subject_df.iloc[trajx]
        
        if np.isnan(trajinfo.RT):
            print('Missing RT')
            continue
        
        ft = int(trajinfo['CT'])
        style = '--' if tstart <= trajx <= tend else '-'
        plt.plot(traj['CursorX'][0:ft], traj['CursorY'][0:ft], style, color=palette[color])
        circle1 = plt.Circle((traj['xTargetPos'][ft], traj['yTargetPos'][ft]), 10, color='r')
        ax.add_patch(circle1)
        ax.axis('equal')
        ax.set(xlim=(-150, 150), ylim=(40, 150))

    plt.savefig(f'{plotsubject}ExampleTraj{trajinfo.Condition}{trajinfo.Duration}{trajinfo.Affected}.pdf', dpi=100, bbox_inches="tight")

def plot_filtered_hand_trajectories(plotsubject, plotday, all_df, allTrajs):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    subject_df = all_df.loc[(all_df['subject'] == plotsubject) & (all_df['day'] == plotday) & 
                            (all_df['Duration'] == 625) & (all_df['Condition'] == 'Interception')]
    trials_to_plot = list(subject_df.index)

    if not trials_to_plot:
        logger.warning(f'No trials to plot for subject {plotsubject} on {plotday}')
        return

    logger.info(f'Plotting {len(trials_to_plot)} trial(s) for subject {plotsubject} on {plotday}')
    
    fig, ax = plt.subplots(1, len(trials_to_plot), figsize=(15, 5), sharey=True)
    if len(trials_to_plot) == 1:
        ax = [ax]

    for x, trajx in enumerate(trials_to_plot):
        try:
            traj = allTrajs[plotsubject + plotday][trajx]
        except KeyError as e:
            logger.error(f'Trajectory data for {plotsubject} on {plotday} trial {trajx} not found: {e}')
            continue
        
        logger.info(f'Plotting trajectory for trial {trajx}')
        ax[x].plot(traj['HandX_filt'], traj['HandY_filt'], label='Hand Path')
        ax[x].plot(traj['CursorX'], traj['CursorY'], label='Cursor Path')
        ax[x].plot(traj['CursorX'][499], traj['CursorY'][499], 'bo', label='Cursor Position at 499')

        # Time-normalized trajectories
        timepoints = 101
        xPath = np.linspace(traj['HandX_filt'][0], traj['HandX_filt'][-1], timepoints)
        yPath = np.linspace(traj['HandY_filt'][0], traj['HandY_filt'][-1], timepoints)
        ax[x].plot(xPath, yPath, label='Time-normalized Hand Path')

        # Interpolations
        points = np.array([traj['HandX_filt'], traj['HandY_filt']]).T
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]
        interpolations_methods = ['slinear', 'quadratic', 'cubic']
        alpha = np.linspace(0, 1, 75)
        
        interpolated_points = {}
        for method in interpolations_methods:
            interpolator = interp1d(distance, points, kind=method, axis=0)
            interpolated_points[method] = interpolator(alpha)
        
        plt.figure(figsize=(7, 7))
        for method_name, curve in interpolated_points.items():
            plt.plot(*curve.T, '-', label=method_name)
            logger.info(f'Plotted {method_name} interpolation for trial {trajx}')
        
        plt.axis('equal')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        logger.info(f'Finished plotting for trial {trajx}')


# def plot_filtered_hand_trajectories(plotsubject, plotday, all_df, allTrajs):
#     subject_df = all_df.loc[(all_df['subject'] == plotsubject) & (all_df['day'] == plotday) & 
#                             (all_df['Duration'] == 625) & (all_df['Condition'] == 'Interception')]
#     trials_to_plot = [0]

#     fig, ax = plt.subplots(1, len(trials_to_plot), figsize=(15, 5), sharey=True)
#     if len(trials_to_plot) == 1:
#         ax = [ax]

#     for x, trajx in enumerate(trials_to_plot):
#         traj = allTrajs[plotsubject + plotday][trajx]
#         ax[x].plot(traj['HandX_filt'], traj['HandY_filt'])
#         ax[x].plot(traj['CursorX'], traj['CursorY'])
#         ax[x].plot(traj['CursorX'][499], traj['CursorY'][499], 'bo')
        
#         # Time-normalized trajectories
#         timepoints = 101
#         xPath = np.linspace(traj['HandX_filt'][0], traj['HandX_filt'][-1], timepoints)
#         yPath = np.linspace(traj['HandY_filt'][0], traj['HandY_filt'][-1], timepoints)
#         ax[x].plot(xPath, yPath)
        
#         #plot velocity underneath
    
#         #plt.plot(traj['HandX_filt'][0],traj['HandY_filt'][0],'go')
#         #plt.plot(traj['HandX_filt'][int(thisData.T[15][0])],traj['HandY_filt'][int(thisData.T[15][0])],'ro')
        
#         # Interpolations
#         points = np.array([traj['HandX_filt'], traj['HandY_filt']]).T
#         distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
#         distance = np.insert(distance, 0, 0) / distance[-1]
#         interpolations_methods = ['slinear', 'quadratic', 'cubic']
#         alpha = np.linspace(0, 1, 75)
        
#         interpolated_points = {}
#         for method in interpolations_methods:
#             interpolator = interp1d(distance, points, kind=method, axis=0)
#             interpolated_points[method] = interpolator(alpha)
        
#         plt.figure(figsize=(7, 7))
#         for method_name, curve in interpolated_points.items():
#             plt.plot(*curve.T, '-', label=method_name)
        
#         plt.axis('equal')
#         plt.legend()
#         plt.xlabel('x')
#         plt.ylabel('y')
#         plt.show()

def preprocess_data(all_df):
    all_df['IA_abs'] = np.abs(all_df['IA_50RT'])
    all_df['pathNorm'] = all_df['pathlength'] / all_df['straightlength']
    all_df['xTargetabs'] = np.abs(all_df['xTargetEnd'])
    return all_df


def save_data_to_csv(all_df,varlist):
    results_dir = RESULTS_DIR
    group_cols = ['group', 'visit', 'studyid', 'subject', 'day', 'Condition', 'Affected']

    numeric_cols = all_df.select_dtypes(include=[np.number]).columns
    all_df_numeric = all_df[group_cols + list(numeric_cols)].copy()

    df_means = all_df_numeric.groupby(['group',"visit",'studyid','subject','day','Condition','Affected']).mean().reset_index()
    df_means.to_csv(os.path.join(results_dir,'means_bysubject.csv'))

    df_meansdur = all_df_numeric.groupby(['group',"visit",'studyid','subject','day','Condition','Affected','Duration']).mean().reset_index()
    df_meansdur.to_csv(os.path.join(results_dir,'means_bysubjectandduration.csv'))
    hit_df = all_df_numeric.loc[all_df['Accuracy']==1]
    df1_means = df_means.loc[df_means['day']=='Day1']
    df1_means[['subject','group','day','Condition','Affected']+varlist].to_csv('Day1_means_bysubject.csv')
    df2_means = df_means.loc[df_means['day']=='Day2']

    #long to wide
    df1_wide = df1_means[['subject',"studyid",'group','day','Condition','Affected']+varlist].pivot_table(index=["subject","studyid","group","day"], 
                        columns=['Condition','Affected'],
                        values=varlist)
    #sort by group
    df1_wide.sort_values(['group','subject'], ascending=True).to_csv(os.path.join(results_dir,'Day1_means_bysubject_wide.csv'))
    return df_means,df1_means,df2_means

def prepare_excel_export(df_means, df_meansdur, results_dir, totalids=88, max_days=5):
    allids = ['cpvib' + str(item).zfill(3) for item in range(1, totalids + 1)]
    varlist = ['Accuracy', 'MT', 'RT', 'pathlength', 'velPeak']
    exceltitle = os.path.join(results_dir, 'UL KINARM Mastersheet Auto Format.xlsx')
    exceltitle2 = os.path.join(results_dir, 'UL KINARM Mastersheet Long Format.xlsx')

    for thisday in range(1, max_days + 1):
        df1_means = df_means.loc[df_means['day'] == 'Day' + str(thisday)]
        df1_wide = df1_means[['subject',"visit","studyid",'group','day','Condition','Affected']+varlist].pivot_table(index=["subject","visit","studyid","group","day"], 
                    columns=['Condition','Affected'],
                    values=varlist)
        df1_wide.columns = df1_wide.columns.to_flat_index()
        df1_wide = df1_wide.reset_index(level=["subject", "visit", "group", "day"])
        missing = list(set(allids) - set(df1_wide.index.values))
        df1_wide = df1_wide.reindex(df1_wide.index.union(missing))
        df1_wide.index.name = 'NMSKL_ID'
        df1_wide = df1_wide.rename(columns={'subject': 'KINARM_ID', 'day': 'Visit_day'})

        df1_meansdur = df_meansdur.loc[df_meansdur['day'] == 'Day' + str(thisday)]
        df1_widedur = df1_meansdur[['subject',"visit","studyid",'group','day','Condition','Affected','Duration']+varlist].pivot_table(index=["subject","visit","studyid","group","day"], 
                    columns=['Condition','Affected','Duration'],
                    values=varlist)
        df1_widedur.columns = df1_widedur.columns.to_flat_index()
        df1_widedur = df1_widedur.reset_index(level=["subject", "visit", "group", "day"])
        missing = list(set(allids) - set(df1_widedur.index.values))
        df1_widedur = df1_widedur.reindex(df1_wide.index.union(missing))
        df1_widedur.index.name = 'NMSKL_ID'
        df1_widedur = df1_widedur.drop(columns=['subject', "visit", 'day', 'group'])

        df_combo = pd.concat([df1_wide, df1_widedur], axis=1, join="inner")
        
        if thisday == 1:
            with pd.ExcelWriter(exceltitle) as writer:
                df_combo.to_excel(writer, sheet_name='Day1_Master_Formatted')
        else:
            with pd.ExcelWriter(exceltitle, engine="openpyxl", mode='a') as writer:
                df_combo.to_excel(writer, sheet_name='Day' + str(thisday) + '_Master_Formatted')

    with pd.ExcelWriter(exceltitle2) as writer:
        df_meansdur.to_excel(writer, sheet_name='AllDays_Master_Formatted')


def corrfunc(x, y, **kws):
    if len(x)>10:
        height = .9
    else:
        height = .1
    r, p = stats.pearsonr(x, y)
    print(f'r= {r}, p= {p}')
    # ax = plt.gca()
    # ax.annotate(f"r = {r:.2f}", xy=(.1, .9 if len(x) > 10 else .1), xycoords=ax.transAxes, fontsize=16)

def plot_age_accuracy(all_df,df1_means):
    palette = sns.color_palette(["#7fc97f", "#998ec3"])
    g = sns.lmplot(x='age', y='Accuracy', hue='group', data=df1_means.groupby(['subject', 'group']).mean().reset_index(),
                   ci=68, palette=palette, hue_order=['TDC', 'CP'], legend=False, markers=['*', '+'], scatter_kws={"s": 150})
    g.set(xlim=(4.5, 13))
    g.map(corrfunc, 'age', 'Accuracy')
    plt.savefig('AgeAccuracy.pdf', dpi=300, bbox_inches="tight")
    p = sns.FacetGrid(all_df.loc[all_df['Condition'] == 'Interception'], col='Condition')
    p.map(sns.jointplot, 'xTargetEnd', 'yTargetEnd', kind='kde')

def plot_kde_by_group_condition(dataframe, var='xTargetEnd'):
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(2.5, 3), dpi=100)
    fig.subplots_adjust(hspace=1)
    cps = ["Blues", "Greens"]
    dataframe['ObjectDir'] = dataframe['xTargetEnd'] / np.abs(dataframe['xTargetEnd'])
    palette = sns.color_palette(["#7fc97f", "#998ec3"])

    for axnum, cond in enumerate(['Reaching', 'Interception']):
        for cp, tc in enumerate(['TDC', 'CP']):
            for n, objdir in enumerate([1, -1]):
                sns.kdeplot(dataframe.loc[(dataframe['Condition'] == cond) & (dataframe['ObjectDir'] == objdir) & (dataframe['group'] == tc)][var],
                            shade=True, cbar=True, color=palette[cp], shade_lowest=False, ax=axs[axnum], legend=False)
                axs[1].set(xlim=(-180, 180))

    axs[1].set_xlabel('Initial Direction', fontsize=10)
    fig.text(-0.05, 0.5, 'Density', ha='center', va='center', rotation='vertical', fontsize=10)
    plt.savefig(var + 'histogram.pdf', dpi=100, bbox_inches="tight")

def calculate_slopes(all_df, df1_means):    
    df_meansdur2 = all_df.loc[all_df['xTargetabs'] < 150].groupby(['group', 'studyid', 'subject', 'day', 'Condition', 'Affected']).mean().reset_index()
    df_tmp = df_meansdur2.loc[df_meansdur2['day'] == 'Day1']
    varbydur = np.reshape(list(df_tmp['velPeak']), (int(len(df_tmp) / 4), 4))
    accbydur = np.reshape(list(df_tmp['Accuracy']), (int(len(df_tmp) / 4), 4))
    durbydur = np.reshape(list(df_tmp['Duration'] / 1000), (int(len(df_tmp) / 4), 4))
    
    slopes = [stats.linregress(x, y).slope for x, y in zip(durbydur, varbydur)]
    acc_slopes = [stats.linregress(x, y).slope for x, y in zip(durbydur, accbydur)]
    df1_means['slope'] = slopes
    df1_means['Aslope'] = acc_slopes
    interception=df1_means.loc[(df1_means['Condition']=='Interception') ] #& (df_means['Affected']=='More Affected')
    reaching=df1_means.loc[(df1_means['Condition']=='Reaching') ] 
    tdcp=df1_means.loc[(df1_means['group']=='TDC') & (df1_means['Condition']=='Reaching')]
    
    x='xTargetabs' #'targetDist'
    y='velPeak'
    g = sns.FacetGrid(df_tmp.loc[df_tmp['Condition']=='Reaching'], col='Affected',hue='group', height=5)
    g.map(sns.regplot, x, y,ci=68)
    g.map(corrfunc, x, y)
    #g.set(xlim=(85, 160))

    g = sns.FacetGrid(df1_means, col="Condition",row="Affected",hue='group',legend_out=True,sharey=False)
    g= g.map(sns.scatterplot,'Accuracy','slope')
    plt.savefig('SlopeRegression.pdf', dpi=100, bbox_inches="tight")

    g = sns.FacetGrid(df_tmp, col="Condition",row="Duration",hue='group',legend_out=True,sharey=False)
    g= g.map(sns.scatterplot,'Accuracy','velPeak')
    plt.savefig('SlopeScatterplot.pdf', dpi=100, bbox_inches="tight")

    return df_tmp, slopes, acc_slopes


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

def prepare_test_retest_data(df1_means, df2_means, varlist):
    df1_means_both = df1_means[df1_means['subject'].str.contains('|'.join(df2_means['subject'].unique()))].reset_index()
    for var in varlist:
        df1_means_both[f'{var}_Day2'] = df2_means.reset_index()[var]
    
    varlist2 = [f'{var}_Day2' for var in varlist]
    
    df1_pivot = df1_means_both[['subject', 'group', 'Condition', 'Affected'] + sorted(varlist + varlist2)].pivot_table(index=["subject", "group"], 
                        columns=['Condition', 'Affected'],
                        values=sorted(varlist + varlist2))
    df1_pivot.columns = df1_pivot.columns.swaplevel(0, 1)
    df1_pivot.columns = df1_pivot.columns.swaplevel(1, 2)
    df1_pivot = df1_pivot.sort_index(axis=1)
    df1_pivot.to_csv('testretest_means_bysubject_wide.csv')
    return df1_means_both


def plot_test_retest(df1_means_both, varlist, reachorder, armorder):
    for var in varlist:
        for cond in reachorder:
            for affected in armorder:
                print(f'{cond} - {affected} - {var}')
                testretest = df1_means_both[(df1_means_both['Condition'] == cond) & (df1_means_both['Affected'] == affected)]
                plt.figure()
                sns.lmplot(x=var, y=f'{var}_Day2', data=testretest, scatter_kws={'s': 70})
                plt.plot([-.1, 1.1], [-.1, 1.1], 'r--')
                print(stats.pearsonr(testretest[var], testretest[f'{var}_Day2']))
                plt.close()

    sns.jointplot(df1_means_both['Accuracy'],df1_means_both['Accuracy2'], kind="reg", stat_func=r2)

    df1_gross_mean = df1_means_both.groupby(['subject','Affected']).mean()
    sns.jointplot(df1_gross_mean['Accuracy'],df1_gross_mean['Accuracy2'], kind="reg", stat_func=r2)



def plot_endpoint_distributions(all_df):
    sns.jointplot(data=all_df, x='xTargetEnd', y='yTargetEnd', kind='kde')
    plt.savefig('EndpointDistributions.pdf', dpi=100, bbox_inches="tight")

def calculate_mean_std(df_tmp, varlist):
    df_means = df_tmp.groupby(varlist).mean().reset_index()
    df_stds = df_tmp.groupby(varlist).std().reset_index()
    df_means['targetVar'] = df_stds['targetlength']
    return df_means, df_stds

def plot_subject_comparisons(all_df, varlist):
    df_tmp = all_df[all_df['day'] == 'Day1']
    df_means, df_stds = calculate_mean_std(df_tmp, varlist)
    df_means.to_csv(df_means, 'Day1_allmeans_bysubject.csv')

    sns.set_context("paper", rc={"lines.linewidth": 1.5, "xtick.labelsize": 14, "ytick.labelsize": 14})
    two_color_palette = sns.color_palette(["#7fc97f", "#998ec3"])

    for var in varlist:
        g = sns.FacetGrid(df_tmp, col="Condition", row="Affected", legend_out=True, sharey=False)
        #g = sns.FacetGrid(df_tmp, col="Condition",row="group",legend_out=True)
        #g = g.map(sns.pointplot,"Duration",vartotest,"Affected",order=[500,625,750,900],hue_order=['Less Affected','More Affected'],
        #palette=sns.color_palette("muted"))  #Blue is Non-Affected (0), Orange is Affected (1) (need to verify)
        g.map(sns.pointplot, "Duration", var, "group", order=[500, 625, 750, 900], hue_order=['TDC', 'CP'], palette=two_color_palette, legend=False)
        g.add_legend()
        g.set_xticklabels(rotation=45)
        plt.savefig(f'{var}_handandcondition.pdf')

    for var in varlist:
        g = sns.FacetGrid(df_tmp, col="Condition", row="group", legend_out=True)
        g.map(sns.pointplot, "Duration", var, "Affected", order=[500, 625, 750, 900], hue_order=['Less Affected', 'More Affected'], palette=sns.color_palette("muted"))
        g.add_legend()
        
        
def plot_by_subject_and_day(all_df, df_means, this_subject, varlist):
    submeans = df_means.loc[df_means['subject'] == this_subject]
    df_dur = all_df.groupby(['group', 'subject', 'day', 'Condition', 'Affected', 'TP']).mean().reset_index()
    submeansdur = df_dur.loc[df_dur['subject'] == this_subject]

    for vartotest in varlist:
        # Plotting subject means
        g = sns.FacetGrid(submeans, col="day", legend_out=True)
        g.map(sns.pointplot, 'Condition', vartotest, 'Affected', order=['Reaching', 'Interception'], hue_order=['Less Affected', 'More Affected'], palette=sns.color_palette("muted"))
        plt.savefig(f'{this_subject}_{vartotest}_by_day_condition.pdf')

        # Plotting duration means
        g = sns.FacetGrid(submeansdur, col="day", legend_out=True)
        g.map(sns.pointplot, 'TP', vartotest, 'Affected', order=[2, 1, 3, 4, 5, 4, 7, 8], palette=sns.color_palette("muted"))
        plt.savefig(f'{this_subject}_{vartotest}_by_day_duration.pdf')




def main():
    all_df, allTrajs = load_data(MASTER_FILE)
    
    plot_single_trajectory('011', 'Day1', 1, all_df, allTrajs)
    print(f'Executed and generated the plot of singel trajectory')
    plot_trajectories_range('011', 'Day1', 90, 100, all_df, allTrajs)
    print(f'Executed and generated plot for in progress trials')
    plot_filtered_hand_trajectories('011', 'Day1', all_df, allTrajs)
    print(f'Executing for filtered hand trajectories')
    
    
    varlist = ['age', 'Accuracy', 'RT', 'MT', 'velPeak', 'pathlength', 'CT', 'xPosError', 'RTalt', 'IA_abs']

    all_df = preprocess_data(all_df)
    df_means, df2_means, df1_means = save_data_to_csv(all_df)
    df_means = pd.read_csv(os.path.join(RESULTS_DIR, 'means_bysubject.csv'))
    df_meansdur = pd.read_csv(os.path.join(RESULTS_DIR, 'means_bysubjectandduration.csv'))
    prepare_excel_export(df_means, df_meansdur, RESULTS_DIR)
    plot_age_accuracy(all_df,df1_means)
    plot_kde_by_group_condition(all_df)
    # calculate_slopes(all_df, df1_means)
    # Example of plotting functions usage
    df_tmp, slopes, acc_slopes = calculate_slopes(all_df)
    df1_means['slope'] = slopes
    df1_means['Aslope'] = acc_slopes
    df1_means_both = prepare_test_retest_data(df1_means, df2_means, varlist)
    reachorder = ['Reaching', 'Interception']
    armorder = ['Less Affected', 'More Affected']
    plot_test_retest(df1_means_both, varlist, reachorder, armorder)
    plot_endpoint_distributions(all_df)
    plot_subject_comparisons(all_df,varlist)
    
    this_subject = '007'
    plot_by_subject_and_day(all_df, df_means, this_subject, varlist)
    
if __name__ == "__main__":
    main()
