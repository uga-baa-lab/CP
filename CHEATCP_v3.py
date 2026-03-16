# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 00:36:58 2020

@author: dab32176

Data analysis for CHEAT CP

This code bypasses all but the essential Matlab processing

NOTES: 
    [RESOLVED] CHEAT016 Day 1is missing 750 ms VaryY Left Hand; need to re-download from KINARM 
    [RESOLVED] Same with CP004-VARYY-625ms-RIGHT Day 2
    [RESOLVED ALL] Same with CHEAT-CP020-VARYY-625ms-RIGHT.zip Day 1, and VARYY-750ms-LEFT.zip and VARYY-900ms-RIGHT.zip
    CHEAT-CP003-STATIC-900ms-LEFT.zip, 
    CHEAT006 Day1 is actually Day2 but first day with both hands figure out what to do)
    
Multiple days:
    002: 3
    003: 2 (3)
    004: 3
    005: 2 (3)
    006: (2) which is Day3
    007: 3
    008: 2
    010: 2 (3)
    017: 2
    019: (2)
"""

import scipy.io
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import CHEATCP_fxns as cf


#Set up paths
#NOTE--> all .mat files formally housed here:
#BASE_DIR = '/Users/barany/OneDrive - University of Georgia/WindowsComputer/Documents/MATLAB/MATLAB Codes UGA'

# BASE_DIR = '/Users/barany/OneDrive - University of Georgia/Research/Projects/CP'
BASE_DIR = 'M:\BrainAndAction\CP\CP\CP'
# mat_files = 'M:\BrainAndAction\CP\CP\CP\data\matfiles'
DATA_DIR = 'M:\BrainAndAction\CP\CP\CP\data'
results_dir = os.path.join(BASE_DIR,'results')
matfiles = os.path.join(DATA_DIR,'matfiles')
defaults = cf.define_defaults()
master = os.path.join(DATA_DIR,
                      'KINARMdataset_SubjectSummary_All Visits_OK_12-20-23.xlsx') #os.path.join(results_dir,'UpperLimb_Mastersheet_OK_12-21-23.xlsx') #'UpperLimb_Mastersheet_OK_11-30-21.xlsx')

#read master sheet
mdf = pd.read_excel(open(master, 'rb'),
              sheet_name='KINARM_AllVisitsMaster')

[all_df,allTrajs] = cf.getDataCP(mdf,matfiles,defaults)


#generate single trial plots for each subject

#the goal: there are 160 trials per subj per day
#10 trials per unique block
#2 x 5 plots (5 trials, each with the x-y position plot with info)
#Information on condition and trial in the title of page

#first step - just the xy plots (plot_singletraj fxn) for 10 trials

# assert 0


import numpy as np
print("Plotting for Sub 11")
plotsubject = '011' #010 (CP) and 011 (TDC) are about the same age
plotday = 'Day1'
trajx = 1
subject_df = all_df.loc[(all_df['subject']==plotsubject) & (all_df['day']==plotday)]
cf.plot_singletraj(plotsubject,plotday,trajx,allTrajs,all_df)
""" 
if plotsubject in TDC:
    color = 0
    tstart = 20 #91-100 for less affected interception 750 ms, 120-130 more affected; 20-30;70-80  for reaching
    tend = 30
    tstart2 = 90
    tend2 = 100
else:
    color = 1
    tstart = 100
    tend = 110 #100-110 for less affected interception 750 ms; 140-150 more affected; 20-30; 50-60 for reaching
    tstart2 = 50
    tend2 = 60
"""
color = 0
tstart = 20 #91-100 for less affected interception 750 ms, 120-130 more affected; 20-30;70-80  for reaching
tend = 30
tstart2 = 90
tend2 = 100

palette = sns.color_palette(["#7fc97f","#998ec3"])
fig, ax = plt.subplots()
for trajx in list(range(tstart2,tend2)): # + list(range(tstart2,tend2)): #range(140,160):
    #cf.plot_singletraj(plotsubject,plotday,trajx,allTrajs,all_df)
    subject_df = all_df.loc[(all_df['subject']==plotsubject) & (all_df['day']==plotday)]
    #subject_df = all_df.loc[(all_df['subject']==plotsubject) & (all_df['day']==plotday) 
    #             & (all_df['Condition']=='Interception') & (all_df['Duration']==750) 
    #             & (all_df['Affected']=='Less Affected')]
    traj = allTrajs[plotsubject+plotday][trajx]
    trajinfo = subject_df.iloc[trajx]
    print(trajinfo.Condition)
    print(trajinfo.Duration)
    print(trajinfo.Affected)
    if np.isnan(trajinfo.RT):
        print('missing RT')
        continue
    
    #plot it
    
    ft = int(trajinfo['CT'])
    if trajx >=tstart2 and trajx <=tend2:
        style = '--'
    else:
        style = '-'
    plt.plot(traj['CursorX'][0:ft],traj['CursorY'][0:ft],'-',color=palette[color])
   # plt.plot(traj['CursorX'][int(trajinfo['RT'])],traj['CursorY'][int(trajinfo['RT'])],'bo')
   # plt.plot(traj['CursorX'][int(trajinfo['RTalt'])],traj['CursorY'][int(trajinfo['RTalt'])],'go')
    #plt.plot(x_new,y_new)
    circle1 = plt.Circle((traj['xTargetPos'][ft],traj['yTargetPos'][ft]), 10, color='r')
    
    #ax.add_patch(circle1)
    ax.axis('equal')
    ax.set(xlim=(-150, 150), ylim=(40, 150))
plt.savefig(plotsubject+'ExampleTraj'+trajinfo.Condition+str(trajinfo.Duration)+trajinfo.Affected+'.pdf',dpi=100, bbox_inches = "tight")
print("")

"""
#plot specific trial(s) IN PROGRESS
plotsubject = '011'
plotday = 'Day1'
subject_df = all_df.loc[(all_df['subject']==plotsubject) & (all_df['day']==plotday)
                        & (all_df['Duration']==625)& (all_df['Condition']=='Interception')]
subject_df = all_df.loc[(all_df['subject']==plotsubject) & (all_df['day']==plotday)]
trials_to_plot = [0]

fig, ax = plt.subplots(1, len(trials_to_plot), figsize=(15, 5), sharey=True)

for x,trajx in enumerate(trials_to_plot):
    traj = allTrajs[plotsubject+plotday][trajx]

    ax[x].plot(traj['HandX_filt'],traj['HandY_filt'])
    ax[x].plot(traj['CursorX'],traj['CursorY'])
    ax[x].plot(traj['CursorX'][499],traj['CursorY'][499],'bo')
    
    #time-normalized trajectories
    import numpy as np
    from scipy import interpolate
    
    
    timepoints = 101 
    xPath = np.linspace(traj['HandX_filt'][0],traj['HandX_filt'][-1],timepoints);
    yPath = np.linspace(traj['HandY_filt'][0],traj['HandY_filt'][-1],timepoints);
    plt.plot(traj['HandX_filt'],traj['HandY_filt'])
    plt.plot(xPath,yPath)
    
    #plot velocity underneath
    
    #plt.plot(traj['HandX_filt'][0],traj['HandY_filt'][0],'go')
    #plt.plot(traj['HandX_filt'][int(thisData.T[15][0])],traj['HandY_filt'][int(thisData.T[15][0])],'ro')
    
    import numpy as np
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    
    # Define some points:
    points = np.array([traj['HandX_filt'],
                       traj['HandY_filt']]).T  # a (nbre_points x nbre_dim) array
    
    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    
    # Interpolation for different methods:
    interpolations_methods = ['slinear', 'quadratic', 'cubic']
    alpha = np.linspace(0, 1, 75)
    
    interpolated_points = {}
    for method in interpolations_methods:
        interpolator =  interp1d(distance, points, kind=method, axis=0)
        interpolated_points[method] = interpolator(alpha)
    
    # Graph:
    plt.figure(figsize=(7,7))
    for method_name, curve in interpolated_points.items():
        plt.plot(*curve.T, '-', label=method_name);
    
   # plt.plot(*points.T, 'ok', label='original points');
    plt.axis('equal'); plt.legend(); plt.xlabel('x'); plt.ylabel('y');
"""

#More calculations
import numpy as np
all_df['IA_abs'] = np.abs(all_df['IA_50RT'])
all_df['pathNorm'] = all_df['pathlength']/all_df['straightlength']
all_df['xTargetabs'] = np.abs(all_df['xTargetEnd'])

#Data cleaning





#Save data to csv
varlist = ['age','Accuracy','RT','MT','velPeak','pathlength','CT','xPosError','RTalt','IA_abs'] #'targetDist','targetlength'

#all_df.to_csv('alltrials_CP.csv') 
df_means = all_df.groupby(['group',"visit",'studyid','subject','day','Condition','Affected']).mean().reset_index()
df_means.to_csv(os.path.join(results_dir,'means_bysubject.csv'))

df_meansdur = all_df.groupby(['group',"visit",'studyid','subject','day','Condition','Affected','Duration']).mean().reset_index()
df_meansdur.to_csv(os.path.join(results_dir,'means_bysubjectandduration.csv'))

hit_df = all_df.loc[all_df['Accuracy']==1]
df1_means = df_means.loc[df_means['day']=='Day1']
df1_means[['subject','group','day','Condition','Affected']+varlist].to_csv('Day1_means_bysubject.csv')
df2_means = df_means.loc[df_means['day']=='Day2']

#long to wide
df1_wide = df1_means[['subject',"studyid",'group','day','Condition','Affected']+varlist].pivot_table(index=["subject","studyid","group","day"], 
                    columns=['Condition','Affected'],
                    values=varlist)
#sort by group
df1_wide.sort_values(['group','subject'], ascending=True).to_csv(os.path.join(results_dir,'Day1_means_bysubject_wide.csv'))



#Owais-style Excel sheet


totalids = 88 #last number of IDs used
allids = ['cpvib'+str(item).zfill(3) for item in range(1,totalids+1)]

#NMSKL_ID, Visit_No, KINARM_ID, Group, Visit_day, [Accuracy, MT, RT, Pathlength, PeakVel,then by duration]
varlist = ['Accuracy','MT','RT','pathlength','velPeak'] #'targetDist','targetlength'
exceltitle = os.path.join(results_dir,'UL KINARM Mastersheet Auto Format.xlsx')
exceltitle2 = os.path.join(results_dir,'UL KINARM Mastersheet Long Format.xlsx')

max_days = 5
for thisday in range(1,max_days+1):
    #for each day
    df1_means = df_means.loc[df_means['day']=='Day'+str(thisday)]
    df1_wide = df1_means[['subject',"visit","studyid",'group','day','Condition','Affected']+varlist].pivot_table(index=["subject","visit","studyid","group","day"], 
                    columns=['Condition','Affected'],
                    values=varlist)
    df1_wide.columns = df1_wide.columns.to_flat_index()
    df1_wide = df1_wide.reset_index(level=["subject","visit","group","day"])
    missing = list(set(allids)-set(df1_wide.index.values))
    df1_wide = df1_wide.reindex(df1_wide.index.union(missing))
   # df1_wide.insert(0, 'Visit_No.', np.nan)
    
    df1_wide.index.name = 'NMSKL_ID'
    df1_wide = df1_wide.rename(columns={'subject': 'KINARM_ID', 'day': 'Visit_day'})
    
    #do the same breaking down by duration
    df1_meansdur = df_meansdur.loc[df_meansdur['day']=='Day'+str(thisday)]
    df1_widedur = df1_meansdur[['subject',"visit","studyid",'group','day','Condition','Affected','Duration']+varlist].pivot_table(index=["subject","visit","studyid","group","day"], 
                    columns=['Condition','Affected','Duration'],
                    values=varlist)
    df1_widedur.columns = df1_widedur.columns.to_flat_index()
    df1_widedur = df1_widedur.reset_index(level=["subject","visit","group","day"])
    missing = list(set(allids)-set(df1_widedur.index.values))
    df1_widedur = df1_widedur.reindex(df1_wide.index.union(missing))
    #df1_widedur.insert(0, 'Visit_No.', np.nan)
    df1_widedur.index.name = 'NMSKL_ID'
    df1_widedur = df1_widedur.drop(columns=['subject',"visit", 'day','group'])
   # df1_widedur = df1_widedur.rename(columns={'subject': 'KINARM_ID', 'day': 'Visit_day'})
    df_combo = pd.concat([df1_wide, df1_widedur], axis=1, join="inner")
    if thisday == 1:
        with pd.ExcelWriter(exceltitle) as writer: 
            df_combo.to_excel(writer, sheet_name='Day1_Master_Formatted')
        #with pd.ExcelWriter(exceltitle2) as writer: 
           # df1_meansdur.to_excel(writer, sheet_name='Day1_Master_Formatted')
    else:
        with pd.ExcelWriter(exceltitle, engine="openpyxl",mode='a') as writer: 
            df_combo.to_excel(writer, sheet_name='Day'+str(thisday)+'_Master_Formatted')
with pd.ExcelWriter(exceltitle2) as writer: 
    df_meansdur.to_excel(writer, sheet_name='AllDays_Master_Formatted')
#with pd.ExcelWriter(exceltitle2, engine="openpyxl",mode='a') as writer: 
#    df_meansdur.to_excel(writer, sheet_name='AllDays'+'_Master_Formatted')   
    




#STUFF FOR POSTER
def corrfunc(x, y, **kws):
    if len(x)>10:
        height = .9
    else:
        height = .1
    r,p = scipy.stats.pearsonr(x, y)
    #ax = plt.gca()
   # ax.annotate("r = {:.2f}".format(r),
   #             xy=(.1, height), xycoords=ax.transAxes,fontsize=16)
    print('r= ',r)
    print('p= ',p)
g=sns.lmplot(x='age',y='Accuracy',hue='group',data=df1_means.groupby(['subject','group']).mean().reset_index(),
             ci=68,palette=palette,hue_order=['TDC','CP'],legend=False,markers=['*','+'],scatter_kws={"s": 150})
g.set(xlim=(4.5,13))
g.map(corrfunc,'age','Accuracy')
plt.savefig('AgeAccuracy.pdf',dpi=300, bbox_inches = "tight")

p = sns.FacetGrid(all_df.loc[all_df['Condition']=='Interception'],col='Condition')
p.map(sns.jointplot,'xTargetEnd', 'yTargetEnd',kind='kde')

testvar='xTargetEnd'
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize = (2.5,3), dpi=100)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
cps = ["Blues","Greens"]
all_df['ObjectDir'] = all_df['xTargetEnd']/np.abs(all_df['xTargetEnd']) 
dataframe = all_df.loc[all_df['Accuracy']==1].dropna(subset=['xTargetEnd'])
for axnum,cond in enumerate(['Reaching','Interception']):
    for cp,tc in enumerate(['TDC','CP']): #Green first, Purple second
        #palette=sns.color_palette(cps[cp])[-2:]
        palette = sns.color_palette(["#7fc97f","#998ec3"])
        for n,objdir in enumerate([1,-1]):
            #print('making kdeplot from all trials')
            sns.kdeplot(dataframe.loc[(dataframe['Condition']==cond) & (dataframe['ObjectDir']==objdir) & (dataframe['group']==tc)][testvar],
                                  shade=True,cbar=True,color=palette[cp], shade_lowest=False,ax=axs[axnum],legend=False)
            axs[1].set(xlim=(-180, 180))    
axs[1].set_xlabel('Initial Direction',fontsize=10)
#d=sns.kdeplot(dvaryright[timepoint+'X'],shade=True,cbar=True, shade_lowest=False,ax=axs[0],label=None)# cmap='Reds
   # d=sns.kdeplot(dstaticright[timepoint+'X'],shade=True,cbar=True, shade_lowest=False,ax=axs[1],label=None)# cmap='Reds
fig.text(-0.05, 0.5, 'Density', ha='center', va='center',rotation='vertical',fontsize=10)


plt.savefig(testvar+'histogram.pdf',dpi=100, bbox_inches = "tight")


#slopes df
df_meansdur2 = all_df.loc[all_df['xTargetabs']<150].groupby(['group','studyid','subject','day','Condition','Affected']).mean().reset_index()
df_tmp = df_meansdur2.loc[df_meansdur2['day']=='Day1']
varbydur = np.reshape(list(df_tmp.velPeak),(int(len(df_tmp)/4),4))
accbydur =np.reshape(list(df_tmp.Accuracy),(int(len(df_tmp)/4),4))
durbydur = np.reshape(list(df_tmp.Duration/1000),(int(len(df_tmp)/4),4))
slopes=[scipy.stats.linregress(x,y).slope for x,y in zip(durbydur,varbydur)]
Accslopes=[scipy.stats.linregress(x,y).slope for x,y in zip(durbydur,accbydur)]
df1_means['slope'] = slopes
df1_means['Aslope'] = Accslopes
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

g = sns.FacetGrid(df_tmp, col="Condition",row="Duration",hue='group',legend_out=True,sharey=False)
g= g.map(sns.scatterplot,'Accuracy','velPeak')
 

#end point distributions

p = sns.jointplot(data=all_df,x='xTargetEnd', y='yTargetEnd',kind='kde')

# Test 1: more affected side child with CP vs. nondominant side TDC static reach
# Test 3: more affected side child with CP vs. nondominant side TDC interception

#Test 5: more affected vs. less affected side child with CP vs. nondominant side vs. dominant side TDC static reach
#for vartotest in varlist:
#     cf.plot_twocolumnbar(vartotest,col='Condition',col_order=reachorder,x='group',hue='Affected',df=df1_means,order=grouporder,
#                       hue_order=armorder)

#Test 2: test-retest reliability (tests one month apart) CP static reach
#Test 4: test-retest reliability (tests one month apart) CP interception

#correlation
#individual 

df1_means_both = (df1_means.loc[df1_means['subject'].str.contains('|'.join(df2_means['subject'].unique()))]).reset_index()
for thisVar in varlist:
    df1_means_both[thisVar+'_Day2'] = df2_means.reset_index()[thisVar]

#sns.relplot(x='Accuracy',y='Accuracy2',hue='subject',data=df1_means_both)
#which two speeds are most predictive? Need to par down task.
varlist2 = [s + '_Day2' for s in varlist]

#save to csv
df1_pivot = df1_means_both[['subject','group','Condition','Affected']+sorted(varlist+varlist2)].pivot_table(index=["subject", "group"], 
                    columns=['Condition','Affected'],
                    values=sorted(varlist+varlist2))

df1_pivot.columns =  df1_pivot.columns.swaplevel(0,1) 
df1_pivot.columns =  df1_pivot.columns.swaplevel(1,2) 
df1_pivot = df1_pivot.sort_index(axis=1)
df1_pivot.to_csv('testretest_means_bysubject_wide.csv')

from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

for thisVar in varlist:
    for cond in reachorder:
        for affected in armorder:
            print(cond)
            print(affected)
            print(thisVar)
            testretest = df1_means_both.loc[(df_means['Condition']==cond) & (df_means['Affected']==affected)]
        
            #sns.jointplot(testretest['Accuracy'],testretest['Accuracy2'], kind="reg", stat_func=r2)
            plt.figure()
            sns.lmplot(x=thisVar,y=thisVar+'_Day2',data=testretest,scatter_kws={'s':70})
            plt.plot([-.1, 1.1], [-.1, 1.1], 'r--')
            print(stats.pearsonr(testretest[thisVar],testretest[thisVar+'_Day2']))
            plt.close()
        

sns.jointplot(df1_means_both['Accuracy'],df1_means_both['Accuracy2'], kind="reg", stat_func=r2)

df1_gross_mean = df1_means_both.groupby(['subject','Affected']).mean()
sns.jointplot(df1_gross_mean['Accuracy'],df1_gross_mean['Accuracy2'], kind="reg", stat_func=r2)



# #interaction (both hands)--CP worse for interception?
# for vartotest in varlist:
#     g = sns.FacetGrid(df1_means,legend_out=True)
#     g = g.map(sns.barplot,'Condition',vartotest,'group',order=['Reaching','Interception'],hue_order=['CP','TDC'],
#                 palette=sns.color_palette("muted"))
#     g = g.map(sns.stripplot,'Condition',vartotest,'group',order=['Reaching','Interception'],hue_order=['CP','TDC'],
#                 palette=sns.color_palette("muted"))
    
#     g = sns.FacetGrid(df2_means,legend_out=True)
#     g = g.map(sns.barplot,'Condition',vartotest,'Affected',order=['Reaching','Interception'],hue_order=['Less Affected','More Affected'],
#               palette=sns.color_palette("muted"))
#     g = g.map(sns.stripplot,'Condition',vartotest,'Affected',order=['Reaching','Interception'],hue_order=['Less Affected','More Affected'],
#               palette=sns.color_palette("muted"))

#plot options


#plot_groupmeans(varlist,df_means)

#plot_byduration(varlist,all_df)

df_tmp = all_df.loc[all_df['day']=='Day1']
df_means = df_tmp.groupby(['group','studyid','subject','day','Condition','Affected','Duration']).mean().reset_index()
df_stds = df_tmp.groupby(['group','studyid','subject','day','Condition','Affected','Duration']).std().reset_index()
df_means['targetVar'] = df_stds['targetlength']
df_means.to_csv('Day1_allmeans_bysubject.csv')
#sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})
sns.set_context("paper", rc={"lines.linewidth": 1.5,"xtick.labelsize":14,"ytick.labelsize":14}) 
two_color_palette = sns.color_palette(["#7fc97f","#998ec3"])
for vartotest in varlist:
    g = sns.FacetGrid(df_tmp, col="Condition",row="Affected",legend_out=True,sharey=False)
    #g = sns.FacetGrid(df_tmp, col="Condition",row="group",legend_out=True)
    #g = g.map(sns.pointplot,"Duration",vartotest,"Affected",order=[500,625,750,900],hue_order=['Less Affected','More Affected'],
    #          palette=sns.color_palette("muted"))  #Blue is Non-Affected (0), Orange is Affected (1) (need to verify)
    g = g.map(sns.pointplot,"Duration",vartotest,"group",order=[500,625,750,900],hue_order=['TDC','CP'],
              palette=two_color_palette,legend=False)  #B
    g.add_legend()
    g.set_xticklabels(rotation=45)
    plt.savefig(vartotest+'handandcondition.pdf')
    
for vartotest in varlist:
    g = sns.FacetGrid(df_tmp, col="Condition",row="group",legend_out=True)
    #g = sns.FacetGrid(df_tmp, col="Condition",row="group",legend_out=True)
    #g = g.map(sns.pointplot,"Duration",vartotest,"Affected",order=[500,625,750,900],hue_order=['Less Affected','More Affected'],
    #          palette=sns.color_palette("muted"))  #Blue is Non-Affected (0), Orange is Affected (1) (need to verify)
    g = g.map(sns.pointplot,"Duration",vartotest,"Affected",order=[500,625,750,900],hue_order=['Less Affected','More Affected'],
              palette=sns.color_palette("muted"))  #B
    g.add_legend()
        

#by subject by day
# thisSubject = '007'
# submeans = df_means.loc[df_means['subject']==thisSubject]
# df_dur = all_df.groupby(['group','subject','day','Condition','Affected','TP']).mean().reset_index()
# submeansdur = df_dur.loc[df_dur['subject']==thisSubject]


# for vartotest in varlist:
    
#     g = sns.FacetGrid(submeans,col="day",legend_out=True)
#     g = g.map(sns.pointplot,'Condition',vartotest,'Affected',order=['Reaching','Interception'],hue_order=['Less Affected','More Affected'],
#               palette=sns.color_palette("muted"))


#     g = sns.FacetGrid(submeansdur,col="day",legend_out=True)
#     g = g.map(sns.pointplot,'TP',vartotest,'Affected',order=[2,1,3,4,5,4,7,8],
#               palette=sns.color_palette("muted"))
