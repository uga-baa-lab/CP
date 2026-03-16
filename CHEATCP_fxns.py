# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 23:00:35 2020

@author: dab32176

Functions for CHEATCP
"""

from scipy import signal
import scipy.io
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

def set_seaborn_preference():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 10
    rcstyle = {'axes.linewidth': 1.0, 'axes.edgecolor': 'black','ytick.minor.size': 5.0}
    sns.set(font_scale=1.0,rc={'figure.figsize':(20,10)})
    sns.set_style('ticks', rcstyle)
    sns.set_context("paper", rc={"lines.linewidth": 1,"xtick.labelsize":10,"ytick.labelsize":10})
    
def define_defaults():
    defaults = dict()
    defaults['fs'] = 1e3; #sampling frequency
    defaults['fc'] = 5; #low pass cut-off (Hz)
    defaults['fdfwd'] = 0.06; #feedforward estimate of hand position used by KINARM (constant)
    #Define condition order for plotting
    defaults['reachorder'] = ['Reaching','Interception']
    defaults['grouporder'] = ['CP','TDC']
    defaults['armorder'] = ['More Affected','Less Affected']
    return defaults

def lowPassFilter(data,fc,fs,filter_order = 4):
    """ fc is cut-off frequency of filter
    fs is sampling rate
    """
    w = fc/(fs/2) #Normalize the frequency
    [b,a] = signal.butter(filter_order/2,w,'low')  #divide filter order by 2
    dataLPfiltfilt = signal.filtfilt(b,a,data)    #apply filtfilt to data
    return dataLPfiltfilt

def dist(x1,y1,x2,y2): 
    dist = np.sqrt(np.square(x1-x2)+np.square(y1-y2))
    return dist

def getHandTrajectories(thisData,defaults):
    #init dictionary
    trajData = dict()
    
    trajData['xTargetPos'] = thisData.T[1]
    trajData['yTargetPos'] = thisData.T[2] #constant y postion of the target
    
    # Get non-filtered and filtered hand positions and velocities
    HandX = thisData.T[4]   #X position of hand
    HandY = thisData.T[5]  #Y position of hand
    velX = thisData.T[6]  #X velocity
    velY = thisData.T[7]   #Y velocity
    
    #filtered data, speed and acceleration
    trajData['HandX_filt'] = lowPassFilter(HandX,defaults['fc'],defaults['fs'])
    trajData['HandY_filt'] = lowPassFilter(HandY,defaults['fc'],defaults['fs'])
    trajData['velX_filt'] = lowPassFilter(velX,defaults['fc'],defaults['fs'])
    trajData['velY_filt'] = lowPassFilter(velY,defaults['fc'],defaults['fs'])
    
    trajData['handspeed'] = np.sqrt(trajData['velX_filt']**2 + trajData['velY_filt']**2);
   # accel = np.append(0,np.diff(handspeed/defaults['fs'])) #Acceleration
    
    # Cursor position Cursor position is based on hand position + velocity *feedforward estimate (velocity-dependent!)
    trajData['CursorX'] = HandX + defaults['fdfwd'] * velX
    trajData['CursorY'] = HandY + defaults['fdfwd'] * velY  
    
    return trajData

def getHandKinematics(thisData,defaults):
    """

    Parameters
    ----------
    thisData : raw hand kinematic data
    defaults : default parameter settings

    Returns
    -------
    kinData : processed hand kinematic data

    What else to get:
    Number of velocity peaks --> NOT NOW
    Path length ratio: total distance / total distance if straight line movement (onset to offset end points)

    """
    #init dictionary
    kinData = dict()
    
    feedbackOn = int(thisData.T[15][0])
    xTargetPos = thisData.T[1]
    yTargetPos = thisData.T[2] #constant y postion of the target
    
    kinData['feedbackOn'] = feedbackOn;
    
    # Get non-filtered and filtered hand positions and velocities
    HandX = thisData.T[4]   #X position of hand
    HandY = thisData.T[5]  #Y position of hand
    velX = thisData.T[6]  #X velocity
    velY = thisData.T[7]   #Y velocity
    
    #filtered data, speed and acceleration
    HandX_filt = lowPassFilter(HandX,defaults['fc'],defaults['fs'])
    HandY_filt = lowPassFilter(HandY,defaults['fc'],defaults['fs'])
    velX_filt = lowPassFilter(velX,defaults['fc'],defaults['fs'])
    velY_filt = lowPassFilter(velY,defaults['fc'],defaults['fs'])
    
    handspeed = np.sqrt(velX_filt**2 + velY_filt**2);
    
    kinData['initVel'] = handspeed[0];
    
    # Cursor position Cursor position is based on hand position + velocity *feedforward estimate (velocity-dependent!)
    CursorX = HandX + defaults['fdfwd'] * velX
    CursorY = HandY + defaults['fdfwd'] * velY

    #find peaks
    [peaks,props] = signal.find_peaks(handspeed,height=max(handspeed)/4,distance=150)
    kinData['peaks'] = peaks
    kinData['props'] = props

    if len(props['peak_heights'])>1 and peaks[0] < 100: #first peak not real if before 100 ms
        #print(props['peak_heights'],peaks)
        print('bad first peak')
        peaks = np.delete(peaks,0)
        props['peak_heights'] = np.delete(props['peak_heights'],0)
        kinData['badfirstpeak'] = True
    else:
        kinData['badfirstpeak'] = False


    if len(peaks)>0:
        kinData['velPeak'] = props['peak_heights'][0] #same as handspeed[peaks][0]; taking first real peak, not overall peak velocity
        kinData['velLoc'] = peaks[0]
    
        #move backwards from the first peak to determine RT
        #first time handspeed went below 5%
        findonset  = handspeed[0:peaks[0]] < props['peak_heights'][0]*.05
        onset  = np.where(findonset==True)[0]
        if len(onset) >0:
            kinData['RT'] = onset[-1] +1
        else:
            kinData['RT'] =np.nan
            kinData['velPeak'] = np.nan
            kinData['velLoc'] = np.nan
            kinData['RTexclusion'] = 'could not find onset'
    else:
        print('no peaks found')
        kinData['RT'] = np.nan
        kinData['velPeak'] = np.nan
        kinData['velLoc'] = np.nan
        kinData['RTexclusion'] = 'no peaks'
            
    #more contingencies for RT: connect be <100 or greater than when feedback comes on
    #also the Y pos at RT cannot exceed the Y position of the target
    if kinData['RT'] < 100 or kinData['RT'] > feedbackOn or CursorY[0]>yTargetPos[0]:
        kinData['RT'] = np.nan
        kinData['velPeak'] = np.nan
        kinData['velLoc'] = np.nan
        kinData['RTexclusion'] = 'outlier value'
    else:
        kinData['RTexclusion'] = 'good RT'
        
    #try RT defined just as first time speed exceeded 1 cm/s (Orban de Xivry is 2 cm/s)
    findonset  = handspeed > 100
    onset  = np.where(findonset==True)[0]
    if len(onset) >0:
        kinData['RTalt'] = onset[0] + 1
    else:
        kinData['RTalt'] =np.nan
    
    #Movement time first approximation -- when did cursor center cross y position of the target?
    if not np.isnan(kinData['RT']):
        #5 and 10 are PLACEHOLDERS
        findCT= np.where((CursorY+5)>(yTargetPos[0]-10))[0]
        if len(findCT) > 0:
            if findCT[0] > feedbackOn + 200:
                kinData['CT'] = feedbackOn + 200
                kinData['CTexclusion'] = 'Likely Missed target'
            else:
                kinData['CT'] = findCT[0]
                kinData['CTexclusion'] = 'Crossed Y pos at CT'
        else:
            kinData['CT'] = np.nan 
            kinData['CTexclusion'] = 'Cursor center did not cross target'
    else:
        kinData['CT'] = np.nan
        kinData['CTexclusion'] = 'no RT'
        
    #Angle of Movement initiation at reaction time (sometimes occurs after
    #target hit)
    for v in ['RT','RTalt']:
        if not np.isnan(kinData[v]) and (kinData[v]+50 < len(HandX_filt)): 
            xdiff = HandX_filt[kinData[v]] - HandX_filt[0];
            ydiff = HandY_filt[kinData[v]] - HandY_filt[0];    
            kinData['IA_'+v] = np.arctan2(xdiff,ydiff) * 180/np.pi;
            xdiff = HandX_filt[kinData[v]+50] - HandX_filt[0];
            ydiff = HandY_filt[kinData[v]+50] - HandY_filt[0];    
            kinData['IA_50'+v] = np.arctan2(xdiff,ydiff) * 180/np.pi;
        else:
            #RT does not exist or occurs too late so initial angle cannot be calculated
            kinData['IA_'+v] = np.nan
            kinData['IA_50'+v] = np.nan
            
   
    #minimum distance between target and cursor (from onset to feedback)
    kinData['minDist'] = np.min(dist(CursorX[0:feedbackOn+10],CursorY[0:feedbackOn+10],xTargetPos[0:feedbackOn+10],yTargetPos[0:feedbackOn+10]))
 
    
    if not np.isnan(kinData['CT']) and kinData['RT'] < kinData['CT']:
        #x position error when y position crossed
        kinData['xTargetEnd'] = xTargetPos[kinData['CT']]
        kinData['yTargetEnd'] = yTargetPos[kinData['CT']]
        kinData['xPosError'] = np.abs(CursorX[kinData['CT']] - xTargetPos[kinData['CT']])
        
        #distance from start position to target position at time y position crossed
        kinData['targetDist'] = dist(xTargetPos[0],yTargetPos[0],xTargetPos[kinData['CT']],yTargetPos[kinData['CT']])
        kinData['handDist'] = dist(HandX_filt[0],HandY_filt[0],xTargetPos[kinData['CT']],yTargetPos[kinData['CT']])
        
        #path length
        lengths = np.sqrt(np.sum(np.diff(list(zip(HandX_filt[0:kinData['CT']],HandY_filt[0:kinData['CT']])), axis=0)**2, axis=1))
        kinData['pathlength'] = np.sum(lengths)
        #target path length
        tlengths = np.sqrt(np.sum(np.diff(list(zip(xTargetPos[0:kinData['CT']],yTargetPos[0:kinData['CT']])), axis=0)**2, axis=1))
        kinData['targetlength']  = np.sum(tlengths)
        #straight-line path length
        pathx = [HandX_filt[0], HandX_filt[kinData['CT']]]
        pathy = [HandY_filt[0], HandY_filt[kinData['CT']]]       
        x_new = np.linspace(start=pathx[0], stop=pathx[-1], num=int(kinData['CT']))        
        y_new = np.linspace(start=pathy[0], stop=pathy[-1], num=int(kinData['CT']))
        slength = np.sqrt(np.sum(np.diff(list(zip(x_new,y_new)), axis=0)**2, axis=1))
        kinData['straightlength']  = np.sum(slength)
        endofMove = np.min([kinData['CT'],feedbackOn])
        
        kinData['CursorX']= CursorX[0:endofMove ]
        kinData['CursorY']= CursorY[0:endofMove ]
        
       # timepoints = kinData['CT'] - kinData['RT'] + 1; #i.e., MT
       # xPath = np.linspace(HandX_filt[kinData['RT']],HandX_filt[kinData['CT']],timepoints)
       # yPath = np.linspace(HandY_filt[kinData['RT']],HandY_filt[kinData['CT']],timepoints)
        
        start_end = [HandX_filt[kinData['CT']]-HandX_filt[kinData['RT']],
                     HandY_filt[kinData['CT']]-HandY_filt[kinData['RT']]]
        start_end_distance = np.sqrt(np.sum(np.square(start_end)))
        start_end.append(0)
        
        perp_distance = []
        for m,handpos in enumerate(HandX_filt[kinData['RT']:kinData['CT']]):
            thispointstart = [[HandX_filt[m]-HandX_filt[kinData['RT']]],
                              [HandY_filt[m]-HandY_filt[kinData['RT']]]]
            thispointstart.append([0])    
            
            thispointstart = [HandX_filt[m]-HandX_filt[kinData['RT']],
                              HandY_filt[m]-HandY_filt[kinData['RT']]]
            thispointstart.append(0)               
            
            p = np.divide(np.sqrt(np.square(np.sum(np.cross(start_end,thispointstart)))),
                      np.sqrt(np.sum(np.square(start_end))))
            
            perp_distance.append(p)
        
       
        pathoffset = np.divide(perp_distance,start_end_distance)
        if len(pathoffset) < 1:
            assert 0
        kinData['maxpathoffset'] = np.max(pathoffset)
        kinData['meanpathoffset']=np.mean(pathoffset)
        
        
    else:
        kinData['xPosError'] = np.nan
        kinData['targetDist'] = np.nan
        kinData['handDist'] = np.nan
        kinData['straightlength'] = np.nan
        kinData['pathlength'] = np.nan
        kinData['targetlength'] = np.nan
        kinData['CursorX']= np.nan
        kinData['CursorY']= np.nan
        kinData['maxpathoffset'] = np.nan
        kinData['meanpathoffset'] = np.nan
        kinData['xTargetEnd'] = np.nan
        kinData['yTargetEnd'] = np.nan

    return kinData

def getDataCP(mdf,matfiles,defaults):
    #initialize dataframe
    all_df = pd.DataFrame()
    
    # initialize trajectory data cell
    allTrajs = {}
    
    for index, row in mdf.iterrows():
        if row['KINARM ID'].startswith('CHEAT'):
            subject = row['KINARM ID'][-3:]     
        else:
            subject = row['KINARM ID']
        print(subject)
        subjectmat = 'CHEAT-CP'+subject+row['Visit_Day']+'.mat' 
        
        mat = os.path.join(matfiles,subjectmat) #
                
        if not os.path.exists(mat):
            print('skipping',mat)
            #if row['Visit_Day'] =='Day1':
            #    assert 0
            continue      
        
        loadmat =  scipy.io.loadmat(mat)
    
        data = loadmat['subjDataMatrix'][0][0]
        
        #data[23] is all data for trial 24 
        #data[23].T[4] is all data for the 5th column of that trial
        
        #simple, for each trial, collect (a) condition--12, (b) TP--11 (c) hitormiss-->14 (d) feedback time from 15
        #(e) hitormis and feedback time from 10 SKIP (f) trial duration 13 ; affectedhand 16 (1 if using affected)
        
        allTrials = []
        subjectTrajs = []
        for i in range(len(data)):
            thisData = data[i]
            trajData = getHandTrajectories(thisData,defaults)
            kinData = getHandKinematics(thisData,defaults)
            row_values = [thisData.T[12][0],thisData.T[16][0],thisData.T[11][0],
                          thisData.T[13][0],thisData.T[14][0],thisData.T[15][0],
                          kinData['RT'],kinData['CT'],kinData['velPeak'],
                          kinData['xPosError'],kinData['minDist'],kinData['targetDist'],kinData['handDist'],kinData['straightlength'],
                          kinData['pathlength'],kinData['targetlength'],kinData['CursorX'],kinData['CursorY'],
                          kinData['IA_RT'],kinData['IA_50RT'],kinData['RTalt'],kinData['IA_RTalt'],
                          kinData['maxpathoffset'],kinData['meanpathoffset'],kinData['xTargetEnd'],kinData['yTargetEnd']]
                        
            allTrials.append(row_values)
            subjectTrajs.append(trajData)
            
        df = pd.DataFrame(allTrials, columns=['Condition', 'Affected','TP', 'Duration','Accuracy','FeedbackTime',
                                              'RT','CT','velPeak','xPosError','minDist','targetDist','handDist','straightlength',
                                              'pathlength','targetlength','cursorX','cursorY','IA_RT','IA_50RT',
                                              'RTalt','IA_RTalt','maxpathoffset','meanpathoffset','xTargetEnd','yTargetEnd'])
        #data cleaning
        df['Affected'] = df['Affected'].map({1:'More Affected',0:'Less Affected'})
        df['Condition'] = df['Condition'].map({1:'Reaching',2:'Interception'})
        df['Duration'] = df['TP'].map({1:625,2:500,3:750,4:900,5:625,6:500,7:750,8:900})
        df['MT'] = df['CT'] - df['RT']
        df['subject'] = subject
        df['age'] = row['Age at Visit (yr)']
        df['visit'] = row['Visit ID']
        df['day'] = row['Visit_Day']
        df['studyid'] = row['Subject ID']
        
        if row['Group']==0:
            df['group'] = 'TDC'
        else:
            df['group'] = 'CP'
            
        df['pathratio'] = df['pathlength'] / df['targetlength']
        all_df = pd.concat([all_df,df])
        
        #combine all trajectories
        allTrajs[subject+row['Visit_Day']] = subjectTrajs
        
    return all_df, allTrajs

#plotting functions
def plot_groupmeans(varlist,df):
    for vartotest in varlist:
        g = sns.FacetGrid(df, col="Condition",legend_out=True,col_order = ['Reaching','Interception'])
        g = g.map(sns.pointplot,'group',vartotest,'Affected',order=['CP','TDC'],hue_order=['Less Affected','More Affected'],
                  palette=sns.color_palette("muted"))
        g = g.map(sns.stripplot,'group',vartotest,'Affected',order=['CP','TDC'],hue_order=['Less Affected','More Affected'],
                  palette=sns.color_palette("muted"))

def plot_byduration(varlist,df):
    """Note this pools TDC and CP"""
    for vartotest in varlist:
        g = sns.FacetGrid(df, col="Condition",legend_out=True)
        g = g.map(sns.pointplot,"Duration",vartotest,"Affected",order=[500,625,750,900],hue_order=['Less Affected','More Affected'],
                  palette=sns.color_palette("muted"))  #Blue is Non-Affected (0), Orange is Affected (1) (need to verify)
        g.add_legend()
        
        
def plot_twocolumnbar(vartotest,col,col_order,x,hue,df,order,hue_order):
    g = sns.FacetGrid(df,col=col,col_order = col_order,legend_out=True)
    g = g.map(sns.barplot,x,vartotest,hue,order=order,hue_order=hue_order,
               palette=sns.color_palette("muted"),alpha=.6)
    g = g.map(sns.stripplot,x,vartotest,hue,order=order,hue_order=hue_order,
               palette=sns.color_palette("muted"),split=True)
    g.savefig(vartotest+col+x+'.jpg', format='jpeg', dpi=300)
    
def plot_singletraj(plotsubject,plotday,trajx,allTrajs,all_df):
    subject_df = all_df.loc[(all_df['subject']==plotsubject) & (all_df['day']==plotday)]
    traj = allTrajs[plotsubject+plotday][trajx]
    trajinfo = subject_df.iloc[trajx]
    
    #plot it
    fig, ax = plt.subplots()
    ft = int(trajinfo['FeedbackTime'])
    plt.plot(traj['CursorX'][0:ft],traj['CursorY'][0:ft])
    plt.plot(traj['CursorX'][int(trajinfo['RT'])],traj['CursorY'][int(trajinfo['RT'])],'bo')
    plt.plot(traj['CursorX'][int(trajinfo['RTalt'])],traj['CursorY'][int(trajinfo['RTalt'])],'go')
    circle1 = plt.Circle((traj['xTargetPos'][ft],traj['yTargetPos'][ft]), 10, color='r')
    
    ax.add_patch(circle1)
    ax.axis('equal')
    ax.set(xlim=(-150, 150), ylim=(40, 200))

"""
# publication plot
y = 'Accuracy'
x = 'Duration'
hue = 'Affected'
col = 'Condition'
two_color_palette = sns.color_palette(["#f1a340","#998ec3"])
data = all_df
savename = (y+'CP2.pdf')
                                       
groupcols = ['subject',x]
for param in [hue,col]:
    if param is not None:
        groupcols.append(param)
data = data.groupby(groupcols).mean().reset_index()
background = data.assign(idhue = lambda df: df['subject'] \
                            .astype(str)+df[hue].astype(str))

#define orderings
x_order =list(background[x].unique())
hue_order = list(background["idhue"].unique()) #purple = more affected, orange = 'less affected'
point_hue_order = list(background[hue].unique())
col_order =  ['Reaching','Interception']  #list(background[col].unique())

g = sns.FacetGrid(background,col=col,col_order=col_order,legend_out=False) 

g.map(sns.pointplot,x,y,"idhue",palette=two_color_palette,scale=.7,markers='',
                    order = x_order,hue_order=hue_order)

backgroundartists = []
for ax in g.axes.flat:
    for l in ax.lines + ax.collections:
        l.set_zorder(1)
        l.set_alpha(0.4)
        backgroundartists.append(l)
#ci=95 is "bootstrap 95% CIs"--> could do bootstrap 68% CI as alternative
g.map(sns.pointplot, x, y, 'hue',unit="subject",palette=two_color_palette,alpha=0,scale=1.5,markers='',errwidth=2,capsize=.1,
      order = list(background[x].unique()),hue_order=point_hue_order)

#create own legend
h=2
w=1
g.fig.set_size_inches(h,w)
#    g.set(xlabel='',ylabel=ylabel)  

for ax in g.axes.flat:
    for l in ax.lines + ax.collections:
        if l not in backgroundartists:
            l.set_zorder(2)
            l.set_alpha(1)
g.savefig(savename)
"""

#Plotting trajectories
#    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize = (5,10), dpi=100)
#    axs = axs.flatten()
#    #plt.subplots_adjust(bottom=0.1, right=9, top=0.9)    
#    # Create a continuous norm to map from data points to colors
#
#    trials=[]
#   # trialends=[]
#    for t,thistrial in enumerate(trialconds):
#        trialcond=thistrial
#           # incorrecttry = 1
#       # trialdir = -1 #left to right
#       # trialvel = 'Fast'
#       # trialacc = 1 #Accurate
#        
#        #locate trial options
#        trials = df.loc[(df['Condition']==trialcond) 
#                             & (df['subject']==subject)]
#        trials = trials.sample(frac=1)
#        for index, trial in trials.iterrows():     
#           # trial = trials.loc[trials.index[i]]
#          #  trialend = int(trial['kinEnd'])
#            if trial['Affected']=='More Affected':
#                color = "#998ec3"
#            else:
#                color = "#f1a340" #orange
#          #      if trial['Decision']=='Decision':
#           #         color = "#2c7fb8" #"#0571b0" #blue
#           #     else:
#           #         if np.random.randint(2)==0: #only plot half the trials
#           #             continue
#           #         color = "#7fcdbb" #"#b2abd2"
#           #         
#           # else:
#            #    color = "#ca0020" #red
#    
#            axs[t].plot(trial['cursorX'],trial['cursorY'],linewidth = .7,c=color) #correct go
#                
#    #set up each subplot the same way
#    for ax in axs:
#        ax.set_xlim(-170, 170)
#        ax.set_ylim(0, 250)
#        ax.set(adjustable='box', aspect='equal')
#        ax.set_xlabel('X position (mm)')
#        ax.set_ylabel('Y position (mm)')
#       # rect = patches.Rectangle((-170,140),340,340,linewidth=1,edgecolor='k',facecolor='none')
#        rect = patches.Rectangle((-170,140),340,340,linewidth=1,edgecolor='k',facecolor='none')
#
#        lcbar = LineCollection([[(-50,50),(50,50)]], colors=np.array([(0, 0, 0, 1)]), linewidths=1.5)
#       # ax.add_patch(rect)
#       # ax.add_collection(lcbar)
#    plt.show()
#    fig.tight_layout(pad=0)
#    fig.savefig('CP_trajectories.pdf')
#    assert 0