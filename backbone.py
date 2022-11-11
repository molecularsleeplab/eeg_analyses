#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:48:28 2022

@author: albert
"""

# %% 

def power2time(dat,ss,fs_v,st_light,sl_light,electrode):
    from scipy import signal
    import numpy as np
    
    scores = dat[3]
    idx_ss = np.array(scores) == ss
    x = np.transpose(dat[electrode-1][idx_ss])
    f, P = signal.periodogram(x, fs = fs_v, window = np.hanning(len(x)), nfft =len(x),axis=0)
    return f, P
    # st_light,sl_light unused for now
    
# %%  Function that converts delimiters in text file  

def Convert(string):
    li = list(string.split("\t"))
    return li

# %%
def front2back(in_ID, in_edf, in_tsv, in_electrode, group_nr, ss):
    import mne
    import numpy as np
    #from array import array
    #import csv
    #import glob
    import os

    
    from scipy import signal
    import matplotlib.pyplot as plt
    # %  EEG Filters
    fs = 400
    cluster_size = 4
    b, a = signal.butter(8/2,[0.3/(fs/2),35/(fs/2)], 'bandpass', analog=False)

    # % Filter for EMG signal 
    b1,a1 = signal.butter(8/2,[49/(fs/2),51/(fs/2)],'bandstop', analog=False)      #% Notch filter 
    b2,a2 = signal.butter(8/2,[0.3/(fs/2),130/(fs/2)],'bandpass', analog=False)    #% Bandpass filter 

    epoch_size = fs*cluster_size
        
    # absolute path to search all text files inside a specific folder
    #path1 = "/Users/albert/Library/CloudStorage/OneDrive-UniversityofCopenhagen/EEG TOOLBOX/data/Baseline1 copy"
    edf_files = []
    tsv_files = []
    id_input = []
    
    
    #Clean input
    for i in range(len(in_edf[group_nr][1])): #Loops over number of possible mice len(tf[1][1]) ==> this could also be a user input, i.e. maximum number of mice for all groups
        if os.path.exists(in_edf[group_nr][1][i][0]):
            edf_files.append(in_edf[group_nr][1][i][0])
        if os.path.exists(in_tsv[group_nr][1][i][0]):
            tsv_files.append(in_tsv[group_nr][1][i][0])
        id_input.append(in_ID[group_nr][1][i][0])
            
            #This loop now only considers group 0 ==> needs to loop over all groups
    
    
    data2 = []
    for i in range(min(len(edf_files), len(tsv_files))):
        #Match tsv and edf file    
        #idx_tsv = tsv_files.index(os.path.splitext(edf_files[i])[0]+".tsv")
        
        #Read edf file
        data = mne.io.read_raw_edf(edf_files[i])
        raw_data = data.get_data()
        
        #Read tsv file
        raw_string = r"{}".format(tsv_files[i])
        with open(raw_string) as fi:
            tsv_file = fi.read().splitlines()
            
        #Find starting row of data in tsv file
        start = 0
        for j in range(len(tsv_file)):
            if tsv_file[j].find('Date', 0) == 0:
                start = j + 1
                break
            
        #Reads scores
        scores = []
        for j in range(start,len(tsv_file)):
            if len(Convert(tsv_file[j])) > 4: ## Needs better error checking
                scores.append(int(float(Convert(tsv_file[j])[4]))) 
        
        #Apply filters
        EEG1, h = signal.freqs(b, a, raw_data[0,:]) #apply filter
        EEG2, h = signal.freqs(b, a, raw_data[1,:]) #apply filter
    
        x, h = signal.freqs(b1, a1, raw_data[2,:]) #apply filter
        EMG, h = signal.freqs(b2, a2, x) #apply filter
    
        #Store data
        #mouse_id = os.path.splitext(edf_files[i])[0]
        #idx = mouse_id.index("Baseline1 copy/")
        data_temp = [id_input[i],[np.reshape(EEG1, (int(len(EEG1)/epoch_size), epoch_size)),
                           np.reshape(EEG2, (int(len(EEG2)/epoch_size), epoch_size)),
                           np.reshape(EMG, (int(len(EMG)/epoch_size), epoch_size)), 
                           scores]]
        data2.append(data_temp)
    
    
    
    
    
    #s_channel = [2,2,2] #Electrode option for each mouse
        
    
    cf_l   = 0.5
    cf_h   = 100 
    p_r = []
    n = min(len(edf_files), len(tsv_files))
    
    for i in range(n):
        #electrode = s_channel[i]
        f, P = power2time(data2[i][1], ss, 400, 'NaN', 'NaN', 1)
        
        st_id       = np.where(f == cf_l)[0][0]
        sl_id  	    = np.where(f == cf_h)[0][0]
        norm_factor = np.mean(P[st_id:sl_id,:])
        p_norm      = P/norm_factor*100 
        p_r.append(np.mean(p_norm,axis = 1))
    
    traces = p_r
    AvgTrace = np.mean(traces, axis = 0)
    SemTrace =  np.std(traces,axis=0)/np.sqrt(n)
    stdP = AvgTrace + SemTrace
    stdM = AvgTrace - SemTrace 
    
    
    return f, AvgTrace, stdP, stdM






