#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:48:28 2022

@author: albert
"""

# Packages
from scipy import signal
import numpy as np
from datetime import datetime
import mne_lib
import os
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# %% 

def match_files(folder):
    # Create a dictionary to store the matched files
    matches = {}

    # Iterate through all files in the folder
    for file in os.listdir(folder):
        # Get the file name without the extension
        name, ext = os.path.splitext(file)

        # If the name is already in the dictionary, add the current file to the list of matches
        if name in matches:
            matches[name].append(file)
        # If the name is not in the dictionary, add it and add the current file as the first match
        else:
            matches[name] = [file]

    # Return the dictionary of matched files
    return matches

# %% 

def add_path_separator(path, file_name):
    path_separator = os.path.sep
    return path + path_separator + file_name

# %% Function that computes power to time signal using periodogram estimation method
def power2time(dat,ss,fs_v,st_light,sl_light):
    
    scores = dat[1][st_light:sl_light] #get scores from data
    dat = dat[0][st_light:sl_light] #get data from the chosen electrode
    idx_ss = np.array(scores) == ss #find indexes of sleep stage in scores
    x = np.transpose(dat[idx_ss]) #extract the sleep stage of interest
    if idx_ss.size == 0:
        P = float("NaN")
        f = float("NaN")
    else:
        f, P = signal.periodogram(x, fs = fs_v, window = np.hanning(len(x)), nfft = len(x), axis=0)
    return f, P

# %%
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


# %% Function that computes start/end index from user defined time of interest
def get_hours(recording, user_hour, user_date):
    user_day0 = datetime.strptime(user_date[0].replace(')', '').replace('(',''),'%m, %d, %Y') #changes user input from GUI to date form
    user_day1 = datetime.strptime(user_date[1].replace(')', '').replace('(',''),'%m, %d, %Y')
    
    if abs((user_day0 - recording[0]).days) <= 2 or abs((user_day1 - recording[1]).days) <= 2:
        if (user_day1 - user_day0).days == 0:                                      #If hours of interest are on same day
            n_hours    = user_hour[1] - user_hour[0]
        else:
            n_hours    = abs((user_day1 - user_day0).days)*24 - (user_hour[0] - user_hour[1])
            
        n_epochs        = ((recording[1] - recording[0]).seconds + 4) / 4          #Compute number of epochs (assumes 4 seconds epochs) 
        epochs_per_hour = n_epochs/( ((recording[1] - recording[0]).seconds + 4)/3600 )    #Epochs per hour
        epochs          = n_hours * epochs_per_hour                                #Compute number of epochs that user wants 
        st_light        = abs(recording[0].hour - user_hour[0])*epochs_per_hour       #Find the begining by comparing user begin time and recording begin time
        sl_light        = st_light + epochs
        return int(st_light), int(min(sl_light,n_epochs))
    else: 
        print("ERROR. Dates provided do not correspond with recorded dates. Automatically setting st_light = 0 and sl_light = 21600")
        st_light = 0; sl_light = 21600;
        return int(st_light), int(sl_light)

#get_hours(id_def_time[0], [9,3], ['(11, 18, 2022)','(11, 19, 2022)']) 




# %%  Function that converts delimiters in text file  
def Convert(string):
    li = list(string.split("\t"))
    return li

# %% Function that find indices of elements in tsv file
def find_element_tsv(element, tsv_file):
    start = 0
    for j in range(len(tsv_file)):
        if tsv_file[j].find(element, 0) == 0:
            start = j
            break
    return start

# %% Function that reads data from all exisiting files within a group

def get_filtered_data(in_ID, in_edf, in_tsv, in_electrode, group_nr, fs, af, bf, cf, folder_path):
    cluster_size = 4; epoch_size = fs*cluster_size                              #cluster and epoch_size
    
    b, a = signal.butter(af/2,[bf/(fs/2),cf/(fs/2)], 'bandpass', analog=False)  #Bandpass filter
    #Initialization
    edf_files = []; tsv_files = []; id_input = []; data2 = []; start = 0; scores = []; id_def_time = []; n = 0; 
    if os.path.exists(folder_path):
        
        print("Using folder path for group " + str(group_nr))
        m = match_files(folder_path); count = 0
        for key in m:
            if len(m[key]) == 2:
                count =+ 1
                for k in range(len(m[key])):
                    if m[key][k].endswith('.edf'): edf_files.append(add_path_separator(folder_path, m[key][k]))
                    if m[key][k].endswith('.tsv'): tsv_files.append(add_path_separator(folder_path, m[key][k]))
                id_input.append(count)
    else:                         
        #Cleaning input such that only files that exist are included
        for i in range(len(in_edf[group_nr][1])): #Loops over number of possible mice len(tf[1][1]) ==> this could also be a user input, i.e. maximum number of mice for all groups
            if os.path.exists(in_edf[group_nr][1][i][0]) and in_edf[group_nr][1][i][0].endswith('.edf'): edf_files.append(in_edf[group_nr][1][i][0])
            if os.path.exists(in_tsv[group_nr][1][i][0]) and in_tsv[group_nr][1][i][0].endswith('.tsv'): tsv_files.append(in_tsv[group_nr][1][i][0]); id_input.append(in_ID[group_nr][1][i][0])
    #Loop through all mice
    if len(range(min(len(edf_files), len(tsv_files)))) > 0:
        for i in range(min(len(edf_files), len(tsv_files))):
            #Read edf file
            data = mne.io.read_raw_edf(edf_files[i]); raw_data = data.get_data()
            #Read tsv file
            raw_string = r"{}".format(tsv_files[i])
            with open(raw_string) as fi: tsv_file = fi.read().splitlines()
            
            #Find indices of information to be retrieved from .tsv
            Date_idx = find_element_tsv('Date',tsv_file)
            Time_start_idx = find_element_tsv('Start',tsv_file)
            Time_end_idx = find_element_tsv('End',tsv_file)
            
            #Read scores from observation after Date index
            for j in range(Date_idx+1,len(tsv_file)):
                if len(Convert(tsv_file[j])) > 4: ## Needs better error checking
                    scores.append(int(float(Convert(tsv_file[j])[4]))) 
            
            #Read time of experiment from beginning of start/end index to be compared with st_light and sl_light
            #print(tsv_file[Time_start_idx])
            Time_start  = datetime.strptime(Convert(tsv_file[Time_start_idx])[2], '%m/%d/%Y %H:%M:%S')
            Time_end    = datetime.strptime(Convert(tsv_file[Time_end_idx])[2], '%m/%d/%Y %H:%M:%S')
            id_def_time.append([Time_start, Time_end]) #id_def_time[#mice][2 (0 = start time, 1 = end time)]
            
            #(in_electrode[group_nr][1][i][0]) Do check if this is provided and numeric, else just electrode 1. Then only filter this electrode to use less computation and storage. Then electrode is not needed to be provided in any other of the backbone functions.
            if in_electrode[group_nr][1][i][0].isnumeric():  electrode= int(in_electrode[group_nr][1][i][0]) - 1
            else: electrode = 0; print("ERROR. Electrode number not numeric for group " + str(group_nr) + " and mouse " + str(i) +". Automatically using electrode = 1.")
            
            EEG = filtfilt(b, a, raw_data[electrode,:]*10E6)
            #EEG1, h = signal.freqs(b, a, raw_data[0,:]); EEG2, h = signal.freqs(b, a, raw_data[1,:]) #apply filter
            #Store data
            data_temp = [id_input[i],[np.reshape(EEG, (int(len(EEG)/epoch_size), epoch_size)),scores]]
            data2.append(data_temp)
            n = min(len(edf_files), len(tsv_files))                                 #indicator for number of proper defined mice
    return data2, n, id_def_time

# %% Function for power across frequencies

def power_freq(data2, n, cf_l, cf_h, ss, st_light, sl_light, fs):
    p_r = []
    cluster_size = 4
    epoch_size = fs*cluster_size
    f_global = np.arange(0,int(fs/2)+fs/epoch_size, fs/epoch_size )

    for i in range(n):
        f, P = power2time(data2[i][1], ss, fs, st_light, sl_light)
        
        st_id       = np.where(f_global == cf_l)[0][0]
        sl_id  	    = np.where(f_global == cf_h)[0][0]
        norm_factor = np.mean(P[st_id:sl_id,:])
        p_norm      = P/norm_factor*100 
        p_r.append(np.mean(p_norm,axis = 1))
    
    traces = p_r
    AvgTrace = np.mean(traces, axis = 0)
    SemTrace =  np.std(traces,axis=0)/np.sqrt(n)
    stdP = AvgTrace + SemTrace
    stdM = AvgTrace - SemTrace 
    
    return f, AvgTrace, stdP, stdM


# %% Power acrosss time

def x_round(x): #Rounds to nearest quarter decimal
    return round(x*4)/4

def power_time(data2, n, cf_l, cf_h, cf_l_norm, cf_h_norm, ss, st_light, sl_light, fs):
    cf_l = x_round(cf_l); cf_h = x_round(cf_h)
    
    pnorm   = []; l_all   = []; t       = []
    cluster_size    = 4
    epoch_size      = fs*cluster_size
    f_global        = np.arange(0,int(fs/2)+fs/epoch_size, fs/epoch_size )
        
    for i in range(n):
        t = range(st_light, sl_light + int(3600/4), int(3600/4))
        f, P = power2time(data2[i][1], ss, fs, st_light, sl_light)
        
        st_id_norm = np.where(f_global == cf_l_norm)[0][0]
        sl_id_norm = np.where(f_global == cf_h_norm)[0][0]
        norm       = np.mean(P[st_id_norm:sl_id_norm,:])
        pnorm.append([])
        
        for j in range(len(t)-1):
            f, P = power2time(data2[i][1], ss, fs, t[j], t[j+1])

            st_id      = np.where(f_global == cf_l)[0][0] 
            sl_id      = np.where(f_global == cf_h)[0][0]
            p1         = np.mean(P[st_id:sl_id,:])
            
            pnorm[i].append(p1/norm*100)

    traces = pnorm
    AvgTrace = np.nanmean(traces, axis = 0)
    SemTrace = np.nanstd(traces,axis=0)/np.sqrt(n)
    stdP = AvgTrace+SemTrace
    stdM = AvgTrace-SemTrace

    return AvgTrace, stdP, stdM 



