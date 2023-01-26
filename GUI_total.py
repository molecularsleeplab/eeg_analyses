#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:42:38 2023

@author: albert
"""

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
import mne
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



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:43:55 2022

@author: albert
"""


import PySimpleGUI as sg
import matplotlib.pyplot as plt
from random import seed
from random import randint
import matplotlib.transforms as mtransforms
from backbone import *

# ----- Design options -------
sg.theme("Reddit")
font = ('Any', 12)
sg.set_options(font=font)



# First the window layout in 2 columns
input_text_column = [
    [sg.T('Input settings',font=('Any 15'))],
    [sg.HSeparator(pad = (0,0))],
    #[sg.T('Number of groups:', font=('Any 12'))],
     [sg.T('Max. number of mice per group:', font=('Any 12'))],
     [sg.T('Sleep stage:', font=('Any 12'))],
    [sg.T('Sample frequency:', font=('Any 12'))],
    [sg.T('Butterworth parameters (N,Wn1,Wn2):', font=('Any 12'))],
    [sg.T('Example:', font = ('Any', 9,'italic'))],
    [sg.T('Limit hours of interest:', font=('Any 12'))],
    [sg.T('', font=('Any 12'))],
    [sg.T('', font=('Any 12'))],
    [sg.Button('Find files', font=('Any 12'),key='-find_files-'), 
     sg.T('Files have been saved',key = "-saved_prompt-",font=("Any", 12, "italic"), visible = False)],
]

input_values_column = [
    [sg.T('',font=('Any 17'))],
    #[sg.HSeparator(pad = (0,0))],
     #[sg.Spin(values=[i for i in range(1, 50)], key="-n_groups-", initial_value=2, size=(2, 1), font=('Any 12'))],
     [sg.Spin(values=[i for i in range(1, 1000)], key="-n_mice-", initial_value=10, size=(2, 1), font=('Any 12'))],
    [sg.Input('1', key="-sleep_stage-", size=(3,1), font=('Any 12'))],
    [sg.Input('400', key="-fs-", size=(3,1), font=('Any 12'))],
    [sg.Input('4', key="-filter_N-", size=(3,1), font=('Any 12')),
     sg.Input('0.3', key="-filter_Wn1-", size=(3,1), font=('Any 12')), 
     sg.Input('30', key="-filter_Wn2-", size=(3,1), font=('Any 12'))],
    [sg.T("a,b = signal.butter(N/2,[Wn1/(fs/2),Wn2/(fs/2)])",  font = ('Any', 9,'italic'))],
    [sg.T('(Time, Date):', font = ('Any 12'), key = "-txt1_hours-", visible = False),
     sg.Spin(values=[i for i in range(0, 24)], key="-st_light-", initial_value=7, size=(2, 1), font=('Any 12'), visible = False),
     #sg.Spin(values=[i for i in range(1, 32)], key="-st_light_day-", initial_value=14, size=(2, 1), font=('Any 12'), visible = False),
     sg.Input(key='-In_date-', size=(8,1), visible = False), sg.CalendarButton('+', close_when_date_chosen=True,  target='-In_date-', location=(0,0), no_titlebar=False, key = "-click_date1-" , visible = False),
     sg.T('until:', font = ('Any 12'), key = "-txt3_hours-", visible = False),
     sg.Input(key='-Out_date-', size=(8,1), visible = False), sg.CalendarButton('+', close_when_date_chosen=True,  target='-Out_date-', location=(0,0), no_titlebar=False, key = "-click_date2-", visible = False),
     sg.Spin(values=[i for i in range(0, 24)], key="-sl_light-", initial_value=7, size=(2, 1), font=('Any 12'), visible = False),
     sg.Checkbox('', change_submits = True, enable_events=True, default=False, key='-limit_hours-', font=('Any 12'))],
    [sg.T('', font=('Any 12'))],
    [sg.Button('', font=('Any 13'),button_color=("white"),border_width=0)],
    [sg.T('', font=('Any 12'))]
]

export_text_column = [
    [sg.T('Export settings',font=('Any 15'))],
    [sg.HSeparator(pad = (0,0))],
    [sg.T('Power diagram:', font=('Any 12'))],
     [sg.T('Power over time:', font=('Any 12'))],
    #[sg.T('Merge mice with identical ID within a group:', font=('Any 12'))],
    [sg.T('Export figure(s):', font=('Any 12'))],
    [sg.T('Export data:', font=('Any 12'))],
    [sg.T('Export to path:',font=('Any 12'))],
    [sg.T('', font=('Any 12'))],
    [sg.Button('', font=('Any 12'),button_color=("white"),border_width=0)], 
    [sg.T('', font=('Any 12'))], 
    [sg.T('', font=('Any 12'))]
]

export_values_column = [
    [sg.T('',font=('Any 17'))],
    [sg.Checkbox('', change_submits = True, enable_events=True, default=False,key='-power_diagram-', font=('Any 11'))],
    [sg.T('Between',key = '-txt1-',font=('Any 12'), visible = False), sg.Input('1', key="-HZ1-", size=(2,1), font=('Any 12'), visible = False),
     sg.T('and',key = '-txt2-',font=('Any 12'), visible = False), sg.Input('3.5', key="-HZ2-", size=(2,1), font=('Any 12'), visible = False),
     sg.T('(Hz)',key = '-txt3-', font = ('Any 12'), visible = False),
     sg.Checkbox('', change_submits = True, enable_events=True, default=False,key='-power_time-', font=('Any 11'))],
    [sg.Checkbox('', change_submits = True, enable_events=True, default=True,key='-figures-', font=('Any 11'))],
    [sg.Checkbox('', change_submits = True, enable_events=True, default=False,key='-data_files-', font=('Any 11'))],
    #[sg.Checkbox('', change_submits = True, enable_events=True, default=False,key='-merging-', font=('Any 12'))],
    [sg.InputText('/Users/albert/Desktop/',key="-export_path-",size=(20,1),font=('Any 12'))],
    [sg.T('', font=('Any 11'))],
    [sg.T('', font=('Any 12'))], 
    [sg.T('', font=('Any 12'))],
    [sg.Button('Run', font=('Any 12'), key='-Run-')]
]


# ----- Full layout -----
layout_power = [
    [sg.Column(input_text_column, element_justification='left'),
     sg.Column(input_values_column, element_justification='right'),
     sg.VSeperator(pad = (20,35)),
     sg.Column(export_text_column, element_justification='left'),
      sg.Column(export_values_column, element_justification='right')]
    ]




#Define tabs in main window
tab_welcome = [[sg.Text('Welcome to the EEG Toolbox.')]]
tab_power = layout_power
tab_placeholder = [[sg.Text('This is a placeholder tab for further analysis')]]

layout1 = [[sg.TabGroup([[sg.Tab('Welcome', tab_welcome, key='-welcome_tab-'),
                         sg.Tab('Power analysis', layout_power, key='-power_tab-'),
                         sg.Tab('Placeholder', tab_placeholder, key='-placeholder_tab-')
                         ]], 
                        key = "-tabs-", tab_location='top', selected_title_color='blue')
            ]]


n_mice = 10


def tab(i,n_mice):
        l = []
        l += [[sg.T("Group label: "),
               sg.InputText(key = f"-group_label-{i}", size = (35,1)),
               sg.T("RGB: ("),
               sg.InputText(randint(0,254),key = f"-group_color_r-{i}", size = (3,1)),
               sg.T(","),
               sg.InputText(randint(0,254),key = f"-group_color_g-{i}", size = (3,1)),
               sg.T(","),
               sg.InputText(randint(0,254),key = f"-group_color_b-{i}", size = (3,1)),
               sg.T(")")],
              [sg.HSeparator()]]
        for j in range(n_mice):
            l += [[sg.T("Mouse ID: "), 
                   sg.InputText(key=f"-id-{i,j}",size=(5,1)), sg.VSeparator(),
                   sg.T("EDF file: "), 
                   sg.InputText(key=f"-edf-{i,j}",change_submits=True,
                                size=(35,1)),
                   sg.FileBrowse(key=f"-edf_browse-{i,j}",file_types=(("EDF files", "*.edf"),)), sg.VSeparator(),
                   sg.T("TSV file: "), 
                   sg.InputText(key=f"-tsv-{i,j}",
                                change_submits=True,size=(35,1)),
                   sg.FileBrowse(key=f"-tsv_browse-{i,j}",file_types=(("TSV files", "*.tsv"),)), sg.VSeparator(),
                   sg.T("Electrode: "), 
                          sg.InputText(key=f"-electrode-{i,j}",size=(3,1))]] 
        l += [[sg.HSeparator()],
              [sg.T("... or direct path to folder only containing the group files that are named exactly equal besides extension:"),
               sg.InputText(key = f"-group_folder-{i}", size = (55,1))]]
        return l

def tab_name(i):
        return f'Group {i}'



index1 = 1
index2 = 1


#DEFINE WINDOW 1
def make_win1(): return sg.Window('EEG Toolbox', layout1,size=(1300, 500), finalize=True)

#DEFINE WINDOW 2
def make_win2(n_mice):
    find_files_tabs = [[sg.Tab(tab_name(i), tab(i,n_mice), 
                        key=f'Group {i}') for i in range(index1)]]
    layout_find_files   = [[sg.TabGroup(find_files_tabs, key='TabGroup',tab_location='topleft', font=('Any 12'))], 
                        [sg.Button('New group', font=('Any 12')), 
                          sg.Button('Save', font=('Any 12'))]]
    layout = layout_find_files
    return sg.Window('Find files', layout,size=(1500, n_mice*40 + 130), finalize=True)


window1, window2 = make_win1(), None # starts with 1 window open


in_ID = []; in_edf = []; in_tsv = []; in_electrode = []; #Initialize user input and computed output arrays
f = []; AvgTrace = []; stdP = [] ;stdM = []; #Initialize computed output arrays

saved_check = 0

while True:
    window, event, values = sg.read_all_windows()
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        window.close()
        if window == window2:       # if closing win 2, mark as closed
            window2 = None
        elif window == window1:     # if closing win 1, exit program
            break

    
    if event == 'New group':
        window2['TabGroup'].add_tab(sg.Tab(f'Group {index1}', tab(index1,n_mice), key=f'Group {index1}'))
        window2[f'Group {index1}'].select()
        index1 += 1    
    
    elif event == 'Save':
        window, event, values = sg.read_all_windows()
        saved_check = 1
        saved_values = values
        window.close()
        if window == window2:       # if closing win 2, mark as closed
            window2 = None
        window1['-saved_prompt-'].update(visible=True)    
        #print('values:', values)
        
    elif event == '-Run-':
        if saved_check:
            window, event, values = sg.read_all_windows() #Read all values from both windows
            
            bin_power_time      = values['-power_time-']
            bin_power_diag      = values['-power_diagram-']
            bin_limit_hour      = values['-limit_hours-']
            bin_export_dat      = values['-data_files-']
            if bin_power_diag: fig1, ax1 = plt.subplots(figsize=(6,4), dpi=600); int1 = range(0,80) # Set plotting range
            if bin_power_time: fig2, ax2 = plt.subplots(figsize=(6,4), dpi=600) #Set plot
            
            
            group_labels = list(range(index1))
            for i in range(index1): # Loop through all groups
                in_ID_temp = []; in_edf_temp = []; in_tsv_temp = []; in_electrode_temp = []; #Initialize storage for user input
                group_labels[i] = saved_values[f'-group_label-{i}'] #Save group labels to use them in naming of the output file
                
                for j in range(n_mice): # Loop through all mice
                    in_ID_temp.append([saved_values[f'-id-({i}, {j})']])
                    in_edf_temp.append([saved_values[f'-edf-({i}, {j})']])
                    in_tsv_temp.append([saved_values[f'-tsv-({i}, {j})']])
                    in_electrode_temp.append([saved_values[f'-electrode-({i}, {j})']])
                    
                #Collect input and files
                in_ID.append([i, in_ID_temp]); in_edf.append([i, in_edf_temp]) # Structure: ==>   in_edf[#groups][0 = group number, 1 = files][files][0 = file]
                in_tsv.append([i, in_tsv_temp]); in_electrode.append([i, in_electrode_temp]); 
                
                if values['-fs-'].isnumeric(): fs = int(values['-fs-']);
                else: print("ERROR. 'fs' not numeric. Automatically choosing sample frewquency s.t. fs = 400"); fs = 400;
                
                
                
                if isfloat(values['-filter_N-']): filter_N = float(values['-filter_N-']);
                else: print("ERROR. 'filter_N' not numeric. Automatically choosing filter_N = 8/2"); filter_N = 8/2;
                if isfloat(values['-filter_Wn1-']): filter_Wn1 = float(values['-filter_Wn1-']);
                else: print("ERROR. 'filter_Wn1' not numeric. Automatically choosing filter_Wn1 = 0.3"); filter_Wn1 = 0.3;
                if isfloat(values['-filter_Wn2-']): filter_Wn2 = float(values['-filter_Wn2-']);
                else: print("ERROR. 'filter_Wn2' not numeric. Automatically choosing filter_Wn2 = 30"); filter_Wn2 = 30;
                
                
                #Compute filtered data to get preprossered data 
                folder_path = saved_values[f"-group_folder-{i}"] 
                data2, n, id_def_time     = get_filtered_data(in_ID, in_edf, in_tsv, in_electrode, i, fs, filter_N, filter_Wn1, filter_Wn2, folder_path)

                #Limit hours
                if bin_limit_hour: 
                    if len(values['-In_date-']) > 6 and len(values['-Out_date-']) > 6:
                        In_date = values['-In_date-']; Out_date = values['-Out_date-']
                    else: 
                        print("ERROR. Dates in limit not provided. Choosing random dates"); In_date = '(12, 7, 2022)'; Out_date = '(12, 7, 2022)';
                    if type(values['-st_light-']) == int and type(values['-sl_light-']) == int: 
                        In_time = int(values['-st_light-']); Out_time = int(values['-sl_light-'])
                    else: "ERROR. Clock in limit hours not provided properly as integer. Automatically choosing all available hours."; In_time = 7; Out_time = 7;
                    
                #Get export relevant input
                if values['-sleep_stage-'].isnumeric(): ss = int(values['-sleep_stage-']);
                else: print("ERROR. Sleep stage not numeric. Automatically choosing sleep stage = 1"); ss = 1;
                bin_limit_hour = values['-limit_hours-']
                
                # Power across frequencies
                if bin_power_diag and n > 0:
                    cf_l   = 0.5; cf_h   = 100;
                    if  bin_limit_hour:
                        st_light, sl_light      =  get_hours(id_def_time[i], [In_time,Out_time], [In_date,Out_date])
                    else:
                        st_light = int(0); sl_light = int(24*60*60/4)
                    f, AvgTrace, stdP, stdM = power_freq(data2, n, cf_l, cf_h, ss, st_light, sl_light, fs)
                    a_freq = np.asarray([ AvgTrace, stdP, stdM  ]);
                    #Plot
                    ax1.set_xlim([0, 20]) # Set xlim wrt. plotting range
                    ax1.set_xlabel("Frequency [Hz]", fontweight='bold') #Maybe these should be user defined as well?
                    ax1.set_ylabel("Relative EEG Power [% average power]") #Maybe these should be user defined as well?
                    
                    ax1.plot(f[int1], AvgTrace[int1], '-', linewidth = 1, 
                            label = saved_values[f'-group_label-{i}'],
                            color = (int(saved_values[f'-group_color_r-{i}'])/256, 
                                     int(saved_values[f'-group_color_g-{i}'])/256, 
                                     int(saved_values[f'-group_color_b-{i}'])/256)
                            ) # Plots power spectrum
                    ax1.fill_between(f[int1], stdP[int1], stdM[int1], alpha = 0.2,
                                    color = (int(saved_values[f'-group_color_r-{i}'])/256, 
                                             int(saved_values[f'-group_color_g-{i}'])/256, 
                                             int(saved_values[f'-group_color_b-{i}'])/256)
                                    )
                    ax1.legend()
                    
                    # write .csv files 
                    if bin_export_dat: 
                        np.savetxt(add_path_separator(values['-export_path-'], 'PowerFreq_' + 'data_' + str(group_labels[i]) + '_group_' + str(i)), a_freq, delimiter=",",header='AvgTrace, stdP, stdM') 

                #Power across time   
                if bin_power_time and n > 0:
                    cf_l = float(values['-HZ1-']); cf_h = float(values['-HZ2-']); cf_l_norm = 0.5; cf_h_norm = 30
                    print(cf_l)
                    if bin_limit_hour:
                        st_light, sl_light      =  get_hours(id_def_time[i], [In_time,Out_time], [In_date,Out_date])
                    else:
                        st_light = int(0); sl_light = int(24*60*60/4)
                    #print(bin_limit_hour); print(st_light); print(sl_light);
                    AvgTrace, stdP, stdM = power_time(data2, n, cf_l, cf_h, cf_l_norm, cf_h_norm, ss, st_light, sl_light, fs)
                    a_time = np.asarray([ AvgTrace, stdP, stdM  ]);
                    #Plot
                    #ax2.set_xlim([0, 24]) # Set xlim wrt. plotting range
                    #ax2.set_ylim([150, 500]) # Set xlim wrt. plotting range
                    ax2.set_xlabel("Hours [ZT]", fontweight='bold') #Maybe these should be user defined as well?
                    ax2.set_ylabel("Relative EEG Power [% average power]") #Maybe these should be user defined as well?
                    
                    light_on = 7 #Should be user defined parameter
                    t_zeitgeber = [];
                    t_start_zeitgeber = (id_def_time[0][0].hour - light_on)
                    for ii in range(24):
                        t_zeitgeber.append((t_start_zeitgeber+ii)%24)
                    

                    ax2.plot(t_zeitgeber, AvgTrace, '-', linewidth =1, 
                            label = saved_values[f'-group_label-{i}'], 
                            color = (int(saved_values[f'-group_color_r-{i}'])/256, 
                                     int(saved_values[f'-group_color_g-{i}'])/256, 
                                     int(saved_values[f'-group_color_b-{i}'])/256)
                            )
                    ax2.fill_between(t_zeitgeber,stdP, stdM, alpha=0.2, 
                                    color = (int(saved_values[f'-group_color_r-{i}'])/256, 
                                             int(saved_values[f'-group_color_g-{i}'])/256, 
                                             int(saved_values[f'-group_color_b-{i}'])/256)
                                    )
                    plt.xticks(np.arange(min(t_zeitgeber), max(t_zeitgeber)+1, 2.0))
                    trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
                    ax2.fill_between(t_zeitgeber, 0, 1, where=np.array(t_zeitgeber) > 11,
                                    facecolor='gray', alpha=0.1, transform=trans)
                    ax2.legend()
                    
                    # write .csv files 
                    if bin_export_dat: 
                        np.savetxt(add_path_separator(values['-export_path-'], 'PowerTime_' + 'data_' + str(group_labels[i]) + '_group_' + str(i)), a_time, delimiter=",",header='AvgTrace, stdP, stdM') 
                    
            #Save figures    
            if bin_power_diag: 
                fig1.savefig(add_path_separator(values['-export_path-'], 'PowerFreq' + 'plot' + '-'.join(group_labels)) )
            if bin_power_time: 
                fig2.savefig(add_path_separator(values['-export_path-'], 'PowerTime' + 'plot' + '-'.join(group_labels) ))
                
        else:
            break
    
    elif event == '-power_time-':
        window1['-power_time-'].update(visible=False)
        window1['-txt1-'].update(visible=values['-power_time-'])
        window1['-HZ1-'].update(visible=values['-power_time-'])
        window1['-txt2-'].update(visible=values['-power_time-'])
        window1['-HZ2-'].update(visible=values['-power_time-'])
        window1['-txt3-'].update(visible=values['-power_time-'])
        window1['-power_time-'].update(visible=True)
    
    # elif event == '-limit_hours-':
    #     window1['-limit_hours-'].update(visible=False)
    #     window1['-txt1_hours-'].update(visible=values['-limit_hours-'])
    #     window1['-st_light-'].update(visible=values['-limit_hours-'])
    #     #window1['-txt2_hours-'].update(visible=values['-limit_hours-'])
    #     window1['-In_date-'].update(visible=values['-limit_hours-'])
    #     window1['-click_date1-'].update(visible=values['-limit_hours-'])
    #     window1['-txt3_hours-'].update(visible=values['-limit_hours-'])
    #     window1['-sl_light-'].update(visible=values['-limit_hours-'])
    #     window1['-Out_date-'].update(visible=values['-limit_hours-'])
    #     window1['-click_date2-'].update(visible=values['-limit_hours-'])
    #     window1['-limit_hours-'].update(visible=True)
    
    
    elif event == '-find_files-':
        if not window2:
            n_mice_prev = n_mice
            if type(values['-n_mice-']) == int: n_mice = int(values['-n_mice-'])
            else: print("ERROR. Number of mice is not numeric. Automatically choosing 10"); n_mice = 10; 
            window2 = make_win2(n_mice)
            #Make sure that user input is saved if reopening the window 2
            if saved_check:
                for i in range(index1):
                    window2[f'-group_label-{i}'].update(saved_values[f'-group_label-{i}'])
                    window2[f'-group_color_r-{i}'].update(saved_values[f'-group_color_r-{i}'])
                    window2[f'-group_color_g-{i}'].update(saved_values[f'-group_color_g-{i}'])
                    window2[f'-group_color_b-{i}'].update(saved_values[f'-group_color_b-{i}'])
                    window2[f'-group_folder-{i}'].update(saved_values[f'-group_folder-{i}'])
                    for j in range(min(n_mice_prev,n_mice)):
                        window2[f'-id-({i}, {j})'].update(saved_values[f'-id-({i}, {j})'])
                        window2[f'-edf-({i}, {j})'].update(saved_values[f'-edf-({i}, {j})'])
                        window2[f'-tsv-({i}, {j})'].update(saved_values[f'-tsv-({i}, {j})'])
                        window2[f'-electrode-({i}, {j})'].update(saved_values[f'-electrode-({i}, {j})'])
                        
    elif event == '-click_date1-':
        date = sg.popup_get_date()
        window1['-In_date-'].update(date)
    elif event == '-click_date2-':
        date = sg.popup_get_date()
        window1['-Out_date-'].update(date)

window.close()


#Include merging option
#Read tsv more stable
#Different deviations - sd or SEM

