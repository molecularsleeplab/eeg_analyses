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
from matplotlib.patches import Rectangle
from backbone import *


# ----- Design options -------
sg.theme("Reddit")
font = ('Any', 12)
sg.set_options(font=font)



# First the window layout in 2 columns
input_text_column = [
    [sg.T('Input settings',font=('Any 15'))],
    [sg.HSeparator(pad = (0,0))],
    [sg.T('', font=('Any 12'))], 
    #[sg.T('Number of groups:', font=('Any 12'))],
     [sg.T('Max. number of mice per group:', font=('Any 12'))],
     [sg.T('Sleep stage:', font=('Any 12'))],
    [sg.T('Sample frequency:', font=('Any 12'))],
    [sg.T('Butterworth parameters (N,Wn1,Wn2):', font=('Any 12'))],
    [sg.T('Example:', font = ('Any', 9,'italic'))],
    [sg.T('Limit hours of interest:', font=('Any 12'))],
    #[sg.T(' ', font=('Any 12'), key = "-whitespace2-", visible = False)],
    [sg.T('Mark dark hours:', font=('Any 12'))],
    [sg.T('Standardization:', font=('Any 12'))],
    [sg.T('', font=('Any 12'))],
    [sg.T(' ', font=('Any 12'), key = "-bout_bins_text4-", visible = False)],
    [sg.Button('Find files', font=('Any 12'),key='-find_files-'), 
     sg.T('Files have been saved',key = "-saved_prompt-",font=("Any", 12, "italic"), visible = False)],
]

input_values_column = [
    [sg.T('',font=('Any 15'))],
    [sg.T('', font=('Any 12'))], 
    #[sg.HSeparator(pad = (0,0))],
     #[sg.Spin(values=[i for i in range(1, 50)], key="-n_groups-", initial_value=2, size=(2, 1), font=('Any 12'))],
     [sg.Spin(values=[i for i in range(1, 1000)], key="-n_mice-", initial_value=10, size=(2, 1), font=('Any 12'))],
    [sg.Input('1', key="-sleep_stage-", size=(3,1), font=('Any 12'))],
    [sg.Input('400', key="-fs-", size=(3,1), font=('Any 12'))],
    [sg.Input('4', key="-filter_N-", size=(3,1), font=('Any 12')),
     sg.Input('0.3', key="-filter_Wn1-", size=(3,1), font=('Any 12')), 
     sg.Input('30', key="-filter_Wn2-", size=(3,1), font=('Any 12'))],
    [sg.T("a,b = signal.butter(N/2,[Wn1/(fs/2),Wn2/(fs/2)])",  font = ('Any', 9,'italic'))],
    [sg.T('From:', font = ('Any 12'), key = "-txt1_hours-", visible = False),
     sg.Spin(values=[i for i in range(0, 24)], key="-st_light-", initial_value=7, size=(2, 1), font=('Any 12'), visible = False),
     sg.Checkbox('', change_submits = True, enable_events=True, default=False, key='-limit_hours-', font=('Any 12')),
     sg.T('until:', font = ('Any 12'), key = "-txt3_hours-", visible = False),
     #sg.Input(key='-Out_date-', size=(8,1), visible = False), sg.CalendarButton('+', close_when_date_chosen=True,  target='-Out_date-', location=(0,0), no_titlebar=False, key = "-click_date2-", visible = False),
     sg.Spin(values=[i for i in range(0, 24)], key="-sl_light-", initial_value=19, size=(2, 1), font=('Any 12'), visible = False)],
    [sg.Checkbox('', change_submits = True, enable_events=True, default=True, key='-bin_dark_hours-', font=('Any 12'))],
    [sg.Checkbox('', change_submits = True, enable_events=True, default=True, key='-bin_standard-', font=('Any 12'))],
    [sg.T(' ', font=('Any 12'), key = "-bout_bins_text3-", visible = False)],
    [sg.Button('', font=('Any 12'),button_color=("white"),border_width=0)],
    [sg.T('', font=('Any 12'))], 
]

export_text_column = [
    [sg.T('Export settings',font=('Any 15'))],
    [sg.HSeparator(pad = (0,0))],
    [sg.T('', font=('Any 12'))], 
    [sg.T('Power diagram:', font=('Any 12'))],
     [sg.T('Power over time:', font=('Any 12'))],
     [sg.T('Bout counts:', font=('Any 12'))],
     [sg.T('', key = "-bout_bins_text2-", font=('Any 12'),visible= False)],
    #[sg.T('Merge mice with identical ID within a group:', font=('Any 12'))],
    [sg.T('Export figure(s):', font=('Any 12'))],
    [sg.T('Export data:', font=('Any 12'))],
    [sg.T('Export to path:',font=('Any 12'))],
    [sg.T('Daylight hours:', font=('Any 12'))],
    [sg.Button('', font=('Any 12'),button_color=("white"),border_width=0)], 
    [sg.T('', font=('Any 12'))],
    [sg.T('', font=('Any 12'))],
]

export_values_column = [
    [sg.T('', font=('Any 12'))], 
    [sg.T('',font=('Any 19'))],
    [sg.T('Between',key = '-txt11-',font=('Any 12'), visible = False), sg.Input('0.5', key="-cf_l-", size=(3,1), font=('Any 12'), visible = False),
     sg.T('and',key = '-txt21-',font=('Any 12'), visible = False), sg.Input('100', key="-cf_h-", size=(3,1), font=('Any 12'), visible = False),
     sg.T('(Hz)',key = '-txt31-', font = ('Any 12'), visible = False),
     sg.Checkbox('', change_submits = True, enable_events=True, default=False,key='-power_diagram-', font=('Any 11'))],
    [sg.T('Between',key = '-txt1-',font=('Any 12'), visible = False), sg.Input('1', key="-HZ1-", size=(2,1), font=('Any 12'), visible = False),
     sg.T('and',key = '-txt2-',font=('Any 12'), visible = False), sg.Input('3.5', key="-HZ2-", size=(2,1), font=('Any 12'), visible = False),
     sg.T('(Hz)',key = '-txt3-', font = ('Any 12'), visible = False),
     sg.Checkbox('', change_submits = True, enable_events=True, default=False,key='-power_time-', font=('Any 11'))],
    [sg.T('Min. bout length',key = '-min_len_bout_text-',font=('Any 12'), visible = False), sg.Spin(values=[i for i in range(1, 24)], key="-min_bout_len-", initial_value=2, size=(1, 1), font=('Any 12'), visible = False),
     sg.Checkbox('', change_submits = True, enable_events=True, default=False,key='-bout_count-', font=('Any 11'))],
    [sg.T('Bout bins (sec.):',key = '-bout_bins_text-',font=('Any 12'), visible = False), 
     sg.InputText('8,16,32,60',key="-bout_bins-",size=(10,1),font=('Any 12'),visible= False)],
    [sg.Checkbox('', change_submits = True, enable_events=True, default=True,key='-figures-', font=('Any 11'))],
    [sg.Checkbox('', change_submits = True, enable_events=True, default=False,key='-data_files-', font=('Any 11'))],
    #[sg.Checkbox('', change_submits = True, enable_events=True, default=False,key='-merging-', font=('Any 12'))],
    [sg.InputText('/Users/albert/Desktop/',key="-export_path-",size=(30,1),font=('Any 12'))],
    [sg.Spin(values=[i for i in range(0, 24)], key="-light_on-", initial_value=7, size=(2, 1), font=('Any 12'), visible = True),
    sg.T('until:', font = ('Any 12'), key = "-txt3_hours-", visible = True),
    sg.Spin(values=[i for i in range(0, 24)], key="-light_off-", initial_value=19, size=(2, 1), font=('Any 12'), visible = True)],
    [sg.T('', font=('Any 11'))],
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


layout1 = layout_power

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
def make_win1(): return sg.Window('Sleep Analysis Toolbox', layout1,size=(1300, 500), finalize=True)

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


maxy = 0; miny = 0; maxx = 0; minx = 0; start_dark_zg = 0; end_dark_zg = 0;  #Initialize plot ranges

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
        saved_values = values
        saved_check = 1
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
            bin_bout            = values['-bout_count-']
            bin_dark            = values['-bin_dark_hours-']
            bin_standard        = values['-bin_standard-']
            if bin_power_diag: fig1, ax1 = plt.subplots(figsize=(6,4), dpi=600); int1 = range(0,80) # Set plotting range
            if bin_power_time: fig2, ax2 = plt.subplots(figsize=(6,4), dpi=600); #Set plot
            if bin_bout:       fig3, ax3 = plt.subplots(figsize=(6,4), dpi=600); #Set plot
            

            #group_labels = list(range(index1))
            group_labels = []
            for i in range(index1): # Loop through all groups
                in_ID_temp = []; in_edf_temp = []; in_tsv_temp = []; in_electrode_temp = []; #Initialize storage for user input
                #group_labels[i] = saved_values[f'-group_label-{i}'] #Save group labels to use them in naming of the output file
                group_labels.append(saved_values[f'-group_label-{i}'])
                
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
                if bin_power_diag or bin_power_time: data2, n, id_def_time = get_filtered_data(in_ID, in_edf, in_tsv, in_electrode, i, fs, filter_N, filter_Wn1, filter_Wn2, folder_path)
                if bin_bout: data3, n3, id_def_time3 = get_filtered_data2(in_ID, in_tsv, i, folder_path); 
                    

                #Limit hours
                if bin_limit_hour: 
                    In_time = int(values['-st_light-']); Out_time = int(values['-sl_light-'])
                    
                #Get export relevant input
                if values['-sleep_stage-'].isnumeric(): ss = int(values['-sleep_stage-']);
                else: print("ERROR. Sleep stage not numeric. Automatically choosing sleep stage = 1"); ss = 1;
                bin_limit_hour = values['-limit_hours-']
                
                #===================== Power across frequencies ===============
                if bin_power_diag and n > 0:
                    #cf_l   = 0.5; cf_h   = 100;
                    cf_l = float(values['-cf_l-']); cf_h = float(values['-cf_h-']);
                    if  bin_limit_hour:
                        #st_light, sl_light      =  get_hours(id_def_time[i], [In_time,Out_time], [In_date,Out_date])
                        #print(id_def_time); print(len(id_def_time)); print(In_time); print(Out_time)
                        st_light, sl_light, hours_dif  =  get_hours2(id_def_time[0], [In_time,Out_time])
                    else:
                        st_light = int(0); sl_light = int(24*60*60/4)
                    f, AvgTrace, stdP, stdM = power_freq(data2, n, cf_l, cf_h, ss, st_light, sl_light, fs, bin_standard)
                    a_freq = np.asarray([ AvgTrace, stdP, stdM  ]);
                    #Plot
                    ax1.set_xlim([0, 20]) # Set xlim wrt. plotting range
                    ax1.set_xlabel("Frequency [Hz]", fontweight='bold') #Maybe these should be user defined as well?
                    if bin_standard: #Maybe these should be user defined as well?
                        ax1.set_ylabel("Relative EEG Power [% average power]") 
                    else: 
                        ax1.set_ylabel("EEG Power") 
                    
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

                #====================== Power across time =====================  
                if bin_power_time and n > 0:
                    cf_l = float(values['-HZ1-']); cf_h = float(values['-HZ2-']); cf_l_norm = 0.5; cf_h_norm = 30
                    print(cf_l)
                    if bin_limit_hour:
                        #st_light, sl_light      =  get_hours(id_def_time[i], [In_time,Out_time], [In_date,Out_date])
                        #print(id_def_time[i]); print(In_time); print(Out_time)
                        st_light, sl_light, hours_dif =  get_hours2(id_def_time[0], [In_time,Out_time])
                    else:
                        st_light = int(0); sl_light = int(24*60*60/4)


                    AvgTrace, stdP, stdM = power_time(data2, n, cf_l, cf_h, cf_l_norm, cf_h_norm, ss, st_light, sl_light, fs, bin_standard)
                    a_time = np.asarray([ AvgTrace, stdP, stdM  ]);
                    
                    #Plot
                    ax2.set_xlabel("Hours [ZT]", fontweight='bold') #Maybe these should be user defined as well?
                    if bin_standard: #Maybe these should be user defined as well?
                        ax2.set_ylabel("Relative EEG Power [% average power]") 
                    else: 
                        ax2.set_ylabel("EEG Power") 
                    
                    
                    
                    light_on = int(values['-light_on-']); light_off = int(values['-light_off-']);
                    t_zeitgeber = [];
                    t_start_zeitgeber = (id_def_time[0][0].hour - light_on)
                    for ii in range(24):
                        t_zeitgeber.append((t_start_zeitgeber+ii)%24)
                    
                    if  bin_limit_hour:
                        idx_start = In_time - id_def_time[0][0].hour
                        idx_end = idx_start + hours_dif
                        x_range = range((idx_end - idx_start)%24)
                    else: 
                        idx_start = 0
                        idx_end = 24
                        x_range = range(24)
                    
                    ax2.plot(x_range, AvgTrace, '-', linewidth =1, 
                            label = saved_values[f'-group_label-{i}'], 
                            color = (int(saved_values[f'-group_color_r-{i}'])/256, 
                                     int(saved_values[f'-group_color_g-{i}'])/256, 
                                     int(saved_values[f'-group_color_b-{i}'])/256)
                            )
                    ax2.fill_between(x_range,stdP, stdM, alpha=0.2, 
                                    color = (int(saved_values[f'-group_color_r-{i}'])/256, 
                                             int(saved_values[f'-group_color_g-{i}'])/256, 
                                             int(saved_values[f'-group_color_b-{i}'])/256)
                                    )
                    
                    plt.xticks(x_range,t_zeitgeber[idx_start:idx_end])
                                        
                    if bin_dark:
                        start_dark = np.where(np.array(t_zeitgeber[idx_start:idx_end]) == 12)[0][0]
                        end_dark = np.where(np.array(t_zeitgeber[idx_start:idx_end]) == max(t_zeitgeber[idx_start:idx_end]))[0][0]
                        dif_dark = min(abs(light_on - light_off), abs(end_dark - start_dark))
                        ax2.axvspan( start_dark, start_dark + dif_dark, facecolor = "gray", alpha = 0.1)   
                    
                    ax2.legend()
                    
                    # write .csv files 
                    if bin_export_dat: 
                        np.savetxt(add_path_separator(values['-export_path-'], 'PowerTime_' + 'data_' + str(group_labels[i]) + '_group_' + str(i)), a_time, delimiter=",",header='AvgTrace, stdP, stdM') 
            
                #======================== Bout bar chart ======================
                if bin_bout and n3 > 0:
                    bins = string_to_list(values['-bout_bins-']) 
                    light_on = int(values['-light_on-']); light_off = int(values['-light_off-']);
                    min_bout_len = int(values['-min_bout_len-']); print("Min bout length is set to " + str(min_bout_len))
                    
                    if bin_dark:
                        if bin_limit_hour:
                            start = In_time; end = Out_time
                            labs0, hist_nr0 = hist_dat(bins, count_scores_all_mice(data3, ss, min_bout_len, id_def_time3, max(start, light_on ), min(end, light_off) , True)) #hours of interests during light
                            labs1not, hist_nr1not = hist_dat(bins, count_scores_all_mice(data3, ss, min_bout_len, id_def_time3, max(end, light_off), min(start, light_on), True)) #hours of no interests during dark
                            labs1, hist_nr1all = hist_dat(bins, count_scores_all_mice(data3, ss, min_bout_len, id_def_time3, light_off, light_on, True)) #all dark hours
                            hist_nr1 = np.array(hist_nr1all) - np.array(hist_nr1not) #hours of interest during dark = all dark hours - hours of no interests during dark
                        else:
                            start = 1; end = 24
                            labs0, hist_nr0 = hist_dat(bins, count_scores_all_mice(data3, ss, min_bout_len, id_def_time3, light_on, light_off, True))
                            labs1, hist_nr1 = hist_dat(bins, count_scores_all_mice(data3, ss, min_bout_len, id_def_time3, light_off, light_on, True))
                                                
                        labs = np.concatenate((labs0, labs1), axis=0)
                        hist_nr = np.concatenate((hist_nr0, hist_nr1),  axis=0)
                    else:
                        labs, hist_nr =   hist_dat(bins, count_scores_all_mice(data3, ss, min_bout_len, id_def_time3, 1, 24, False))
                    
                    a_bout = np.asarray([hist_nr]);
                    ind = np.arange(len(labs))
                    
                    width = 1/(index1+1)
                    ax3.bar(ind + i*width, hist_nr, width, 
                            color = (int(saved_values[f'-group_color_r-{i}'])/256, 
                                     int(saved_values[f'-group_color_g-{i}'])/256, 
                                     int(saved_values[f'-group_color_b-{i}'])/256), 
                            label = saved_values[f'-group_label-{i}'])
                    
                    
                    ax3.set_xlabel("Episode duration Bout [B]", fontweight='bold') #Maybe these should be user defined as well?
                    ax3.set_ylabel("# of episodes") #Maybe these should be user defined as well?
                    factor = (n3)/2 - 1 
                    
                    #ax3.xticks(X_axis + factor*width, X)
                    #ax3.xticks(rotation=15)
                    X = labs
                    X_axis = np.arange(len(X))
                    
                    ax3.set_xticks(X_axis + width*i/2)
                    ax3.set_xticklabels(X, fontdict=None, rotation=90, fontsize = 8)
                    plt.tight_layout()
                    
                    
                    if bin_dark and i == index1:
                        ax3.fill_between(range(idx_end - idx_start), indic ,edgecolor="white", alpha = 0.1, color = "gray")
                    
                    ax3.legend()
                    
                    
                    # write .csv files 
                    if bin_export_dat: 
                        np.savetxt(add_path_separator(values['-export_path-'], 'Bout_' + 'data_' + str(group_labels[i]) + '_group_' + str(i)), a_bout, delimiter=",",header='Scores') 
                    
                    
            
            
            
            #Save figures    
            if bin_power_diag: 
                fig1.savefig(add_path_separator(values['-export_path-'], 'PowerFreq' + 'plot' + '-'.join(group_labels)) )
            if bin_power_time: 
                fig2.savefig(add_path_separator(values['-export_path-'], 'PowerTime' + 'plot' + '-'.join(group_labels) ))
            if bin_bout: 
                fig3.savefig(add_path_separator(values['-export_path-'], 'Bout' + 'plot' + '-'.join(group_labels) ))
                
        #else:
        #    break
    
    elif event == '-power_time-':
        window1['-power_time-'].update(visible=False)
        window1['-txt1-'].update(visible=values['-power_time-'])
        window1['-HZ1-'].update(visible=values['-power_time-'])
        window1['-txt2-'].update(visible=values['-power_time-'])
        window1['-HZ2-'].update(visible=values['-power_time-'])
        window1['-txt3-'].update(visible=values['-power_time-'])
        window1['-power_time-'].update(visible=True)
        
    elif event == '-power_diagram-':
        window1['-power_diagram-'].update(visible=False)
        window1['-txt11-'].update(visible=values['-power_diagram-'])
        window1['-cf_l-'].update(visible=values['-power_diagram-'])
        window1['-txt21-'].update(visible=values['-power_diagram-'])
        window1['-cf_h-'].update(visible=values['-power_diagram-'])
        window1['-txt31-'].update(visible=values['-power_diagram-'])
        window1['-power_diagram-'].update(visible=True)
    
    elif event == '-limit_hours-':
        window1['-limit_hours-'].update(visible=False)
        window1['-txt1_hours-'].update(visible=values['-limit_hours-'])
        window1['-st_light-'].update(visible=values['-limit_hours-'])
        window1['-txt3_hours-'].update(visible=values['-limit_hours-'])
        window1['-sl_light-'].update(visible=values['-limit_hours-'])
        window1['-limit_hours-'].update(visible=True)
    
    elif event == '-bout_count-':
        window1['-bout_count-'].update(visible=False)
        window1['-min_len_bout_text-'].update(visible=values['-bout_count-'])
        window1['-min_bout_len-'].update(visible=values['-bout_count-'])
        window1['-bout_bins_text-'].update(visible=values['-bout_count-'])
        window1['-bout_bins_text2-'].update(visible=values['-bout_count-'])
        window1['-bout_bins_text3-'].update(visible=values['-bout_count-'])
        window1['-bout_bins_text4-'].update(visible=values['-bout_count-'])
        window1['-bout_bins-'].update(visible=values['-bout_count-'])
        window1['-bout_count-'].update(visible=True)
        
        
        
    
    
    elif event == '-find_files-':
        if not window2:
            n_mice_prev = n_mice
            if type(values['-n_mice-']) == int: n_mice = int(values['-n_mice-'])
            else: print("ERROR. Number of mice is not numeric. Automatically choosing 10"); n_mice = 10; 
            window2 = make_win2(n_mice)
            #User input is saved if reopening the window 2
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

window.close()

#Handle "mark dark hours". Should work fine now for power across time. Check one more time and implement hereafter on boutplots.
#every 2nd tick on power plot
#Files not updated proberly after 1st figure produced
#Export all bout data
#Different deviations - sd or SEM

