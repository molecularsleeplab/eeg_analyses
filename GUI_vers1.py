#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:32:54 2022

@author: albert
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 07:05:01 2022

@author: albert
"""

import PySimpleGUI as sg
import matplotlib.pyplot as plt

from backbone import *


sg.theme("Reddit")
font = ('Courier New', 11)
sg.set_options(font=font)



layout0 = [[sg.T("Export (path + filename): "), 
            sg.InputText('/Users/albert/Desktop/',key="-export_path-",size=(35,1)), 
            sg.T(" + "),
            sg.InputText('EEGplot1',key="-export_filename-",size=(15,1))],
            [sg.T('Sleep stage: '), sg.Input('1', key="-sleep_stage-", size=(3,1))
            #sg.Text("EEG Toolbox", size=(200,40))
            ]] #Welcome layout

n_mice = 20

def tab(i):
    if i == 0:
        return layout0                                                          # Welcome layout
    else:
        l = []
        for j in range(n_mice):
            l += [[sg.T("Mouse ID: "), 
                   sg.InputText(key=f"-id-{i,j}",size=(5,1)), sg.VSeparator(),
                   sg.T("EDF file: "), 
                   sg.InputText(key=f"-edf-{i,j}",change_submits=True,
                                size=(35,1)),
                   sg.FileBrowse(key=f"-edf_browse-{i,j}"), sg.VSeparator(),
                   sg.T("TSV file: "), 
                   sg.InputText(key=f"-tsv-{i,j}",
                                change_submits=True,size=(35,1)),
                   sg.FileBrowse(key=f"-tsv_browse-{i,j}"), sg.VSeparator(),
                   sg.T("Electrode: "), 
                          sg.InputText(key=f"-electrode-{i,j}",size=(3,1))]] 
        return l


#layout = [[sg.T("")], [sg.Text("Choose a folder: "), sg.Input(key="-IN2-" ,change_submits=True), sg.FolderBrowse(key="-IN-")],[sg.Button("Submit")]]



def tab_name(i):
    if i == 0:
        return 'Welcome'                                                       #Group 0 is the welcome page
    else:
        return f'Group {i}'


#def new_layout(j):
#    return [[sg.T("Question: "), sg.InputText(key=("-q-", j)), sg.T("Answer"), sg.InputText(key=("-ans-", j))]]


index1 = 1
index2 = 1

tabgroup = [[sg.Tab(tab_name(i), tab(i), 
                    key=f'Group {i}') for i in range(index1)]]


#Layout with scroll
# layout   = [[sg.Column([[
#                 sg.TabGroup(tabgroup, key='TabGroup',tab_location='topleft')], 
#                     [sg.Text('Sleep stage: '), sg.Input(key="-ss-", size=(3,1)),
#                       sg.Button('New group'), 
#                       sg.Button('Save'),sg.Button('Close')]],
#             scrollable=True,
#             vertical_scroll_only=True)
#               ]]

layout   = [[sg.TabGroup(tabgroup, key='TabGroup',tab_location='topleft')], 
                    [sg.Button('New group'), 
                      sg.Button('Save')],
                    [sg.Text('Does Something', key='-print_values-', visible=False)]]



window   = sg.Window('EEG Toolbox', layout,size=(1150, 700))

in_ID = []; in_edf = []; in_tsv = []; in_electrode = []; #Initialize user input and computed output arrays
f = []; AvgTrace = []; stdP = [] ;stdM = []; #Initialize computed output arrays

while True:

    event, values = window.read(timeout=10)
    #print('event:', event)
    #print('values:', values)
    
    if event == sg.WIN_CLOSED: #or event in (None, 'Close'):
        break

    
    if event == 'New group':
        window['TabGroup'].add_tab(sg.Tab(f'Group {index1}', tab(index1), key=f'Group {index1}'))
        window[f'Group {index1}'].select()
        #window[f'Group {index1}'].update(visible=True)
        index1 += 1
        
    if event == 'Save':
        
        fig, ax = plt.subplots(figsize=(6,4), dpi=600) #Set plot
        int1 = range(0,80) # Set plotting range
        plt.xlim([0, 20]) # Set xlim wrt. plotting range
        plt.xlabel("Frequency [Hz]", fontweight='bold') 
        plt.ylabel("Relative EEG Power [% average power]")
        
        ss = int(values['-sleep_stage-'])
        for i in range(1,index1): # Loop through all groups
            in_ID_temp = []; in_edf_temp = []; in_tsv_temp = []; in_electrode_temp = []; #Initialize storage for user input
            
            for j in range(n_mice): # Loop through all mice
                in_ID_temp.append([values[f'-id-({i}, {j})']])
                in_edf_temp.append([values[f'-edf-({i}, {j})']])
                in_tsv_temp.append([values[f'-tsv-({i}, {j})']])
                in_electrode_temp.append([values[f'-electrode-({i}, {j})']])
                
            in_ID.append([i, in_ID_temp]); in_edf.append([i, in_edf_temp]) # Structure: ==>   in_edf[#groups][0 = group number, 1 = files][files][0 = file]
            in_tsv.append([i, in_tsv_temp]); in_electrode.append([i, in_electrode_temp])
            
            ss = 1  
            f, AvgTrace, stdP, stdM = front2back(in_ID, in_edf, in_tsv, in_electrode, i-1, ss) # Sends user input to backbone
            
            ax.plot(f[int1], AvgTrace[int1], '-', linewidth = 1, label = f'Group {i}') # Plots power spectrum
            ax.fill_between(f[int1], stdP[int1], stdM[int1], alpha = 0.2)
            ax.legend()
            
        fig.savefig(values['-export_path-'] + values['-export_filename-'])
        break
    
    
    

window.close()



#Include output file txt FFT
#Include show figure/save figures to directory
#Include filter options
#Include merging option
#Include color option
#Include mouse specific electrode option
#Include "fs" option
#Include st_light,sl_light
#Include "path"-option if all files are named accordingly
#Include extensive error checking

#Filter out single sleep stages occurances


