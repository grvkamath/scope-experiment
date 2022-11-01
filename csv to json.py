# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:04:39 2022

@author: gaura
"""
import os
## REPLACE PATH VARIABLE WITH LOCAL PATH
path = "C:\\Users\\gaura\\Documents\\GitHub\\scope-experiment-tentative-pilot"

os.chdir(path)
import csv
import json
csv_as_list = []
list_for_conv = []

with open("Continuations+ControlsAttempt1Sheet.csv", mode='r', encoding='utf-8-sig') as temp:
    csvreader = csv.reader(temp)
    header_entry = True
    for i in csvreader:
        if header_entry == True:
            headers=i
            header_entry = False
            continue
        csv_as_list_entry = dict()
        for j in range(len(headers)):
            csv_as_list_entry[headers[j]]=i[j]
        csv_as_list.append(csv_as_list_entry)
    linecount = 1
    for i in csv_as_list:
        if i['S']!='NA':
            list_for_conv_entry_S_f1 = {"idx":linecount, "sentence":i['S']+"...", "followup":i['F1'],
                                        "stype":"S", "ftype":"F1", "wide": i['F1_label'],
                                        "OP1_type": i['OP1_Type'], "OP2_type": i['OP2_Type'], 
                                        "OP1":i["OP1"], "OP2":i["OP2"]}
            list_for_conv_entry_S_f2 = {"idx":linecount, "sentence":i['S']+"...", "followup":i['F2'],
                                        "stype":"S", "ftype":"F2", "wide": i['F2_label'],
                                        "OP1_type": i['OP1_Type'], "OP2_type": i['OP2_Type'], 
                                        "OP1":i["OP1"], "OP2":i["OP2"]}
            list_for_conv_entry_Sc_f1 = {"idx":linecount, "sentence":i['Sc']+"...", "followup":i['F1'],
                                        "stype":"Sc", "ftype":"F1", "wide": "NA_CONTROL",
                                        "OP1_type": "NA_CONTROL", "OP2_type": "NA_CONTROL", 
                                        "OP1":"NA_CONTROL", "OP2":"NA_CONTROL"}
            list_for_conv_entry_Sc_f2 = {"idx":linecount, "sentence":i['Sc']+"...", "followup":i['F2'],
                                        "stype":"Sc", "ftype":"F2", "wide": "NA_CONTROL",
                                        "OP1_type": "NA_CONTROL", "OP2_type": "NA_CONTROL", 
                                        "OP1":"NA_CONTROL", "OP2":"NA_CONTROL"}
            list_for_conv.append(list_for_conv_entry_S_f1)
            list_for_conv.append(list_for_conv_entry_S_f2)
            list_for_conv.append(list_for_conv_entry_Sc_f1)
            list_for_conv.append(list_for_conv_entry_Sc_f2)
            linecount+=1

n_splits = 7

def list_slicer(List, N_splits):
    list_for_output = []
    for i in range(N_splits):
        list_for_output.append([])
    group = 0
    for i in List:
        list_for_output[group].append(i)
        group+=1
        if group == N_splits:
            group=0
    return list_for_output

list_for_conv = list_slicer(list_for_conv,n_splits)

stimuli_as_jsonStr = json.dumps(list_for_conv)

with open("js\\stimuli.js", "w", encoding="utf-8") as file:
    file.write("var all_stims = "+"\n"+stimuli_as_jsonStr)
    
