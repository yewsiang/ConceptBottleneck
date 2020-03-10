#!/usr/bin/env python
# coding: utf-8

# In[24]:


# Global dependencies
from config import Y_COLS, ATTRIBUTES_WO_KLG, ATTRIBUTES_BALANCED, CLINICAL_CONTROL_COLUMNS, OUTPUTS_DIR, BASE_DIR

import os
import pdb
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_fscore_support
from analysis import plot, convert_continuous_back_to_ordinal


# ----------------- Analysis -----------------
def summarize_results(experiment_folders, final=True, individual=False, keys=[], KLG_suffix=''):
    from analysis import convert_continuous_back_to_ordinal
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    ys = Y_COLS
    As = ATTRIBUTES_BALANCED 
    
    final_y_rmses, final_A_rmses = [], []
    for i, experiment_folder in enumerate(experiment_folders):
        results_path = os.path.join(experiment_folder, 'results.pkl')
        assert os.path.exists(results_path)
        
        results = pickle.load(open(results_path, 'rb'))
        if len(keys) == 0: 
            epochs = [result for result in results.keys() if isinstance(result, int)]
            last_epoch = epochs[-1]
        else:
            last_epoch = keys[i]
        
        try:
            final_A_rmses.append(np.mean([results['test_set_results']['test_%s_rmse' % attribute] for attribute in As]))
        except:
            try:
                final_A_rmses.append(np.mean([results[keys[i]]['test_%s_rmse' % attribute] for attribute in As]))
            except:
                pass
        try:
            final_y_rmses.append(results['test_set_results']['test_xrkl%s_rmse' % KLG_suffix])
        except:
            try:
                final_y_rmses.append(results[keys[i]]['test_xrkl%s_rmse' % KLG_suffix])
            except:
                pass
        continue

    string, string1, string2 = '%17s | %17s', '', ''
    if len(final_y_rmses) > 0: 
        avg, sd2 = np.mean(final_y_rmses), 2 * np.std(final_y_rmses, ddof=1)
        string1 = 'y: %.3f +- %.3f' % (avg, sd2)
    if len(final_A_rmses) > 0: 
        avg, sd2 = np.mean(final_A_rmses), 2 * np.std(final_A_rmses, ddof=1)
        string2 = 'C: %.3f +- %.3f' % (avg, sd2)
    print(string % (string1, string2))


# ==== Task and concept results ====
print('  Task and concept results')
# Independent
summarize_results(['outputs/OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0', 'outputs/OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0', 'outputs/OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0', 'outputs/OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0', 'outputs/OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0'], keys=['outputs/XtoA_A0.1_FC50_Opt1/model_weights.pth', 'outputs/XtoA_A0.1_FC50_Opt2/model_weights.pth', 'outputs/XtoA_A0.1_FC50_Opt3/model_weights.pth', 'outputs/XtoA_A0.1_FC50_Opt4/model_weights.pth', 'outputs/XtoA_A0.1_FC50_Opt5/model_weights.pth'])
# Sequential
summarize_results(['outputs/XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt1-_Opt1', 'outputs/XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt2-_Opt2', 'outputs/XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt3-_Opt3', 'outputs/XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt4-_Opt4', 'outputs/XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt5-_Opt5'])
# Joint
summarize_results(['outputs/XtoAtoY_Lambda1_FC50_Opt1', 'outputs/XtoAtoY_Lambda1_FC50_Opt2', 'outputs/XtoAtoY_Lambda1_FC50_Opt3', 'outputs/XtoAtoY_Lambda1_FC50_Opt4', 'outputs/XtoAtoY_Lambda1_FC50_Opt5'])

# Standard
summarize_results(['outputs/XtoY_FC50_Opt1', 'outputs/XtoY_FC50_Opt2', 'outputs/XtoY_FC50_Opt3', 'outputs/XtoY_FC50_Opt4', 'outputs/XtoY_FC50_Opt5'])
# Standard, No Bottleneck
summarize_results(['outputs/XtoY_Opt1', 'outputs/XtoY_Opt2', 'outputs/XtoY_Opt3', 'outputs/XtoY_Opt4', 'outputs/XtoY_Opt5'])
# Multitask
summarize_results(['outputs/XtoAY_A0.1_FC50_Opt1', 'outputs/XtoAY_A0.1_FC50_Opt2', 'outputs/XtoAY_A0.1_FC50_Opt3', 'outputs/XtoAY_A0.1_FC50_Opt4', 'outputs/XtoAY_A0.1_FC50_Opt5'])

# ==== Probe results ====
print('\n  Probe results')
# Standard [Probe]
summarize_results(['outputs/AProbes_Conv4_XtoAtoY_Lambda0_FC50_Opt1', 'outputs/AProbes_Conv4_XtoAtoY_Lambda0_FC50_Opt2', 'outputs/AProbes_Conv4_XtoAtoY_Lambda0_FC50_Opt3', 'outputs/AProbes_Conv4_XtoAtoY_Lambda0_FC50_Opt4'])
# SENN [Probe]
summarize_results(['outputs/XtoAFromFreezedSENNNewCpt10_Conv4_A0.1_Linear_1', 'outputs/XtoAFromFreezedSENNNewCpt10_Conv4_A0.1_Linear_4'])


# In[7]:


# ==== y RMSE vs A RMSE ====
def add_data_point(name, data, experiment_folders, keys=[], folder_prefix='outputs/'):
    x_values = []
    y_values = []
    for i, experiment_folder in enumerate(experiment_folders):
        results_path = os.path.join(folder_prefix + experiment_folder, 'results.pkl')
        assert os.path.exists(results_path), results_path
        
        # Specifically because of "Independent" experiment
        key = keys[i] if len(keys) > 0 else 'test_set_results'

        results = pickle.load(open(results_path, 'rb'))
        try:
            A_rmse = np.mean([results[key]['test_%s_rmse' % attribute] for attribute in ATTRIBUTES_BALANCED])
        except:
            A_rmse = 1.0
            
        y_rmse = results[key]['test_xrkl_rmse']
        x_values.append(A_rmse)
        y_values.append(y_rmse)
    print(name, np.mean(x_values), np.mean(y_values), np.std(x_values), np.std(y_values))
    data.append((name, np.mean(x_values), np.mean(y_values)))

# ==== Counting # of points for each Error bin ====
def add_data_point_for_bins(name, data, experiment_folders, correlation=True, folder_prefix='outputs/'):
    counts_per_bin_list = []
    for experiment_folder in experiment_folders:
        results_path = os.path.join(folder_prefix + experiment_folder, 'results.pkl')
        assert os.path.exists(results_path), results_path

        results = pickle.load(open(results_path, 'rb'))
        if correlation:
            values = [results['test_set_results']['test_%s_r' % attribute] for attribute in ATTRIBUTES_BALANCED]
        else:
            A_pred = [results['test_set_results']['test_%s_pred' % attribute] for attribute in ATTRIBUTES_BALANCED]
            A_true = [results['test_set_results']['test_%s_true' % attribute] for attribute in ATTRIBUTES_BALANCED]
            A_abs_error = np.array([np.abs(pred - true) for pred, true in zip(A_pred, A_true)])
            values = np.mean(A_abs_error, axis=0)
        
        bin_ids = np.digitize(values, bins)
        counts_per_bin = [np.sum(bin_ids == (i + 1)) for i in range(len(bins))]
        counts_per_bin_list.append(counts_per_bin)
    data.append((name, np.mean(np.array(counts_per_bin_list), axis=0)))

# ==== Data Efficiency ====
def add_data_efficiency_plots(name, data_proportions, experiment_folders_list, plot=None, color=None, variable='test_xrkl_rmse', folder_prefix='outputs/'):
    x = data_proportions
    y = []
    for experiment_folders in experiment_folders_list:
        y_values = []
        for experiment_folder in experiment_folders:
            results_path = os.path.join(folder_prefix + experiment_folder, 'results.pkl')
            assert os.path.exists(results_path), results_path

            results = pickle.load(open(results_path, 'rb'))
            if results.get('test_set_results') is None:
                keys = [k for k in results.keys()]
                assert len(keys) == 1
                y_values.append(results[keys[0]][variable])
            else:
                y_values.append(results['test_set_results'][variable])
        y.append(np.mean(y_values))
    
    subplt = plt if plot is None else plot
    subplt.plot(x, y, marker='s', fillstyle='none', label=name, color=color)

    
# ==================== Combined plots ====================
SMALL_SIZE  = 11
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE+1) # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9), dpi=300)

# ========= y vs C performance =========
# ---- OAI Data ----
data = []
add_data_point('Standard', data, ['XtoY_FC50_Opt1', 'XtoY_FC50_Opt2', 'XtoY_FC50_Opt3', 'XtoY_FC50_Opt4', 'XtoY_FC50_Opt5'])
add_data_point('Joint, $\lambda$ = 0.001', data, ['XtoAtoY_Lambda0.001_FC50_Opt1', 'XtoAtoY_Lambda0.001_FC50_Opt2', 'XtoAtoY_Lambda0.001_FC50_Opt3', 'XtoAtoY_Lambda0.001_FC50_Opt4', 'XtoAtoY_Lambda0.001_FC50_Opt5'])
add_data_point('Joint, $\lambda$ = 0.01', data, ['XtoAtoY_Lambda0.01_FC50_Opt1', 'XtoAtoY_Lambda0.01_FC50_Opt2', 'XtoAtoY_Lambda0.01_FC50_Opt3', 'XtoAtoY_Lambda0.01_FC50_Opt4', 'XtoAtoY_Lambda0.01_FC50_Opt5'])
add_data_point('Independent', data, ['OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0'], keys=['outputs/XtoA_A0.1_FC50_Opt1/model_weights.pth', 'outputs/XtoA_A0.1_FC50_Opt2/model_weights.pth', 'outputs/XtoA_A0.1_FC50_Opt3/model_weights.pth', 'outputs/XtoA_A0.1_FC50_Opt4/model_weights.pth', 'outputs/XtoA_A0.1_FC50_Opt5/model_weights.pth'])
add_data_point('Joint, $\lambda$ = 0.1', data, ['XtoAtoY_Lambda0.1_FC50_Opt1', 'XtoAtoY_Lambda0.1_FC50_Opt2', 'XtoAtoY_Lambda0.1_FC50_Opt3', 'XtoAtoY_Lambda0.1_FC50_Opt4', 'XtoAtoY_Lambda0.1_FC50_Opt5'])
add_data_point('Joint, $\lambda$ = 1', data, ['XtoAtoY_Lambda1_FC50_Opt1', 'XtoAtoY_Lambda1_FC50_Opt2', 'XtoAtoY_Lambda1_FC50_Opt3', 'XtoAtoY_Lambda1_FC50_Opt4', 'XtoAtoY_Lambda1_FC50_Opt5'])
add_data_point('Sequential', data, ['XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt1-_Opt1', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt2-_Opt2', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt3-_Opt3', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt4-_Opt4', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt5-_Opt5'])
colors = ['#9467bd', '#ff7f0e', '#ff7f0e', '#d62728', '#ff7f0e', '#ff7f0e', '#2ca02c']
x_unit, y_unit = 0.0125, 0.00125
delta_oai = [(-4,-1.7), (-6,-1.75), (1.3,-0.2), (1.4,-0.25), (1.4,-0.3), (1.2,-0.35), (-4.0,-1.7)]

subplt = axes[0, 0]
line = [d for i, d in enumerate(data) if i in [0, 2, 3, 6]]
x_fill_1 = [line[-1][1], line[-1][1], 1.05]
y_fill_1 = [line[-1][2], line[-1][2] + 0.5, line[-1][2] + 0.5]
y_fill_2 = [line[-1][2], line[-1][2], line[-1][2]]
subplt.set_ylim(bottom=0.415, top=0.445)
subplt.set_xlim(left=0.47, right=1.05)
subplt.fill_between(x_fill_1, y_fill_1, y_fill_2, where=y_fill_2 <= y_fill_1, facecolor='#7f7f7f', alpha=0.1)
marker_style = { 'marker': 's', 'facecolors': 'none', 'edgecolors': '#1f77b4' } 
subplt.scatter([d[1] for d in data], [d[2] for d in data], color=colors, **marker_style)
for (name, x, y), (del_x, del_y) in zip(data, delta_oai):
    del_x, del_y = del_x * x_unit, del_y * y_unit
    subplt.annotate(name, (x + del_x, y + del_y))
subplt.set_title('OAI')
subplt.set_xlabel('Concept ($c$) RMSE')
subplt.set_ylabel('Task ($y$) RMSE')
# ---- CUB Data ----
data = [('Standard'                , 50.0, 82.1),
        ('Joint, $\lambda$ = 0.001', 75.7, 82.5),
        ('Joint, $\lambda$ = 0.01' , 96.1, 81.0),
        ('Joint, $\lambda$ = 0.1'  , 96.8, 78.4),
        ('Sequential'              , 96.9, 77.8),
        ('Joint, $\lambda$ = 1'    , 96.9, 78.3),
        ('Independent'             , 96.9, 77.4)]
colors = ['#9467bd', '#ff7f0e', '#ff7f0e', '#ff7f0e', '#2ca02c', '#ff7f0e', '#d62728']
CUB_SCALE = 100.
x_unit, y_unit = 2.5/CUB_SCALE, 0.25/CUB_SCALE
delta_cub = [(-4.0,-0.3), (0.8,-0.4), (0.7,-0.35), (0.5,-1.2), (0.6,-0.5), (0.6,-0.1), (0.6,-0.5)]

subplt = axes[1, 0]
line = [d for i, d in enumerate(data) if i in [0, 1, 2, 3, 4, 6]]
subplt.scatter([(100 - d[1])/CUB_SCALE for d in data], [(100 - d[2])/CUB_SCALE for d in data], color=colors, **marker_style)
x_fill_1 = [x/CUB_SCALE for x in [3.1, 3.9, 24.3, 52]]
y_fill_1 = [x/CUB_SCALE for x in [23, 23, 23, 23]]
y_fill_2 = [x/CUB_SCALE for x in [21.2, 19, 17.5, 17.5]]
subplt.set_ylim(bottom=17/CUB_SCALE, top=23/CUB_SCALE)
subplt.set_xlim(left=0, right=52/CUB_SCALE)
subplt.fill_between(x_fill_1, y_fill_1, y_fill_2, where=y_fill_2 <= y_fill_1, facecolor='#7f7f7f', alpha=0.1)
for (name, x, y), (del_x, del_y) in zip(data, delta_cub):
    del_x, del_y = del_x * x_unit, del_y * y_unit
    subplt.annotate(name, ((100 - x)/CUB_SCALE + del_x, (100 - y)/CUB_SCALE + del_y))
subplt.set_title('CUB')
subplt.set_xlabel('Concept ($c$) error')
subplt.set_ylabel('Task ($y$) error')

# ========= Counts vs A performance =========
start, end, width = (0, 1.01, 0.1)
bins = np.arange(start, end, width)
# ---- OAI ----
data = []
add_data_point_for_bins('Joint', data, ['XtoAtoY_Lambda1_FC50_Opt1', 'XtoAtoY_Lambda1_FC50_Opt2', 'XtoAtoY_Lambda1_FC50_Opt3', 'XtoAtoY_Lambda1_FC50_Opt4', 'XtoAtoY_Lambda1_FC50_Opt5'])
add_data_point_for_bins('Sequential / Independent', data, ['XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt1-_Opt1', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt2-_Opt2', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt3-_Opt3', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt4-_Opt4', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt5-_Opt5'])
subplt = axes[0, 1]
x = np.arange(len(bins))  # the bin locations
bar_width, bar_gap = 0.5, 0.1
colors = ['#ff7f0e', '#2ca02c']
for i, d in enumerate(data):
    name, counts = d
    rects = subplt.bar(x + bar_width/2 + i * bar_width, counts, bar_width - bar_gap, color=colors[i], label=name)
# Add some text for labels, title and custom x-axis tick labels, etc.
subplt.set_xlabel('Pearson correlation')
subplt.set_ylabel('Average counts')
subplt.set_title('OAI')
subplt.set_xticks(x)
subplt.set_xticklabels(['%.1f' % b for b in bins])
subplt.set_xlim(left=0., right=10.)
subplt.legend()
# ---- CUB ----
subplt = axes[1, 1]
data = [('Joint', [0, 0, 0, 0, 0, 0, 0, 0, 49, 63, 0]),
        ('Sequential / Independent', [0, 0, 0, 0, 0, 0, 0, 0, 38, 74, 0])]
xlabel, xticklabels = 'F1', ['%.1f' % b for b in bins]
bar_width, bar_gap = 0.5, 0.1
for i, d in enumerate(data):
    name, counts = d
    rects = subplt.bar(x + bar_width/2 + i * bar_width, counts, bar_width - bar_gap, color=colors[i], label=name)
# Add some text for labels, title and custom x-axis tick labels, etc.
subplt.set_xlabel('F1')
subplt.set_ylabel('Average counts')
subplt.set_title('CUB')
subplt.set_xticks(x)
subplt.set_xticklabels(xticklabels)
subplt.set_xlim(left=0., right=10.)
subplt.legend()

# ========= Data efficiency =========
# ---- OAI ----
# ---- Seeded ----
subplt = axes[0, 2]
add_data_efficiency_plots('Standard', [10, 20, 50, 100], 
    [['XtoY_FC50_DataEffSeed0.1_1', 'XtoY_FC50_DataEffSeed0.1_2', 'XtoY_FC50_DataEffSeed0.1_3'],
     ['XtoY_FC50_DataEffSeed0.2_1', 'XtoY_FC50_DataEffSeed0.2_2', 'XtoY_FC50_DataEffSeed0.2_3'],
     ['XtoY_FC50_DataEffSeed0.5_1', 'XtoY_FC50_DataEffSeed0.5_2', 'XtoY_FC50_DataEffSeed0.5_3'],
     ['XtoY_FC50_Opt1', 'XtoY_FC50_Opt2', 'XtoY_FC50_Opt3', 'XtoY_FC50_Opt4', 'XtoY_FC50_Opt5']],
    plot=subplt, color='#9467bd')
add_data_efficiency_plots('Joint', [10, 20, 50, 100], 
    [['XtoAtoY_Lambda1_FC50_DataEffSeed0.1_1', 'XtoAtoY_Lambda1_FC50_DataEffSeed0.1_2', 'XtoAtoY_Lambda1_FC50_DataEffSeed0.1_3'],
    ['XtoAtoY_Lambda1_FC50_DataEffSeed0.2_1', 'XtoAtoY_Lambda1_FC50_DataEffSeed0.2_2', 'XtoAtoY_Lambda1_FC50_DataEffSeed0.2_3'],
    ['XtoAtoY_Lambda1_FC50_DataEffSeed0.5_1', 'XtoAtoY_Lambda1_FC50_DataEffSeed0.5_2', 'XtoAtoY_Lambda1_FC50_DataEffSeed0.5_3'],
    ['XtoAtoY_Lambda1_FC50_Opt1', 'XtoAtoY_Lambda1_FC50_Opt2', 'XtoAtoY_Lambda1_FC50_Opt3', 'XtoAtoY_Lambda1_FC50_Opt4', 'XtoAtoY_Lambda1_FC50_Opt5']],
    plot=subplt, color='#ff7f0e')
add_data_efficiency_plots('Sequential', [10, 20, 50, 100], 
    [['XtoAhat_AhatToY_PRE-XtoA_A0.1_DataEff0.1_1-_FC50_DataEff0.1_1', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_DataEff0.1_2-_FC50_DataEff0.1_2', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_DataEff0.1_3-_FC50_DataEff0.1_3'],
    ['XtoAhat_AhatToY_PRE-XtoA_A0.1_DataEff0.2_1-_FC50_DataEff0.2_1', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_DataEff0.2_2-_FC50_DataEff0.2_2', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_DataEff0.2_3-_FC50_DataEff0.2_3'],
    ['XtoAhat_AhatToY_PRE-XtoA_A0.1_DataEff0.5_1-_FC50_DataEff0.5_1', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_DataEff0.5_2-_FC50_DataEff0.5_2', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_DataEff0.5_3-_FC50_DataEff0.5_3'],
    ['XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt1-_Opt1', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt2-_Opt2', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt3-_Opt3', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt4-_Opt4', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt5-_Opt5']],
    plot=subplt, color='#2ca02c')
add_data_efficiency_plots('Independent', [10, 20, 50, 100], 
    [['OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff0.1_1', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff0.1_2', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff0.1_3'],
     ['OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff0.2_1', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff0.2_2', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff0.2_3'],
     ['OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff0.5_1', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff0.5_2', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff0.5_3'],
     ['OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff1.0_1', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff1.0_2', 'OracleAtoY_ontop_XtoA_MLP_FC50,50,1_Regul0_DataEff1.0_3']],
    plot=subplt, color='#d62728')
subplt.set_title('OAI')
subplt.set_xlim(left=0, right=105)
subplt.legend(loc='upper right')
subplt.set_xlabel('Data proportion (%)')
subplt.set_ylabel('Task ($y$) RMSE')
subplt.yaxis.grid(True, linestyle='--')
# ---- CUB ----
data = [('Standard', '#9467bd', [14.8, 37.1, 58.8, 67.7, 75.0, 79.0, 82.1]),
        ('Joint', '#ff7f0e', [22, 46.9, 63.6, 70.5, 75.6, 77.8, 81.0]),
        ('Sequential', '#2ca02c', [40.8, 53.3, 63, 67.6, 71.2, 75.5, 77.8]),
        ('Independent', '#d62728', [20.2, 41.9, 59.5, 66.8, 69.7, 75.2, 77.4])]
x = [3.33, 10, 23.36, 33.37, 50, 66.6, 100]
subplt = axes[1, 2]
for name, color, y in data:
    y = [1 - d/100. for d in y]
    subplt.plot(x, y, marker='s', fillstyle='none', label=name, color=color)
subplt.set_title('CUB')
subplt.set_xlim(left=0, right=105)
subplt.legend(loc='upper right')
subplt.set_xlabel('Data proportion (%)')
subplt.set_ylabel('Task ($y$) error')
subplt.yaxis.grid(True, linestyle='--')

plt.subplots_adjust(hspace=0.4)
plt.show()


# In[2]:


# ==================== Test-time intervention ==================== 
def analyse_test_time_intervention_results(experiment_folder):
    results_path = os.path.join(experiment_folder, 'results.pkl')
    assert os.path.exists(results_path)
    
    results = pickle.load(open(results_path, 'rb'))
    N_attributes_given_list = [result for result in results.keys() if isinstance(result, int)]
    
    metrics = ['rmse', 'macro_F1']

    # ---- Plot train / val performance with more attributes given ----
    analyse_performance_over_time(Y_COLS, metrics, results, N_attributes_given_list, splits=['test'])
    
def print_latex_coordinates(experiment_folder, variable='test_xrkl_rmse'):
    results_path = os.path.join(experiment_folder, 'results.pkl')
    assert os.path.exists(results_path)
    
    results = pickle.load(open(results_path, 'rb'))
    N_attributes_given_list = [result for result in results.keys() if isinstance(result, int)]
    
    str = ''
    for n in N_attributes_given_list:
        str += '(%d,%.3f)' % (n, results[n][variable])
    print(str)

def compare_tti_plots(experiments, variable='test_xrkl_rmse', folder_prefix='outputs/', colors=None, plot=None):
    subplt = plt if plot is None else plot
    for i, (name, experiment_folders) in enumerate(experiments):
        
        xs, ys = [], []
        for experiment_folder in experiment_folders:
            results_path = os.path.join(folder_prefix + experiment_folder, 'results.pkl')
            assert os.path.exists(results_path), results_path

            results = pickle.load(open(results_path, 'rb'))
            N_attributes_given_list = [result for result in results.keys() if isinstance(result, int)]

            x, y = [], []
            for n in N_attributes_given_list:
                x.append(n)
                y.append(results[n][variable])
            xs.append(x)
            ys.append(y)
            
        xs = np.mean(np.array(xs), axis=0)
        ys = np.mean(np.array(ys), axis=0)
        color = colors[i] if colors is not None else None
        subplt.plot(xs, ys, marker='s', fillstyle='none', color=color, label=name)     

SMALL_SIZE  = 11
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE+1) # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
       

# ------------------------------------------------------------------------------
# TTI Figures
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4), dpi=300)
# ---- OAI ----
compare_tti_plots([('Control', ['XtoAtoY_Lambda0.01_FC50_Opt1_TTISeed_OrderBestImprovValBased', 'XtoAtoY_Lambda0.01_FC50_Opt2_TTISeed_OrderBestImprovValBased', 'XtoAtoY_Lambda0.01_FC50_Opt3_TTISeed_OrderBestImprovValBased', 'XtoAtoY_Lambda0.01_FC50_Opt4_TTISeed_OrderBestImprovValBased', 'XtoAtoY_Lambda0.01_FC50_Opt5_TTISeed_OrderBestImprovValBased']),
                   ('Joint', ['XtoAtoY_Lambda1_FC50_Opt1_TTISeed_OrderBestImprovValBased', 'XtoAtoY_Lambda1_FC50_Opt2_TTISeed_OrderBestImprovValBased', 'XtoAtoY_Lambda1_FC50_Opt3_TTISeed_OrderBestImprovValBased', 'XtoAtoY_Lambda1_FC50_Opt4_TTISeed_OrderBestImprovValBased', 'XtoAtoY_Lambda1_FC50_Opt5_TTISeed_OrderBestImprovValBased']),
                   ('Sequential', ['XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt1-_Opt1_TTISeed_OrderBestImprov', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt2-_Opt2_TTISeed_OrderBestImprov', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt3-_Opt3_TTISeed_OrderBestImprov', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt4-_Opt4_TTISeed_OrderBestImprov', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_FC50_Opt5-_Opt5_TTISeed_OrderBestImprov']),
                   ('Independent', ['XtoAhat_OracleAToY_PRE-XtoA_A0.1_FC50_Opt1-_Opt1_TTI_OrderBestImprov', 'XtoAhat_OracleAToY_PRE-XtoA_A0.1_FC50_Opt2-_Opt2_TTI_OrderBestImprov', 'XtoAhat_OracleAToY_PRE-XtoA_A0.1_FC50_Opt3-_Opt3_TTI_OrderBestImprov', 'XtoAhat_OracleAToY_PRE-XtoA_A0.1_FC50_Opt4-_Opt4_TTI_OrderBestImprov', 'XtoAhat_OracleAToY_PRE-XtoA_A0.1_FC50_Opt5-_Opt5_TTI_OrderBestImprov'])
                   ], plot=axes[0])
axes[0].set_ylim(bottom=0.15, top=0.5)
axes[0].set_title(r'OAI (Nonlinear $c \rightarrow y$)')
axes[0].legend(loc='lower left', prop={'size': 9.5})
axes[0].set_xlabel('Number of concepts intervened')
axes[0].set_ylabel('Task ($y$) RMSE')
axes[0].yaxis.grid(True, linestyle='--')
# ---- OAI ----
compare_tti_plots([('Joint', ['XtoAtoY_Lambda1_Opt1_TTISeed_OrderBestImprov', 'XtoAtoY_Lambda1_Opt2_TTISeed_OrderBestImprov', 'XtoAtoY_Lambda1_Opt3_TTISeed_OrderBestImprov', 'XtoAtoY_Lambda1_Opt4_TTISeed_OrderBestImprov', 'XtoAtoY_Lambda1_Opt5_TTISeed_OrderBestImprov']),
                   ('Sequential', ['XtoAhat_AhatToY_PRE-XtoA_A0.1_Opt1-_Opt1_TTISeed_OrderBestImprov', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_Opt2-_Opt2_TTISeed_OrderBestImprov', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_Opt3-_Opt3_TTISeed_OrderBestImprov', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_Opt4-_Opt4_TTISeed_OrderBestImprov', 'XtoAhat_AhatToY_PRE-XtoA_A0.1_Opt5-_Opt5_TTISeed_OrderBestImprov']),
                   ('Independent', ['XtoAhat_OracleAToY_PRE-XtoA_A0.1_Opt1-_Opt1_TTI_OrderBestImprov', 'XtoAhat_OracleAToY_PRE-XtoA_A0.1_Opt2-_Opt2_TTI_OrderBestImprov', 'XtoAhat_OracleAToY_PRE-XtoA_A0.1_Opt3-_Opt3_TTI_OrderBestImprov', 'XtoAhat_OracleAToY_PRE-XtoA_A0.1_Opt4-_Opt4_TTI_OrderBestImprov', 'XtoAhat_OracleAToY_PRE-XtoA_A0.1_Opt5-_Opt5_TTI_OrderBestImprov'])
                   ], plot=axes[1], colors=['#ff7f0e', '#2ca02c', '#d62728'])
axes[1].set_ylim(bottom=0.15, top=0.5)
axes[1].set_title(r'OAI (Linear $c \rightarrow y$)')
axes[1].legend(loc='lower left', prop={'size': 9.5})
axes[1].set_xlabel('Number of concepts intervened')
axes[1].yaxis.grid(True, linestyle='--')

# ---- CUB ----
data = [(r'Joint, from sigmoid', '#17becf', [77.6, 78.1, 78.6, 79.2, 79.7, 80.4, 81.0, 81.9, 82.7, 83.7, 85.0, 85.9, 86.9, 88.3, 89.4, 90.6, 91.6, 92.8, 93.5, 93.9, 94.7, 95.3, 95.4, 95.3, 95.6, 95.4, 95.3, 95.3, 95.3]),
        ('Joint', '#ff7f0e', [81.0, 81.6, 81.8, 82.3, 82.8, 83.2, 83.6, 84.0, 84.1, 84.7, 85.4, 85.0, 85.5, 85.3, 85.4, 85.6, 85.6, 85.6, 85.2, 84.9, 85.0, 84.9, 84.1, 84.0, 83.2, 82.7, 82.1, 81.5, 81.3]),
        ('Sequential', '#2ca02c', [77.8, 79.1, 79.5, 80.3, 81.2, 81.7, 82.7, 83.6, 84.3, 85.6, 86.2, 86.8, 87.7, 88.3, 88.5, 89.0, 90.1, 90.3, 91.0, 91.1, 91.5, 91.7, 91.6, 91.8, 91.8, 92.0, 92.1, 91.8, 91.5]),
        ('Independent', '#d62728', [77.5, 78.2, 78.7, 79.2, 80.0, 81.0, 81.8, 82.6, 83.7, 84.6, 86.2, 87.4, 89.0, 90.0, 91.5, 92.3, 93.8, 94.5, 95.5, 96.1, 96.5, 96.7, 97.4, 97.5, 97.5, 97.6, 97.7, 97.6, 97.6])]
xs = range(29)
for name, color, ys in data:
    ys = [1 - y/100. for y in ys]
    axes[2].plot(xs, ys, marker='s', fillstyle='none', color=color, label=name)
axes[2].set_title('CUB')
axes[2].legend(loc='lower left', prop={'size': 9.5})
axes[2].set_xlabel('Number of concept groups intervened')
axes[2].set_ylabel('Task ($y$) error')
axes[2].yaxis.grid(True, linestyle='--')
plt.subplots_adjust(wspace=0.25)
plt.show()

