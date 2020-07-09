
import numpy as np
import matplotlib.pyplot as plt

r = {
    # Normal experiments
    'Independent': np.genfromtxt('IndependentModel__WithValSigmoid/results.txt'),
    'Sequential': np.genfromtxt('SequentialModel__WithVal/results.txt'),
    'Sequential_ConceptsBreakdown': np.genfromtxt('SequentialModel__WithVal/concepts.txt'),
    'Joint0.001': np.genfromtxt('Joint0.001Model/results.txt'),
    'Joint0.01': np.genfromtxt('Joint0.01Model/results.txt'),
    'Joint0.01_ConceptsBreakdown': np.genfromtxt('Joint0.01Model/concepts.txt'),
    'Joint0.1': np.genfromtxt('Joint0.1Model/results.txt'),
    'Joint1': np.genfromtxt('Joint1Model/results.txt'),
    'Standard': np.genfromtxt('Joint0Model/results.txt'),
    'Standard Probe': np.genfromtxt('Joint0Model_LinearProbe/results.txt'),
    'Standard No Bottleneck': np.genfromtxt('StandardNoBNModel/results.txt'),
    'Multitask': np.genfromtxt('MultitaskModel/results.txt'),

    # Data efficiency experiments
    'StandardModel_DataEffN1': np.genfromtxt('Joint0Model_DataEffN1_Result/results.txt'),
    'StandardModel_DataEffN3': np.genfromtxt('Joint0Model_DataEffN3_Result/results.txt'),
    'StandardModel_DataEffN7': np.genfromtxt('Joint0Model_DataEffN7_Result/results.txt'),
    'StandardModel_DataEffN10': np.genfromtxt('Joint0Model_DataEffN10_Result/results.txt'),
    'StandardModel_DataEffN15': np.genfromtxt('Joint0Model_DataEffN15_Result/results.txt'),

    'Joint0.01Model_DataEffN1': np.genfromtxt('Joint0.01Model_DataEffN1_Result/results.txt'),
    'Joint0.01Model_DataEffN3': np.genfromtxt('Joint0.01Model_DataEffN3_Result/results.txt'),
    'Joint0.01Model_DataEffN7': np.genfromtxt('Joint0.01Model_DataEffN7_Result/results.txt'),
    'Joint0.01Model_DataEffN10': np.genfromtxt('Joint0.01Model_DataEffN10_Result/results.txt'),
    'Joint0.01Model_DataEffN15': np.genfromtxt('Joint0.01Model_DataEffN15_Result/results.txt'),

    'IndependentModel_DataEffN1': np.genfromtxt('IndependentModel_WithVal_DataEffN1_Result/results.txt'),
    'IndependentModel_DataEffN3': np.genfromtxt('IndependentModel_WithVal_DataEffN3_Result/results.txt'),
    'IndependentModel_DataEffN7': np.genfromtxt('IndependentModel_WithVal_DataEffN7_Result/results.txt'),
    'IndependentModel_DataEffN10': np.genfromtxt('IndependentModel_WithVal_DataEffN10_Result/results.txt'),
    'IndependentModel_DataEffN15': np.genfromtxt('IndependentModel_WithVal_DataEffN15_Result/results.txt'),

    'SequentialModel_DataEffN1': np.genfromtxt('SequentialModel_WithVal_DataEffN1_Result/results.txt'),
    'SequentialModel_DataEffN3': np.genfromtxt('SequentialModel_WithVal_DataEffN3_Result/results.txt'),
    'SequentialModel_DataEffN7': np.genfromtxt('SequentialModel_WithVal_DataEffN7_Result/results.txt'),
    'SequentialModel_DataEffN10': np.genfromtxt('SequentialModel_WithVal_DataEffN10_Result/results.txt'),
    'SequentialModel_DataEffN15': np.genfromtxt('SequentialModel_WithVal_DataEffN15_Result/results.txt'),

    # TTI experiments
    'TTI_Joint0.01Model': np.genfromtxt('TTI__Joint0.01Model/results.txt'),
    'TTI_Joint0.01SigmoidModel': np.genfromtxt('TTI__Joint0.01SigmoidModel/results.txt'),
    'TTI_SequentialModel': np.genfromtxt('TTI__SequentialModel_WithVal/results.txt'),
    'TTI_IndependentModel': np.genfromtxt('TTI__IndependentModel_WithValSigmoid/results.txt'),

    # Adversarial experiments
    'StandardAdversarialModel': np.genfromtxt('Joint0AdversarialModel/results.txt'),
    'Joint0.01AdversarialModel': np.genfromtxt('Joint0.01AdversarialModel/results.txt'),
    'SequentialAdversarialModel': np.genfromtxt('SequentialAdversarialModel/results.txt'),
    'IndependentAdversarialModel': np.genfromtxt('IndependentAdversarialSigmoidModel/results.txt'),
}
# =============================================================================================
# ======================================== Table 1 & 2 ========================================
# =============================================================================================
exps = ['Independent', 'Sequential', 'Joint0.01', 'Standard', 'Standard Probe', 'Standard No Bottleneck', 'Multitask']
print('Table 1 & 2')
output_string = '                                    y Error    |    c Error    \n'
for exp in exps:
    if r[exp][0] >= 0:
        output_string += '%30s  %.3f +- %.3f | ' % (exp, r[exp][0], r[exp][1] * 2)
    else:
        output_string += '%30s         -       | ' % exp

    if r[exp][2] >= 0:
        output_string += '%.3f +- %.3f\n' % (r[exp][2], r[exp][3] * 2)
    else:
        output_string += '       -     \n'
print(output_string)

# =============================================================================================
# ========================================= Figure 2 ==========================================
# =============================================================================================
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
marker_style = { 'marker': 's', 'facecolors': 'none', 'edgecolors': '#1f77b4' }
data = [('Standard'                , 1.000, 0.440),
        ('Joint, $\lambda$ = 0.001', 0.829, 0.440),
        ('Joint, $\lambda$ = 0.01' , 0.595, 0.441),
        ('Independent'             , 0.529, 0.435),
        ('Joint, $\lambda$ = 0.1'  , 0.548, 0.432),
        ('Joint, $\lambda$ = 1'    , 0.543, 0.418),
        ('Sequential'              , 0.527, 0.418),]
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
subplt.scatter([d[1] for d in data], [d[2] for d in data], color=colors, **marker_style)
for (name, x, y), (del_x, del_y) in zip(data, delta_oai):
    del_x, del_y = del_x * x_unit, del_y * y_unit
    subplt.annotate(name, (x + del_x, y + del_y))
subplt.set_title('OAI')
subplt.set_xlabel('Concept ($c$) RMSE')
subplt.set_ylabel('Task ($y$) RMSE')

# ---- CUB Data ----
data = [('Standard'                , 0.5, r['Standard'][0]),
        ('Joint, $\lambda$ = 0.001', r['Joint0.001'][2], r['Joint0.001'][0]),
        ('Joint, $\lambda$ = 0.01' , r['Joint0.01'][2], r['Joint0.01'][0]),
        ('Joint, $\lambda$ = 0.1'  , r['Joint0.1'][2], r['Joint0.1'][0]),
        ('Sequential'              , r['Sequential'][2], r['Sequential'][0]),
        ('Joint, $\lambda$ = 1'    , r['Joint1'][2], r['Joint1'][0]),
        ('Independent'             , r['Independent'][2], r['Independent'][0])]
colors = ['#9467bd', '#ff7f0e', '#ff7f0e', '#ff7f0e', '#2ca02c', '#ff7f0e', '#d62728']
CUB_SCALE = 100.
x_unit, y_unit = 2.5/CUB_SCALE, 0.25/CUB_SCALE
delta_cub = [(-3.9,-0.5), (0.8,-0.4), (0.3,1.3), (0.5,-0.8), (0.6,-0.7), (0.6,-0.2), (0.6,-0.8)]

subplt = axes[1, 0]
subplt.scatter([d[1] for d in data], [d[2] for d in data], color=colors, **marker_style)
x_fill_1 = [x/CUB_SCALE for x in [3.12, 3.23, 14.21, 52]]
y_fill_1 = [x/CUB_SCALE for x in [25.5, 25.5, 25.5, 25.5]]
y_fill_2 = [x/CUB_SCALE for x in [24.3, 19.9, 17.0, 17.1]]
subplt.set_ylim(bottom=16/CUB_SCALE, top=25.5/CUB_SCALE)
subplt.set_xlim(left=0, right=52/CUB_SCALE)
subplt.fill_between(x_fill_1, y_fill_1, y_fill_2, where=y_fill_2 <= y_fill_1, facecolor='#7f7f7f', alpha=0.1)
for (name, x, y), (del_x, del_y) in zip(data, delta_cub):
    del_x, del_y = del_x * x_unit, del_y * y_unit
    subplt.annotate(name, (x + del_x, y + del_y))
subplt.set_title('CUB')
subplt.set_xlabel('Concept ($c$) error')
subplt.set_ylabel('Task ($y$) error')

# ========= Counts vs A performance =========
# ---- OAI ----
bins = np.arange(0, 1.01, 0.1)
x = np.arange(len(bins))  # the bin locations
bar_width, bar_gap = 0.5, 0.1
colors = ['#ff7f0e', '#2ca02c']

subplt = axes[0, 1]
data = [('Joint', [0, 0, 0, 0, 0, 0, 0, 2.2, 6.8, 1., 0]),
        ('Sequential / Independent', [0, 0, 0, 0, 0, 0, 0, 2., 7., 1., 0])]
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
data = [('Joint', r['Joint0.01_ConceptsBreakdown']),
        ('Sequential / Independent', r['Sequential_ConceptsBreakdown'])]
xlabel, xticklabels = 'F1', ['%.1f' % b for b in bins]
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
data = [('Standard', '#9467bd', [0.5089946, 0.47227314, 0.4460999, 0.44069982]),
        ('Joint', '#ff7f0e', [0.46282497, 0.45027956, 0.43035713, 0.4180873]),
        ('Sequential', '#2ca02c', [0.4682201081476601, 0.4541488508626818, 0.43791661291392225, 0.4296507395092277]),
        ('Independent', '#d62728', [0.4548289, 0.4435927, 0.42583558, 0.4179975])]
x = [10, 20, 50, 100]
subplt = axes[0, 2]
for name, color, y in data:
    subplt.plot(x, y, marker='s', fillstyle='none', label=name, color=color)
subplt.set_title('OAI')
subplt.set_xlim(left=0, right=105)
subplt.legend(loc='upper right')
subplt.set_xlabel('Data proportion (%)')
subplt.set_ylabel('Task ($y$) RMSE')
subplt.yaxis.grid(True, linestyle='--')

# ---- CUB ----
data = [('Standard', '#9467bd', [r['StandardModel_DataEffN1'][0], r['StandardModel_DataEffN3'][0], r['StandardModel_DataEffN7'][0], r['StandardModel_DataEffN10'][0], r['StandardModel_DataEffN15'][0], r['Standard'][0]]),
        ('Joint', '#ff7f0e', [r['Joint0.01Model_DataEffN1'][0], r['Joint0.01Model_DataEffN3'][0], r['Joint0.01Model_DataEffN7'][0], r['Joint0.01Model_DataEffN10'][0], r['Joint0.01Model_DataEffN15'][0], r['Joint0.01'][0]]),
        ('Sequential', '#2ca02c', [r['SequentialModel_DataEffN1'][0], r['SequentialModel_DataEffN3'][0], r['SequentialModel_DataEffN7'][0], r['SequentialModel_DataEffN10'][0], r['SequentialModel_DataEffN15'][0], r['Sequential'][0]]),
        ('Independent', '#d62728', [r['IndependentModel_DataEffN1'][0], r['IndependentModel_DataEffN3'][0], r['IndependentModel_DataEffN7'][0], r['IndependentModel_DataEffN10'][0], r['IndependentModel_DataEffN15'][0], r['Independent'][0]])]
x = [3.33, 10, 23.36, 33.37, 50, 100]
subplt = axes[1, 2]
for name, color, y in data:
    subplt.plot(x, y, marker='s', fillstyle='none', label=name, color=color)
subplt.set_title('CUB')
subplt.set_xlim(left=0, right=105)
subplt.legend(loc='upper right')
subplt.set_xlabel('Data proportion (%)')
subplt.set_ylabel('Task ($y$) error')
subplt.yaxis.grid(True, linestyle='--')

plt.subplots_adjust(hspace=0.4)
plt.tight_layout()
plt.savefig('figure2.png')

# ===============================================================================================
# ======================================== Figure 4: TTI ========================================
# ===============================================================================================

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4), dpi=300)

# ---- OAI ----
data = [(r'Control', '#1f77b4',
         [0.441, 0.424, 0.411, 0.407, 0.418, 0.43 , 0.458, 0.459, 0.446, 0.456, 0.456]),
        ('Joint', '#ff7f0e',
         [0.418, 0.361, 0.306, 0.284, 0.271, 0.26 , 0.237, 0.235, 0.241, 0.245, 0.244]),
        ('Sequential', '#2ca02c',
         [0.418, 0.364, 0.304, 0.288, 0.262, 0.247, 0.231, 0.23 , 0.235, 0.233, 0.235]),
        ('Independent', '#d62728',
         [0.43 , 0.384, 0.3  , 0.282, 0.241, 0.203, 0.161, 0.16 , 0.16 , 0.159, 0.159])]
xs = range(11)
for name, color, ys in data:
    axes[0].plot(xs, ys, marker='s', fillstyle='none', color=color, label=name)
axes[0].set_ylim(bottom=0.15, top=0.5)
axes[0].set_title(r'OAI (Nonlinear $c \rightarrow y$)')
axes[0].legend(loc='lower left', prop={'size': 9.5})
axes[0].set_xlabel('Number of concepts intervened')
axes[0].set_ylabel('Task ($y$) RMSE')
axes[0].yaxis.grid(True, linestyle='--')

# ---- OAI ----
data = [('Joint', '#ff7f0e',
         [0.419, 0.376, 0.364, 0.482, 0.469, 0.445, 0.442, 0.464, 0.461, 0.454, 0.451]),
        ('Sequential', '#2ca02c',
         [0.441, 0.414, 0.383, 0.378, 0.36 , 0.355, 0.354, 0.37 , 0.372, 0.372, 0.366]),
        ('Independent', '#d62728',
         [0.446, 0.417, 0.376, 0.37 , 0.351, 0.344, 0.34 , 0.339, 0.339, 0.34 , 0.339])]
xs = range(11)
for name, color, ys in data:
    axes[1].plot(xs, ys, marker='s', fillstyle='none', color=color, label=name)
axes[1].set_ylim(bottom=0.15, top=0.5)
axes[1].set_title(r'OAI (Linear $c \rightarrow y$)')
axes[1].legend(loc='lower left', prop={'size': 9.5})
axes[1].set_xlabel('Number of concepts intervened')
axes[1].yaxis.grid(True, linestyle='--')

# ---- CUB ----
data = [(r'Joint, from sigmoid', '#17becf', r['TTI_Joint0.01SigmoidModel'][:, 1]),
        ('Joint', '#ff7f0e', r['TTI_Joint0.01Model'][:, 1]),
        ('Sequential', '#2ca02c', r['TTI_SequentialModel'][:, 1]),
        ('Independent', '#d62728', r['TTI_IndependentModel'][:, 1])]
xs = range(29)
for name, color, ys in data:
    ys = [1 - y / 100. for y in ys]
    axes[2].plot(xs, ys, marker='s', fillstyle='none', color=color, label=name)
axes[2].set_title('CUB')
axes[2].legend(loc='lower left', prop={'size': 9.5})
axes[2].set_xlabel('Number of concept groups intervened')
axes[2].set_ylabel('Task ($y$) error')
axes[2].yaxis.grid(True, linestyle='--')
plt.subplots_adjust(wspace=0.25, bottom=0.15)
plt.savefig('figure4.png')

# ======================================================================================================
# ======================================== Table 3: Adversarial ========================================
# ======================================================================================================
exps = ['StandardAdversarialModel', 'Joint0.01AdversarialModel', 'SequentialAdversarialModel', 'IndependentAdversarialModel']
output_string = '                                    y Error    |    c Error    \n'
for exp in exps:
    output_string += '%30s  %.3f +- %.3f | ' % (exp, r[exp][0], r[exp][1] * 2)
    if r[exp][2] >= 0:
        output_string += '%.3f +- %.3f\n' % (r[exp][2], r[exp][3] * 2)
    else:
        output_string += '       -     \n'
print(output_string)
