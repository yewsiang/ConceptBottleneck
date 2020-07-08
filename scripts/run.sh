
# ------- Linear -------

#python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_Linear_LR00005_Opt1 --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --seed 603844 --lr 0.00005
#python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_Linear_LR00005_Opt2 --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --seed 841538 --lr 0.00005
#python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_Linear_LR00005_Opt3 --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --seed 620523 --lr 0.00005

python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_Linear_LR0005_Opt1 --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --seed 603844 --lr 0.0005
python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_Linear_LR0005_Opt2 --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --seed 841538 --lr 0.0005
python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_Linear_LR0005_Opt3 --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --seed 620523 --lr 0.0005

#python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_Linear_LR00005_Opt1_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_Linear_LR00005_Opt1/model_weights.pth --sampling uniform --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --C_weight 1 --seed 603844
#python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_Linear_LR00005_Opt2_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_Linear_LR00005_Opt2/model_weights.pth --sampling uniform --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --C_weight 1 --seed 841538
#python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_Linear_LR00005_Opt3_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_Linear_LR00005_Opt3/model_weights.pth --sampling uniform --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --C_weight 1 --seed 620523

python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_Linear_LR0005_Opt1_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_Linear_LR0005_Opt1/model_weights.pth --sampling uniform --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --C_weight 1 --seed 603844
python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_Linear_LR0005_Opt3_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_Linear_LR0005_Opt3/model_weights.pth --sampling uniform --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --C_weight 1 --seed 620523
python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_Linear_LR0005_Opt2_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_Linear_LR0005_Opt2/model_weights.pth --sampling uniform --fc_layers 20 1 --y_fc_name fc2 --C_fc_name fc1 --C_weight 1 --seed 841538

# ------- Nonlinear -------

#python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_FC50_LR00005_Opt1 --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --seed 603844 --lr 0.00005
#python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_FC50_LR00005_Opt2 --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --seed 841538 --lr 0.00005
#python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_FC50_LR00005_Opt3 --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --seed 620523 --lr 0.00005

#python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_FC50_LR0005_Opt1 --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --seed 603844 --lr 0.0005
#python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_FC50_LR0005_Opt2 --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --seed 841538 --lr 0.0005
#python experiments.py oai StandardWithAuxC --name XtoYwithAuxC_FC50_LR0005_Opt3 --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --seed 620523 --lr 0.0005

#python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_FC50_LR00005_Opt1_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_FC50_LR00005_Opt1/model_weights.pth --sampling uniform --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed 603844
python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_FC50_LR00005_Opt2_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_FC50_LR00005_Opt2/model_weights.pth --sampling uniform --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed 841538
python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_FC50_LR00005_Opt3_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_FC50_LR00005_Opt3/model_weights.pth --sampling uniform --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed 620523

#python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_FC50_LR0005_Opt1_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_FC50_LR0005_Opt1/model_weights.pth --sampling uniform --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed 603844
python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_FC50_LR0005_Opt2_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_FC50_LR0005_Opt2/model_weights.pth --sampling uniform --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed 841538
python experiments.py oai TTI --test_time_intervention_analysis --name XtoYwithAuxC_FC50_LR0005_Opt3_TTIBestImprov --test_time_intervention_model X_to_y_with_aux_C --test_time_intervention_method ordered --pretrained outputs/XtoYwithAuxC_FC50_LR0005_Opt3/model_weights.pth --sampling uniform --fc_layers 20 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed 620523

#nlprun -q jag -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8,jagupard9,jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15 --mem=54G -p high --output outputs/rebuttal4.out 'scripts/run.sh'