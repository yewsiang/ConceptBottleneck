
# Concept Bottleneck Models - OAI Dataset

## Dataset preprocessing

1) Request OAI dataset from the following url: https://nda.nih.gov/oai/

2) Run preprocessing steps according to Pierson et al and as described in the paper.

## Experiments

1) Update the BASE_DIR and DATA_DIR directories in config.py to point to the base directory of the repository and the directory of the above preprocessed code.

2) Run experiments and replace [SEED] with the following seeds {603844, 841538, 620523, 217182, 84137}, and [ID] with the id of the seed. EG. The seed of 841538 has id 2.

### a. Task and concept tables (Table 1 and 2)

##### 1. Independent
```
python train.py --name XtoC_C0.1_FC50_Opt[ID] --exp Concept_XtoC --fc_layers 10 --C_fc_name fc1 --C_weight 0.1 --seed [SEED] --lr 0.0005

python train.py --name OracleCtoY_ontop_XtoC_MLP_FC50,50,1_Regul0 --exp Independent_CtoY --fc_layers 10 --C_fc_name fc1 --C_weight 0.1 --lr 0.0005 --eval_model X_to_C --oracle_C_to_y_model mlp --oracle_C_to_y_model_params "{ \"hidden_layer_sizes\": [50,50], \"alpha\": 0 }" \
--pretrained outputs/XtoC_C0.1_FC50_Opt[ID]/model_weights.pth --seed [SEED]
```

python experiments.py oai --name XtoC_C0.1_FC50_Opt1 --exp Concept_XtoC --fc_layers 10 --C_fc_name fc1 --C_weight 0.1 --seed 1 --lr 0.0005 --use_small_subset
python train.py --name OracleCtoY_ontop_XtoC_MLP_FC50,50,1_Regul0 --exp Independent_CtoY --fc_layers 10 --C_fc_name fc1 --C_weight 0.1 --lr 0.0005 --eval_model X_to_C --oracle_C_to_y_model mlp --oracle_C_to_y_model_params "{ \"hidden_layer_sizes\": [50,50], \"alpha\": 0 }" \
--pretrained outputs/XtoC_C0.1_FC50_Opt1/model_weights.pth --seed 1

##### 2. Sequential
```
python train.py --name XtoChat_ChatToY_PRE-XtoC_C0.1_FC50_Opt[ID]-_Opt[ID] --exp Sequential_CtoY --pretrained outputs/XtoC_C0.1_FC50_Opt[ID]/model_weights.pth --front_fc_layers_to_freeze 1 --fc_layers 10 50 50 1 --C_fc_name fc1 --y_fc_name fc4 --C_weight 0.1 --seed [SEED] --lr 0.0005
```
python experiments.py oai --name XtoChat_ChatToY_PRE-XtoC_C0.1_FC50_Opt1-_Opt1 --exp Sequential_CtoY --pretrained outputs/XtoC_C0.1_FC50_Opt1/model_weights.pth --front_fc_layers_to_freeze 1 --fc_layers 10 50 50 1 --C_fc_name fc1 --y_fc_name fc4 --C_weight 0.1 --seed 1 --lr 0.0005 --use_small_subset

##### 3. Joint
```
python train.py --name XtoCtoY_Lambda1_FC50_Opt[ID] --exp Joint --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed [SEED] --lr 0.0005
```
python experiments.py oai --name XtoCtoY_Lambda1_FC50_Opt1 --exp Joint --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed 1 --lr 0.0005 --use_small_subset

##### 4. Standard
```
python train.py --name XtoY_FC50_Opt[ID] --exp train_X_to_y --fc_layers 10 50 50 1 --y_fc_name fc4 --seed [SEED] --lr 0.00005
```
python experiments.py oai --name XtoY_FC50_Opt1 --exp train_X_to_y --fc_layers 10 50 50 1 --y_fc_name fc4 --seed 1 --lr 0.00005 --use_small_subset

##### 4b. Standard, No Bottleneck
```
python train.py --name XtoY_Opt[ID] --exp train_X_to_y --fc_layers 1 --y_fc_name fc1 --lr 0.00005 --seed [SEED]
```
python experiments.py oai --name XtoY_Opt1 --exp train_X_to_y --fc_layers 1 --y_fc_name fc1 --lr 0.00005 --seed 1 --use_small_subset

##### 5. Multitask
```
python train.py --name XtoCY_C0.1_FC50_Opt[ID] --exp train_X_to_Cy --fc_layers 10 50 50 11 --y_fc_name fc4 --C_fc_name fc4 --C_weight 0.1 --seed [SEED] --lr 0.0005
```
python experiments.py oai --name XtoCY_C0.1_FC50_Opt1 --exp train_X_to_Cy --fc_layers 10 50 50 11 --y_fc_name fc4 --C_fc_name fc4 --C_weight 0.1 --seed 1 --lr 0.0005 --use_small_subset

##### 6. Standard [Probe]
```
python train.py --name AProbes_Conv4_XtoCtoY_Lambda0_FC50_Opt[ID] --exp train_X_to_C_with_freezed_convs --fc_layers 10 --C_weight 0.1 --lr 0.0005 --pretrained outputs/XtoCtoY_Lambda0_FC50_Opt[ID]/model_weights.pth --input_conv_layers conv4 --pretrained_exclude_vars fc1 fc2 fc3 fc4 --conv_layers_before_end_to_unfreeze 0 --C_fc_name fc1 --seed [SEED]
```
python experiments.py oai --name AProbes_Conv4_XtoCtoY_Lambda0_FC50_Opt1 --exp train_X_to_C_with_freezed_convs --fc_layers 10 --C_weight 0.1 --lr 0.0005 --pretrained outputs/XtoCtoY_Lambda0_FC50_Opt[ID]/model_weights.pth --input_conv_layers conv4 --pretrained_exclude_vars fc1 fc2 fc3 fc4 --conv_layers_before_end_to_unfreeze 0 --C_fc_name fc1 --seed 1 --use_small_subset

### b. General results (Figure 2)

##### 1. Joint Models With Varying Lambda
To run Joint models of various lambda, replace [LAMBDA] and [LEARN_RATE] with appropriate values below. In our experiments we used [LAMBDA] in {0.001, 0.01, 0.1, 1} and the corresponding [LEARN_RATE] in {0.00005, 0.00005, 0.00005, 0.0005}. EG. Lambda of 0.1 has learning rate 0.00005 while lambda of 1 has learning rate 0.0005.
```
python train.py --name XtoCtoY_Lambda[LAMBDA]_FC50_Opt[ID] --exp Joint --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight [LAMBDA] --seed [SEED] --lr [LEARN_RATE]
```

For the data efficiency experiments below, replace [PROPORTION] with {0.1, 0.2, 0.5} to train on different data proportions.
##### 2. (Data Efficiency) Independent
```
python train.py --name XtoC_C0.1_DataEff[PROPORTION]_[ID] --exp Concept_XtoC --fc_layers 10 --C_fc_name fc1 --C_weight 0.1 --lr 0.0005 --data_proportion [PROPORTION]

python train.py --name OracleCtoY_ontop_XtoC_MLP_FC50,50,1_Regul0_DataEff[PROPORTION]_[ID] --exp Independent_CtoY --fc_layers 10 --C_fc_name fc1 --C_weight 0.1 --lr 0.0005 --eval_model X_to_C --oracle_C_to_y_model mlp --oracle_C_to_y_model_params "{ \"hidden_layer_sizes\": [50,50], \"alpha\": 0 }" --pretrained outputs/XtoC_C0.1_DataEff[PROPORTION]_[ID]/model_weights.pth --data_proportion [PROPORTION] --seed [SEED]
```

##### 3. (Data Efficiency) Sequential
```
python train.py --name XtoChat_ChatToY_PRE-XtoC_C0.1_DataEff[PROPORTION]_[ID]-_FC50_DataEff[PROPORTION]_[ID] --seed [SEED] --exp Sequential_CtoY --pretrained outputs/XtoC_C0.1_DataEff[PROPORTION]_[ID]/model_weights.pth --front_fc_layers_to_freeze 1 --fc_layers 10 50 50 1 --C_fc_name fc1 --y_fc_name fc4 --C_weight 0.1 --lr 0.0005 --data_proportion [PROPORTION]
```

##### 4. (Data Efficiency) Joint
```
python train.py --name XtoCtoY_Lambda1_FC50_DataEffSeed[PROPORTION]_[ID] --seed [SEED] --exp Joint --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --lr 0.0005 --data_proportion [PROPORTION]
```

##### 5. (Data Efficiency) Standard
```
python train.py --name XtoY_FC50_DataEffSeed[PROPORTION]_[ID] --seed [SEED] --exp train_X_to_y --fc_layers 10 50 50 1 --y_fc_name fc4 --lr 0.00005 --data_proportion [PROPORTION]
```

### c. TTI results (Figure 4)

##### 1. Control
```
# Nonlinear
python train.py --name XtoCtoY_Lambda0.01_FC50_Opt[ID]_TTISeed_OrderBestImprovValBased --exp TTI --test_time_intervention_model X_to_C_to_y --test_time_intervention_method ordered --intervention_order 3 7 4 8 6 0 1 2 5 9 --pretrained outputs/XtoCtoY_Lambda0.01_FC50_Opt[ID]/model_weights.pth --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 0.01 --seed [SEED]
```

##### 2. Joint
```
# Nonlinear
python train.py --name XtoCtoY_Lambda1_FC50_Opt[ID]_TTISeed_OrderBestImprovValBased --exp TTI --test_time_intervention_model X_to_C_to_y --test_time_intervention_method ordered --intervention_order 3 2 5 8 0 7 4 1 6 9 --pretrained outputs/XtoCtoY_Lambda1_FC50_Opt[ID]/model_weights.pth --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed [SEED]

# Linear
python train.py --name XtoCtoY_Lambda1_Opt[ID]_TTISeed_OrderBestImprov --exp TTI --test_time_intervention_model X_to_C_to_y --test_time_intervention_method ordered --intervention_order 2 3 4 7 1 8 9 0 6 5 --pretrained outputs/XtoCtoY_Lambda1_Opt[ID]/model_weights.pth --fc_layers 10 1 --y_fc_name fc2 --C_fc_name fc1 --C_weight 1 --seed [SEED]
```

##### 3. Sequential
```
# Nonlinear
python train.py --name XtoChat_ChatToY_PRE-XtoC_C0.1_FC50_Opt[ID]-_Opt[ID]_TTISeed_OrderBestImprov --exp TTI --test_time_intervention_model X_to_Chat__Chat_to_y --test_time_intervention_method ordered --intervention_order 3 2 8 7 5 0 4 6 9 1 --pretrained outputs/XtoChat_ChatToY_PRE-XtoC_C0.1_FC50_Opt[ID]-_Opt[ID]/model_weights.pth --front_fc_layers_to_freeze 0 --fc_layers 10 50 50 1 --C_fc_name fc1 --y_fc_name fc4 --C_weight 0.1 --seed [SEED]

# Linear
python train.py --name XtoChat_ChatToY_PRE-XtoC_C0.1_Opt[ID]-_Opt[ID]_TTISeed_OrderBestImprov --exp TTI --test_time_intervention_model X_to_Chat__Chat_to_y --test_time_intervention_method ordered --intervention_order 3 2 8 7 5 0 4 6 9 1 --pretrained outputs/XtoChat_ChatToY_PRE-XtoC_C0.1_Opt[ID]-_Opt[ID]/model_weights.pth --front_fc_layers_to_freeze 0 --fc_layers 10 1 --C_fc_name fc1 --y_fc_name fc2 --C_weight 0.1 --seed [SEED]
```

##### 4. Independent
```
# Nonlinear
python train.py --name XtoChat_OracleAToY_PRE-XtoC_C0.1_FC50_Opt[ID]-_Opt[ID]_TTI_OrderBestImprov --exp TTI --test_time_intervention_model X_to_Chat__OracleA_to_y --test_time_intervention_method ordered --intervention_order 3 2 8 7 5 0 4 6 9 1 --pretrained outputs/XtoChat_ChatToY_PRE-XtoC_C0.1_FC50_Opt[ID]-_Opt[ID]/model_weights.pth --front_fc_layers_to_freeze 0 --fc_layers 10 50 50 1 --C_fc_name fc1 --y_fc_name fc4 --C_weight 0.1 --seed [SEED] --oracle_C_to_y_model mlp --oracle_C_to_y_model_params "{ \"hidden_layer_sizes\": [50,50], \"alpha\": 0 }"

# Linear
python train.py --name XtoChat_OracleAToY_PRE-XtoC_C0.1_Opt[ID]-_Opt[ID]_TTI_OrderBestImprov --exp TTI --test_time_intervention_model X_to_Chat__OracleA_to_y --test_time_intervention_method ordered --intervention_order 3 2 8 7 5 0 4 6 9 1 --pretrained outputs/XtoChat_ChatToY_PRE-XtoC_C0.1_Opt[ID]-_Opt[ID]/model_weights.pth --front_fc_layers_to_freeze 0 --fc_layers 10 1 --C_fc_name fc1 --y_fc_name fc2 --C_weight 0.1 --seed [SEED] --oracle_C_to_y_model lr
```

## Results and plots
To obtain the experimental results and generate the plots in the paper, run 
```
python plots.py
```