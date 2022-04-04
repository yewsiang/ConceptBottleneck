
# Concept Bottleneck Models - OAI Dataset

## Dataset preprocessing

The NIH Osteoarthritis Initiative (OAI) dataset requires an application for data access, so we are unable to provide the raw data here. To access that data, please first obtain data access permission from the [Osteoarthritis Initiative](https://nda.nih.gov/oai/), and then refer to this [Github repository](https://github.com/epierson9/pain-disparities) for data processing code. If you use it, please cite the Pierson et al. paper corresponding to that repository as well.

## Experiments

1) Update the `BASE_DIR` and `DATA_DIR` directories in `config.py` to point to the base directory of the repository and the directory of the above preprocessed code.

2) Run experiments and replace `[SEED]` with the following seeds {603844, 841538, 620523, 217182, 84137}, and `[ID]` with the id of the seed. EG. The seed of 841538 has id 2.

### a. Task and concept tables (Table 1 and 2)

##### 1. Independent
```
python experiments.py oai Concept_XtoC --name XtoC_C0.1_FC50_Opt[ID] --fc_layers 10 --C_fc_name fc1 --C_weight 0.1 --seed [SEED] --lr 0.0005

python experiments.py oai Independent_CtoY --name OracleCtoY_ontop_XtoC_MLP_FC50,50,1_Regul0 --fc_layers 10 --C_fc_name fc1 --C_weight 0.1 --lr 0.0005 --eval_model X_to_C --oracle_C_to_y_model mlp --oracle_C_to_y_model_params "{ \"hidden_layer_sizes\": [50,50], \"alpha\": 0 }" \
--pretrained outputs/XtoC_C0.1_FC50_Opt[ID]/model_weights.pth --seed [SEED]
```

##### 2. Sequential
```
python experiments.py oai Sequential_CtoY --name XtoChat_ChatToY_PRE-XtoC_C0.1_FC50_Opt[ID]-_Opt[ID] --pretrained outputs/XtoC_C0.1_FC50_Opt[ID]/model_weights.pth --front_fc_layers_to_freeze 1 --fc_layers 10 50 50 1 --C_fc_name fc1 --y_fc_name fc4 --C_weight 0.1 --seed [SEED] --lr 0.0005
```

##### 3. Joint
```
python experiments.py oai Joint --name XtoCtoY_Lambda1_FC50_Opt[ID] --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed [SEED] --lr 0.0005
```

##### 4. Standard
```
python experiments.py oai train_X_to_y --name XtoY_FC50_Opt[ID] --fc_layers 10 50 50 1 --y_fc_name fc4 --seed [SEED] --lr 0.00005
```

##### 4b. Standard, No Bottleneck
```
python experiments.py oai train_X_to_y --name XtoY_Opt[ID] --fc_layers 1 --y_fc_name fc1 --lr 0.00005 --seed [SEED]
```

##### 5. Multitask
```
python experiments.py oai train_X_to_Cy --name XtoCY_C0.1_FC50_Opt[ID] --fc_layers 10 50 50 11 --y_fc_name fc4 --C_fc_name fc4 --C_weight 0.1 --seed [SEED] --lr 0.0005
```

##### 6. Standard [Probe]
```
python experiments.py oai train_X_to_C_with_freezed_convs --name AProbes_Conv4_XtoCtoY_Lambda0_FC50_Opt[ID] --fc_layers 10 --C_weight 0.1 --lr 0.0005 --pretrained outputs/XtoCtoY_Lambda0_FC50_Opt[ID]/model_weights.pth --input_conv_layers conv4 --pretrained_exclude_vars fc1 fc2 fc3 fc4 --conv_layers_before_end_to_unfreeze 0 --C_fc_name fc1 --seed [SEED]
```

### b. General results (Figure 2)

##### 1. Joint Models With Varying Lambda
To run Joint models of various lambda, replace `[LAMBDA]` and `[LEARN_RATE]` with appropriate values below. In our experiments we used `[LAMBDA]` in {0.001, 0.01, 0.1, 1} and the corresponding `[LEARN_RATE]` in {0.00005, 0.00005, 0.00005, 0.0005}. 

EG. Lambda of 0.1 has learning rate 0.00005 while lambda of 1 has learning rate 0.0005.
```
python experiments.py oai Joint --name XtoCtoY_Lambda[LAMBDA]_FC50_Opt[ID] --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight [LAMBDA] --seed [SEED] --lr [LEARN_RATE]
```

For the data efficiency experiments below, replace `[PROPORTION]` with {0.1, 0.2, 0.5} to train on different data proportions.
##### 2. (Data Efficiency) Independent
```
python experiments.py oai Concept_XtoC --name XtoC_C0.1_DataEff[PROPORTION]_[ID] --fc_layers 10 --C_fc_name fc1 --C_weight 0.1 --lr 0.0005 --data_proportion [PROPORTION]

python experiments.py oai Independent_CtoY --name OracleCtoY_ontop_XtoC_MLP_FC50,50,1_Regul0_DataEff[PROPORTION]_[ID] --fc_layers 10 --C_fc_name fc1 --C_weight 0.1 --lr 0.0005 --eval_model X_to_C --oracle_C_to_y_model mlp --oracle_C_to_y_model_params "{ \"hidden_layer_sizes\": [50,50], \"alpha\": 0 }" --pretrained outputs/XtoC_C0.1_DataEff[PROPORTION]_[ID]/model_weights.pth --data_proportion [PROPORTION] --seed [SEED]
```

##### 3. (Data Efficiency) Sequential
```
python experiments.py oai Sequential_CtoY --name XtoChat_ChatToY_PRE-XtoC_C0.1_DataEff[PROPORTION]_[ID]-_FC50_DataEff[PROPORTION]_[ID] --seed [SEED] --pretrained outputs/XtoC_C0.1_DataEff[PROPORTION]_[ID]/model_weights.pth --front_fc_layers_to_freeze 1 --fc_layers 10 50 50 1 --C_fc_name fc1 --y_fc_name fc4 --C_weight 0.1 --lr 0.0005 --data_proportion [PROPORTION]
```

##### 4. (Data Efficiency) Joint
```
python experiments.py oai Joint --name XtoCtoY_Lambda1_FC50_DataEffSeed[PROPORTION]_[ID] --seed [SEED] --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --lr 0.0005 --data_proportion [PROPORTION]
```

##### 5. (Data Efficiency) Standard
```
python experiments.py oai train_X_to_y --name XtoY_FC50_DataEffSeed[PROPORTION]_[ID] --seed [SEED] --fc_layers 10 50 50 1 --y_fc_name fc4 --lr 0.00005 --data_proportion [PROPORTION]
```

### c. TTI results (Figure 4)

##### 0. Determine intervention ordering
For each of the model types below, run the following script to determine the order of intervention on the concepts. For example, we show an example for the Joint model.
```
python experiments.py oai TTI --test_time_intervention_analysis --name XtoCtoY_Lambda1_FC50_Opt1_TTIAnalysis --test_time_intervention_model X_to_C_to_y --test_time_intervention_method random --pretrained outputs/XtoCtoY_Lambda1_FC50_Opt1/model_weights.pth --sampling uniform --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed [SEED]
```

It is possible to run a different intervention ordering for each different seed, but we will use the ordering determined by only the first seed to reduce time. We fill in the argument of `intervention_order` with the ordering determined from the script above. For the nonlinear Joint model, the ordering is `3 2 5 8 0 7 4 1 6 9`. 

##### 1. Control
```
# Nonlinear
python experiments.py oai TTI --name XtoCtoY_Lambda0.01_FC50_Opt[ID]_TTISeed_OrderBestImprovValBased --test_time_intervention_model X_to_C_to_y --test_time_intervention_method ordered --intervention_order 3 7 4 8 6 0 1 2 5 9 --pretrained outputs/XtoCtoY_Lambda0.01_FC50_Opt[ID]/model_weights.pth --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 0.01 --seed [SEED]
```

##### 2. Joint
```
# Nonlinear
python experiments.py oai TTI --name XtoCtoY_Lambda1_FC50_Opt[ID]_TTISeed_OrderBestImprovValBased --test_time_intervention_model X_to_C_to_y --test_time_intervention_method ordered --intervention_order 3 2 5 8 0 7 4 1 6 9 --pretrained outputs/XtoCtoY_Lambda1_FC50_Opt[ID]/model_weights.pth --fc_layers 10 50 50 1 --y_fc_name fc4 --C_fc_name fc1 --C_weight 1 --seed [SEED]

# Linear
python experiments.py oai TTI --name XtoCtoY_Lambda1_Opt[ID]_TTISeed_OrderBestImprov --test_time_intervention_model X_to_C_to_y --test_time_intervention_method ordered --intervention_order 2 3 4 7 1 8 9 0 6 5 --pretrained outputs/XtoCtoY_Lambda1_Opt[ID]/model_weights.pth --fc_layers 10 1 --y_fc_name fc2 --C_fc_name fc1 --C_weight 1 --seed [SEED]
```

##### 3. Sequential
```
# Nonlinear
python experiments.py oai TTI --name XtoChat_ChatToY_PRE-XtoC_C0.1_FC50_Opt[ID]-_Opt[ID]_TTISeed_OrderBestImprov --test_time_intervention_model X_to_Chat__Chat_to_y --test_time_intervention_method ordered --intervention_order 3 2 8 7 5 0 4 6 9 1 --pretrained outputs/XtoChat_ChatToY_PRE-XtoC_C0.1_FC50_Opt[ID]-_Opt[ID]/model_weights.pth --front_fc_layers_to_freeze 0 --fc_layers 10 50 50 1 --C_fc_name fc1 --y_fc_name fc4 --C_weight 0.1 --seed [SEED]

# Linear
python experiments.py oai TTI --name XtoChat_ChatToY_PRE-XtoC_C0.1_Opt[ID]-_Opt[ID]_TTISeed_OrderBestImprov --test_time_intervention_model X_to_Chat__Chat_to_y --test_time_intervention_method ordered --intervention_order 3 2 8 7 5 0 4 6 9 1 --pretrained outputs/XtoChat_ChatToY_PRE-XtoC_C0.1_Opt[ID]-_Opt[ID]/model_weights.pth --front_fc_layers_to_freeze 0 --fc_layers 10 1 --C_fc_name fc1 --y_fc_name fc2 --C_weight 0.1 --seed [SEED]
```

##### 4. Independent
```
# Nonlinear
python experiments.py oai TTI --name XtoChat_OracleAToY_PRE-XtoC_C0.1_FC50_Opt[ID]-_Opt[ID]_TTI_OrderBestImprov --test_time_intervention_model X_to_Chat__OracleA_to_y --test_time_intervention_method ordered --intervention_order 3 2 8 7 5 0 4 6 9 1 --pretrained outputs/XtoChat_ChatToY_PRE-XtoC_C0.1_FC50_Opt[ID]-_Opt[ID]/model_weights.pth --front_fc_layers_to_freeze 0 --fc_layers 10 50 50 1 --C_fc_name fc1 --y_fc_name fc4 --C_weight 0.1 --seed [SEED] --oracle_C_to_y_model mlp --oracle_C_to_y_model_params "{ \"hidden_layer_sizes\": [50,50], \"alpha\": 0 }"

# Linear
python experiments.py oai TTI --name XtoChat_OracleAToY_PRE-XtoC_C0.1_Opt[ID]-_Opt[ID]_TTI_OrderBestImprov --test_time_intervention_model X_to_Chat__OracleA_to_y --test_time_intervention_method ordered --intervention_order 3 2 8 7 5 0 4 6 9 1 --pretrained outputs/XtoChat_ChatToY_PRE-XtoC_C0.1_Opt[ID]-_Opt[ID]/model_weights.pth --front_fc_layers_to_freeze 0 --fc_layers 10 1 --C_fc_name fc1 --y_fc_name fc2 --C_weight 0.1 --seed [SEED] --oracle_C_to_y_model lr
```
