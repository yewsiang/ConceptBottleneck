# Concept Bottleneck Models - CUB Dataset
## Dataset preprocessing
1) Download the CUB dataset from the official website: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html and save it into a 'datasets/' folder
2) Run *data_processing.py* to obtain train/ val/ test splits as well as to extract all relevant task and concept metadata into pickle files
3) Run preprocessing steps as found in *utils.py*
## Experiments
1) Update the BASE_DIR in the scripts (where applicable) to point to the base directory of the repository
2) Run *hyperparam_tune.py* to perform hyperparameter search for all types of models described in this paper
3) The commands for retraining on both train & val sets, as well as for running inference on the official test data can be found in *postprocessing.py*. Otherwise, inference using any trained model can be run separately using *inference.py* 

### a. Task and concept tables (Table 1 and 2)
To train on both train & val sets combined, include the flag '-ckpt [SOME_MODEL_CKPT]' when running *train.py*.
##### 1. Independent
Train the x -> c part:
```
python train.py -log_dir [LOG_DIR] -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir [DATA_DIR] -n_attributes 112 -normalize_loss -b 64 -weight_decay [WEIGHT_DECAY] -lr [LR] -scheduler_step [LR_DECAY_STEP] -bottleneck
```
Train the c -> y part:
```
python train.py -log_dir [LOG_DIR] -optimizer sgd -e 1000 -pretrained -use_aux -use_attr -data_dir [DATA_DIR] -n_attributes 112 -no_img -b 64 -weight_decay [WEIGHT_DECAY] -lr [LR] -scheduler_step [LR_DECAY_STEP] 
```
##### 2. Sequential
Train the x -> c part:
```
python train.py -log_dir [LOG_DIR] -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir [DATA_DIR] -n_attributes 112 -normalize_loss -b 64 -weight_decay [WEIGHT_DECAY] -lr [LR] -scheduler_step [LR_DECAY_STEP] -bottleneck
```
Extract the c logits from the first stage model and save the new data into train/ val/ test pickle files using create_logits_data() in *generate_new_data.py*.

(YS)
```
python CUB/generate_new_data.py ExtractConcepts 
```

Then train the c -> y part, with [DATA_DIR] set accordingly:
```
python train.py -log_dir [LOG_DIR] -optimizer sgd -e 1000 -pretrained -use_aux -use_attr -data_dir [DATA_DIR] -n_attributes 112 -no_img -b 64 -weight_decay [WEIGHT_DECAY] -lr [LR] -scheduler_step [LR_DECAY_STEP] 
```
##### 3. Joint
```
python train.py -log_dir [LOG_DIR] -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir [DATA_DIR] -n_attributes 112 -attr_loss_weight [LAMBDA] -normalize_loss -b 64 -weight_decay [WEIGHT_DECAY] -lr [LR] -scheduler_step [LR_DECAY_STEP] -end2end
```
To train a joint model with a sigmoid layer included between x -> c and c -> y, simply include '-use_sigmoid' flag.
##### 4. Standard
```
python train.py -log_dir [LOG_DIR] -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir [DATA_DIR] -n_attributes 112 -attr_loss_weight 0 -normalize_loss -b 64 -weight_decay [WEIGHT_DECAY] -lr [LR] -scheduler_step [LR_DECAY_STEP] -end2end
```
##### 4b. Standard, No Bottleneck
```
python train.py -log_dir [LOG_DIR] -e 1000 -optimizer sgd -pretrained -use_aux -data_dir [DATA_DIR] -b 64 -weight_decay [WEIGHT_DECAY] -lr [LR] -scheduler_step [LR_DECAY_STEP]
```
##### 5. Multitask
```
python train.py -log_dir [LOG_DIR] -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir [DATA_DIR] -n_attributes 112 -attr_loss_weight [LAMBDA] -normalize_loss -b 64 -weight_decay [WEIGHT_DECAY] -lr [LR] -scheduler_step [LR_DECAY_STEP]
```
##### 6. Standard [Probe]
First extract the logits from the desired layer of the trained standard model and save the new data into train/ val/ test pickle files using get_representation_linear_probe() in *generate_new_data.py*.

(YS)
```
python CUB/generate_new_data.py ExtractProbeRepresentations
```
Then use *linear_probe.py* for training and inference, with [DATA_DIR] set accordingly:
```
python linear_probe.py -log_dir [LOG_DIR] -data_dir [DATA_DIR] -n_attributes 112 -b 64 -e 1000 -weight_decay [WEIGHT_DECAY] -lr [LR] -scheduler_step [LR_DECAY_STEP]

python linear_probe.py -eval -model_dir [PATH_TO_TRAINED_MODEL]
```
### b. General results (Figure 2)
##### 1. Joint Models With Varying Lambda
To run Joint models with a different lambda value, replace [LAMBDA] in the command in section (a)(3), with the desired value. In our experiments we searched over [LAMBDA] in [0.001, 0.01, 0.1, 1].

##### 2. Data Efficiency

For the data efficiency experiments, first create smaller datasets based on a specified number of images (i.e. 1, 3, 7, 10, 15, 20 shots) sampled from each class using get_few_shot_data() in *generate_new_data.py*, and update the [DATA_DIR] in the commands in section (a) to point to the new pickle files correspondingly.

(YS)
```
python CUB/generate_new_data.py DataEfficiencySplits
```

### c. TTI results (Figure 4)

##### 1. Joint
```
python attribute_correction.py -model_dir [TRAINED_MODEL] -use_attr -mode random -n_trials 5 -use_invisible -class_level
```
To run on a joint model where a sigmoid layer was included between x -> c and c -> y, simply include '-use_sigmoid' flag.
##### 2. Sequential
```
python attribute_correction.py -model_dir [X_TO_C_TRAINED_MODEL] -model_dir2 [C_TO_Y_TRAINED_MODEL] -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level
```
##### 3. Independent
```
python attribute_correction.py -model_dir [X_TO_C_TRAINED_MODEL] -model_dir2 [C_TO_Y_TRAINED_MODEL] -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level
```
### d. Robustness results (Table 3)
Create the adversarial data where backgrounds are modified to include spurious correlation by running *gen_cub_synthetic.py*.
Model training commands are similar to the ones described in section (a), with an additional flag '-image_dir CUB_adversarial/CUB_fixed/train/' for training and '-image_dir CUB_adversarial/CUB_fixed/test/' for inference.
(YS)
```
python gen_cub_synthetic.py 
```