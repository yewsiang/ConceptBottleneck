# Concept Bottleneck Models - CUB Dataset
## Dataset preprocessing
1) Download the [official CUB dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) (`CUB_200_2011`), processed CUB data (`CUB_processed`), places365 dataset (`places365`) and pretrained Inception V3 models (`pretrained`) from our [Codalab worksheet](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2).   

OR) You can get `CUB_processed` from Step 1. above with the following steps
1) Run `data_processing.py` to obtain train/ val/ test splits as well as to extract all relevant task and concept metadata into pickle files. 
2) Run `generate_new_data.py` to obtain other versions of training data (class-level attributes, few-shot training, etc.) from the metadata.
## Experiments
1) Update the paths (e.g. `BASE_DIR`, `-log_dir`, `-out_dir`, `--model_path`) in the scripts (where applicable) to point to your dataset and outputs.
2) Run the scripts below to get the results for 1 seed. Change the seed values and corresponding file names to get results for more seeds.
3) All of the scripts together with different seeds are available in `scripts/experiments.sh`, read it to get a complete picture of the experiments. 
3) (Optional) Run `hyperparam_tune.py` to perform hyperparameter search for all types of models described in this paper 

### a. Task and concept tables (Table 1 and 2)
##### 1. Independent
Train the x -> c model and extract the predicted c logits:
```
python3 src/experiments.py cub Concept_XtoC --seed 1 -ckpt 1 -log_dir ConceptModel__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck
python3 src/CUB/generate_new_data.py ExtractConcepts --model_path ConceptModel__Seed1/outputs/best_model_1.pth --data_dir CUB_processed/class_attr_data_10 --out_dir ConceptModel1__PredConcepts
```
Train the c -> y model:
```
python3 src/experiments.py cub Independent_CtoY --seed 1 -log_dir IndependentModel_WithVal___Seed1/outputs/ -e 500 -optimizer sgd -use_attr -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 1000 
```
Inference on test set when you have 3 seeds:
```
python3 src/CUB/inference.py -model_dirs ConceptModel__Seed1/outputs/best_model_1.pth ConceptModel__Seed2/outputs/best_model_2.pth ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 IndependentModel_WithVal___Seed1/outputs/best_model_1.pth IndependentModel_WithVal___Seed2/outputs/best_model_2.pth IndependentModel_WithVal___Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -bottleneck -use_sigmoid -log_dir IndependentModel__WithValSigmoid/outputs
```
##### 2. Sequential
Train the x -> c model:

- There is nothing to run, just use the x -> c models from the `Independent` model. 

Then train the c -> y model:
```
python3 src/experiments.py cub Sequential_CtoY --seed 1 -log_dir SequentialModel_WithVal__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -data_dir ConceptModel1__PredConcepts -n_attributes 112 -no_img -b 64 -weight_decay 0.00004 -lr 0.001 -scheduler_step 1000 
```
Inference on test set when you have 3 seeds:
```
python3 src/CUB/inference.py -model_dirs ConceptModel__Seed1/outputs/best_model_1.pth ConceptModel__Seed2/outputs/best_model_2.pth ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 SequentialModel_WithVal__Seed1/outputs/best_model_1.pth SequentialModel_WithVal__Seed2/outputs/best_model_2.pth SequentialModel_WithVal__Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -bottleneck -feature_group_results -log_dir SequentialModel__WithVal/outputs
```
##### 3. Joint
```
python3 src/experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint0.01Model__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end
```
Inference on test set when you have 3 seeds:
```
python3 src/CUB/inference.py -model_dirs Joint0.001Model_Seed1/outputs/best_model_1.pth Joint0.001Model_Seed2/outputs/best_model_2.pth Joint0.001Model_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -log_dir Joint0.001Model/outputs
```
##### 3b. Joint model with a sigmoid layer included between x -> c and c -> y:
```
python3 src/experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint0.01SigmoidModel__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end -use_sigmoid
```
##### 4. Standard
```
python3 src/experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint0Model_Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end
```
Inference on test set when you have 3 seeds:
```
python3 src/CUB/inference.py -model_dirs Joint0Model_Seed1/outputs/best_model_1.pth Joint0Model_Seed2/outputs/best_model_2.pth Joint0Model_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -log_dir Joint0Model/outputs
```
##### 4b. Standard, No Bottleneck
```
python3 src/experiments.py cub Standard --seed 1 -ckpt 1 -log_dir StandardNoBNModel_Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir CUB_processed/class_attr_data_10 -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20
```
Inference on test set when you have 3 seeds:
```
python3 src/CUB/inference.py -model_dirs StandardNoBNModel_Seed1/outputs/best_model_1.pth StandardNoBNModel_Seed2/outputs/best_model_2.pth StandardNoBNModel_Seed3/outputs/best_model_3.pth -eval_data test -data_dir CUB_processed/class_attr_data_10 -log_dir StandardNoBNModel/outputs
```
##### 5. Multitask
```
python3 src/experiments.py cub Multitask --seed 1 -ckpt 1 -log_dir MultitaskModel_Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20
```
Inference on test set when you have 3 seeds:
```
python3 src/CUB/inference.py -model_dirs MultitaskModel_Seed1/outputs/best_model_1.pth MultitaskModel_Seed2/outputs/best_model_2.pth MultitaskModel_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -log_dir MultitaskModel/outputs
```
##### 6. Standard [Probe]
First extract the logits from the desired layer of the trained standard model and save the new data into train/ val/ test pickle files using get_representation_linear_probe() in `generate_new_data.py`.
```
python3 src/CUB/generate_new_data.py ExtractProbeRepresentations --model_path Joint0Model_Seed1/outputs/best_model_1.pth --layer_idx -1 --data_dir CUB_processed/class_attr_data_10 --out_dir Joint0Model_Seed1_ExtractProbeRep
```
Then use `probe.py` for training and inference:
```
python3 src/CUB/probe.py -data_dir Joint0Model_Seed1_ExtractProbeRep -n_attributes 112 -log_dir Joint0Model_Seed1_LinearProbe/outputs -lr 0.001 -scheduler_step 1000 -weight_decay 0.00004
```
Inference on test set when you have 3 seeds:
```
python3 src/CUB/probe.py -data_dirs Joint0Model_Seed1_ExtractProbeRep Joint0Model_Seed2_ExtractProbeRep Joint0Model_Seed3_ExtractProbeRep -log_dir Joint0Model_LinearProbe -eval -model_dirs Joint0Model_Seed1_LinearProbe/outputs/best_model.pth Joint0Model_Seed2_LinearProbe/outputs/best_model.pth Joint0Model_Seed3_LinearProbe/outputs/best_model.pth
```
### b. General results (Figure 2)
##### 1. Joint Models With Varying Lambda
To run Joint models with different lambda values, simply replace `-attr_loss_weight` in the previous scripts.

##### 2. Data Efficiency

For the data efficiency experiments, first create smaller datasets based on a specified number of images (i.e. 1, 3, 7, 10, 15 shots) sampled from each class using `generate_new_data.py`. The following is for N = 1:
```
python3 src/CUB/generate_new_data.py DataEfficiencySplits --n_samples 1 --out_dir DataEfficiencySplits_N1 --splits_dir CUB_processed/class_attr_data_10
```
Model training and inference: 
- Commands are similar to the ones described in section (a), except that we replace the data directories (`-data_dir`). 
Read `scripts/experiments.sh` for the exact scripts.

### c. TTI results (Figure 4)
These experiments show the performance gain when we randomly select a group of concepts one by one, and replace all predicted concepts in that group with their true labels (or the 5th and 95th percentiles of their predicted logit values where appropriate).
##### 1. Joint
```
python3 src/CUB/tti.py -model_dirs Joint0.01Model__Seed1/outputs/best_model_1.pth Joint0.01Model__Seed2/outputs/best_model_2.pth -use_attr -mode random -n_trials 5 -use_invisible -class_level -data_dir2 CUB -data_dir CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01Model
```
##### 1b. Joint with Sigmoid
```
python3 src/CUB/tti.py -model_dirs Joint0.01SigmoidModel__Seed1/outputs/best_model_1.pth Joint0.01SigmoidModel__Seed2/outputs/best_model_2.pth Joint0.01SigmoidModel__Seed3/outputs/best_model_3.pth -use_sigmoid -use_attr -mode random -n_trials 5 -use_invisible -class_level -data_dir2 CUB -data_dir CUB_processed/class_attr_data_10 -log_dir TTI__Joint0.01SigmoidModel
```
##### 2. Sequential
```
python3 src/CUB/tti.py -model_dirs ConceptModel__Seed1/outputs/best_model_1.pth ConceptModel__Seed2/outputs/best_model_2.pth ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 SequentialModel_WithVal__Seed1/outputs/best_model_1.pth SequentialModel_WithVal__Seed2/outputs/best_model_2.pth SequentialModel_WithVal__Seed3/outputs/best_model_3.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 CUB -data_dir CUB_processed/class_attr_data_10 -log_dir TTI__SequentialModel_WithVal
```
##### 3. Independent
```
python3 src/CUB/tti.py -model_dirs ConceptModel__Seed1/outputs/best_model_1.pth ConceptModel__Seed2/outputs/best_model_2.pth ConceptModel__Seed3/outputs/best_model_3.pth -model_dirs2 IndependentModel_WithVal___Seed1/outputs/best_model_1.pth IndependentModel_WithVal___Seed2/outputs/best_model_2.pth IndependentModel_WithVal___Seed3/outputs/best_model_3.pth -use_attr -bottleneck -mode random -n_trials 5 -use_invisible -class_level -data_dir2 CUB -data_dir CUB_processed/class_attr_data_10 -use_sigmoid -log_dir TTI__IndependentModel_WithValSigmoid
```
### d. Robustness results (Table 3)
Create the adversarial data where backgrounds are modified to include spurious correlation by running `gen_cub_synthetic.py`.
```
python3 src/CUB/gen_cub_synthetic.py --cub_dir CUB_200_2011 --places_dir places365 --out_dir AdversarialData
```
Model training and inference: 
- Commands are similar to the ones described in section (a), with an additional flag `-image_dir CUB_adversarial/CUB_fixed/train/` for training and `-image_dir CUB_adversarial/CUB_fixed/test/` for inference. Read `scripts/experiments.sh` for the exact scripts.
