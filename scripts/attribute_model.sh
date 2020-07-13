
# Concept model
python3 src/experiments.py cub Concept_XtoC --seed 1 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 32 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck
python3 src/experiments.py cub Concept_XtoC --seed 2 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 32 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck
python3 src/experiments.py cub Concept_XtoC --seed 3 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 32 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck

# Independent model
python src/experiments.py cub Independent_CtoY --seed 1 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -no_img -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 10
python src/experiments.py cub Independent_CtoY --seed 2 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -no_img -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 10
python src/experiments.py cub Independent_CtoY --seed 3 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -no_img -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 10

# Sequential model
python src/experiments.py cub Sequential_CtoY --seed 1 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -no_img -b 64 -weight_decay 0.00004 -lr 0.001 -scheduler_step 1000
python src/experiments.py cub Sequential_CtoY --seed 2 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -no_img -b 64 -weight_decay 0.00004 -lr 0.001 -scheduler_step 1000
python src/experiments.py cub Sequential_CtoY --seed 3 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -no_img -b 64 -weight_decay 0.00004 -lr 0.001 -scheduler_step 1000

# Joint (Lambda = 0.01)
python3 src/experiments.py cub Joint --seed 1 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end
python3 src/experiments.py cub Joint --seed 2 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end
python3 src/experiments.py cub Joint --seed 3 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end

# Standard (Lambda = 0)
python3 src/experiments.py cub Standard --seed 1 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end
python3 src/experiments.py cub Standard --seed 2 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end
python3 src/experiments.py cub Standard --seed 3 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end

# Standard, no bottleneck
python3 src/experiments.py cub Standard --seed 1 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir CUB_processed/class_attr_data_10 -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20
python3 src/experiments.py cub Standard --seed 2 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir CUB_processed/class_attr_data_10 -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20
python3 src/experiments.py cub Standard --seed 3 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir CUB_processed/class_attr_data_10 -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20

# Multitask
python3 src/experiments.py cub Multitask --seed 1 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.00001 -scheduler_step 20
python3 src/experiments.py cub Multitask --seed 2 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.00001 -scheduler_step 20
python3 src/experiments.py cub Multitask --seed 3 -log_dir outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.00001 -scheduler_step 20
