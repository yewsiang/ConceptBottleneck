"""
Automatically extract best config and epoch and retrain the model on both train + val sets
"""
import os
import subprocess
import re
import argparse
from hyperparam_checking import find_best_config_hyperparam_tune, find_best_perf


def retrain(hyperparam_tune_path, save_path, all_model_types=[], all_lambdas=[], shots=[], adversarial=False):
    """
    Retrain only the best hyperparam config for each model type on both train + val sets
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    best_records = find_best_config_hyperparam_tune(hyperparam_tune_path)
    all_data_dir = []
    if shots:
        for n_shots in shots:
            all_data_dir.append('class_attr_data_10_%d_shot' % n_shots)
    else:
        all_data_dir.append('class_attr_data_10')

    for data_dir in all_data_dir:
        for model_type, v in best_records.items():
            _, epoch, config_dir = v
            if all_model_types and not any([t in model_type for t in all_model_types]):
                continue
            model_path = os.path.join(config_dir, '%d_model.pth' % epoch)
            log_dir = os.path.join(save_path, config_dir.split('/')[-1] + '_' + data_dir)
            command = 'python train_sigmoid.py -log_dir %s -e 1000 -optimizer sgd -pretrained -use_aux %s'
            if 'simple_finetune' in model_path:
                model_suffix = ''
            else:
                lambda_val = float(re.findall(r"attr_loss_weight_\d*\.\d+", config_dir)[0].split('_')[-1])
                if any([t in model_type for t in ['multitask', 'end2end']]) and (all_lambdas and lambda_val not in all_lambdas):
                    continue
                model_suffix = '-use_attr -weighted_loss multiple -data_dir %s -n_attributes 112 -attr_loss_weight %.3f -normalize_loss' % (data_dir, lambda_val)
                if 'relu' in hyperparam_tune_path:
                    model_suffix += ' -use_relu'
                    print("USE RELU")

                if 'end2end' in model_path:
                    model_suffix += ' -end2end'
                elif 'bottleneck' in model_path:
                    model_suffix += ' -bottleneck'
                elif 'onlyAttr' in model_path:
                    model_suffix += ' -no_img'
            scheduler_step = int(re.findall(r"scheduler_step_\d*", config_dir)[0].split('_')[-1])
            weight_decay = float(re.findall(r"weight_decay_\d*\.\d+", config_dir)[0].split('_')[-1])
            lr = float(re.findall(r"lr_\d*\.\d+", config_dir)[0].split('_')[-1])
 
            model_suffix = model_suffix + " -batch_size %d -weight_decay %f -lr %f -scheduler_step %d" % (64, weight_decay, lr, scheduler_step)    
            command = command % (log_dir, model_suffix)
            if not shots: #also train on val set
                command += (' -ckpt %s' % model_path)
            if adversarial:
                command += ' -image_dir CUB_adversarial/CUB_fixed/train/'
            print(command)
            subprocess.run([command])

def run_inference(retrain_path, model_types=[], all_lambdas=[], feature_group=False, sequential=False):
    """
    Assuming there is only one model of each (model type, lambda value) in retrain_path
    Run inference on test set using the best epoch obtained from retraining
    if model_type is specified, then only run inference for that model type
    """
    for config in os.listdir(retrain_path):
        config_dir = os.path.join(retrain_path, config)
        if not os.path.isdir(config_dir):
            continue
        if 'bottleneck' in config:
            model_type = 'bottleneck'
        elif 'end2end' in config:
            model_type = 'end2end'
        elif 'use_attr' in config and 'onlyAttr' not in config:
            model_type = 'multitask'
        elif 'onlyAttr' not in config:
            model_type = 'simple_finetune'
        else:
            model_type = 'onlyAttr'
        if model_types and model_type not in model_types:
            continue
        all_val_acc = find_best_perf(os.path.join(config_dir, 'log.txt'))
        epoch = all_val_acc.index(max(all_val_acc))
        #epoch = round(epoch, -1) - 20
        if epoch < 0:
            print(config_dir, ' has not started training')
        print(epoch, '\t', config)
        model_path = os.path.join(config_dir, '%d_model.pth' % epoch)
        if 'attr_loss_weight' in model_path:
            lambda_val = float(re.findall(r"attr_loss_weight_\d*\.\d+", config_dir)[0].split('_')[-1])
        else:
            lambda_val = 1
        if any([t in model_types for t in ['multitask', 'end2end']]) and (all_lambdas and lambda_val not in all_lambdas):
            continue
        if 'NEW_SIGMOID_MODEL' in retrain_path or 'NEW_MODEL' in retrain_path:
            command = 'python inference_sigmoid.py -model_dir %s -eval_data test' % model_path
        else:
            command = 'python inference.py -model_dir %s -eval_data test' % model_path
        if feature_group:
            command += ' -feature_group_results' 
        if 'use_attr' in model_path:
            command += ' -use_attr -n_attributes 112 -data_dir class_attr_data_10'
        if 'onlyAttr' in model_path:
            continue
        if 'bottleneck' in model_path:
            def find_onlyAttr_dir(retrain_path, model_path):
                if 'few_shots' in retrain_path:
                    n_shots = re.findall(r"\d+_shot", model_path)[0]
                    if sequential:
                        dir_name = [c for c in os.listdir(retrain_path) if 'onlyAttr_Ahat' in c and n_shots in c][0]
                    else:
                        dir_name = [c for c in os.listdir(retrain_path) if 'onlyAttr' in c and 'onlyAttr_Ahat' not in c and n_shots in c][0]                    
                else: 
                    if sequential:
                        dir_name = [c for c in os.listdir(retrain_path) if 'onlyAttr_Ahat' in c][0]
                    else:
                        dir_name = [c for c in os.listdir(retrain_path) if 'onlyAttr' in c and 'onlyAttr_Ahat' not in c][0]
                return os.path.join(retrain_path, dir_name)

            onlyAttr_dir = find_onlyAttr_dir(retrain_path, model_path)
            val_acc = find_best_perf(os.path.join(onlyAttr_dir, 'log.txt'))
            model2_path = os.path.join(onlyAttr_dir, '%d_model.pth' % (val_acc.index(max(val_acc))))
            config_dir = os.path.join(retrain_path, config)
            command += (' -model_dir2 %s -bottleneck' % model2_path)
            if 'onlyAttr_Ahat' not in model2_path:
                command += ' -use_sigmoid'
        if 'adversarial' in retrain_path:
            command += ' -image_dir CUB_adversarial/CUB_fixed/test/'
        subprocess.run([command])
    #TODO: write test inference results to a separate folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-save_path', default=None, help='where the trained models are saved')
    parser.add_argument('-results_path', default=None, help='where to parse for the best performance')
    args = parser.parse_args()
    #retrain(args.results_path, args.save_path, all_model_types=['bottleneck', 'end2end'], all_lambdas=['0.01'], shots=[])
    run_inference(args.results_path, model_types=['end2end'], all_lambdas=[0.001], sequential=True)
