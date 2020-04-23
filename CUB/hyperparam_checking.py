import os
import re
import sys
import datetime
import subprocess
from CUB.config import BASE_DIR


def find_early_stop_epoch(log_file, patience):
    all_best_epochs = find_all_best_epochs(log_file)
    #print("last best epoch:", all_best_epochs[-1])
    for i, epoch in enumerate(all_best_epochs):
        if i == len(all_best_epochs) - 1:
            return epoch
        if epoch > 500 and all_best_epochs[i+1] - epoch > patience:
            return epoch
    return None if not all_best_epochs else all_best_epochs[-1]

def find_best_config_retrain(path, patience=1000):
    best_records = dict()
    for config in os.listdir(path):
        config_path = os.path.join(path, config)
        if not os.path.isdir(config_path):
            continue
        #lambda_val = re.findall(r"attr_loss_weight_\d*\.\d+", config)[0].split('_')[-1]
        #if float(lambda_val) > 1:
        #    continue
        if 'end2end' in config:
            model_type = 'end2end'
        elif 'bottleneck' in config:
            model_type = 'bottleneck'
        elif 'onlyAttr' in config:
            model_type = 'onlyAttr'
        elif 'simple_finetune' in config:
            model_type = 'simple_finetune'
        else:
            model_type = 'multitask'
        log_file = os.path.join(config_path, 'log.txt')
        all_val_acc = find_best_perf(log_file)
        epoch = find_early_stop_epoch(log_file, patience)
        if epoch is None:
            continue
        print(config_path)
        print (model_type, epoch, all_val_acc[epoch], '\n')


if __name__ == "__main__":
    cmd = sys.argv[1]
    if 'invalid' in cmd:
        find_invalid_runs(BASE_DIR)
    else:
        results_dir = os.path.join(BASE_DIR, sys.argv[2])
        if 'retrain' in cmd:
           find_best_config_retrain(results_dir)
        else:
           find_best_config_hyperparam_tune(results_dir)
