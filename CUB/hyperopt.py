"""
Tune hyperparameters for end2end and multitask models with different lambda values
"""
import os
import sys
import argparse
import subprocess

BASE_DIR = ''
DATA_DIR = 'class_attr_data_10'
N_ATTR = 112
USE_RELU = False
USE_SIGMOID = False

all_lr = [0.01, 0.001]
all_optimizer = ['SGD'] #, 'RMSprop']
all_batch_size = [64]
all_lambda_val = [0.001, 0.01, 0.1, 0, 1]
all_scheduler_step = [1000, 20, 10, 15] # large scheduler step = constant lr
all_weight_decay = [0.0004, 0.00004]
all_model_type = ['simple_finetune', 'onlyAttr', 'bottleneck', 'multitask', 'end2end']


all_configs = [{'model_type': m, 'lr': lr, 'batch_size': b, 'optimizer': o, 'lambda': l, 'scheduler_step': s, 'weight_decay': w}
               for m in all_model_type for lr in all_lr for b in all_batch_size for o in all_optimizer for l in all_lambda_val for s in all_scheduler_step for w in all_weight_decay]
BASE_COMMAND = 'python train.py -e 300 -pretrained -use_aux %s'

def launch_job(config, save_dir):
    save_path = os.path.join(BASE_DIR, save_dir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if config['model_type'] not in ['multitask', 'end2end']:
        if config['lambda'] != 1:
            return

    if config['model_type'] == 'simple_finetune':
        model_suffix = ''
    else:
        model_suffix = '-use_attr -weighted_loss multiple -data_dir %s -n_attributes %d -attr_loss_weight %.3f -normalize_loss' % (DATA_DIR, N_ATTR, config['lambda'])
        if USE_RELU:
            model_suffix += ' -use_relu'
        if USE_SIGMOID:
            model_suffix += ' -use_sigmoid'

        if config['model_type'] == 'end2end':
            model_suffix += ' -end2end'
        elif config['model_type'] == 'bottleneck':
            model_suffix += ' -bottleneck'
        elif config['model_type'] == 'onlyAttr':
            model_suffix += ' -no_img'
    command = model_suffix + ' -batch_size %d -lr %f -optimizer %s -weight_decay %f -scheduler_step %s' % (config['batch_size'], config['lr'], config['optimizer'], config['weight_decay'], config['scheduler_step'])
    log_dir = os.path.join(save_path, config['model_type'])
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, '_'.join(command.split(' ')))
    log_dir = log_dir.replace('-', '')
    command = command + ' -log_dir %s' % log_dir
    command = BASE_COMMAND % command
    print("Launch command:", command, '\n')
    subprocess.run([command])

def parse_arguments(parser=None):
    if parser is None: parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to the data used for evaluation')
    return parser.parse_args()

def run(args):
    for config in all_configs:
        launch_job(config, args.save_dir)


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
