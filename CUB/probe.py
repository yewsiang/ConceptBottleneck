
import pdb
import os
import sys
import math
import torch
import pickle
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

from CUB.models import MLP
from CUB.dataset import find_class_imbalance
from analysis import AverageMeter, Logger, binary_accuracy
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE


class LinearProbeDataset(Dataset):
    def __init__(self, pkl_file_paths):
        self.data = []
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        return img_data['representation_logits'], img_data['attribute_label']

def run_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion_list, args, is_training):
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            inputs = torch.stack(inputs).t().float()
        if isinstance(labels, list):
            labels = torch.stack(labels).t().float()
        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var
        
        outputs = model(inputs_var)
        #loss
        loss = 0
        for i in range(len(criterion_list)):
            loss += criterion_list[i](outputs[:, i], labels_var[:, i])    
        loss = loss / args.n_attributes
        #acc
        acc = binary_accuracy(torch.nn.Sigmoid()(outputs), labels)
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))

        if is_training:
            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters
    return loss_meter, acc_meter

def linear_probe(args):
    if os.path.exists(args.log_dir): # job restarted by cluster
        for f in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, f))
    else:
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.flush()

    model = MLP(args.n_attributes, args.n_attributes, expand_dim=None)
    model = model.cuda() if torch.cuda.is_available() else model
    #calculate imbalance
    imbalance = find_class_imbalance(os.path.join(BASE_DIR, args.data_dir, 'train.pkl'), True)
    attr_criteria = []
    for ratio in imbalance:
        r = torch.FloatTensor([ratio])
        r = r.cuda() if torch.cuda.is_available() else r
        attr_criteria.append(torch.nn.BCEWithLogitsLoss(weight=r))

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)
    train_dataset = LinearProbeDataset([train_data_path])
    val_dataset = LinearProbeDataset([val_data_path])
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, drop_last=False)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        train_loss_meter, train_acc_meter = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, attr_criteria, args, is_training=True)
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        with torch.no_grad():
            val_loss_meter, val_acc_meter = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, attr_criteria, args, is_training=False)

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))

        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                'Val loss: %.4f\tVal acc: %.4f\t'
                'Best val epoch: %d\n'
                % (epoch, train_loss_meter.avg, train_acc_meter.avg, val_loss_meter.avg, val_acc_meter.avg, best_val_epoch)) 
        logger.flush()
        
        if epoch <= stop_epoch:
            scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_lr())

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break

def eval_linear_probe(args):
    model = torch.load(args.model_dir)
    model.eval()
    test_data_path = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')
    test_dataset = LinearProbeDataset([test_data_path])
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, drop_last=False)
    test_acc_meter = AverageMeter()

    all_attr_labels, all_attr_outputs_sigmoid = [], []
    for _, data in enumerate(test_loader):
        inputs, labels = data
        if isinstance(inputs, list):
            inputs = torch.stack(inputs).t().float()
        if isinstance(labels, list):
            labels = torch.stack(labels).t().float()
        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        outputs = model(inputs_var)
        sigmoid_outputs = torch.nn.Sigmoid()(outputs)
        all_attr_labels.extend(list(labels.flatten().data.cpu().numpy()))
        all_attr_outputs_sigmoid.extend(list(sigmoid_outputs.flatten().data.cpu().numpy()))
        acc = binary_accuracy(sigmoid_outputs, labels)
        test_acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        #print(test_acc_meter.avg, test_acc_meter.sum, test_acc_meter.count)
    all_attr_outputs_int = np.array(all_attr_outputs_sigmoid) >= 0.5
    f1 = f1_score(all_attr_labels, all_attr_outputs_int)
    print('F1 score on test set: %.4f' % f1)
    print('Accuracy on test set: %.4f' % test_acc_meter.avg)
    return f1, test_acc_meter.avg

def run(args):
    if args.eval:
        f1s, c_results = [], []
        for data_dir, model_dir in zip(args.data_dirs, args.model_dirs):
            args.data_dir = data_dir
            args.model_dir = model_dir
            f1, acc = eval_linear_probe(args)
            f1s.append(f1)
            c_results.append(1 - acc / 100.)

        values = (-1, -1, np.mean(c_results), np.std(c_results))
        output_string = '%.4f %.4f %.4f %.4f' % values
        print_string = 'Error of y: %.4f +- %.4f, Error of C: %.4f +- %.4f' % values
        print(print_string)
        output = open(os.path.join(args.log_dir, 'results.txt'), 'w')
        output.write(output_string)
    else:
        args.data_dir = args.data_dirs[0]
        linear_probe(args)

def parse_arguments(parser=None):
    if parser is None: parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-data_dirs', nargs='+', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='number of attributes used')
    parser.add_argument('-scheduler_step', type=int, help='Number of steps before decaying current learning rate by half', default=1000)
    parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', default=64, type=int, help='mini-batch size')
    parser.add_argument('-epochs', '-e', default=1000, type=int, help='epochs for training process')
    parser.add_argument('-save_step', default=10, type=int, help='number of epochs to save model')
    parser.add_argument('-lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-model_dirs', nargs='+', help='where the trained models are saved')
    parser.add_argument('-eval', action='store_true', help='whether to evaluate on test set')
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    args = parse_arguments()
    run(args)
