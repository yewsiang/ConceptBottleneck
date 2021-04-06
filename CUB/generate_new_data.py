"""
Create variants of the initial CUB dataset
"""
import os
import sys
import copy
import torch
import random
import pickle
import argparse
import numpy as np
from PIL import Image
from shutil import copyfile
import torchvision.transforms as transforms
from collections import defaultdict as ddict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from CUB.config import N_ATTRIBUTES, N_CLASSES


def get_few_shot_data(n_samples, out_dir, data_file='train.pkl'):
    """
    For few shot training: from data_file, sample n_samples randomly and save the metadata corresponding to these samples to out_dir
    """
    random.seed(0)
    data = pickle.load(open(data_file, 'rb'))
    class_to_data = ddict(list)
    for d in data:
        class_to_data[d['class_label']].append(d)
    new_data = []
    for c, data_list in class_to_data.items():
        if len(data_list) < n_samples:
            print("Class %d does not have enough samples. Add all of %d instances to the new dataset" % (c, len(data_list)))
            new_data.extend(data_list)
        else:
            new_data.extend(random.sample(data_list, n_samples))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    f = open(os.path.join(out_dir, data_file.split('/')[-1]), 'wb')
    pickle.dump(new_data, f)

def get_fraction_data(fraction, out_dir, data_file='train.pkl'):
    """
    For data efficiency: from data file, extract fraction of data at random and write it to out_dir
    """
    random.seed(0)
    train_data = pickle.load(open(data_file, 'rb'))
    split_train = int(fraction * len(train_data))
    all_class_present = False
    while not all_class_present:
        random.shuffle(train_data)
        new_data = train_data[:(split_train)]
        distinct_class = set()
        for d in new_data:
            distinct_class.add(d['class_label'])
        if len(distinct_class) == N_CLASSES:
            all_class_present = True
        else:
            print(len(distinct_class))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    f = open(os.path.join(out_dir, data_file), 'wb')
    pickle.dump(new_data, f)

def get_class_attributes_data(min_class_count, out_dir, modify_data_dir='', keep_instance_data=False):
    """
    Use train.pkl to aggregate attributes on class level and only keep those that are predominantly 1 for at least min_class_count classes
    Transform data in modify_data_dir file using the class attribute statistics and save the new dataset to out_dir
    If keep_instance_data is True, then retain the original values of the selected attributes. Otherwise save aggregated class level attributes
    In our paper, we set min_class_count to be 10 and only use the following 112 attributes of indices 
    [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]
    """
    data = pickle.load(open('train.pkl', 'rb'))
    class_attr_count = np.zeros((N_CLASSES, N_ATTRIBUTES, 2))
    for d in data:
        class_label = d['class_label']
        certainties = d['attribute_certainty']
        for attr_idx, a in enumerate(d['attribute_label']):
            if a == 0 and certainties[attr_idx] == 1: #not visible
                continue
            class_attr_count[class_label][attr_idx][a] += 1

    class_attr_min_label = np.argmin(class_attr_count, axis=2)
    class_attr_max_label = np.argmax(class_attr_count, axis=2)
    equal_count = np.where(class_attr_min_label == class_attr_max_label) #check where 0 count = 1 count, set the corresponding class attribute label to be 1
    class_attr_max_label[equal_count] = 1

    attr_class_count = np.sum(class_attr_max_label, axis=0)
    mask = np.where(attr_class_count >= min_class_count)[0] #select attributes that are present (on a class level) in at least [min_class_count] classes
    class_attr_label_masked = class_attr_max_label[:, mask]
    if keep_instance_data:
        collapse_fn = lambda d: list(np.array(d['attribute_label'])[mask])
    else:
        collapse_fn = lambda d: list(class_attr_label_masked[d['class_label'], :])
    create_new_dataset(out_dir, 'attribute_label', collapse_fn, data_dir=modify_data_dir)

def shuffle_class(out_dir, data_dir):
    """
    Assume data_dir contains class level attributes so that each class is mapped to a unique attribute setting
    Shuffle (by rolling backwards) the mapping between attribute setting and class label
    """
    data = pickle.load(open(os.path.join(data_dir, 'train.pkl'), 'rb'))
    class_to_attr_setting = dict()
    for d in data:
        if len(class_to_attr_setting) == N_CLASSES:
            break
        if d['class_label'] in class_to_attr_setting:
            continue
        else:
            class_to_attr_setting[d['class_label']] = d['attribute_label']
    n_roll = 10
    new_to_old_label = {i : (i + n_roll) % N_CLASSES for i in range(N_CLASSES)}
    shuffle_fn = lambda d: class_to_attr_setting[new_to_old_label[d['class_label']]] #change attribute labels instead of class labels
    create_new_dataset(out_dir, 'attribute_label', shuffle_fn, data_dir=data_dir)

def get_attr_groups(attr_name_file):
    """
    Read attribute names one by one from attr_name_file and based on the common prefix, separate them into different attribute groups
    Return list of starting indices of those groups
    """
    new_group_idx = [0]
    with open(attr_name_file, 'r') as f:
        all_lines = f.readlines()
        line0 = all_lines[0]
        prefix = line0.split()[1][:10]
        for i, line in enumerate(all_lines[1:]):
            curr = line.split()[1][:10] 
            if curr != prefix:
                new_group_idx.append(i+1)
                prefix = curr
    return new_group_idx

def split_into_groups(single_attr, attr_name_file):
    """
    Create train, val and test datasets to predict either individual attributes or attribute groups (if single_attr is False)
    """
    if not single_attr: #attribute group
        start_indices = get_attr_groups(attr_name_file)
        end_indices = start_indices[1:] + [N_ATTRIBUTES]
    else:
        start_indices = list(range(N_ATTRIBUTES))[:10] #generate data for the first 10 individual attributes to experiment
        end_indices = [s + 1 for s in start_indices]
    groups = list(zip(start_indices, end_indices))
    count = 0
    for dataset in ['train', 'val', 'test']:
        data = pickle.load(open(dataset + '.pkl', 'rb'))
        for i, (s, e) in enumerate(groups):
            if not single_attr:
                save_dir = 'attribute_group_' + str(i)
            else:
                save_dir = 'attribute_' + str(i)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            new_data = []
            for d in data:
                new_d = copy.deepcopy(d)
                new_d['attribute_label'] = d['attribute_label'][s:e]
                new_d['attribute_certainty'] = d['attribute_certainty'][s:e]
                new_data.append(new_d)
            f = open(save_dir + '/' + dataset + '.pkl', 'wb')
            pickle.dump(new_data, f)
            f.close()

def change_img_dir_data(new_image_folder, datasets, data_dir='', out_dir='masked_datasets/'):
    """
    Change the prefix of img_path data in data_dir to new_image_folder
    """
    compute_fn = lambda d: os.path.join(new_image_folder, d['img_path'].split('/')[-2], d['img_path'].split('/')[-1]) 
    create_new_dataset(out_dir, 'img_path', datasets=datasets, compute_fn=compute_fn, data_dir=data_dir)

def create_logits_data(model_path, out_dir, data_dir='', use_relu=False, use_sigmoid=False):
    """
    Replace attribute labels in data_dir with the logits output by the model from model_path and save the new data to out_dir
    """
    model = torch.load(model_path)
    get_logits_train = lambda d: inference(d['img_path'], model, use_relu, use_sigmoid, is_train=True)
    get_logits_test = lambda d: inference(d['img_path'], model, use_relu, use_sigmoid, is_train=False)
    create_new_dataset(out_dir, 'attribute_label', get_logits_train, datasets=['train'], data_dir=data_dir)
    create_new_dataset(out_dir, 'attribute_label', get_logits_train, datasets=['val', 'test'], data_dir=data_dir)

def get_representation_linear_probe(model_path, layer_idx, out_dir, data_dir):
    model = torch.load(model_path)
    get_representation_train = lambda d: inference_no_grad(d['img_path'], model, use_relu=False, use_sigmoid=False, is_train=True, layer_idx=layer_idx)
    get_representation_test = lambda d: inference_no_grad(d['img_path'], model, use_relu=False, use_sigmoid=False, is_train=False, layer_idx=layer_idx) 
    create_new_dataset(out_dir, 'representation_logits', get_representation_train, datasets=['train'], data_dir=data_dir)
    create_new_dataset(out_dir, 'representation_logits', get_representation_test, datasets=['val', 'test'], data_dir=data_dir)


def inference(img_path, model, use_relu, use_sigmoid, is_train, resol=299, layer_idx=None):
    """
    For a single image stored in img_path, run inference using model and return A\hat (if layer_idx is None) or values extracted from layer layer_idx 
    """
    model.eval()
    # see utils.py
    if is_train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])

    # Trim unnecessary paths
    try:
        idx = img_path.split('/').index('CUB_200_2011')
        img_path = '/'.join(img_path.split('/')[idx:])
    except:
        img_path_split = img_path.split('/')
        split = 'train' if is_train else 'test'
        img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    input_var = torch.autograd.Variable(img_tensor).cuda()
    if layer_idx is not None:
        all_mods = list(model.modules())
        cropped_model = torch.nn.Sequential(*list(model.children())[:layer_idx])  # nn.ModuleList(all_mods[:layer_idx])
        print(type(input_var), input_var.shape, input_var)
        return cropped_model(input_var)

    outputs = model(input_var)
    if use_relu:
        attr_outputs = [torch.nn.ReLU()(o) for o in outputs]
    elif use_sigmoid:
        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
    else:
        attr_outputs = outputs

    attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1).squeeze()
    return list(attr_outputs.data.cpu().numpy())


def inference_no_grad(img_path, model, use_relu, use_sigmoid, is_train, resol=299, layer_idx=None):
    """
    Extract activation from layer_idx of model for input from img_path (for linear probe)
    """
    with torch.no_grad():
        attr_outputs = inference(img_path, model, use_relu, use_sigmoid, is_train, resol, layer_idx)
    #return [list(o.cpu().numpy().squeeze())[0] for o in attr_outputs]
    return [o.cpu().numpy().squeeze()[()] for o in attr_outputs]


def convert_3_class(data_dir, out_dir='three_class/'):
    """
    Transform attribute labels in the dataset in data_dir to create a separate prediction class for not visible attributes
    """

    def filter_not_visible(d):
        certainty = np.array(d['attribute_certainty'])
        not_visible_idx = np.where(certainty == 1)[0]
        attr_label = np.array(d['attribute_label'])
        attr_label[not_visible_idx] = 2
        return list(attr_label)

    create_new_dataset(out_dir, 'attribute_label', filter_not_visible, data_dir)


def create_new_dataset(out_dir, field_change, compute_fn, datasets=['train', 'val', 'test'], data_dir=''):
    """
    Generic function that given datasets stored in data_dir, modify/ add one field of the metadata in each dataset based on compute_fn
                          and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
                          and return the updated value for field_change
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, 'rb'))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)
            new_value = compute_fn(d)
            if field_change in d:
                old_value = d[field_change]
                assert (type(old_value) == type(new_value))
            new_d[field_change] = new_value
            new_data.append(new_d)
        f = open(os.path.join(out_dir, dataset + '.pkl'), 'wb')
        pickle.dump(new_data, f)
        f.close()


def mask_image(file_path, out_dir_name, remove_bkgnd=True):
    """
    Remove background or foreground (if remove_bkgnd is False) using segmentation label stored in segmentations/ folder in CUB dataset
    Save the masked image to the directory specified by out_dir_name
    """
    im = np.array(Image.open(file_path).convert('RGB'))
    segment_path = file_path.replace('images', 'segmentations').replace('.jpg', '.png')
    segment_im = np.array(Image.open(segment_path).convert('L'))
    mask = segment_im.astype(float) / 255
    if not remove_bkgnd:  # remove bird in the foreground instead
        mask = 1 - mask
    new_im = (im * mask[:, :, None]).astype(np.uint8)
    Image.fromarray(new_im).save(file_path.replace('/images/', out_dir_name))


def mask_dataset(pkl_file, out_dir_name, remove_bkgnd=True):
    """
    Apply mask_image() to each image stored in pkl_file
    """
    data = pickle.load(open(pkl_file, 'rb'))
    file_paths = [d['img_path'] for d in data]
    for file_path in file_paths:
        mask_image(file_path, out_dir_name, remove_bkgnd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str,
                        choices=['ExtractConcepts', 'ExtractProbeRepresentations', 'DataEfficiencySplits', 'ChangeAdversarialDataDir'],
                        help='Name of experiment to run.')
    parser.add_argument('--model_path', type=str, help='Path of model')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--adv_data_dir', type=str, help='Adversarial data directory')
    parser.add_argument('--train_splits', type=str, nargs='+', help='Train splits to use')
    parser.add_argument('--use_relu', action='store_true', help='Use Relu')
    parser.add_argument('--use_sigmoid', action='store_true', help='Use Sigmoid')
    parser.add_argument('--layer_idx', type=int, default=None, help='Layer id to extract probe representations')
    parser.add_argument('--n_samples', type=int, help='Number of samples for data efficiency split')
    parser.add_argument('--splits_dir', type=str, help='Data dir of splits')
    args = parser.parse_args()

    if args.exp == 'ExtractConcepts':
        create_logits_data(args.model_path, args.out_dir, args.data_dir, args.use_relu, args.use_sigmoid)
    elif args.exp == 'ExtractProbeRepresentations':
        get_representation_linear_probe(args.model_path, args.layer_idx, args.out_dir, args.data_dir)
    elif args.exp == 'ChangeAdversarialDataDir':
        change_img_dir_data(args.adv_data_dir, datasets=args.train_splits, data_dir=args.data_dir, out_dir=args.out_dir)
    elif args.exp == 'DataEfficiencySplits':
        get_few_shot_data(args.n_samples, args.out_dir, os.path.join(args.splits_dir, 'train.pkl'))
        get_few_shot_data(args.n_samples, args.out_dir, os.path.join(args.splits_dir, 'val.pkl'))
        copyfile(os.path.join(args.splits_dir, 'test.pkl'), os.path.join(args.out_dir, 'test.pkl'))
