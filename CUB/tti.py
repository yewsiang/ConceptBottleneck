#!/usr/bin/env python
# coding: utf-8
import os
import sys
import torch
import pickle
import random
from scipy.stats import entropy
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.inference import *
from CUB.config import N_CLASSES, N_ATTRIBUTES
from CUB.utils import get_class_attribute_names


def get_stage2_pred(a_hat):
    stage2_inputs = torch.from_numpy(np.array(a_hat)).cuda().float()
    class_outputs = model2(stage2_inputs)
    class_outputs = torch.nn.Softmax()(class_outputs)
    return class_outputs.data.cpu().numpy()


def simulate_group_intervention(mode, replace_val, preds_by_attr, ptl_5, ptl_95, model2, attr_group_dict, b_attr_binary_outputs, b_class_labels, b_class_logits,
                                b_attr_outputs, b_attr_outputs_sigmoid, b_attr_outputs2, b_attr_labels,
                                instance_attr_labels, uncertainty_attr_labels, use_not_visible, min_uncertainty,
                                n_replace, use_relu, use_sigmoid, n_trials=1, connect_CY=False):
    assert len(instance_attr_labels) == len(b_attr_labels), 'len(instance_attr_labels): %d, len(b_attr_labels): %d' % (
    len(instance_attr_labels), len(b_attr_labels))
    assert len(uncertainty_attr_labels) == len(
        b_attr_labels), 'len(uncertainty_attr_labels): %d, len(b_attr_labels): %d' % (
    len(uncertainty_attr_labels), len(b_attr_labels))

    all_class_acc = []

    for _ in range(n_trials):
        b_attr_new = np.array(b_attr_outputs[:])

        if mode == 'random':
            replace_fn = lambda attr_preds: replace_random(attr_preds)

            def replace_random(attr_preds):
                replace_idx = []
                group_replace_idx = list(random.sample(list(range(args.n_groups)), n_replace))
                for i in group_replace_idx:
                    replace_idx.extend(attr_group_dict[i])
                return replace_idx

        else:  # entropy
            replace_fn = lambda attr_preds, attr_preds_sigmoid, attr_labels, img_id, n_replace, replace_cached: \
                replace_entropy_adaptive(attr_preds, attr_preds_sigmoid, attr_labels, img_id, n_replace, replace_cached)

            def attr_entropy_diff(attr_idx, attr_preds, attr_preds_sigmoid):
                init_entropy = entropy(get_stage2_pred(attr_preds))  # constant -> doesn't matter
                attr_logit = attr_preds_sigmoid[attr_idx]
                if int(attr_logit):
                    p1 = attr_logit
                    p0 = 1 - p1
                else:
                    p0 = attr_logit
                    p1 = 1 - p0
                a_hat_0 = attr_preds[:]
                a_hat_0[attr_idx] = ptl_5[attr_idx]
                a_hat_1 = attr_preds[:]
                a_hat_1[attr_idx] = ptl_95[attr_idx]
                expected_entropy = p0 * entropy(get_stage2_pred(a_hat_0)) + p1 * entropy(get_stage2_pred(a_hat_1))
                return init_entropy - expected_entropy

            def group_entropy_diff(group_attr_idx, attr_preds, attr_preds_sigmoid):
                total_diff = 0
                for attr_idx in group_attr_dict[group_attr_idx]:
                    total_diff += attr_entropy_diff(attr_idx, attr_preds, attr_preds_sigmoid)
                return total_diff / len(group_attr_dict[group_attr_idx])

            def replace_entropy_non_adaptive(attr_preds, attr_preds_sigmoid, chosen=[], n=1):
                # print("n replace:", n_replace, "replace cached:", chosen)
                all_entropy_change = []
                for group_attr_idx in range(args.n_groups):
                    all_entropy_change.append(group_attr_entropy_diff(group_attr_idx, attr_preds, attr_preds_sigmoid))
                group_replace_idx = np.argsort(all_entropy_change)[::-1]
                if n == 1:
                    i = 0
                    while replace_idx[i] in chosen:
                        i += 1
                    return replace_idx[i]
                else:
                    return replace_idx[:n]

            def replace_entropy_adaptive(attr_preds, attr_preds_sigmoid, attr_labels, img_id, n_replace,
                                         replace_cached):
                attr_preds_new = attr_preds[:]
                if n_replace == 1:
                    group_replace_idx = []
                else:
                    group_replace_idx = replace_cached[img_id * (n_replace - 1): (img_id + 1) * (n_replace - 1)]
                    for j in group_replace_idx:
                        for k in attr_group_dict[j]:
                            attr_preds_new[k] = (1 - attr_labels[k]) * ptl_5[k] + attr_labels[k] * ptl_95[k]

                idx = replace_entropy_non_adaptive(attr_preds_new, attr_preds_sigmoid, chosen=group_replace_idx, n=1)
                group_replace_idx.append(idx)
                return group_replace_idx

        attr_replace_idx = []
        all_attr_ids = []
        global replace_cached
        if n_replace == 1:
            replace_cached = []

        for img_id in range(len(b_class_labels)):
            attr_preds = b_attr_outputs[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
            attr_preds_sigmoid = b_attr_outputs_sigmoid[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
            attr_preds2 = b_attr_outputs2[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
            if mode == 'entropy':
                attr_labels = b_attr_labels[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                replace_idx = replace_fn(attr_preds, attr_preds_sigmoid, attr_labels, img_id, n_replace, replace_cached)
            else:
                replace_idx = replace_fn(attr_preds)
            all_attr_ids.extend(replace_idx)
            attr_replace_idx.extend(np.array(replace_idx) + img_id * args.n_attributes)

        # print(n_replace, len(all_attr_ids)/len(b_class_labels))
        replace_cached = all_attr_ids
        pred_vals = b_attr_binary_outputs[attr_replace_idx]
        true_vals = np.array(b_attr_labels)[attr_replace_idx]
        # print("acc among the replaced values:", (pred_vals == true_vals).mean())

        if replace_val == 'class_level':
            b_attr_new[attr_replace_idx] = np.array(b_attr_labels)[attr_replace_idx]
        else:
            b_attr_new[attr_replace_idx] = np.array(instance_attr_labels)[attr_replace_idx]

        if use_not_visible:
            not_visible_idx = np.where(np.array(uncertainty_attr_labels) == 1)[0]
            for idx in attr_replace_idx:
                if idx in not_visible_idx:
                    b_attr_new[idx] = 0

        if use_relu or not use_sigmoid:  # replace with percentile values
            binary_vals = b_attr_new[attr_replace_idx]
            for j, replace_idx in enumerate(attr_replace_idx):
                attr_idx = replace_idx % args.n_attributes
                b_attr_new[replace_idx] = (1 - binary_vals[j]) * ptl_5[attr_idx] + binary_vals[j] * ptl_95[attr_idx]

        # stage 2
        K = [1, 3, 5]
        model2.eval()

        b_attr_new = b_attr_new.reshape(-1, args.n_attributes)
        stage2_inputs = torch.from_numpy(np.array(b_attr_new)).cuda()
        if connect_CY:  # class_outputs is currently contributed by C --> Y
            new_cy_outputs = model2(stage2_inputs)
            old_stage2_inputs = torch.from_numpy(np.array(b_attr_outputs).reshape(-1, args.n_attributes)).cuda()
            old_cy_outputs = model2(old_stage2_inputs)
            class_outputs = torch.from_numpy(b_class_logits).cuda() + (new_cy_outputs - old_cy_outputs)
        else:
            class_outputs = model2(stage2_inputs)
        _, preds = class_outputs.topk(1, 1, True, True)
        b_class_outputs_new = preds.data.cpu().numpy().squeeze()
        class_acc = np.mean(np.array(b_class_outputs_new) == np.array(b_class_labels))
        all_class_acc.append(class_acc * 100)

    return max(all_class_acc)


def parse_arguments(parser=None):
    if parser is None: parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', nargs='+', help='where the trained model is saved')
    parser.add_argument('-model_dirs2', nargs='+', default=None, help='where another trained model is saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (val/ test) to be used')
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-no_background', help='whether to test on images with background removed', action='store_true')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-data_dir2', default='class_attr_data_10', help='directory to the raw data')
    parser.add_argument('-n_attributes', type=int, default=112, help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-feature_group_results', help='whether to print out performance of individual atttributes', action='store_true')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-class_level', help='Whether to correct with class- (if set) or instance- (if not set) level values', action='store_true')
    parser.add_argument('-use_invisible', help='Whether to include attribute visibility information', action='store_true')
    parser.add_argument('-mode', help='Which mode to use for correction. Choose from wrong_idx, entropy, uncertainty, random', default='entropy')
    parser.add_argument('-n_trials', help='Number of trials to run, when mode is random', type=int, default=1)
    parser.add_argument('-n_groups', help='Number of groups', type=int, default=28)
    parser.add_argument('-connect_CY', help='Whether to use concepts as auxiliary features (in multitasking) to predict Y', action='store_true')
    args = parser.parse_args()
    return args

def run(args):
    class_to_folder, attr_id_to_name = get_class_attribute_names()

    data = pickle.load(open(os.path.join(args.data_dir2, 'train.pkl'), 'rb'))
    class_attr_count = np.zeros((N_CLASSES, N_ATTRIBUTES, 2))
    for d in data:
        class_label = d['class_label']
        certainties = d['attribute_certainty']
        for attr_idx, a in enumerate(d['attribute_label']):
            if a == 0 and certainties[attr_idx] == 1:  # not visible
                continue
            class_attr_count[class_label][attr_idx][a] += 1

    class_attr_min_label = np.argmin(class_attr_count, axis=2)
    class_attr_max_label = np.argmax(class_attr_count, axis=2)
    equal_count = np.where(class_attr_min_label == class_attr_max_label)  # check where 0 count = 1 count, set the corresponding class attribute label to be 1
    class_attr_max_label[equal_count] = 1

    attr_class_count = np.sum(class_attr_max_label, axis=0)
    mask = np.where(attr_class_count >= 10)[0]  # select attributes that are present (on a class level) in at least [min_class_count] classes

    instance_attr_labels, uncertainty_attr_labels = [], []
    test_data = pickle.load(open(os.path.join(args.data_dir2, 'test.pkl'), 'rb'))
    for d in test_data:
        # if 'class_attr_data' in args.data_dir or 'end2end' in args.model_dir:
        instance_attr_labels.extend(list(np.array(d['attribute_label'])[mask]))
        uncertainty_attr_labels.extend(list(np.array(d['attribute_certainty'])[mask]))
        # else:
        #    instance_attr_labels.extend(list(np.array(d['attribute_label'])))
        #    uncertainty_attr_labels.extend(list(np.array(d['attribute_certainty'])))

    class_attr_id_to_name = dict()
    for k, v in attr_id_to_name.items():
        if k in mask:
            class_attr_id_to_name[list(mask).index(k)] = v

    attr_group_dict = dict()
    curr_group_idx = 0
    with open('CUB_200_2011/attributes/attributes.txt', 'r') as f:
        all_lines = f.readlines()
        line0 = all_lines[0]
        prefix = line0.split()[1][:10]
        attr_group_dict[curr_group_idx] = [0]
        for i, line in enumerate(all_lines[1:]):
            curr = line.split()[1][:10]
            if curr != prefix:
                curr_group_idx += 1
                prefix = curr
                attr_group_dict[curr_group_idx] = [i + 1]
            else:
                attr_group_dict[curr_group_idx].append(i + 1)

    for group_id, attr_ids in attr_group_dict.items():
        new_attr_ids = []
        for attr_id in attr_ids:
            if attr_id in mask:
                new_attr_ids.append(attr_id)
        attr_group_dict[group_id] = new_attr_ids

    total_so_far = 0
    for group_id, attr_ids in attr_group_dict.items():
        class_attr_ids = list(range(total_so_far, total_so_far + len(attr_ids)))
        total_so_far += len(attr_ids)
        attr_group_dict[group_id] = class_attr_ids

    class_attr_id = 0
    for i in range(len(mask)):
        class_attr_id_to_name[i] = attr_id_to_name[mask[i]]

    # stage 1
    _, _, b_class_labels, b_topk_class_outputs, b_class_logits, b_attr_labels, b_attr_outputs, b_attr_outputs_sigmoid, \
        b_wrong_idx, b_attr_outputs2 = eval(args)
    b_class_outputs = b_topk_class_outputs[:, 0]
    b_attr_binary_outputs = np.rint(b_attr_outputs_sigmoid).astype(int)

    preds_by_attr, ptl_5, ptl_95 = dict(), dict(), dict()
    for i, val in enumerate(b_attr_outputs):
        attr_idx = i % args.n_attributes
        if attr_idx in preds_by_attr:
            preds_by_attr[attr_idx].append(val)
        else:
            preds_by_attr[attr_idx] = [val]

    for attr_idx in range(args.n_attributes):
        preds = preds_by_attr[attr_idx]
        ptl_5[attr_idx] = np.percentile(preds, 5)
        ptl_95[attr_idx] = np.percentile(preds, 95)

    N_TRIALS = args.n_trials
    MIN_UNCERTAINTY_GAP = 0
    assert args.mode in ['wrong_idx', 'entropy', 'uncertainty', 'random']
    if args.class_level:
        REPLACE_VAL = 'class_level'
    else:
        REPLACE_VAL = 'instance_level'

    # stage 2
    model = torch.load(args.model_dir)
    if args.model_dir2:
        if 'rf' in args.model_dir2:
            model2 = load(args.model_dir2)
        else:
            model2 = torch.load(args.model_dir2)
    else:  # end2end, split model into 2
        all_mods = list(model.modules())
        # model = ListModule(all_mods[:-1])
        model2 = all_mods[-1]  # last fully connected layer

    results = []
    for n_replace in list(range(args.n_groups + 1)):
        if 'random' not in args.mode:
            N_TRIALS = 1
        acc = simulate_group_intervention(args.mode, REPLACE_VAL,
                                          preds_by_attr, ptl_5, ptl_95,
                                          model2,
                                          attr_group_dict,
                                          b_attr_binary_outputs,
                                          b_class_labels,
                                          b_class_logits,
                                          b_attr_outputs,
                                          b_attr_outputs_sigmoid,
                                          b_attr_outputs2,
                                          b_attr_labels,
                                          instance_attr_labels,
                                          uncertainty_attr_labels,
                                          args.use_invisible,
                                          MIN_UNCERTAINTY_GAP,
                                          n_replace, args.use_relu,
                                          args.use_sigmoid,
                                          n_trials=N_TRIALS,
                                          connect_CY=args.connect_CY)
        print(n_replace, acc)
        results.append([n_replace, acc])
    return results

if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    args = parse_arguments()
    all_values = []
    for i, model_dir in enumerate(args.model_dirs):
        print('----------')
        args.model_dir = model_dir
        args.model_dir2 = args.model_dirs2[i] if args.model_dirs2 else None
        values = run(args)
        all_values.append(values)

    output_string = ''
    no_intervention_groups = np.array(all_values[0])[:, 0]
    values = sum([np.array(values)[:, 1] / len(all_values) for values in all_values])
    for no_intervention_group, value in zip(no_intervention_groups, values):
        output_string += '%.4f %.4f\n' % (no_intervention_group, value)
    print(output_string)
    output = open(os.path.join(args.log_dir, 'results.txt'), 'w')
    output.write(output_string)
