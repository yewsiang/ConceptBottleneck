
import pdb
import os
import sys
import copy
import json
import pickle
import argparse
import datetime
from itertools import product
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import analysis
import numpy as np
from torch.utils.data import DataLoader

# Global dependencies
from OAI.config import Y_COLS, CONCEPTS_WO_KLG, CONCEPTS_BALANCED, CLASSES_PER_COLUMN, OUTPUTS_DIR, BASE_DIR, \
    EST_TIME_PER_EXP, N_DATALOADER_WORKERS, CACHE_LIMIT, TRANSFORM_STATISTICS_TRAIN

# Models and dataset
from OAI.models import ModelXtoC, ModelXtoC_SENN, ModelOracleCtoY, ModelXtoChat_ChatToY, ModelXtoCtoY, ModelXtoY, \
    ModelXtoCY, ModelXtoYWithAuxC
from OAI.dataset import load_non_image_data, load_data_from_different_splits, PytorchImagesDataset, \
    get_image_cache_for_split


# ----------------- Training Experiments -----------------
def train_X_to_C(args, dataset_kwargs, model_kwargs):

    dataloaders, datasets, dataset_sizes = load_data_from_different_splits(**dataset_kwargs)

    # ---- Model fitting ----
    if args.use_senn_model:
        model = ModelXtoC_SENN(model_kwargs)
    else:
        model = ModelXtoC(model_kwargs)
    results = model.fit(dataloaders=dataloaders, dataset_sizes=dataset_sizes)

    # ---- Save results ----
    save_model_results(model, results, args, dataset_kwargs, model_kwargs)

def train_oracle_C_to_y_and_test_on_Chat(args, dataset_kwargs, model_kwargs):

    params = json.loads(args.oracle_C_to_y_model_params) if args.oracle_C_to_y_model_params else {}
    y_cols = Y_COLS
    C_cols = args.C_cols
    TRAIN_SPLIT = 'train'
    TEST_SPLIT = 'test'

    # ---- Training Oracle C -> y ----
    C_train, y_train = load_non_image_data(TRAIN_SPLIT, C_cols, y_cols,
                                           transform_statistics=TRANSFORM_STATISTICS_TRAIN,
                                           zscore_C=dataset_kwargs['zscore_C'],
                                           zscore_Y=dataset_kwargs['zscore_Y'],
                                           return_CY_only=True,
                                           merge_klg_01=True,
                                           truncate_C_floats=True,
                                           shuffle_Cs=False)
    y_train = np.squeeze(y_train, axis=1)
    model_CtoY = ModelOracleCtoY(model_type=args.oracle_C_to_y_model, new_kwargs=params)
    model_CtoY.fit(C_train, y_train)

    # ---- Dataset to pass to ModelXtoC to generate Chats ----
    limit = 500 if args.use_small_subset else CACHE_LIMIT
    cache_test = get_image_cache_for_split(TEST_SPLIT, limit=limit)
    dataset = PytorchImagesDataset(dataset=TEST_SPLIT,
                                   transform_statistics=TRANSFORM_STATISTICS_TRAIN,
                                   C_cols=dataset_kwargs['C_cols'],
                                   y_cols=dataset_kwargs['y_cols'],
                                   zscore_C=dataset_kwargs['zscore_C'],
                                   zscore_Y=dataset_kwargs['zscore_Y'],
                                   cache=cache_test,
                                   truncate_C_floats=True,
                                   data_proportion=dataset_kwargs['data_proportion'],
                                   shuffle_Cs=False,
                                   merge_klg_01=True,
                                   transform='None',
                                   use_small_subset=dataset_kwargs['use_small_subset'],
                                   downsample_fraction=dataset_kwargs['downsample_fraction'])
    dataloader = DataLoader(dataset,
                            batch_size=dataset_kwargs['batch_size'],
                            shuffle=False,
                            num_workers=N_DATALOADER_WORKERS)
    # Get y_test information separately
    C_test, y_test = load_non_image_data(TEST_SPLIT, C_cols, y_cols,
                                         transform_statistics=TRANSFORM_STATISTICS_TRAIN,
                                         zscore_C=dataset_kwargs['zscore_C'],
                                         zscore_Y=dataset_kwargs['zscore_Y'],
                                         return_CY_only=True,
                                         merge_klg_01=True,
                                         truncate_C_floats=True,
                                         shuffle_Cs=False)
    y_test = np.squeeze(y_test, axis=1)

    dataloaders = {TEST_SPLIT: dataloader}
    if args.use_small_subset:
        dataset_sizes = {TEST_SPLIT: 500}
    else:
        dataset_sizes = {TEST_SPLIT: len(dataset)}

    # ---- Restore pretrained C model to get Chats ----
    results = {}
    for pretrained_path in model_kwargs['pretrained_paths']:
        model_kwargs['pretrained_path'] = pretrained_path

        # Sanity check
        y_train_hat = model_CtoY.predict(C_train)
        y_train_rmse = np.sqrt(np.mean((y_train - y_train_hat) ** 2))
        y_test_hat = model_CtoY.predict(C_test)
        y_test_rmse = np.sqrt(np.mean((y_test - y_test_hat) ** 2))

        if args.eval_model == 'X_to_C':
            model_class = ModelXtoC
        elif args.eval_model == 'X_to_C_to_y':
            model_class = ModelXtoCtoY

        model_XtoC = model_class(model_kwargs)
        metrics = model_XtoC.train_or_eval_dataset(dataloaders, dataset_sizes, TEST_SPLIT)

        # C_true = np.array([metrics['%s_%s_true' % (TEST_SPLIT, C)] for C in C_cols]) # Same as C_test
        C_hat = np.stack([metrics['%s_%s_pred' % (TEST_SPLIT, C)] for C in C_cols], axis=1)
        y_hat = model_CtoY.predict(C_hat)

        C_rmse = np.mean([metrics['%s_%s_rmse' % (TEST_SPLIT, C)] for C in C_cols])
        y_rmse = np.sqrt(np.mean((y_test - y_hat) ** 2))

        metrics['y_train_rmse'] = y_train_rmse
        metrics['y_test_rmse'] = y_test_rmse
        metrics['test_C_rmse'] = C_rmse
        metrics_y = analysis.assess_performance(y=y_test[:, None], yhat=y_hat[:, None],
                                                names=y_cols,
                                                prediction_type='continuous_ordinal',
                                                prefix='test',
                                                verbose=False)
        metrics.update(metrics_y)
        results[pretrained_path] = metrics

    # ---- Save results ----
    unique_name = get_results_name(args.name)
    # Save results
    results_path = os.path.join(OUTPUTS_DIR, unique_name, 'results.pkl')
    pickle.dump(results, open(results_path, 'wb'))

def train_Chat_to_y_and_test_on_Chat(args, dataset_kwargs, model_kwargs):

    dataloaders, datasets, dataset_sizes = load_data_from_different_splits(**dataset_kwargs)

    # ---- Model fitting ----
    model = ModelXtoChat_ChatToY(model_kwargs)
    results = model.fit(dataloaders=dataloaders, dataset_sizes=dataset_sizes)

    # ---- Save results ----
    save_model_results(model, results, args, dataset_kwargs, model_kwargs)

def train_X_to_C_to_y(args, dataset_kwargs, model_kwargs):

    dataloaders, datasets, dataset_sizes = load_data_from_different_splits(**dataset_kwargs)

    # ---- Model fitting ----
    model = ModelXtoCtoY(model_kwargs)
    results = model.fit(dataloaders=dataloaders, dataset_sizes=dataset_sizes)

    # ---- Save results ----
    save_model_results(model, results, args, dataset_kwargs, model_kwargs)

def train_X_to_y(args, dataset_kwargs, model_kwargs):

    dataloaders, datasets, dataset_sizes = load_data_from_different_splits(**dataset_kwargs)

    # ---- Model fitting ----
    model = ModelXtoY(model_kwargs)
    results = model.fit(dataloaders=dataloaders, dataset_sizes=dataset_sizes)

    # ---- Save results ----
    save_model_results(model, results, args, dataset_kwargs, model_kwargs)

def train_X_to_y_with_aux_C(args, dataset_kwargs, model_kwargs):

    dataloaders, datasets, dataset_sizes = load_data_from_different_splits(**dataset_kwargs)

    # ---- Model fitting ----
    model = ModelXtoYWithAuxC(model_kwargs)
    results = model.fit(dataloaders=dataloaders, dataset_sizes=dataset_sizes)

    # ---- Save results ----
    save_model_results(model, results, args, dataset_kwargs, model_kwargs)

def train_X_to_Cy(args, dataset_kwargs, model_kwargs):

    dataloaders, datasets, dataset_sizes = load_data_from_different_splits(**dataset_kwargs)

    # ---- Model fitting ----
    model = ModelXtoCY(model_kwargs)
    results = model.fit(dataloaders=dataloaders, dataset_sizes=dataset_sizes)

    # ---- Save results ----
    save_model_results(model, results, args, dataset_kwargs, model_kwargs)

def train_probe(args, dataset_kwargs, model_kwargs):
    train_X_to_C(args, dataset_kwargs, model_kwargs)

def save_model_results(model, results, args, dataset_kwargs, model_kwargs, exp_name=None):
    experiment_name = exp_name if exp_name else args.name
    experiment_to_run = args.exp
    timestring = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_').replace('-', '_')
    unique_name = timestring if experiment_name is None else experiment_name

    print("Saving code, weights, config, and results as: %s" % unique_name)

    # Save code
    src_models_code_dir = os.path.join(BASE_DIR, 'OAI')
    dest_code_dir = os.path.join(OUTPUTS_DIR, unique_name, 'code')
    if not os.path.exists(dest_code_dir): os.makedirs(dest_code_dir)
    # Models code
    for model_code_file in os.listdir(src_models_code_dir):
        src_file = os.path.join(src_models_code_dir, model_code_file)
        dest_file = os.path.join(dest_code_dir, model_code_file)
        os.system('cp %s %s' % (src_file, dest_file))
    # Dataset code
    src_file = os.path.join(BASE_DIR, 'dataset.py')
    dest_file = os.path.join(dest_code_dir, 'dataset.py')
    os.system('cp %s %s' % (src_file, dest_file))

    # Save configs
    config = {'dataset_kwargs': dataset_kwargs, 'model_kwargs': model_kwargs, 'experiment_to_run': experiment_to_run}
    config_path = os.path.join(OUTPUTS_DIR, unique_name, 'config.pkl')
    pickle.dump(config, open(config_path, 'wb'))

    # Save results
    results_path = os.path.join(OUTPUTS_DIR, unique_name, 'results.pkl')
    pickle.dump(results, open(results_path, 'wb'))

    # Save model weights.
    weights_path = os.path.join(OUTPUTS_DIR, unique_name, 'model_weights.pth')
    try:
        print('Model saved A')
        torch.save(model.state_dict, weights_path)
    except:
        print('Model saved B')
        torch.save(model.state_dict(), weights_path)

# ----------------- Test-time Intervention Experiments -----------------
def test_time_intervention(args, dataset_kwargs, model_kwargs):

    PHASE = args.test_time_intervention_split
    np.random.seed(0)
    N_CONCEPTS_TO_GIVE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # ---- Dataset ----
    limit = 500 if args.use_small_subset else CACHE_LIMIT
    cache_test = get_image_cache_for_split(PHASE, limit=limit)
    dataset = PytorchImagesDataset(dataset=PHASE,
                                   transform_statistics=TRANSFORM_STATISTICS_TRAIN,
                                   C_cols=dataset_kwargs['C_cols'],
                                   y_cols=dataset_kwargs['y_cols'],
                                   zscore_C=dataset_kwargs['zscore_C'],
                                   zscore_Y=dataset_kwargs['zscore_Y'],
                                   cache=cache_test,
                                   truncate_C_floats=True,
                                   shuffle_Cs=False,
                                   merge_klg_01=True,
                                   transform='None',
                                   use_small_subset=dataset_kwargs['use_small_subset'],
                                   downsample_fraction=dataset_kwargs['downsample_fraction'])
    dataloader = DataLoader(dataset,
                            batch_size=dataset_kwargs['batch_size'],
                            shuffle=False,
                            num_workers=N_DATALOADER_WORKERS)

    dataloaders = {PHASE: dataloader}
    if args.use_small_subset:
        dataset_sizes = {PHASE: 500}
    else:
        dataset_sizes = {PHASE: len(dataset)}

    # ---- Model inference ----
    if args.test_time_intervention_model == 'X_to_C_to_y':
        model = ModelXtoCtoY(model_kwargs)
    elif args.test_time_intervention_model == 'X_to_y_with_aux_C':
        model = ModelXtoYWithAuxC(model_kwargs)
    elif args.test_time_intervention_model == 'X_to_Chat__Chat_to_y':
        model = ModelXtoChat_ChatToY(model_kwargs)
    elif args.test_time_intervention_model == 'X_to_Chat__OracleC_to_y':

        # ---- Training Oracle C -> y ----
        C_cols, y_cols = dataset_kwargs['C_cols'], dataset_kwargs['y_cols']
        params = json.loads(args.oracle_C_to_y_model_params) if args.oracle_C_to_y_model_params else {}
        C_train, y_train = load_non_image_data('train', C_cols, y_cols,
                                               transform_statistics=TRANSFORM_STATISTICS_TRAIN,
                                               zscore_C=dataset_kwargs['zscore_C'],
                                               zscore_Y=dataset_kwargs['zscore_Y'],
                                               return_CY_only=True,
                                               merge_klg_01=True,
                                               truncate_C_floats=True,
                                               shuffle_Cs=False)
        y_train = np.squeeze(y_train, axis=1)
        model_CtoY = ModelOracleCtoY(model_type=args.oracle_C_to_y_model, new_kwargs=params)
        model_CtoY.fit(C_train, y_train)

        class ModelXtoChat_OracleCToY(ModelXtoChat_ChatToY):
            def __init__(self, cfg, model_CtoY, build=True):
                ModelXtoChat_ChatToY.__init__(self, cfg, build=build)
                self.model_CtoY = model_CtoY

            def forward_with_intervention(self, inputs, labels):
                outputs = super(ModelXtoChat_ChatToY, self).forward_with_intervention(inputs, labels)
                # Instead of a Chat->Y model to intervene, we intervene with an OracleC->Y model
                C_hat_intervened = outputs['C']
                y_hat = self.model_CtoY.predict(C_hat_intervened.cpu().detach().numpy())
                outputs['y'] = torch.Tensor(y_hat[:, None]).cuda()
                return outputs

        model = ModelXtoChat_OracleCToY(model_kwargs, model_CtoY)

    # Determine intervention ordering (only applicable if using ordered intervention)
    if args.test_time_intervention_analysis:
        metrics_all = model.intervention_analysis(dataloaders, dataset_sizes, PHASE)
        intervention_order = model.get_intervention_ordering(metrics_all)
        model.intervention_order = intervention_order

    metrics_all = {}
    for n in N_CONCEPTS_TO_GIVE_LIST:
        print('----- Intervention on %d concepts given -----' % n)
        model.set_intervention_N_concepts(n)
        metrics = model.train_or_eval_dataset(dataloaders, dataset_sizes, PHASE,
                                              intervention=True)
        metrics_all[n] = metrics

    # ---- Save results ----
    save_test_time_intervention_results(model, metrics_all, args, dataset_kwargs, model_kwargs)

def save_test_time_intervention_results(model, results, args, dataset_kwargs, model_kwargs):
    experiment_name = args.name
    experiment_to_run = args.exp
    timestring = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_').replace('-', '_')
    unique_name = timestring if experiment_name is None else experiment_name

    print("Saving code, weights, config, and results as: %s" % unique_name)

    # Save code
    src_models_code_dir = os.path.join(BASE_DIR, 'OAI')
    dest_code_dir = os.path.join(OUTPUTS_DIR, unique_name, 'code')
    if not os.path.exists(dest_code_dir): os.makedirs(dest_code_dir)
    # Models code
    for model_code_file in os.listdir(src_models_code_dir):
        src_file = os.path.join(src_models_code_dir, model_code_file)
        dest_file = os.path.join(dest_code_dir, model_code_file)
        os.system('cp %s %s' % (src_file, dest_file))
    # Dataset code
    src_file = os.path.join(BASE_DIR, 'dataset.py')
    dest_file = os.path.join(dest_code_dir, 'dataset.py')
    os.system('cp %s %s' % (src_file, dest_file))

    # Save configs
    config = {'dataset_kwargs': dataset_kwargs, 'model_kwargs': model_kwargs, 'experiment_to_run': experiment_to_run}
    config_path = os.path.join(OUTPUTS_DIR, unique_name, 'config.pkl')
    pickle.dump(config, open(config_path, 'wb'))

    # Save results
    results_path = os.path.join(OUTPUTS_DIR, unique_name, 'results.pkl')
    pickle.dump(results, open(results_path, 'wb'))

# ----------------- Hyperparameter Optimization -----------------
def hyperparameter_optimization(args, dataset_kwargs, model_kwargs):

    dataloaders, datasets, dataset_sizes = load_data_from_different_splits(**dataset_kwargs)

    args.hyperopt_params = json.loads(args.hyperopt_params)
    args.hyperopt_additional = json.loads(args.hyperopt_additional) if args.hyperopt_additional else {}
    if args.hyperopt_model == 'X_to_y':
        model_class = ModelXtoY
    elif args.hyperopt_model == 'X_to_Cy':
        model_class = ModelXtoCY
    elif args.hyperopt_model == 'X_to_C_to_y':
        model_class = ModelXtoCtoY
    elif args.hyperopt_model == 'X_to_Chat__Chat_to_y':
        model_class = ModelXtoC

    # ---- Generate candidate parameters ----
    candidate_parameters = []
    if args.hyperopt_search == 'random':
        raise NotImplementedError()
    elif args.hyperopt_search == 'grid':
        keys = [key for key in args.hyperopt_params.keys()]
        values = [value for value in args.hyperopt_params.values()]
        candidate_parameters = [tup for tup in product(*values)]

    def recursive_set_attr(kwargs, key, value):
        if len(key) == 1:
            kwargs[key[0]] = value
            return
        new_kwargs = kwargs[key[0]]
        new_key = key[1:]
        return recursive_set_attr(new_kwargs, new_key, value)

    def convert_params_to_kwargs(names, parameters, model_kwargs):
        model_kwargs_new = copy.deepcopy(model_kwargs)
        for name, parameter in zip(names, parameters):
            recursive_set_attr(model_kwargs_new, name.split('.'), parameter)
        return model_kwargs_new

    def get_exp_name(model_name, cand_id, trial_id, keys, params):
        string = 'opt/%s_Cand%d_Trial%d' % (model_name, cand_id, trial_id)
        for key, param in zip(keys, params):
            string += '_%s@%s' % (key, param)
        return string

    # ---- Run evaluations for each candidate parameters ----
    N_exps = len(candidate_parameters) * args.hyperopt_n_repeats
    print('Running a total of %d params X %d repeats = %d experiments' %
          (len(candidate_parameters), args.hyperopt_n_repeats, N_exps))
    print('Estimated time: %d H' % (N_exps * EST_TIME_PER_EXP))
    candidate_scores = []
    for i, parameters in enumerate(candidate_parameters):
        print(' ------ Evaluating candidate %d/%d ------' % (i + 1, len(candidate_parameters)))
        scores = []
        for j in range(args.hyperopt_n_repeats):
            print(' ---------- Trial %d ----------' % (j + 1))
            exp_name = os.path.join(args.name, get_exp_name(args.hyperopt_model, i + 1, j + 1, keys, parameters))
            model_kwargs_param = convert_params_to_kwargs(keys, parameters, model_kwargs)
            model = model_class(model_kwargs_param)
            results = model.fit(dataloaders=dataloaders, dataset_sizes=dataset_sizes)
            save_model_results(model, results, args, dataset_kwargs, model_kwargs_param, exp_name=exp_name)

            if args.hyperopt_model == 'X_to_Chat__Chat_to_y':
                print(' ----- Training Chat_to_y -----')
                extra_params = {'pretrained_path': os.path.join(OUTPUTS_DIR, exp_name, 'model_weights.pth'),
                                'front_fc_layers_to_freeze': args.hyperopt_additional['front_fc_layers_to_freeze'],
                                'fc_layers': args.hyperopt_additional['fc_layers'],
                                'y_fc_name': args.hyperopt_additional['y_fc_name']}
                model_kwargs_param_new = copy.deepcopy(model_kwargs_param)
                model_kwargs_param_new.update(extra_params)
                model = ModelXtoChat_ChatToY(model_kwargs_param_new)
                results = model.fit(dataloaders=dataloaders, dataset_sizes=dataset_sizes)
                save_model_results(model, results, args, dataset_kwargs, model_kwargs_param_new)

            # Use last epoch validation result as the score
            score = results[model_kwargs['num_epochs'] - 1][args.hyperopt_score_metric]
            if args.hyperopt_negate_score: score *= -1.
            scores.append(score)
        candidate_scores.append(np.mean(scores))

    # ---- Report the best hyperparameter ----
    print(' ------ Results ------')
    print(' Parameter names: %s' % (str(keys)))
    best_idx = np.argmax(candidate_scores)
    for i, (score, parameters) in enumerate(zip(candidate_scores, candidate_parameters)):
        best = '[Best] ' if i == best_idx else ''
        print('   Score: %.3f %s| Parameters: %s ' % (score, best, str(parameters)))

# ----------------- Boilerplate -----------------
def get_results_name(experiment_name):
    timestring = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_').replace('-', '_')
    unique_name = timestring if experiment_name is None else experiment_name
    return unique_name

def generate_configs(args):

    concepts_to_use = args.C_cols
    N_concepts = len(concepts_to_use)
    dataset_kwargs = {
        'batch_size': args.batch_size,
        'C_cols': concepts_to_use,
        'y_cols': Y_COLS,
        'zscore_C': False,
        'zscore_Y': False,
        'sampling_strategy': 'uniform',
        'sampling_args': None,
        'merge_klg_01': True,
        'max_horizontal_translation': 0.1,
        'max_vertical_translation': 0.1,
        'C_hat_path': None,
        'use_small_subset': False,
        'downsample_fraction': None
    }

    model_kwargs = {
        'C_cols': concepts_to_use,
        'y_cols': Y_COLS,
        # ---- Needed for classification ----
        'classes_per_C_col': None,
        'classes_per_y_col': None,
        # ------- Test-time inference -------
        'test_time_intervention_model': None,
        'test_time_intervention_method': None,
        'intervention_order': None,
        # -----------------------------------
        'additional_loss_weighting': [1.] * len(concepts_to_use),
        'conv_layers_before_end_to_unfreeze': 12,
        'fc_layers': None, # Need to define depending on experiment
        'num_epochs': 15,
        'optimizer_kwargs': {
            'betas': [
                0.9,
                0.999
            ],
            'lr': 0.0005
        },
        'optimizer_name': 'adam',
        'pretrained_path': None,
        'pretrained_model_name': 'resnet18',
        'scheduler_kwargs': {
            'additional_kwargs': {},
            'lr_scheduler_type': None,
        },
        'verbose': {
            'time_100batches': True,
            'time_breakdown': False,
            'layer_magnitudes': False,
        }
    }

    # Create output dir
    experiment_name = args.name
    if experiment_name:
        experiment_dir = os.path.join(OUTPUTS_DIR, experiment_name)
        # assert not os.path.exists(experiment_dir)
        if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    # Modify configs according to argparse
    if args.y_fc_name:
        assert args.y_fc_name[:2] == 'fc'
        fc_layer_y = int(args.y_fc_name[2:]) - 1
    if args.C_fc_name:
        assert args.C_fc_name[:2] == 'fc'
        fc_layer_C = int(args.C_fc_name[2:]) - 1

    # The number of classes per C and y (needed because classification needs to know the no. of classes)
    assert dataset_kwargs['merge_klg_01'] and CLASSES_PER_COLUMN['xrkl'] == 4
    model_kwargs['classes_per_C_col'] = [CLASSES_PER_COLUMN[variable] for variable in model_kwargs['C_cols']]
    model_kwargs['classes_per_y_col'] = [CLASSES_PER_COLUMN[variable] for variable in model_kwargs['y_cols']]

    # Should not drop majority
    assert args.dropout <= 0.5

    # Assertions to ensure correct setup for FC layers
    N_y_cols = len(model_kwargs['y_cols'])
    if args.y_loss_type == 'reg':
        y_dims = N_y_cols
    elif args.y_loss_type == 'cls':
        y_dims = sum(model_kwargs['classes_per_y_col'])
        assert 'Y' not in args.zscore

    if args.C_loss_type == 'reg':
        C_dims = N_concepts
    elif args.C_loss_type == 'cls':
        C_dims = sum(model_kwargs['classes_per_C_col'])
        assert 'C' not in args.zscore

    experiment_name = args.exp
    if experiment_name == 'TTI':
        # Test-time Intervention
        model_kwargs['test_time_intervention_method'] = args.test_time_intervention_method
        model_kwargs['intervention_order'] = args.intervention_order
        if model_kwargs['test_time_intervention_method'] == 'ordered':
            assert (args.intervention_order is not None) or args.test_time_intervention_analysis
            if args.intervention_order:
                assert len(set(args.intervention_order)) == N_concepts
                assert 0 <= max(args.intervention_order) < N_concepts
        else:
            # TTI analysis determines ordering only for an ordered intervention method
            assert not args.test_time_intervention_analysis

    if experiment_name == 'Independent_CtoY':
        # Need pretrained model to get Chats
        assert args.pretrained is not None
        assert args.oracle_C_to_y_model is not None
        # Because we are loading a ModelXtoC
        model_kwargs['use_input_conv_layers'] = False

    elif experiment_name == 'Sequential_CtoY' or \
            args.test_time_intervention_model in ['X_to_Chat__Chat_to_y', 'X_to_Chat__OracleC_to_y']:
        # model_kwargs['fc_layers'] = [N_concepts, 50, 50, 1]
        assert C_dims == args.fc_layers[fc_layer_C], print(C_dims, args.fc_layers[fc_layer_C])
        assert y_dims == args.fc_layers[fc_layer_y], print(y_dims, args.fc_layers[fc_layer_y])
        assert args.pretrained is not None
        assert args.front_fc_layers_to_freeze is not None

    elif experiment_name == 'Standard':
        # Put as N_concepts + 1 to share code with train_X_to_Cy, but only train the y portion
        # model_kwargs['fc_layers'] = [N_concepts + 1]
        assert y_dims == args.fc_layers[fc_layer_y], print(y_dims, args.fc_layers[fc_layer_y])

    elif experiment_name == 'StandardWithAuxC':
        assert y_dims == args.fc_layers[fc_layer_y], print(y_dims, args.fc_layers[fc_layer_y])

    elif experiment_name == 'Multitask':
        # model_kwargs['fc_layers'] = [N_concepts + 1]
        assert (C_dims + y_dims) == args.fc_layers[fc_layer_y], print(C_dims + y_dims, args.fc_layers[fc_layer_y])
        assert fc_layer_y == fc_layer_C, print(fc_layer_y, fc_layer_C)

    elif experiment_name == 'Joint' or \
            args.test_time_intervention_model in ['X_to_C_to_y']:
        # model_kwargs['fc_layers'] = [N_concepts, 50, 50, 1]
        assert C_dims == args.fc_layers[fc_layer_C], print(C_dims, args.fc_layers[fc_layer_C])
        assert y_dims == args.fc_layers[fc_layer_y], print(y_dims, args.fc_layers[fc_layer_y])

    elif experiment_name in ['Concept_XtoC', 'Probe']:
        # model_kwargs['fc_layers'] = [N_concepts]
        assert C_dims == args.fc_layers[fc_layer_C], print(C_dims, args.fc_layers[fc_layer_C])
        model_kwargs['use_input_conv_layers'] = False if args.use_senn_model else \
                (experiment_name == 'Probe')

    elif experiment_name == 'HyperparameterSearch' or args.test_time_intervention_model in ['X_to_y_with_aux_C']:
        pass
    else:
        raise NotImplementedError('Experiment not implemented: %s' % experiment_name)

    # Set learning rate scheduler parameters
    model_kwargs['optimizer_kwargs']['lr'] = args.lr
    model_kwargs['scheduler_kwargs']['lr_scheduler_type'] = args.lr_scheduler
    kwargs = model_kwargs['scheduler_kwargs']['additional_kwargs']
    if args.lr_scheduler == 'plateau':
        # Every N epochs where there is no improvement of some metric, reduce LR by a factor L
        kwargs['factor'] = 0.5
        kwargs['mode'] = 'max'
        kwargs['patience'] = 1
        kwargs['verbose'] = True
    elif args.lr_scheduler == 'step':
        # Every step size of K, reduce LR by a factor L
        kwargs['step_size'] = 10
        kwargs['gamma'] = 0.5
    else:
        raise Exception('Unknown scheduler: %s' % args.lr_scheduler)

    # Different sampling weights according to the sample's C and y
    if args.sampling in ['weigh_C', 'weigh_Cy']:
        dataset_kwargs['sampling_strategy'] = 'weighted'
        dataset_kwargs['sampling_args'] = {
            'mode': args.sampling,
            'min_count': 5,
        }

    # Variables to zscore
    if args.zscore == 'CY':
        dataset_kwargs['zscore_C'] = True
        dataset_kwargs['zscore_Y'] = True
    elif args.zscore == 'C':
        dataset_kwargs['zscore_C'] = True
    elif args.zscore == 'Y':
        dataset_kwargs['zscore_Y'] = True

    # Loss weighting for EACH concept
    if len(args.C_weight) == 1:
        model_kwargs['additional_loss_weighting'] = args.C_weight * len(concepts_to_use)
    elif len(args.C_weight) == len(concepts_to_use):
        model_kwargs['additional_loss_weighting'] = args.C_weight
    else:
        raise Exception('Your C weights should be the same as dimensions of A: %s' % str(args.C_weight))

    if args.use_small_subset:
        dataset_kwargs['use_small_subset'] = True

    # Loss weighting (according to class frequencies) for EACH class within EACH concept
    model_kwargs['C_loss_weigh_class'] = args.C_loss_weigh_class

    # Assigning argparse configs to our configs
    dataset_kwargs['augment'] = args.augment
    dataset_kwargs['data_proportion'] = args.data_proportion
    dataset_kwargs['shuffle_Cs'] = args.shuffle_Cs
    model_kwargs['num_epochs'] = args.num_epochs
    model_kwargs['pretrained_paths'] = args.pretrained
    model_kwargs['pretrained_path'] = args.pretrained[0] if args.pretrained else None # Assumes there is only one model provided
    model_kwargs['pretrained_exclude_vars'] = args.pretrained_exclude_vars
    model_kwargs['conv_layers_before_end_to_unfreeze'] = args.conv_layers_before_end_to_unfreeze
    model_kwargs['prefixes_of_vars_to_freeze'] = args.prefixes_of_vars_to_freeze
    model_kwargs['front_fc_layers_to_freeze'] = args.front_fc_layers_to_freeze
    model_kwargs['fc_layers'] = args.fc_layers
    model_kwargs['input_conv_layers'] = args.input_conv_layers
    model_kwargs['C_fc_name'] = args.C_fc_name
    model_kwargs['y_fc_name'] = args.y_fc_name
    model_kwargs['dropout'] = args.dropout
    model_kwargs['C_loss_type'] = args.C_loss_type
    model_kwargs['y_loss_type'] = args.y_loss_type

    return dataset_kwargs, model_kwargs

def parse_arguments(experiment):
    # ---- General ----
    # Get argparse configs from user
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Name of the dataset.')
    parser.add_argument('exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'StandardWithAuxC', 'Multitask', 'Joint', 'Probe',
                                 'TTI', 'HyperparameterSearch'],
                        help='Name of experiment to run.')
    parser.add_argument('--name', type=str, help='Name of the experiment will be saved as.')
    parser.add_argument('--seed', type=int, required=True, help='Seed for numpy and torch.')
    parser.add_argument('--folder', type=str, help='Input experiment folder for functions') # For get_Chat_from_models
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train for.')
    parser.add_argument('--use_small_subset', action='store_true', help='To use small subset of data or not.')
    parser.add_argument('--data_proportion', type=float, default=1., help='Proportion of data to use.')
    parser.add_argument('--shuffle_Cs', action='store_true', help='Shuffle the training set C values for each A.')
    parser.add_argument('--silent', action='store_true', default=True, help='Whether to be silent in logging or not.')

    # ---- Evaluation ----
    parser.add_argument('--eval_split', type=str, choices=['train', 'val', 'test'],
                        help='Which split to evaluate on.')
    parser.add_argument('--eval_model', type=str, choices=['X_to_y', 'X_to_Cy', 'X_to_C_to_y',
                                                           'X_to_C', 'X_to_Chat__Chat_to_y'])

    # ---- Optimization ----
    parser.add_argument('--batch_size', '--b', type=int, default=8, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--lr_scheduler', default='step', choices=['plateau', 'step'],
                        help='Learning rate scheduler for training.')

    # ---- Loading of Pretrained Models ----
    parser.add_argument('--pretrained', type=str, nargs='+', default=None, help='Pretrained model path.')
    parser.add_argument('--pretrained_exclude_vars', default=[], type=str, nargs='+',
                        help='List of variables to exclude loading.')
    parser.add_argument('--conv_layers_before_end_to_unfreeze', type=int, default=12,
                        help='Conv layers before the end to unfreeze.')
    parser.add_argument('--prefixes_of_vars_to_freeze', type=str, nargs='+',
                        help='List of variables with specified prefixes to freeze.')
    # Only used by ModelXtoChat_ChatToY
    parser.add_argument('--front_fc_layers_to_freeze', type=int, default=None,
                        help='FC layers to freeze from pretrained model.')

    # ---- Preprocessing ----
    parser.add_argument('--C_cols', default=CONCEPTS_BALANCED, nargs='+', help='Concepts to use.')
    parser.add_argument('--zscore', default='C', choices=['None', 'C', 'Y', 'CY'], help='Variables to zscore.')
    parser.add_argument('--augment', type=str, default='random_translation', choices=['None', 'random_translation'],
                        help='Image augmentation to use.')

    # ---- FC layers ----
    parser.add_argument('--fc_layers', type=int, nargs='+', default=None, help='FC layers after ResNet.')

    parser.add_argument('--C_fc_name', type=str, help='Select the FC layer to be used for C. EG. fc2')
    parser.add_argument('--y_fc_name', type=str, help='Select the FC layer to be used for y. EG. fc2')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Probability of dropping after each FC layer. 0 means no dropping.')

    # ---- Special experiments ----
    # train_X_to_C_with_freezed_convs
    parser.add_argument('--input_conv_layers', type=str, nargs='+', default=None,
                        help='Conv layers to use as inputs for training model.')
    # Oracle C to y related experiments
    parser.add_argument('--oracle_C_to_y_model', type=str, choices=['lr', 'svm', 'mlp'],
                        help='Model to use for oracle C -> y.')
    parser.add_argument('--oracle_C_to_y_model_params', type=str, default=None,
                        help='String representing a dictionary for the parameters of the oracle C to y model.')
    #   train_X_to_C_with_freezed_convs
    parser.add_argument('--use_senn_model', action='store_true', help='Whether to use the model from senn model.')

    # ---- Loss ----
    parser.add_argument('--C_weight', type=float, nargs='+', default=[1.0], help='Weights of concepts for loss.')
    parser.add_argument('--C_loss_type', default='reg', choices=['reg', 'cls'], help='Loss type for the C outputs.')
    parser.add_argument('--y_loss_type', default='reg', choices=['reg', 'cls'], help='Loss type for the y outputs.')

    # ---- Test time intervention ----
    parser.add_argument('--test_time_intervention_model', default=None,
                        choices=['X_to_C_to_y', 'X_to_y_with_aux_C', 'X_to_Chat__Chat_to_y', 'X_to_Chat__OracleC_to_y'],
                        help='Model to be used for test time inference.')
    parser.add_argument('--test_time_intervention_split', type=str, default='test', help='Dataset split to use.')
    parser.add_argument('--test_time_intervention_method', default=None,
                        choices=['random', 'ordered', 'max_info_gain', 'highest_uncertainty',
                                 'max_info_gain_with_dropout_model', 'highest_uncertainty_with_dropout_model'],
                        help='Method to select As to be given ground truth during test time.')
    parser.add_argument('--test_time_intervention_analysis', action='store_true',
                        help='Analyse the effects of test time inference instead of using a sampling scheme.')
    # Used for test_time_intervention_method of 'ordered'
    parser.add_argument('--intervention_order', type=int, nargs='+', default=None,
                        help='Order in which to provide intervention.')
    # Other arguments: --exp, --pretrained

    # ---- Hyperparameter optimization ----
    parser.add_argument('--hyperopt_model', type=str, choices=['X_to_y', 'X_to_Cy', 'X_to_C_to_y',
                                                               'X_to_Chat__Chat_to_y'],
                        help='Model to hyperparameter optimize.')
    parser.add_argument('--hyperopt_params', type=str, help='String representing a parameter dictionary that will be'
                        'used for the search. Will be parsed by json library.'
                        ' EG: ... --hyperopt_params \'{ "optimizer_kwargs.lr": [0.00005, 0.0005, 0.005] }\' '
                        ' This changes model_kwargs["optimizer_kwargs"]["lr"] to the specified values.')
    parser.add_argument('--hyperopt_additional', type=str, default='', help='String representing a dictionary that can'
                        'be used to pass additional configs needed. Currently only used for X_to_Chat__Chat_to_y.')
    parser.add_argument('--hyperopt_search', type=str, default='grid', choices=['random', 'grid'],
                        help='Search method for hyperparameters')
    parser.add_argument('--hyperopt_n_repeats', type=int, default=3,
                        help='Number of repeats for each parameter setting.')
    parser.add_argument('--hyperopt_score_metric', type=str, default='val_xrkl_rmse',
                        help='Metric to determine the goodness of each experiment.')
    parser.add_argument('--hyperopt_negate_score', action='store_true',
                        help='Multiply the metric above with -1 (used when model is better if metric is lower.')
    # Other arguments: Usual training algorithm parameters

    # ---- Imbalanced classes ----
    parser.add_argument('--C_loss_weigh_class', action='store_true',
                        help='Give rare classes of concepts more weights for loss.')
    parser.add_argument('--sampling', type=str, default='uniform', choices=['uniform', 'weigh_C', 'weigh_Cy'],
                        help='Sampling strategy for minibatch.')
    args = parser.parse_args()

    # Generate dataset and model configs according to experiment name
    dataset_kwargs, model_kwargs = generate_configs(args)

    import platform
    node_name = platform.node().split('.')[0]
    if not args.silent: print('Running code on %s' % node_name)

    if not args.silent: print('Argparse args', json.dumps(vars(args), indent=4))
    if not args.silent: print('Dataset kwargs', json.dumps(dataset_kwargs, indent=4))
    if not args.silent: print('Model kwargs', json.dumps(model_kwargs, indent=4))
    return args, dataset_kwargs, model_kwargs


if __name__ == '__main__':
    run_experiments(*parse_arguments())
