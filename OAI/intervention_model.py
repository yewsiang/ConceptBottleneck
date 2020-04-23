
import torch
import numpy as np
from OAI.template_model import extend_dicts


class InterventionModelOnC(object):
    """
    This class implements some functions that provide intervention methods on C (concepts).
    It must be inherited together with some other deep learning model.
    """
    def __init__(self, cfg):
        # The number of concepts given during test time inference
        self.test_time_intervention_N_concepts_gven = None
        self.test_time_intervention_method = cfg['test_time_intervention_method']
        self.intervention_order = cfg['intervention_order']

    def set_intervention_N_concepts(self, N_concepts):
        self.test_time_intervention_N_concepts_gven = N_concepts

    def forward_with_intervention(self, inputs, labels):

        # These parameters must not be None
        assert self.test_time_intervention_method
        assert self.test_time_intervention_N_concepts_gven is not None

        # --------- Repeat of forward ---------
        x = inputs['image']
        pooled = self.compute_cnn_features(x)
        x = self.dropout(pooled)
        # -------------------------------------

        # Usage of ground truth labels as intervention
        C_feats = labels['C_feats']

        outputs = {}  # { 'pool': x }
        N_layers = len(self.fc_layers)
        for i, layer in enumerate(self.fc_layers):
            fc_name = 'fc' + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)
            if fc_name == self.C_fc_name:

                # ---------- Overriding the outputs ----------
                # Note that overriding will be different depending on whether outputs are cls / reg
                N_samples = 10
                B, N_concepts = x.shape
                N_given = self.test_time_intervention_N_concepts_gven

                if self.test_time_intervention_method == 'random':
                    if self.C_loss_type == 'reg':
                        rand_ids = np.array([np.random.choice(range(N_concepts), N_given,
                                                              replace=False) for _ in range(B)]) # (B,N_given)
                        for ids in rand_ids.T:
                            x[np.arange(B), ids] = C_feats[np.arange(B), ids]

                    elif self.C_loss_type == 'cls':
                        raise NotImplementedError()

                elif self.test_time_intervention_method == 'ordered':
                    if self.C_loss_type == 'reg':
                        ordered_ids = self.intervention_order[:N_given]
                        x[:, ordered_ids] = C_feats[:, ordered_ids]

                    elif self.C_loss_type == 'cls':
                        raise NotImplementedError()

                outputs['C'] = x
                continue
            elif fc_name == self.y_fc_name:
                assert i == N_layers - 1
                # No ReLu for y layer
                outputs['y'] = x
                continue

            x = self.relu(x)
            # x = self.dropout(x)
            # Can choose to keep track of this as well
            # outputs[fc_name] = x

        return outputs

    def intervention_analysis(self, dataloaders, dataset_sizes, phase):
        """
        Instead of intervening with some sampling scheme, perform an analysis of results and
        """
        assert phase in ['val', 'test']
        use_gpu = torch.cuda.is_available()
        self.train(False)  # Set model to evaluate mode

        n_batches_loaded = 0

        # Iterate over data.
        concatenated_labels = {}
        concatenated_outputs = {}
        for data in dataloaders[phase]:
            n_batches_loaded += 1
            if n_batches_loaded % 100 == 0: print('Processed %d/%d' % (n_batches_loaded, len(dataloaders[phase])))

            # Get the inputs
            data_dict = self.get_data_dict_from_dataloader(data)
            inputs = data_dict['inputs']
            labels = data_dict['labels']

            # Forward
            outputs = self.intervention_analysis_step(inputs, labels)

            # Keep track of everything for correlations
            extend_dicts(concatenated_labels, labels)
            extend_dicts(concatenated_outputs, outputs)

        return { 'pred': concatenated_outputs,
                 'true': concatenated_labels }

    def intervention_analysis_step(self, inputs, labels):

        # --------- Repeat of forward ---------
        x = inputs['image']
        pooled = self.compute_cnn_features(x)
        # x = self.dropout(pooled)
        x = pooled
        # -------------------------------------

        # Usage of ground truth labels as intervention
        C_feats = labels['C_feats']

        outputs = {}  # { 'pool': x }
        N_layers = len(self.fc_layers)
        N_concepts = len(self.C_cols) #self.fc_layers[int(self.C_fc_name[2:]) - 1]

        outputs_C = []
        outputs_y = []
        for a_id in range(N_concepts + 1):
            x = pooled
            for i, layer in enumerate(self.fc_layers):
                fc_name = 'fc' + str(i + 1)
                fn = getattr(self, fc_name)
                x = fn(x)
                if fc_name == self.C_fc_name:

                    # ---------- Overriding the outputs ----------
                    # Keep the actual Ahats for the first run
                    if a_id > 0:
                        x[:, a_id - 1] = C_feats[:, a_id - 1]
                    outputs_C.append(x)
                    continue
                elif fc_name == self.y_fc_name:
                    assert i == N_layers - 1
                    # No ReLu for y layer
                    outputs_y.append(x)
                    continue

                x = self.relu(x)
                # x = self.dropout(x)
                # Can choose to keep track of this as well
                # outputs[fc_name] = x
        outputs['C'] = torch.stack(outputs_C, dim=2)
        outputs['y'] = torch.stack(outputs_y, dim=2)
        return outputs

    def get_intervention_ordering(self, results, discretize=False):
        pred = results['pred']
        true = results['true']
        y_true = true['y']  # (N,1)
        C_true = true['C_feats']  # (N,C)
        C_pred = pred['C']  # (N,C,C+1)
        C_pred_orig = C_pred[:, :, 0]  # (N,C)
        y_pred_orig = pred['y'][:, 0, :1]  # (N,1)
        y_pred_tti = pred['y'][:, 0, 1:]  # (N,C)
        N, C = C_true.shape

        if discretize:
            # Discretization
            C_true_list = []
            C_pred_list = []
            for i in range(C):
                C_true_list.append(convert_continuous_back_to_ordinal(C_true[:, i], C_true[:, i], use_integer_bins=True)[0])
                C_pred_list.append(
                    convert_continuous_back_to_ordinal(C_true[:, i], C_pred_orig[:, i], use_integer_bins=True)[0])
            C_true = np.stack(C_true_list, axis=1)
            C_pred_orig = np.stack(C_pred_list, axis=1)
            y_pred_orig, _ = convert_continuous_back_to_ordinal(y_true, y_pred_orig, use_integer_bins=True)
            y_pred_tti, _ = convert_continuous_back_to_ordinal(y_true, y_pred_tti, use_integer_bins=True)

        # Analyse change in error
        y_error_orig = np.abs(y_pred_orig - y_true)  # (N,1)
        y_error_tti = np.abs(y_pred_tti - y_true)  # (N,C)
        # Change in error after TTI. Lower is better
        y_error_delta = y_error_tti - y_error_orig  # (N,C)

        # Find best improvement per example
        y_best_improvement_C_ids = np.argmin(y_error_delta, axis=1)  # (N,)
        y_improvement = y_error_delta[np.arange(N), y_best_improvement_C_ids]  # (N)

        # Find best improvement amongst all examples
        # y_improvement_asc_ids = np.argsort(y_improvement)

        # Analysis of As to provide an ordering for intervention on Cs
        # In order of best improvement
        best_improvement_ordering = np.argsort(-np.bincount(y_best_improvement_C_ids))
        print('Best improvement intervention: %s' % str(best_improvement_ordering))
        # In order of highest C error
        # mean_abs_error = np.mean(np.abs(C_pred_orig - C_true), axis=0)  # (C,)
        # worst_error_ordering = np.argsort(-mean_abs_error)
        # print('     Worst error intervention: %s\n' % str(worst_error_ordering))

        return best_improvement_ordering
