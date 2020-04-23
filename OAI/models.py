
import pdb
import torch
import torch.nn as nn
import analysis
import numpy as np

from sklearn.svm import SVR
from torch.autograd import Variable
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from OAI.template_model import PretrainedResNetModel
from OAI.intervention_model import InterventionModelOnC


class ModelXtoC(PretrainedResNetModel):
    def __init__(self, cfg):
        super().__init__(cfg, build=False)
        self.C_cols = cfg['C_cols']

        # Whether to use different conv layers
        self.use_input_conv_layers = cfg['use_input_conv_layers']
        self.input_conv_layers = cfg['input_conv_layers']
        if self.use_input_conv_layers:
            # If we want to use input_conv_layers, we need to redefine fc1
            previous_layer_dims = sum([self.conv_layer_dims[layer] for layer in self.input_conv_layers])
            self.fc1 = nn.Linear(previous_layer_dims, self.fc_layers[0])
        self.build()

        # FC layers to use for outputs
        self.C_fc_name = cfg['C_fc_name']

        self.C_loss_weigh_class = cfg['C_loss_weigh_class']
        self.additional_loss_weighting = cfg['additional_loss_weighting']
        self.metric_to_use_as_stopping_criterion = 'val_epoch_neg_loss'

    def get_data_dict_from_dataloader(self, data):
        # Retrieves the relevant data from dataloader and store into a dict
        X = data['image']  # X
        C_feats = data['C_feats']
        C_feats_not_nan = data['C_feats_not_nan']
        C_feats_loss_class_wts = data['C_feats_loss_class_wts']

        # Wrap them in Variable
        assert len(self.C_cols) > 0
        X = Variable(X.float().cuda())
        C_feats = Variable(C_feats.float().cuda())
        C_feats_not_nan = Variable(C_feats_not_nan.float().cuda())

        inputs = { 'image': X }
        labels = { 'C_feats': C_feats,
                   'C_feats_not_nan': C_feats_not_nan,
                   'C_feats_loss_class_wts': C_feats_loss_class_wts }

        data_dict = {
            'inputs': inputs,  # Will be used to compute outputs = self.foward(inputs)
            'labels': labels,  # Will be used for comparison with outputs
        }
        return data_dict

    def forward(self, inputs):
        x = inputs['image']
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)
        convs = {
            'conv1': conv1,
            'conv2': conv2,
            'conv3': conv3,
            'conv4': conv4,
        }
        if self.use_input_conv_layers:
            res_convs = [self.avgpool(convs[layer]) for layer in self.input_conv_layers]
            res_convs = [conv.view(conv.size(0), -1) for conv in res_convs]
            x = torch.cat(res_convs, dim=1)
        else:
            x = self.avgpool(conv4)
            x = x.view(x.size(0), -1)

        outputs = {} # { 'pool': x }
        N_layers = len(self.fc_layers)
        for i, layer in enumerate(self.fc_layers):
            fc_name = 'fc' + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)
            if fc_name == self.C_fc_name:
                assert i == N_layers - 1 # To ensure that we are at last layer
                # No ReLu
                outputs['C'] = x
                continue

            x = self.relu(x)
            # Can choose to keep track of this as well
            # outputs[fc_name] = x

        return outputs

    def loss(self, outputs, data_dict):
        # Data
        C_feats = data_dict['labels']['C_feats']
        C_feats_not_nan = data_dict['labels']['C_feats_not_nan']
        C_feats_loss_class_wts = data_dict['labels']['C_feats_loss_class_wts']

        # Parse outputs from deep model
        C_hat = outputs['C']
        assert C_hat.shape[1] == len(self.C_cols), print('C_hat: %s, len C_cols: %s' % (C_hat.shape, len(self.C_cols)))

        # Loss for A
        # Compute loss only if feature is not NaN.
        loss = ((C_feats - C_hat) ** 2) * C_feats_not_nan
        # We upweigh rare classes (within each concept) to allow the model to pay attention to it.
        if self.C_loss_weigh_class:
            loss_class_wts = C_feats_loss_class_wts
            loss *= loss_class_wts.float().cuda()
        loss *= torch.FloatTensor([self.additional_loss_weighting]).cuda()
        loss = loss.sum(dim=1).mean(dim=0)
        loss_details = {}
        return loss, loss_details

    def analyse_predictions(self, y_true, y_pred, info={}):
        # This function is called at the end of every train / val epoch in DeepLearningModel and is
        # used to analyse the predictions over the entire train / val dataset.
        metrics_all = {}
        phase = info['phase']
        dataset_size = info['dataset_size']
        epoch_loss = info['epoch_loss']

        C_hat = y_pred['C']
        C = y_true['C_feats']
        assert len(C_hat) == dataset_size

        metrics_C = analysis.assess_performance(y=C, yhat=C_hat,
                                                names=self.C_cols,
                                                prediction_type='continuous_ordinal',
                                                prefix=phase)

        rmses = [metrics_C['%s_%s_rmse' % (phase, C_col)] for C_col in self.C_cols]
        print('%s epoch loss for C: %2.4f (n=%i)' % (phase, epoch_loss, len(C_hat)))
        print('Average RMSE: %2.4f, RMSEs: %s' % (np.mean(rmses), str(rmses)))
        metrics_all['%s_epoch_loss' % phase] = epoch_loss
        metrics_all['%s_epoch_neg_loss' % phase] = -epoch_loss

        metrics_all.update(metrics_C)
        return metrics_all

class ModelXtoC_SENN(ModelXtoC):
    def load_pretrained(self):
        prefix = 'conceptizer.'
        # Our own trained model
        assert self.pretrained_path is not None
        print('[A] Loading our own pretrained model')
        own_state = self.state_dict()
        pretrained_state = torch.load(self.pretrained_path)['state_dict']

        for name, param in pretrained_state.items():
            encoder_fc_prefix = prefix + 'fc'
            if not name.startswith(prefix) or any([name.startswith(var) for var in self.pretrained_exclude_vars]):
                print('  Skipping %s' % name)
                continue

            print('  Loading %s' % name)
            if name.startswith(encoder_fc_prefix) and not self.use_input_conv_layers:
                # Only restore SENN's FC layers if we are going from SENN Output -> C instead of Convs -> A
                own_state_name = 'fc1' + name[len(encoder_fc_prefix):]
                own_state[own_state_name].copy_(param)
            else:
                assert not name.startswith('fc')
                own_state_name = name[len(prefix):]
                own_state[own_state_name].copy_(param)
        return

class ModelOracleCtoY(object):
    def __init__(self, model_type='lr', new_kwargs={}):
        if model_type == 'lr':
            model = LinearRegression
            model_kwargs = {}
        elif model_type == 'svm':
            model = SVR
            model_kwargs = {'C': 5.0,
                            'kernel': 'rbf',
                            'gamma': 'scale'}
            if new_kwargs:
                model_kwargs.update(new_kwargs)
        elif model_type == 'mlp':
            model = MLPRegressor
            model_kwargs = {'hidden_layer_sizes': [50, 50],
                            'activation': 'relu',
                            'solver': 'adam',
                            'alpha': 0.,
                            'learning_rate': 'adaptive',
                            'learning_rate_init': 0.001,
                            'batch_size': 16,
                            'max_iter': 2000,
                            'early_stopping': True}
            if new_kwargs:
                model_kwargs.update(new_kwargs)
        self.model = model(**model_kwargs)

    def fit(self, C, y):
        return self.model.fit(C, y)

    def predict(self, C):
        return self.model.predict(C)

    def analyse_predictions(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape, print('y_true: %s, y_pred: %s' % (y_true.shape, y_pred.shape))
        metrics = {
            'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2))
        }
        return metrics

class ModelXtoChat_ChatToY(InterventionModelOnC, PretrainedResNetModel):
    """
    In this model, we assume a pretrained X -> Chat model (freezed), then we train the Chat -> y model.
    """
    def __init__(self, cfg, build=True):
        self.front_fc_layers_to_freeze = cfg['front_fc_layers_to_freeze']
        InterventionModelOnC.__init__(self, cfg)
        PretrainedResNetModel.__init__(self, cfg, build=build)

        self.C_cols = cfg['C_cols']
        self.y_cols = cfg['y_cols']

        # FC layers to use for outputs
        self.C_fc_name = cfg['C_fc_name']
        self.y_fc_name = cfg['y_fc_name']

        self.C_loss_type = cfg['C_loss_type']
        self.y_loss_type = cfg['y_loss_type']
        self.classes_per_C_col = cfg['classes_per_C_col']
        self.classes_per_y_col = cfg['classes_per_y_col']

        self.metric_to_use_as_stopping_criterion = 'val_xrkl_negative_rmse'

    def load_pretrained(self):
        # Load our X->C model
        if self.pretrained_path:
            print('Loading our own pretrained model')
            incompatible, unexpected = self.load_state_dict(torch.load(self.pretrained_path), strict=False)
            print('Incompatible weights: %s' % str(incompatible))
            print('Unexpected weights  : %s' % str(unexpected))
            assert len(unexpected) == 0
            return

    def unfreeze_conv_layers(self, conv_layers_before_end_to_unfreeze):
        to_be_trained = []
        for i, layer in enumerate(self.fc_layers):
            if i >= self.front_fc_layers_to_freeze:
                to_be_trained.append('fc' + str(i + 1) + '.weight')
                to_be_trained.append('fc' + str(i + 1) + '.bias')
        print('FC layers to train: %s' % str(to_be_trained))

        for name, param in self.named_parameters():
            if name in to_be_trained:
                # We always unfreeze these layers.
                print("Param %s is UNFROZEN" % name, param.data.shape)
            else:
                print("Param %s is FROZEN" % name, param.data.shape)
                param.requires_grad = False

    def get_data_dict_from_dataloader(self, data):
        # Retrieves the relevant data from dataloader and store into a dict
        X = data['image']  # X
        y = data['y']  # y
        C_feats = data['C_feats']
        C_feats_not_nan = data['C_feats_not_nan']

        # wrap them in Variable
        X = Variable(X.float().cuda())
        y = Variable(y.float().cuda())
        if len(self.C_cols) > 0:
            C_feats = Variable(C_feats.float().cuda())
            C_feats_not_nan = Variable(C_feats_not_nan.float().cuda())

        inputs = { 'image': X }
        labels = { 'y': y,
                   'C_feats': C_feats,
                   'C_feats_not_nan': C_feats_not_nan }

        data_dict = {
            'inputs': inputs,
            'labels': labels,
        }
        return data_dict

    def forward(self, inputs):
        x = inputs['image']
        x = self.compute_cnn_features(x)

        outputs = {}  # { 'pool': x }
        N_layers = len(self.fc_layers)
        for i, layer in enumerate(self.fc_layers):
            fc_name = 'fc' + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)
            if fc_name == self.C_fc_name:
                # No ReLu for concept layer
                outputs['C'] = x
                continue
            elif fc_name == self.y_fc_name:
                assert i == N_layers - 1
                # No ReLu for y layer
                outputs['y'] = x
                continue

            x = self.relu(x)
            # Can choose to keep track of this as well
            # outputs[fc_name] = x

        return outputs

    def loss(self, outputs, data_dict):
        # Data
        y = data_dict['labels']['y']

        # Parse outputs from deep model
        y_hat = outputs['y']
        assert y_hat.shape[1] == len(self.y_cols), print('y_hat: %s, len y_cols: %s' % (y_hat.shape, len(self.y_cols)))

        # Loss for y
        loss = nn.MSELoss()(input=y_hat, target=y)
        loss_details = {}
        return loss, loss_details

    def analyse_predictions(self, y_true, y_pred, info={}):
        # This function is called at the end of every train / val epoch in DeepLearningModel and is
        # used to analyse the predictions over the entire train / val dataset.
        metrics_all = {}
        phase = info['phase']
        dataset_size = info['dataset_size']
        epoch_loss = info['epoch_loss']

        y_hat = y_pred['y']
        C_hat = y_pred['C']
        y = y_true['y']
        C = y_true['C_feats']
        assert len(y_hat) == dataset_size

        metrics_y = analysis.assess_performance(y=y, yhat=y_hat,
                                                names=self.y_cols,
                                                prediction_type='continuous_ordinal',
                                                prefix=phase,
                                                verbose=True)
        metrics_C = analysis.assess_performance(y=C, yhat=C_hat,
                                                names=self.C_cols,
                                                prediction_type='continuous_ordinal',
                                                prefix=phase)

        print('%s epoch loss for %s: %2.6f; RMSE %2.6f; correlation %2.6f (n=%i)' %
              (phase, str(self.y_cols), epoch_loss, metrics_y['%s_xrkl_rmse' % phase], metrics_y['%s_xrkl_r' % phase],
               len(y_hat)))
        metrics_all['%s_epoch_loss' % phase] = epoch_loss

        rmses = [metrics_C['%s_%s_rmse' % (phase, C_col)] for C_col in self.C_cols]
        print('  Average RMSE for C: %2.4f, RMSEs: %s' % (np.mean(rmses), str(rmses)))

        metrics_all.update(metrics_y)
        metrics_all.update(metrics_C)
        return metrics_all

class ModelXtoY(PretrainedResNetModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.y_cols = cfg['y_cols']

        # FC layers to use for outputs
        self.y_fc_name = cfg['y_fc_name']

        self.metric_to_use_as_stopping_criterion = 'val_xrkl_negative_rmse'

    def get_data_dict_from_dataloader(self, data):
        # Retrieves the relevant data from dataloader and store into a dict
        X = data['image']  # X
        y = data['y']  # y

        # Wrap them in Variable
        X = Variable(X.float().cuda())
        y = Variable(y.float().cuda())

        inputs = { 'image': X }
        labels = { 'y': y }

        data_dict = {
            'inputs': inputs, # Will be used to compute outputs = self.foward(inputs)
            'labels': labels, # Will be used for comparison with outputs (Might contain C + y). *Must be same dims*.
        }
        return data_dict

    def forward(self, inputs):
        x = inputs['image']
        x = self.compute_cnn_features(x)

        outputs = {} # { 'pool': x }
        N_layers = len(self.fc_layers)
        for i, layer in enumerate(self.fc_layers):
            fc_name = 'fc' + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)
            if fc_name == self.y_fc_name:
                assert i == N_layers - 1 # To ensure that we are at last layer
                outputs['y'] = x # No ReLu
                continue

            x = self.relu(x)
            x = self.dropout(x)
            # Can choose to keep track of this as well
            # outputs[fc_name] = x

        return outputs

    def loss(self, outputs, data_dict):
        # Data
        y = data_dict['labels']['y']

        # Parse outputs from deep model
        y_hat = outputs['y']
        assert y_hat.shape[1] == len(self.y_cols), print('y_hat: %s, len y_cols: %s' % (y_hat.shape, len(self.y_cols)))

        # Loss for y
        loss = nn.MSELoss()(input=y_hat, target=y)
        loss_details = {}
        return loss, loss_details

    def analyse_predictions(self, y_true, y_pred, info={}):
        # This function is called at the end of every train / val epoch in DeepLearningModel and is
        # used to analyse the predictions over the entire train / val dataset.
        metrics_all = {}
        phase = info['phase']
        dataset_size = info['dataset_size']
        epoch_loss = info['epoch_loss']

        y_hat = y_pred['y']
        y = y_true['y']
        assert len(y_hat) == dataset_size

        metrics_y = analysis.assess_performance(y=y, yhat=y_hat,
                                                names=self.y_cols,
                                                prediction_type='continuous_ordinal',
                                                prefix=phase,
                                                verbose=True)

        print('%s epoch loss for %s: %2.6f; RMSE %2.6f; correlation %2.6f (n=%i)' %
              (phase, str(self.y_cols), epoch_loss, metrics_y['%s_xrkl_rmse' % phase], metrics_y['%s_xrkl_r' % phase],
               len(y_hat)))
        metrics_all['%s_epoch_loss' % phase] = epoch_loss

        metrics_all.update(metrics_y)
        return metrics_all

class ModelXtoCY(PretrainedResNetModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.C_cols = cfg['C_cols']
        self.y_cols = cfg['y_cols']

        # FC layers to use for outputs
        self.C_fc_name = cfg['C_fc_name']
        self.y_fc_name = cfg['y_fc_name']

        self.C_loss_type = cfg['C_loss_type']
        self.y_loss_type = cfg['y_loss_type']
        self.classes_per_C_col = cfg['classes_per_C_col']
        self.classes_per_y_col = cfg['classes_per_y_col']

        self.C_loss_weigh_class = cfg['C_loss_weigh_class']
        self.additional_loss_weighting = cfg['additional_loss_weighting']
        self.metric_to_use_as_stopping_criterion = 'val_xrkl_negative_rmse'

    def get_data_dict_from_dataloader(self, data):
        # Retrieves the relevant data from dataloader and store into a dict
        X = data['image']  # X
        y = data['y']  # y
        C_feats = data['C_feats']
        C_feats_not_nan = data['C_feats_not_nan']
        C_feats_loss_class_wts = data['C_feats_loss_class_wts']

        # Wrap them in Variable
        X = Variable(X.float().cuda())
        y = Variable(y.float().cuda())
        if len(self.C_cols) > 0:
            C_feats = Variable(C_feats.float().cuda())
            C_feats_not_nan = Variable(C_feats_not_nan.float().cuda())

        inputs = { 'image': X }
        labels = { 'y': y,
                   'C_feats': C_feats,
                   'C_feats_not_nan': C_feats_not_nan,
                   'C_feats_loss_class_wts': C_feats_loss_class_wts }

        data_dict = {
            'inputs': inputs,
            'labels': labels,
        }
        return data_dict

    def forward(self, inputs):
        x = inputs['image']
        x = self.compute_cnn_features(x)

        outputs = {} # { 'pool': x }
        N_layers = len(self.fc_layers)
        for i, layer in enumerate(self.fc_layers):
            fc_name = 'fc' + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)
            if fc_name == self.y_fc_name:
                assert i == N_layers - 1 # To ensure that we are at last layer
                assert fc_name == self.C_fc_name # C + y should be at the same fc layer
                N_y_cols = len(self.y_cols)
                N_C_cols = len(self.C_cols)
                # No ReLu
                outputs['y'] = x[:, :N_y_cols]
                outputs['C'] = x[:, N_y_cols:N_y_cols + N_C_cols]
                continue

            x = self.relu(x)
            # Can choose to keep track of this as well
            # outputs[fc_name] = x

        return outputs

    def loss(self, outputs, data_dict):
        # Data
        y = data_dict['labels']['y']
        C_feats = data_dict['labels']['C_feats']
        C_feats_not_nan = data_dict['labels']['C_feats_not_nan']
        C_feats_loss_class_wts = data_dict['labels']['C_feats_loss_class_wts']

        # Parse outputs from deep model
        C_hat = outputs['C']
        y_hat = outputs['y']

        # Loss for y
        if self.y_loss_type == 'reg':
            loss_y = nn.MSELoss()(input=y_hat, target=y)
        elif self.y_loss_type == 'cls':
            y_long = y.long()
            loss_y = []
            start_id = 0
            for i, N_cls in enumerate(self.classes_per_y_col):
                loss_y.append(nn.CrossEntropyLoss(reduction='none')(y_hat[:, start_id:start_id + N_cls],
                                                                    y_long[:, i]))
                start_id += N_cls
            loss_y = torch.stack(loss_y, dim=1)
            loss_y = loss_y.sum(dim=1).mean(dim=0)
            assert start_id == sum(self.classes_per_y_col)

        # Loss for C
        if self.C_loss_type == 'reg':
            loss_C = ((C_feats - C_hat) ** 2)
        elif self.C_loss_type == 'cls':
            C_feats_long = C_feats.long()
            loss_C = []
            start_id = 0
            for i, N_cls in enumerate(self.classes_per_C_col):
                loss_C.append(nn.CrossEntropyLoss(reduction='none')(C_hat[:, start_id:start_id + N_cls],
                                                                    C_feats_long[:, i]))
                start_id += N_cls
            loss_C = torch.stack(loss_C, dim=1)
            assert start_id == sum(self.classes_per_C_col)

        # Compute loss only if feature is not NaN.
        loss_C = loss_C * C_feats_not_nan
        # We upweigh rare classes (within each concept) to allow the model to pay attention to it.
        if self.C_loss_weigh_class:
            loss_class_wts = C_feats_loss_class_wts
            loss_C *= loss_class_wts.float().cuda()
        loss_C *= torch.FloatTensor([self.additional_loss_weighting]).cuda()
        loss_C = loss_C.sum(dim=1).mean(dim=0)

        # Final loss
        loss = loss_y + loss_C
        loss /= (sum(self.additional_loss_weighting) + 1.)

        loss_y_float = loss_y.data.cpu().numpy().flatten()
        loss_C_float = loss_C.data.cpu().numpy().flatten()
        loss_ratio = loss_y_float / loss_C_float

        # Use y only and no C
        # loss = nn.MSELoss()(input=y_hat, target=y)

        loss_details = {
            'loss_ratio': loss_ratio
        }
        return loss, loss_details

    def analyse_predictions(self, y_true, y_pred, info={}):
        # This function is called at the end of every train / val epoch in DeepLearningModel and is
        # used to analyse the predictions over the entire train / val dataset.
        metrics_all = {}
        phase = info['phase']
        dataset_size = info['dataset_size']
        epoch_loss = info['epoch_loss']
        loss_ratios = np.concatenate([x['loss_ratio'] for x in info['loss_details']])

        print('Loss_y divided by loss_C is %2.3f (median ratio across batches)' % np.median(loss_ratios))

        y_hat = y_pred['y']
        C_hat = y_pred['C']
        y = y_true['y']
        C = y_true['C_feats']
        assert len(y_hat) == dataset_size

        if self.y_loss_type == 'cls':
            y_hat_new = []
            start_id = 0
            for i, N_cls in enumerate(self.classes_per_y_col):
                y_hat_new.append(np.argmax(y_hat[:, start_id:start_id + N_cls], axis=1))
                start_id += N_cls
            y_hat = np.stack(y_hat_new, axis=1).astype(np.float32)
            assert start_id == sum(self.classes_per_y_col)

        if self.C_loss_type == 'cls':
            C_hat_new = []
            start_id = 0
            for i, N_cls in enumerate(self.classes_per_C_col):
                C_hat_new.append(np.argmax(C_hat[:, start_id:start_id + N_cls], axis=1))
                start_id += N_cls
            C_hat = np.stack(C_hat_new, axis=1).astype(np.float32)
            assert start_id == sum(self.classes_per_C_col)

        metrics_y = analysis.assess_performance(y=y, yhat=y_hat,
                                                names=self.y_cols,
                                                prediction_type='continuous_ordinal',
                                                prefix=phase,
                                                verbose=True)
        metrics_C = analysis.assess_performance(y=C, yhat=C_hat,
                                                names=self.C_cols,
                                                prediction_type='continuous_ordinal',
                                                prefix=phase)

        print('%s epoch loss for %s: %2.6f; RMSE %2.6f; correlation %2.6f (n=%i)' %
              (phase, str(self.y_cols), epoch_loss, metrics_y['%s_xrkl_rmse' % phase], metrics_y['%s_xrkl_r' % phase],
               len(y_hat)))
        metrics_all['%s_epoch_loss' % phase] = epoch_loss

        rmses = [metrics_C['%s_%s_rmse' % (phase, C_col)] for C_col in self.C_cols]
        print('  Average RMSE for C: %2.4f, RMSEs: %s' % (np.mean(rmses), str(rmses)))

        metrics_all.update(metrics_y)
        metrics_all.update(metrics_C)
        return metrics_all

class ModelXtoYWithAuxC(InterventionModelOnC, ModelXtoCY):
    def __init__(self, cfg):
        InterventionModelOnC.__init__(self, cfg)
        ModelXtoCY.__init__(self, cfg)

    def forward(self, inputs):
        x = inputs['image']
        x = self.compute_cnn_features(x)
        x = self.dropout(x)

        outputs = {}  # { 'pool': x }
        N_layers = len(self.fc_layers)
        N_concepts = len(self.C_cols)

        for i, layer in enumerate(self.fc_layers):
            fc_name = 'fc' + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)
            if fc_name == self.C_fc_name:
                # No ReLu for concept layer
                x_concepts = x[:, :N_concepts]
                x_latent = self.relu(x[:, N_concepts:])
                x = torch.cat([x_concepts, x_latent], dim=1)
                outputs['C'] = x_concepts
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
        N_concepts = len(self.C_cols)

        for i, layer in enumerate(self.fc_layers):
            fc_name = 'fc' + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)
            if fc_name == self.C_fc_name:

                # ---------- Overriding the outputs ----------
                # Note that overriding will be different depending on whether outputs are cls / reg
                N_samples = 10
                B, _ = x.shape
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

                x_concepts = x[:, :N_concepts]
                x_latent = self.relu(x[:, N_concepts:])
                x = torch.cat([x_concepts, x_latent], dim=1)
                outputs['C'] = x_concepts
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
        N_concepts = len(self.C_cols)

        outputs_C = []
        outputs_y = []
        for C_id in range(N_concepts + 1):
            x = pooled
            for i, layer in enumerate(self.fc_layers):
                fc_name = 'fc' + str(i + 1)
                fn = getattr(self, fc_name)
                x = fn(x)
                if fc_name == self.C_fc_name:

                    # ---------- Overriding the outputs ----------
                    # Keep the actual Ahats for the first run
                    if C_id > 0:
                        x[:, C_id - 1] = C_feats[:, C_id - 1]

                    x_concepts = x[:, :N_concepts]
                    x_latent = self.relu(x[:, N_concepts:])
                    x = torch.cat([x_concepts, x_latent], dim=1)
                    outputs_C.append(x_concepts)
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

class ModelXtoCtoY(InterventionModelOnC, ModelXtoCY):
    def __init__(self, cfg):
        InterventionModelOnC.__init__(self, cfg)
        ModelXtoCY.__init__(self, cfg)

    def forward(self, inputs):
        x = inputs['image']
        x = self.compute_cnn_features(x)
        x = self.dropout(x)

        outputs = {}  # { 'pool': x }
        N_layers = len(self.fc_layers)
        for i, layer in enumerate(self.fc_layers):
            fc_name = 'fc' + str(i + 1)
            fn = getattr(self, fc_name)
            x = fn(x)
            if fc_name == self.C_fc_name:
                # No ReLu for concept layer
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
