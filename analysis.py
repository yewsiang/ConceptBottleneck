
import pdb
import os
import sys
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, accuracy_score, precision_score, recall_score, balanced_accuracy_score, classification_report


# ---------------------- OAI ----------------------
def plot(x, y, **kw):
    if kw.get('multiple_plots'):
        # MANY lines on MANY plots
        assert not kw.get('multiple_plot_cols') is None
        ncols = kw['multiple_plot_cols']
        titles = kw['multiple_plot_titles']
        suptitle = kw['suptitle']
        sharex = kw['sharex'] if kw.get('sharex') else False
        sharey = kw['sharey'] if kw.get('sharey') else False
        nplots = len(x)
        nrows = np.ceil(nplots / ncols).astype(np.int32)
        fig_dims_w = 16
        fig_dims_h = nrows * (0.25 * fig_dims_w)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(fig_dims_w, fig_dims_h),
                                 sharex=sharex, sharey=sharey)
        if len(axes.shape) == 1: axes = axes[None,:]

        for n in range(nplots):
            i, j = n // ncols, n % ncols
            subplt = axes[i, j]
            for k, (x_, y_) in enumerate(zip(x[n], y[n])):
                plot_types = kw.get('plot_types')
                if plot_types:
                    plot_type, plot_args = plot_types[n][k]
                    if plot_type == 'line':
                        subplt.plot(x_, y_, **plot_args)
                    elif plot_type == 'scatter':
                        subplt.scatter(x_, y_, **plot_args)
                else:
                    subplt.plot(x_, y_)
                handle_plot_kwargs(subplt, **kw)
            subplt.set_title(titles[n])
        fig.suptitle(**suptitle)
        plt.tight_layout()
    else:
        # ONE line on ONE plot
        plt.plot(x, y)
        plot_template_ending(**kw)
    plt.show()

def handle_plot_kwargs(subplot=None, **kw):
    curr_plot = subplot if subplot else plt
    if kw.get('title'): curr_plot.title(kw['title'])
    if kw.get('xlabel'): curr_plot.xlabel(kw['xlabel'])
    if kw.get('ylabel'): curr_plot.ylabel(kw['ylabel'])
    if kw.get('margins'): curr_plot.margins(kw['margins'])
    if kw.get('xticks'): curr_plot.xticks(**kw['xticks'])
    if kw.get('yticks'): curr_plot.yticks(**kw['yticks'])
    if kw.get('xlim'): curr_plot.xlim(**kw['xlim'])
    if kw.get('ylim'): curr_plot.ylim(**kw['ylim'])
    if kw.get('set_xlim'): curr_plot.set_xlim(**kw['set_xlim'])
    if kw.get('set_ylim'): curr_plot.set_ylim(**kw['set_ylim'])
    if kw.get('subplots_adjust'): curr_plot.subplots_adjust(**kw['subplots_adjust'])

def plot_template_ending(**kw):
    # Standard template ending for the plots
    handle_plot_kwargs(**kw)
    plt.show()

def plot_violin(x_category, y, **kw):
    unique = np.unique(x_category)
    plot_x = range(len(unique))
    plot_y = [y[x_category == val] for val in unique]
    plt.violinplot(plot_y, plot_x, points=60, widths=0.7, showmeans=False, showextrema=True,
                   showmedians=True, bw_method=0.5)
    plot_template_ending(**kw)

def plot_rmse(y_true, y_pred, **kw):
    unique = np.unique(y_true)
    plot_x = range(len(unique))
    ids = [y_true == val for val in unique]
    rmses = [np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])) for idx in ids]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    kw['title'] = 'RMSE = %.3f' % rmse
    plot(plot_x, rmses, **kw)

def plot_distributions(data, names, discrete=True):
    assert data.shape[1] == len(names)
    x = data.astype(np.int32)

    nplots = data.shape[1]
    ncols = 4
    nrows = np.ceil(nplots / ncols).astype(np.int32)
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(12,12))
    for n in range(nplots):
        i, j = n // ncols, n % ncols
        data = x[:,n]
        if discrete:
            nbins = len(np.unique(data))
        axes[i, j].hist(data, bins=nbins)
        axes[i, j].set_title(names[n])
    plt.tight_layout()
    plt.show()

def assign_value_to_bins(value, bins, use_integer_bins=True):
    shape = value.shape
    value_vec = value.reshape(-1)
    dist = np.abs(value_vec[:,None] - bins[None,:])
    bin_id = np.argmin(dist, axis=1)
    if use_integer_bins:
        new_values = bin_id
    else:
        new_values = bins[bin_id]
    new_values = new_values.reshape(shape)
    return new_values

def convert_continuous_back_to_ordinal(y_true, y_pred, use_integer_bins=False):
    # Convert y_true into categories
    unique_y_true = np.unique(y_true)  # (C,)
    N_classes = len(unique_y_true)
    one_hot_y_true = (y_true[:, None] == unique_y_true[None, :]) # (N,C)
    cat_y_true = np.dot(one_hot_y_true, np.arange(N_classes))  # (N,)
    y_pred_binned_i = assign_value_to_bins(y_pred, unique_y_true, use_integer_bins=use_integer_bins)
    return y_pred_binned_i, cat_y_true

def assess_performance(y, yhat, names, prediction_type, prefix, verbose=False):
    """
    Return standard metrics of performance of y and yhat.
    """
    assert y.shape == yhat.shape, print('(%s) y: %s, yhat: %s' % (prefix, str(y.shape), str(yhat.shape)) )
    assert y.shape[1] == len(names), print('%s) y: %s, len(names): %d' % (prefix, str(y.shape), len(names)) )

    metrics = {}
    for i, name in enumerate(names):
        # This is to give each variable a unique key in the metrics dict
        prefix_name = '%s_%s_' % (prefix, name)
        # y and yhat can be (N,D), we analyse col by col
        y_i = y[:,i]
        yhat_i = yhat[:,i]

        if prediction_type == 'binary':
            assert set(np.unique(y_i)) == {0, 1}
            assert set(np.unique(yhat_i)) != {0, 1}
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=y_i, y_score=yhat_i)
            auc = sklearn.metrics.roc_auc_score(y_true=y_i, y_score=yhat_i)
            auprc = sklearn.metrics.average_precision_score(y_true=y_i, y_score=yhat_i)
            metrics.update({
                prefix_name+'auc': auc,
                prefix_name+'auprc': auprc,
                prefix_name+'tpr': tpr,
                prefix_name+'fpr': fpr
            })

        elif prediction_type == 'multiclass':
            precision, recall, fbeta, support = precision_recall_fscore_support(y_i, yhat_i)
            metrics.update({
                prefix_name+'precision': precision,
                prefix_name+'recall': recall,
                prefix_name+'fbeta': fbeta,
                prefix_name+'support': support,
                prefix_name+'macro_precision': np.mean(precision),
                prefix_name+'macro_recall': np.mean(recall),
                prefix_name+'macro_F1': np.mean(fbeta),
            })

        elif prediction_type in ['continuous', 'continuous_ordinal']:
            r = pearsonr(y_i, yhat_i)[0]
            spearman_r = spearmanr(y_i, yhat_i)[0]
            rmse = np.sqrt(np.mean((y_i - yhat_i) ** 2))
            metrics.update({
                prefix_name+'r': r,
                prefix_name+'rmse': rmse,
                prefix_name+'negative_rmse': -rmse,
                prefix_name+'r^2': r ** 2,
                prefix_name+'spearman_r': spearman_r,
                prefix_name+'spearman_r^2': spearman_r ** 2,
            })

            # Continuous ordinal means that the class is categorical & ordinal in nature but represented as continuous
            if prediction_type == 'continuous_ordinal':
                yhat_round_i, cat_y_i = convert_continuous_back_to_ordinal(y_i, yhat_i, use_integer_bins=True)
                precision, recall, fbeta, support = precision_recall_fscore_support(cat_y_i, yhat_round_i)

                metrics.update({
                    prefix_name+'precision': precision,
                    prefix_name+'recall': recall,
                    prefix_name+'F1': fbeta,
                    prefix_name+'acc': accuracy_score(cat_y_i, yhat_round_i),
                    prefix_name+'support': support,
                    prefix_name+'macro_precision': np.mean(precision),
                    prefix_name+'macro_recall': np.mean(recall),
                    prefix_name+'macro_F1': np.mean(fbeta),
                })

            metrics[prefix_name+'pred'] = yhat_i
            metrics[prefix_name+'true'] = y_i

        if verbose:
            if prediction_type in ['multiclass', 'continuous_ordinal']:
                N_classes = len(np.unique(y_i))
                out = ('%11s |' % prefix_name[:-1]) + ('%8s|' * N_classes) % tuple([str(i) for i in range(N_classes)])
                for metric in ['precision', 'recall', 'F1', 'support']:
                    out += ('\n%11s |' % metric)
                    for cls_id in range(N_classes):
                        if metric == 'support':
                            out += ' %6d |' % (metrics[prefix_name+metric][cls_id])
                        else:
                            out += '  %04.1f  |' % (metrics[prefix_name+metric][cls_id] * 100.)
                out += '\nMacro precision: %2.1f' % ((metrics[prefix_name+'macro_precision']) * 100.)
                out += '\nMacro recall   : %2.1f' % ((metrics[prefix_name+'macro_recall']) * 100.)
                out += '\nMacro F1       : %2.1f' % ((metrics[prefix_name+'macro_F1']) * 100.)
                print(out)

    for metric in metrics:
        metric_type = '_'.join(metric.split('_')[2:])
        if metric_type in ['tpr', 'fpr', 'precision', 'recall', 'F1', 'support', 'pred', 'true']:
            continue
        if np.isnan(metrics[metric]):
            pass
            # print(metric, metrics[metric])
            # raise Exception("%s is a nan, something is weird about your predictor" % metric)
    return metrics

# ---------------------- CUB ----------------------
class Logger(object):
    """
    Log results to a file and flush() to view instant updates
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output and target are Torch tensors
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def binary_accuracy(output, target):
    """
    Computes the accuracy for multiple binary predictions
    output and target are Torch tensors
    """
    pred = output.cpu() >= 0.5
    #print(list(output.data.cpu().numpy()))
    #print(list(pred.data[0].numpy()))
    #print(list(target.data[0].numpy()))
    #print(pred.size(), target.size())
    acc = (pred.int()).eq(target.int()).sum()
    acc = acc*100 / np.prod(np.array(target.size()))
    return acc

def multiclass_metric(output, target):
    """
    Return balanced accuracy score (average of recall for each class) in case of class imbalance,
    and classification report containing precision, recall, F1 score for each class
    """
    balanced_acc = balanced_accuracy_score(target, output)
    report = classification_report(target, output)
    return balanced_acc, report
