
# Global dependencies
from OAI.config import IMG_DIR_PATH, CLINICAL_CONTROL_COLUMNS, IMG_CODES_FILENAME, NON_IMG_DATA_FILENAME, \
    N_DATALOADER_WORKERS, CACHE_LIMIT

import os
import pdb
import copy
import random
import pickle

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, sampler


class PytorchImagesDataset(Dataset):
    """
    A class for loading in images one at a time.
    Follows pytorch dataset tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self,
                 dataset,
                 transform_statistics,
                 C_cols,
                 y_cols,
                 merge_klg_01,
                 transform,
                 cache=None,
                 truncate_C_floats=True,
                 data_proportion=1.,
                 shuffle_Cs=False,
                 zscore_C=True,
                 zscore_Y=False,
                 max_horizontal_translation=None,
                 max_vertical_translation=None,
                 C_hat_path=None,
                 use_small_subset=False,
                 downsample_fraction=None):
        assert dataset in ['train', 'val', 'test']
        assert all([(y_col in CLINICAL_CONTROL_COLUMNS) for y_col in y_cols])
        assert all([(C_col in CLINICAL_CONTROL_COLUMNS) for C_col in C_cols])
        assert len(set(y_cols).intersection(set(C_cols))) == 0  # A and y should not share variables

        assert transform in ['None', 'random_translation'], 'transform is: %s' % str(transform)
        if transform != 'None': assert max_horizontal_translation > 0
        if transform != 'None': assert max_vertical_translation > 0

        self.cache = cache
        self.dataset = dataset
        self.transform_statistics = transform_statistics
        self.C_cols = C_cols
        self.y_cols = y_cols
        self.merge_klg_01 = merge_klg_01
        self.transform = transform
        self.data_proportion = data_proportion
        self.max_horizontal_translation = max_horizontal_translation
        self.max_vertical_translation = max_vertical_translation
        self.C_hat_path = C_hat_path
        self.use_small_subset = use_small_subset
        self.downsample_fraction = downsample_fraction

        # ----- Data processing -----
        self.base_dir_for_images, self.non_image_data, data_transform_statistics = \
            load_non_image_data(self.dataset, C_cols, y_cols, zscore_C, zscore_Y,
                                transform_statistics=transform_statistics,
                                merge_klg_01=merge_klg_01,
                                truncate_C_floats=truncate_C_floats,
                                shuffle_Cs=shuffle_Cs,
                                check=True)

        if self.transform_statistics is None:
            self.transform_statistics = data_transform_statistics

        # Select subset of ids if we are only using a proportion of data
        N = len(self.non_image_data)
        if self.data_proportion < 1.:
            N_selected = int(data_proportion * N)
            selected_ids = np.random.choice(N, N_selected, replace=False)
            self.selected_ids = selected_ids
        else:
            self.selected_ids = np.arange(N)

        if C_hat_path:
            # Attribute prediction available from previous model
            pass

        if len(C_cols) > 0:
            self.C_feats = copy.deepcopy(self.non_image_data[C_cols].values)

            # This is the weight given to EACH class WITHIN EACH attribute in order to correct the imbalance.
            variables = [C_col + '_loss_class_wt' for C_col in C_cols]
            self.C_feats_loss_class_wts = copy.deepcopy(self.non_image_data[variables].values)
        self.y_feats = copy.deepcopy(self.non_image_data[y_cols].values)

        print('Dataset %s has %i rows' % (dataset, len(self.non_image_data)))

    def __len__(self):
        if self.use_small_subset:
            return 500
        return len(self.selected_ids)

    def __getitem__(self, idx):
        new_idx = self.selected_ids[idx]
        if self.cache:
            image = self.cache.get(new_idx)
            cache_hit = image is not None
        if not self.cache or not cache_hit:
            image_path = os.path.join(self.base_dir_for_images, 'image_%i.npz' % new_idx)
            image = self.load_image(image_path)

        # ----- Data augmentation -----
        if self.transform == 'random_translation':
            image = random_transform_image(image, self.transform,
                                           max_horizontal_translation=self.max_horizontal_translation,
                                           max_vertical_translation=self.max_vertical_translation)

        if self.downsample_fraction:
            image = downsample_image(image, self.downsample_fraction)

        image = np.tile(image, [3, 1, 1])

        # ----- Data processing -----
        if self.C_cols:
            C_feats = self.C_feats[new_idx, :]
            C_feats_loss_class_wts = self.C_feats_loss_class_wts[new_idx]
            C_feats_not_nan = ~np.isnan(C_feats)
            C_feats[~C_feats_not_nan] = 0
            C_feats_not_nan = C_feats_not_nan * 1.
        y_feats = self.y_feats[new_idx]
        assert ~any(np.isnan(y_feats))

        sample = {'image': image,
                  'C_feats': C_feats,  # C_cols that are Z-scored
                  'C_feats_not_nan': C_feats_not_nan,  # C_cols that are Z-scored but not nan
                  'C_feats_loss_class_wts': C_feats_loss_class_wts,
                  'y': y_feats}
        return sample

    def load_image(self, path):
        return np.load(path)['arr_0']

def load_non_image_data(dataset_split, C_cols, y_cols, zscore_C, zscore_Y,
                        transform_statistics=None, merge_klg_01=True, truncate_C_floats=True,
                        shuffle_Cs=False, return_CY_only=False, check=True, verbose=True):
    base_dir_for_images = get_base_dir_for_individual_image(dataset_split)
    image_codes = pickle.load(open(os.path.join(base_dir_for_images, IMG_CODES_FILENAME), 'rb'))
    non_image_data = pd.read_csv(os.path.join(base_dir_for_images, NON_IMG_DATA_FILENAME), index_col=0)
    if check: ensure_barcodes_match(non_image_data, image_codes)

    # Clip xrattl from [0,3] to [0,2]. Basically only for the 2 examples with Class = 3
    # which do not appear in train dataset
    if verbose: print('Truncating xrattl')
    non_image_data['xrattl'] = np.minimum(2, non_image_data['xrattl'])

    # Data processing for non-image data
    if merge_klg_01:
        if verbose: print('Merging KLG')
        # Merge KLG 0,1 + Convert KLG scale to [0,3]
        non_image_data['xrkl'] = np.maximum(0, non_image_data['xrkl'] - 1)

    # Truncate odd decimals
    if truncate_C_floats:
        if verbose: print('Truncating A floats')
        for variable in C_cols + y_cols:
            # Truncate decimals
            non_image_data[variable] = non_image_data[variable].values.astype(np.int64).astype(np.float64)

    # Mix up the As for the training set to see if performance of KLG worsens
    if shuffle_Cs:
        if verbose: print('Shuffling As')
        for variable in C_cols:
            N = len(non_image_data)
            permutation = np.random.permutation(N)
            non_image_data[variable] = non_image_data[variable].values[permutation]

    # Give weights for each class within each attribute, so that it can be used to reweigh the loss
    for variable in C_cols:
        new_variable = variable + '_loss_class_wt'
        attribute = non_image_data[variable].values
        unique_classes = np.unique(attribute)
        N_total = len(attribute)
        N_classes = len(unique_classes)
        weights = np.zeros(len(attribute))
        for cls_val in unique_classes:
            belongs_to_cls = attribute == cls_val
            counts = np.sum(belongs_to_cls)
            # Since each class has 'counts', the total weight allocated to each class = 1
            # weights[belongs_to_cls] = 1. / counts
            weights[belongs_to_cls] = (N_total - counts) / N_total
        non_image_data[new_variable] = weights

    # Z-scoring of the Ys
    new_transform_statistics = {}
    y_feats = None
    if zscore_Y:
        y_feats = copy.deepcopy(non_image_data[y_cols].values)
        for i in range(len(y_cols)):
            not_nan = ~np.isnan(y_feats[:, i])
            if transform_statistics is None:
                std = np.std(y_feats[not_nan, i], ddof=1)
                mu = np.mean(y_feats[not_nan, i])
                new_transform_statistics[y_cols[i]] = { 'mu': mu, 'std': std }
            else:
                std = transform_statistics[y_cols[i]]['std']
                mu = transform_statistics[y_cols[i]]['mu']
            if verbose: print('Z-scoring additional feature %s with mean %2.3f and std %2.3f' % (y_cols[i], mu, std))
            non_image_data['%s_original' % y_cols[i]] = y_feats[:, i]
            non_image_data[y_cols[i]] = (y_feats[:, i] - mu) / std
            y_feats[:, i] = non_image_data[y_cols[i]]

    # Z-scoring of the attributes
    C_feats = None
    if zscore_C:
        C_feats = copy.deepcopy(non_image_data[C_cols].values)
        for i in range(len(C_cols)):
            not_nan = ~np.isnan(C_feats[:, i])
            if transform_statistics is None:
                std = np.std(C_feats[not_nan, i], ddof=1)
                mu = np.mean(C_feats[not_nan, i])
                new_transform_statistics[C_cols[i]] = {'mu': mu, 'std': std}
            else:
                std = transform_statistics[C_cols[i]]['std']
                mu = transform_statistics[C_cols[i]]['mu']
            if verbose: print('Z-scoring additional feature %s with mean %2.3f and std %2.3f' % (C_cols[i], mu, std))
            non_image_data['%s_original' % C_cols[i]] = C_feats[:, i]
            non_image_data[C_cols[i]] = (C_feats[:, i] - mu) / std
            C_feats[:, i] = non_image_data[C_cols[i]]

    if return_CY_only:
        if y_feats is None:
            y_feats = copy.deepcopy(non_image_data[y_cols].values)
        if C_feats is None:
            C_feats = copy.deepcopy(non_image_data[C_cols].values)
        return C_feats, y_feats

    return base_dir_for_images, non_image_data, new_transform_statistics

def load_attributes(image_codes, non_image_data, all_cols, y_cols, merge_klg_01=False, zscore_C=False, zscore_Y=False):
    C_feats = copy.deepcopy(non_image_data[all_cols].values)
    if zscore_C:
        for i in range(len(all_cols)):
            not_nan = ~np.isnan(C_feats[:, i])
            std = np.std(C_feats[not_nan, i], ddof=1)
            mu = np.mean(C_feats[not_nan, i])
            print('Z-scoring additional feature %s with mean %2.3f and std %2.3f' % (all_cols[i], mu, std))
            C_feats[:, i] = (C_feats[:, i] - mu) / std

    y_feats = copy.deepcopy(non_image_data[y_cols].values)
    if merge_klg_01:
        assert 'xrkl' in y_cols
        y_feats = np.maximum(0, y_feats - 1)
    if zscore_Y:
        for i in range(len(y_cols)):
            not_nan = ~np.isnan(y_feats[:, i])
            std = np.std(y_feats[not_nan, i], ddof=1)
            mu = np.mean(y_feats[not_nan, i])
            if verbose: print('Z-scoring additional feature %s with mean %2.3f and std %2.3f' % (y_cols[i], mu, std))
            y_feats[:, i] = (y_feats[:, i] - mu) / std

    print('A shape: %s' % str(C_feats.shape))
    print('y shape: %s' % str(y_feats.shape))
    return C_feats, y_feats

def get_sampling_weights(sampling_strategy, sampling_args, train_dataset, C_cols, y_cols):
    """
    Get different weights for each data point according to the sampling strategy.
    """
    print('\n-------------- Sampling strategy: %s --------------' % sampling_strategy)
    if sampling_strategy == 'weighted':

        C_data = train_dataset.non_image_data[C_cols].values
        y_data = train_dataset.non_image_data[y_cols].values
        N_C_cols = len(C_cols)
        N_y_cols = len(y_cols)
        N = C_data.shape[0]
        weights = np.zeros(N)
        if sampling_args['mode'] in ['weigh_C', 'weigh_Cy']:
            # Select an example according to the class distribution of As.
            # For an attribute, if an example is part of the rarer class, it will receive a higher weight.
            # However, if the class is too rare, we assume give it a uniform weight to prevent overfitting.
            for i in range(N_C_cols):
                string = '%02d [%7s] ' % (i + 1, C_cols[i])
                attribute = C_data[:, i]
                ids_all = np.arange(N)
                classes = np.unique(attribute)
                N_classes = len(classes)
                total_wt_per_class = 1. / N_classes

                for cls_val in classes:
                    counts = np.sum(attribute == cls_val)
                    if counts < sampling_args['min_count']:
                        # If there are too few of this attribute class, we just sample this point uniformly
                        # (to prevent constant sampling of this data point and overfitting).
                        # Weight is (N_C_cols / N) instead of (1 / N) because we are adding magnitudes of 1
                        # to the 'weights' vector N_C_cols times.Dis
                        print('Setting Attribute %s (Class %f) to uniform due to small size of %d' %
                              (C_cols[i], cls_val, counts))
                        additive_weights = N_C_cols / N
                    else:
                        # Each data point given weight w such that each class should be equally sampled.
                        # EG. If there are 25 pos and 100 neg, we want sampling of 0.5:0.5.
                        # So, we give 0.5/25 = 0.02 weight to pos and 0.5/100=0.005 weight to neg.
                        additive_weights = total_wt_per_class / (counts + 0)
                    weights[attribute == cls_val] += additive_weights
                    string += '(%.7f)' % additive_weights
                print(string)

            if sampling_args['mode'] == 'weigh_Cy':
                for i in range(N_y_cols):
                    string = '%02d [%7s] ' % (i + 1, y_cols[i])
                    y_value = y_data[:, i]
                    ids_all = np.arange(N)
                    classes = np.unique(y_value)
                    N_classes = len(classes)
                    total_wt_per_class = 1. / N_classes

                    for cls_val in classes:
                        counts = np.sum(y_value == cls_val)
                        if counts < sampling_args['min_count']:
                            print('Setting Attribute %s (Class %f) to uniform due to small size of %d' %
                                  (y_cols[i], cls_val, counts))
                            additive_weights = N_C_cols / N
                        else:
                            additive_weights = total_wt_per_class / (counts + 0)
                        weights[y_value == cls_val] += additive_weights
                        string += '(%.7f)' % additive_weights

            hist_counts, hist_weights = np.histogram(weights * 1000, bins=20)
            print('Histogram counts               : ', [x for x in hist_counts])
            print('Histogram weights              : ', ' '.join(['%.1f' % x for x in hist_weights]))
            # While each rare class has high weight, its 'mass' is small since it has low count
            # The below unif_prob_mass_per_bin and prob_mass_per_bin show what the rare class representation would be
            # like before and after resampling. Ideally, the change should not be overly drastic, but the common class
            # representation should smooth out.
            unif_prob_mass = (1 / N) * hist_counts
            unif_prob_mass_per_bin = unif_prob_mass / np.sum(unif_prob_mass)
            print('Uniform prob mass (%)          :', ' '.join(['%.1f' % (x * 100) for x in unif_prob_mass_per_bin]))
            prob_mass = hist_weights[:-1] * hist_counts
            prob_mass_per_bin = prob_mass / np.sum(prob_mass)
            print('Histogram weights prob mass (%): ', ' '.join(['%.1f' % (x * 100) for x in prob_mass_per_bin]), '\n')

        train_sampler = sampler.WeightedRandomSampler(weights, len(weights))
        shuffle = False

    elif sampling_strategy == 'uniform':
        train_sampler = None
        shuffle = True

    return train_sampler, shuffle

def get_image_cache_for_split(dataset_split, limit=None):
    print('Building image cache for %s split' % dataset_split)
    cache = {}
    base_dir_for_images = get_base_dir_for_individual_image(dataset_split)
    non_image_data = pd.read_csv(os.path.join(base_dir_for_images, NON_IMG_DATA_FILENAME), index_col=0)
    N = len(non_image_data) if limit is None else min(int(limit), len(non_image_data))

    num_workers = 8
    def get_images(ids_group, result):
        for idx in ids_group:
            image_path = os.path.join(base_dir_for_images, 'image_%i.npz' % idx)
            image = np.load(image_path)['arr_0']
            result[idx] = image

    rounds = 20 # Split into multiple rounds to pass smaller sized data
    import threading

    ids_groups_list = np.array_split(range(N), num_workers * rounds)
    for round in range(rounds):
        print('  Iter %d/%d' % (round + 1, rounds))
        ids_groups = ids_groups_list[round * num_workers:(round + 1) * num_workers]

        results = []
        threads = []
        for i, ids_group in enumerate(ids_groups):
            result = {}
            t = threading.Thread(target=get_images, args=(ids_group, result))
            t.start()
            threads.append(t)
            results.append(result)

        for i, t in enumerate(threads):
            t.join()

        for i, result in enumerate(results):
            cache.update(result)

    return cache

def load_data_from_different_splits(batch_size,
                                    C_cols,
                                    y_cols,
                                    zscore_C,
                                    zscore_Y,
                                    data_proportion,
                                    shuffle_Cs,
                                    merge_klg_01,
                                    max_horizontal_translation,
                                    max_vertical_translation,
                                    augment=None,
                                    sampling_strategy=None,
                                    sampling_args=None,
                                    C_hat_path=None,
                                    use_small_subset=False,
                                    downsample_fraction=None):
    """
    Load dataset a couple images at a time using DataLoader class, as shown in pytorch dataset tutorial.
    Checked.
    """
    limit = 500 if use_small_subset else CACHE_LIMIT
    cache_train = get_image_cache_for_split('train', limit=limit)
    train_dataset = PytorchImagesDataset(dataset='train',
                                         transform_statistics=None,
                                         C_cols=C_cols,
                                         y_cols=y_cols,
                                         zscore_C=zscore_C,
                                         zscore_Y=zscore_Y,
                                         cache=cache_train,
                                         truncate_C_floats=True,
                                         data_proportion=data_proportion,
                                         shuffle_Cs=shuffle_Cs,
                                         merge_klg_01=merge_klg_01,
                                         transform=augment,
                                         max_horizontal_translation=max_horizontal_translation,
                                         max_vertical_translation=max_vertical_translation,
                                         use_small_subset=use_small_subset)
    # Sampler for training
    train_sampler, shuffle = get_sampling_weights(sampling_strategy, sampling_args, train_dataset, C_cols, y_cols)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=N_DATALOADER_WORKERS,
                                  sampler=train_sampler, pin_memory=False)

    cache_val = get_image_cache_for_split('val', limit=limit)
    val_dataset = PytorchImagesDataset(dataset='val',
                                       transform_statistics=train_dataset.transform_statistics,
                                       C_cols=C_cols,
                                       y_cols=y_cols,
                                       zscore_C=zscore_C,
                                       zscore_Y=zscore_Y,
                                       cache=cache_val,
                                       truncate_C_floats=True,
                                       data_proportion=data_proportion,
                                       shuffle_Cs=False,
                                       merge_klg_01=merge_klg_01,
                                       transform='None',
                                       use_small_subset=use_small_subset,
                                       downsample_fraction=downsample_fraction)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=N_DATALOADER_WORKERS)

    test_dataset = PytorchImagesDataset(dataset='test',
                                        transform_statistics=train_dataset.transform_statistics,
                                        C_cols=C_cols,
                                        y_cols=y_cols,
                                        zscore_C=zscore_C,
                                        zscore_Y=zscore_Y,
                                        cache=None,
                                        truncate_C_floats=True,
                                        shuffle_Cs=False,
                                        merge_klg_01=merge_klg_01,
                                        transform='None',
                                        use_small_subset=use_small_subset,
                                        downsample_fraction=downsample_fraction)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=N_DATALOADER_WORKERS)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    if use_small_subset:
        dataset_sizes = {'train': 500, 'val': 500, 'test': 500}
    else:
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    return dataloaders, datasets, dataset_sizes

def get_base_dir_for_individual_image(dataset_split):
    """
    Get the path for an image.
    """
    assert dataset_split in ['train', 'val', 'test']
    base_dir = IMG_DIR_PATH % dataset_split
    return base_dir

def ensure_barcodes_match(combined_df, image_codes):
    """
    Sanity check: make sure non-image data matches image data.
    """
    print("Ensuring that barcodes line up.")
    assert len(combined_df) == len(image_codes)
    for idx in range(len(combined_df)):
        barcode = str(combined_df.iloc[idx]['barcdbu'])
        if len(barcode) == 11:
            barcode = '0' + barcode
        side = str(combined_df.iloc[idx]['side'])
        code_in_df = barcode + '*' + side

        if image_codes[idx] != code_in_df:
            raise Exception("Barcode mismatch at index %i, %s != %s" % (idx, image_codes[idx], code_in_df))
    print("All %i barcodes line up." % len(combined_df))

# ----------------- Data augmentations -----------------
def random_horizontal_vertical_translation(img, max_horizontal_translation, max_vertical_translation):
    """
    Translates the image horizontally/vertically by a fraction of its width/length.
    To keep the image the same size + scale, we add a background color to fill in any space created.
    """
    assert max_horizontal_translation >= 0 and max_horizontal_translation <= 1
    assert max_vertical_translation >= 0 and max_vertical_translation <= 1
    if max_horizontal_translation == 0 and max_vertical_translation == 0:
        return img

    img = img.copy()

    assert len(img.shape) == 3
    channels = img.shape[0]
    assert img.shape[1] >= img.shape[2]

    height = img.shape[1]
    width = img.shape[2]

    translated_img = img
    horizontal_translation = int((random.random() - .5) * max_horizontal_translation * width)
    vertical_translation = int((random.random() - .5) * max_vertical_translation * height)
    background_color = img[:, -10:, -10:].mean(axis=1).mean(axis=1)

    # first we translate the image.
    if horizontal_translation != 0:
        if horizontal_translation > 0:
            translated_img = translated_img[:, :, horizontal_translation:]  # cuts off pixels on the left of image.
        else:
            translated_img = translated_img[:, :, :horizontal_translation]  # cuts off pixels on the right of image.

    if vertical_translation != 0:
        if vertical_translation > 0:
            translated_img = translated_img[:, vertical_translation:, :]  # cuts off pixels on the top of image.
        else:
            translated_img = translated_img[:, :vertical_translation, :]  # cuts off pixels on the bottom of image.

    # then we keep the dimensions the same.
    new_height = translated_img.shape[1]
    new_width = translated_img.shape[2]
    new_image = []
    for i in range(channels):  # loop over RGB
        background_square = np.ones([height, width]) * background_color[i]
        if horizontal_translation < 0:
            if vertical_translation < 0:
                # I don't really know if the signs here matter all that much -- it's just whether we're putting the
                # translated images on the left or right.
                background_square[-new_height:, -new_width:] = translated_img[i, :, :]
            else:
                background_square[:new_height, -new_width:] = translated_img[i, :, :]
        else:
            if vertical_translation < 0:
                background_square[-new_height:, :new_width] = translated_img[i, :, :]
            else:
                background_square[:new_height, :new_width] = translated_img[i, :, :]
        new_image.append(background_square)
    new_image = np.array(new_image)

    return new_image

def random_transform_image(image, transform, max_horizontal_translation, max_vertical_translation):
    assert transform in ['random_translation_and_then_random_horizontal_flip', 'random_translation']
    image = random_horizontal_vertical_translation(image, max_horizontal_translation, max_vertical_translation)
    if transform == 'random_translation_and_then_random_horizontal_flip':
        if random.random() < 0.5:
            image = image[:, :, ::-1].copy()
    return image

def downsample_image(image, downsample_fraction):
    assert 0 < downsample_fraction < 1  # this argument is the downsample fraction
    new_image = []
    for i in range(3):  # RGB
        img = image[i, :, :].copy()
        original_size = img.shape  # note have to reverse arguments for cv2.
        img2 = cv2.resize(img,
                          (int(original_size[1] * downsample_fraction), int(original_size[0] * downsample_fraction)))
        new_image.append(cv2.resize(img2, tuple(original_size[::-1])))
        # image[0:1, :, :] = gaussian_filter(image[0:1, :, :], sigma=self.gaussian_blur_filter)
    new_image = np.array(new_image)
    assert new_image.shape == image.shape
    return new_image


