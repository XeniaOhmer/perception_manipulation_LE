import numpy as np
import math
import pickle
import os
import inspect
import pathlib
import h5py

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1 import ConfigProto, Session
from pathlib import Path

from PIL import Image
import random

import pdb
import argparse

def create_exp_identifier(model_params):
    
    conv_depths = model_params['conv_depths']
    fc_depths = model_params['fc_depths']

    n_conv = len(conv_depths)
    n_fc = len(fc_depths)
    conv_nodes = 0
    fc_nodes = 0
    for node_count in conv_depths:
        conv_nodes += node_count
    for node_count in fc_depths:
        fc_nodes += node_count

    exp_identifier = 'conv_{:02d}_{:03d}_fc_{:02d}_{:04d}'.format(n_conv, conv_nodes,
                                                                  n_fc, fc_nodes)
    return exp_identifier


def load_default_model_params(dataset='shapes'):
    model_params = dict()
    if dataset == 'shapes':
        model_params['conv_depths'] = [32,32]
        model_params['fc_depths'] = [16,16]
        model_params['conv_pool'] = [True, False]
    else:
        raise ValueError('Default model for dataset \'{}\' not defined.'.format(dataset))

    return model_params

def sum_label_components(labels, rel_comps=None, factor=0.1, verbose=False, relational_split=None):
    if rel_comps is None:
        raise ValueError('must set kwarg rel_comps to use this function.')
    if verbose:
        orig_labels = labels.copy()
    if relational_split is not None:
        rel_comps = calculate_multitrait_labels(rel_comps, relational_split)

    labels = labels.astype('float')
    for (idx, l) in enumerate(labels):
        true_class = int(np.nonzero(l)[0])
        labels[idx] = (1-factor)*l + factor*rel_comps[true_class]

    if verbose is True:
        print("[INFO] smoothing amount: {}".format(factor))
        print("[INFO] before smoothing: {}".format(orig_labels[0]))
        print("[INFO] after smoothing: {}".format(labels[0]))
        print("[INFO] compiling model...")

    return labels

def coarse_generic_relational_labels(label_map, trait_name=None):
    
    num_classes = np.max([int(k) for k in label_map.keys()]) + 1
    output_labels = [[] for _ in range(num_classes)]
    output_dicts = [{} for _ in range(len(label_map[0]))] 
    for i in range(num_classes):
        traits_i = label_map[i]
        tmp_labels = [np.zeros(num_classes) for _ in range(len(traits_i))]
        for j in range(num_classes):
            if i == j:
                continue
            traits_j = label_map[j]
            for ii in range(len(traits_j)):
                if traits_i[ii] == traits_j[ii]:
                    tmp_labels[ii][j] = 1

        for j in range(len(tmp_labels)):
            tmp_labels[j] /= np.sum(tmp_labels[j])
            output_dicts[j][i] = tmp_labels[j]

    output_dicts = tuple(output_dicts)

    return output_dicts

def load_data(input_shape, normalize=True, 
              subtract_mean=True, 
              balance_traits=True,
              balance_type=2,
              analysis_run=False,
              output_samples=False,
              sample_dir='samples',
              data_path=None):

    meta = None
    if data_path is None:
        data_path = '../data/3dshapes.h5'
    
    dataset = h5py.File(data_path, 'r')
    data = dataset['images'][:]
    full_labels = dataset['labels'][:]
    #visualize_3dshapes_data(data, full_labels)
    labels_reg, labels_relational, keeper_idxs = get_shape_color_labels(full_labels,
                                                    balance_traits=balance_traits,
                                                    balance_type=balance_type)
    if keeper_idxs is not None:
        data = np.array([data[idx] for idx in keeper_idxs])

    meta = labels_relational
    (train_data, test_data, train_labels, test_labels) = train_test_split(data, labels_reg,
                                                                          test_size=0.25,
                                                                          random_state=42)
    if K.image_data_format() == "channels_first":
        train_data = train_data.reshape((train_data.shape[0], 
                                         input_shape[2], input_shape[1], input_shape[0]))
        test_data = test_data.reshape((test_data.shape[0], 
                                       input_shape[2], input_shape[1], input_shape[0]))
    else:
        train_data = train_data.reshape((train_data.shape[0], 
                                         input_shape[0], input_shape[1], input_shape[2]))
        test_data = test_data.reshape((test_data.shape[0], 
                                       input_shape[0], input_shape[1], input_shape[2]))
    if output_samples:
        num_classes = len(train_labels[0]) + 1
        outlabel_digits = math.ceil(math.log10(num_classes))
        idx_list = [i for i in range(num_classes)]
        count = 0
        for (i, label) in enumerate(train_labels):
            if len(train_labels.shape) > 1 and train_labels.shape[1] > 1:
                class_idx = np.argmax(label)
            elif not isinstance(label, int):
                class_idx = int(label[0])
            else:
                class_idx = label
            if not class_idx in idx_list:
                continue
            samp_img = Image.fromarray(train_data[i])
            samp_img = samp_img.resize((128,128))
            tmp_outpath = os.path.join(sample_dir, '{:02d}.jpeg'.format(class_idx))
            samp_img.save(tmp_outpath)
            idx_list.remove(class_idx)

    if normalize:
        train_data = train_data.astype("float32") / 255.0
        test_data = test_data.astype("float32") / 255.0
    if subtract_mean:
        axes_to_average = tuple(range(len(train_data.shape)-1))
        if K.image_data_format() == "channels_first":
            tmp_data = train_data.reshape(train_data.shape[1], -1)
        else:
            tmp_data = train_data.reshape(train_data.shape[-1], -1)

        mean = np.mean(tmp_data, axis=1)
        # sanity check because np.mean over multiple axes seems to behave strangely some times, 
        # may be worth diving into
        if abs( np.mean(mean) - np.mean(train_data) ) > 1e-3:
            raise ValueError("results of mean calculation suspicious, please double check before continuing")
        # print('channel means = ' + str(mean) + ', data mean = ' + str(np.mean(train_data)))
        train_data = train_data - mean
        test_data = test_data - mean

    if analysis_run:
        X = test_data
        Y = test_labels
        X_sorted = sort_data(X, np.argmax(Y, axis=1))
        return X, Y, X_sorted
    else:
        if len(train_labels.shape) == 1 or train_labels.shape[1] == 1:
            le = LabelBinarizer()
            train_labels = le.fit_transform(train_labels)
            test_labels = le.transform(test_labels)
            target_names = [str(x) for x in le.classes_] 
        else:
            target_names = [str(x) for x in range(test_labels.shape[1])]

        validation_data = (test_data, test_labels)
        train_data = (train_data, train_labels)
    
        if meta is not None:
            return train_data, validation_data, target_names, meta
        else:
            return train_data, validation_data, target_names

def sort_data(data, labels):
    data_sorted = dict()
    for i in range(0,np.max(labels)+1):
        data_sorted[str(i)] = []
    for (idx,entry) in enumerate(data):
        try:
            data_sorted[str(labels[idx])].append(list(entry))
        except:
            pdb.set_trace()
    
    for k in data_sorted.keys():
        data_sorted[k] = np.array(data_sorted[k])
    
    return data_sorted

def get_shape_color_labels(full_labels, trait_idxs=(2,3,4), balance_traits=True, balance_type=2):
    possible_values = [[] for _ in range(len(trait_idxs))]
    trait_names_by_idx = ['floorHue', 'wallHue', 'objectHue', 'scale', 'shape', 'orientation'] 
    extracted_traits = [tuple(entry) for entry in list(full_labels[:,trait_idxs])]
    
    for tup in extracted_traits:
        for (idx, entry) in enumerate(tup):
            possible_values[idx].append(entry)
    for (idx,p) in enumerate(possible_values):
        possible_values[idx] = sorted(set(p))
    if balance_traits:
        # 4 (sort of) equally spaced options without generalized implementation for now because I 
        # procrastinate like a douche and force myself into hacky solutions
        if balance_type == 0:
            idxes_to_keep = [ [0,3,6,9], [0,3,5,7] ]
        elif balance_type == 1:
            idxes_to_keep = [ [1,2,4,8], [0,3,5,7] ]
        elif balance_type == 2:
            idxes_to_keep = [ [0,2,4,8], [0,3,5,7] ]
        values_to_keep = [[], []]
        for idx in [0,1]:
            for val_idx in idxes_to_keep[idx]:
                values_to_keep[idx].append(possible_values[idx][val_idx])
        filtered_traits = []
        keeper_idxs = []
        for (idx, traits) in enumerate(extracted_traits):
            if traits[0] in values_to_keep[0] and traits[1] in values_to_keep[1]:
                filtered_traits.append(traits)
                keeper_idxs.append(idx)
        extracted_traits = filtered_traits
    else:
        keeper_idxs = None

    trait_names = [trait_names_by_idx[i] for i in trait_idxs]
    unique_traits = sorted(set(extracted_traits))
    labels = np.zeros((len(extracted_traits),len(unique_traits)))
    label2trait_map = dict()
    trait2label_map = dict()

    for (i, traits) in enumerate(unique_traits):
        trait2label_map[traits] = i
        label2trait_map[i] = traits

    coarse_labels = coarse_generic_relational_labels(label2trait_map)
    relational_labels = dict()
    for (i,k) in enumerate(trait_names):
        relational_labels[k] = coarse_labels[i]
    relational_labels['all'] = dict()
    for k in coarse_labels[0].keys():
        relational_labels['all'][k] = 0
        for lab in coarse_labels:
            relational_labels['all'][k] += 1/len(coarse_labels)*lab[k]

    for (i, traits) in enumerate(extracted_traits):
        labels[i, trait2label_map[traits]] = 1
    
    return labels, relational_labels, keeper_idxs

def configure_gpu_options():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    set_session(sess)

def get_command_line_args(run_type='analysis'):
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoints", type=str, default="checkpoints")
    ap.add_argument("-g", "--gpu", type=int, default=0)
    ap.add_argument("-i", "--dataset", type=str, default='3D-shapes')
    
    ap.add_argument("-s", "--smoothing", type=float, default=None,
            help="amount of label smoothing to be applied")
    ap.add_argument("-p", "--params", type=str, default=None)

    return vars(ap.parse_args())
