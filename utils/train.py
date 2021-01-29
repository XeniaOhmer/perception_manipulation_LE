import numpy as np
import math
import pickle
import os
import inspect
import pathlib
import h5py

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1 import ConfigProto, Session
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import random

import pdb
import argparse

HP_PARAM_DICT = {'n_epochs'      : [500],
                 'vocab_size'    : [3,6,9],
                 'message_length': [3,6,9],
                 'entropy_coeff' : [0., .01],
                 'length_cost'   : [0., .1],
                 'embed_dim'     : [64, 128],
                 'hidden_dim'    : [64, 128],
                 'learning_rate' : [1e-4],
                 'batch_size'    : [64, 128],
                 'activation'    : ['linear', 'tanh']}


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

def get_random_hyperparams(param_options=HP_PARAM_DICT):
    param_keys = sorted(param_options.keys())
    output_params = dict() 
    for k in param_keys:
        output_params[k] = random.choice(param_options[k])

    return output_params


def write_param_files(params=None, 
                       base_fn_name='model_params',
                       outdir='param_files/'):
    if params is None:
        params = load_default_boundary_params()
    # filter ranges should be a 2-d tuple with first entry representing minimum
    # filter size and second entry representing max filter number
    cnn_chan_range = params['nfilter'][0]
    fc_chan_range = params['nfilter'][1]
    cnn_max = cnn_chan_range[1]
    fc_max = fc_chan_range[1]

    model_params = dict()
    model_params['conv_depths'] = [cnn_chan_range[0]] 
    model_params['fc_depths'] = [fc_chan_range[0]] 

    file_count = 0
    for i in range(params['layers'][0]):
        for j in range(params['layers'][1]):
            if np.mod(j,2) == 0:
                continue
            print('j = ' + str(j))
            model_params['conv_depths'] = [cnn_chan_range[0]]*(i+1)
            model_params['conv_depths'] = [min(x, cnn_max) for x in model_params['conv_depths']]

            if i < 3:
                model_params['conv_pool'] = [True] + [False]*(len(model_params['conv_depths'])-1)
            else:
                model_params['conv_pool'] = [True, False, True, False] + \
                                                [False]*(len(model_params['conv_depths'])-4)

            model_params['fc_depths'] = [fc_chan_range[0]]*(j+1)
            out_path = outdir + base_fn_name + "_{:04d}".format(file_count) + '.pkl.'
#           with open(out_path, 'wb') as f:
#               pickle.dump(model_params, f)
            while True:
                out_path = outdir + base_fn_name + "_{:04d}".format(file_count) + '.pkl'

                n_conv = len(model_params['conv_depths'])
                n_convnodes = sum(model_params['conv_depths'])
                n_fc = len(model_params['fc_depths'])
                n_fcnodes = sum(model_params['fc_depths'])
                with open(out_path, 'wb') as f:
                    pickle.dump(model_params, f)
                    file_count += 1
                print('file count = ' + str(file_count))
                print('conv_depths : ' + str(model_params['conv_depths']))
                print('conv_pool : ' + str(model_params['conv_pool']))
                print('fc_depths : ' + str(model_params['fc_depths']))
                print()
                if (np.min(model_params['fc_depths']) == fc_chan_range[1] and
                        np.min(model_params['conv_depths']) == cnn_chan_range[1]):
                    break
                                                                                      
                model_params['conv_depths'] = [min(x*2, cnn_max) for x in model_params['conv_depths']]
                model_params['fc_depths'] = [min(x*2, fc_max) for x in model_params['fc_depths']]
                out_path = outdir + base_fn_name + "_{:04d}".format(file_count) + '.pkl.'

def load_default_boundary_params():
    params = dict()
    params['nfilter'] = ((2, 4), (4,16))
    params['layers'] = (4,8)

    return params


def load_default_model_params(dataset='shapes'):
    model_params = dict()
    if dataset == 'shapes':
        model_params['conv_depths'] = [32,32]
        model_params['fc_depths'] = [16,16]
        model_params['conv_pool'] = [True, False]
    else:
        raise ValueError('Default model for dataset \'{}\' not defined.'.format(dataset))

    return model_params

# expects labels as integers
def relational_labels(labels, factor=0.1, verbose=False, dataset='3D-shapes', rel_type='both'):
    if verbose:
        orig_labels = labels.copy()
    label_dict = dict()

    labels = labels.astype('float')
    
    if dataset == 'mnist':
        for i in range(0,10):
            tmp_vec = np.zeros(10)
            for idx,val in enumerate(tmp_vec):
                if i == idx:
                    tmp_vec[idx] = 0
                else:
                    tmp_vec[idx] = 1/(abs(i - idx))
            tmp_vec = (tmp_vec / np.sum(tmp_vec))*factor
            tmp_vec[i] = (1-factor)
            label_dict[str(i)] = tmp_vec
        for idx,val in enumerate(labels):
            int_val = [i for (i,x) in enumerate(val) if x][0] 
            labels[idx] = label_dict[str(int_val)]
    elif dataset == '2D-shapes':
        rel_component = [[] for _ in range(9)]
        if rel_type == 'shape':
            rel_component[0] = factor*np.array([0, 0.5, 0.5, 0, 0, 0, 0, 0, 0])
            rel_component[1] = factor*np.array([0.5, 0, 0.5, 0, 0, 0, 0, 0, 0])
            rel_component[2] = factor*np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0, 0])
            rel_component[3] = factor*np.array([0, 0, 0, 0, 0.5, 0.5, 0, 0, 0])
            rel_component[4] = factor*np.array([0, 0, 0, 0.5, 0, 0.5, 0, 0, 0])
            rel_component[5] = factor*np.array([0, 0, 0, 0.5, 0.5, 0, 0, 0, 0])
            rel_component[6] = factor*np.array([0, 0, 0, 0, 0, 0, 0, 0.5, 0.5])
            rel_component[7] = factor*np.array([0, 0, 0, 0, 0, 0, 0.5, 0, 0.5])
            rel_component[8] = factor*np.array([0, 0, 0, 0, 0, 0, 0.5, 0.5, 0])
        elif rel_type == 'color':
            rel_component[0] = factor*np.array([0, 0, 0, 0.5, 0, 0, 0.5, 0, 0])
            rel_component[1] = factor*np.array([0, 0, 0, 0, 0.5, 0, 0, 0.5, 0])
            rel_component[2] = factor*np.array([0, 0, 0, 0, 0, 0.5, 0, 0, 0.5])
            rel_component[3] = factor*np.array([0.5, 0, 0, 0, 0, 0, 0.5, 0, 0])
            rel_component[4] = factor*np.array([0, 0.5, 0, 0, 0, 0, 0, 0.5, 0])
            rel_component[5] = factor*np.array([0, 0, 0.5, 0, 0, 0, 0, 0, 0.5])
            rel_component[6] = factor*np.array([0.5, 0, 0, 0.5, 0, 0, 0, 0, 0])
            rel_component[7] = factor*np.array([0, 0.5, 0, 0, 0.5, 0, 0, 0, 0])
            rel_component[8] = factor*np.array([0, 0, 0.5, 0, 0, 0.5, 0, 0, 0])
        else:
            rel_component[0] = factor*np.array([0, 0.25, 0.25, 0.25, 0, 0, 0.25, 0, 0])
            rel_component[1] = factor*np.array([0.25, 0, 0.25, 0, 0.25, 0, 0, 0.25, 0])
            rel_component[2] = factor*np.array([0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0.25])
            rel_component[3] = factor*np.array([0.25, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0])
            rel_component[4] = factor*np.array([0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0])
            rel_component[5] = factor*np.array([0, 0, 0.25, 0.25, 0.25, 0, 0, 0, 0.25])
            rel_component[6] = factor*np.array([0.25, 0, 0, 0.25, 0, 0, 0, 0.25, 0.25])
            rel_component[7] = factor*np.array([0, 0.25, 0, 0, 0.25, 0, 0.25, 0, 0.25])
            rel_component[8] = factor*np.array([0, 0, 0.25, 0, 0, 0.25, 0, 0.25, 0.25])
        for (idx, l) in enumerate(labels):
            true_class = int(np.nonzero(l)[0])
            labels[idx] = (1-factor)*l + rel_component[true_class]
    if verbose is True:
        print("[INFO] smoothing amount: {}".format(factor))
        print("[INFO] before smoothing: {}".format(orig_labels[0]))
        print("[INFO] after smoothing: {}".format(labels[0]))
        print("[INFO] compiling model...")

    return labels

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

def calculate_multitrait_labels(components, coeffs):
    assert components.keys() == coeffs.keys(), "keys of components dict must match keys of coeffs dict"
    coeffs_sum = 0
    tmp_labels = dict()
    for (i,kt) in enumerate(components.keys()):
        assert coeffs[kt] > 0 and coeffs[kt] < 1, "coeffs must be between 0 and 1"
        coeffs_sum += coeffs[kt]
        for (idx, kc) in enumerate(components[kt].keys()):
            if i == 0:
                tmp_labels[kc] = components[kt][kc]*coeffs[kt]
            else:
                try:
                    tmp_labels[kc] += components[kt][kc]*coeffs[kt]
                except:
                    pdb.set_trace()
    output_labels = tmp_labels
    assert coeffs_sum == 1, "coeffs must sum to one"
    return output_labels


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

def get_validation_data(dataset, normalize=True):
    if dataset == 'mnist':
        ((_, _), (X, Y)) = mnist.load_data()
        if K.image_data_format() == "channels_first":
            X = X.reshape((X.shape[0], 1, 28, 28))
        else:
            X = X.reshape((X.shape[0], 28, 28, 1))
        if normalize:
            X = X.astype("float") / 255.0 
            X = X - np.mean(X, axis=0)
            
    return X, Y

def load_data(input_shape, dataset='3D-shapes', normalize=True, 
                                                subtract_mean=True, 
                                                balance_traits=True,
                                                balance_type=2,
                                                analysis_run=False,
                                                output_samples=False,
                                                sample_dir='samples'):
    meta = None
    if dataset == 'mnist':
        ((train_data, train_labels), (test_data, test_labels)) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        ((train_data, train_labels), (test_data, test_labels)) = fashion_mnist.load_data()
    elif dataset == 'cifar10':
        ((train_data, train_labels), (test_data, test_labels)) = cifar10.load_data()
    elif dataset == 'cifar100':
        ((train_data, train_labels), (test_data, test_labels)) = cifar100.load_data()
    elif dataset == '2D-shapes':
        data = np.load('../data/images.npy')
        labels = np.load('../data/labels.npy')
        (train_data, test_data, train_labels, test_labels) = train_test_split(data, labels,
                                                                              test_size=0.25,
                                                                              random_state=42)
    elif dataset == '3D-shapes':
        # data_path = '../data/3dshapes.h5'
        data_path = '/net/store/cogmod/users/xenohmer/PycharmProjects/SimilarityGames/data/3dshapes.h5'
        parent_dir = str(pathlib.Path().absolute()).split('/')[-1]
        if parent_dir == 'SimilarityGames':
            data_path = data_path[3:]
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
    else:
        raise ValueError('Function not defined for dataset \'{}\'.'.format(dataset))

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
        print('channel means = ' + str(mean) + ', data mean = ' + str(np.mean(train_data)))
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
            target_names = []
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

def visualize_3dshapes_data(images, labels):
    fig = plt.figure()
    figs_shown = []
    for (img, label) in zip(images, labels):
        if [label[2], label[4]] in figs_shown:
            continue
        figs_shown.append([label[2], label[4]])
        plt.imshow(img)
        plt.title('label = {:01f},{}'.format(label[2],label[4]))
        plt.show()
    plt.close(fig)



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
    ap.add_argument("-g", "--gpu", type=int, default=1)
    ap.add_argument("-i", "--dataset", type=str, default='3D-shapes')
    
    ap.add_argument("-s", "--smoothing", type=float, default=None,
            help="amount of label smoothing to be applied")
    ap.add_argument("-p", "--params", type=str, default=None)
    ap.add_argument("-b", "--bash", type=int, default=0)

    return vars(ap.parse_args())
