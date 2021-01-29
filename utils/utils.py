import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from pathlib import Path
import pathlib
import h5py
import pdb
import pickle
import shutil
        

def init_ctd_entry(num_classes, layer_keys):

    ctd_dict = dict()
    for kl in layer_keys:
        ctd_dict[kl] = list()

    return ctd_dict


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


def sort_data(data, labels):
    data_sorted = dict()
    for i in range(0, 9):
        data_sorted[str(i)] = []
    for (idx, entry) in enumerate(data):
        try:
            data_sorted[str(labels[idx])].append(list(entry))
        except:
            pdb.set_trace()
    
    for k in data_sorted.keys():
        data_sorted[k] = np.array(data_sorted[k])
    
    return data_sorted


def get_validation_data(dataset, normalize=True, sort_x=True, binarize=True, subtract_mean=True,
                        balance_traits=True, input_shape=None):
    if dataset == 'mnist':
        ((_, _), (X, Y)) = mnist.load_data()
        if K.image_data_format() == "channels_first":
            X = X.reshape((X.shape[0], 1, 28, 28))
        else:
            X = X.reshape((X.shape[0], 28, 28, 1))
        if normalize:
            X = X.astype("float") / 255.0 
            X = X - np.mean(X, axis=0)
    elif dataset == '3D-shapes':
        data_path = '../data/3dshapes.h5'
        parent_dir = str(pathlib.Path().absolute()).split('/')[-1]
        if parent_dir == 'SimilarityGames':
            data_path = data_path[3:]
        dataset = h5py.File(data_path, 'r')
        data = dataset['images'][:]
        full_labels = dataset['labels'][:]
        labels_reg, labels_relational, keeper_idxs = get_shape_color_labels(full_labels, balance_traits=balance_traits)
        if keeper_idxs is not None:
            data = np.array([data[idx] for idx in keeper_idxs])

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
        if normalize:
            train_data = train_data.astype("float32") / 255.0
            test_data = test_data.astype("float32") / 255.0
        if subtract_mean:
            if K.image_data_format() == "channels_first":
                tmp_data = train_data.reshape(train_data.shape[1], -1)
            else:
                tmp_data = train_data.reshape(train_data.shape[-1], -1)

            mean = np.mean(tmp_data, axis=1)
            # sanity check because np.mean over multiple axes seems to behave strangely some times, 
            # may be worth diving into
            if abs(np.mean(mean) - np.mean(train_data)) > 1e-3:
                raise ValueError("results of mean calculation suspicious, please double check before continuing")
            print('channel means = ' + str(mean) + ', data mean = ' + str(np.mean(train_data)))
            test_data = test_data - mean
        X = test_data
        Y = test_labels

    if sort_x:
        X_sorted = sort_data(X, Y)
    if binarize:
        lb = LabelBinarizer()
        Y = lb.fit_transform(Y)

    if sort_x:
        return X, Y, X_sorted
    else:
        return X, Y


def load_data(input_shape, dataset='3D-shapes', normalize=True,
                                                subtract_mean=True,
                                                balance_traits=True,
                                                balance_type=2,
                                                analysis_run=False):

    if dataset == '3D-shapes':
        data_path = '../data/3dshapes.h5'
        parent_dir = str(pathlib.Path().absolute()).split('/')[-1]
        if parent_dir == 'SimilarityGames':
            data_path = data_path[3:]
        dataset = h5py.File(data_path, 'r')
        data = dataset['images'][:]
        full_labels = dataset['labels'][:]
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
    if normalize:
        train_data = train_data.astype("float32") / 255.0
        test_data = test_data.astype("float32") / 255.0
    if subtract_mean:
        if K.image_data_format() == "channels_first":
            tmp_data = train_data.reshape(train_data.shape[1], -1)
        else:
            tmp_data = train_data.reshape(train_data.shape[-1], -1)

        mean = np.mean(tmp_data, axis=1)
        # sanity check because np.mean over multiple axes seems to behave strangely some times, 
        # may be worth diving into
        if abs(np.mean(mean) - np.mean(train_data)) > 1e-3:
            raise ValueError("results of mean calculation suspicious, please double check before continuing")
        train_data = train_data - mean
        test_data = test_data - mean
    if analysis_run:
        X = test_data
        Y = test_labels
        X_sorted = sort_data(X, np.argmax(Y, axis=1))
        return X, Y, X_sorted
    else:
    
        le = LabelBinarizer()
        train_labels = le.fit_transform(train_labels)
        test_labels = le.transform(test_labels)
        target_names = [str(x) for x in le.classes_] 
        validation_data = (test_data, test_labels)
        train_data = (train_data, train_labels)
    
        if meta is not None:
            return train_data, validation_data, target_names, meta
        else:
            return train_data, validation_data, target_names


def get_shape_color_labels(full_labels, trait_idxs=(2, 3, 4), balance_traits=True, balance_type=2):
    possible_values = [[] for _ in range(len(trait_idxs))]
    trait_names_by_idx = ['floorHue', 'wallHue', 'objectHue', 'scale', 'shape', 'orientation'] 
    extracted_traits = [tuple(entry) for entry in list(full_labels[:,trait_idxs])]
    
    for tup in extracted_traits:
        for (idx, entry) in enumerate(tup):
            possible_values[idx].append(entry)
    for (idx, p) in enumerate(possible_values):
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
        for idx in [0, 1]:
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


def coarse_generic_relational_labels(label_map):
    
    num_classes = np.max([int(k) for k in label_map.keys()]) + 1
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


def load_experiment_variables(params_path):
    try:
        with open(params_path, 'rb') as f:
            (model_history, time_str, c_matrix_orig) = pickle.load(f)
            return model_history, time_str, c_matrix_orig
    except:
        print('[info] params file {} does not exist, skipping '.format(params_path))
        return None


def load_completed_analysis_data(BASE_OUT_DIR, trial):
    data_path = os.path.join(BASE_OUT_DIR, trial, 'data_backup.pkl')
    cross_exp_path = os.path.join(BASE_OUT_DIR, 'cross_experiment_backup.pkl')
    try:
        with open(data_path, 'rb') as f:
            (dist_matrices, hist_data, c_matrices, _) = pickle.load(f)
        with open(cross_exp_path, 'rb') as f:
            cross_trial_data = pickle.load(f)
        return dist_matrices, hist_data, c_matrices, cross_trial_data
    except:
        try:
            with open(data_path, 'rb') as f:
                (dist_matrices, hist_data, c_matrices, cross_trial_data) = pickle.load(f)
            return dist_matrices, hist_data, c_matrices, cross_trial_data
        except:
            print('[info] failed to load data, deleting directory {}'.format(os.path.join(BASE_OUT_DIR,trial)))
            shutil.rmtree(os.path.join(BASE_OUT_DIR, trial))
            # should generalize this to flexible list of children directories
            setup_output_directories(os.path.join(BASE_OUT_DIR, trial), 
                                     create_parents=False, 
                                     child_dir='histograms')
            return None


def load_model_from_hdf5(checkpoint_path):
    from tensorflow.keras.models import load_model
    try:
        return load_model(checkpoint_path)
    except:
        pdb.set_trace()
        raise ValueError('model could not be loaded from path {}'.format(checkpoint_path))


def dump_variables(data, out_path, mode='wb'):
    try:
        with open(out_path, mode) as f:
            pickle.dump(data, f)
    except:
        pdb.set_trace()
        print('![!INFO!]! error writing to file {}, continuing analysis'.format(out_path))


def setup_output_directories(path, create_parents=False, child_dir=None):
    Path(path).mkdir(parents=create_parents)
    if child_dir is not None:
        child_path = os.path.join(path, child_dir)
        Path(child_path).mkdir(parents=False)


def search(values, target):
    for k in values:
        for v in values[k]:
            if target in v:
                return k
    return None


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# def backup_dictfile(name):
#     src = 'obj/' + name + '.pkl'
#     dst = 'obj/tmp_check/' + name + '.pkl'
#     copyfile(src, dst)
