# USAGE
# python run_mnist_training.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# import the necessary packages
from models import GenericNet
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow import device
import tensorflow as tf

import utils
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gc
import pdb

from pathlib import Path
from datetime import datetime
import shutil
import time
import sys


# to fix cudnn handle error
utils.train.configure_gpu_options()
args = utils.train.get_command_line_args(run_type='train')

if args['smoothing'] is not None:
    FACTOR = args['smoothing']
    smooth_labels = True
else:
    FACTOR = 0.0
    smooth_labels = False

if args['dataset'] == 'mnist':
    from config import mnist_config as config 
elif args['dataset'] == '3D-shapes':
    from config import shapes_config as config 


input_shape = config.DATASET_PARAMS['input_shape']
num_classes = config.DATASET_PARAMS['num_classes']
epochs, init_lr, batch_size, verbose = config.get_default_training_config()
epochs = 200

# load model parameters
if args['params'] is None:
    model_params = utils.train.load_default_model_params()
else:
    with open(args['params'], 'rb') as f:
        model_params = pickle.load(f)

# set up experiment directory for checkpoints
params_str = utils.train.create_exp_identifier(model_params)
# checkpoints_path = os.path.join(args['checkpoints'], args['dataset'], params_str)
# eliminated dataset ID folder in path just for testing

print("[INFO] loading data...")
train_data, validation_data, target_names, relational_labels = utils.train.load_data(input_shape,
                                                                                     dataset='3D-shapes',
                                                                                     balance_traits=False,
                                                                                     output_samples=True,
                                                                                     sample_dir='data_samples/full/')
# change this to be dynamic with filtered dataset
num_classes = 64

smooth_func = utils.train.sum_label_components
relational_split = {'objectHue':0.5, 'shape':0.5}
#test = utils.train.sum_label_components(train_data[1], relational_labels,
#                                        relational_split=relational_split)
# train the network
for trait_to_enforce in ['objectHue', 'scale', 'shape', 'all']:
# for trait_to_enforce in ['scale', 'shape', 'all']:
    if trait_to_enforce == 'all':
        sf_list = [0.2, 0.5, 0.6, 0.8, 0.0]
        sf_list = [0.5, 0.6, 0.8, 0.0]
    else:
        sf_list = [0.2, 0.5, 0.6]
        sf_list = [0.6, 0.5]
    for FACTOR in sf_list: 
        with device('/gpu:' + str(args["gpu"])): 
            opt = SGD(lr=init_lr)
            model = GenericNet.build(*input_shape, num_classes, **model_params)
            model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
            checkpoints_path = os.path.join(args['checkpoints'], trait_to_enforce, params_str, 'sf_'+ str(FACTOR).replace('.', '-'))
#           checkpoints_path = os.path.join('tmp', trait_to_enforce, params_str, 'sf_'+ str(FACTOR).replace('.', '-'))
            if not os.path.isdir(checkpoints_path):
                Path(checkpoints_path).mkdir(parents=True)

            if args['params'] is not None:
                shutil.copyfile(args['params'], os.path.join(checkpoints_path, 'model_params.pkl'))

            if args['bash']:
                orig_stdout = sys.stdout
                log_file = os.path.join(checkpoints_path, "log_sf-{:.2f}.txt".format(FACTOR))
                sys.stdout = open(log_file, 'w')
            fname = os.path.sep.join([checkpoints_path, 
                                      "weights-sfactor-{:.2f}".format(FACTOR) + "-{epoch:03d}-{val_loss:.4f}.hdf5"])

            callbacks = [ModelCheckpoint(fname,  monitor='val_loss', save_freq='epoch')]
            training_args = dict() 
            param_kws = ('validation_data', 'batch_size', 'epochs', 'callbacks', 'verbose')
            for kw in param_kws:
                training_args[kw] = locals()[kw]
            if trait_to_enforce == 'both':
                sf_args = {'factor': FACTOR, 
                           'verbose' : True, 
                           'rel_comps' : relational_labels,
                           'relational_split' : relational_split}
            else:
                sf_args = {'factor': FACTOR, 
                           'verbose' : True, 
                           'rel_comps' : relational_labels[trait_to_enforce]}
            pdb.set_trace()
            mt = utils.ModelTrainer(model, train_data,
                                    train_args=training_args,
                                    eval_args={'batch_size':batch_size},
                                    sfunc=smooth_func,
                                    func_args=sf_args,
                                    classnames=target_names)
            print(mt.c_matrix)
            print(mt.report)
            save_vars = (mt.H.history, mt.report, mt.c_matrix)
            fn_out = os.path.sep.join([checkpoints_path, 
                                        'exp_data_sf-{:.2f}-trait-{}.pkl'.format(FACTOR,trait_to_enforce)])
            with open(fn_out, 'wb') as f:
                pickle.dump(save_vars, f)
            tf.keras.backend.clear_session()
            gc.collect()
            
            if args['bash']:
                sys.stdout = orig_stdout
            
            # plot the training loss and accuracy
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, epochs), mt.H.history["loss"], label="train_loss")
            plt.plot(np.arange(0, epochs), mt.H.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, epochs), mt.H.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, epochs), mt.H.history["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(os.path.join(checkpoints_path, 'training_plot.png'))
