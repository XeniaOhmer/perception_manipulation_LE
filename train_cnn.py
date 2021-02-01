# USAGE
# python train_cnn.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from models import GenericNet
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow import device
import tensorflow as tf

import utils
import matplotlib.pyplot as plt
import gc
import pdb

from pathlib import Path

from config import shapes_config as config 

# to fix cudnn handle error in tf 2.X
utils.train.configure_gpu_options()
args = utils.train.get_command_line_args(run_type='train')

input_shape = config.DATASET_PARAMS['input_shape']
num_classes = config.DATASET_PARAMS['num_classes']
epochs, init_lr, batch_size, verbose = config.get_default_training_config()
epochs = 200

# needs to be set to location of 3dshapes.h5 file from tensorflow 3dshapes dataset
data_path = '3dshapes.h5'
model_params = utils.train.load_default_model_params()

print("[INFO] loading data...")
train_data, validation_data, target_names, relational_labels = utils.train.load_data(input_shape,
                                                                                     balance_traits=True,
                                                                                     output_samples=False,
                                                                                     data_path=data_path)
num_classes = 64

smooth_func = utils.train.sum_label_components

for trait_to_enforce in ['objectHue', 'scale', 'shape', 'all']:
    if trait_to_enforce == 'all':
        sf_list = [0.5, 0.6, 0.8, 0.0]
    else:
        sf_list = [0.6, 0.5]
    for FACTOR in sf_list: 
        with device('/gpu:' + str(args["gpu"])): 
            opt = SGD(lr=init_lr)
            model = GenericNet.build(*input_shape, num_classes, **model_params)
            model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
            checkpoints_path = os.path.join(args['checkpoints'], trait_to_enforce, 'sf_'+ str(FACTOR).replace('.', '-'))
            if not os.path.isdir(checkpoints_path):
                Path(checkpoints_path).mkdir(parents=True)

            fname = os.path.sep.join([checkpoints_path, 
                                      "weights-sfactor-{:.2f}".format(FACTOR) + "-{epoch:03d}-{val_loss:.4f}.hdf5"])

            callbacks = [ModelCheckpoint(fname,  monitor='val_loss', save_freq='epoch')]
            training_args = dict() 
            param_kws = ('validation_data', 'batch_size', 'epochs', 'callbacks', 'verbose')
            for kw in param_kws:
                training_args[kw] = locals()[kw]
            sf_args = {'factor': FACTOR, 
                       'verbose' : True, 
                       'rel_comps' : relational_labels[trait_to_enforce]}
            mt = utils.ModelTrainer(model, train_data,
                                    train_args=training_args,
                                    eval_args={'batch_size':batch_size},
                                    sfunc=smooth_func,
                                    func_args=sf_args,
                                    classnames=target_names)
            tf.keras.backend.clear_session()
            gc.collect()
