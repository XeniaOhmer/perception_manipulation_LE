import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import AveragePooling2D

import numpy as np
from numpy.core.umath_tests import inner1d

import pickle
import pdb
from pathlib import Path
from datetime import datetime
from . import utils
from . import train

time_fmt = "%d/%m/%Y %H:%M:%S"
# placeholder, should be taken from config or inferred
input_shape = (64, 64, 3) 
num_classes = 64

def load_default_configuration(dataset):
    if dataset == '3D-shapes':
        from config import shapes_config as config
    else:
        raise ValueError("no default config file for dataset \'{}\'".format(dataset))
    
def generalized_cosine(A,B):
    numerator = np.sum(inner1d(A, B))
    denominator = np.sqrt( np.sum(inner1d(A, A)) * np.sum(inner1d(B, B)) )
    cosAB = numerator / denominator

    return cosAB

class TrainingManager:
    def __init__(self, 
                 dataset_id, 
                 model,
                 config=None,
                 data=None, 
                 output_path='', 
                 kwargs=None):
        self.correlations = None
        self.distances = None
        self.model = None
        self.meta = {'dataset' : dataset_id,
                     'layer_info' : None,
                     'model_file' : model_file,
                     'datetime' : datetime.now,
                     'output_path' : output_path}
        if config is None:
            self.exp_config = load_default_configuration(dataset_id)
        else:
            self.exp_config = config

    def run_default_training(self, save_weights=False, 
                                            verbose=False, 
                                            data_kwargs={}):
        train.configure_gpu_options()
        if verbose:
            now = datetime.now()
            print("[INFO-{}] beginning calculations for file {}".format(now.strftime(time_fmt),
                                                                        self.meta['model_file']))
            print("[INFO-{}] loading model and dataset {}...".format(now.strftime(time_fmt),
                                                                        self.meta['dataset']))
        self.model = utils.load_model_from_hdf5(self.meta['model_file'])
        X, Y, X_sorted = train.load_data(input_shape,
                                         dataset=self.meta['dataset'],
                                         analysis_run=True,
                                         **data_kwargs)
        if verbose:
            now = datetime.now()
            print("[INFO-{}] gathering activations...".format(now.strftime(time_fmt)))
        activations, self.meta['layer_info'] = self.gather_activations(self.model, X_sorted,
                                                                save_weights=save_weights)
        if verbose:
            now = datetime.now()
            print("[INFO-{}] calculating correlations...".format(now.strftime(time_fmt)))
        self.calculate_correlations(activations)

        if verbose:
            now = datetime.now()
            print("[INFO-{}] calculating distances...".format(now.strftime(time_fmt)))
        self.calculate_distances(num_classes)
        if verbose:
            print("[INFO-{}] calculations complete.")

