import matplotlib
matplotlib.use("Agg")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from models import GenericNet
from tensorflow.keras.datasets import mnist
from tensorflow import device 
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1 import ConfigProto, Session
import numpy as np
import argparse
import pdb
import glob

from pathlib import Path
import pickle
import shutil

import utils

def get_distances_groundtruth(smoothing_factor):
    smoothing_factor
    _,_,_,relational_targets = get_relational_targets(smoothing_factor)
    mat_dim = relational_targets.shape[1]
    gt = np.zeros((mat_dim,mat_dim))
    for i in range(mat_dim):
        i_norm = np.linalg.norm(relational_targets[i])
        for j in range(i, mat_dim):
            j_norm = np.linalg.norm(relational_targets[j])
            gt[i,j] = (np.dot(relational_targets[i], relational_targets[j]) / (i_norm*j_norm))**2
            gt[j,i] = gt[i,j]
    gt = 1-gt
    
    return gt

# to fix cudnn handle error
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)
set_session(sess)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--outdir", type=str, default='plots/')
ap.add_argument("-c", "--checkpoints", type=str, default="checkpoints/")
ap.add_argument("-e", "--exp", type=str, default='conv_01_002_fc_02_0008')
ap.add_argument("-i", "--dataset", type=str, default="3D-shapes")
ap.add_argument("-g", "--gpu", type=int, default=1)
ap.add_argument("-b", "--bools", type=dict, default=None)
args = vars(ap.parse_args())


if args['dataset'] == 'mnist':
    from config import mnist_config as config 
    (BASE_OUTDIR, EXP_DIR, CTRL_BOOLS, FN_PATTERN) = config.get_parameter_variables(args)
    input_shape = config.DATASET_PARAMS['input_shape']
    num_classes = config.DATASET_PARAMS['num_classes']
    files_of_interest = config.DATASET_PARAMS['foi']
elif args['dataset'] == '3D-shapes':
    from config import shapes_config as config 
    (BASE_OUTDIR, BASE_EXP_DIR, CTRL_BOOLS, FN_PATTERN) = config.get_parameter_variables(args)
    input_shape = config.DATASET_PARAMS['input_shape']
    num_classes = config.DATASET_PARAMS['num_classes']
    files_of_interest = config.DATASET_PARAMS['foi']

architecture_id = 'conv_02_064_fc_02_0032'
num_classes = 64
cross_trial_data = dict()
write_plots = True
model_params = utils.train.load_default_model_params()
for i in range(num_classes):
    cross_trial_data[str(i)] = dict()

exp_identifier = '201227'
BASE_OUTDIR = os.path.join(BASE_OUTDIR, exp_identifier)
BASE_OUTDIR = os.path.join('tmp', exp_identifier)

X,Y, X_sorted = utils.train.load_data(input_shape, 
                                      dataset='3D-shapes',
                                      analysis_run=True, 
                                      balance_traits=True)
for trait in ['all', 'objectHue', 'shape', 'scale']:
    EXP_DIR = os.path.join(BASE_EXP_DIR, exp_identifier, trait, architecture_id)
    EXP_OUTDIR = os.path.join(BASE_OUTDIR, trait)
    for trial in os.listdir(EXP_DIR): 
        sf = trial[-3:].replace('-', '.') + '0'
        params_path = os.path.join(EXP_DIR, trial,'exp_data_sf-' + sf + '-trait-' + trait + '.pkl')
        # model_params_path = os.path.join(EXP_DIR, trial,'model_params.pkl')
        tmp = utils.utils.load_experiment_variables(params_path)
        if tmp is None: 
            break
        (model_history, report, c_matrices_orig) = tmp
#       with open(model_params_path, 'rb') as f:
#           model_params = pickle.load(f)
        checkpoint_dir = os.path.join(EXP_DIR, trial)
        OUT_DIR = os.path.join(EXP_OUTDIR, trial)
        pdb.set_trace()
        if not os.path.isdir(OUT_DIR):
            utils.utils.setup_output_directories(OUT_DIR, create_parents=True)
        else:
            print('warning: directory {} aready exists'.format(OUT_DIR))
        
        dist_matrices = {} 
        hist_data = {} 
        weights = {} 
        c_matrices = {} 
        fn_list = glob.glob(os.path.join(checkpoint_dir, FN_PATTERN))
        
        pdb.set_trace()
        with device('/gpu:' + str(args["gpu"])):
            for checkpoint_path in sorted(fn_list):
                fn = checkpoint_path.split('/')[-1]
                ke = fn[21:24]
                if int(ke) not in files_of_interest:
                    continue
                
                model = utils.utils.load_model_from_hdf5(checkpoint_path)
        print('[INFO] analysis complete.')
#if CTRL_BOOLS['save_overview']:      
#    utils.utils.dump_variables(cross_trial_data, 
#                         os.path.join(BASE_OUTDIR, 'cross_experiment_backup.pkl'))
#
#if CTRL_BOOLS['compiled_analysis']:
#    trial_outdir = os.path.join(BASE_OUTDIR, 'cross_trial_analysis')
#    if not os.path.isdir(trial_outdir):
#        Path(trial_outdir).mkdir(parents=True)
#    labels = dict()
#    labels['data'] = list(range(10))
#    labels['xlabel'] = 'digit'
#    print("[INFO] writing cross training analysis plots...")
#    utils.analysis.plot.write_crosstraining_plots(cross_trial_data, trial_outdir)
#    print("[INFO] ...complete.")
