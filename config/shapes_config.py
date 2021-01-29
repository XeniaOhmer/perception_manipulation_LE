import os

# set dataset-specific parameters
DATASET_PARAMS = dict()
DATASET_PARAMS['input_shape'] = (64, 64, 3)
num_classes = 64
DATASET_PARAMS['num_classes'] = num_classes
#DATASET_PARAMS['labels'] = ['red square', 'green square', 'blue square', \
#                            'red circle', 'green circle', 'blue circle', \
#                            'red triangle', 'green triangle', 'blue triangle']
DATASET_PARAMS['labels'] = [str(i) for i in range(num_classes)]
DATASET_PARAMS['foi'] = [10, 25, 50, 100, 150]

_DEFAULT_OUTPUT_DIRECTORY = 'plots'
_DEFAULT_CHECKPOINT_DIRECTORY = 'checkpoints'


def get_parameter_variables(args):
    if 'outdir' not in args.keys():
        args['outdir'] = _DEFAULT_OUTPUT_DIRECTORY
    if 'checkpoints' not in args.keys():
        args['checkpoints'] = _DEFAULT_CHECKPOINT_DIRECTORY
    if 'exp' not in args.keys():
        args['exp'] = ''
    if 'fn_pattern' not in args.keys():
        FN_PATTERN = 'weights-sfactor-*.hdf5'
    else:
        FN_PATTERN = args['fn_pattern']

    BASE_OUT_DIR = os.path.join(args['outdir'])
    EXP_DIR = os.path.join(args['checkpoints'])

    if args['bools'] is None:
        BOOL_CONFIG = dict()
        BOOL_CONFIG['normalize_dists'] = False
        BOOL_CONFIG['print_confusion'] = True
        BOOL_CONFIG['compiled_analysis'] = False
        BOOL_CONFIG['save_overview'] = True
    else:
        BOOL_CONFIG = args['bools']

    return (BASE_OUT_DIR, EXP_DIR, BOOL_CONFIG, FN_PATTERN)

def get_default_training_config():
    epochs = 150
    init_lr = 1e-3
    batch_size = 128
    verbose = True

    return (epochs, init_lr, batch_size, verbose)
