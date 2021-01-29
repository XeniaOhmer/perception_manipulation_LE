
cnn_paths = {
    '3Dshapes_subset': {
        'default0-0': 'trained_cnns/3d-shapes_subset_weights/all-weights-sfactor-0.00-200-0.0021.hdf5',
        'all0-5': 'trained_cnns/3d-shapes_subset_weights/all-weights-sfactor-0.50-200-0.9039.hdf5',
        'all0-6': 'trained_cnns/3d-shapes_subset_weights/all-weights-sfactor-0.60-200-1.0843.hdf5',
        'all0-8': 'trained_cnns/3d-shapes_subset_weights/all-weights-sfactor-0.80-200-1.7984.hdf5',
        'color0-5': 'trained_cnns/3d-shapes_subset_weights/objectHue-weights-sfactor-0.50-200-0.8520.hdf5',
        'color0-6': 'trained_cnns/3d-shapes_subset_weights/objectHue-weights-sfactor-0.60-200-1.2947.hdf5',
        'scale0-5': 'trained_cnns/3d-shapes_subset_weights/scale-weights-sfactor-0.50-200-0.8453.hdf5',
        'scale0-6': 'trained_cnns/3d-shapes_subset_weights/scale-weights-sfactor-0.60-200-1.0954.hdf5',
        'shape0-5': 'trained_cnns/3d-shapes_subset_weights/shape-weights-sfactor-0.50-200-0.8635.hdf5',
        'shape0-6': 'trained_cnns/3d-shapes_subset_weights/shape-weights-sfactor-0.60-200-1.1110.hdf5'
    }
}

attribute_dict = {
    '3Dshapes_subset': {0: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                        1: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        2: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                        3: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        4: [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                        5: [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                        6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                        7: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        8: [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                        9: [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                        10: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                        11: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                        12: [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        13: [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                        14: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        15: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        16: [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                        17: [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        18: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                        19: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        20: [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                        21: [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                        22: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                        23: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        24: [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                        25: [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                        26: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                        27: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                        28: [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        29: [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                        30: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        31: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        32: [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                        33: [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        34: [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                        35: [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        36: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                        37: [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                        38: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                        39: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        40: [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                        41: [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                        42: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                        43: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                        44: [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        45: [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                        46: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        47: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        48: [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                        49: [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                        50: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                        51: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                        52: [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                        53: [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                        54: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                        55: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                        56: [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                        57: [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                        58: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                        59: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                        60: [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                        61: [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                        62: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                        63: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
                        }
}


image_dim = {'3Dshapes_subset': 64}
n_classes = {'3Dshapes_subset': 64}
feature_dims = {'3Dshapes_subset': [4, 4, 4]}
feature_order = {'3Dshapes_subset': ('color', 'scale', 'shape')}
zero_shot_categories = {'3Dshapes_subset': [0, 21, 42, 63]}


def get_feature_information(dataset='3Dshapes_subset'):
    return feature_dims[dataset], feature_order[dataset]


def get_attribute_dict(dataset='3Dshapes_subset'):
    return attribute_dict[dataset]


def get_config(dataset='3Dshapes_subset'):
    return (cnn_paths[dataset], 
            image_dim[dataset], 
            n_classes[dataset], 
            feature_dims[dataset], 
            zero_shot_categories[dataset]
           )

