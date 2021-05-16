from utils.train import load_data
from communication_game.utils.referential_data import *
from communication_game.utils.config import *
from communication_game.nn.agents import *
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

dataset = '3Dshapes_subset'
all_cnn_paths, image_dim, n_classes, feature_dims, zero_shot_cats = get_config(dataset)
layer_name = {0: 'dense_2', 1: 'dense_1', 2: 'dense', 3: 'conv2d_1', 4: 'conv2d', 5: 'conv2d_input'}


color_indices = [np.arange(0,16), np.arange(16,32), np.arange(32,48), np.arange(48, 64)]
scale_indices = [np.concatenate([[0+x, 1+x, 2+x, 3+x] for x in range(0, 64, 16)]),
                 np.concatenate([[0+x, 1+x, 2+x, 3+x] for x in range(4, 64, 16)]),
                 np.concatenate([[0+x, 1+x, 2+x, 3+x] for x in range(8, 64, 16)]),
                 np.concatenate([[0+x, 1+x, 2+x, 3+x] for x in range(12, 64, 16)])]
shape_indices = [np.arange(0, 64, 4), np.arange(1,64,4), np.arange(2, 64, 4), np.arange(3, 64, 4)]


all_indices = np.arange(64)
other_color_indices = [all_indices[~np.isin(all_indices, color_indices[i])] for i in range(4)]
other_shape_indices = [all_indices[~np.isin(all_indices, shape_indices[i])] for i in range(4)]
other_scale_indices = [all_indices[~np.isin(all_indices, scale_indices[i])] for i in range(4)]

layer_name = {0: 'dense_2',
              1: 'dense_1',
              2: 'dense'}


def class_similarity_matrix(features, n_examples, n_classes=64):
    """ calculates matrix with pairwise object (class) similarities

    :param features:    CNN features extracted for objects, sorted by class (first n_examples belong to class 0 etc)
    :param n_examples:  number of examples per class
    :param n_classes:   number of classes
    :return:            matrix with pairwise class similarities, entry i,j is similarity between class i and class j
    """
    
    similarity = 1 - pdist(features, metric='cosine')
    similarity = squareform(similarity)
   
    np.fill_diagonal(similarity, np.nan)
    
    sim_matrix = np.zeros((n_classes, n_classes))
    for ind_i, i in enumerate(range(0, n_classes*n_examples, n_examples)):
        for ind_j, j in enumerate(range(0, n_classes*n_examples, n_examples)):
            sim_matrix[ind_i, ind_j] = np.nanmean(similarity[i:i+n_examples, j:j+n_examples])
    return sim_matrix


def featurewise_similarity(similarity_matrix):
    """ calculate similarities to other classes sharing the same feature value for either color, size or shape

    :param similarity_matrix: similarity matrix as calculated with class_similarity_matrix
    :return: for each class, similarities with respect to other classes sharing the same feature for either color
             size or shape --> dimensionality is [n_classes, n_features], here [64,3]
    """
    
    similarities = np.zeros((n_classes, 3))
    
    for c in range(n_classes):
        
        for color_ind in color_indices: 
            color_ind_list = list(color_ind)
            if c in color_ind: 
                color_ind_list.remove(c)
                similarities[c, 0] = np.mean(similarity_matrix[c, color_ind_list])

        for scale_ind in scale_indices: 
            scale_ind_list = list(scale_ind)
            if c in scale_ind: 
                scale_ind_list.remove(c)
                similarities[c, 1] = np.mean(similarity_matrix[c, scale_ind_list])
                
        for shape_ind in shape_indices: 
            shape_ind_list = list(shape_ind)
            if c in shape_ind: 
                shape_ind_list.remove(c)
                similarities[c, 2] = np.mean(similarity_matrix[c, shape_ind_list])
        
    return similarities


def featurewise_dissimilarity(similarity_matrix):
    """ calculate similarities to other classes NOT sharing the same feature value for either color, size or shape

        :param similarity_matrix: similarity matrix as calculated with class_similarity_matrix
        :return:    for each class, similarities with respect to other classes NOT sharing the same feature for either
                    color size or shape --> dimensionality is [n_classes, n_features], here [64,3]
        """
    
    similarities = np.zeros((n_classes, 3))
    similarities_all = np.zeros((n_classes, 3))
    
    for c in range(n_classes): 
        
        for j, color_ind in enumerate(color_indices): 
            color_ind_list = list(other_color_indices[j])   
            if c in color_ind: 
                similarities[c, 0] = np.mean(similarity_matrix[c, color_ind_list])
                collect_color = color_ind_list
                
        for j, scale_ind in enumerate(scale_indices): 
            scale_ind_list = list(other_scale_indices[j])
            if c in scale_ind: 
                similarities[c, 1] = np.mean(similarity_matrix[c, scale_ind_list])
                collect_scale = scale_ind_list
                
        for j, shape_ind in enumerate(shape_indices): 
            shape_ind_list = list(other_shape_indices[j])
            if c in shape_ind: 
                similarities[c, 2] = np.mean(similarity_matrix[c, shape_ind_list])
                collect_shape = shape_ind_list
                
        alternative_colors = [ind for ind in set(collect_color) if (ind in collect_scale and ind in collect_shape)]
        alternative_scales = [ind for ind in set(collect_scale) if (ind in collect_color and ind in collect_shape)]
        alternative_shapes = [ind for ind in set(collect_shape) if (ind in collect_color and ind in collect_scale)]
        similarities_all[c, 0] = np.mean(similarity_matrix[c, alternative_colors])
        similarities_all[c, 1] = np.mean(similarity_matrix[c, alternative_scales])
        similarities_all[c, 2] = np.mean(similarity_matrix[c, alternative_shapes])
    return similarities, similarities_all


def show_vision_modules_similarities(cnn_keys, n_examples=50, layer=1, plot=True, print_sim=True, path='../../'):
    """ Calculate vision module biases for each feature, and plot the similarity matrices. """
    
    if not plot:
        sim_matrices = []
        
    for cnn_key in cnn_keys: 
        path_vision = all_cnn_paths[cnn_key]
        vision = tf.keras.models.load_model(path + path_vision)
        vision = tf.keras.Model(inputs=vision.input, outputs=vision.get_layer(layer_name[layer]).output)
        
        val_data, val_labels, _ = load_data((64, 64, 3),
                                            analysis_run=True,
                                            balance_type=2,
                                            balance_traits=True)
        
        labels = np.argmax(val_labels, axis=1)

        data_new = np.zeros((n_classes*n_examples, 64, 64, 3), dtype=np.float32)
        i = 0
        for c in range(n_classes):
            data_new[i:i+n_examples] = val_data[labels == c][0:n_examples]
            i = i+n_examples

        features = np.reshape(vision(data_new), (len(data_new), -1))
        similarity_matrix = class_similarity_matrix(features, n_examples)
        
        if plot: 
            plt.imshow(similarity_matrix, vmin=0., vmax=1.)
            plt.title(cnn_key)
            plt.colorbar()
            plt.show()
        else: 
            sim_matrices.append(similarity_matrix)
        
        if print_sim: 
            featurewise_sim = np.mean(featurewise_similarity(similarity_matrix), axis=0)
            dissim, dissim_all = featurewise_dissimilarity(similarity_matrix)
            featurewise_dissim = np.mean(dissim, axis=0)
            print('\nsim        color scale shape')
            print(cnn_key, np.round(featurewise_sim - featurewise_dissim, 3))
            if 'all' in cnn_key: 
                featurewise_dissim = np.mean(dissim_all, axis=0)
                print('\nsim               color scale shape')
                print(cnn_key + ' corrected', np.round(featurewise_sim - featurewise_dissim, 3))

    if not plot:
        return sim_matrices
