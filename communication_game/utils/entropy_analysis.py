import numpy as np
import tensorflow as tf
import sys
from communication_game.nn import agents
from communication_game.utils.config import get_config, get_attribute_dict
from sklearn.metrics import normalized_mutual_info_score as nmiscore
from sklearn.metrics import mutual_info_score as miscore
from utils.train import load_data
from scipy.stats import entropy
from communication_game.utils.referential_data import make_referential_data
import pandas as pd

dataset = '3Dshapes_subset'
all_cnn_paths, image_dim, n_classes, feature_dims, zero_shot_cats = get_config(dataset)
attribute_dict = get_attribute_dict(dataset)
nsamples = 10000
result_path = '3Dshapes_subset/'
path_prefix='../../'
vocab_size = 4
message_length = 3


def load_train():
    """ load the training data""" 
    (data, labels), _, _, _ = load_data((64, 64, 3), balance_type=2, balance_traits=True)
    return data, labels

def calc_entropy(X, base=None): 
    value, counts = np.unique(X, return_counts=True)
    return entropy(counts, base=base)

def joint_entropy(X, Y, base=None):
    XY = np.array([str(X[i]) + str(Y[i]) for i in range(len(X))])
    value, counts = np.unique(XY, return_counts=True)
    return entropy(counts, base=base)
    
def conditional_entropy(X, Y, base=None, normalizer='arithmetic'): 
    
    X_given_Y = joint_entropy(X, Y, base=base) - calc_entropy(Y, base=base)
    
    if normalizer is None: 
        normalized_entropy = X_given_Y
    elif normalizer == 'marginal': 
        normalized_entropy = X_given_Y / calc_entropy(X, base=base)
    if normalizer == 'arithmetic': 
        normalized_entropy = X_given_Y / (0.5 * (calc_entropy(Y, base=base) + calc_entropy(X, base=base)))
    elif normalizer == 'joint':
        normalized_entropy = X_given_Y / joint_entropy(X, Y, base=base)
        
    return normalized_entropy

def conditional_metric(X, Y, normalizer = 'joint'):
    return 1 - conditional_entropy(X, Y, normalizer=normalizer)


def get_effectiveness(conditions=['default', 'color', 'scale', 'shape', 'all'], runs=10, normalizer='marginal', mode='basic'):

    data, labels = load_train()
    data = data[0:nsamples]
    labels = np.argmax(labels[0:nsamples], axis=1)
    attributes = np.array([attribute_dict[l] for l in labels])

    color_labels = np.argmax(attributes[:, 0:4], axis=1) % 4
    scale_labels = np.argmax(attributes[:, 4:8], axis=1) % 4 
    shape_labels = np.argmax(attributes[:, 8:12], axis=1) % 4
    
    all_results = {}
    
    
    for condition in conditions: 
        
        if '_' in condition: 
            mode = 'mixed'
        else: 
            mode = 'basic'

        mi_scores = [[],[],[],[]]
        effectiveness_scores = [[],[],[],[]]
        efficiency_scores = [[],[],[],[]]
    
        for i in range(runs):

            run = condition + str(i)
            path = (path_prefix + 'communication_game/results/' + result_path + mode + '/' + run + '/vs' +  
                    str(vocab_size) + '_ml' + str(message_length) + '/')

            cnn_sender = tf.keras.models.load_model(path_prefix + all_cnn_paths['default0-0'])
            vision_module_sender = tf.keras.Model(inputs=cnn_sender.input, outputs=cnn_sender.get_layer('dense_1').output)
            sender = agents.Sender(vocab_size, message_length, 128, 128, activation='tanh', vision_module=vision_module_sender)

            sender.load_weights(path + 'sender_weights_epoch149/')
                
            messages = sender.forward(data, training=False)
            messages = np.array(messages[0][:])
            message_labels = []
            message_label_dict = {}

            label = -1
            for m in messages: 
                if str(m) in message_label_dict.keys(): 
                    message_labels.append(message_label_dict[str(m)])
                else: 
                    label = label + 1
                    message_labels.append(label)
                    message_label_dict[str(m)] = label
            
            effectiveness_scores[0].append(conditional_metric(color_labels, message_labels, normalizer=normalizer))
            effectiveness_scores[1].append(conditional_metric(scale_labels, message_labels, normalizer=normalizer))
            effectiveness_scores[2].append(conditional_metric(shape_labels, message_labels, normalizer=normalizer))

        all_results[condition] = {
                                  'effectiveness_scores': effectiveness_scores, 
                                  }
        
    return all_results
    

def get_residual_entropy(conditions=['default', 'all'], runs=10):
    
    data, labels = load_train()
    data = data[0:nsamples]
    labels = np.argmax(labels[0:nsamples], axis=1)
    attributes = np.array([attribute_dict[l] for l in labels])

    color_labels = np.argmax(attributes[:, 0:4], axis=1) % 4
    scale_labels = np.argmax(attributes[:, 4:8], axis=1) % 4 
    shape_labels = np.argmax(attributes[:, 8:12], axis=1) % 4
    
    all_results = {}
    residual_entropy = []

    for c, condition in enumerate(conditions): 

        residual_entropy_condition = []
        
        if '_' in condition: 
            mode = 'mixed'
        else: 
            mode = 'basic'

        for i in range(runs):

            run = condition + str(i)
            path = (path_prefix + 'communication_game/results/' + result_path + mode + '/' + run + '/vs' +  
                    str(vocab_size) + '_ml' + str(message_length) + '/')
            cnn_sender = tf.keras.models.load_model(path_prefix + all_cnn_paths['default0-0'])
            vision_module_sender = tf.keras.Model(inputs=cnn_sender.input, outputs=cnn_sender.get_layer('dense_1').output)
            sender = agents.Sender(vocab_size, message_length, 128, 128, activation='tanh', 
                                   vision_module=vision_module_sender)

            sender.load_weights(path + 'sender_weights_epoch149/')

            messages = sender.forward(data[0:nsamples], training=False)
            messages = np.array(messages[0][:])
            
            potential_re = []
            for partition in [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]:

                feature_wise_re = []
                for f in range(3): 
                    message_labels = messages[:, partition[f]]
                    feature_labels = [color_labels, scale_labels, shape_labels][f]
                    feature_wise_re.append(conditional_entropy(feature_labels, message_labels, normalizer='marginal'))
                potential_re.append(np.mean(feature_wise_re))
            residual_entropy_condition.append(min(potential_re))
                
        all_results[condition] = residual_entropy_condition
    
    return all_results


def get_effectiveness_dataframes(entropy_both_biased, entropy_sender_biased, entropy_receiver_biased, runs=10):
    
    all_dataframes = []
    for e, entropy_dict in enumerate([entropy_both_biased, entropy_sender_biased, entropy_receiver_biased]):
        
        data = {}
        biases = []
        effectiveness = []
        features = []

        conditions = [['default', 'color', 'scale', 'shape', 'all'],
                      ['default', 'color_default', 'scale_default', 'shape_default', 'all_default'],
                      ['default', 'default_color', 'default_scale', 'default_shape', 'default_all']]

        for b, bias in enumerate(conditions[e]):
            for f, feature in enumerate(['color', 'scale', 'shape']):
                for run in range(runs): 
                    biases.append(conditions[0][b])
                    features.append(feature)
                    effectiveness.append(entropy_dict[bias]['effectiveness_scores'][f][run])

        biases = [b if b!='scale' else 'size' for b in biases]
        features = [f if f!='scale' else 'size' for f in features]

        data['bias'] = biases
        data['feature'] = features
        data['effectiveness'] = effectiveness

        df = pd.DataFrame(data = data)
        all_dataframes.append(df)
    return all_dataframes