from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten, Input

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from numpy.core.umath_tests import inner1d

import argparse
import os
import pdb
import pickle


def generalized_cosine(A,B,speedup=True):
#   cos_AB = np.sum(inner1d(A, B)) / \
#       np.sqrt( np.sum(inner1d(A, A)) * np.sum(inner1d(B, B)) )
    @tf.function
    def f(x,y):
        return tf.linalg.trace(tf.linalg.matmul(x,y)) / \
                    tf.math.sqrt( tf.linalg.trace(tf.linalg.matmul(x,x)) * \
                             tf.linalg.trace(tf.linalg.matmul(y,y)) )

    a = tf.constant(A)
    b = tf.constant(B)
    cos_AB = f(a,b).numpy()
    
    return cos_AB



def calc_R(data,normalize=False,white_noise=0):


#   if normalize == True:
#       data = data / np.sqrt(np.sum(X**2,aXis=1)[:,np.newaxis])

#   result = np.tensordot(data,data,axes=[0,0]) / np.shape(data)[0]

    @tf.function
    def f(x):
        return tf.linalg.matmul(x,x, transpose_a=True) / x.shape[0]

    X = block_diagonal(data)
    R = f(X).numpy()
    
    return R

def block_diagonal(matrices, dtype=tf.float16):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.
  
    Args:
      matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
        matrices with the same batch dimension).
      dtype: Data type to use. The Tensors in `matrices` must match this dtype.
    Returns:
      A matrix with the input matrices stacked along its main diagonal, having
      shape [..., \sum_i N_i, \sum_i M_i].
  
    """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.compat.v1.Dimension(0)
    blocked_cols = tf.compat.v1.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
      full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
      batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
      blocked_rows += full_matrix_shape[-2]
      blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
      matrix_shape = tf.shape(matrix)
      ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
      matrix_shape = tf.shape(matrix)
      row_before_length = current_column
      current_column += matrix_shape[-1]
      row_after_length = ret_columns - current_column
      row_blocks.append(tf.pad(
          tensor=matrix,
          paddings=tf.concat(
              [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
               [(row_before_length, row_after_length)]],
              axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))

    return blocked

def sort_data(data, labels):
    data_sorted = dict()
    for i in range(0,10):
        data_sorted[str(i)] = []
    for (idx,entry) in enumerate(data):
        data_sorted[str(labels[idx])].append(list(entry))
    
    for k in data_sorted.keys():
        data_sorted[k] = np.array(data_sorted[k])
    
    return data_sorted

def gather_activations(base_model, data_in,
                       write_hists=False,
                       outdir='',
                       epoch=None,
                       layer_keys=None,
                       include_input=False):
    activations = dict()
    weights = dict()
    layer_info = []
    for (idx,k) in enumerate(data_in.keys()):
        activations[k] = []
        weights[k] = []
        if include_input:
            out = data_in[k].reshape(data_in[k].shape[0], -1)
            activations[k].append(out)
            weights[k].append([])
            
        for layer in base_model.layers:
            if layer.name.find('flatten') >= 0:
                continue
            if idx == 0:
                layer_info.append(layer.get_config())

            model = Model(base_model.input, outputs=layer.output)
            if layer.name.find('conv') >= 0 or \
                    layer.name.find('pool') >= 0:
                flatten = Flatten()(layer.output)
                model = Model(inputs=base_model.input, outputs=flatten)

            out = model.predict(data_in[k])
            activations[k].append(out)
            try:
                weights[k].append(layer.weights)
            except:
                weights[k].append([])
        if layer_keys is None:
            layer_keys = [str(x) for x in range(len(activations[k]))]
    test = activations['0']

    if write_hists:
        hist_data = write_histograms(activations, weights, layer_keys,
                                     epoch=epoch,
                                     outdir=outdir)
        return activations, layer_keys, hist_data
    else:
        return activations, layer_keys
