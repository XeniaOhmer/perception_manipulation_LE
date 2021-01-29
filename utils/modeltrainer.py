import pdb

from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import pickle

def transform_labels(train_y, f, **args):
    transformed_y = f(train_y, **args)
    return transformed_y

class ModelTrainer:
    def __init__(self, model, train_input, 
                 train_args=None,
                 do_eval=True,
                 eval_args=None,
                 sfunc=None,
                 func_args=None,
                 classnames=None,
                 fit_func=None,
                 **args):
        self.model = model
        self.smooth_f= sfunc
        self.train_args = train_args
        self.meta = dict()
        self.class_names = classnames
        if fit_func is None:
            self.fit_func = model.fit
        else:
            self.fit_func = fit_func
        # making the assumption here that train_input is either a 2-d tuple of
        # form (x,y) or a generator, in which case any label smoothing should be
        # managed within the generator itself rather than with the sfunc parameter
        if sfunc is not None and isinstance(train_input,tuple):
            self.meta['smoothf_args'] = func_args
            train_y = transform_labels(train_input[1], sfunc, **func_args)
            train_input = (train_input[0], train_y)
        if not isinstance(train_input,tuple):
            if isinstance(train_input,list):
                train_input = tuple(train_input)
            else:
                train_input = (train_input,)
        self.train_model(train_input, 
                         **self.train_args)
        if do_eval:
            if eval_args is not None:
                self.evaluate_model(*train_args['validation_data'], **eval_args)
            else:
                self.evaluate_model(*train_args['validation_data'])


    def train_model(self, train_input, **train_args):
            if train_args is not None:
                self.H = self.fit_func(*train_input, **train_args)
            else:
                self.H = self.fit_func(*train_input)

    def evaluate_model(self, test_x, test_y, **args):
        if args is not None:
            predictions = self.model.predict(test_x, **args)
            self.report = classification_report(test_y.argmax(axis=1),                                                predictions.argmax(axis=1),
                                                predictions.argmax(axis=1),
                                                target_names=self.class_names)
            self.c_matrix = confusion_matrix(test_y.argmax(axis=1),
                                             predictions.argmax(axis=1))
        else:
            predictions = self.model.predict(test_x)
            self.report = classification_report(test_y)
            self.c_matrix = confusion_matrix(test_y.argmax(axis=1),
                                             predictions.argmax(axis=1))
    
    # should be updated to save weights in hdf5, or point to checkpoint file optionally
    # using pickle will be inefficient for saving the model itself
    def save_modelinfo(self, filepath):
        with open(filepath) as f:
            pickle.dump(self,f)
