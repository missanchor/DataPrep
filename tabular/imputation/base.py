import tensorflow as tf
from dataprep.base import BaseEstimator

class Imputation_Estimator(BaseEstimator):
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.sess is not None:
            vars_list = tf.compat.v1.global_variables()
            vars_vals = self.sess.run(vars_list)
            state['saved_weights_'] = {v.name: val for v, val in zip(vars_list, vars_vals)}
        if 'sess' in state: del state['sess']
        if 'graph' in state: del state['graph']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sess = None
        self.graph = None
