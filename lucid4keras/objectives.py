import numpy as np
import tensorflow as tf
from keras.layers import Lambda,Input
import keras.backend as K
from decorator import decorator

class keras_Objective(object):
    def __init__(self, objective_func, name="", description="",batch_n=1):
        self.objective_func = objective_func
        self.name = name
        self.description = description
        self.batch_n = batch_n

    def __add__(self, other):
        if isinstance(other, (int, float)):

            objective_func = lambda T: other + self(T)
            name = self.name
            description = self.description
            batch_n = int(self.batch_n)
        else:
            objective_func = lambda T: self(T) + other(T)
            name = ", ".join([self.name, other.name])
            
            if (self.description[self.description.find("]") - 1]) != (other.description[other.description.find("]") - 1]):
                #temp nasty coding. will replace by a regular expression in future. Y.T.
                #above means finding batch dementions [neuron, 0] same batch, dont add batch n but add1 if different
                batch_n = int(self.batch_n) + 1
                description = "Align(" + " +\n".join([self.description, other.description]) + ")"
            else:
                batch_n = int(self.batch_n)
                description = "Sum(" + " +\n".join([self.description, other.description]) + ")"

        return keras_Objective(objective_func, name=name, description=description,batch_n=batch_n)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-1 * other)

    @staticmethod
    def sum(objs):
        objective_func = lambda T: sum([obj(T) for obj in objs])
        descriptions = [obj.description for obj in objs]
        description = "Sum(" + " +\n".join(descriptions) + ")"
        names = [obj.name for obj in objs]
        name = ", ".join(names) 
        return keras_Objective(objective_func, name=name, description=description,batch_n=obj.batch_n)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda T: other * self(T)
        else:
            objective_func = lambda T: self(T) * other(T)
        return keras_Objective(objective_func, name=self.name, description=self.description)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __call__(self, T):
        return self.objective_func(T)

def _make_arg_str(arg):
  arg = str(arg)
  too_big = len(arg) > 15 or "\n" in arg
  return "..." if too_big else arg

@decorator
def wrap_objective(f, *args, **kwds):
    """Decorator for creating Objective factories.

    Changes f from the closure: (args) => () => TF Tensor
    into an Obejective factory: (args) => Objective

    while perserving function name, arg info, docs... for interactive python.
    """
    objective_func = f(*args, **kwds)
    objective_name = f.__name__
    args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
    description = objective_name.title() + args_str

    #sum and interpolation identification
    if "Interpolate" in description:
        batch_n = [_make_arg_str(arg) for arg in args][-1]
    else:
        batch_n = 1 #for summing up layers
    return keras_Objective(objective_func, objective_name, description, batch_n)

@wrap_objective
def channel(n_channel, batch=None):
    """Visualize a single channel"""
    if batch is None:
        return lambda T: K.mean(T.output[..., n_channel])
    else:
        return lambda T: K.mean(T.output[batch,..., n_channel])

def _dot(x, y):
    return tf.reduce_sum(x * y, -1)

def _dot_cossim(x, y, cossim_pow=0):
    eps = 1e-4
    xy_dot = _dot(x, y)
    if cossim_pow == 0: return tf.reduce_mean(xy_dot)
    x_mags = tf.sqrt(_dot(x,x))
    y_mags = tf.sqrt(_dot(y,y))
    cossims = xy_dot / (eps + x_mags ) / (eps + y_mags)
    floored_cossims = tf.maximum(0.1, cossims)

    return tf.reduce_mean(xy_dot * floored_cossims**cossim_pow)

@wrap_objective
def direction(vec, batch=None, cossim_pow=0):
    """Visualize a direction"""
    if batch is None:
        vec = vec[None, None, None]
        return lambda T: _dot_cossim(T.output, vec)
    else:
        vec = vec[None, None]
        return lambda T: _dot_cossim(T.output[batch], vec)

@wrap_objective
def channel_interpolate(n_channel1, n_channel2, batch_n):
    def inner(T):
        #batch_n = T(layer1).get_shape().as_list()[0]
        #Note Y.T.
        #i do not understand the above code. in keras, its number of neurons? then batch should be really large. passing argv for now.
        #batch_n = T.output.get_shape().as_list()[0]
        arr1 = T.output[..., n_channel1]
        arr2 = T.output[..., n_channel2]

        weights = (np.arange(batch_n)/float(batch_n-1))
        S = 0
        for n in range(batch_n):
          S += (1-weights[n]) * tf.reduce_mean(arr1[n])
          S += weights[n] * tf.reduce_mean(arr2[n])
        
        arr3 = batch_n

        return S
    return inner


def as_objective(obj):
    if isinstance(obj, keras_Objective):
        return obj
    elif callable(obj):
        return obj
    elif isinstance(obj, str):
        return channel(int(obj))
    elif isinstance(obj, int):
        return channel(obj)
