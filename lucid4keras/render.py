import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.layers import Lambda,Input
from keras.models import Model

import lucid.optvis.param as param
from lucid.optvis.render import make_t_image, make_transform_f, make_optimizer
from lucid.misc.io import show
from lucid.misc.io.serialize_array import _normalize_array as normalize_array
from . objectives import *
from . optimizers import *


def keras_render_vis(input_model, objective_f, param_f=None, optimizer=None,
               transforms=None, thresholds=[512], print_objectives=None,
               verbose=True, use_fixed_seed=False,raw = False):
    if use_fixed_seed:
        tf.set_random_seed(0)
        
    t_image, loss, train = keras_make_vis_T(input_model,
                                            objective_f,
                                            param_f,
                                            optimizer,
                                            transforms)
    if thresholds == int:
        thresholds = list(thresholds)
        
    cache_m = None
    cache_v = None
    lr = 0.05
    beta1 = 0.9
    beta2 = 0.999
    iters = 0
    images = []

    try:
        for i in range(max(thresholds)+1):
            loss,grads = train([t_image,0])
            step, cache_m, cache_v, iters = adam(grads,cache_m,cache_v,iters,lr,beta1,beta2)
            t_image += step
            if i in thresholds:
                vis = t_image
                images.append(vis)
                if verbose:
                    print(i, loss)
                    show(np.hstack(vis))
        try:
            del loss, train #clear graphs
            del _model
        except:
            pass
        
        return np.array(images)
    
    except KeyboardInterrupt:
        print("Interrupted optimization at step: ",i)
        print("will return the last iteration image only")
        vis = t_image
        show(np.hstack(vis))
        
        del loss, train #clear graphs
        return normalize_array(np.hstack(vis))

def keras_make_vis_T(input_model, objective_f, param_f=None, optimizer=None,
               transforms=None):
    def connect_model(bottom,input_size,transform_f):
        '''
        connect the keras model with transformation function defined by transform_f with lambda
        '''
        input_tensor = Input(shape=input_size)
        transformed = Lambda(lambda inputs: transform_f(inputs),name="transform_layer")(input_tensor)
        top = Model(inputs=input_tensor,outputs=transformed)

        _model = Model(inputs = top.input,
                  outputs = bottom(top.output))
        return _model
    

    _t_image = make_t_image(param_f)
    _t_image = K.eval(_t_image)
    #have to be isolated from gpu for fast calculation. maybe
    #maybe if the adam calculation is on pure gpu, becomes faster and K.eval is no longer needed

    input_size = _t_image.shape[1:]

    transform_f = make_transform_f(transforms)
    target_model =connect_model(input_model,input_size,transform_f)
        
    
    objective_f = as_objective(objective_f)
    
    
    #create (batch,size,size,channel) in case object function requires multiple input by add or subtraction
    try:
        batch = int(objective_f.batch_n)
    except:
        batch = 1

    if batch > 1:
        t_image = np.zeros((batch,_t_image.shape[1],_t_image.shape[2],_t_image.shape[3]))
        for k in range(batch):
            t_image[k] = _t_image[0]
    else:
        #or do nothing
        t_image = _t_image

    #elif :
    #    pass
        #or interpolation
    

    #optimizer = make_optimizer(optimizer, [])
    loss = objective_f(target_model)
    grads = K.gradients(loss,target_model.input)[0]
    train = K.function([target_model.input, K.learning_phase()], [loss, grads])    
    return t_image, loss, train