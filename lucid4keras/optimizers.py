import numpy as np

def rmsprop(grads, cache=None, decay_rate=0.95):
    if cache is None:
        cache = np.zeros_like(grads)
    cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
    step = -grads / np.sqrt(cache + 1e-8)
    
    return step, cache

def adam(grads, cache_m,cache_v,iters,
         lr, beta1, beta2):
    if cache_m is None:
        cache_m = np.zeros_like(grads)
    if cache_v is None:
        cache_v = np.zeros_like(grads)

    iters += 1
    
    cache_m = (beta1 * cache_m) + (1. - beta1) * grads
    cache_v = (beta2 * cache_v) + (1 - beta2) * (grads * grads)
    mc = cache_m / (1. - (beta1 ** iters))
    vc = cache_v / (1. - (beta2 ** iters))
    
    lr_t = lr * np.sqrt(1. - (beta2 ** iters)) / (1. - (beta1 ** iters)) #learning rate
    step = (lr_t * mc)/ (np.sqrt(vc) + 1e-8)
    
    return step, cache_m, cache_v, iters