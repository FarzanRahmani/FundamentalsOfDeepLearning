import numpy as np

def forward_batchnorm(Z, gamma, beta, eps, cache_dict, beta_avg, mode):
    """
    Performs the forward propagation through a BatchNorm layer.

    Arguments:
    Z -- input, with shape (num_examples, num_features) -> (N, D)
    gamma -- vector, BN layer parameter -> (D,) for scaling
    beta -- vector, BN layer parameter -> (D,) for shifting
    eps -- scalar, BN layer hyperparameter -> epsilon for numerical stability
    beta_avg -- scalar, beta value to use for moving averages (or call it momentum parameter) -> 0.9
    cache_dict -- dictionary, containing the moving averages of mu and var, and the current mode
    mode -- boolean, indicating whether used at 'train' or 'test' time

    Returns:
    out -- output, with shape (num_examples, num_features) -> (N, D)
    """

    if mode == 'train':
        # TODO: Mean of Z across first dimension
        mu = np.mean(Z, axis=0) # (N, D) -> (D,)

        # TODO: Variance of Z across first dimension
        var = np.var(Z, axis=0) # (N, D) -> (D,)
        # var = np.mean((Z - mu)**2, axis=0) 

        # Take moving average for cache_dict['mu']
        cache_dict['mu'] = beta_avg * cache_dict['mu'] + (1-beta_avg) * mu

        # Take moving average for cache_dict['var']
        cache_dict['var'] = beta_avg * cache_dict['var'] + (1-beta_avg) * var

    elif mode == 'test':
        # TODO: Load moving average of mu
        mu = cache_dict['mu']

        # TODO: Load moving average of var
        var = cache_dict['var']

    # TODO: Apply z_norm transformation
    Z_norm = (Z - mu) / np.sqrt(var + eps) # (N, D) -> (N, D)

    # TODO: Apply gamma and beta transformation to get Z tiled
    out = gamma * Z_norm + beta # (N, D) -> (N, D) 

    return out


# test the above function in a simple case
Z = np.random.randn(3, 5) + 100
print("Z:")
print(Z)
# gamma = np.random.randn(5)
# beta = np.random.randn(5)
gamma = np.ones(5)
beta = np.zeros(5)
eps = 1e-5
beta_avg = 0.9
mode = 'train'
cache_dict = {'mu': np.zeros((5,)), 'var': np.zeros((5,))}

print("Train mode")
out = forward_batchnorm(Z, gamma, beta, eps, cache_dict, beta_avg, mode)
print("out:")
print(out)
print("mu of out:")
print(np.mean(out, axis=0))
print("var of out:")
print(np.var(out, axis=0))
print("cache_dict:")
print(cache_dict)

print("-----------------------------------")
print("Test mode")
# test it in test mode
mode = 'test'
cache_dict = {'mu': np.mean(Z, axis=0), 'var': np.var(Z, axis=0)}
out = forward_batchnorm(Z, gamma, beta, eps, cache_dict, beta_avg, mode)
print("out:")
print(out)
print("mu of out:")
print(np.mean(out, axis=0))
print("var of out:")
print(np.var(out, axis=0))
