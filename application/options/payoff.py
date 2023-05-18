import numpy as np

def european_payoff(x, K, option_type):
    type = option_type.upper()

    if type == 'CALL':
        return np.maximum(x-K, 0.0)
    elif type == 'PUT':
        return np.maximum(K-x, 0.0)
    else:
        raise NotImplemented('The passed option-type ({}) has not been implemented'.format(type))

