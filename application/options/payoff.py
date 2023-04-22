import numpy as np

def european_payoff(x, k, type):
    type = type.upper()

    if type == 'CALL':
        return np.maximum(x-k, 0)
    elif type == 'PUT':
        return np.maximum(k-x, 0)
    else:
        raise NotImplemented('The passed option-type ({}) has not been implemented'.format(type))

