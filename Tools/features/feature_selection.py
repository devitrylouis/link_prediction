import pandas as pd
import numpy as np

def log_transform(training_features, list_names = ['comm_auth', 'in_in', 'out_out', 'in_out', 'out_in']):
    for name in list_names:
        if inplace:
            training_features[[name]] = np.log(training_features[[name]] + 1)
        else:
            training_features[['log_' + name]] = np.log(training_features[[name]] + 1)
    return(training_features)
