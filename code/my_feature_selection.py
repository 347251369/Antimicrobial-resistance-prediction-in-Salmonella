from sklearn.feature_selection import chi2
import pandas as pd
from fastCMIM import fast_cmim

def fs_chi2(X, y, p_values_threshold=0.01):
    _, p_values = chi2(X, y)
    p_values = pd.DataFrame(p_values, columns=['p_value'], index=X.columns)
    feats = p_values[p_values['p_value'] < p_values_threshold].index
    return feats

def fs_cmim(X, y, n_selected_features, verbose=False):
    tmp = X.values
    idxs, maxCMI = fast_cmim(tmp, y, n_selected_features=n_selected_features, verbose=verbose)
    feats  = X.iloc[:, idxs].columns
    return feats, maxCMI