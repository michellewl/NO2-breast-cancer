import numpy as np

def mape_score(targets, predictions):
    zero_indices = np.where(targets == 0)
    targets_drop_zero = np.delete(targets, zero_indices)
    prediction_drop_zero = np.delete(predictions, zero_indices)
    mape = np.sum(np.abs(targets_drop_zero - prediction_drop_zero)/targets_drop_zero) * 100/len(targets_drop_zero)
    return mape