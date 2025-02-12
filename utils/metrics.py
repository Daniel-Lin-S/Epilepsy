import numpy as np


def calculate_c_index(
        y_true_time: np.ndarray, y_pred_time: np.ndarray,
        event_observed: np.ndarray) -> float:
    """
    Computes the Concordance Index (C-index) for survival analysis.
    Only positive times (pre-event) are considered.

    Parameters:
    -----------
    y_true_time : numpy.ndarray
        Ground truth survival times (excluding np.inf for censored cases).
    y_pred_time : numpy.ndarray
        Predicted survival times.
    event_observed : numpy.ndarray
        Binary indicator (1 = event occurred, 0 = event not observed).

    Return
    -------
    float
        Concordance Index (C-index).
    """
    # Mask to ignore censored cases (where event_observed == 0)
    mask_event = (event_observed == 1) & (y_true_time >= 0)
    true_times = y_true_time[mask_event]
    predicted_times = y_pred_time[mask_event]

    concordant = 0
    discordant = 0
    tied = 0

    # Compare pairs of samples
    for i in range(len(true_times)):
        for j in range(i + 1, len(true_times)):
            if true_times[i] != true_times[j]:  # Ignore identical true times
                if (true_times[i] < true_times[j] and predicted_times[i] < predicted_times[j]) or \
                   (true_times[i] > true_times[j] and predicted_times[i] > predicted_times[j]):
                    concordant += 1
                elif (true_times[i] < true_times[j] and predicted_times[i] > predicted_times[j]) or \
                     (true_times[i] > true_times[j] and predicted_times[i] < predicted_times[j]):
                    discordant += 1
                else:
                    tied += 1

    c_index = (concordant + 0.5 * tied) / (concordant + discordant + tied)
    return c_index
