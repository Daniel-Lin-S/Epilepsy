import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt


def calculate_c_index(
        y_true_time: np.ndarray, y_pred_time: np.ndarray,
        event_observed: np.ndarray) -> float:
    """
    Computes the Concordance Index (C-index) for survival analysis.

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
    mask_event = event_observed == 1
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


def plot_pr_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
    """
    Plot the Precision-Recall curve.
    Used to evaluate probablistic binary classifiers.

    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth labels.
    y_pred_proba : numpy.ndarray
        Predicted probabilities.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_pred_proba)
    
    plt.plot(thresholds, precision[:-1],
             label="Precision", linestyle="--", color="blue")
    plt.plot(thresholds, recall[:-1],
             label="Recall", linestyle="-", color="red")

    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title("Recall vs. Threshold")
    plt.legend()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    # Compute AUC (Area Under Curve)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray",
             label="Random Model (AUC = 0.50)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid(True)
    plt.show()