from mne.io.edf.edf import RawEDF
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def plot_channels(raw : RawEDF,
                  start_time: float, end_time: float,
                  channel_idxs: Union[list, np.ndarray[np.int_]],
                  mark_time: Optional[float]=None,
                  mark_name: Optional[str]=None,
                  title: str='EEG Signals',
                  file_name: Optional[str]=None) -> None:
    """
    Plot first few channels of the data file
    across a specified time interval.
    
    Parameters
    ----------
    raw : RawEDF
        the data file
    start_time, end_time : float
        the start and end time (in seconds)
        of the interval to be plotted
    channel_idxs : list or numpy.ndarray
        the channel indices to be plotted. \n
    mark_time : float, optional
        If given, a red vertical dotted line will
        be plotted at that time.
    mark_time : str, optional
        If given, it will be used to label
        the red line when mark_time
        is provided.
    title : str, optional
        Main title of the plot. \n
        Default is 'EEG Signals'
    file_name : str, optional
        if provided,
        the figure will be saved instead of shown. \n
        e.g. 'figures/EEG.png'
    """
    _check_times(start_time, end_time, mark_time)

    channel_names = raw.info['ch_names']
    
    # Extract data for the selected channels in [start_time, end_time]
    # Note: mne times are in seconds
    sf = raw.info['sfreq']  # sampling frequency
    start_stamp = int(start_time * sf)
    end_stamp = int(end_time * sf)
    data, times = raw[:, start_stamp:end_stamp]  # data : (channels, length)

    # Plot the selected channels
    n_channels_plot = len(channel_idxs)
    fig, axes = plt.subplots(
        n_channels_plot, 1, figsize=(10, 2 * n_channels_plot),
        sharex=True, constrained_layout=True)

    for i, channel_idx in enumerate(channel_idxs):
        ax = axes[i] if n_channels_plot > 1 else axes
        channel_idx = channel_idxs[i]
        ax.plot(times, data[channel_idx, :], label=channel_names[channel_idx])
        ax.set_title(f"Channel: {channel_names[channel_idx]}", fontsize=10)

        if mark_time is not None:
            ax.axvline(x=mark_time, color='red', linestyle='--')
            
    if mark_name is not None and mark_time is not None:
        fig.text(0.95, 0.99, f"--- {mark_name}",
                 ha='right', va='top', fontsize=12, color='red')

    fig.suptitle(title, fontsize=16)
    if n_channels_plot > 1:
        axes[-1].set_xlabel('Time (s)')
    else:
        axes.set_xlabel('Time (s)')
    
    if file_name:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()


def _check_times(start_time: float, end_time: float,
                 mark_time: Optional[float]=None) -> None:
    """
    Raise
    -----
    ValueError
        if the three times does not match.
    """
    if start_time > end_time:
        raise ValueError(
            f'start time {start_time}, should be earlier than'
            f' end time {end_time}. ')
    
    if mark_time is not None:
        if mark_time < start_time or mark_time > end_time:
            raise ValueError(
                f'mark time {mark_time} should be in the interval'
                f' [{start_time}, {end_time}]. '
            )


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

    plt.xlabel("Threshold (%)")
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


def plot_predicted_vs_actual_times(
        times_true: np.ndarray, times_pred: np.ndarray,
        file_name: Optional[str]=None
    ) -> None:
    """
    Generate a scatter plot of predicted vs.
    ground truth times to event for survival analysis.
    Only non-negative and finite ground truth times
    will be plotted.

    Parameters
    ----------
    times_true : np.ndarray
        Ground truth event times.
    times_pred : np.ndarray
        Predicted event times.
    file_name : str, optional
        If provided, the plot will be saved
        instead of shown. \n
        e.g. 'figures/predicted_vs_actual.png'
    """

    # Select only non-negative and finite ground truth values
    valid_mask = (times_true >= 0) & np.isfinite(times_true)
    y_true_filtered = times_true[valid_mask]
    y_pred_filtered = times_pred[valid_mask]

    inf_mask = np.isinf(y_pred_filtered)

    # Set a fixed max value for infinite predictions
    if np.any(~inf_mask):
        max_finite = max(
            np.max(y_pred_filtered[~inf_mask]), np.max(y_true_filtered))
        inf_value = 1.5 * max_finite
    else:
        inf_value = 1.5 * np.max(y_true_filtered)

    y_pred_filtered[inf_mask] = inf_value

    # Plot finite values as scatter points
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_filtered[~inf_mask], y_pred_filtered[~inf_mask], 
                color='blue', alpha=0.6)

    # Plot infinite predictions as crosses ('x')
    plt.scatter(y_true_filtered[inf_mask], y_pred_filtered[inf_mask], 
                color='red', marker='x', s=100,
                label='Infinite (False Negative)')

    # Labels and title
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title("Scatter Plot of Predicted vs. Ground Truth Times to Event")
    plt.legend()
    plt.grid(True)

    if file_name:
        plt.savefig(file_name, dpi=300)
        plt.close()
    else:
        plt.show()
