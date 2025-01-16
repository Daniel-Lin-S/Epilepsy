from mne.io.edf.edf import RawEDF
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union

from preprocess import get_raw_data, get_seizure_times


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
        e.g. 'figures/ECG.png'
    """
    _check_times(start_time, end_time, mark_time)

    channel_names = raw.info['ch_names']
    
    # Extract data for the selected channels in [start_time, end_time]
    # Note: mne times are in seconds
    sf = raw.info['sfreq']  # sampling frequency
    start_stamp = int(start_time * sf)
    end_stamp = int(end_time * sf)
    data, times = raw[:, start_stamp:end_stamp]  # (channels, length)

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


if __name__ == '__main__':
    patient_id = 'DA0441I5'
    l_forward = 30
    l_backward = 20
    channels = np.arange(21, 26)
    raw = get_raw_data('./data/seizure-data-annotated', patient_id)

    onset_dict = get_seizure_times(raw, in_seconds=True, verbose=False)

    ids = list(onset_dict.keys())
    seizure_id = ids[0]

    seizure_start_time = onset_dict[seizure_id]['start']
    plot_channels(raw, seizure_start_time-l_forward,
                  seizure_start_time+l_backward,
                  channel_idxs = channels,
                  mark_time=seizure_start_time,
                  mark_name='seizure starts',
                  title=f'ECG signals before and after seizure {seizure_id} starts',
                  file_name=f'./figures/{patient_id}_seiz{seizure_id}.png')
    
    seizure_start_time = onset_dict[seizure_id]['start']
    plot_channels(raw, seizure_start_time-(l_forward+l_backward),
                  seizure_start_time,
                  channel_idxs = channels,
                  title=f'ECG signals before seizure starts',
                  file_name=f'./figures/{patient_id}_seiz{seizure_id}_before.png')
    
    plot_channels(raw, 0,
                  l_forward+l_backward,
                  channel_idxs = channels,
                  title=f'ECG signals with no seizure',
                  file_name=f'./figures/{patient_id}_no_seiz.png')
