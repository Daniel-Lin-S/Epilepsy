import os
import numpy as np
import h5py
import mne
from mne.io.edf.edf import RawEDF
import random
from typing import List, Tuple, Optional, Union
import warnings

from utils.preprocess import read_seizure_times
from utils.tools import check_labels, check_seizure_overlap, distance_to_closest_seizure


# TODO - add other modes of sample balancing (only downsampling now)

def build_samples(
        folder_path: str, output_file: Optional[str]=None,
        selected_channels: Optional[List[str]]=None,
        seed: int=2025,
        verbose: bool=True, mode: str='classification',
        **kwargs) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
    """    
    Obtain positive (ictal) and negative (non-ictal) samples from EEG signals
    stored in .edf and .edf.seizure files,
    and transform them as x y pairs for model training.

    Parameters
    ----------
    folder_path : str
        The folder path containing the `.edf` files
        and `.edf.seizures` files.
    
    output_file : str
        If provided, the extracted dataset
        and labels will be saved in HDF5 format,
        and this function returns nothing.
    
    selected_channels : List[str], optional
        If given, only data from these channels will be kept.
    
    verbose : bool, optional
        if True, print the information of
        classification samples extracted.

    mode : str
        The type of samples being built.
        - 'classification' : samples stored with labels
        - 'clustering' : unsupervised samples for clustering

    kwargs : Any
        Passed to `get_{mode}_samples`. \n
        For classification: e.g. sample_time, preicetal_time,
        see docstring of get_classification_samples. \n
        For clustering: e.g. window_width, overlap,
        see docstring of get_clustering_samples.

    Return
    -------
    xs, ys : numpy.ndarray
        arrays of shape [num_samples, sample_length, num_channels]
        and [num_samples,] respectively. \n
        Only returned if output_file is not given. \n
        xs are the EEG signals and meaning of ys depends on
        the mode. 
        - 'classification' : labels, 0 for nonictal and 1 for preictal
        - 'clustering' : distance to the closest seizure period.
    """
    xs = []    # store EEG signals
    ys = []    # store labels / tags
    flag = True

    # Go through all the files in the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".edf"):
                if verbose:
                    print(f'Processing {file}')
                flag = False
                patient_id = os.path.splitext(file)[0]
                edf_file_path = os.path.join(root, file)

                raw_data = mne.io.read_raw_edf(
                    edf_file_path, preload=False, verbose=False)

                seizure_file = os.path.join(root, f"{patient_id}.edf.seizures")
                if not os.path.exists(seizure_file):
                    raise FileNotFoundError(
                        'Cannot find seizure time stamps '
                        'Please run annotation.py before '
                        'loading data.'
                    )
                seizure_times = read_seizure_times(seizure_file)

                if mode == 'classification':
                    ind_samples, ind_labels = get_classification_samples(
                        raw_data, seizure_times, selected_channels, seed, **kwargs)
                elif mode == 'clustering':
                    ind_samples, ind_labels = get_clustering_samples(
                        raw_data, seizure_times, selected_channels, seed, **kwargs)
                else:
                    raise ValueError(
                        'Unsupported mode, must be one of [classification, clustering].'
                    )

                if ind_samples.shape[0] != 0:
                    xs.append(ind_samples)
                    ys.append(ind_labels)

    if flag:
        raise FileNotFoundError(
            'Found no edf file in the provided folder'
            f' {folder_path}.'
        )

    # Convert lists to numpy arrays and store them
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)

    if mode == 'classification':
        check_labels(ys)

    if verbose:
        print(f'Extracted {xs.shape[0]} samples '
              f'with {xs.shape[1]} channels and '
              f'{xs.shape[2]} time stamps')
        
        if mode == 'classification':
            print(f'Positive samples: {sum(ys == 1)}')
            print(f'Negative samples: {sum(ys == 0)}')

    if output_file:
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('x', data=xs)
            f.create_dataset('y', data=ys)
    else:
        return xs, ys


def read_samples(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read samples and labels from h5 file.
    (assumed to be created by `process_data_and_store`)

    Parameter
    ---------
    file_path : str
        The path where the extracted dataset
        is saved in HDF5 format.
    
    Return
    -------
    x, y : numpy.ndarray
        arrays of shape [num_samples, num_channels, sample_length]
        and [num_samples,] respectively. \n
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Cannot find file {file_path}')

    with h5py.File(file_path, 'r') as f:
        return f['x'][:], f['y'][:]


def get_classification_samples(
        raw_data: RawEDF,
        seizure_times: List[Tuple[int, int]],
        selected_channels: Optional[List[str]]=None,
        seed: int=2025,
        sample_time: float=10.0, preictal_time: float=10.0,
        n_negative: int=-1,
        safe_gap: float=300.0
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract positive and negative samples from a RawEDF object
    based on seizure times (given as indices). \n
    The samples are taken from interval
    [seizure_start - sample_length - preictal_time,
      seizure_start - preictal_time]

    Parameters
    ----------
    raw_data : mne.io.edf.raw.RawEDF
        The RawEDF object containing the EEG data for the patient.
    
    seizure_times : List[Tuple[int, int]]
        A list of tuples representing the
        seizure start and end time stamps.

    selected_channels : List[str]
        If given, only data from these channels will be kept.
    
    seed : int, optional
        The random seed used for taking negative samples. \n
        Default is 2025.
    
    sample_time : float, optional
        The length of each sample in samples. \n
        Default is 60.0 seconds.
    
    preictal_time : float, optional
        The time in seconds before the seizure to
        extract as positive samples. \n
        Default is 10.0 seconds.

    n_negative : int, optional
        Number of negative samples to draw. \n
        Default is 5. \n
        Set it to -1 to use same
        number of negative samples
        as positive samples.
    
    safe_gap : float, optional
        The minimum distance (in seconds) between negative samples
        and seizure times to avoid overlap.
        Default is 300.0 seconds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays:
        - samples of shape (n_samples, n_channels, sample_length)
        - labels of shape (n_samples,) 0 for not preictal, 1 for preictal
    """
    random.seed(seed)

    positive_samples = []
    negative_samples = []

    sfreq = raw_data.info['sfreq']

    sample_length = int(sample_time * sfreq)

    if selected_channels:
        raw_data_selected = raw_data.pick(selected_channels)
    else:
        raw_data_selected = raw_data

    # Extract positive samples
    for start_idx, _ in seizure_times:
        pre_ictal_end = int(start_idx) - int(preictal_time * sfreq)
        pre_ictal_start = pre_ictal_end - sample_length

        if pre_ictal_start >= 0:
            positive_samples.append(
                raw_data_selected.get_data(
                    start=pre_ictal_start, stop=pre_ictal_end))
        else:
            raise Exception(
                'Not enough length before ictal stage. '
                f'required length: ({preictal_time} + {sample_time}) s'
                f', actual length {start_idx / sfreq} s. \n'
                'Please reduce preictal_time or sample_length.'
            )
    
    if n_negative == -1:
        n_negative = len(positive_samples)

    # Extract negative samples (random samples away from seizures)
    attempts = 0
    distance = int(safe_gap * sfreq)
    while len(negative_samples) < n_negative and attempts < 100:
        random_start = random.randint(0, len(raw_data.times) - sample_length)
        random_end = random_start + sample_length

        # ensure safe distance from seizure periods
        overlap = check_seizure_overlap(
            seizure_times, interval=(random_start, random_end),
            distance=distance)
        
        if not overlap:
            negative_samples.append(
                raw_data_selected.get_data(start=random_start, stop=random_end))
        attempts += 1

    if len(negative_samples) < n_negative:
        raise Exception(
            f"Only {len(negative_samples)} negative samples "
            "were extracted due to overlap issues. "
            "Try reduce sample_time or safe_gap."
        )
    
    positive_samples = np.array(positive_samples)   # (n, dim)
    negative_samples = np.array(negative_samples)   # (n, dim)
    samples = np.concatenate([positive_samples, negative_samples], axis=0)
    labels = np.concatenate(
        [np.ones(positive_samples.shape[0]), np.zeros(negative_samples.shape[0])],
        axis=0)

    return samples, labels


def get_clustering_samples(
        raw_data: RawEDF, seizure_times: List[Tuple[int, int]],
        selected_channels: Optional[List[str]]=None,
        seed: int=2025,
        window_width: float=5.0, overlap: float=1.0,
        preictal_interval: float=600.,
        preictal_width: Optional[float]=None,
        random_samples: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract samples for clustering from the long series
    of RawEDF object using sliding windows.
    - fixed windows before seizure starting time
    - random windows (not overlapping with ictal periods)

    Parameters
    ----------
    raw_data : RawEDF
        The long time series data.
    seizure_times : list of tuple of int
        List of tuples where each tuple contains
        the start and end times of a seizure period
        (the exact time stamps). \n
        Assumed to be the output of `utils.preprocess.read_seizure_times`.
    selected_channels : List[str]
        If given, only data from these channels will be kept.
    seed : int, optional
        The random seed used for taking negative samples. \n
        Default is 2025.
    window_width : float
        The length of the windows in seconds.
    overlap : float
        The overlap between consecutive windows in seconds,
        used for taking windows in preictal stage.
    preictal_interval : float, optional
        The time interval before onset time from which to
        take the preictal samples (in seconds) \n
        Default is 600 s (10 minutes)
    preictal_width : float, optioanl
        If given, the window width of preictal stage
        will be taken as this value. Otherwise,
        it is the same as the other windows.
    random_samples : int, optional, default=500
        The number of random segments to sample from
        the time series outside preictal periods.

    Returns
    -------
    samples : numpy.ndarray
        Array of shape
        [n_samples, n_channels, window_width]
        containing the EEG samples
    dists_to_seizure : np.ndarray
        1 dimensional array of same length as n_samples,
        each value (integer) represents the distance
        to the closest seizure period
        of the sample. \n
        Positive distance: preictal,
        negative distance: postictal,
        nan: no seizure in the file.
    """
    random.seed(seed)

    samples = []
    dists_to_seizure = []

    sfreq = raw_data.info['sfreq']

    # Extract preictal windows
    if preictal_width is None:
        preictal_width = window_width
    
    if selected_channels:
        raw_data_selected = raw_data.pick(selected_channels)
    else:
        raw_data_selected = raw_data

    for i, (seizure_start, seizure_end) in enumerate(seizure_times):
        if i == 0:
            prev_end = 0

        # slide windows backwards
        sample_start = seizure_start - int(preictal_width * sfreq)
        while sample_start > seizure_start - int(preictal_interval * sfreq):
            if sample_start < prev_end:
                warnings.warn(
                    "Not enough temporal length in the interictal stage. "
                    f"Requested: {preictal_interval}s, available: "
                    f"{(seizure_start - prev_end) / sfreq:.4f}s"
                )
                break

            sample_end = sample_start + int(preictal_width * sfreq)
            sample = raw_data_selected.get_data(
                start=sample_start, stop=sample_end)
            samples.append(sample)
            dists_to_seizure.append(seizure_start-sample_end)
            sample_start -= int((preictal_width - overlap) * sfreq)
        
        prev_end = seizure_end
    
    # Add random intervals
    total_samples = len(raw_data_selected)
    for _ in range(random_samples):
        sample_start = random.randint(0, total_samples - int(window_width * sfreq))
        sample_end = sample_start + int(preictal_width * sfreq)

        overlap = check_seizure_overlap(
            seizure_times, interval=(sample_start, sample_end))

        if not overlap:
            sample = raw_data_selected.get_data(
                start=sample_start, stop=sample_end)
            samples.append(sample)
            if len(seizure_times) == 0: # no seizure in the file
                dist = np.inf
            else:
                dist = distance_to_closest_seizure(
                    (sample_start, sample_end), seizure_times)
            dists_to_seizure.append(dist)
    
    return np.array(samples), np.array(dists_to_seizure)
