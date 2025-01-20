import os
import numpy as np
import h5py
import mne
from mne.io.edf.edf import RawEDF
import random
from typing import List, Tuple, Optional, Union

from utils.preprocess import read_seizure_times
from utils.tools import check_labels

def _extract_samples(
        raw_data: RawEDF,
        seizure_times: List[Tuple[int, int]],
        sample_time: float=10.0, preictal_time: float=10.0,
        n_negative: int=5,
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
    
    sample_time : float, optional
        The length of each sample in samples. \n
        Default is 60.0 seconds.
    
    preictal_time : float, optional
        The time in seconds before the seizure to
        extract as positive samples. \n
        Default is 10.0 seconds.

    n_negative : int, optional
        Number of negative samples to draw. \n
        Default is 5.
    
    safe_gap : float, optional
        The minimum distance (in seconds) between negative samples
        and seizure times to avoid overlap.
        Default is 300.0 seconds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays:
        - Positive samples (pre-ictal segments)
        - Negative samples (random segments far from seizures)
    """
    positive_samples = []
    negative_samples = []

    sfreq = raw_data.info['sfreq']

    sample_length = int(sample_time * sfreq)

    # Extract positive samples
    for start_idx, _ in seizure_times:
        pre_ictal_start = int(start_idx) - int(preictal_time * sfreq)
        pre_ictal_end = pre_ictal_start + sample_length

        if pre_ictal_start >= 0:
            positive_samples.append(
                raw_data.get_data(start=pre_ictal_start, stop=pre_ictal_end))
        else:
            raise Exception(
                'Not enough length before ictal stage. '
                f'required length: ({preictal_time} + {sample_length}) s'
                f', actual length {start_idx / sfreq} s. \n'
                'Please reduce preictal_time or sample_length.'
            )

    # Extract negative samples (random samples away from seizures)
    attempts = 0
    distance = int(safe_gap * sfreq)
    while len(negative_samples) < n_negative and attempts < 100:
        random_start = random.randint(0, len(raw_data.times) - sample_length)
        random_end = random_start + sample_length

        overlap = False
        for seizure_start, seizure_end in seizure_times:
            if (random_start < (seizure_end + distance)
                and random_end > (seizure_start - distance)):
                overlap = True
                break
        
        if not overlap:
            negative_samples.append(
                raw_data.get_data(start=random_start, stop=random_end))
        attempts += 1

    if len(negative_samples) < n_negative:
        raise Exception(
            f"Only {len(negative_samples)} negative samples "
            "were extracted due to overlap issues. "
            "Try reduce sample_time or safe_gap."
        )

    return np.array(positive_samples), np.array(negative_samples)


def build_classification_samples(
        folder_path: str, output_file: Optional[str]=None,
        **kwargs) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
    """    
    Obtain positive and negative samples from EEG signals
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

    kwargs : Any
        Passed to extract_samples. e.g. sample_time, preicetal_time,
        n_negative, safe_gap. (see docstring of extract_samples)

    Return
    -------
    samples, labels : numpy.ndarray
        of shape [num_samples, sample_length, num_channels]
        and [num_samples,] respectively. \n
        The labels are 0 and 1s.
        0 - no seizure
        1 - seizure
    """
    samples = []  # store EEG signals
    labels = []   # store 0 1 labels
    flag = True

    # Go through all the files in the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".edf"):
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

                # Extract positive and negative samples
                pos_samples, neg_samples = _extract_samples(
                    raw_data, seizure_times, **kwargs)

                if pos_samples.shape[0] > 0:
                    samples.append(pos_samples)
                    labels.append(np.ones(pos_samples.shape[0]))
                if neg_samples.shape[0] > 0:
                    samples.append(neg_samples)
                    labels.append(np.zeros(neg_samples.shape[0]))

    if flag:
        raise FileNotFoundError(
            'Found no edf file in the provided folder'
            f' {folder_path}.'
        )

    # Convert lists to numpy arrays and store them
    samples = np.concatenate(samples, axis=0)
    labels = np.concatenate(labels, axis=0)
    check_labels(labels)

    if output_file:
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('x', data=samples)
            f.create_dataset('y', data=labels)
    else:
        return samples, labels


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
    samples, labels : numpy.ndarray
        of shape [num_samples, num_channels, sample_length]
        and [num_samples,] respectively. \n
        The labels are 0 and 1s.
        0 - no seizure
        1 - seizure
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Cannot find file {file_path}')

    with h5py.File(file_path, 'r') as f:
        return f['x'][:], f['y'][:]
