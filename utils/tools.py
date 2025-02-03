import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from mne.io.edf.edf import RawEDF
import pickle
import argparse
import json
import pandas as pd


def check_labels(y: Union[np.ndarray, List[int]]):
    """Check whether y has both positive (1) and negative (0) samples"""
    if isinstance(y, list):
        y = np.array(y)
    if y.ndim != 1:
        raise ValueError(
            'y should be a 1 dimensional array'
        )
    labels = np.unique(y)
    if 0 not in labels or 1 not in labels:
        raise Exception(
            'y must have both positive (1) and negative (0) labels'
            f': 0 and 1. Labels: {labels}'
        )

def slice_raw(
        raw: RawEDF,
        interval: Tuple[float],
        channel_idxs: Optional[List[Union[int, np.int_, str]]]=None
    ) -> RawEDF:
    """
    Slice a raw EDF file on given time-interval and channels
    without losing meta-data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data object.
    channel_idxs : list of int or str
        the channel indices (or names) to be picked.
    interval : tuple of floats
        two floats indicating starting time and ending time
        (in seconds)
    """
    if channel_idxs:
        selected_channels = channel_ids_to_names(
            channel_idxs, raw.info['ch_names'])

        raw_select = raw.copy().pick(selected_channels)
    else:
        raw_select = raw.copy()
    raw_select = raw_select.crop(tmin=interval[0], tmax=interval[1])

    return raw_select


def channel_ids_to_names(
    channel_idxs: Union[np.int_, int, str],
    channel_names: List[str]
):
    if isinstance(channel_idxs[0], str):
        return channel_idxs
    elif isinstance(channel_idxs[0], int) or isinstance(
        channel_idxs[0], np.int_):  # using indices
        channel_names = np.array(channel_names)
        return channel_names[channel_idxs]
    else:
        raise TypeError(
            'Unsupported type of list items of channel_idxs,'
            ' Please use strings or integers. '
        )

def check_time_range(
        raw: RawEDF,
        time: Union[int, float],
        in_seconds: bool=False
    ) -> None:
    """
    Check whether the given time is in the range of raw file.

    Parameters
    ----------
    raw : RawEDF
        The raw EDF object.
    time : int | float
        The time to check. It can be either in seconds
        or samples (stamps), depending on `in_seconds`.
    in_seconds : bool, optional
        If True, `time` is considered in seconds. \n
        If False, `time` is considered in samples (stamps). \n
        Default is False (samples).

    Raise
    -----
    ValueError
        If time is not in the range of raw.
    """
    
    n_samples = raw.n_times

    if in_seconds:
        # Convert the time to samples
        time_in_samples = time * raw.info['sfreq']
    else:
        time_in_samples = time

    if time_in_samples < 0 or time_in_samples >= n_samples:
        raise ValueError(
            f"Time {time} is out of the raw data's range."
        )


def save_set_to_file(data_set: set, file_path: str) -> None:
    """
    Save a set to a file using pickle.

    Parameters
    ----------
    data_set : set
        The set to be saved.
    file_path : str 
        The path to the file where the set will be stored.
        The postfix should be 'pkl'.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data_set, f)

    print(f"Set successfully saved to {file_path}")


def load_set_from_file(
        file_path: str, as_list: bool=False
    ) -> Union[set, list]:
    """
    Load a set from a file using pickle.

    Parameters
    ----------
    file_path : str
        The path to the file where the set is stored.
    as_list : bool, optional
        Whether to return the data as a list. \n
        Default is False (returns a set).

    Returns
    -------
    set or list
        The loaded data, either as a set or a list,
        based on the `as_list` parameter.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if as_list:
        return list(data)
    else:
        return data


def print_args(args: argparse.Namespace,
               description: Optional[str]=None,
               return_str: bool=True) -> Union[None, str]:
    """
    Print all the arguments stored in the
    given argparse.Namespace object.
    
    Parameter
    ----------
    args : argparse.Namespace
        The namespace object containing parsed arguments.
    description : str, optional
        A description inserted in front of the arguments.
    return_str : bool
        if True, the arguments will be printed. \n
        Otherwise, returned as a str.
    
    Return
    ------
    str
        the formatted arguments, if return_str=False.
    """
    args_list = [f"{arg}={value}" for arg, value in vars(args).items()]

    if description:
        args_str = f"{description}: {', '.join(args_list)}"
    else:
        args_str = ', '.join(args_list)
    
    if return_str:
        return args_str
    else:
        print(args_str)


def flatten_dict(d: dict) -> dict:
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))

    return dict(items)


def print_dict(
    d: dict, return_str: bool=True
) -> Union[str, None]:
    """
    Turn a dictionary into a string.
    The dictionary can be nested.

    d : dict
        The dictionary being processed
    return_str : bool
        If true, return the formatted dict.
        Otherwise, printed.
    """
    flattened_dict = flatten_dict(d)
    dict_str = ', '.join(
        [f"{k} : {v}" for k, v in flattened_dict.items()])
    if return_str:
        return dict_str
    else:
        print(dict_str)


def read_config(config_file: str) -> Dict:
    """
    Read the JSON configuration file and
    return the parameters as a dictionary.

    Parameters:
    ----------
    config_file : str
        Path to the configuration JSON file.

    Returns:
    -------
    dict
        Dictionary containing model parameters for each classifier.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def confusion_matrix_to_str(cm: np.ndarray, class_labels: np.ndarray):
    """
    Format a confusion matrix to str
    
    Parameters
    ----------
    cm : np.ndarray
        The confusion matrix
    class_labels : np.ndarray
        The corresponding labels
    """
    if class_labels.ndim != 1:
        raise TypeError('class_labels must be a 1-dimensional array')
    elif class_labels.shape[0] != cm.shape[0]:
        raise TypeError(
            'The length of class_labels must be consistent with cm'
        )

    cm_str = "Confusion Matrix:\n"
    cm_str += f"              {', '.join(map(str, class_labels))}\n"
    
    for i, row in enumerate(cm):
        cm_str += f"True Label {class_labels[i]}: {', '.join(map(str, row))}\n"
    return cm_str


def save_to_csv(data: dict, file_path: str):
    """
    Append a single data to csv file. \n
    Create the file if not exist.
    """
    df = pd.DataFrame([data])

    header = not pd.io.common.file_exists(file_path)

    df.to_csv(file_path, mode='a', header=header, index=False)


def check_seizure_overlap(
        seizure_times: List[Tuple[int, int]],
        interval: Tuple[int, int],
        distance: int=0
    ) -> bool:
    """
    Check whether the given time interval overlaps with any
    of the seizure time.

    Parameters
    ----------
    seizure_times : list of tuple of int
        A list of tuples where each tuple represents
        the start and end time stamps of a seizure period.
    interval : tuple of int
        A tuple representing the start and end time stamps
        of a given interval to check for overlap,
        where the first element is the start time
        and the second element is the end time.
    distance : int, optional
        A safe distance from the seizure interval.
        Default is 0.

    Returns
    -------
    bool
        Returns `True` if the interval overlaps with any of the seizure times, `False` otherwise.
    """
    interval_start, interval_end = interval

    for seizure_start, seizure_end in seizure_times:
        # Check if the interval overlaps with the seizure period
        if interval_end > (seizure_start - distance) and (
            interval_start < (seizure_end + distance)):
            return True

    return False


def distance_to_closest_seizure(
        interval: Tuple[int, int],
        seizure_times: List[Tuple[int, int]]
    ) -> int:
    """
    Find the distance from a given interval to the closest seizure time.

    Parameters
    ----------
    interval : tuple of int
        A tuple representing the start and end time stamps
        of a given interval to check for overlap,
        where the first element is the start time
        and the second element is the end time.
    seizure_times : list of tuple of int
        A list of tuples where each tuple represents
        the start and end time stamps of a seizure period.

    Returns
    -------
    dist_to_seizure : float
        The distance to the closest seizure start or end time. \n
        Positive if the interval is preictal, 
        negative if it is postictal,
        0 if it is right next to an ictal stage.
    
    Raise
    -----
    Exception
        if the given interval is 
    """
    if check_seizure_overlap(seizure_times, interval):
        raise Exception(
            'The interval is overlapping with a seizure (ictal) period, '
            ' cannot calculate distance')
    
    if len(seizure_times) == 0:
        raise ValueError(
            'No seizure given in seizure_times'
        )

    interval_start, interval_end = interval
    closest_distance = float('inf')

    for (seizure_start, seizure_end) in seizure_times:
        dist_pre = seizure_start - interval_end
        dist_post = interval_start - seizure_end

        if dist_pre >= 0 and dist_pre < closest_distance:
            # interval before seizure
            closest_distance = dist_pre
            postictal = False
        elif dist_post >= 0 and dist_post < closest_distance:
            # interval after seizure
            closest_distance = dist_post
            postictal = True
    
    if postictal:
        return - closest_distance
    else:
        return closest_distance
