import numpy as np
from typing import Union, List, Tuple, Optional
from mne.io.edf.edf import RawEDF
import pickle
import argparse


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
