from mne.io import read_raw_edf
from mne.io.edf.edf import RawEDF
import os
import re
from typing import Optional, Dict, Union, List, Tuple


def load_raw_data(folder_path: str, patient_id: str,
                  preload: bool=False) -> RawEDF:
    """
    Read raw data of a patient's EEG graph stored in
    an edf file.

    Parameters
    ----------
    folder_path : str
        the folder in which edf data files are stored.
    patient_id : str
        the unique patient id, e.g. DA0441I2
    preload : bool
        If True, the entire data will be loaded
        to RAM for faster process. Otherwise,
        the data is stored in disk.
    
    Return
    ------
    raw : mne.io.edf.edf.RawEDF
        the extracted data.
        Use raw[c, t] to extract signal at time stamp
        t of channel c.
        Or slice to get numpy.ndarray of a section of
        of signal(s).
        raw.info has the meta informations including
        - 'sfreq' (float) : sampling frequency in Hz
        - 'nchan' (int) : number of channels
        - 'ch_names' (list) :  channel names
        - 'meas_date' (datetime) : the starting time of the recording
        - 'highpass' (float) : signals with frequencies lower than
        this are filtered out (in Hz)
        - 'lowpass' (float) : signals with frequencies higher than
        this are filtered out (in Hz)
    
    Raise
    -----
    FileNotFoundError
        if the patient's edf file does not exist.
    """
    file_path = os.path.join(folder_path, f'{patient_id}.edf')

    # check existence of edf file
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Corresponding EDF file not found: {file_path}")

    raw = read_raw_edf(file_path, preload=preload, verbose=False)

    return raw


def load_raw_data_from_folder(
        folder_path: str, preload: bool=False) -> List[RawEDF]:
    """
    Load all EEG data (.edf files) in the folder
    into a list of RawEDF objects.

    Parameters
    ----------
    folder_path : str
        the folder in which edf data files are stored.
    preload : bool
        If True, the entire data will be loaded
        to RAM for faster process. Otherwise,
        the data is stored in disk. \n
        Default is False

    Return
    ------
    raw_data_list : List[mne.io.edf.edf.RawEDF]
        each item is a EDF dataset
    """
    raw_data_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.edf'):
                eeg_file = os.path.join(root, file)
                raw_data = read_raw_edf(eeg_file, preload=preload)
                raw_data_list.append(raw_data)
    return raw_data_list


def get_seizure_times(
        raw : RawEDF,
        output_path: Optional[str]=None,
        in_seconds: bool=False,
        verbose: bool=True,
    ) -> Dict[str, Dict[Union[int, float], Union[int, float]]]:
    """
    Extract the start and end time stamps
    of seizure (ictal stage)
    and return a dictionary of dictionary of the form
    {"start" : start_time, "end" : end_time}. \n
    The values can be optionally stored in a file,
    otherwise printed in console.

    Parameters
    ----------
    raw : RawEDF
        the extracted data
    output_path : str, optional
        file path to the output file,
        e.g. DA0441I2.edf.seizures,
        to which the seizure time will be saved.
    in_seconds : bool
        if True, the time stamps will be in seconds
    verbose : bool
        if True, the seizure times will be printed
        if output_path is not given
    
    Return
    ------
    onset_dict : dict
        a dictionary with keys being onset id (str)
        and values being dictionaries with
        pairs of onset start and end time stamps
        indexed by "start" and "end".
    """

    fs = raw.info['sfreq']  # Sampling frequency

    # Standardize annotation descriptions to lowercase
    events = {ann['description'].lower(): ann['onset']
              for ann in raw.annotations}
    
    # Find onset pairs
    onset_dict = _get_onsets(fs, events, in_seconds)

    # write into file or print
    if len(onset_dict.keys()) == 0 and verbose:
        print('No seizure found')
    else:  # seizure exists
        if output_path:
            with open(output_path, 'w') as f:
                for value in onset_dict.values():
                    start_time = value['start']
                    end_time = value['end']
                    f.write(f"{start_time} {end_time}\n")
        elif verbose:
            for value in onset_dict.values():
                start_time = value['start']
                end_time = value['end']
                if in_seconds:
                    print(f"start {start_time}s, end {end_time}s\n")
                else:
                    print(f"start {start_time}, end {end_time}\n")

    return onset_dict


def _get_onsets(
        fs: float, events: Dict[str, float],
        in_seconds : bool
    ) -> Dict[str, Dict[int, int]]:
    """
    Find pairs of onsets from given event marks.
    i.e. 'onset[id]' and 'onset[id]end' pairs

    Parameters
    ----------
    fs : int
        the sampling frequency (in Hz)

    events : dict
        keys are the event type (text description),
        and values are the corresponding
        time stamps of the event.

    in_seconds : bool
        if True, the time stamps will be in seconds

    Return
    ------
    onset_dict : dict
        a dictionary with keys being onset id
        and values being dictionaries with
        pairs of onset start and end time stamps
        indexed by "start" and "end".
    """
    onset_dict = {}

    for desc, onset_time in events.items():
        # Match onset
        if "onset" in desc and "end" not in desc:
            onset_id = ''.join(filter(str.isdigit, desc))
            if onset_id:
                if in_seconds:
                    onset_dict[onset_id] = {
                        "start": onset_time, "end": None}
                else:
                    onset_dict[onset_id] = {
                        "start": int(onset_time * fs), "end": None}
        # Match end
        elif "onset" in desc and "end" in desc:
            onset_id = ''.join(filter(str.isdigit, desc))
            if onset_id and onset_id in onset_dict:
                if in_seconds:
                    onset_dict[onset_id]["end"] = onset_time
                else:
                    onset_dict[onset_id]["end"] = int(onset_time * fs)

    # remove invalid onsets
    onset_dict = {key: value for key, value in onset_dict.items()
                  if value["end"] is not None}

    n_seizures = len(onset_dict.keys())

    if n_seizures > 1:
        _onset_overlap_check(onset_dict)

    return onset_dict

def _onset_overlap_check(onset_dict: Dict[str, Dict[int, int]]) -> None:
    """
    check whether intervals defined in
    onset_dict are non-overlapping.

    Raise
    -----
    AssertionError
        if overlapping intervals found.
    """
    intervals = [(value["start"], value["end"])
                 for value in onset_dict.values()]
    
    intervals.sort(key=lambda x: x[0])

    # Check for overlaps
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            raise AssertionError(
                "Overlap detected: "
                f"Interval {i-1} ({intervals[i-1][0]}, {intervals[i-1][1]}) "
                f"overlaps with Interval {i} ({intervals[i][0]}, {intervals[i][1]})"
            )

def read_seizure_times(seizure_file: str) -> List[Tuple[int, int]]:
    """
    Read seizure time stamps from a file,
    supposed to be `{id}.edf.seizures`.

    Parameter
    ---------
    seizure_file : str
        path to the file storing seizure times

    Return
    ------
    seizure_times : List[Tuple[int, int]]
        each item contains a pair of
        integers representing seizure
        starting time and ending time
    """
    seizure_times = []
    with open(seizure_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Start and end time stamps
            seizure_times.append((int(parts[0]), int(parts[1])))
    return seizure_times

def read_sampling_rate(file_path: str) -> int:
    """
    Extracts the sampling rate (in Hz)
    as an integer from the 'summary.txt' file.
    
    :param file_path: str - The path to the summary.txt file.
    :return: int - The extracted sampling rate, or None if not found.
    """
    try:
        # Open the file
        with open(file_path, 'r') as file:
            for line in file:
                # Search for the pattern "Data Sampling Rate: <number> Hz"
                match = re.search(r"Data Sampling Rate:\s*(\d+)\s*Hz", line)
                if match:
                    return int(match.group(1))
        # Return None if the line is not found
        print(f"Sampling rate not found in the file: {file_path}")
        return None
    except FileNotFoundError:
        print(
            f"The file '{file_path}' was not found. "
            "Please also check that you have run annotation.py"
            " before running this function."
        )
        return None
