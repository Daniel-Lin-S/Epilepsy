from scipy.stats import pearsonr, ks_2samp, mannwhitneyu, ttest_ind
from collections import defaultdict
from mne.io.edf.edf import RawEDF
import re
from typing import Dict, Tuple, Optional, Union, List, Set
import numpy as np
import os

from utils.preprocess import read_seizure_times, load_raw_data
from utils.tools import check_time_range, slice_raw


def channel_corr(
        raw: RawEDF, prefix_pattern=r"^[a-zA-Z]+",
        verbose: bool=True
    ) -> Dict[Tuple[str, str], float]:
    """
    Compute Pearson correlations between EEG channels with the same prefix.
    Channels that have a numeric suffix (e.g., 'Fp1', 'F1') are included in 
    the correlation calculation, while channels without a numeric suffix 
    (e.g., 'Pz') are ignored for correlation.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data object.
    prefix_pattern : str, optional
        A regular expression pattern to extract channel prefixes. \n
        Default is r"^[a-zA-Z]+" to match the first alphabetic characters.
    verbose: bool
        If true, the correlations will be printed
    
    Return
    ------
    Dict[Tuple[str, str], float]
        A dictionary containing pairs of channel names as
        keys and Pearson correlations as values.
    """

    channel_names = raw.info['ch_names']

    # Group channels by the starting letter(s)
    grouped_channels = defaultdict(list)
    for ch in channel_names:
        # Match the prefix
        match = re.match(prefix_pattern, ch)
        if match:
            prefix = match.group(0)
            # This ensures the channel ends with a number 
            if ch[-1].isdigit():  
                grouped_channels[prefix].append(ch)

    # Compute the Pearson correlation within each group
    correlations = {}
    for prefix, channels in grouped_channels.items():
        data = raw.get_data(picks=channels)
        n_channels = len(channels)
        
        # all pairs of channels
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                ch_i_data = data[i]
                ch_j_data = data[j]
                corr, _ = pearsonr(ch_i_data, ch_j_data)
                correlations[(channels[i], channels[j])] = corr

    if verbose and len(correlations) > 0:
        print('------ Pearson Correlations -------')
        for pair, corr in correlations.items():
            print(f"Channel {pair[0]} and Channel {pair[1]}: {corr}")

    return correlations


def seizure_channel_diff(
    raw: RawEDF, seizure_time: float,
    channel_idxs: Optional[List[Union[np.int_, int]]]=None,
    width: float=30.0,
    test: str='ks'
) -> Dict[str, float]:
    """
    Perform a hypothesis test on whether
    the distributions of channels
    before and after a given seizure time
    are different.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data object.
    channel_idxs : list of int
        the channel indices (or names) to be picked.
    seizure_time : float
        the starting time of seizure
        (in seconds)
    width : float, optional
        the width of window before and after
        seizure_time on which the distribution will
        be compared. \n
        Default is 30.0s
    test : str, optional
        The test used to measure distribution
        difference. One of the following: 
        - 'ks' : Kolmogorov-Smirnov (KS)
        - 'u' : Mann-Whitney U Test (Wilcoxon rank-sum test)
        - 't' : T-test. (assumes normality)

    Return
    ------
    Dict[str, float]
        keys: channel names, 
        values: the p-values
        of the test
    """
    check_time_range(raw, seizure_time, in_seconds=True)

    sf = raw.info['sfreq']
    start_stamp = int((seizure_time - width) * sf)
    seizure_stamp = int(seizure_time * sf)
    end_stamp = int((seizure_time + width) * sf)

    data_before, _ = raw[:, start_stamp:seizure_stamp]
    data_seizure, _ = raw[:, seizure_stamp:end_stamp]

    # store test scores
    results = {}

    channel_names = raw.info['ch_names']

    if channel_idxs is None:
        channel_idxs = np.arange(raw.info['nchan'])

    for idx in channel_idxs:
        ch_name = channel_names[idx]
        # Perform KS test between data before and after the seizure
        if test == 'ks':
            _, p_value = ks_2samp(data_before[idx, :], data_seizure[idx, :])
        elif test == 'u':
            _, p_value = mannwhitneyu(data_before[idx, :], data_seizure[idx, :])
        elif test == 't':
            _, p_value = ttest_ind(data_before[idx, :], data_seizure[idx, :])

        results[ch_name] = p_value

    return results


def find_significant_channels(
        folder_path: str,
        width: float = 30.,
        test: str = 'ks',
        p_threshold: float = 0.05,
        ratio_threshold: float=0.9,
        corr_threshold: float=0.5
    ) -> Tuple[Dict[str, float], Set[str]]:
    """
    Analyze which channels are significantly different before
    and after seizure starting time (i.e. preictal vs ictal stage)
    for most seizures across all patients in the folder.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing the raw EDF files and seizure files.
    
    width : float, optional, default=30.
        Time window width (in seconds or samples)
        for seizure detection. \n

    test : str, optional, default='ks'
        The test used to measure distribution
        difference. One of the following: 
        - 'ks' : Kolmogorov-Smirnov (KS)
        - 'u' : Mann-Whitney U Test (Wilcoxon rank-sum test)
        - 't' : T-test. (assumes normality)
    
    p_threshold : float, optional, default=0.05
        The threshold below which the difference in
        distribution is treated as significant. \n
        Must be in (0., 1.)
    
    ratio_threshold : float, optional, default=0.9
        The threshold above which the proportion
        of seizures in which the channel was significant
        will be returned. \n
        Set to 0. to return all channels.

    corr_threshold : float, optional, default=0.5
        The threshold above which the two channels
        will be treated as the same
        and only one channel will be kept.
    
    Returns
    -------
    Dict[str, int]
        information of channels which are significant
        for more than half of the seizures. \n
        Keys: channel names;
        Values: proportion of seizures in which the channel was significant
    Set[str]
        the names of channels remaining after correlation filtering
        and distribution difference filtering.
    """
    channel_significance_count = {}

    patient_files = [f for f in os.listdir(folder_path) if f.endswith('.edf')]

    seizure_count = 0
    
    correlations : Dict[Tuple[str, str], list] = {}

    for raw_file in patient_files:
        patient_id = raw_file.split('.')[0]
        raw = load_raw_data(folder_path, patient_id)

        seizure_file = os.path.join(folder_path, f"{patient_id}.edf.seizures")
        seizure_times = read_seizure_times(seizure_file)  

        # Check each seizure time for significance
        for start_stamp, _ in seizure_times:
            seizure_count += 1
            seizure_time = start_stamp / raw.info['sfreq']
            significance_scores = seizure_channel_diff(
                raw, seizure_time, width=width, test=test)

            # Count significant channels
            for channel_name, p_value in significance_scores.items():
                if p_value < p_threshold:
                    if channel_name in channel_significance_count:
                        channel_significance_count[channel_name] += 1
                    else:
                        channel_significance_count[channel_name] = 1
            
            # compute correlations
            sliced_raw = slice_raw(
                raw, interval=(seizure_time - width, seizure_time + width))
            seizure_corr = channel_corr(
                sliced_raw, verbose=False)

            # Store the correlation values for this seizure
            for (ch1, ch2), corr_value in seizure_corr.items():
                if (ch1, ch2) not in correlations:
                    correlations[(ch1, ch2)] = []
                correlations[(ch1, ch2)].append(corr_value)

    # Find channels significant for most seizures (more than half)
    significant_scores_all = {
        channel: count / seizure_count
        for channel, count in channel_significance_count.items()
        if count / seizure_count > ratio_threshold}
    
    # correlation filtering
    kept_channels = set(significant_scores_all.keys())

    for ch1, ch2 in correlations.keys():
        avg_corr = np.mean(correlations[(ch1, ch2)])

        if avg_corr >= corr_threshold:
            # remove channel
            if ch1 in kept_channels and ch2 in kept_channels:
                kept_channels.remove(ch2)

    significant_score_dict = {}
    for channel in kept_channels:
        significant_score_dict[channel] = significant_scores_all[channel]

    return significant_score_dict, kept_channels
