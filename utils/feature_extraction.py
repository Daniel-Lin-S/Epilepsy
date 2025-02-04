import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Union
import neurokit2 as nk
import os
from tqdm import tqdm
from multiprocessing import Pool


def extract_features_timefreq(
    x: np.ndarray, sfreq: int,
    timefreq_method: str, n_jobs: int
) -> np.ndarray:
    """
    Extract time-frequency graph summary statistics, processed in parallel. \n
    Summary statistics: 
    * **mav** Mean absolute value
    * **std** Standard deviation, variation around the mean
    * **rms** Root mean squares, indicating power of the signal
    * **skewness** Asymmetry of the data distribution around its mean
    * **kurtosis** tells how much data are in the tails
    compared to around mean
    * **mobility** first-order variation:
    relative variation in the rate of change. \n
    \t Mobility = sqrt( Var(Delta x) / Var(x) ) \n
    \t where Delta x is the first-order difference of x.
    * **complexity** second-order variation \n
    \t Complexity = Mobility(Delta x) / Mobility(x)

    Parameters
    ----------
    x : numpy.ndarray
        the input signal of shape (samples, channels, length)
    sfreq : int
        the sampling frequency
    timefreq_method : str
        the method used to turn x into time-frequency graph.
        one of stft, cwt, wvd, pwvd
    n_jobs : int
        number of parallel processors used to convert the
        samples.

    Return
    ------
    features : np.ndarray
        of shape (samples, n_features)
    """
    n_samples, n_channels, _ = x.shape
    # store features from all channels
    all_features = []

    if n_jobs == -1:
        n_jobs = os.cpu_count()

    args = [(i, x, n_channels, sfreq, timefreq_method)
            for i in range(n_samples)]

    with Pool(processes=n_jobs) as pool:
        with tqdm(total=n_samples, desc="Processing samples") as pbar:
            def update_progress(_):
                pbar.update(1)
            
            # Use apply_async to handle the tasks asynchronously
            results = []
            for arg in args:
                result = pool.apply_async(
                    _process_sample, arg, callback=update_progress)
                results.append(result)
            
            # Wait for all processes to finish and collect the results
            all_features = [result.get() for result in results]

    return np.array(all_features)

def _process_sample(
        i: int, x: np.ndarray, n_channels: int,
        sfreq: int, timefreq_method: str
    ) -> np.ndarray:
    """
    Process a single sample (patient) to extract its features.
    """
    sample_features = []
    
    for c in range(n_channels):
        # Perform time-frequency decomposition
        freqs, _, x_timefreq = nk.signal_timefrequency(
            x[i, c, :], sampling_rate=sfreq,
            method=timefreq_method,
            min_frequency=1., max_frequency=80.,
            show=False
        )
        
        # Extract summary statistics
        features = timefreq_summary_features(x_timefreq, freqs, labels=False)
        sample_features.append(features.flatten())

    return np.concatenate(sample_features)


def timefreq_summary_features(
        x_timefreq: np.ndarray,
        freqs: np.ndarray,
        labels: bool=False
    ) -> Union[Dict[str, Dict[str, float]], np.ndarray]:
    """
    Extracts summary statistics from each frequency band
    of the timefreq transformed signal. \n
    Bands:
    * **Gamma** (30-80 Hz)
    * **Beta** (13-30 Hz)
    * **Alpha** (8-13 Hz)
    * **Theta** (4-8 Hz)
    * **Delta** (1-4 Hz) \n
    Summary statistics: 
    * **mav** Mean absolute value
    * **std** Standard deviation, variation around the mean
    * **rms** Root mean squares, indicating power of the signal
    * **skewness** Asymmetry of the data distribution around its mean
    * **kurtosis** tells how much data are in the tails
    compared to around mean
    * **mobility** first-order variation:
    relative variation in the rate of change. \n
    \t Mobility = sqrt( Var(Delta x) / Var(x) ) \n
    \t where Delta x is the first-order difference of x.
    * **complexity** second-order variation \n
    \t Complexity = Mobility(Delta x) / Mobility(x)
    
    Parameters
    ----------
    x_timefreq : np.ndarray
        The timefreq transformed signal with shape (n_freqs, length),
        where n_freqs is the number of frequency channels.
    freqs : np.ndarray
        The array of frequencies (in Hz) corresponding to
        each of the frequency channels.
    labels : bool
        If true, keep the frequency band and summary
        statistic labels. Otherwise,
        simply return an array of features.
    
    Returns
    -------
    Union[Dict[str, Dict[str, float]], np.ndarray]
        If labels = True, returns a dictionary
        where the keys are the frequency bands,
        and the values are dictionaries of summary features.
        Each dictionary contains the calculated features:
        mean absolute value (MAV), standard deviation (STD),
        skewness, kurtosis, RMS, mobility,
        signal complexity, and MAV ratio (for adjacent bands). \n
        If labels = False, returns a 1 dimensional np.ndarray
        of all the features (float)
    """
    
    # Store features for each frequency band
    features = {}

    segmented_timefreq = _segment_timefreq_bands(x_timefreq, freqs)

    band_names = list(segmented_timefreq.keys())
    
    # Iterate through each frequency band
    for i, band in enumerate(band_names):
        band_data = segmented_timefreq[band]
        
        # Compute the summary features
        mav = np.mean(np.abs(band_data), axis=1)  # Mean absolute value
        std = np.std(band_data, axis=1)  # Standard deviation
        skewness = skew(band_data, axis=1, nan_policy='omit')
        kurt = kurtosis(band_data, axis=1, nan_policy='omit')
        rms = np.sqrt(np.mean(np.square(band_data), axis=1))  # Root mean square

        # compute complexity and mobility
        complexities = []
        mobilities = []
        for c in range(band_data.shape[0]):
            complexity, act_dict = nk.complexity_hjorth(band_data[c, :])
            complexities.append(complexity)
            mobilities.append(act_dict["Mobility"])

        # Compute MAV ratio with previous band
        if i > 0:
            prev_band = band_names[i-1]
            prev_band_mav = np.mean(np.abs(segmented_timefreq[prev_band]))
            mav_ratio = np.mean(mav) / prev_band_mav
        else:
            mav_ratio = np.nan  # No previous band for first frequency band

        features[band] = {
            "mav": np.mean(mav),
            "std": np.mean(std),
            "skewness": np.mean(skewness),
            "kurtosis": np.mean(kurt),
            "rms": np.mean(rms),
            "mobility": np.mean(mobilities),
            "complexity": np.mean(complexities),
            "mav_ratio": mav_ratio
        }
    
    if labels:
        return features
    else:
        return _nest_dict_to_numpy_array(features)


def _nest_dict_to_numpy_array(
        features_dict: Dict[str, Dict[str, float]]
    ) -> np.ndarray:
    """
    Converts the nested dictionary (output of `timefreq_summary_features`)
    of features into a 1D numpy array.

    Parameters
    ----------
    features_dict : Dict[str, Dict[str, float]]
        A nested dictionary.

    Returns
    -------
    np.ndarray
        A 1D numpy array containing all the feature values,
        flattened and concatenated.
    """
    all_features = []

    for _, inner_dict in features_dict.items():
        for key, value in inner_dict.items():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                print(f'{key} shape: {value.shape}')
            if not np.isnan(value):
                all_features.append(value)

    return np.array(all_features)


def _segment_timefreq_bands(
        x_timefreq: np.ndarray, freqs: np.ndarray
    ) -> Dict[str, np.ndarray]:
    """
    Segments the timefreq transformed signal into different frequency bands.
    * **Gamma** (30-80 Hz)
    * **Beta** (13-30 Hz)
    * **Alpha** (8-13 Hz)
    * **Theta** (4-8 Hz)
    * **Delta** (1-4 Hz)
    
    Parameters
    ----------
    x_timefreq : np.ndarray
        The timefreq transformed signal with shape (n_freqs, length),
        where n_freqs is the number of frequency channels.
    
    freqs : np.ndarray
        The array of frequencies corresponding to each of the 250 frequency channels.
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary where the keys are the frequency bands
        and the values are the segmented timefreq signals. \n
        Each value is an ndarray of shape (num_freqs, 5000),
        where num_freqs is the number of frequency channels
        in that band.
        The five keys are: 'gamma', 'beta', 'alpha', 'theta', 'delta'
    """
    freq_bands = {
        "gamma": (30, 80),
        "beta": (13, 30),
        "alpha": (8, 13),
        "theta": (4, 8),
        "delta": (1, 4)
    }

    # Dictionary to store segmented timefreq data for each frequency band
    segmented_data: Dict[str, np.ndarray] = {}

    for band, (low, high) in freq_bands.items():
        indices = np.where((freqs >= low) & (freqs <= high))[0]

        segmented_data[band] = x_timefreq[indices, :]
        
    return segmented_data
