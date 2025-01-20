import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Union
import neurokit2 as nk


def cwt_summary_features(
        x_cwt: np.ndarray,
        freqs: np.ndarray,
        labels: bool=False
    ) -> Union[Dict[str, Dict[str, float]], np.ndarray]:
    """
    Extracts summary statistics from each frequency band
    of the cwt transformed signal. \n
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
    x_cwt : np.ndarray
        The CWT transformed signal with shape (n_freqs, length),
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
        If labels = False, returns a np.ndarray
        of all the features (float)
    """
    
    # Store features for each frequency band
    features = {}

    segmented_cwtm = _segment_cwt_bands(x_cwt, freqs)

    band_names = list(segmented_cwtm.keys())
    
    # Iterate through each frequency band
    for i, band in enumerate(band_names):
        band_data = segmented_cwtm[band]
        
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
            prev_band_mav = np.mean(np.abs(segmented_cwtm[prev_band]))
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
    Converts the nested dictionary (output of `cwt_summary_features`)
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


def _segment_cwt_bands(
        x_cwt: np.ndarray, freqs: np.ndarray
    ) -> Dict[str, np.ndarray]:
    """
    Segments the CWT transformed signal into different frequency bands.
    * **Gamma** (30-80 Hz)
    * **Beta** (13-30 Hz)
    * **Alpha** (8-13 Hz)
    * **Theta** (4-8 Hz)
    * **Delta** (1-4 Hz)
    
    Parameters
    ----------
    x_cwt : np.ndarray
        The CWT transformed signal with shape (n_freqs, length),
        where n_freqs is the number of frequency channels.
    
    freqs : np.ndarray
        The array of frequencies corresponding to each of the 250 frequency channels.
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary where the keys are the frequency bands
        and the values are the segmented CWT signals. \n
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

    # Dictionary to store segmented CWT data for each frequency band
    segmented_data: Dict[str, np.ndarray] = {}

    for band, (low, high) in freq_bands.items():
        indices = np.where((freqs >= low) & (freqs <= high))[0]

        segmented_data[band] = x_cwt[indices, :]
        
    return segmented_data
