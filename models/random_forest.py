import numpy as np
import neurokit2 as nk
import os
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from feature_extraction import cwt_summary_features
from utils.tools import check_labels


def rf_classifier_timefreq(
    x: np.ndarray, y: np.ndarray, sfreq: float,
    timefreq_method: str='cwt',
    n_estimators: int = 100, max_depth: int = 10,
    seed: int=2025, evaluate: bool=True,
    test_ratio: float = 0.2,
    verbose: int=1,
    n_jobs: int=-1
) -> RandomForestClassifier:
    """
    Build and train a Random Forest classifier on
    features extracted using time-frequency decomposition.

    Parameters
    ----------
    x : np.ndarray
        Input data, with shape [n_samples, channels, time_stamps],
        representing the raw signal data.
    
    y : np.ndarray
        Labels for classification, with shape [n_samples].
        Each label corresponds to a class for the respective sample.
    
    sfreq : float
        The sampling frequency of the signals (Hz).
    
    timefreq_method : str
        The method use for time-frequency decomposition. \n
        Must be one of ['stft', 'cwt', 'wvd', 'pwvd']. \n
        See docstring of `nk.signal_timefrequency`
        for introduction of each method.
    
    n_estimators : int, optional, default=100
        The number of trees in the random forest.
    
    max_depth : int, optional, default=10
        The maximum depth of the trees in the random forest.
    
    seed : int, optional, default=2025
        Random seed used for train-test split
        and training the random forest.
    
    evaluate : bool, optional, default=True
        If true, the performance of trained classifier
        will be evaluated and printed on the test
        set.

    test_ratio : float, optional, default=0.2
        The ratio of test set.
        Set to 0. if no testing is desired.
    
    verbose : int, optional, default=1
        Level of stage-wise reports.
        If set to 0, nothing will be printed.
    
    n_jobs : int, optional, default=-1
        Number of jobs to run in parallel. \n
        Default is -1, which means using all available cores.
    
    Returns
    -------
    RandomForestClassifier
        The trained random forest classifier.
    """
    check_labels(y)

    ex_verbose = True if verbose > 1 else False

    all_features = _extract_features_timefreq(
        x, sfreq, timefreq_method, n_jobs, ex_verbose)

    X = np.array(all_features)

    if verbose > 0:
        print(f'all features extracted, dimension = {X.shape[1]}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed)

    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed)

    if verbose > 0:
        print(f'Training random forest with {n_estimators} trees ...')

    rf.fit(X_train, y_train)

    if verbose > 0:
        print('finished training.')

    if evaluate:
        if verbose > 0:
            print('Evaluating ...')
        y_pred = rf.predict(X_test)
        print(classification_report(y_test, y_pred))

    return rf


def _extract_features_timefreq(
    x: np.ndarray, sfreq: int,
    timefreq_method: str, n_jobs: int,
    verbose: bool
) -> np.ndarray:
    n_samples, n_channels, _ = x.shape
    # store features from all channels
    all_features = []

    if n_jobs == -1:
        n_jobs = os.cpu_count()

    args = [(i, x, n_channels, sfreq, timefreq_method, verbose)
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

def _process_sample(i: int, x: np.ndarray, n_channels: int,
                   sfreq: int, timefreq_method: str, verbose: bool):
    """
    Process a single sample (patient) to extract its features.
    """
    if verbose:
        print(f'Handling sample {i} ....')
    sample_features = []
    
    for c in range(n_channels):
        # Perform time-frequency decomposition
        freqs, _, x_cwt = nk.signal_timefrequency(
            x[i, c, :], sampling_rate=sfreq,
            method=timefreq_method,
            min_frequency=1., max_frequency=80.,
            show=False
        )
        
        # Extract summary statistics
        features = cwt_summary_features(x_cwt, freqs, labels=False)
        sample_features.append(features.flatten())

    if verbose:
        print(f'Finished feature extraction of sample {i}')
    return np.concatenate(sample_features)
