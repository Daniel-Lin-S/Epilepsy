import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from argparse import Namespace
from typing import Optional

from feature_extraction import extract_features_timefreq
from utils.tools import check_labels, print_args


def rf_classifier_timefreq(
    x: np.ndarray, y: np.ndarray, sfreq: float,
    timefreq_method: str='cwt', seed: int=2025,
    n_estimators: int = 100, max_depth: int = 10,
    evaluate: bool=True, save: bool=False,
    args: Optional[Namespace]=None,
    test_ratio: float = 0.2,
    verbose: int=1,
    n_jobs: int=-1,
    **kwargs
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

    seed : int, optional, default=2025
        Random seed used for train-test split
        and training the random forest.
    
    n_estimators : int, optional, default=100
        The number of trees in the random forest.
    
    max_depth : int, optional, default=10
        The maximum depth of the trees in the random forest.
    
    evaluate : bool, optional, default=True
        If true, the performance of trained classifier
        will be evaluated and printed on the test
        set.
    
    save : bool, optioanl, default=False
        If true, the evaluation report
        will be saved into the rf_results.txt
        file.
    
    args : argparse.namespace, optional
        Must be defined when save=True.
        Used to record experiment
        settings.

    test_ratio : float, optional, default=0.2
        The ratio of test set.
        Set to 0. if no testing is desired.
    
    verbose : int, optional, default=1
        Level of stage-wise reports.
        If set to 0, nothing will be printed.
    
    n_jobs : int, optional, default=-1
        Number of jobs to run in parallel. \n
        Default is -1, which means using all available cores.
    
    kwargs : Any
        keyword arguments passed to sklearn.ensemble.RandomForestClassifier
    
    Returns
    -------
    RandomForestClassifier
        The trained random forest classifier.
    """
    check_labels(y)

    ex_verbose = True if verbose > 1 else False

    all_features = extract_features_timefreq(
        x, sfreq, timefreq_method, n_jobs, ex_verbose)

    X = np.array(all_features)

    if verbose > 0:
        print(f'all features extracted, dimension = {X.shape[1]}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed)

    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, **kwargs)

    if verbose > 0:
        print(f'Training random forest with {n_estimators} trees ...')

    rf.fit(X_train, y_train)

    if verbose > 0:
        print('finished training.')

    if evaluate:
        if verbose > 0:
            print('Evaluating ...')
        evaluate_rf(X_test, y_test, rf, save, args)

    return rf


def evaluate_rf(X_test : np.ndarray, y_test : np.ndarray,
                rf : RandomForestClassifier, save : bool,
                args: Optional[Namespace]=None):
    y_pred = rf.predict(X_test)
    # metrics based on confusion matrix
    report = classification_report(y_test, y_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    class_labels = np.unique(y_test)

    # Format confusion matrix for better readability
    cm_str = "Confusion Matrix:\n"
    cm_str += f"              {', '.join(map(str, class_labels))}\n"
    
    for i, row in enumerate(cm):
        cm_str += f"True Label {class_labels[i]}: {', '.join(map(str, row))}\n"

    full_report = f"{cm_str}\n\n{report}"

    if save:
        if args is None:
            raise ValueError('args must be provided when save=True.')
        # 40 dashes as a divider
        separator = "=" * 40 + "\n"  
        file_name = "rf_results.txt"
        config_str = print_args(args, 'Settings', return_str=True)

        with open(file_name, 'a') as f:
            f.write(separator)
            f.write(config_str + "\n\n")
            f.write(full_report)
            f.write("\n")
        print(f'Evaluation report saved to {file_name}.')
    else:
        print(report)
