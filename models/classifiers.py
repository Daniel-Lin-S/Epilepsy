import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from argparse import Namespace
from typing import Optional, List, Dict

from feature_extraction import extract_features_timefreq
from utils.tools import check_labels, print_args, print_dict


def classifier_timefreq(
    x: np.ndarray, y: np.ndarray, sfreq: float,
    model_params: Dict[str, dict], model_name: str,
    timefreq_method: str='cwt', seed: int=2025,
    evaluate: bool=True, save: bool=False,
    args: Optional[Namespace]=None,
    test_ratio: float = 0.2,
    verbose: int=1,
    n_jobs: int=-1,
    num_experiments: int=5,
    result_file: str='result.txt'
) -> List[RandomForestClassifier]:
    """
    Build and train a classifier on
    features extracted using time-frequency decomposition. \n
    Train-test separation included.

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

    model_params : Dict[str, dict]
        The hyper-parameters of each classification model.
    
    model_name : str
        The classification model used. \n
        Options: 
        - 'rf' : random forest
        - 'svm' : Support Vector Machine (CSVM)
    
    timefreq_method : str
        The method use for time-frequency decomposition. \n
        Must be one of ['stft', 'cwt', 'wvd', 'pwvd']. \n
        See docstring of `nk.signal_timefrequency`
        for introduction of each method.

    seed : int, optional, default=2025
        Random seed used for train-test split
        and training the random forest.
    
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

    num_experiments : int, optional, default=5
        Number of times to repeat the experiment
        using various seeds.
    
    result_file : str, optional
        The name of the file in which results
        should be saved. \n
        Only valid if save=True
    
    Returns
    -------
    list
        List of trained classifiers.
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

    models = []

    statis_models = ['svm']
    if num_experiments > 1 and model_name in statis_models:
        num_experiments = 1
        print('CSVM have no randomness, '
                'setting number of experiments to 1.')

    for i in range(num_experiments):
        print(f'Iteration {i}')

        if model_name == 'rf':
            n_estimators = model_params['rf']['n_estimators']
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed+i,
                **model_params['rf'].get('kwargs', {}))

            if verbose > 0:
                print(
                    "Training random forest with "
                    f"{n_estimators} trees ...")
        elif model_name == 'svm':
            kernel = model_params['svm']['kernel']
            model = SVC(
                C=model_params['svm']['C'],
                kernel=kernel,
                random_state=seed+i,
                **model_params['svm'].get('kwargs', {})
            )

            if verbose > 0:
                print(
                    "Training CSVM with "
                    f"{kernel} kernel ...")
        else:
            raise ValueError(
                f'Unsupport model_name {model_name}'
            )

        model.fit(X_train, y_train)

        if evaluate:
            y_pred = model.predict(X_test)
            evaluate_classifier(
                y_test, y_pred, save,
                model_params=model_params[model_name],
                exp_id=i,
                file_name=result_file,
                args=args)
        
        models.append(model)

    return models


def evaluate_classifier(
        y_test: np.ndarray,
        y_pred: np.ndarray, save: bool,
        model_params : dict,
        exp_id: int=0,
        file_name : str='result.txt',
        args: Optional[Namespace]=None):
    """
    Evaluate classification result based on 
    confusion matrix, and optionally save the results
    to a text file.

    Parameters
    ----------
    X_test, y_test : np.ndarray
        the test set and labels
    y_pred : np.ndarray
        The predicted labels
    save : bool
        If true, the results
        will be saved. Otherwise,
        all printed.
    exp_id : int, optional, default=0
        Used to store multiple repeititions
        of an experiment with the same
        settings. Use 0 to include header.
    file_name : str, optional
        The name of the file in which results
        should be saved. 
    args : Namespace, optional
        The experiment settings. \n
        If provided, will be added
        at the beginning.
    """
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

    if exp_id == 0:
        # 40 dashes as a divider
        header = "=" * 40 + "\n"
        if args is not None:
            config_str = print_args(args, 'Settings', return_str=True)
            header = header + config_str + "\n"
        # Model Hyperparameters
        params_str = print_dict(model_params)
        header = header + 'Classifier hyper-parameters: ' + params_str + "\n"
        header = header + "\n"
        full_report = header + full_report
    else:
        separator = f"-------- Iteration {exp_id} --------" + "\n"
        full_report = separator + full_report

    if save:
        with open(file_name, 'a') as f:
            f.write(full_report)
            f.write("\n")
        print(f'Evaluation report saved to {file_name}.')
    else:
        print(full_report)
