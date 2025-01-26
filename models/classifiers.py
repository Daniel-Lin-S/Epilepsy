import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    accuracy_score
)
from sklearn.decomposition import FastICA
from argparse import Namespace
from typing import Optional, Dict
import os
import h5py

from feature_extraction import extract_features_timefreq
from utils.tools import check_labels, print_args, print_dict, confusion_matrix_to_str
from data_loader import read_samples


def classifier_timefreq(
    x: np.ndarray, y: np.ndarray, sfreq: float,
    model_params: Dict[str, dict],
    seed: int=2025,
    store_features: bool=False,
    evaluate: bool=True, save: bool=False,
    args: Optional[Namespace]=None,
    verbose: int=1,
    n_jobs: int=-1,
    n_folds: int=5,
    result_file: str='result.txt',
    detailed_log: bool=False
) -> list:
    """
    Build and train a classifier on
    features extracted using time-frequency decomposition. \n
    Train-test separation included, using K-Fold validation.

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

    seed : int, optional, default=2025
        Random seed used for train-test split
        and training the random forest.
    
    store_features : bool, optional
        If true, the features will be stored
        as h5 file for reuse.

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
        settings. Must contain the following:
        - 'n_features' (int) : Number of features to keep in
        ICA to reduce the dimension. If negative, all features used.
        - 'sample_length' (float): Length of sample interval (in seconds)
        - 'preictal_time' (float): The time before seizure start
        (in seconds) to take the sample
        - 'timefreq_method' (str) : The method use for time-frequency decomposition.
        Must be one of ['stft', 'cwt', 'wvd', 'pwvd'].
        - 'model_name' (str) : The classification model used. \n
        Options: 'rf' : random forest; 'svm' : Support Vector Machine (CSVM)

    verbose : int, optional, default=1
        Level of stage-wise reports.
        If set to 0, nothing will be printed.

    n_jobs : int, optional, default=-1
        Number of jobs to run in parallel. \n
        Default is -1, which means using all available cores.

    n_folds : int, optional, default=5
        Number of folds in the K-fold cross-validation.

    result_file : str, optional
        The name of the file in which results
        should be saved. \n
        Only valid if save=True

    detailed_log : bool, optional
        If true, the confusion matrix and classification
        report of each fold will be added.

    Returns
    -------
    list
        List of trained classifiers.
    """
    check_labels(y)

    ex_verbose = True if verbose > 1 else False

    if store_features:
        feature_folder = './samples/timefreq/'
        feature_file = (f'len[{args.sample_length}]-start[{args.preictal_time}]'
                        f'-timefreq[{args.timefreq_method}].h5')
        
        feature_path = os.path.join(feature_folder, feature_file)

        if not os.path.exists(feature_folder):
            os.makedirs(feature_folder)

        if os.path.exists(feature_path):
            all_features, y = read_samples(feature_path)
        else:
            all_features = extract_features_timefreq(
                x, sfreq, args.timefreq_method, n_jobs, ex_verbose)
            with h5py.File(feature_path, 'w') as f:
                f.create_dataset('x', data=all_features)
                f.create_dataset('y', data=y)
    else:
        all_features = extract_features_timefreq(
                x, sfreq, args.timefreq_method, n_jobs, ex_verbose)
    
    dim_total = all_features.shape[1]

    if verbose > 0:
        print(f'Extracted {dim_total} features')

    if args.n_features <= 0:
        X = np.array(all_features)
    elif args.n_features <= dim_total:
        print(f'Reducing dimension to {args.n_features} ...')
        ica = FastICA(n_components=args.n_features)
        X = ica.fit_transform(all_features)
    else:
        raise ValueError(
            f'n_features ({args.n_features}) cannot exceed total '
            f'number of features: {dim_total}.'
        )

    models = []
    
    if evaluate:
        accuracies = []
        recalls = []
        fscores = []
        precisions = []
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(f'Fold {i}')

        if args.model_name == 'rf':
            try:
                n_estimators = model_params['rf']['n_estimators']
            except KeyError:
                raise KeyError(
                    "'n_estimators' must be defined "
                    "in the configurations for rf")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                **model_params['rf'].get('kwargs', {}))

            if verbose > 0:
                print(
                    "Training random forest with "
                    f"{n_estimators} trees ...")
        elif args.model_name == 'svm':
            try:
                kernel = model_params['svm']['kernel']
                C = model_params['svm']['C']
            except KeyError:
                raise KeyError(
                    "'kernel' and 'C' must be defined "
                    "in the configurations for svm")
            model = SVC(
                C=C,
                kernel=kernel,
                random_state=seed,
                **model_params['svm'].get('kwargs', {})
            )

            if verbose > 0:
                print(
                    "Training CSVM with "
                    f"{kernel} kernel and C={C} ...")
        elif args.model_name == 'logreg':  # Add case for Logistic Regression
            try:
                penalty = model_params['logreg']['penalty']
            except KeyError:
                raise KeyError(
                    "'C' must be defined "
                    "in the configurations for logistic regression")
            model = LogisticRegression(
                penalty=penalty,
                random_state=seed,
                **model_params['logreg'].get('kwargs', {})
            )

            if verbose > 0:
                print(
                    "Training Logistic Regression with "
                    f"{penalty} penalty ...")
        else:
            raise ValueError(
                f'Unsupport model_name {args.model_name}'
            )
    
        model.fit(X_train, y_train)

        if evaluate:
            y_pred = model.predict(X_test)
            precision, recall, fscore, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            accuracies.append(accuracy_score(y_test, y_pred))
            recalls.append(recall)
            precisions.append(precision)
            fscores.append(fscore)
            if detailed_log or i == 0:
                evaluate_classifier(
                    y_test, y_pred, save,
                    model_params=model_params[args.model_name],
                    exp_id=i,
                    file_name=result_file,
                    args=args,
                    header_only=not detailed_log)
        
        models.append(model)
    
    if evaluate and n_folds > 1:
        separator = f"-------- Aggregate --------" + "\n"
        metrics = (
            f"Accuracy : {np.mean(accuracies)} ± {np.std(accuracies)}"
            f"; Sensitivity (recall) : {np.mean(recalls)} ± {np.std(recalls)}"
            f"; \n Precision : {np.mean(precisions)} ± {np.std(precisions)}"
            f"; F-score : {np.mean(fscores)} ± {np.std(fscores)}"
        )
        with open(result_file, 'a') as f:
            f.write(separator + metrics)
            f.write("\n")
        print(f'Summary metrics saved to {result_file}.')

    return models


def evaluate_classifier(
        y_test: np.ndarray,
        y_pred: np.ndarray, save: bool,
        model_params: dict,
        exp_id: int=0,
        file_name : str='result.txt',
        args: Optional[Namespace]=None,
        header_only: bool=False
    ) -> None:
    """
    Evaluate classification result based on 
    confusion matrix, and optionally save the results
    to a text file.

    Parameters
    ----------
    y_test, y_pred : np.ndarray
        the ground truth and predicted labels
    save : bool
        If true, the results
        will be saved. Otherwise,
        all printed.
    model_params : dict
        The hyper-parameters of the model
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
    header_only : bool
        If true, only generate the header.
        With no classification report
    """
    # set header with configurations
    if exp_id == 0:
        # 40 dashes as a divider
        header = "=" * 40 + "\n"
        if args is not None:
            config_str = print_args(args, 'Settings', return_str=True)
            header = header + config_str + "\n"
        # Model Hyperparameters
        params_str = print_dict(model_params)
        header = header + 'Classifier hyper-parameters: ' + params_str + "\n"

    # get confusion matrix and all metrics
    if not header_only:
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        class_labels = np.unique(y_test)
        cm_str = confusion_matrix_to_str(cm, class_labels)

        full_report = f"{cm_str}\n\n{report}"

        header = header + f"-------- Fold {exp_id} --------" + "\n"
        full_report = header + full_report
    else:
        full_report = header

    if save:
        with open(file_name, 'a') as f:
            f.write(full_report)
            f.write("\n")
    else:
        print(full_report)
