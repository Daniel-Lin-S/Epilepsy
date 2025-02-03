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
from utils.tools import (
    check_labels, print_args, print_dict, confusion_matrix_to_str,
    save_to_csv
)
from data_loader import read_samples


# TODO modify classifier_timefreq to return a model trained on all x instead of cross-validation. 
# Perhaps make this into a class

def classifier_timefreq(
    x: np.ndarray, y: np.ndarray, sfreq: float,
    model_params: Dict[str, dict],
    args: Namespace,
    seed: int=2025,
    evaluate: bool=True,
    verbose: int=1,
    n_jobs: int=-1,
    n_folds: int=5,
    result_file: str='classifier_results.csv',
    save_confusion: bool=False
) -> list:
    """
    Build and train a classifier on
    features extracted using time-frequency decomposition. \n
    Train-test separation included, using K-Fold validation. \n
    All metrics will be saved to a csv file
    if evaluate=True (by default).

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
        The hyper-parameters of each classification model. \n
        The keys should be the model names and values being
        detailed hyper-parameter settings.
        - 'rf': must define 'n_estimators',
        - 'svm' : must define 'C' and 'kernel'
        - 'logreg': must define 'penalty' \n
        other arguments should be defined in key 'kwargs', with
        its value being a dictionary of the arguments.

    args : argparse.namespace, optional
        Used to record experiment
        settings. Must contain the following keys:
        - 'n_features' (int) : Number of features to keep in
        ICA to reduce the dimension. If negative, all features used.
        - 'sample_length' (float): Length of sample interval (in seconds)
        - 'preictal_time' (float): The time before seizure start
        (in seconds) to take the sample
        - 'timefreq_method' (str) : The method use for time-frequency decomposition.
        Must be one of ['stft', 'cwt', 'wvd', 'pwvd'].
        - 'model_name' (str) : The classification model used. \n
        Options: 'rf' : random forest; 'svm' : Support Vector Machine (CSVM)
        - 'store_features' (bool) : If true, the time-frequency features
        will be stored in a h5 file for reuse.

    seed : int, optional, default=2025
        Random seed used for train-test split
        and training the random forest.

    evaluate : bool, optional, default=True
        If true, the performance of trained classifier
        will be evaluated and printed on the test
        set.

    verbose : int, optional, default=1
        Level of stage-wise reports.
        0 - nothing printed. \n
        1 - experiment progress. \n
        2 - details of classifier included.

    n_jobs : int, optional, default=-1
        Number of jobs to run in parallel. \n
        Default is -1, which means using all available cores.

    n_folds : int, optional, default=5
        Number of folds in the K-fold cross-validation.
    
    result_file : str, optional
        Name of the file in which classification metrics
        are saved. \n
        Default is 'classifier_results.csv'

    save_confusion : bool, optioanl, default=False
        If true, the confusion matrix and
        detailed classification report of each fold
        will be saved into the '{model_name}_results.txt'
        file.

    Returns
    -------
    list
        List of trained classifiers.
    """
    check_labels(y)

    # Extract features
    if args.store_features:
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
                x, sfreq, args.timefreq_method, n_jobs)
            with h5py.File(feature_path, 'w') as f:
                f.create_dataset('x', data=all_features)
                f.create_dataset('y', data=y)
    else:
        all_features = extract_features_timefreq(
                x, sfreq, args.timefreq_method, n_jobs)
    
    dim_total = all_features.shape[1]
    n_samples = all_features.shape[0]

    if verbose > 0:
        print(f'Extracted {dim_total} features')

    # perform dimension reduction (optionally)
    if args.n_features <= 0:
        X = np.array(all_features)
    elif args.n_features <= min(n_samples ,dim_total):
        print(f'Reducing dimension to {args.n_features} ...')
        ica = FastICA(n_components=args.n_features)
        X = ica.fit_transform(all_features)
    else:
        raise ValueError(
            f'n_features ({args.n_features}) cannot exceed total '
            f'number of features: {dim_total} and number of samples {n_samples}.'
        )

    models = []

    model_verbose = verbose > 1
    
    if evaluate:
        accuracies = []
        recalls = []
        fscores = []
        precisions = []

    # use K-fold cross-validation
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
                verbose=model_verbose,
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
                verbose=model_verbose,
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
                    "'penalty' must be defined "
                    "in the configurations for logistic regression")
            model = LogisticRegression(
                penalty=penalty,
                random_state=seed,
                verbose=model_verbose,
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
        models.append(model)

        if evaluate:
            y_pred = model.predict(X_test)
            precision, recall, fscore, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            accuracies.append(accuracy_score(y_test, y_pred))
            recalls.append(recall)
            precisions.append(precision)
            fscores.append(fscore)

            if save_confusion:
                evaluate_classifier(
                    y_test, y_pred, save_confusion,
                    model_params=model_params[args.model_name],
                    exp_id=i,
                    file_name=f'{args.model_name}_results.txt',
                    args=args
                )

                if verbose > 0 and i == (n_folds-1):
                    print(
                        'Confusion matrices saved to '
                        f'{args.model_name}_results.txt.')

    if evaluate: # save aggregated metrics
        data = {
            "model_name": args.model_name,
            "model_params": print_dict(model_params),
            "sample_length": args.sample_length,
            "preictal_time": args.preictal_time,
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "accuracies": accuracies,
            "recall_mean": np.mean(recalls),
            "recall_std": np.std(recalls),
            "recalls": recalls,
            "precision_mean": np.mean(precisions),
            "precision_std": np.std(precisions),
            "precisions": precisions,
            "fscore_mean": np.mean(fscores),
            "fscore_std": np.std(fscores),
            "fscores": fscores
        }
        save_to_csv(data, result_file)
        if verbose > 0:
            print(f'Metrics saved to {result_file}')

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
