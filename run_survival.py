import numpy as np
import pandas as pd
import argparse
import os
import torch
from collections import defaultdict

from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE

from lifelines import CoxPHFitter

from models.risk_predictor import RiskPredictionNN
from models.clustering import SeizureClustering
from utils.preprocess import read_sampling_rate
from utils.metrics import calculate_c_index
from utils.visualise import (
    plot_pr_curve, plot_roc_curve, plot_predicted_vs_actual_times
)


parser = argparse.ArgumentParser(description="Run clustering model.")

parser.add_argument('--random_seed', type=int, default=2025,
                    help="Random seed for reproducibility. Default: 2025")

# samples
parser.add_argument('--window_width', type=float, 
                    default=5.0, help="Window width (in seconds) "
                    "used to extract samples. Default: 5.0")
parser.add_argument('--overlap', type=float, default=0.,
                    help="Length of overlap (in seconds) between windows. "
                    "Default: 0.")
parser.add_argument('--preictal_interval', type=float, default=300.,
                    help="Time interval before seizure to take samples (in seconds)"
                    "Default: 300. (5 minutes)")
parser.add_argument('--random_samples', type=int, default=50,
                    help="Number of random samples to include on top of preictal samples"
                    "Default: 50")

# files and io
parser.add_argument('--data_folder', type=str, default='./dataset',
                    help="Path to the folder under which edf files are stored. "
                    "Should also have summary.txt (run `annotation.py`)")
parser.add_argument('--feature_folder', type=str, default='./samples/timefreq',
                    help="Path to the folder under which the h5 files "
                    "of processed time-frequency features are stored. \n"
                    "Default: './samples/timefreq'")

# anomaly detection
parser.add_argument('--nu', type=float, default=0.15,
                   help="nu parameter for one-class SVM. Default: 0.15. "
                   "A higher value of nu will result in more samples "
                   "being classified as anomalies.")
parser.add_argument('--n_ica', type=int, default=200,
                    help="Number of components in dimension reduction before clustering. "
                    "Note: ICA (independent component analysis) used. \n"
                    "Set to 0 for no dimension reduction.")

# survival model training
parser.add_argument('--test_size', type=float, default=0.2,
                    help="Fraction of samples to use as test set. Default: 0.2")
parser.add_argument('--threshold', type=float, default=0.5,
                    help="Threshold for risk prediction. Default: 0.5. \n"
                    "If risk is greater than threshold, predict event. "
                    "Otherwise, predict no event.")
parser.add_argument('--hidden_dim', type=int, default=128,
                    help="Number of hidden units in the risk prediction model "
                     " and survival model."
                     " Default: 128")
parser.add_argument('--n_epochs', type=int, default=50,
                    help="Number of epochs to train the survival model. Default: 50")


if __name__ == "__main__":
    args = parser.parse_args()

    torch.random.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # get samples using one-class SVM
    sample_configs = (
        f'width[{args.window_width}]-overlap[{args.overlap}]-'
        f'preictal[{args.preictal_interval}]-'
        f'nrand[{args.random_samples}]'
        )
    
    feature_filename = f'{sample_configs}-timefreq[cwt].h5'
    feature_file = os.path.join(args.feature_folder, feature_filename)

    if not os.path.exists(feature_file):
        raise FileNotFoundError(
            f"Feature file {feature_file} not found. "
            "Run `feature_extraction.py` first.")

    sfreq = read_sampling_rate(os.path.join(args.data_folder, 'summary.txt'))

    clustering = SeizureClustering(
        sfreq, model=OneClassSVM(nu=args.nu), n_ica=args.n_ica,
        scaling_method='patient')

    clustering.fit(feature_file=feature_file)

    clustering.evaluate_cluster_pre_seizure(-1)

    # extract samples for survival model
    mask = clustering.labels == -1  # only use anomaly cases
    event_times = clustering.features_meta['distances'][mask]
    risk_labels = (event_times != np.inf) # 1 if event, 0 if not

    # pre-seizure period ids: tuples (file_id, seizure_id)
    id_seizures = list(
        zip(clustering.features_meta['id'][mask],
            clustering.features_meta['seizure_id'][mask]))

    (X_train, X_test, y_train_time, y_test_time,
      y_train_labels, y_test_labels, id_seizure_train, id_seizure_test) = train_test_split(
        clustering.features[mask],  # Feature matrix
        event_times,                # time to next event
        risk_labels,                # event labels
        id_seizures,                # id for each pre-seizure period
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=clustering.labels[mask]
    )

    n_features = X_train.shape[1]

    # turn into tensors
    X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
    X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
    y_train_time = torch.from_numpy(y_train_time).to(dtype=torch.float32)
    y_train_labels = torch.from_numpy(y_train_labels).to(dtype=torch.int8)

    ### train risk model ###
    print('Training risk prediction model...')
    risk_model = RiskPredictionNN(
        input_dim=n_features, hidden_dim=args.hidden_dim)
    
    # over-sample minority class (non-event cases)
    smote = SMOTE(sampling_strategy='auto', random_state=args.random_seed)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train_labels)
    X_resampled = torch.from_numpy(X_resampled).to(dtype=torch.float32)
    y_resampled = torch.from_numpy(y_resampled).to(dtype=torch.int8)

    risk_model.train_model(
        X_resampled, y_resampled,
        batch_size=32, num_epochs=args.n_epochs)

    ### train survival model ###
    print('Training Cox Proportional Hazard survival model...')
    surv_model = CoxPHFitter()

    # train on pre-event cases
    event_mask = (y_train_labels == 1) & (y_train_time >= 0)
    print(f'Number of training samples: {torch.sum(event_mask)}')
    features_pre = X_train[event_mask]
    # dimensionality reduction
    pca = PCA(n_components=50)
    latents_pre = pca.fit_transform(features_pre.numpy())
    df_train = pd.DataFrame(latents_pre)
    df_train['duration'] = y_train_time[event_mask]
    df_train['event'] = y_train_labels[event_mask]

    surv_model.fit(df_train, duration_col='duration', event_col='event')

    ### evaluate models ###
    # risk prediction evaluation
    y_train_pred_risks, y_train_pred_labels = risk_model.predict_risk(
        X_train, threshold=args.threshold)

    y_test_pred_risks, y_test_pred_labels = risk_model.predict_risk(
        X_test, threshold=args.threshold)
    
    plot_roc_curve(y_test_labels, y_test_pred_risks)
    plot_pr_curve(y_test_labels, y_test_pred_risks)

    print('---------------- Risk Prediction Evaluations ----------------')
    print('Confusion matrix: ')
    print(confusion_matrix(y_test_labels, y_test_pred_labels))

    # pre-seizure period evaluation
    # { (id, seizure_id) -> [non_seizure_count, seizure_count] }
    pre_seizure_test_counts = defaultdict(lambda: [0, 0]) 

    for sid, pred_label in zip(id_seizure_test, y_test_pred_labels):
        if sid[1] != -1:  # Only count pre-seizure periods
            pre_seizure_test_counts[sid][pred_label] += 1

    print('Pre-seizure identification of test set: ')
    for sid, (label0, label1) in pre_seizure_test_counts.items():
        print(f'{sid} identified seizures: {label1} / {label1 + label0}')
    print('----------------')
    # false alarms
    false_alarms_train = np.sum((y_train_pred_labels == 1) & (y_train_labels.numpy() == 0))
    neg_samples_train = np.sum([y_train_labels == 0])
    print(f'False alarms in train set: {false_alarms_train} / {neg_samples_train}')

    false_alarms_test = np.sum((y_test_pred_labels == 1) & (y_test_labels == 0))
    neg_samples_test = np.sum([y_test_labels == 0])
    print(f'False alarms in test set: {false_alarms_test} / {neg_samples_test}')

    # survival time prediction evaluation
    print('---------------- Survival Time Prediction Evaluations ----------------')
    # only predict time for pre-seizure periods
    event_mask_pred = (y_test_pred_labels == 1) & (y_test_time >= 0)
    y_test_time_pred = np.full(len(X_test), np.inf)

    X_test_latents = pca.transform(X_test[event_mask_pred].numpy())
    hazard_ratios_pred = np.exp(
        np.dot(X_test_latents, surv_model.params_))
    baseline_surv = surv_model.baseline_survival_

    survival_function = []
    # Scale the baseline survival function based on the hazard ratio
    for hazard_ratio in hazard_ratios_pred:
        individual_survival = baseline_surv ** hazard_ratio
        survival_function.append(individual_survival)

    # Convert to a NumPy array or DataFrame for easier handling
    survival_function = np.array(survival_function)

    # Find the median survival time for each individual
    median_survival_times = []
    for i in range(survival_function.shape[0]):
        median_time_index = np.where(survival_function[i, :] <= 0.5)[0][0]
        median_survival_times.append(baseline_surv.index[median_time_index])

    y_test_time_pred[event_mask_pred] = median_survival_times
    
    plot_predicted_vs_actual_times(y_test_time, y_test_time_pred)

    # compute C-index
    c_index = calculate_c_index(
        y_test_time, y_test_time_pred, y_test_pred_labels)
    print('C-index: ', c_index)
