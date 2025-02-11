from pycox.models import CoxPH
from torchtuples.practical import MLPVanilla
from torchtuples.optim import Adam
import numpy as np
import argparse
import os
import torch

from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split

from models.risk_predictor import RiskPredictionNN
from models.clustering import SeizureClustering
from utils.preprocess import read_sampling_rate
from utils.metrics import calculate_c_index, plot_pr_curve, plot_roc_curve


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
parser.add_argument('--n_ica', type=int, default=50,
                    help="Number of components in dimension reduction before clustering. "
                    "Note: ICA (independent component analysis) used. \n"
                    "Set to 0 for no dimension reduction.")

# survival model training
parser.add_argument('--test_size', type=float, default=0.2,
                    help="Fraction of samples to use as test set. Default: 0.2")
parser.add_argument('--threshold', type=float, default=0.3,
                    help="Threshold for risk prediction. Default: 0.3. \n"
                    "If risk is greater than threshold, predict event. "
                    "Otherwise, predict no event.")
parser.add_argument('--hidden_dim', type=int, default=128,
                    help="Number of hidden units in the risk prediction model "
                     " and survival model."
                     " Default: 128")
parser.add_argument('--n_epochs', type=int, default=80,
                    help="Number of epochs to train the survival model. Default: 80")


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

    # extract samples for survival model
    mask = clustering.labels == -1  # only use anomaly cases
    # TODO - test how many seizure events have been identified
    # may need to modify labelling and `SeizureClustering.cluster_seizure_comparison`
    event_times = clustering.features_meta['distances'][mask]
    risk_labels = (event_times != np.inf) # 1 if event, 0 if not

    X_train, X_test, y_train_time, y_test_time, y_train_labels, y_test_labels = train_test_split(
        clustering.features[mask],  # Feature matrix
        event_times,  # time to next event
        risk_labels,  # event labels
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
    risk_model.train_model(
        X_train, y_train_labels,
        batch_size=32, num_epochs=args.n_epochs)

    ### train survival model ###
    print('Training Cox Proportional Hazard survival model...')
    net = MLPVanilla(
        in_features=n_features,
        num_nodes=[n_features, args.hidden_dim],
        out_features=1)
    surv_model = CoxPH(net, optimizer=Adam)

    # train on event cases
    event_mask = y_train_labels == 1
    surv_model.fit(
        X_train[event_mask],
        (y_train_time[event_mask], y_train_labels[event_mask]),
        batch_size=32, epochs=args.n_epochs, verbose=False)

    ### evaluate models ###
    # risk prediction evaluation
    y_test_pred_risks, y_test_pred_labels = risk_model.predict_risk(
        X_test, threshold=args.threshold)
    
    accuracy = accuracy_score(y_test_labels, y_test_pred_labels)
    precision = precision_score(y_test_labels, y_test_pred_labels)
    recall = recall_score(y_test_labels, y_test_pred_labels)
    f1 = f1_score(y_test_labels, y_test_pred_labels)
    print('---------------- Evaluations ----------------')
    print(f'Metrics for threshold {args.threshold}: ')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 score: ', f1)

    plot_pr_curve(y_test_labels, y_test_pred_risks)
    plot_roc_curve(y_test_labels, y_test_pred_risks)

    # survival time prediction evaluation
    event_mask_pred = y_test_pred_labels == 1
    y_test_time_pred = np.full(len(X_test), np.inf)
    y_test_time_pred[event_mask_pred] = surv_model.predict(
        X_test[event_mask_pred, :]).flatten()

    # compute C-index
    c_index = calculate_c_index(
        y_test_time, y_test_time_pred, y_test_pred_labels)
    print('C-index: ', c_index)
