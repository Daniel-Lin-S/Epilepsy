from data_loader import read_samples
from models.random_forest import rf_classifier_timefreq
import argparse
import os

from utils.tools import load_set_from_file, save_set_to_file, print_args
from utils.preprocess import read_sampling_rate
from utils.channel_selection import find_significant_channels
from data_loader import build_classification_samples


parser = argparse.ArgumentParser(description='Train Random Forest for seizure prediction')
    
# Basic arguments
parser.add_argument('--data_folder', type=str, default='./dataset',
                    help='The folder in which all data are stored')
# Drawing samples
parser.add_argument('--sample_length', type=float, default=10.0,
                    help='Length of sample interval (in seconds)')
parser.add_argument('--preictal_time', type=float, default=5.0,
                    help='The time before seizure start (in seconds) to take the sample')
parser.add_argument('--sample_mode', type=str, default='undersample',
                    help='The method used to balance two classes')
# Feature selection
parser.add_argument('--select_channels', action='store_true',
                    help='Only use selected channels')

# Feature extraction
parser.add_argument('--timefreq_method', type=str, default='cwt',
                    help='The method use for time-frequency decomposition. '
                    "Must be one of ['stft', 'cwt', 'wvd', 'pwvd'], "
                    "see docstring of `nk.signal_timefrequency`.")
# Random forest
parser.add_argument('--n_estimators', type=int, default=100,
                    help='The number of trees in the random forest.')
parser.add_argument('--max_depth', type=int, default=10,
                    help='The maximum depth of the trees in the random forest.')

args = parser.parse_args()

if __name__ == '__main__':
    sample_folder = './samples'
    sample_file = (f'len[{args.sample_length}]-'
                   f'start[{args.preictal_time}]-{args.sample_mode}.h5')
    sample_path = os.path.join(sample_folder, sample_file)
    channel_path = os.path.join(args.data_folder, 'selected_channels.pkl')

    print('------------ Starting Random Forest Experiment --------------')
    print_args(args, description='Settings')
    
    # create samples
    if not os.path.exists(sample_path):
        if args.select_channels and not os.path.exists(channel_path):
            print('Selecting channels ...')
            _, selected_channels = find_significant_channels(
                './dataset')

            save_set_to_file(
                selected_channels, './dataset/selected_channels.pkl')
        elif args.select_channels:
            selected_channels = load_set_from_file(
                os.path.join(args.data_folder, 'selected_channels.pkl'),
                as_list=True)
        else: # use all channels
            selected_channels = None

        print('Extracting samples ...')
        build_classification_samples(
            args.data_folder, output_file=sample_path,
            selected_channels=selected_channels,
            sample_time=args.sample_length, preictal_time=args.preictal_time,
            verbose=False)

    x, y = read_samples(sample_path)
    print(f'Number of samples {x.shape[0]}, '
        f'Number of channels {x.shape[1]}, '
        f'Number of time stamps {x.shape[2]}')

    sfreq = read_sampling_rate(os.path.join(args.data_folder, 'summary.txt'))
    rf_classifier_timefreq(x[:10], y[:10], sfreq, save=True,
                           n_estimators=args.n_estimators,
                           max_depth=args.max_depth,
                           timefreq_method=args.timefreq_method,
                           args=args)

    print("-" * 40)
