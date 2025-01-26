from data_loader import read_samples
from models.classifiers import classifier_timefreq
import argparse
import os

from utils.tools import load_set_from_file, save_set_to_file, print_args, read_config
from utils.preprocess import read_sampling_rate
from utils.channel_selection import find_significant_channels
from data_loader import build_classification_samples


parser = argparse.ArgumentParser(description='Train Random Forest for seizure prediction')
    
# Basic arguments
parser.add_argument('--data_folder', type=str, default='./dataset',
                    help='The folder in which all data are stored')
parser.add_argument('--config_file', type=str, default='configs/classifiers.json',
                    help='Configurations for classifier')
parser.add_argument('--model_name', type=str, default='rf',
                    help='The classification model used.')

# Drawing samples
parser.add_argument('--sample_length', type=float, default=10.0,
                    help='Length of sample interval (in seconds)')
parser.add_argument('--preictal_time', type=float, default=20.0,
                    help='The time before seizure start (in seconds) to take the sample')
parser.add_argument('--sample_mode', type=str, default='undersample',
                    help='The method used to balance two classes')

# Feature selection and extraction
parser.add_argument('--select_channels', action='store_true',
                    help='Only use selected channels')
parser.add_argument('--timefreq_method', type=str, default='cwt',
                    help='The method use for time-frequency decomposition. '
                    "Must be one of ['stft', 'cwt', 'wvd', 'pwvd'], "
                    "see docstring of `nk.signal_timefrequency`.")

if __name__ == '__main__':
    args = parser.parse_args()

    sample_folder = './samples'
    sample_file = (f'len[{args.sample_length}]-'
                   f'start[{args.preictal_time}]-{args.sample_mode}.h5')
    sample_path = os.path.join(sample_folder, sample_file)
    channel_path = os.path.join(args.data_folder, 'selected_channels.pkl')

    print('------------ Starting Random Forest Experiment --------------')
    print_args(args, description='Settings', return_str=False)
    
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
    model_params = read_config(args.config_file)
    classifier_timefreq(
        x, y, sfreq,
        model_params=model_params, model_name=args.model_name,
        save=True, timefreq_method=args.timefreq_method,
        args=args, result_file=f'{args.model_name}_results.txt')

    print("-" * 40)
