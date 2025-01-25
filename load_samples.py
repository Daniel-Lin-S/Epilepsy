from data_loader import build_classification_samples
from utils.preprocess import read_sampling_rate
from utils.tools import load_set_from_file
from utils.channel_selection import find_significant_channels

import os
import argparse


parser = argparse.ArgumentParser(description="Build classification samples for EEG data.")

parser.add_argument('--sample_times', type=float, nargs='+', default=[5.0, 10.0, 20.0],
                    help="List of sample times / length of samples (in seconds). "
                    "Default: [5.0, 10.0, 20.0]")
parser.add_argument('--preictal_times', type=float, nargs='+',
                    default=[10.0, 20.0, 30.0, 40.0, 60.0],
                    help="List of intervals between seizure time and sample (in seconds). "
                    "Default: [10.0, 20.0, 30.0, 40.0, 60.0]")
parser.add_argument('--n_negative', type=int, default=-1, 
                    help="The number of negative samples to generate for each edf file. "
                    "Default: -1 (same number as positive samples)")
parser.add_argument('--filter_channels', type=bool, default=True,
                    help="Whether to filter out insignificant channels")
parser.add_argument('--sample_folder', type=str, default='./samples',
                    help="Folder path to save the generated samples. Default: './samples'")
parser.add_argument('--data_folder', type=str, default='./dataset',
                    help="Path to the folder under which edf files are stored. "
                    "Should also have summary.txt (run `annotation.py`) and "
                    "selected_channels.pkl (run `channel_selection.py`)")

if __name__ == '__main__':
    args = parser.parse_args()

    sfreq = read_sampling_rate(os.path.join(args.data_folder, 'summary.txt'))

    for sample_time in args.sample_times:
        for preictal_time in args.preictal_times:
            if args.filter_channels:
                channels_file = os.path.join(args.data_folder, 'selected_channels.pkl')

                # load selected channels
                if not os.path.exists(channels_file):
                    _, selected_channels = find_significant_channels(args.data_folder)
                else:
                    selected_channels = load_set_from_file(
                        channels_file, as_list=True)
            else:
                selected_channels = None

            sample_file = f'len[{sample_time}]-start[{preictal_time}]-undersample.h5'
            file_path = os.path.join(args.sample_folder, sample_file)

            if not os.path.exists(args.sample_folder):
                os.makedirs(args.sample_folder)

            if not os.path.exists(file_path):
                print(f'Building samples with length {round(sample_time, 1)}s'
                    f' from {round(preictal_time, 1)}s before onset start time')
                build_classification_samples(
                    args.data_folder, output_file=file_path,
                    selected_channels=selected_channels,
                    sample_time=sample_time, preictal_time=preictal_time,
                    n_negative=args.n_negative)
