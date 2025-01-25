from data_loader import build_classification_samples
from utils.preprocess import read_sampling_rate
from utils.tools import load_set_from_file

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
                    help="The number of negative samples to generate. "
                    "Default: -1 (same number as positive samples)")
parser.add_argument('--folder_path', type=str, default='./samples',
                    help="Folder path to save the generated samples. Default: './samples'")
parser.add_argument('--summary_file', type=str, default='./dataset/summary.txt',
                    help="Path to the summary file to read the sampling rate. "
                    "Default: './dataset/summary.txt'")
parser.add_argument('--selected_channels_file', type=str,
                    default='./dataset/selected_channels.pkl',
                    help="File containing selected channels. "
                    "Default: './dataset/selected_channels.pkl'")

args = parser.parse_args()

for sample_time in args.sample_times:
    for preictal_time in args.preictal_times:
        sfreq = read_sampling_rate(args.summary_file)

        if not os.path.exists(args.folder_path):
            os.makedirs(args.folder_path)

        sample_file = f'len[{sample_time}]-start[{preictal_time}]-undersample.h5'

        selected_channels = load_set_from_file(
            args.selected_channels_file, as_list=True)

        file_path = os.path.join(args.folder_path, sample_file)
        if not os.path.exists(file_path):
            print(f'Building samples with length {round(sample_time, 1)}s'
                  f' from {round(preictal_time, 1)}s before onset start time')
            build_classification_samples(
                './dataset/', output_file=file_path,
                selected_channels=selected_channels,
                sample_time=sample_time, preictal_time=preictal_time,
                n_negative=args.n_negative)
