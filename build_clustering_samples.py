from utils.data_loader import build_samples
from utils.tools import load_set_from_file
from utils.channel_selection import find_significant_channels

import os
import argparse


parser = argparse.ArgumentParser(description="Build clustering samples.")

# sample generation
parser.add_argument('--window_widths', type=float, nargs='+',
                    default=[5.0, 10.0],
                    help="List of window widths to extract. "
                    "Default: [5.0, 10.0]")
parser.add_argument('--overlaps', type=float, nargs='+',
                    default=[0.0, 1.0, 2.5],
                    help="List of overlap lengths between windows. "
                    "Default: [0.0, 1.0, 2.5]")
parser.add_argument('--preictal_interval', type=float, default=300.,
                    help="Time interval before seizure to take samples (in seconds)"
                    "Default: 300. (5 minutes)")
parser.add_argument('--random_samples', type=int, default=50,
                    help="Number of random samples to include on top of preictal samples"
                    "Default: 50")
parser.add_argument('--filter_channels', type=bool, default=True,
                    help="Whether to filter out insignificant channels")

# io and dataset
parser.add_argument('--sample_folder', type=str, default='./samples/clustering',
                    help="Folder path to save the generated samples. Default: './samples'")
parser.add_argument('--data_folder', type=str, default='./dataset',
                    help="Path to the folder under which edf files are stored. "
                    "Should also have summary.txt (run `annotation.py`) and "
                    "selected_channels.pkl (run `channel_selection.py`)")

if __name__ == '__main__':
    args = parser.parse_args()

    for width in args.window_widths:
        for overlap in args.overlaps:
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

            sample_file = (f'width[{width}]-overlap[{overlap}]-'
                           f'preictal[{args.preictal_interval}]-'
                           f'nrand[{args.random_samples}].h5'
                           )
            file_path = os.path.join(args.sample_folder, sample_file)

            if not os.path.exists(args.sample_folder):
                os.makedirs(args.sample_folder)

            if not os.path.exists(file_path):
                print(f'Building clustering samples with window width {width}s'
                    f' and {overlap}s overlap between adjacent windows')
                build_samples(
                    args.data_folder, output_file=file_path,
                    selected_channels=selected_channels,
                    mode='clustering',
                    window_width=width, overlap=overlap,
                    preictal_interval=args.preictal_interval,
                    random_samples=args.random_samples)
            else:
                print(f'Sample file already exists: {file_path}')
