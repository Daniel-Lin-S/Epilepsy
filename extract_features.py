from utils.feature_extraction import extract_features_timefreq
from utils.data_loader import read_samples
from utils.preprocess import read_sampling_rate

import argparse
import h5py

parser = argparse.ArgumentParser(
    description="Extract features from samples and save to a specified file.")

# io and data
parser.add_argument('--sample_file', type=str, default='./samples/sample.h5',
                    help="Path to the file in which the samples are stored.")
parser.add_argument('--feature_file', type=str, default='./samples/timefreq/sample.h5',
                    help="Path to the file in which the "
                    "extracted features should be stored. ")
parser.add_argument('--summary_file', type=str, default='./dataset/summary.txt',
                    help="The file in which meta information are stored"
                    ", it would be produced by `annotation.py`")

# time-frequency feature extraction
parser.add_argument('--timefreq_method', type=str, default='cwt',
                    help="The method used to generate time-frequency graph")
parser.add_argument('--n_jobs', type=int, default=-1,
                    help="Number of parallel processors used to extract features")

if __name__ == '__main__':
    args = parser.parse_args()

    sample_dict = read_samples(
        args.sample_file, samples_only=False, handle_str=False)

    sfreq = read_sampling_rate(args.summary_file)

    features = extract_features_timefreq(
        x=sample_dict['x'], sfreq=sfreq,
        timefreq_method=args.timefreq_method,
        n_jobs=args.n_jobs)
    
    with h5py.File(args.feature_file, 'w') as f:
        f.create_dataset('features', data=features)
        for key in sample_dict.keys():
            if key != 'x':
                f.create_dataset(key, data=sample_dict[key])
    print(f'Features saved to {args.feature_file}')
