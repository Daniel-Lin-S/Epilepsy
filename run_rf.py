from data_loader import read_samples
from models.random_forest import rf_classifier_timefreq
import argparse


# TODO - add channel selection

parser = argparse.ArgumentParser(description='Train Random Forest for seizure prediction')
    
# Define the arguments
parser.add_argument('--sfreq', type=int, default=500, help='Sampling frequency')
parser.add_argument('--file_path', type=str,
                    default='./samples/len[10.0]-start[5.0]-undersample.h5',
                    help='Path to the sample file')

args = parser.parse_args()

if __name__ == '__main__':
    x, y = read_samples(args.file_path)
    print(f'Number of samples {x.shape[0]}, '
        f'Number of channels {x.shape[1]}, '
        f'Number of time stamps {x.shape[2]}')

    rf_classifier_timefreq(x[:, :33, :], y, args.sfreq)
