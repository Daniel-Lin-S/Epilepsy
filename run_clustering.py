from data_loader import build_samples, read_samples
from utils.tools import load_set_from_file
from utils.preprocess import read_sampling_rate
from utils.channel_selection import find_significant_channels
from models.clustering import SeizureClustering

from sklearn.cluster import AgglomerativeClustering
import os
import argparse


parser = argparse.ArgumentParser(description="Run clustering model.")

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
parser.add_argument('--filter_channels', type=bool, default=True,
                    help="Whether to filter out insignificant channels")

# clustering
parser.add_argument('--n_ica', type=int, default=30,
                    help="Number of components in dimension reduction before clustering. "
                    "Note: ICA (independent component analysis) used. \n"
                    "Set to 0 for no dimension reduction.")
parser.add_argument('--n_clusters', type=int, default=2,
                    help="Number of clusters")

# data
parser.add_argument('--data_folder', type=str, default='./dataset',
                    help="Path to the folder under which edf files are stored. "
                    "Should also have summary.txt (run `annotation.py`) and "
                    "selected_channels.pkl (run `channel_selection.py`)")


if __name__ == '__main__':
    args = parser.parse_args()

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

    # store samples and time-frequency features into an h5 file
    feature_folder = './samples/timefreq'
    if not os.path.exists(feature_folder):
        os.mkdir(feature_folder)

    sample_folder = './samples/clustering'
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    sample_configs = (
        f'width[{args.window_width}]-overlap[{args.overlap}]-'
        f'preictal[{args.preictal_interval}]-'
        f'nrand[{args.random_samples}]'
        )
    
    feature_filename = f'{sample_configs}-timefreq[cwt].h5'
    sample_filename = f'{sample_configs}.h5'

    feature_file = os.path.join(feature_folder, feature_filename)
    sample_file = os.path.join(sample_folder, sample_filename)

    sfreq = read_sampling_rate(os.path.join(args.data_folder, 'summary.txt'))

    model = AgglomerativeClustering(n_clusters=args.n_clusters)
    clustering = SeizureClustering(sfreq, model=model, n_ica=args.n_ica)

    if os.path.exists(feature_file):
        clustering.fit(feature_file=feature_file)
    else:
        if os.path.exists(sample_file):
            sample_dict = read_samples(sample_file, samples_only=False)
        else:
            sample_dict = build_samples(
                args.data_folder, selected_channels=selected_channels,
                mode='clustering', output_file=sample_file,
                window_width=args.window_width, overlap=args.overlap,
                preictal_interval=args.preictal_interval,
                random_samples=args.random_samples)

        clustering.fit(sample_dict['x'], sample_dict,
                    feature_file=feature_file)

    figure_folder = (f'figures/clustering-{sample_configs}-k[{args.n_clusters}]'
                     f'-nica[{args.n_ica}]'
                     )

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

    clustering.plot_clusters(
        file_path=os.path.join(figure_folder, f'tsne.png'))
    clustering.visualize_cluster_distances(
        file_path=os.path.join(figure_folder, f'cluster_dist.png'))
    cluster_sizes = clustering.get_cluster_sizes()

    # minimum size of a cluster to be considered as not noise
    min_cluster_size = 5

    # search for major and minor classes
    major_size = 0
    major_class = None

    for label, size in cluster_sizes.items():
        if size > major_size and label != -1:
            major_size = size
            major_class = label

    if major_class is not None:
        # collect minor classes
        minor_classes = []
        for index, size in cluster_sizes.items():
            if index != major_class and size >= min_cluster_size:
                minor_classes.append(index)

        for index in minor_classes:
            clustering.evaluate_cluster(index)

        clustering.plot_histogram_dists(
            minor_classes, bins=100,
            file_path=os.path.join(figure_folder, 'hist_dist.png')) 
