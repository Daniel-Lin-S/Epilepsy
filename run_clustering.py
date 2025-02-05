from utils.data_loader import build_samples, read_samples
from utils.tools import load_set_from_file
from utils.preprocess import read_sampling_rate
from utils.channel_selection import find_significant_channels
from models.clustering import SeizureClustering

from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
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
parser.add_argument('--n_clusters', type=int, default=3,
                    help="Number of clusters. "
                    "Will not be used by DBSCAN.")
parser.add_argument('--model', type=str, default='agglo',
                    help="The model used for clustering, "
                    "one of ['agglo', 'kmeans', 'dbscan', 'gmm']"
                    " where 'agglo' means Agglomerative Clustering, "
                    "a hierarchical clustering method")

# data
parser.add_argument('--data_folder', type=str, default='./dataset',
                    help="Path to the folder under which edf files are stored. "
                    "Should also have summary.txt (run `annotation.py`) and "
                    "selected_channels.pkl (run `channel_selection.py`)")
parser.add_argument('--sample_folder', type=str, default='./samples/clustering',
                    help="Path to the folder under which the h5 files "
                    "of samples are stored. Default: './samples/clustering'")
parser.add_argument('--feature_folder', type=str, default='./samples/timefreq',
                    help="Path to the folder under which the h5 files "
                    "of processed time-frequency features are stored. \n"
                    "Default: './samples/timefreq'")
parser.add_argument('--patient_id', type=str, default=None,
                    help="If provided, only samples from this patient will be used. \n"
                    "a folder with name patient_id should be created in "
                    "the sample folder and feature folder to store the "
                    "samples of this patient."
                    "If not provided, use all patients in the data folder.")


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

    if args.patient_id is not None:
        feature_folder = os.path.join(args.feature_folder, args.patient_id)
        sample_folder = os.path.join(args.sample_folder, args.patient_id)
    else:
        feature_folder = args.feature_folder
        sample_folder = args.sample_folder

    # store samples and time-frequency features into an h5 file
    if not os.path.exists(feature_folder):
        os.mkdir(feature_folder)

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

    if args.model == 'agglo':
        model = AgglomerativeClustering(n_clusters=args.n_clusters)
    elif args.model == 'kmeans':
        model = KMeans(n_clusters=args.n_clusters, random_state=2025)
    elif args.model == 'dbscan':
        model = DBSCAN()
    elif args.model == 'gmm':
        model = GaussianMixture(
            n_components=args.n_clusters, covariance_type='diag',
            random_state=2025)
    else:
        raise ValueError("Invalid model name. Must be one of ['agglo', 'kmeans',"
                         " 'dbscan', 'gmm']")

    clustering = SeizureClustering(sfreq, model=model, n_ica=args.n_ica)

    if os.path.exists(feature_file):
        clustering.fit(feature_file=feature_file)
    else:
        print(f'Did not find feature file {feature_file}')
        if os.path.exists(sample_file):
            sample_dict = read_samples(sample_file, samples_only=False)
        else:
            print('Sample file does exists, proceeding to build samples')
            sample_dict = build_samples(
                args.data_folder, selected_channels=selected_channels,
                mode='clustering', output_file=sample_file,
                window_width=args.window_width, overlap=args.overlap,
                preictal_interval=args.preictal_interval,
                random_samples=args.random_samples,
                patient_id=args.patient_id)

        clustering.fit(sample_dict['x'], sample_dict,
                    feature_file=feature_file)

    if args.patient_id is not None:
        figure_folder = (
            f'figures/{args.patient_id}-clustering-{args.model}'
            f'-{sample_configs}-k[{args.n_clusters}]-nica[{args.n_ica}]'
        )
    else:
        figure_folder = (
            f'figures/clustering-{args.model}-{sample_configs}'
            f'-k[{args.n_clusters}]-nica[{args.n_ica}]'
        )

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

    clustering.plot_clusters(
        file_path=os.path.join(figure_folder, f'tsne.png'))
    clustering.visualize_cluster_distances(
        file_path=os.path.join(figure_folder, f'cluster_dist.png'))
    cluster_sizes = clustering.get_cluster_sizes()

    # minimum and maximum size of a cluster to be
    # considered as a seizure-related cluster
    min_cluster_size = 5
    if args.patient_id is None:
        # more allowance for clustering of all patients
        max_cluster_size = 200
    else:
        max_cluster_size = 100

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
            if index != major_class and (
                size >= min_cluster_size and size <= max_cluster_size):
                minor_classes.append(index)

        if args.patient_id is None:
            # check presence of each cluster for each seizure
            for index in minor_classes:
                clustering.evaluate_cluster(index)

        if len(minor_classes) > 0:
            clustering.plot_histogram_dists(
                minor_classes, bins=100,
                file_path=os.path.join(figure_folder, 'hist_dist.png')) 
