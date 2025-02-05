import numpy as np
from sklearn.base import ClusterMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from typing import Optional, List
import os
import h5py
import warnings

from utils.feature_extraction import extract_features_timefreq


class SeizureClustering:
    """
    A model for unsupervised clustering of time series data
    The model uses clustering algorithms to group the data into clusters.
    """

    def __init__(self, sfreq: int, timefreq_method: str='cwt',
                 model: Optional[ClusterMixin]=None, n_jobs: int = -1,
                 scaling: bool=True, n_ica: int=0,
                 seed: int=2025):
        """
        Parameters
        ----------
        sfreq : int
            Sampling frequency of the time series data.
        timefreq_method : str
            Method used for feature extraction.
            See docstring of `feature_extraction.extract_features_timefreq`
        model : sklearn.base.ClusterMixin, optional
            Instance of a clustering model specified in sklearn,
            or self-defined instance with functions fit_predict
            and predict provided. \n
            Default is sklearn.cluster.AgglomerativeClustering
            with 2 clusters.
        n_jobs : int, optional
            Number of parallel jobs to run. \n
            Default is -1, which uses all available processors.
        scaling : bool, optional
            if true, the features will be standardised to ensure they
            are on the same scale.
        n_ica : int, optional
            If a positive value given, ICA will be performed
            to reduce the dimension of the dataset.
        seed : int, optional
            The random seed used for training the clustering model
            to ensure reproducibility
        """
        self.sfreq = sfreq
        self.timefreq_method = timefreq_method
        if model is None:
            self.model = AgglomerativeClustering(n_clusters=2)
        else:
            self.model = model
        self.scaling = scaling
        self.n_ica = n_ica
        if scaling:
            self.scaler = StandardScaler()
        if self.n_ica > 0:
            self.ica = FastICA(n_components=self.n_ica)
        self.n_jobs = n_jobs
        self.seed = seed
        self.labels = None
        self.features = None
        self.features_meta = {}

    def extract_features(
            self, samples: Optional[np.ndarray]=None,
            sample_dict: Optional[dict]=None,
            feature_file: Optional[str]=None):
        """
        Extract features from the time series samples
        using the specified time-frequency method. \n
        self.features will be assigned an array of shape
        (num_samples, n_features)
        representing the extracted features.

        Parameters
        ----------
        samples : np.ndarray, optional
            Array of shape (num_samples, num_channels, sample_length)
            representing the time series samples. \n
            If not provided, will be read from h5 file
        sample_dict : np.ndarray, optional.
            Dictionary with meta-information relating
            to the samples, e.g. distance to seizure,
            sample id (original file). \n
            If not given, only the features themselves
            are stored in the h5 file.
        feature_file : str, optional
            If provided, the extracted features and dists will be saved
            or read from the file. \n
            If the file exists, feature will be directly read from this folder.
            Otherwise, the extracted features will be saved to a
            h5 file.
        """
        # clear previous entries
        self.features = None
        self.features_meta = {}

        if feature_file is not None and os.path.exists(feature_file):
            with h5py.File(feature_file, 'r') as f:
                self.features = f['features'][:]

                for key in f.keys():
                    if key == 'y' or key == 'dist':
                        self.features_meta['distances'] = f[key][:]
                    elif key != 'features':
                        self.features_meta[key] = f[key][:]

        elif samples is not None:
            print('Extracting features ...')
            self.features = extract_features_timefreq(
                samples, self.sfreq, self.timefreq_method, self.n_jobs)
            if sample_dict:  # record meta information
                self.features_meta['distances'] = sample_dict['y']
                for key in sample_dict.keys():
                    if key != 'x' and key != 'y':
                        self.features_meta[key] = sample_dict[key]

            # Store features as a dataset in the HDF5 file
            if feature_file is not None:
                with h5py.File(feature_file, 'w') as f:
                    f.create_dataset('features', data=self.features)
                    for key, item in self.features_meta.items():
                        f.create_dataset(key, data=item)

        else:
            raise Exception(
                'samples must be provided '
                'if features have not been extracted. \n'
                f'{feature_file} cannot be found.'
            )

    def fit(
            self,
            samples: Optional[np.ndarray]=None,
            sample_dict: Optional[dict]=None,
            feature_file: Optional[str]=None,
            verbose: bool=True
        ):
        """
        Fit the clustering model using the extracted features. \n
        The distances to seizure periods are treated as labels.

        Parameters
        ----------
        samples : np.ndarray, optional
            Array of shape (num_samples, num_channels, sample_length)
            representing the time series samples. \n
            Must be provided if feature_file is not given
            or features have not been extracted.
        sample_dict : np.ndarray
            Dictionary with meta-information relating
            to the samples, e.g. distance to seizure,
            sample id (original file)
        feature_file : str, optional
            If provided, the extracted features will be saved
            or read from the file. \n
            If the file exists, feature will be directly read from this folder.
            Otherwise, the extracted features will be saved to a
            h5 file.
        verbose : bool, optional
            If true, print the training processes
            and the cluster sizes
        """
        self.extract_features(
            samples, sample_dict, feature_file)

        if self.scaling:
            features = self.scaler.fit_transform(self.features)
        else:
            features = self.features
        
        if self.n_ica > 0 and self.n_ica < min(features.shape[0], features.shape[1]):
            if verbose:
                print(f'Reducing dimension to {self.n_ica} using ICA ...')
            features = self.ica.fit_transform(features)
        elif self.n_ica > 0:
            raise ValueError(
                f'n_ica ({self.n_ica}) cannot exceed total '
                f'number of features: {features.shape[1]} and '
                f'number of samples {features.shape[0]}.'
            )

        if verbose:
            print('fitting clustering model')
        self.labels = self.model.fit_predict(features)
        if verbose:
            self.get_cluster_sizes(verbose=True)

    def predict(
            self, new_samples: np.ndarray,
            sample_dict: Optional[dict]) -> np.ndarray:
        """
        Predict the cluster labels for new samples.

        Parameters
        ----------
        new_samples : np.ndarray
            Array of shape (num_samples, num_channels, sample_length)
            representing the new time series samples.
        sample_dict : np.ndarray
            Dictionary with meta-information relating
            to the samples, e.g. distance to seizure,
            sample id (original file)

        Returns
        -------
        np.ndarray
            1 dimensional array of predicted cluster labels for the new samples.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' first.")

        self.extract_features(new_samples, sample_dict)

        if self.scaling:
            features = self.scaler.transform(self.features)
        else:
            features = self.features

        if self.n_ica:
            features = self.ica.transform(features)

        self.labels = self.model.predict(features)

        return self.labels

    def get_cluster_sizes(self, verbose: bool=False):
        """
        Prints the sizes of each cluster based
        on the cluster labels stored in self.labels. \n
        """
        if not hasattr(self, 'labels'):
            raise AttributeError("Labels have not been computed yet.")

        unique_labels, counts = np.unique(self.labels, return_counts=True)

        cluster_sizes = dict(zip(unique_labels, counts))

        if verbose:
            for label, size in cluster_sizes.items():
                if label == -1:  # noise points in DBSCAN
                    print(f"Noise points (label = {label}): {size}")
                else:
                    print(f"Cluster {label}: {size} points")
            
        return cluster_sizes

    def get_cluster_dists(self, cluster_id: int) -> np.ndarray:
        """
        Return distances to seizure of samples under this cluster
        """
        cluster_indices = np.where(self.labels == cluster_id)[0]
        if 'distances' in self.features_meta:
            dists = self.features_meta['distances']
        else:
            print(f"Keys in features_meta: {self.features_meta.keys()}")
            raise ValueError(
                "Distances must be set before visualizing."
                )

        return dists[cluster_indices]

    def plot_clusters(self, file_path: Optional[str]=None) -> None:
        """
        Visualize the clustering results
        in 2D using t-SNE (t-Distributed Stochastic Neighbor Embedding)
        of the previous call of the model. \n
        (`fit` or `predict`)

        Parameter
        ---------
        file_path : str, optional
            If provided, the figure will be saved
            instead of shown.
        """
        if self.labels is None:
            raise ValueError(
                "Model is not fitted yet. "
                "Please call `fit` first.")
        
        # Reduce features to 2D for visualization
        tsne = TSNE(n_components=2, random_state=self.seed)

        if self.scaling:
            features = self.scaler.transform(self.features)
        else:
            features = self.features

        if self.n_ica:
            features = self.ica.transform(features)

        features_tsne = tsne.fit_transform(features)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            features_tsne[:, 0], features_tsne[:, 1],
            c=self.labels, cmap='viridis')

        # Add a color legend based on the scatter plot's colors
        unique_labels = np.unique(self.labels)
        handles = []
        for label in unique_labels:
            color = scatter.cmap(scatter.norm(label))
            handle = plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=color, markersize=10)
            handles.append(handle)

        plt.legend(
            handles,
            [f"Cluster {label}" for label in unique_labels],
            title="Clusters", loc='best')

        plt.title(f"tSNE Clustering Visualisation")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")

        if file_path:
            plt.savefig(file_path, dpi=300)
            print(f'figure saved to {file_path}')
            plt.close()
        else:
            plt.show()

    def plot_k_distance(
            self, feature_file: str, n_neighbors: int=5,
            file_path: Optional[str]=None
        ) -> None:
        """
        Used to judge the distance between points
        for determining a suitable value of eps.

        Parameters
        -----------
        feature_file : str
            The file in which extracted features are saved.
            It should contain a numpy array of shape
            (n_samples, n_features)

        n_neighbors : int, optional
            Number of neighbours to consider for
            each neighbourhood.

        file_path : str, optional
            If provided, the figure will be saved
            instead of shown.
        """
        self.extract_features(feature_file=feature_file)

        nn = NearestNeighbors(n_neighbors=n_neighbors)
        if self.scaling:
            features = self.scaler.fit_transform(self.features)
        else:
            features = self.features

        if self.n_ica:
            features = self.ica.fit_transform(features)

        nn.fit(features)
        distances, _ = nn.kneighbors(features)
        distances = np.sort(distances[:, -1], axis=0)

        # Plot the k-distance graph
        plt.plot(distances)
        plt.xlabel('Points')
        plt.ylabel(f'Distance to {n_neighbors}th nearest neighbor')
        plt.title('k-distance Graph')

        if file_path:
            plt.savefig(file_path)
            print(f'figure saved to {file_path}')
            plt.close()
        else:
            plt.show()

    def visualize_cluster_distances(
            self, file_path: Optional[str]=None) -> None:
        """
        Visualises the relation between cluster labels
        and distances to the closest seizure period.

        Parameter
        ---------
        file_path : str, optional
            If provided, the figure will be saved
            instead of shown.

        Notes
        -----
        np.inf distance values (no seizure in the recording)
        will be replaced with a value larger than the
        maximum distance for visualisation.
        """
        if self.labels is None:
            raise ValueError(
                "Labels must be set before visualizing."
                "run `fit` or `predict` before running this function"
                )

        if 'distances' in self.features_meta:
            dists = self.features_meta['distances']
        else:
            print(f"Keys in features_meta: {self.features_meta.keys()}")
            raise ValueError(
                "Distances must be set before visualizing."
                )

        max_distance = np.nanmax(
            dists[~np.isinf(dists)])
        dists_visual = np.where(
            np.isinf(dists),
            2*max_distance, dists)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.labels, dists_visual)

        # Labeling
        plt.title('Cluster Labels vs. Distance to Closest Seizure Period')
        plt.xlabel('Cluster Labels')
        plt.ylabel('Distance to Closest Seizure Period')

        if file_path:
            plt.savefig(file_path)
            print(f'figure saved to {file_path}')
            plt.close()
        else:
            plt.show()

    def plot_histogram_dists(
            self, cluster_ids: List[int], bins: int=30,
            file_path: Optional[str]=None
            ):
        """
        Plot the histogram of distances to the closest seizure period 
        for samples in the specified cluster.

        Parameters
        ----------
        cluster_ids : list of int or int
            The indices of the cluster(s) for
            which to plot the histogram of distances.
        bins : int, optional
            Number of bins used to plot the histogram.
        file_path : str, optional
            If provided, the figure will be saved
            instead of shown.

        Raises
        ------
        ValueError
            If the provided cluster_id is invalid (not found in self.labels).

        Notes
        -----
        For visualisation, np.inf values will not be plotted. \n
        But number of points with infinite distance
        in this cluster will be printed. 
        """
        if isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]

        all_valid_dists = []

        # Check if the cluster_id is valid
        for cluster_id in cluster_ids:
            if cluster_id not in np.unique(self.labels):
                raise ValueError(
                    f"Invalid cluster ID: {cluster_id}. "
                    "Please provide a valid cluster ID.")

            dists_in_cluster = self.get_cluster_dists(cluster_id)
            valid_dists = dists_in_cluster[np.isfinite(dists_in_cluster)]

            all_valid_dists.extend(valid_dists)
            if valid_dists.shape[0] < dists_in_cluster.shape[0]:
                print(
                    f'{dists_in_cluster.shape[0] - valid_dists.shape[0]} samples'
                    f' in cluster {cluster_id} come from files with no seizure record'
                )

        # Plot the histogram
        plt.figure(figsize=(8, 6))
        plt.hist(all_valid_dists, bins=bins, color='skyblue', edgecolor='black')
        plt.title("Histogram of Distances to Seizure Periods"+ "\n" +
                  f"for Clusters {cluster_ids}")
        plt.xlabel("Distance to Seizure (s)")
        plt.ylabel("Frequency")
        plt.grid(True)

        if file_path:
            plt.savefig(file_path, dpi=300)
            print(f'figure saved to {file_path}')
            plt.close()
        else:
            plt.show()

    def evaluate_cluster(self, cluster_id: int):
        """
        Analyses the presence of given cluster index (or indices)
        in each pre-ictal segment.
        
        Parameters:
        ----------
        cluster_ids : int
            The ID of the cluster to analyze in `self.labels`.

        Returns:
        -------
        None
            Prints the analysis results:
            number of samples in the cluster in each pre-ictal segment,
            proportion of pre-ictal segments in which the cluster is present,
            and distribution of distances.
        """
        if cluster_id not in np.unique(self.labels):
            raise ValueError(f"Cluster ID {cluster_id} not found in labels.")
        if self.labels is None or (
                'preictal_flag' not in self.features_meta):
            raise ValueError(
                "Labels and distances must be set before visualizing."
                "run `fit` or `predict` before running this function"
                )
        
        if 'distances' in self.features_meta:
            dists = self.features_meta['distances']
        else:
            print(f"Keys in features_meta: {self.features_meta.keys()}")
            raise ValueError(
                "Distances must be set before visualizing."
                )
        
        # Initialise variables for counting and analysis
        num_segments = 0
        num_segments_with_cluster = 0
        cluster_counts = []
        distance_values = []
        
        # Loop over consecutive segments of 1s in preictal_flags
        start_idx = None
        for i, flag in enumerate(self.features_meta['preictal_flag']):
            if flag == 1:
                if start_idx is None:
                    start_idx = i  # Start a new segment
            elif flag == 0 and start_idx is not None: # end of segment
                segment_labels = self.labels[start_idx:i]
                segment_dists = dists[start_idx:i]

                cluster_count = np.sum(segment_labels == cluster_id)
                cluster_counts.append(cluster_count)

                distance_values.extend(segment_dists[segment_labels == cluster_id])
                if cluster_count > 0:
                    num_segments_with_cluster += 1
                num_segments += 1
                
                # reset for a new segment
                start_idx = None
        
        # Handle edge case: last segment if it ends at the last index
        if start_idx is not None:
            segment_labels = self.labels[start_idx:]
            segment_dists = dists[start_idx:]
            
            cluster_count = np.sum(segment_labels == cluster_id)
            cluster_counts.append(cluster_count)
            distance_values.extend(segment_dists[segment_labels == cluster_id])
            
            if cluster_count > 0:
                num_segments_with_cluster += 1
            num_segments += 1
        
        if num_segments == 0:
            warnings.warn("No preictal segments found in `self.preictal_flags`.")
            return

        cluster_proportion = (num_segments_with_cluster / num_segments
                              if num_segments > 0 else 0)
        
        # Print results
        print(f"Total number of preictal segments: {num_segments}")
        print(f"Number of segments containing cluster {cluster_id}"
              f": {num_segments_with_cluster}")
        print(f"Proportion of segments containing cluster {cluster_id}"
              f": {cluster_proportion:.2f}")
        print(f"Cluster {cluster_id} sample counts in each segment: {cluster_counts}")
        
        # Print the distribution of distances for samples in the cluster
        distance_values = np.array(distance_values)
        if distance_values.size > 0:
            print(f"Distribution of distances for cluster {cluster_id}"
                  " samples in preictal segments:")
            print(f"Min: {np.min(distance_values)}, Max: {np.max(distance_values)}, "
                  f"Mean: {np.mean(distance_values)}, Std: {np.std(distance_values)}")
        else:
            print(f"No samples found for cluster {cluster_id} in any preictal segment.")