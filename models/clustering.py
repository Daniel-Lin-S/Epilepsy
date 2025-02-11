import numpy as np
from sklearn.base import ClusterMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Dict
import os
import h5py
import warnings

from utils.feature_extraction import extract_features_timefreq
from layers.normaliser import BatchNormaliser


class SeizureClustering:
    """
    A model for unsupervised clustering of time series data
    The model uses clustering algorithms to group the data into clusters.
    """

    def __init__(self, sfreq: int, timefreq_method: str='cwt',
                 model: Optional[ClusterMixin]=None, n_jobs: int = -1,
                 scaling_method: str='standard', n_ica: int=0,
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
        scaling_method : str, optional
            Method used for scaling the features. \n
            One of ['standard', 'none', 'patient']. \n
            - 'standard': StandardScaler from sklearn.preprocessing
            - 'none': No scaling applied
            - 'patient': normalise by patient id stored in self.features_meta['id']
            Default is 'standard'.
        n_ica : int, optional
            If a positive value given, ICA will be performed
            to reduce the dimension of the dataset.
        seed : int, optional
            The random seed used for training the clustering model
            to ensure reproducibility
        """
        self.sfreq = sfreq
        self.timefreq_method = timefreq_method
        self.n_ica = n_ica
        self.n_jobs = n_jobs
        self.seed = seed

        if model is None:
            self.model = AgglomerativeClustering(n_clusters=2)
        else:
            self.model = model

        self.scaling_method = scaling_method

        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'patient':
            self.scaler = BatchNormaliser()
        elif scaling_method == 'none':
            self.scaler = None
        else:
            raise ValueError(
                f"Scaling method {scaling_method} not recognised. "
                "Please provide one of ['standard', 'none', 'patient']"
            )
        
        if self.n_ica > 0:
            self.ica = FastICA(n_components=self.n_ica)

        # runtime attributes
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

        if self.scaling_method == 'standard':
            features = self.scaler.fit_transform(self.features)
        elif self.scaling_method == 'patient':
            features = self.scaler.fit_transform(
                self.features, self.features_meta['id']
            )
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

        if self.scaling_method == 'standard':
            features = self.scaler.transform(self.features)
        elif self.scaling_method == 'patient':
            features = self.scaler.transform(
                self.features, self.features_meta['id']
            )
        else:
            features = self.features

        if self.n_ica:
            features = self.ica.transform(features)

        self.labels = self.model.predict(features)

        return self.labels

    def get_cluster_sizes(
            self, verbose: bool=False
        ) -> Dict[int, int]:
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

        if self.scaling_method == 'standard':
            features = self.scaler.transform(self.features)
        elif self.scaling_method == 'patient':
            features = self.scaler.transform(
                self.features, self.features_meta['id']
            )
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

        plt.title(f"tSNE Clustering Visualisation", fontsize=20)
        plt.xlabel("t-SNE Component 1", fontsize=16)
        plt.ylabel("t-SNE Component 2", fontsize=16)

        if file_path:
            plt.savefig(file_path, dpi=300)
            print(f'figure saved to {file_path}')
            plt.close()
        else:
            plt.show()

    def plot_cluster_seizure_distances(
            self, file_path: Optional[str]=None) -> None:
        """
        Visualises the relation between cluster labels
        and distances to the closest seizure period.

        Parameter
        ---------
        file_path : str, optional
            If provided, the figure will be saved
            instead of shown.
        """
        if self.labels is None:
            raise ValueError(
                "Labels must be set before visualizing."
                "run `fit` or `predict` before running this function"
                )

        if 'distances' in self.features_meta:
            dists = self.features_meta['distances'] / self.sfreq
        else:
            print(f"Keys in features_meta: {self.features_meta.keys()}")
            raise ValueError(
                "Distances must be set before visualizing."
                )

        finite_mask = np.isfinite(dists)
        infinite_mask = ~finite_mask

        if np.all(infinite_mask):
            warnings.warn(
                'None of self.dists is finite. '
                'Histogram will not be plotted.'
            )
            return

        plt.figure(figsize=(10, 6))
        plt.scatter(self.labels[finite_mask], dists[finite_mask], alpha=0.7)

        # plot infinite points
        if np.any(infinite_mask):
            plt.scatter(self.labels[infinite_mask],
                np.full(np.sum(infinite_mask), np.max(dists[finite_mask]) * 1.5),
                color='red', marker='x', label="Infinite Distances", alpha=0.7)

        # Labeling
        plt.title('Cluster Labels vs. Distance to Closest Seizure Period', fontsize=20)
        plt.xlabel('Cluster Labels', fontsize=16)
        plt.ylabel('Time to Closest Seizure (s)', fontsize=16)
        plt.legend(loc='lower center')

        if file_path:
            plt.savefig(file_path)
            print(f'figure saved to {file_path}')
            plt.close()
        else:
            plt.show()

    def plot_distances_seizures(
            self, cluster_id: int,
            file_path: Optional[str]=None):
        """
        Generate a scatter plot of:
        - Distances to seizure (`self.features_meta['distances']`) vs.
        - Euclidean distance to the centroid of the largest cluster.

        Parameters
        ----------
        cluster_id : int
            The cluster whose samples will be plotted.
        file_path : str, optional
            If provided, the figure will be saved.
            Otherwise, the figure will be shown.

        Raises
        ------
        ValueError
            If the cluster_id is not found in self.labels.
        """
        cluster_sizes = self.get_cluster_sizes()
        if cluster_id not in cluster_sizes:
            raise ValueError(
                f"Invalid cluster ID {cluster_id}. "
                f"Available clusters: {list(cluster_sizes.keys())}")

        largest_cluster_id = max(cluster_sizes, key=cluster_sizes.get)

        # Extract samples in the target cluster
        cluster_mask = self.labels == cluster_id
        cluster_features = self.features[cluster_mask]
        cluster_dists = np.array(self.features_meta['distances'])[cluster_mask] / self.sfreq

        finite_mask = np.isfinite(cluster_dists)
        infinite_mask = ~finite_mask

        # Extract samples in the largest cluster
        largest_cluster_mask = self.labels == largest_cluster_id
        largest_cluster_features = self.features[largest_cluster_mask]

        centroid_largest_cluster = np.mean(largest_cluster_features, axis=0)

        euclidean_distances = cdist(
            cluster_features, centroid_largest_cluster.reshape(1, -1)
            ).flatten()

        # scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(
            cluster_dists[finite_mask],
            euclidean_distances[finite_mask], alpha=0.7, edgecolors='k')
        
        if np.any(infinite_mask):
            plt.scatter(
                np.full(
                    np.sum(infinite_mask),
                    np.max(cluster_dists[finite_mask]) * 1.1), 
                euclidean_distances[infinite_mask], color='red', marker='x', 
                label="Infinite Distances", alpha=0.7)

        plt.xlabel("Time to Closest Seizure (s)", fontsize=16)
        plt.ylabel("Distance to Largest Cluster", fontsize=16)
        plt.title(
            f"Cluster {cluster_id}: Seizure Distance vs. Distance to Largest Cluster",
            fontsize=20)
        plt.legend()
        plt.grid(True)

        if file_path:
            plt.savefig(file_path, dpi=300)
            print(f'figure saved to {file_path}')
            plt.close()
        else:
            plt.show()

    def evaluate_clusters(self):
        if self.labels is None:
            raise ValueError("Labels have not been computed yet.")

        if self.features is None:
            raise ValueError("Features have not been extracted yet.")

        unique_labels = np.unique(self.labels)
        num_clusters = len(unique_labels)

        within_distances = []
        between_distances = []

        for i in range(num_clusters):
            cluster_indices = np.where(self.labels == unique_labels[i])[0]
            cluster_features = self.features[cluster_indices]

            # Calculate within-cluster distance
            cluster_distance = np.mean(
                np.linalg.norm(
                    cluster_features - np.mean(cluster_features, axis=0), axis=1
                )
            )
            within_distances.append(cluster_distance)

            # Calculate between-cluster distance
            # by averaging the distance between the cluster centroid
            # and the centroid of all other clusters (combined)
            other_clusters_indices = np.where(self.labels != unique_labels[i])[0]
            other_clusters_features = self.features[other_clusters_indices]
            between_cluster_distance = np.mean(
                np.linalg.norm(
                    cluster_features - np.mean(other_clusters_features, axis=0),
                    axis=1
                )
            )
            between_distances.append(between_cluster_distance)

        # Calculate other metrics
        mean_within_distance = np.mean(within_distances)
        mean_between_distance = np.mean(between_distances)
        separation_index = mean_between_distance / mean_within_distance

        # Print evaluation metrics
        print("Cluster Distance Evaluation:")
        print(f"Number of clusters: {num_clusters}")
        print(f"Within-cluster distances: {within_distances}")
        print(f"Between-cluster distances: {between_distances}")
        print(f"Average within-cluster distance: {mean_within_distance}")
        print(f"Average between-cluster distance: {mean_between_distance}")
        print(f"Separation index: {separation_index}")
