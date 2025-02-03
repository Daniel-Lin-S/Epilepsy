import numpy as np
from sklearn.base import ClusterMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from typing import Optional
import os
import h5py

from feature_extraction import extract_features_timefreq


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
        self.dists = None

    def extract_features(
            self, samples: Optional[np.ndarray]=None,
            dists: Optional[np.ndarray]=None,
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
        dists : np.ndarray, optional.
            Array of shape (num_samples,) representing the distance
            to the closest seizure for each sample. \n
            If not provided, will be read from h5 file
        feature_file : str, optional
            If provided, the extracted features and dists will be saved
            or read from the file. \n
            If the file exists, feature will be directly read from this folder.
            Otherwise, the extracted features will be saved to a
            h5 file.
        """
        if feature_file is not None and os.path.exists(feature_file):
            with h5py.File(feature_file, 'r') as f:
                self.features = f['features'][:]
                self.dists = f['distances'][:]
        elif samples is not None and dists is not None:
            self.features = extract_features_timefreq(
                samples, self.sfreq, self.timefreq_method, self.n_jobs)
            self.dists = dists
            
            if feature_file is not None:
                with h5py.File(feature_file, 'w') as f:
                    # Store features as a dataset in the HDF5 file
                    f.create_dataset('features', data=self.features)
                    f.create_dataset('distances', data=dists)
        else:
            raise Exception(
                'samples and dists must be provided '
                'if features have not been extracted. \n'
                f'{feature_file} cannot be found.'
            )

    def fit(
            self,
            samples: Optional[np.ndarray]=None,
            dists: Optional[np.ndarray]=None,
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
        dists : np.ndarray
            Array of shape (num_samples,) representing the distance
            to the closest seizure for each sample.
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
            samples, dists, feature_file)

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

    def predict_labels(
            self, new_samples: np.ndarray,
            dists: np.ndarray) -> np.ndarray:
        """
        Predict the cluster labels for new samples.

        Parameters
        ----------
        new_samples : np.ndarray
            Array of shape (num_samples, num_channels, sample_length)
            representing the new time series samples.
        dists : np.ndarray
            Array of shape (num_samples,) representing the distance
            to the closest seizure for each new sample.

        Returns
        -------
        np.ndarray
            1 dimensional array of predicted cluster labels for the new samples.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' first.")

        self.extract_features(new_samples, dists)

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

    def plot_clusters(self, file_path: Optional[str]=None) -> None:
        """
        Visualize the clustering results
        in 2D using t-SNE (t-Distributed Stochastic Neighbor Embedding)
        of the previous call of the model. \n
        (`fit` or `predict_labels`)

        Parameter
        ---------
        file_path : str, optional
            If provided, the figure will be saved
            instead of shown.
        """
        if self.labels is None:
            raise ValueError(
                "Model is not fitted yet. Please call 'fit' first.")
        
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
        if self.labels is None or self.dists is None:
            raise ValueError(
                "Labels and distances must be set before visualizing.")

        max_distance = np.nanmax(self.dists[~np.isinf(self.dists)])
        dists_visual = np.where(np.isinf(self.dists), 2*max_distance, self.dists)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.labels, dists_visual)

        # Labeling
        plt.title('Cluster Labels vs. Distance to Closest Seizure Period')
        plt.xlabel('Cluster Labels')
        plt.ylabel('Distance to Closest Seizure Period')

        if file_path:
            plt.savefig(file_path)
            print(f'figure saved to {file_path}')
        else:
            plt.show()

    def plot_histogram_dists(
            self, cluster_id: int, bins: int=30,
            file_path: Optional[str]=None
            ):
        """
        Plot the histogram of distances to the closest seizure period 
        for samples in the specified cluster.

        Parameters
        ----------
        cluster_id : int
            The ID of the cluster for which to plot the histogram of distances.
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

        # Check if the cluster_id is valid
        if cluster_id not in np.unique(self.labels):
            raise ValueError(
                f"Invalid cluster ID: {cluster_id}. "
                "Please provide a valid cluster ID.")

        cluster_indices = np.where(self.labels == cluster_id)[0]
        dists_in_cluster = self.dists[cluster_indices]
        valid_dists = dists_in_cluster[np.isfinite(dists_in_cluster)]
        if valid_dists.shape[0] < dists_in_cluster.shape[0]:
            print(
                f'{dists_in_cluster.shape[0] - valid_dists.shape[0]} samples'
                ' in this cluster come from files with no seizure record'
            )

        # Plot the histogram
        plt.figure(figsize=(8, 6))
        plt.hist(valid_dists, bins=bins, color='skyblue', edgecolor='black')
        plt.title(f"Histogram of Distances to Seizure Periods (Cluster {cluster_id})")
        plt.xlabel("Distance to Seizure (s)")
        plt.ylabel("Frequency")
        plt.grid(True)

        if file_path:
            plt.savefig(file_path, dpi=300)
            print(f'figure saved to {file_path}')
        else:
            plt.show()
