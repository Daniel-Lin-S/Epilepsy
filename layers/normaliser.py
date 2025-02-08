import numpy as np


class BatchNormaliser:
    """
    Normalise features based on feature ids. 
    e.g. if the features are grouped by patient id.
    """
    def __init__(self):
        self.means = {}  # Store means per batch
        self.stds = {}   # Store standard deviations per batch

    def fit_transform(
            self, features: np.ndarray, feature_ids: list
        ) -> np.ndarray:
        """
        Perform batch normalization on features, grouped by patient ID.

        Parameters
        ----------
        features : np.ndarray
            The feature matrix of shape (n_samples, n_features).
        feature_ids : list
            A list of patient IDs corresponding to each row in features.

        Returns
        -------
        np.ndarray
            The batch-normalized feature matrix, where each patient's data
            is normalized independently (mean=0, std=1 per patient).
        """
        normalized_features = np.zeros_like(features)
        unique_ids = np.unique(feature_ids)

        for patient_id in unique_ids:
            mask = np.array(feature_ids) == patient_id
            patient_features = features[mask]

            # Compute per-patient mean & std, store them for inversion
            mean = np.mean(patient_features, axis=0)
            std = np.std(patient_features, axis=0) + 1e-8  # Avoid division by zero

            self.means[patient_id] = mean
            self.stds[patient_id] = std

            normalized_features[mask] = (patient_features - mean) / std

        return normalized_features

    def transform(self, features: np.ndarray, feature_ids: list) -> np.ndarray:
        """
        Normalize features using previously stored mean and std.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        feature_ids : list
            List of patient IDs corresponding to each row in features.

        Returns
        -------
        np.ndarray
            The normalized feature matrix.

        Raises
        ------
        ValueError
            If a patient ID is encountered that was not seen in `fit_transform()`.
        """
        normalized_features = np.zeros_like(features)

        for i, patient_id in enumerate(feature_ids):
            if patient_id not in self.means:
                raise ValueError(
                    f"Patient ID {patient_id} was not seen in fit_transform(). "
                     "Cannot normalize.")
            
            mean = self.means[patient_id]
            std = self.stds[patient_id]

            # Normalize using trained mean & std
            normalized_features[i] = (features[i] - mean) / std

        return normalized_features

    def inverse_transform(
            self, normalized_features: np.ndarray, feature_ids: list
        ) -> np.ndarray:
        """
        Reverse the batch normalization process to recover original feature values.

        Parameters
        ----------
        normalized_features : np.ndarray
            The batch-normalized feature matrix.
        feature_ids : list
            A list of patient IDs corresponding to each row in normalized_features.

        Returns
        -------
        np.ndarray
            The original feature matrix (before normalization).
        """
        original_features = np.zeros_like(normalized_features)
        unique_ids = np.unique(feature_ids)

        for patient_id in unique_ids:
            mask = np.array(feature_ids) == patient_id
            if patient_id in self.means and patient_id in self.stds:
                # x_original = x_normalized * std + mean
                original_features[mask] = (
                    normalized_features[mask] * self.stds[patient_id]
                    ) + self.means[patient_id]

        return original_features
