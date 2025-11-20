import joblib
import numpy as np

from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset
from DashAI.back.models.unsupervised.clustering_model import ClusteringModel


class SklearnLikeClusteringModel(ClusteringModel):
    """Class for handling sklearn like clustering models."""

    def save(self, filename: str) -> None:
        """Save the model in the specified path."""
        joblib.dump(self, filename)

    @staticmethod
    def load(filename: str) -> None:
        """Load the model of the specified path."""
        model = joblib.load(filename)
        return model

    def fit(self, x: DashAIDataset) -> "SklearnLikeClusteringModel":
        """Fits sklearn clustering model.

        Parameters
        ----------
        x: DashAIDataset
            Dataset containing the input features.

        Returns
        -------
        self
            The fitted model.
        """
        x_pandas = x.to_pandas()
        return super().fit(x_pandas)

    def predict(self, x: DashAIDataset) -> np.ndarray:
        """Assigns a cluster to each sample in x.

        Parameters
        ----------
        x: DashAIDataset
            Dataset containing the input features.

        Returns
        -------
        np.ndarray
            Array with the clusters assignments.
        """
        x_pandas = x.to_pandas()
        try:
            return super().predict(x_pandas)
        except AttributeError:
            if hasattr(self, "labels_"):
                return self.labels_
            raise AttributeError(
                "Not implement predict and labels_ is missing."
            ) from None

    def fit_predict(self, x: DashAIDataset) -> np.ndarray:
        """Fits the clustering model and return cluster assignments.

        Parameters
        ----------
        x: DashAIDataset
            Dataset containing the input features.

        Returns
        -------
        np.ndarray
            Cluster assignment for each sample.
        """
        x_pandas = x.to_pandas()

        if hasattr(super(), "fit_predict"):
            labels = super().fit_predict(x_pandas)
        else:
            super().fit(x_pandas)
            labels = self.predict(x)

        self.metadata = {"labels": labels}

        return labels
