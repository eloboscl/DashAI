from sklearn.cluster import KMeans as _Kmeans

from DashAI.back.models.unsupervised.scikit_learn.sklearn_like_clustering_model import (
    SklearnLikeClusteringModel,
)


class KMeans(SklearnLikeClusteringModel, _Kmeans):
    """Scikit-learn's K-Means clustering algorithm wrapper for DashAI."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
