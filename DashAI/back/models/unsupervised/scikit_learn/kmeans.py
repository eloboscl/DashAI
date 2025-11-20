from pydantic import conint
from sklearn.cluster import KMeans as _Kmeans

from DashAI.back.core.schema_fields import BaseSchema, enum_field, schema_field
from DashAI.back.models.unsupervised.scikit_learn.sklearn_like_clustering_model import (
    SklearnLikeClusteringModel,
)


class KMeansSchema(BaseSchema):
    """K-Means is an unsupervised clustering method that groups data into
    K clusters based on distances.
    """

    n_clusters: schema_field(
        conint(ge=1),
        placeholder=3,
        description="Number of clusters to form as well as the number of "
        "centroids to generate.",
    )  # type: ignore
    init: schema_field(
        enum_field(enum=["k-means++", "random"]),
        placeholder="k-means++",
        description="Method for initialization: 'k-means++' selects initial "
        "cluster centers intelligently, 'random' chooses random initial centers.",
    )  # type: ignore
    max_iter: schema_field(
        conint(ge=1),
        placeholder=300,
        description="Maximum number of iterations of the k-means algorithm "
        "for a single run.",
    )  # type: ignore
    random_state: schema_field(
        conint(ge=0),
        placeholder=42,
        description="Random seed for reproducibility of results.",
    )  # type: ignore
    algorithm: schema_field(
        enum_field(enum=["auto", "full", "elkan"]),
        placeholder="auto",
        description="KMeans algorithm to use.",
    )  # type: ignore


class KMeans(SklearnLikeClusteringModel, _Kmeans):
    """Scikit-learn's K-Means clustering algorithm wrapper for DashAI."""

    SCHEMA = KMeansSchema
    DISPLAY_NAME: str = "K-Means"
    COLOR: str = "#4DB6AC"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
