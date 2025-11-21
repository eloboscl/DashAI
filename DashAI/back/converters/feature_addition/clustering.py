from typing import Type, Union

from DashAI.back.converters.base_converter import BaseConverter
from DashAI.back.converters.category.feature_addition import FeatureAdditionConverter
from DashAI.back.core.schema_fields import BaseSchema, enum_field, schema_field
from DashAI.back.dataloaders.classes.dashai_dataset import (
    DashAIDataset,
    to_dashai_dataset,
)
from DashAI.back.models.unsupervised.scikit_learn.kmeans import KMeans


class ClusteringSchema(BaseSchema):
    selected_algorithm: schema_field(
        enum_field(["KMeans"]),
        "KMeans",
        "Clustering algorithm to use",
    )  # type: ignore


class Clustering(FeatureAdditionConverter, BaseConverter):
    """Converter that adds a cluster column to the dataset using
    any available model.
    """

    SCHEMA = ClusteringSchema
    DESCRIPTION = "Adds a column with the clusters assignations."
    CATEGORY = "Feature Addition"
    DISPLAY_NAME = "Clustering Feature"

    ALGORITHMS = {
        "KMeans": KMeans,
    }

    def __init__(self, algorithm_name: str = "KMeans", **kwargs):
        super().__init__()

        self.algorithm_name = algorithm_name
        if self.algorithm_name not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{self.algorithm_name}'")

        AlgorithmClass = self.ALGORITHMS[self.algorithm_name]

        self.model = AlgorithmClass(**kwargs)
        self.labels_ = None

    @classmethod
    def get_schema(cls):
        base_schema = super().get_schema()

        properties = base_schema.get("properties", {})
        algorithm_field = properties.get("selected_algorithm", {})
        default_algorithm = algorithm_field.get("placeholder")

        AlgorithmClass = cls.ALGORITHMS[default_algorithm]

        if hasattr(AlgorithmClass, "SCHEMA"):
            schema_algorithm = AlgorithmClass.SCHEMA.model_json_schema()

        for field_name, field_def in schema_algorithm.items():
            if field_name == "selected_algorithm":
                continue
            base_schema[field_name] = field_def

        return base_schema

    def fit(
        self, x: DashAIDataset, y: Union[DashAIDataset, None] = None
    ) -> Type["Clustering"]:
        """Fit the clustering model and store labels."""
        self.labels_ = self.model.fit_predict(x)

        return self

    def transform(
        self, x: DashAIDataset, y: Union[DashAIDataset, None] = None
    ) -> DashAIDataset:
        """Transform the dataset adding a cluster column."""
        if self.labels_ is None:
            raise RuntimeError("Need to call fit() before transform()")

        x_pandas = x.to_pandas()
        x_pandas["cluster"] = self.labels_

        return to_dashai_dataset(x_pandas)
