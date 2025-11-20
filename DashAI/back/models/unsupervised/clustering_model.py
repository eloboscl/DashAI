from typing import Any, Final

from DashAI.back.models.base_model import BaseModel


class ClusteringModel(BaseModel):
    """Base class for unsupervised clustering models.

    Provides a shared metadata dictionary.
    """

    TYPE: Final[str] = "Clustering"

    def __init__(self, **kwargs):
        """Save the metadata after running clustering."""
        super().__init__(**kwargs)
        self.metadata: dict[str, Any] = {}
