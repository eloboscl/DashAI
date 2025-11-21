from typing import Final

from DashAI.back.converters.base_converter import BaseConverter


class FeatureAdditionConverter(BaseConverter):
    CATEGORY: Final[str] = "Feature Addition"
    COLOR: Final[str] = "rgb(147, 112, 219)"
