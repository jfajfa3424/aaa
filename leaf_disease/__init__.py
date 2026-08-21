"""8-class plant leaf-disease image classification.

This package is the single locked domain for the tutoring listing:
VGG / ResNet / MobileNet / EfficientNet (+ ResNet-50 with CBAM).
"""

from .config import CLASS_NAMES, NUM_CLASSES
from .models import MODEL_NAMES, build_model

__all__ = [
    "CLASS_NAMES",
    "NUM_CLASSES",
    "MODEL_NAMES",
    "build_model",
]
