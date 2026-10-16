"""
Shared fixtures and test helpers for struct_load tests.
"""

from pathlib import Path
import sys
import pytest

_tests_dir = str(Path(__file__).resolve().parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from helpers import build_3d_frame_model
from struct_core import StructuralModel


@pytest.fixture
def model_3d_frame() -> StructuralModel:
    return build_3d_frame_model()
