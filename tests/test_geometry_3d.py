"""
Tests for 3D/2D geometry sensing and story elevation detection in struct_load.
"""

import pytest
from struct_core import Node, StructuralModel
from struct_load.engine.geometry import (
    detect_model_dimensionality,
    detect_story_elevations,
    get_beam_elevation,
    get_beam_length,
    get_beam_span_and_orientation,
    is_horizontal_beam,
)
from helpers import build_3d_frame_model


def test_detect_dimensionality():
    model_3d = build_3d_frame_model()
    assert detect_model_dimensionality(model_3d) == "3d"

    # 2D frame (z=0 for all nodes)
    nodes_2d = [
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=6.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=3.5, z=0.0),
        Node(id=4, x=6.0, y=3.5, z=0.0),
    ]
    model_2d = StructuralModel(nodes=nodes_2d, elements=[])
    assert detect_model_dimensionality(model_2d) == "2d"


def test_detect_story_elevations_3d():
    model_3d = build_3d_frame_model()
    elevs = detect_story_elevations(model_3d)
    assert elevs == [3.5, 7.0]


def test_is_horizontal_beam_3d():
    model_3d = build_3d_frame_model()
    nodes_map = {n.id: n for n in model_3d.nodes}
    c1 = next(e for e in model_3d.elements if e.id == "C1")
    b1 = next(e for e in model_3d.elements if e.id == "B1")

    assert not is_horizontal_beam(c1, nodes_map, is_3d=True)
    assert is_horizontal_beam(b1, nodes_map, is_3d=True)
    assert get_beam_elevation(b1, nodes_map, is_3d=True) == 3.5

    length, orientation = get_beam_span_and_orientation(b1, nodes_map, is_3d=True)
    assert length == pytest.approx(6.0)
    assert orientation == "x"

    b2 = next(e for e in model_3d.elements if e.id == "B2")
    length2, orientation2 = get_beam_span_and_orientation(b2, nodes_map, is_3d=True)
    assert length2 == pytest.approx(5.0)
    assert orientation2 == "y"
