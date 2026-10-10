"""
Shared fixtures and test helpers for struct_load tests.
"""

import pytest
from struct_core import (
    BeamElement,
    ElasticMaterial,
    Node,
    RectangularSection,
    StructuralModel,
)


def build_3d_frame_model() -> StructuralModel:
    """Build a 2-story 3D frame with Z as elevation."""
    # Bay X = 6.0m, Bay Y = 5.0m, Story heights Z = 3.5m, 7.0m
    nodes = [
        # Ground: Z = 0
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=6.0, y=0.0, z=0.0),
        Node(id=3, x=6.0, y=5.0, z=0.0),
        Node(id=4, x=0.0, y=5.0, z=0.0),
        # Story 1: Z = 3.5
        Node(id=5, x=0.0, y=0.0, z=3.5),
        Node(id=6, x=6.0, y=0.0, z=3.5),
        Node(id=7, x=6.0, y=5.0, z=3.5),
        Node(id=8, x=0.0, y=5.0, z=3.5),
        # Story 2 (Roof): Z = 7.0
        Node(id=9, x=0.0, y=0.0, z=7.0),
        Node(id=10, x=6.0, y=0.0, z=7.0),
        Node(id=11, x=6.0, y=5.0, z=7.0),
        Node(id=12, x=0.0, y=5.0, z=7.0),
    ]

    mat = ElasticMaterial(id="CONC", E=2.5e7, nu=0.2, rho=2400.0)
    col_sec = RectangularSection(id="COL", name="C400", b=0.4, h=0.4)
    bm_sec = RectangularSection(id="BEAM", name="B300", b=0.3, h=0.5)

    elements = [
        # Story 1 columns
        BeamElement(id="C1", start_node=1, end_node=5, material_id="CONC", section_id="COL"),
        BeamElement(id="C2", start_node=2, end_node=6, material_id="CONC", section_id="COL"),
        BeamElement(id="C3", start_node=3, end_node=7, material_id="CONC", section_id="COL"),
        BeamElement(id="C4", start_node=4, end_node=8, material_id="CONC", section_id="COL"),
        # Story 1 beams
        BeamElement(id="B1", start_node=5, end_node=6, material_id="CONC", section_id="BEAM"),
        BeamElement(id="B2", start_node=6, end_node=7, material_id="CONC", section_id="BEAM"),
        BeamElement(id="B3", start_node=7, end_node=8, material_id="CONC", section_id="BEAM"),
        BeamElement(id="B4", start_node=8, end_node=5, material_id="CONC", section_id="BEAM"),
        # Story 2 columns
        BeamElement(id="C5", start_node=5, end_node=9, material_id="CONC", section_id="COL"),
        BeamElement(id="C6", start_node=6, end_node=10, material_id="CONC", section_id="COL"),
        BeamElement(id="C7", start_node=7, end_node=11, material_id="CONC", section_id="COL"),
        BeamElement(id="C8", start_node=8, end_node=12, material_id="CONC", section_id="COL"),
        # Story 2 beams
        BeamElement(id="B5", start_node=9, end_node=10, material_id="CONC", section_id="BEAM"),
        BeamElement(id="B6", start_node=10, end_node=11, material_id="CONC", section_id="BEAM"),
        BeamElement(id="B7", start_node=11, end_node=12, material_id="CONC", section_id="BEAM"),
        BeamElement(id="B8", start_node=12, end_node=9, material_id="CONC", section_id="BEAM"),
    ]

    return StructuralModel(
        nodes=nodes,
        elements=elements,
        materials=[mat],
        sections=[col_sec, bm_sec],
    )


@pytest.fixture
def model_3d_frame() -> StructuralModel:
    return build_3d_frame_model()
