"""Differential geometry: metrics, connections and curvature."""

from relativisticpy.diffgeom.metric import (
    Metric,
    differential,
    geodesic_equations,
    metric_from_line_element,
)
from relativisticpy.tensors.simplification import simplify_component

__all__ = [
    "Metric",
    "differential",
    "geodesic_equations",
    "metric_from_line_element",
    "simplify_component",
]
