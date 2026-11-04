"""The descriptor nodes call other nodes' functions directly.

They used ``Node()._func``, an attribute of the old ``core`` API that aiflow
nodes no longer have, so both raised ``AttributeError`` whenever a structure
was given.
"""

import numpy as np
from ase.build import bulk

from aiflow_nodes.atomistic.structure.calc import (
    LinearInterpolationDescriptor,
    SplineDescriptor,
)


def _structure():
    return bulk("Al", cubic=True).repeat(2)


def test_spline_descriptor_returns_one_value_per_point():
    descriptor = SplineDescriptor(structure=_structure(), num_points=21).run()
    assert descriptor.shape == (21,)
    assert np.isfinite(descriptor).all()


def test_linear_interpolation_descriptor_returns_one_count_per_bin():
    r_bins = np.linspace(2.5, 7, 31)
    counts = LinearInterpolationDescriptor(structure=_structure(), r_bins=r_bins).run()
    assert counts.shape == (31,)
    assert counts.sum() > 0


def test_descriptors_without_structure_return_none():
    assert SplineDescriptor().run() is None
    assert LinearInterpolationDescriptor().run() is None
