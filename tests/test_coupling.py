"""Coupling-tensor shape, conventions, and Hermitian projection."""
import numpy as np

from rn_cwt import (
    build_C_notes,
    build_coupling_tensor,
    coupling_hermitian_defect,
    hermitian_adjoint_notes,
)


EXAMPLE_MODES = [
    {"n": 1, "l": 0, "m": 0},
    {"n": 2, "l": 0, "m": 0},
    {"n": 7, "l": 0, "m": 0},
    {"n": 11, "l": 0, "m": 0},
]


def test_C_shape_and_dtype():
    C = build_C_notes(EXAMPLE_MODES, alpha=1.0)
    N = len(EXAMPLE_MODES)
    assert C.shape == (N, N, N, N)
    assert C.dtype == complex


def test_build_coupling_tensor_alias_matches_build_C_notes():
    """`build_coupling_tensor` is a backwards-compat alias for `build_C_notes`."""
    assert build_coupling_tensor is build_C_notes


def test_C_H_invariant_under_notes_adjoint():
    """C_H = 0.5*(C + Cdag) must equal its own notes-convention adjoint."""
    C = build_C_notes(EXAMPLE_MODES, alpha=1.0)
    Cdag = hermitian_adjoint_notes(C)
    CH = 0.5 * (C + Cdag)
    CHdag = hermitian_adjoint_notes(CH)
    assert np.allclose(CH, CHdag, atol=1e-12, rtol=1e-12)


def test_C_A_anti_invariant_under_notes_adjoint():
    """C_A = 0.5*(C - Cdag) must be the negative of its adjoint."""
    C = build_C_notes(EXAMPLE_MODES, alpha=1.0)
    Cdag = hermitian_adjoint_notes(C)
    CA = 0.5 * (C - Cdag)
    CAdag = hermitian_adjoint_notes(CA)
    assert np.allclose(CA, -CAdag, atol=1e-12, rtol=1e-12)


def test_double_adjoint_is_identity():
    """(C^dag)^dag = C."""
    C = build_C_notes(EXAMPLE_MODES, alpha=1.0)
    Cdd = hermitian_adjoint_notes(hermitian_adjoint_notes(C))
    assert np.allclose(C, Cdd, atol=1e-14, rtol=1e-14)


def test_hermitian_defect_returns_nonnegative_triple():
    C = build_C_notes(EXAMPLE_MODES, alpha=1.0)
    ratio, normH, normA = coupling_hermitian_defect(C)
    assert ratio >= 0
    assert normH >= 0
    assert normA >= 0
