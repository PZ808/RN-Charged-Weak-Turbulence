"""Angular overlap Q and Wigner-3j sanity checks."""
import numpy as np

from rn_cwt import Q_element, W3j


def test_W3j_trivial():
    assert W3j(0, 0, 0, 0, 0, 0) == 1.0


def test_W3j_magnetic_selection():
    assert W3j(1, 1, 0, 1, 0, 0) == 0.0


def test_W3j_triangle_inequality():
    # |j1 - j2| <= j3 <= j1 + j2; here j3 = 4 > 1 + 2 = 3.
    assert W3j(1, 2, 4, 0, 0, 0) == 0.0


def test_Q_element_all_zero_modes():
    """All four modes l=m=0: Y_00 = 1/sqrt(4 pi), so the integrand is (1/4pi)^2,
    and the integral over S^2 (= 4 pi) gives 1/(4 pi)."""
    val = Q_element(0, 0, 0, 0, 0, 0, 0, 0)
    assert np.isclose(val, 1.0 / (4.0 * np.pi))


def test_Q_element_magnetic_selection():
    """m_i + m_j must equal m_k + m_p."""
    assert Q_element(2, 1, 2, 0, 2, 0, 2, 0) == 0.0


def test_Q_element_symmetric_in_unbarred_pair():
    """Y_i Y_j is symmetric in (i, j), so Q^{ij}_{kp} = Q^{ji}_{kp}."""
    val_ij = Q_element(2, 1, 2, -1, 2, 0, 2, 0)
    val_ji = Q_element(2, -1, 2, 1, 2, 0, 2, 0)
    assert np.isclose(val_ij, val_ji)


def test_Q_element_symmetric_in_barred_pair():
    """bar(Y_k) bar(Y_p) is symmetric in (k, p), so Q^{ij}_{kp} = Q^{ij}_{pk}."""
    val_kp = Q_element(2, 1, 2, -1, 2, 1, 2, -1)
    val_pk = Q_element(2, 1, 2, -1, 2, -1, 2, 1)
    assert np.isclose(val_kp, val_pk)
