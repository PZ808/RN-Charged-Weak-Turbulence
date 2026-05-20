"""Background parameter / QNM-frequency identities."""
import numpy as np

from rn_cwt import bar, h, h0, iq, q, s, w


def test_h_l0_matches_module_constant():
    assert np.isclose(h(0), h0)


def test_h_real_for_small_charge():
    """For q << 1 and l >= 0, h(l) is essentially real (csqrt is complex-typed
    but the imaginary part is zero up to roundoff)."""
    for l in range(6):
        assert abs(h(l).imag) < 1e-14


def test_h_zimmerman_formula():
    """h_l = 1/2 + sqrt(1/4 + l(l+1) - q^2) (Zimmerman 2017 eq. 1)."""
    for l in [0, 1, 2, 4]:
        expected = 0.5 + np.sqrt(0.25 + l * (l + 1) - q**2)
        assert np.isclose(h(l).real, expected)


def test_w_and_s_consistency():
    """w(n,l) = -i * (n + h + iq) / 2 ;  s(n,l) = -(n + h + iq) / 2 ; w = -i * s_arg."""
    for n, l in [(0, 0), (1, 0), (3, 2), (5, 1)]:
        assert np.isclose(w(n, l), -1j * (n + h(l) + iq) / 2.0)
        assert np.isclose(s(n, l), -(n + h(l) + iq) / 2.0)


def test_w_imag_part_is_decay_rate():
    """For n >= 0, near-extremal RN QNMs decay: Im(w) < 0 (e^{-i w t} decays)."""
    for n in range(5):
        assert w(n, 0).imag < 0


def test_bar_is_conj():
    assert bar(1 + 2j) == 1 - 2j
    assert bar(complex(3.0, -4.0)) == complex(3.0, 4.0)
