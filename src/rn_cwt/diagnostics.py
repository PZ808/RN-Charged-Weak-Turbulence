"""Plot helpers and solution inspection (no scientific content)."""
import numpy as np


def mode_label(k):
    """Pretty label '{n=…, l=…, m=…}' for a mode dict."""
    return f"n={k['n']}, l={k['l']}, m={k.get('m', 0)}"


def inspect_solution(sol, modes):
    """Print success/range/initial-final amplitudes/total-power drift.

    Returns ``(t, A, amps, powers)`` for further plotting.
    """
    A = sol.y.T
    t = sol.t

    print("success:", sol.success)
    print("message:", sol.message)
    print("nfev:", sol.nfev)
    print("t range:", t[0], t[-1])
    print("A shape:", A.shape)

    amps = np.abs(A)
    powers = amps**2

    print("\ninitial amplitudes:")
    for a, k in zip(A[0], modes):
        print(mode_label(k), a, " |A|=", abs(a))

    print("\nfinal amplitudes:")
    for a, k in zip(A[-1], modes):
        print(mode_label(k), a, " |A|=", abs(a))

    print("\nmax |A| by mode:")
    for j, k in enumerate(modes):
        print(mode_label(k), np.max(amps[:, j]))

    print("\ntotal power:")
    print("initial:", np.sum(powers[0]))
    print("final:  ", np.sum(powers[-1]))
    print(
        "relative drift:",
        (np.sum(powers[-1]) - np.sum(powers[0])) / np.sum(powers[0]),
    )

    return t, A, amps, powers
