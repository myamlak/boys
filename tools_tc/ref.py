"""Independent high-precision reference for the Boys function.

F_n(x) = int_0^1 t^(2n) exp(-x t^2) dt

Two independent routes, cross-validated against each other:

  * ``F_quad``   - the integral, by mpmath's adaptive quadrature. This is the
                   definition, and nothing in this tree computes it.
  * ``F_series`` - the convergent series (Boys 1950; the form used as the
                   stable series in VikhamarSandberg 2025):

                       exp(-x)   _inf_      x^l
                 F_n = ------- * >     ------------
                          2     ---    (n + l + 1/2)
                                l=0

                   all terms positive, so no cancellation in the sum; the
                   only error is the truncation, reported by the caller's
                   tail floor.

``sanity_check`` compares them on the region-A span and returns the worst
disagreement, so the dense survey can lean on the fast series.
"""
import mpmath as mp

DPS = 50
TAIL = mp.mpf("1e-45")
MAX_TERMS = 4000


def F_series(n, x, dps=DPS, tail=TAIL, max_terms=MAX_TERMS):
    """The convergent series, all terms positive."""
    with mp.workdps(dps):
        x = mp.mpf(x)
        if x == 0:
            return mp.mpf(1) / (2 * n + 1)
        s = mp.mpf(0)
        term = mp.mpf(1) / (n + mp.mpf("0.5"))
        for l in range(max_terms):
            s += term
            if l > 20 and term < tail:
                break
            else:
                term *= x / (n + l + mp.mpf("1.5"))
        return mp.exp(-x) / 2 * s


def F_quad(n, x, dps=DPS, **kw):
    """The definition, by quadrature."""
    with mp.workdps(dps):
        x = mp.mpf(x)
        return mp.quad(lambda t: t ** (2 * n) * mp.exp(-x * t * t), [0, 1], **kw)


def sanity_check(points=None, dps=DPS):
    """Worst |series - quad| / |quad| over a spread of (n, x)."""
    if points is None:
        points = [(n, x) for n in (0, 1, 2, 5, 9, 17, 25, 32)
                  for x in ("0.001", "0.25", "1", "3", "5.94992407605424223",
                            "6.5", "10", "11.89", "20", "28.98")]
    worst = (mp.mpf(0), None)
    for (n, x) in points:
        a = F_series(n, x, dps=dps)
        b = F_quad(n, x, dps=dps)
        rel = abs(a - b) / abs(b) if b != 0 else abs(a - b)
        if rel > worst[0]:
            worst = (rel, (n, x))
    return worst


if __name__ == "__main__":
    rel, where = sanity_check()
    print(f"worst |series - quad| / |quad| = {mp.nstr(rel, 4)} at (n, x) = {where}")
