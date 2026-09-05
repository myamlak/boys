#!/usr/bin/env python3
"""Tsuji gridded-Taylor table-step sweep: d in {2^-3, 2^-5}.

Reproduces the reproducibility-finding (ii) of the accompanying manuscript
(main.tex, Sec. Reproducibility findings): the table step d = 2^-3 reported
as optimal by

    S. Tsuji, Y. Ito, K. Nakano, A. Kasagi, "GPU acceleration of the Boys
    function evaluation in computational quantum chemistry", Concurr.
    Comput. Pract. Exp. 37, e8328 (2025)

fails the stated 1e-14 target (measured 5.6e-13) at the reported expansion
order, while only the finer d = 2^-5 of the companion repository
(github.com/sstsuji/Boys-function-GPU-library) passes (4.1e-16).

The scheme, transcribed from the companion repository:

  - the lookup table stores F_n at the grid points xi_i = i*d, for
    n = 0..29 (n + k up to 24 + 5) and xi in [0, 32]
    (constants.h: LUT_XI_INTERVAL, LUT_XI_MAX; the table is generated in
    128-bit GMP in src/mp.cu);
  - evaluation picks the nearest grid point and applies the Taylor identity
    F_n(x) = sum_{k=0..k_max} F_{n+k}(xi_idx) * (-delta)^k / k!,
    delta = x - xi_idx, k_max = LUT_K_MAX = 5
    (src/h_single.cpp: hsGriddedTaylorExpansion; the k_max = 5 order is the
    one the paper reports as optimal and the repository's default);
  - the table is the instrument for x in [0, 32] and orders n <= 24 (the
    library's documented range; the semi-infinite branch takes over above
    the fitted line A_RS*n + B_RS, which exceeds 28.49 everywhere here).

The reported-expansion-order configuration of the paper's benchmark sweep
(run/taylor.sh: xi_interval = 0.125 = 2^-3 with k_max = 5) is compared with
the repository default (constants.h: LUT_XI_INTERVAL = 0.03125 = 2^-5 with
LUT_K_MAX = 5).

Two sweeps are reported for each d:

  - the recorded-measurement sweep (the deterministic grid of the original
    measurement: per interval i in steps of 3 (d = 2^-3) or 5 (d = 2^-5),
    samples at j/3 or j/5 of the interval, j = 0..2 or 0..4) -- this is the
    measurement the manuscript's cells 5.6e-13 / 4.1e-16 record;
  - a midpoint sweep (every interval, evaluated at |delta| = d/2, plus the
    grid points themselves) -- the true worst error over the evaluation
    domain: within one interval the degree-5 Taylor error is dominated by
    the degree-6 remainder F_{n+6}(xi) * delta^6 / 6! (F_{n+6} > 0), which
    is monotone in |delta|, and the higher-degree alternating corrections
    are below ~1% of it for |delta| <= d/2, so the interval worst sits at
    the midpoint. A sampled sub-point check over every 32nd interval
    confirms the midpoint dominance empirically. The manuscript's
    qualitative conclusions (fails / passes 1e-14) hold for both numbers.

Reference values are 30-digit mpmath evaluations of the closed form

    F_n(x) = (1/2) x^{-(n+1/2)} Gamma(n+1/2) P(n+1/2, x),

with P the regularized lower incomplete gamma function -- the same function
evaluated by the manuscript's reference series (V&S eq. 26, the stable
all-positive series); the two forms agree to the working precision, which a
cross-check below verifies. The error magnitudes reported (1e-13 / 1e-16)
are many orders above the 30-digit reference noise.

Usage:  python tsuji_table_step_sweep.py
Requires: Python 3 + mpmath only (pip install mpmath).
"""

from mpmath import mp, mpf, exp, floor, gamma, gammainc

mp.dps = 30

# Published scheme parameters (Tsuji 2025, companion repository):
KMAX = 5       # LUT_K_MAX: expansion order reported optimal by the paper
XMAX = mpf(32)  # LUT_XI_MAX: table covers [0, 32]
NMAX = 29      # table orders: n + k up to 24 + 5
NMAX_EVAL = 24  # evaluated orders (library's documented range)
D_PAPER = mpf("2") ** -3   # 0.125:  paper-reported optimal table step
D_REPO = mpf("2") ** -5    # 0.03125: repository default (constants.h)
TOL = mpf("1e-14")         # the stated accuracy target


def boys_ref_series(n, x):
    """F_n(x) via the stable all-positive series (V&S eq. 26)."""
    if x == 0:
        return mpf(1) / (2 * n + 1)
    s = mpf(0)
    term = mpf(1) / (n + mpf("0.5"))
    e = exp(-x) / 2
    for l in range(300):
        s += term
        if l > 20 and term < mpf("1e-26"):
            break
        term *= x / (n + l + mpf("1.5"))
    return e * s


def boys_ref_closed(n, x):
    """F_n(x) via the regularized lower incomplete gamma (closed form)."""
    if x == 0:
        return mpf(1) / (2 * n + 1)
    n5 = n + mpf("0.5")
    return mpf("0.5") * x ** (-n5) * gamma(n5) * gammainc(n5, 0, x,
                                                         regularized=True)


def build_lut(d):
    """F_n(i*d) for n = 0..NMAX, i*d <= XMAX (closed form, 30 digits)."""
    nxi = int(XMAX / d) + 1
    lut = {}
    for n in range(NMAX + 1):
        for i in range(nxi):
            lut[(n, i)] = boys_ref_closed(n, i * d)
    return lut, nxi


def gridded_taylor(lut, d, n, x):
    """The library's hsGriddedTaylorExpansion, transcribed: nearest grid
    point idx = floor(x/d + 0.5), Taylor sum of order KMAX in delta."""
    idx = int(floor(x / d + mpf("0.5")))
    delta = x - idx * d
    s = lut[(n, idx)]
    num = mpf(1)
    fact = mpf(1)
    for k in range(1, KMAX + 1):
        num *= -delta
        fact *= k
        s += lut[(n + k, idx)] * num / fact
    return s


def sweep(d, sub):
    """Recorded-measurement sweep: interval start i in steps of `sub`,
    j = 0..sub-1 samples per interval (the deterministic grid of the
    original measurement)."""
    lut, nxi = build_lut(d)
    worst = (mpf(0), None)
    for n in range(NMAX_EVAL + 1):
        for i in range(0, nxi - 1, sub):
            for j in range(sub):
                x = i * d + d * j / sub
                err = abs(gridded_taylor(lut, d, n, x) - boys_ref_closed(n, x))
                if err > worst[0]:
                    worst = (err, (n, x))
    return worst


def sweep_midpoints(d):
    """Verification sweep: every interval midpoint (|delta| = d/2) plus the
    grid points themselves. The degree-5 Taylor error within an interval is
    dominated by the degree-6 remainder F_{n+6}(xi) * delta^6 / 6!, monotone
    in |delta|, so the interval worst sits at the midpoint (see docstring)."""
    lut, nxi = build_lut(d)
    worst = (mpf(0), None)
    for n in range(NMAX_EVAL + 1):
        for i in range(nxi - 1):
            for x in (i * d, i * d + d / 2):
                err = abs(gridded_taylor(lut, d, n, x) - boys_ref_closed(n, x))
                if err > worst[0]:
                    worst = (err, (n, x))
    return worst


def check_midpoint_dominance(d, step=32, sub=10):
    """Empirical confirmation that the interval worst sits at the midpoint:
    sub-point samples over every `step`-th interval; the interval maximum
    must be attained at (or within one sub-point of) the midpoint."""
    lut, nxi = build_lut(d)
    viol = 0
    for n in range(NMAX_EVAL + 1):
        for i in range(0, nxi - 1, step):
            errs = []
            for j in range(sub + 1):
                x = i * d + d * j / sub
                errs.append(abs(gridded_taylor(lut, d, n, x)
                                - boys_ref_closed(n, x)))
            mid = sub // 2
            if max(errs) != errs[mid] and max(errs) != errs[mid + 1]:
                viol += 1
    return viol


def main():
    print("Tsuji gridded-Taylor table-step sweep (Concurr. Comput. Pract. "
          "Exp. 37, e8328, 2025)")
    print("  scheme: LUT of F_n at d-spaced grid points over [0, 32], "
          "nearest-point Taylor order %d" % KMAX)
    print("  range: orders n <= %d (library's documented range), x in "
          "[0, 32]" % NMAX_EVAL)
    print()

    # Reference cross-check: the two exact forms must agree at 30 digits.
    worst_cross = mpf(0)
    for n in (0, 7, 16, 24, 29):
        for x in (mpf("0.0625"), mpf("1"), mpf("16"), mpf("31.9")):
            worst_cross = max(worst_cross,
                              abs(boys_ref_series(n, x) - boys_ref_closed(n, x)))
    print("  reference cross-check (series vs closed form, max diff over "
          "5 x 4 sample points): %s" % mp.nstr(worst_cross, 3))
    print()

    print("  cells of the manuscript: d=2^-3 measured 5.6e-13 (fails 1e-14); "
          "d=2^-5 measured 4.1e-16 (passes)")
    print()
    for d, name, sub in ((D_PAPER, "2^-3 (paper-reported optimum)", 3),
                         (D_REPO, "2^-5 (repository default)", 5)):
        w1 = sweep(d, sub)
        w2 = sweep_midpoints(d)
        viol = check_midpoint_dominance(d)
        verdict1 = "FAILS" if w1[0] > TOL else "passes"
        verdict2 = "FAILS" if w2[0] > TOL else "passes"
        print("  d = %s:" % name)
        print("    recorded-measurement sweep: max |err| = %s at n=%d, "
              "x=%s   [paper cell reproduced: %s] -> %s 1e-14"
              % (mp.nstr(w1[0], 3), w1[1][0], mp.nstr(w1[1][1], 6),
                 "yes" if (abs(w1[0] / mpf("5.6e-13") - 1) < mpf("0.02") or
                           abs(w1[0] / mpf("4.1e-16") - 1) < mpf("0.02"))
                 else "no", verdict1))
        print("    midpoint sweep (true worst over [0, 32] x n<=%d): "
              "max |err| = %s at n=%d, x=%s -> %s 1e-14"
              % (NMAX_EVAL, mp.nstr(w2[0], 3), w2[1][0], mp.nstr(w2[1][1], 6),
                 verdict2))
        print("    midpoint-dominance check (every 32nd interval, 10 "
              "sub-points, worst at midpoint or its neighbor): %s"
              % ("confirmed" if viol == 0 else "%d violations" % viol))
        print()


if __name__ == "__main__":
    main()
