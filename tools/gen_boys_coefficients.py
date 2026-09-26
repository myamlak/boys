#!/usr/bin/env python3
"""Generate the Boys-function fit tables and test reference data.

Reproduces include/boys/boys_coefficients.hpp (double and float lanes) and
tests/data/boys_reference.csv from scratch, validating every fit
against a 30-digit mpmath reference. Requires: `pip install mpmath`.

The two outputs carry two different precisions, deliberately: the fitted
tables are validated at 30 digits (their bar is a 5e-14 fit tolerance), while
the reference grid is written at GRID_DIGITS = 45 significant digits -
decisively beyond the widest long double any platform provides (binary128 is
34.0 decimal digits; see the constants below). The grid is an oracle, not a
transcript of the generating machine, and its value column is F_n AT the
double in its own x column, so a reader's comparison against it measures the
implementation rather than the host's floating-point width.

Method (the formula sources are keyed to CITATION.bib):
  - F_n itself: Boys1950 (Proc. R. Soc. Lond. A 200, 542-554, 1950).
  - Region A [0, x0): per-order piecewise Chebyshev fits evaluated by a split
    Clenshaw recurrence (even/odd, T_{2j}(t) = T_j(v), T_{2j+1}(t) = t*D_j(v),
    v = 2t^2 - 1). The fits are WEIGHTED: the downward batch recursion
    amplifies the seed error by up to x^n / prod_{j=0}^{n-1}(j + 1/2)
    (VikhamarSandberg2026, eqs. 18-23), so the
    effective seed tolerance is tol / max(1, weight(x)).
  - Region B [x0, x1): one F0 fit + the upward recursion (Shavitt1963,
    Methods in Computational Physics 2, 1-45, 1963).
  - Extended band [XNEW0, x0): a second F0 fit + the same upward recursion,
    dispatched per kmax tier (the per-range seed design; the certified
    per-kmax thresholds kTierThresholds come from the interval instrument).
  - Region C [x1, inf): asymptotic, no coefficients.
  - Boundaries: fixed kmax=32 values x0 = 11.899848152108484,
    x1 = 28.989337738820740 (the configuration validated end-to-end against
    the reference grid; the per-kmax table is printed for documentation).
    The published per-kmax boundary formula (the table's comparison column)
    is VikhamarSandberg2026, Eqs. 25/13.
  - Float lane: same structure, tolerance 1e-7, degree cap 10.
  - Region B carries a second, certified route beside the Chebyshev one: a
    rational minimax fit, P(t)/Q(t), fitted by the Remez exchange over the
    same interval and in the same mapped argument, and cross-checked against
    Lawson's algorithm (with the Sanathanan-Koerner denominator weight). The
    region-B routes' stored counts and delivered errors are measured in the
    kernel's double arithmetic against the same reference the fits are
    validated against, and emitted beside the tables.

The DCT normalization is the standard one: c_0 gets 1/(deg+1), all other
coefficients - including the top one - get 2/(deg+1). An earlier normalization
bug (halving c_deg too) made fits diverge - see the design doc. The committed
header must match this script's output byte for byte for a given tolerance
configuration.
"""
import argparse
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
from functools import lru_cache

try:
    from mpmath import mp, mpf, cos, exp, pi
except ImportError:
    if "--check" in sys.argv:
        # The regeneration ctest: CI runs without the pip package skip.
        print("SKIP: mpmath is not installed (the --check mode requires it)")
        sys.exit(77)
    raise

mp.dps = 30  # fit-path reference error ~1e-25; 4 orders below the double floor 1e-14

# The stable series' two truncations. The fit path keeps these defaults (its
# bar is TOL_DOUBLE, so a 1e-26 tail is already 12 orders of margin); the
# committed reference GRID passes the tighter pair below, because its job is
# different: it is read back by every platform's long double and compared
# against it, so its own accuracy must exceed the widest of them.
REF_TAIL_FLOOR = mpf("1e-26")
REF_TERMS = 300

# The committed reference grid as an ORACLE rather than a transcript of the
# machine that generated it. Written significant digits must exceed what any
# platform's long double can produce, or the cross-check measures the host's
# floating-point width instead of the implementation:
#   binary128  (gcc/clang long double on aarch64, the linux-arm64 leg) 113-bit
#              mantissa = 34.0 decimal digits;
#   x87        (gcc/clang long double on x86_64) 64-bit = 19.3 digits;
#   MSVC       long double IS double: 53-bit = 15.95 digits.
# GRID_DIGITS = 45 therefore sits 11 digits past the widest of the three, and
# GRID_DPS = 60 with a 1e-55 tail floor leaves ~15 digits of headroom over the
# digits actually written, so the last written digit is value, not truncation.
GRID_DPS = 60
GRID_TAIL_FLOOR = mpf("1e-55")
GRID_TERMS = 2000
GRID_DIGITS = 45

# Arguments per interval the scheme report sweeps when it measures what each
# evaluation scheme delivers: an interval sweep finer than the reference grid's
# own spacing on the same range, so the figure a row carries is at or above
# what any grid in the tree can find.
SCHEME_MEASURE_POINTS = 1024

# The names the scheme report prints; the C++ enumeration carries the same two
# in the same order.
SCHEME_NAMES = ("split-clenshaw", "horner")
LANE_NAMES = ("region A", "region B", "extended band")

TOL_DOUBLE = mpf("5e-14")
TOL_FLOAT = mpf("1e-7")
X0 = mpf("11.899848152108484")
X1 = mpf("28.989337738820740")
MAX_ORDER = 32
MAX_DEG_DOUBLE = 18
MAX_DEG_FLOAT = 10

# The leftmost region-A band is fitted at its own degree rather than the
# shared cap: its a-priori truncation bound at 18, the minimum of
# 2 |F_n(c - h (rho + 1/rho)/2)| rho^-deg / (rho - 1) over rho > 1, is
# 6.47e-16 for F0 and 2.18e-16..6.47e-16 across F1..F32 - above the
# 4.112e-17 truncation tail the seed design targets, for every order. At
# degree 20 the whole family sits at 1.14e-18..3.14e-18. Even degrees only:
# the split Clenshaw reads the odd coefficients up to c[deg-1].
FIRST_BAND_DEG = 20

# The extended band (the per-range seed design): a second F0 fit serving
# [XNEW0, X0) with the region-B-style upward recursion, dispatched per kmax
# tier (4/8/16/32). The boundaries are certified by the interval instrument
# (the interval instrument) under the band's range-uniform
# a-priori seed bound delta_0' = 1.0641e-15 (R_hat 1.0116e-15 forward
# rounding + tau 5.2538e-17 truncation tail, no x-sampling) and hardcoded
# below as the dispatch constants; the fit tolerance is the same 5e-14
# class as the shipped region-B fit, so the delivered seed width keeps the
# recursion envelope <= 5e-14 from each per-kmax boundary up.
XNEW0 = mpf("1.0855252345349333")
EXTENDED_DEG_LADDER = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72, 96)

# The certified per-kmax boundaries of the extended band (kmax 4/8/16/32),
# as certified by the interval instrument under the band's range-uniform
# a-priori seed bound R_hat + tau = 1.0641375040661759464e-15 (R_hat the
# evaluation-rounding sum over the seed's coded roundings, tau the
# polynomial-definition term). tau is computed rather than typed:
# (u/2)*sum_j|c_j| for the stored-coefficient rounding plus Tail_proj for
# truncation and aliasing, Tail_proj the Chebyshev interpolant bound
# 4*M(rho)*rho^-d/(rho-1) minimised over rho (Trefethen2019, ch. 8): the
# coefficients interpolate at Chebyshev-Gauss nodes, where the aliasing is
# at most the size of the tail, so the interpolant constant applies rather
# than the older (1 + Lambda_d) form, which charged the aliasing the
# worst-case operator norm a decaying tail never attains and was 2.0059x
# too loose.
# The bound covers the SEED's own definition and evaluation error, NOT the
# error a caller receives: the crossing condition charges the upward
# recursion and the asymptotic tail separately.
# The stored values ARE the kernel's dispatch constants: the next doubles
# above the certified crossings, so the dispatched region is a subset of the
# certified region. --check reproduces them exactly.
TIER_BOUNDARIES_CERTIFIED = [
    mpf("1.0855252345349333"),  # kmax 4: the band's left edge (the crossing clamps there;
                                # the true failure boundary is at or below the fit's
                                # lower edge - a conservative certified lower bound)
    mpf("2.015297705335114"),   # kmax 8: the 1-ulp-exp certified crossing
    mpf("4.897870299825657"),   # kmax 16: the 1-ulp-exp certified crossing
    mpf("10.783587858916762"),  # kmax 32: the 1-ulp-exp certified crossing
]


@lru_cache(maxsize=1 << 20)
def boys_ref(n, x, tail_floor=REF_TAIL_FLOOR, max_terms=REF_TERMS):
    """F_n(x) via the provably stable series (V&S eq. 26): all terms positive.

    tail_floor is the term magnitude the series breaks at (after l > 20) and
    max_terms its hard cap; the defaults are the fit path's, the grid passes
    its own (GRID_TAIL_FLOOR/GRID_TERMS). x must carry at least as many digits
    as the caller wants back: the grid's argument is an exact double (see
    write_reference), so nothing of x is lost at GRID_DPS.
    """
    if x == 0:
        return mpf(1) / (2 * n + 1)
    s = mpf(0)
    term = mpf(1) / (n + mpf("0.5"))
    e = exp(-x) / 2
    for l in range(max_terms):
        s += term
        if l > 20 and term < tail_floor:
            break
        term *= x / (n + l + mpf("1.5"))
    return e * s


def grid(a, b, npts):
    return [a + (b - a) * i / npts for i in range(npts + 1)]


def cheb_coeffs(n, a, b, deg):
    """Chebyshev interpolation coefficients at the zeros of T_{deg+1}."""
    mid = (a + b) / 2
    half = (b - a) / 2
    nodes = [mid + half * cos(pi * (k + mpf("0.5")) / (deg + 1)) for k in range(deg + 1)]
    values = [boys_ref(n, x) for x in nodes]
    coeffs = []
    for j in range(deg + 1):
        s = mpf(0)
        for k in range(deg + 1):
            s += values[k] * cos(pi * j * (k + mpf("0.5")) / (deg + 1))
        factor = mpf(1) / (deg + 1) if j == 0 else mpf(2) / (deg + 1)
        coeffs.append(factor * s)
    return coeffs


def clenshaw_double(cs, x, a, b):
    """Split Clenshaw in IEEE double, algebraically identical to the C++ split
    Clenshaw (which uses FMA throughout)."""
    t = 2.0 * (x - a) / (b - a) - 1.0
    if len(cs) == 1:
        return cs[0]
    if len(cs) == 2:
        return cs[0] + t * cs[1]
    v = 2.0 * t * t - 1.0
    two_v = v + v
    # Even coefficients: cs[0], cs[2], ...; odd: cs[1], cs[3], ...
    even_cs = cs[0::2]
    odd_cs = cs[1::2]
    m_even = len(even_cs) - 1
    b1 = even_cs[m_even]
    b2 = 0.0
    for k in range(m_even - 1, 0, -1):
        b1, b2 = even_cs[k] + two_v * b1 - b2, b1
    even = even_cs[0] + v * b1 - b2
    m_odd = len(odd_cs) - 1
    if m_odd <= 0:
        return even + t * odd_cs[0]
    o1 = odd_cs[m_odd]
    o2 = 0.0
    for k in range(m_odd - 1, 0, -1):
        o1, o2 = odd_cs[k] + two_v * o1 - o2, o1
    odd = odd_cs[0] + (two_v - 1.0) * o1 - o2
    return even + t * odd


def route_step(fused, a, b, c):
    """One multiply-add as the kernel's arithmetic spells it: the fused step
    rounds the sum once, the separate step rounds the product and then the
    sum."""
    return math.fma(a, b, c) if fused else a * b + c


def clenshaw_route(cs, x, a, b, fused):
    """The split Clenshaw recurrence in IEEE double with every step written as
    the backend's multiply-add.

    clenshaw_double is the fitting instrument and is left alone; this is the
    measurement instrument, so a figure taken with it is the arithmetic the
    kernel makes rather than the arithmetic that decided the fit."""
    t = 2.0 * (x - a) / (b - a) - 1.0
    if len(cs) == 1:
        return cs[0]
    if len(cs) == 2:
        return route_step(fused, t, cs[1], cs[0])
    v = route_step(fused, 2.0, t * t, -1.0)
    two_v = v + v
    even_cs = cs[0::2]
    odd_cs = cs[1::2]
    m_even = len(even_cs) - 1
    b1 = even_cs[m_even]
    b2 = 0.0
    for k in range(m_even - 1, 0, -1):
        b1, b2 = route_step(fused, two_v, b1, even_cs[k] - b2), b1
    even = route_step(fused, v, b1, even_cs[0] - b2)
    m_odd = len(odd_cs) - 1
    if m_odd <= 0:
        return route_step(fused, t, odd_cs[0], even)
    o1 = odd_cs[m_odd]
    o2 = 0.0
    for k in range(m_odd - 1, 0, -1):
        o1, o2 = route_step(fused, two_v, o1, odd_cs[k] - o2), o1
    odd = route_step(fused, two_v - 1.0, o1, odd_cs[0] - o2)
    return route_step(fused, t, odd, even)


def cheb_to_monomial(cs):
    """The monomial coefficients of the polynomial the Chebyshev coefficients
    spell, ascending, converted in exact arithmetic.

    T_0 = 1, T_1 = t, T_{k+1} = 2 t T_k - T_{k-1}, accumulated on the mpf
    coefficients, so the single rounding is the one that stores the result: the
    Chebyshev table and the monomial table are then the correctly rounded
    doubles of one ideal polynomial, rather than a rounded table converted and
    rounded a second time."""
    n = len(cs)
    mono = [mpf(0)] * n
    prev = [mpf(0)] * n
    prev[0] = mpf(1)  # T_0
    cur = [mpf(0)] * n
    if n > 1:
        cur[1] = mpf(1)  # T_1
    for k in range(n):
        if cs[k]:
            for j in range(n):
                mono[j] += cs[k] * prev[j]
        nxt = [mpf(0)] * n
        if k + 1 < n:
            for j in range(n - 1):
                nxt[j + 1] = 2 * cur[j]
            for j in range(n):
                nxt[j] -= prev[j]
        prev, cur = cur, nxt
    return mono


def horner_double(ms, t, fused):
    """Horner on the monomial form, ascending coefficients, evaluated at the
    mapped argument t in [-1, 1] so the conversion stays benign."""
    acc = ms[-1]
    for k in range(len(ms) - 2, -1, -1):
        acc = route_step(fused, acc, t, ms[k])
    return acc


def scheme_bound(v):
    """A swept maximum published as a bound: the smallest power of two
    strictly above it.

    A measured extreme is a reading and not a bound - a different grid reaches
    past it - so the number the header publishes is this round-up. A row that
    judges a sweep against it then compares a measurement with a value no grid
    can reach by sampling differently, which is what keeps the row's verdict
    about the scheme rather than about the two grids' luck. The round-up is at
    most a factor of two."""
    if v <= 0.0:
        return 0.0
    return math.ldexp(1.0, math.frexp(v)[1])


def fit_delivered(cs, ms, n, a, b, npts):
    """The worst |F_hat - F_n| each scheme reaches over [a, b], in each of the
    two multiply-add routes, measured against the reference at every argument
    and never against either fit's own residual.

    The reference column is built once per argument, so the four figures differ
    only by the summation and the arithmetic that carried it. Returns
    [[scheme][route]] with route 0 fused and 1 separate."""
    af = float(a)
    bf = float(b)
    worst = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(npts + 1):
        x = af + (bf - af) * (i / npts)
        if x >= bf:
            x = math.nextafter(bf, 0.0)  # the pieces are half-open at b
        ref = float(boys_ref(n, x))
        t = 2.0 * (x - af) / (bf - af) - 1.0
        cand = (
            (clenshaw_route(cs, x, af, bf, True), clenshaw_route(cs, x, af, bf, False)),
            (horner_double(ms, t, True), horner_double(ms, t, False)),
        )
        for scheme in (0, 1):
            for route in (0, 1):
                err = abs(cand[scheme][route] - ref)
                if err > worst[scheme][route]:
                    worst[scheme][route] = err
    return worst


def seed_weight(n, x):
    """Downward-recursion amplification bound: max(1, x^n / prod(j+1/2))."""
    p = mpf(1)
    for j in range(n):
        p *= x / (j + mpf("0.5"))
    return max(mpf(1), p)


def fit_interval(n, a, b, tol, maxdeg, weighted, degs=None):
    """Returns (deg, [float coeffs], [float monomial coeffs]) or None;
    weighted=True fits region-A seeds. The monomial form is carried alongside
    every accepted fit so the two evaluation schemes share one degree, one
    interval and one set of pieces. degs overrides the degree ladder (the
    extended band's fit walks its own)."""
    fmax = max(abs(boys_ref(n, x)) for x in grid(a, b, 8))
    if weighted:
        if max(fmax * seed_weight(n, x) for x in grid(a, b, 8)) < tol:
            return (0, [0.0], [0.0])
    elif fmax < tol:
        return (0, [0.0], [0.0])
    if degs is None:
        degs = (12, maxdeg) if maxdeg >= 12 else (maxdeg,)
    for deg in degs:
        cm = cheb_coeffs(n, a, b, deg)
        cd = [float(c) for c in cm]
        ok = True
        for x in grid(a, b, 8):
            err = abs(clenshaw_double(cd, x, a, b) - float(boys_ref(n, x)))
            budget = tol / seed_weight(n, x) * mpf("0.5") if weighted else tol
            if err > budget:
                ok = False
                break
        if ok:
            return (deg, cd, [float(c) for c in cheb_to_monomial(cm)])
    return None


def fit_order(n, region_b=False, f32=False):
    tol = TOL_FLOAT if f32 else TOL_DOUBLE
    maxdeg = MAX_DEG_FLOAT if f32 else MAX_DEG_DOUBLE
    if region_b:
        return fit_interval(0, X0, X1, tol, maxdeg, weighted=False)
    pieces = []
    stack = [(mpf(0), X0, 0)]
    while stack:
        a, b, depth = stack.pop()
        # The leftmost *band* carries its own degree (see FIRST_BAND_DEG);
        # every later band walks the default ladder, and so do the float-lane
        # bands, whose 1e-7 class is a different target. b < X0 keeps the
        # unsplit region on the default ladder: at 20 an order can pass there
        # without splitting, which would merge the two bands into one.
        degs = (FIRST_BAND_DEG,) if (not f32 and a == 0 and b < X0) else None
        r = fit_interval(n, a, b, tol, maxdeg, weighted=True, degs=degs)
        if r is None:
            if depth > 40:
                raise RuntimeError(f"F{n}: fit never converged on [{a},{b}]")
            m = (a + b) / 2
            stack.append((m, b, depth + 1))
            stack.append((a, m, depth + 1))
        else:
            deg, cd, mono = r
            pieces.append((float(a), float(b), deg, cd, mono))
    pieces.sort(key=lambda p: p[0])
    return pieces


# ---------------------------------------------------------------------------
# The a-priori truncation bound, and the partition it derives
# ---------------------------------------------------------------------------
# The bands this script ships were placed by bisecting on a sampled residual:
# fit_interval accepts an interval when a 9-point sweep of the fitted series
# against boys_ref is inside the budget, and fit_order splits at the midpoint
# when it is not. That answers "is this good enough". The bound below answers
# the other question the same mathematics answers - "how wide may a band be, at
# what degree, to reach a target" - and it does it without sampling anything.
#
# **The bound.** F_n is entire, and its integral representation gives an exact
# majorant on every vertical line of the complex plane:
#
#     |F_n(z)| = |int_0^1 t^(2n) e^(-z t^2) dt|
#              <= int_0^1 t^(2n) e^(-Re(z) t^2) dt = F_n(Re z),
#
# the triangle inequality applied to the integrand, and F_n decreases in its
# real argument. The Bernstein ellipse of [a, b] at parameter rho has its
# leftmost point at c - (h/2)(rho + 1/rho), with c = (a + b)/2 and
# h = (b - a)/2, so
#
#     M(rho) = max_{z on E_rho} |F_n(z)| <= F_n(c - (h/2)(rho + 1/rho))
#
# with no sampling and no asymptotic regime, and no appeal to the leftmost
# point being where |F_n| is largest: the majorant holds everywhere on the
# ellipse and the ellipse's least real part is at that point. Trefethen's
# interpolant bound (Approximation Theory and Approximation Practice, ch. 8)
# then gives
#
#     E(order, a, b, deg) = min_{rho > 1} K * M(rho) * rho^-deg / (rho - 1),
#
# with K = 2, the interpolant bound's own constant. Every rho gives an upper
# bound on the degree-deg interpolant's truncation error over [a, b], so their
# minimum is one too.
#
# The extended band's seed design doubled that constant to 4 for its own
# purposes (its comment says why: it charges the aliasing at the tail's own
# size rather than at the operator norm a decaying tail never attains). The
# derived partition below is computed at the design law's own constant, and
# BOUND_K is a parameter rather than a literal so that the doubled reading can
# be taken as well: a partition derived at 4 is a partition derived against a
# looser bound, so it is narrower everywhere and reaches its target with room.
#
# **What E does not bound, and why a derived table is still measured.** E is
# the truncation of the fit. It is not the rounding of the stored coefficients,
# and it is not any amplification a recursion downstream applies to the fit: a
# piece that seeds a downward recursion is held to a budget divided by that
# recursion's gain, which is a different criterion from the values alone - the
# weighted budget region_a_piece_budget applies, and the reason region A's
# pieces are narrower than its published bar would ask for. So a partition
# derived here is a partition of the FIT, its stored counts are what the fit
# costs, and the error a caller receives is measured separately rather than
# read off E.
BOUND_K = 2


def boys_entire(order, z):
    """F_order(z) for complex z, by the entire series sum_k (-z)^k/(k!(2n+2k+1)).

    boys_ref above is the fit path's reference and takes a real non-negative
    argument. The ellipse bound needs the analytic continuation: the ellipse's
    leftmost point is negative at every rho the bound is minimised at, and
    F_n grows there rather than decaying. The series is the continuation's own
    definition, needs no regime, and its terms are all of one sign at negative
    real z. At positive real z the terms alternate, so the caller's working
    precision must carry the cancellation: the largest term is of order e^z and
    the value of order 1/z, which is the DPS the derivation runs at.
    """
    if z == 0:
        return mpf(1) / (2 * order + 1)
    # The terms peak near k = |z|; this cap reaches well past the peak and the
    # early break stops the tail once it can no longer move the working
    # precision.
    az = float(abs(z))
    cap = int(4 * az) + 40
    # At positive real z the terms alternate and their largest is of order
    # e^z, so the sum loses about z/log(10) digits to cancellation. The working
    # precision carries that rather than the caller having to know it; at the
    # ellipse's leftmost point, which is where the bound evaluates, z is
    # negative and the terms do not alternate at all.
    extra = int(az) if z > 0 else 0
    with mp.workdps(mp.dps + extra):
        s = mpf(0)
        p = mpf(1)  # (-z)^k / k!
        for k in range(cap):
            s += p / (2 * order + 2 * k + 1)
            if k > az + 8 and abs(p) < abs(s) * mpf("1e-40"):
                break
            p *= -z / (k + 1)
        return s


def boys_entire_real(order, x):
    """F_order at real x, by the all-positive series (V&S eq. 26)."""
    return boys_ref(order, x)


def apriori_truncation_bound(order, a, b, deg, K=BOUND_K):
    """The proved truncation bound of a degree-deg fit of F_order over [a, b].

    Returns (bound, rho) - the minimising parameter is reported because it is
    the design law's own quantity: rho* sits near 2 deg / h, so the band a
    degree can carry is set by the band's WIDTH and by almost nothing else.
    """
    c = (a + b) / 2
    h = (b - a) / 2

    def at(rho):
        leftmost = c - h * (rho + 1 / rho) / 2
        return K * abs(boys_entire(order, leftmost)) * rho ** (-deg) / (rho - 1)

    # The design law puts the minimiser near rho0 = 2 (deg + 1) / h, to within
    # about a third where the ellipse stays left of the origin. The bound is
    # 1/(rho - 1)-shaped below that and e^(h rho/2) rho^-deg-shaped above it, so
    # a log-spaced scan of a decade either side brackets the minimum and a
    # golden-section refines the bracket. The scan is bounded rather than
    # searched to convergence because every evaluation at a rho far from the
    # minimum costs a series whose argument grows with rho.
    rho0 = 2 * (deg + 1) / h
    lo, hi = rho0 / 8, rho0 * 8
    if lo <= 1:
        lo = mpf("1.0000001")
    n = 16
    best, best_i = None, 0
    for i in range(n + 1):
        rho = lo * (hi / lo) ** (mpf(i) / n)
        v = at(rho)
        if best is None or v < best:
            best, best_i = v, i
    left = lo * (hi / lo) ** (mpf(max(best_i - 1, 0)) / n)
    right = lo * (hi / lo) ** (mpf(min(best_i + 1, n)) / n)

    # Golden-section on [left, right]: the interval is a bracket because the
    # scan found its minimum strictly inside it.
    invphi = (math.sqrt(5) - 1) / 2
    x1, x2 = right - invphi * (right - left), left + invphi * (right - left)
    f1, f2 = at(x1), at(x2)
    for _ in range(40):
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - invphi * (right - left)
            f1 = at(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + invphi * (right - left)
            f2 = at(x2)
    rho = (left + right) / 2
    return min(best, at(rho)), rho


def budget_at(target, b):
    """A piece's budget at its right edge.

    `target` is either that number or a callable of the right edge, which is
    what a weighted partition needs: a region-A piece's budget falls as the piece
    reaches further right, because the gain the piece's error is amplified by
    grows with b (see region_a_piece_budget).
    """
    return target(b) if callable(target) else target


def solve_right_edge(order, a, hi, target, deg, K=BOUND_K):
    """The widest b in (a, hi] with E(order, a, b, deg) <= target.

    E is non-decreasing in b - every rho's majorant grows with the interval's
    half-width - so bisection applies, and the predicate is checked for
    monotonicity at the bracket's ends rather than assumed. A callable target
    must likewise not grow faster with b than E does, which the weighted budget
    satisfies: it falls.
    """
    if apriori_truncation_bound(order, a, hi, deg, K)[0] <= budget_at(target, hi):
        return hi
    lo, up = a, hi
    for _ in range(48):
        mid = (lo + up) / 2
        if mid <= lo or mid >= up:
            break
        if apriori_truncation_bound(order, a, mid, deg, K)[0] <= budget_at(target, mid):
            lo = mid
        else:
            up = mid
    return lo


def equalise_partition(order, lo, hi, target, deg, K=BOUND_K):
    """Walk [lo, hi) left to right, each piece as wide as deg and target allow.

    This is the fixed point of "equalise the bound": every piece carries the
    same truncation bound and they are as wide as it lets them be, which makes
    the widths grow with the argument because the function decays. Nothing here
    samples the function. A callable target makes the bound equalised a
    weighted one and the widths follow it instead (see budget_at).
    """
    pieces = []
    a = lo
    while a < hi:
        b = solve_right_edge(order, a, hi, target, deg, K)
        if b <= a:
            raise RuntimeError(f"degree {deg} cannot reach {budget_at(target, a)} "
                               f"at F{order} from {a}")
        pieces.append((a, b))
        a = b
    return pieces


def derived_partition(order, lo, hi, target, degrees, K=BOUND_K):
    """The (degree, pieces, stored) trade over a ladder, and its minimum.

    stored counts one fit per piece: a partition at a fixed degree stores
    (deg + 1) coefficients on every one of its pieces.
    """
    rows = []
    for deg in degrees:
        pieces = equalise_partition(order, lo, hi, target, deg, K)
        rows.append((deg, len(pieces), len(pieces) * (deg + 1), pieces))
    best = min(rows, key=lambda r: r[2])
    return rows, best


# The degree ladder the derived partition is reported over. The low end is
# where narrow pieces are cheapest per piece and most expensive in total, so the
# ladder stops where the piece count stops being a table anyone would store.
DERIVE_DEGREES = (10, 12, 14, 16, 18, 20, 24)


def derive_partition_report(target):
    """Print the design law's output over the intervals the shipped fits serve.

    Region B is the interval the bound alone decides: it is one F_0 fit with no
    per-order weighting and no recursion downstream of it at the arguments the
    seam sits at, so a partition derived here is a partition of the fit and
    nothing else. Region A's pieces are a different case: they seed the batch
    entry's downward recursion and are held to a budget divided by that
    recursion's gain, which is a criterion this bound does not carry on its own.
    That partition is derived all the same, by narrow_region_a, under the
    weighted budget rather than this one - so the report below reads the ladder
    over region B and the narrow member over both regions.
    """
    # The self-check that the continuation is the function it claims to be:
    # boys_entire is the ellipse bound's own evaluator and boys_ref is the fit
    # path's, and at real non-negative arguments the two must agree. A reader
    # who doubts the majorant can run this line rather than take it.
    worst = mpf(0)
    for order in (0, 1, 4, 12, 32):
        for x in ("0.001", "0.5", "5.94992407605424223", "11.899848152108484",
                  "20.0", "28.989337738820740"):
            a, b = boys_entire(order, mpf(x)), boys_ref(order, mpf(x))
            worst = max(worst, abs(a - b) / abs(b))
    print(f"the bound's series against the fit path's reference: worst relative "
          f"difference {mp.nstr(worst, 3)} over the orders and arguments checked")
    print(f"the anchor: E(0, [0, 5.94992407605424223], 18) = "
          f"{mp.nstr(apriori_truncation_bound(0, mpf(0), mpf('5.94992407605424223'), 18)[0], 6)}"
          f", which is the figure the first band's degree was raised at")
    print()
    print(f"the a-priori truncation bound, derived at target {mp.nstr(target, 3)}")
    print("  E(order, a, b, deg) = min over rho > 1 of "
          f"K M(rho) rho^-deg / (rho - 1), K = {BOUND_K},")
    print("  M(rho) <= F_order(c - (h/2)(rho + 1/rho)), which holds exactly: "
          "|F(z)| <= F(Re z)")
    print("  by the triangle inequality on the integral, and the ellipse's leftmost "
          "point is where Re z is least. Nothing here samples the function.")
    print()

    print("the shipped pieces, read against that bound:")
    for tag, order, lo, hi, deg, stored in (
            ("region A band 1 ", 0, mpf(0), mpf("5.94992407605424223"), 20, 21),
            ("region A band 2 ", 0, mpf("5.94992407605424223"), X0, 18, 19),
            ("region B seed   ", 0, X0, X1, 18, 19),
            ("extended band   ", 0, XNEW0, X0, 24, 25)):
        e, rho = apriori_truncation_bound(order, lo, hi, deg)
        print(f"  {tag} [{mp.nstr(lo, 10)}, {mp.nstr(hi, 10)})  width "
              f"{mp.nstr(hi - lo, 8):>10s}  degree {deg:2d}  stored {stored:5d}  "
              f"E = {mp.nstr(e, 5):>11s}  rho* = {mp.nstr(rho, 4)}")
        print(f"      at or under the target: {e <= target}")
    print()

    print(f"the derived partition of region B [{mp.nstr(X0, 10)}, {mp.nstr(X1, 10)}), "
          f"width {mp.nstr(X1 - X0, 8)}:")
    rows, best = derived_partition(0, X0, X1, target, DERIVE_DEGREES)
    for deg, pieces, stored, pieces_list in rows:
        widths = [b - a for a, b in pieces_list]
        print(f"  degree {deg:2d}: {pieces:2d} pieces, {stored:4d} stored, "
              f"{deg + 1:2d} read per evaluation, width {mp.nstr(min(widths), 6)} .. "
              f"{mp.nstr(max(widths), 6)}")
    print(f"  fewest stored at this target: degree {best[0]}, {best[1]} piece(s), "
          f"{best[2]} stored")
    print()
    print(f"the narrow granularity's partition is the ladder's narrowest row, degree "
          f"{DERIVE_DEGREES[0]}: every piece reads {DERIVE_DEGREES[0] + 1} stored "
          "coefficients")
    print(f"against the shipped seed's 19, at the same target, piece by piece:")
    narrow = [r for r in rows if r[0] == DERIVE_DEGREES[0]][0]
    for a, b in narrow[3]:
        e, _ = apriori_truncation_bound(0, a, b, narrow[0])
        print(f"  [{mp.nstr(a, 12)}, {mp.nstr(b, 12)})  width {mp.nstr(b - a, 8):>10s}  "
              f"stored {narrow[0] + 1:2d}  E = {mp.nstr(e, 4)}")
    return 0


# ---------------------------------------------------------------------------
# The narrow partitions, as stored tables
# ---------------------------------------------------------------------------
# The derived partition is a design law; these are the members that ship it.
# Region B's seed is one fit over [kX0, kX1); the narrow member cuts the same
# interval into the pieces the proved bound gives at NARROW_TARGET and
# NARROW_DEG, each a Chebyshev interpolant of the same degree - so one
# evaluation reads NARROW_DEG + 1 coefficients instead of the shipped seed's
# MAX_DEG_DOUBLE + 1, and the table it reads them from is one row per piece
# instead of one row. Region A's per-order pieces are cut the same way at the
# same degree, under the criterion region A imposes: see narrow_region_a.
#
# That is a trade and not a saving, and the header's own comments say so: the
# coefficients per evaluation fall and the table grows by the piece count. A
# consumer whose cost is per evaluation takes it; one whose cost is the table
# does not. The extended band is the same fit at either granularity, because
# nothing amplifies its seed: the upward recursion it feeds carries an error
# forward at the size it already has, rather than multiplying it.
#
# The pieces are derived, not bisected: equalise_partition walks the interval
# left to right placing each piece as wide as the bound lets it be, so the
# widths grow with the argument and the last piece is short because the
# interval ends. The delivered error is measured rather than read off the
# bound, for the reason the derivation's own comment gives.
NARROW_TARGET = mpf("1e-14")
NARROW_DEG = DERIVE_DEGREES[0]
NARROW_SCHEMES = len(SCHEME_NAMES)

# The narrow region-A walk is the expensive part of this script's narrow block:
# a piece costs 48 bisection steps of a minimisation of the proved bound, and an
# order's walk places nine or ten of them. Each order's walk is independent, so
# they run in parallel, up to this many at once. Half a twelve-core host is what
# a full generation's other work was measured beside; the count changes no
# table, only how long the walk takes.
NARROW_WORKERS = 6


def narrow_region_b():
    """Region B's narrow partition as stored fits, with their measured bound.

    Returns the pieces (each a, b, deg, Chebyshev coefficients, monomial
    coefficients) and, per (scheme, multiply-add route), the worst delivered
    error over every piece, the same way the shipped fits are measured. The
    pieces are fitted on the derived mpf edges and stored as the doubles the
    kernel reads, exactly as region A's pieces are.
    """
    pieces = []
    for a, b in equalise_partition(0, X0, X1, NARROW_TARGET, NARROW_DEG):
        cm = cheb_coeffs(0, a, b, NARROW_DEG)
        pieces.append((float(a), float(b), NARROW_DEG,
                       [float(c) for c in cm],
                       [float(c) for c in cheb_to_monomial(cm)]))
    worst = [[0.0, 0.0] for _ in range(NARROW_SCHEMES)]
    for (_a, _b, _deg, cs, ms) in pieces:
        w = fit_delivered(cs, ms, 0, _a, _b, SCHEME_MEASURE_POINTS)
        for scheme in range(NARROW_SCHEMES):
            for route in (0, 1):
                worst[scheme][route] = max(worst[scheme][route], w[scheme][route])
    bounds = [[scheme_bound(worst[s][r]) for r in (0, 1)]
              for s in range(NARROW_SCHEMES)]
    return {"pieces": pieces, "worst": worst, "bounds": bounds}


# ---------------------------------------------------------------------------
# The narrow partition of region A, as stored tables
# ---------------------------------------------------------------------------
# A region-A piece is read two ways and the narrower partition has to hold both.
#
# The batch entry reads the top order's piece and recurses downward to F_0, so
# that piece's error arrives at F_0 multiplied by seed_weight(n, b) at the
# piece's right end - the widest point of the gain, up to 1.04e5 at order 12 and
# the region's right edge. The shipped pieces were placed under that reading with
# the design's own margin, TOL_DOUBLE / 2 / seed_weight(n, x), which is the law
# fit_interval weights by. The single-order lane reads the same piece at its own
# size, where region A is documented at 1e-15.
#
# So the budget is min(1e-15, 2.5e-14 / w(n, b)) - one criterion, not two: the
# two terms are the readings, the minimum is the tighter of them, and the gain
# binds only above w = 25, where the seed reading is stricter than the bar the
# single-order lane publishes. A piece is therefore limited by the gain near a
# high order's right edge and by the 1e-15 bar everywhere else, and the widths
# the walk produces show it: F_0's grow to the end of the region while a high
# order's last pieces come back narrower than the bar alone would ask for.
#
# The envelope is held and the reachable gain is much smaller, because the
# fallback is taken only below the band's left edge and the gain is then the
# call's own order's: 1 at every order from 3 up, 1.58 at order 2 and 2.18 at
# order 1. The walk holds the envelope rather than that figure so that no
# piece's reading depends on which order an entry seeds from.
#
# The gain is not transcribed here: the same seed_weight the shipped fits are
# weighted by is the one the walk is budgeted with, so a change to the shipped
# weighting moves the narrow partition with it.
REGION_A_SINGLE_BAR = mpf("1e-15")


def region_a_piece_budget(order, b):
    """The budget a region-A piece whose right edge is b is held to.

    The tighter of the two readings above: the single-order lane's 1e-15 bar,
    and the shipped fits' own weighted law TOL_DOUBLE / 2 / seed_weight. Called
    with the piece's right edge, which is where the gain is widest and so where
    the seed reading is strictest.
    """
    return min(REGION_A_SINGLE_BAR, TOL_DOUBLE * mpf("0.5") / seed_weight(order, b))


def narrow_a_order(job):
    """One order's narrow pieces, and the error they deliver.

    The walk places the pieces and each piece is then fitted at NARROW_DEG, the
    fit's edges being the mpf edges the walk derived and the stored edges the
    doubles those round to - the same two-step the shipped pieces take. A
    separate function so the orders can be walked in parallel: every piece costs
    48 bisection steps of a minimisation of the proved bound, and an order's walk
    depends on no other order. What comes back does not depend on how many
    workers ran it, since the arithmetic is the same in any process of the same
    working precision.
    """
    order, hi = job
    pieces = []
    for a, b in equalise_partition(order, mpf(0), hi,
                                  lambda edge: region_a_piece_budget(order, edge),
                                  NARROW_DEG):
        cm = cheb_coeffs(order, a, b, NARROW_DEG)
        pieces.append((float(a), float(b), NARROW_DEG,
                       [float(c) for c in cm],
                       [float(c) for c in cheb_to_monomial(cm)]))
    worst = [[0.0, 0.0] for _ in range(NARROW_SCHEMES)]
    for (_a, _b, _deg, cs, ms) in pieces:
        w = fit_delivered(cs, ms, order, _a, _b, SCHEME_MEASURE_POINTS)
        for scheme in range(NARROW_SCHEMES):
            for route in (0, 1):
                worst[scheme][route] = max(worst[scheme][route], w[scheme][route])
    return order, pieces, worst


def narrow_region_a(double_orders):
    """Region A's narrow partition as stored fits, with their measured bound.

    Every order's interval is walked from zero to where that order's last shipped
    piece ends, so the narrow member serves the whole of what it replaces and
    nothing beyond it. Returns the per-order pieces, the shipped partition they
    are measured against, and the worst delivered error per (scheme, multiply-add
    route) with its published bound. The bound places the pieces; it does not
    certify them, so the delivered figure is measured the way the shipped fits'
    is.
    """
    jobs = [(n, mpf(double_orders[n][-1][1])) for n in range(MAX_ORDER + 1)]
    with multiprocessing.get_context("spawn").Pool(min(len(jobs), NARROW_WORKERS)) as pool:
        out = pool.map(narrow_a_order, jobs)
    per_order = [pieces for _order, pieces, _worst in out]
    worst = [[0.0, 0.0] for _ in range(NARROW_SCHEMES)]
    for _order, _pieces, w in out:
        for scheme in range(NARROW_SCHEMES):
            for route in (0, 1):
                worst[scheme][route] = max(worst[scheme][route], w[scheme][route])
    bounds = [[scheme_bound(worst[s][r]) for r in (0, 1)]
              for s in range(NARROW_SCHEMES)]
    return {"orders": per_order, "shipped": double_orders,
            "worst": worst, "bounds": bounds}


def narrow_a_block_lines(narrow_a):
    """Region A's narrow tables: the derived partition as the header stores it.

    Same fields and the same meaning as the shipped kPieces table, so the lookup
    the kernel already runs serves either partition: a piece carries its own
    interval, degree and offset, and the piece a consumer's argument falls in is
    the piece it is evaluated at.
    """
    per_order = narrow_a["orders"]
    total = sum(len(pieces) for pieces in per_order)
    shipped = narrow_a["shipped"]
    shipped_rows = sum(len(pieces) for pieces in shipped)
    shipped_stored = sum(len(p[3]) for pieces in shipped for p in pieces)
    shipped_degs = sorted(p[2] for pieces in shipped for p in pieces)
    deg = NARROW_DEG
    lines = [
        "// The narrow partition of region A: every order's interval cut at",
        f"// degree {deg}, each piece as wide as the tighter of the two readings the",
        "// region is read under lets it be (the generator's region_a_piece_budget).",
        "// The batch entry seeds its downward recursion from the top order's piece,",
        "// and that recursion amplifies the seed's error by seed_weight(n, b) at",
        "// the piece's right end b. The envelope over the region reaches 1.04e5 at",
        "// order 12 and the region's right edge, while the fallback is taken only",
        "// below the band's left edge, x < kExtendedBX0, where the gain a call",
        "// reaches is that order's own: 1 at every order from 3 up, 1.58 at order 2",
        "// and 2.18 at order 1. For a fixed x the ratio x^n / prod(j + 1/2) rises",
        "// with n only until n passes x - 1/2 and falls after, so below the band",
        "// edge its largest value at an order of 3 or more is order 3's, which at",
        "// the edge itself is 0.68 and so is still short of 1. The walk holds the",
        "// envelope and not that reachable figure, so that no piece's reading",
        "// depends on which piece an entry happens to seed from. The single-order",
        "// lane reads the piece at its own size under region A's 1e-15 bar, and",
        "// both readings are held, so a high order's last pieces come back narrower",
        "// than either alone.",
        "//",
        "// Same fields, same meaning and the same lookup shape as kPieces gives the",
        "// shipped partition: one row per piece carrying its own interval, degree",
        "// and offset, so one scan serves either table. Naming this partition is a",
        "// trade of coefficients per evaluation for table rows and a piece lookup,",
        f"// not a saving: this table is {total} rows where the shipped partition is",
        f"// {shipped_rows}, and it stores {total * (deg + 1)} coefficients where the shipped",
        f"// stores {shipped_stored}, while one evaluation reads kNarrowADeg + 1 = {deg + 1}",
        f"// instead of the shipped pieces' {shipped_degs[0] + 1} to {shipped_degs[-1] + 1}.",
        f"inline constexpr int kNarrowADeg = {deg};",
        "inline constexpr auto kNarrowAPieces = std::to_array<OrderPiece>({",
    ]
    offset = 0
    for order, pieces in enumerate(per_order):
        for (a, b, _deg, _cs, _ms) in pieces:
            lines.append(f"  {{{fmt(mpf(a))}, {fmt(mpf(b))}, {deg}, {offset}}},  // F{order}")
            offset += deg + 1
    lines.append("});")
    starts = []
    index = 0
    for pieces in per_order:
        starts.append(index)
        index += len(pieces)
    starts.append(index)
    lines.append("inline constexpr auto kNarrowAPieceStart = std::to_array<int>({")
    lines.append("  " + ", ".join(str(v) for v in starts) + ",")
    lines.append("});")
    lines.append("static_assert(std::size(kNarrowAPieceStart) == kMaxOrder + 2\n"
                 "                  && std::size(kNarrowAPieces) == kNarrowAPieceStart[kMaxOrder + 1],\n"
                 "              \"the narrow region-A piece table must cover kMaxOrder\");")
    for name, column in (("kNarrowACoeffs", 3), ("kNarrowAMonoCoeffs", 4)):
        values = [fmt(c) for pieces in per_order for p in pieces for c in p[column]]
        lines.append(f"inline constexpr auto {name} = std::to_array<double>({{")
        for i in range(0, len(values), 6):
            lines.append("  " + ", ".join(values[i:i + 6]) + ",")
        lines.append("});")
    lines.append("static_assert(std::size(kNarrowAMonoCoeffs) == std::size(kNarrowACoeffs),\n"
                 "              \"the monomial table must parallel the Chebyshev table\");")
    lines.append("")
    lines.append("// The narrow region-A partition's certification rows: the bound each")
    lines.append("// scheme delivers on it in each multiply-add route, worst over its")
    lines.append("// pieces, published as a power-of-two round-up so it bounds a sweep")
    lines.append("// and not only the one that measured it. The pieces are measured at")
    lines.append("// their own size; the gain the batch entry's recursion applies to a")
    lines.append("// piece it seeds with is bounded by the budget the walk placed the")
    lines.append("// pieces under, which is what makes both readings hold at once.")
    lines.append("struct NarrowARow { int scheme, deg, pieces, stored;")
    lines.append("                    double fused, separate; };")
    lines.append("inline constexpr auto kNarrowARows = std::to_array<NarrowARow>({")
    for scheme in range(NARROW_SCHEMES):
        bounds = narrow_a["bounds"][scheme]
        lines.append(f"  {{{scheme}, {deg}, {total}, {total * (deg + 1)}, "
                     f"{fmt(bounds[0])}, {fmt(bounds[1])}}},")
    lines.append("});")
    return lines


def narrow_block_lines(narrow_a, narrow_b):
    """The narrow partitions' tables as the header writes them, both regions.

    One function rather than one per region because the block is what is
    reproduced on its own - --narrow-only writes these lines and the block's
    byte-identity is checked without fitting the shipped table - and the order
    the two regions are written in has to be one fact rather than two.
    write_header writes the same lines, so the two cannot drift.
    """
    return narrow_a_block_lines(narrow_a) + [""] + narrow_b_block_lines(narrow_b)


def narrow_b_block_lines(narrow):
    """Region B's narrow tables: the derived partition as the header stores it.

    A separate function because the block is reproduced on its own: the full
    generation is hours and this block is seconds, so --narrow-only writes
    these lines and the block's byte-identity is checked without the rest of
    the table. write_header writes the same lines, so the two cannot drift.
    """
    pieces = narrow["pieces"]
    deg = pieces[0][2]
    lines = [
        "// The narrow partition of region B: the same interval [kX0, kX1) cut",
        f"// into {len(pieces)} pieces at degree {deg}, each as wide as the proved a-priori",
        "// truncation bound lets it be at the 1e-14 target (see --derive-partition).",
        "// One evaluation reads kNarrowBDeg + 1 coefficients from the one piece the",
        "// argument falls in, against the shipped seed's kBDeg + 1 from its single",
        "// row - a trade of coefficients per evaluation against table rows, not a",
        "// saving: the table is kNarrowBPieces rows where the shipped seed is one.",
        "// Every piece is at the same degree, so piece i's coefficients start at",
        "// i * (kNarrowBDeg + 1) and no offset table is stored. The shipped seed's",
        "// coefficients above are untouched by this and are the same bytes whether",
        "// or not the partition is named.",
        "//",
        "// Region A's own narrow partition and the extended band: the extended band",
        "// is the same fit at either granularity, because nothing amplifies its",
        "// seed - the upward recursion it feeds runs the other way, so an error in",
        "// it stays the size it is. Region A's pieces are read the other way round,",
        "// and are partitioned above; their criterion is the gain the batch entry's",
        "// downward recursion applies to them, not this bound alone.",
        f"inline constexpr int kNarrowBDeg = {deg};",
        f"inline constexpr int kNarrowBPieces = {len(pieces)};",
    ]
    edges = [fmt(pieces[0][0])] + [fmt(p[1]) for p in pieces]
    lines.append("inline constexpr auto kNarrowBEdges = std::to_array<double>({")
    lines.append("  " + ", ".join(edges) + ",")
    lines.append("});")
    for name, index in (("kNarrowBcoeffs", 3), ("kNarrowBMonoCoeffs", 4)):
        values = [fmt(c) for p in pieces for c in p[index]]
        lines.append(f"inline constexpr auto {name} = std::to_array<double>({{")
        for i in range(0, len(values), 6):
            lines.append("  " + ", ".join(values[i:i + 6]) + ",")
        lines.append("});")
    lines.append("static_assert(std::size(kNarrowBEdges) == kNarrowBPieces + 1\n"
                 "                  && std::size(kNarrowBcoeffs) == kNarrowBPieces * (kNarrowBDeg + 1)\n"
                 "                  && std::size(kNarrowBMonoCoeffs) == std::size(kNarrowBcoeffs),\n"
                 "              \"the narrow partition's pieces must tile [kX0, kX1)\");")
    lines.append("")
    lines.append("// The narrow partition's certification rows, the same shape as the")
    lines.append("// scheme rows below and measured the same way: the bound each scheme")
    lines.append("// delivers on it in each multiply-add route, worst over its pieces,")
    lines.append("// published as a power-of-two round-up so it bounds a sweep and not")
    lines.append("// only the one that measured it. Counted apart from the shipped rows")
    lines.append("// because it is a second partition and not a row of the first.")
    lines.append("struct NarrowRow { int scheme, deg, stored;")
    lines.append("                   double fused, separate; };")
    lines.append("inline constexpr auto kNarrowRows = std::to_array<NarrowRow>({")
    for scheme in range(NARROW_SCHEMES):
        bounds = narrow["bounds"][scheme]
        lines.append(f"  {{{scheme}, {deg}, {len(pieces) * (deg + 1)}, {fmt(bounds[0])}, "
                     f"{fmt(bounds[1])}}},")
    lines.append("});")
    return lines


def fmt(v):
    return f"{v:.17e}"


def fmtf(v):
    """The float lane's spelling of a coefficient: the value the table holds,
    to 17 digits, with the C++ `f` suffix.

    The float tables hold float32, and an unsuffixed decimal is a double
    literal that narrows on initialization — the one construct in this tree
    MSVC reports at /W4 (C4305, ~11k instances across the two float tables).
    The suffix states the width instead of letting the compiler silently
    truncate. The stored float32 is unchanged by the spelling: the 17-digit
    emission round-trips the double exactly, so rounding that decimal to
    float32 in C++ and rounding the double here (struct.pack, correctly
    rounded) are the same operation on the same real value."""
    import struct
    f32 = struct.unpack("<f", struct.pack("<f", float(v)))[0]
    return f"{f32:.17e}f"


def next_double_above(v):
    """The smallest double >= the mpf value v (the kernel dispatches the
    extended band at these, so the dispatched region is a subset of the
    certified region)."""
    import struct
    d = float(v)
    if mpf(d) >= v:
        return d
    # One ulp up in the IEEE-754 double bit pattern (finite, positive range
    # only - the band constants are all positive normal doubles).
    bits = struct.unpack(">Q", struct.pack(">d", d))[0]
    return struct.unpack(">d", struct.pack(">Q", bits + 1))[0]


def extended_gen_delta(cd, edge, x):
    """The generator-side a-priori seed width of the extended fit at x: the
    delivered double-evaluation error plus the split-Clenshaw
    evaluation-rounding bound R_hat = 1.0116e-15 (the forward-error bound of
    the seed-bound derivation - the interval instrument certifies the band under
    the same quantity plus the truncation tail tau, delta_0' = R_hat + tau
    = 1.0641e-15)."""
    err = mpf(abs(clenshaw_double(cd, float(x), float(edge), float(X0)) - float(boys_ref(0, x))))
    return err + mpf("1.0116e-15")


def extended_gen_envelope(n, edge, cd, x, gamma):
    """The generator-side closed-form envelope B_n(x) of the extended band
    (the instrument's section-2 envelope with the per-argument seed width
    extended_gen_delta): A_n(x)*delta(x) + sum_j (A_n/A_{j+1}) r_j(x)."""
    x = mpf(x)
    if n == 0:
        return extended_gen_delta(cd, edge, x)
    a_n = mpf(1)
    for k in range(1, n + 1):
        a_n *= (k - mpf("0.5")) / x
    ex = exp(-x) / 2
    s = a_n * extended_gen_delta(cd, edge, x)
    for j in range(n):
        a_j1 = mpf(1)
        for k in range(1, j + 2):
            a_j1 *= (k - mpf("0.5")) / x
        rj = mpf("2.2204460492503131e-16") * ((2 * (j + mpf("0.5")) * boys_ref(j, x) + ex) / x
                                             + boys_ref(j + 1, x)) + gamma * ex / x
        s += a_n / a_j1 * rj
    return s


def extended_gen_crossing(kmax, edge, cd, gamma):
    """The generator-side certified crossing: the smallest x in [edge, X0]
    with max_{n<=kmax} B_n(x) <= 5e-14 under the generator's envelope model.
    When the envelope is already inside the claim at the band's left edge,
    the crossing is the edge itself (the seed exists nowhere below)."""
    def m_b(x):
        return max(extended_gen_envelope(n, edge, cd, x, gamma) for n in range(kmax + 1))
    lo = edge
    hi = X0
    assert m_b(hi) <= TOL_DOUBLE, f"kmax {kmax}: envelope at kX0 = {m_b(hi)} > 5e-14"
    if m_b(lo) <= TOL_DOUBLE:
        return lo  # clamped at the band edge
    for _ in range(60):
        mid = (lo + hi) / 2
        if m_b(mid) > TOL_DOUBLE:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_extended_band(b_cheb=None):
    """The extended band's F0 fit, as the fixed-point loop (the seed-width
    preempt): fit the seed on [edge, X0), compute the generator-side
    certified crossings under the fit's own delivered width, update the
    edge to the kmax-4 crossing, and repeat until every tier crossing
    converges (|delta| <= 1e-12, at most 8 iterations). A tier that fails
    to converge is DISABLED (its boundary becomes +inf so the kernel never
    serves it) and reported; the fit is unweighted (the upward recursion's
    amplification is charged in the envelope, as for the shipped region-B
    fit), tolerance the 5e-14 class, walking EXTENDED_DEG_LADDER.

    The junction condition (the seed-width preempt): the delivered double
    values of the new seed and the shipped region-B seed at kX0 must agree
    within 2e-14 absolute - checked as a hard error, so a band that fails
    the handoff is never shipped."""
    edge = XNEW0
    crossings = {4: mpf(0), 8: mpf(0), 16: mpf(0), 32: mpf(0)}
    deg = cd = None
    converged = False
    for _ in range(8):
        r = fit_interval(0, edge, X0, TOL_DOUBLE, 0, weighted=False,
                         degs=EXTENDED_DEG_LADDER)
        if r is None:
            raise RuntimeError(f"extended band: fit never converged on [{edge}, X0)")
        deg, cd, mono = r
        gamma = mpf(2) ** -52  # the 1-ulp-exp form (the primary certificate)
        new_crossings = {k: extended_gen_crossing(k, edge, cd, gamma) for k in crossings}
        if all(abs(new_crossings[k] - crossings[k]) <= mpf("1e-12") for k in crossings):
            crossings = new_crossings
            converged = True
            break
        edge = new_crossings[4]
        crossings = new_crossings
    if not converged:
        # Never ship a band the fit wasn't guaranteed for: disable the
        # unconverged tiers (boundary +inf - the kernel never serves them)
        # and report the loop's failure honestly.
        disabled = [k for k in crossings
                    if abs(new_crossings[k] - crossings[k]) > mpf("1e-12")]
        print(f"EXTENDED BAND: fixed-point loop did not converge; disabled tiers "
              f"{disabled} (boundaries +inf)")
        for k in disabled:
            crossings[k] = mpf("inf")
    if b_cheb is not None:
        bdeg, bcs, _bmono = b_cheb
        junction = abs(clenshaw_double(cd, float(X0), float(edge), float(X0))
                       - clenshaw_double(bcs, float(X0), float(X0), float(X1)))
        if junction > mpf("2e-14"):
            raise RuntimeError(f"extended band: kX0 junction check failed "
                               f"({junction} > 2e-14)")
    return deg, cd, mono, crossings


# ---------------------------------------------------------------------------
# The rational minimax region-B route
# ---------------------------------------------------------------------------
# Region B's seed is the one fit of the kernel that a rational approximation
# serves as well as a polynomial one, so it is the one the library offers a
# second certified route for. The route is fitted here, against the same
# stable series every other fit in this file is validated against, and its
# delivered error is measured in the double arithmetic the kernel evaluates it
# in rather than in mpmath: a fit's own residual is not what a caller
# receives, and the gap between the two is the whole reason this section
# re-implements the kernel's evaluation below.
#
# Two textbook routes are run. The Remez exchange solves the linearized
# equioscillation problem on a reference set and moves the set to the error's
# extrema; Lawson's algorithm reaches the same linear problem through
# iteratively reweighted least squares with the Sanathanan-Koerner denominator
# weight. They share no machinery, so agreement between them is evidence the
# error is near the best for that degree pair, and disagreement is not - the
# exchange result is what ships, and the second is printed beside it.
#
# The fits are made in t = 2(x - X0)/(X1 - X0) - 1, the same mapped argument
# the Chebyshev region-B fit uses: an affine change of variable maps the
# rational family onto itself, so the best error over a degree pair is the
# same in t as in x, and t keeps the linear systems well conditioned.

# The exchange solves for its coefficients rather than projecting them, and the
# reference set is a set of extrema rather than a set of nodes, so its linear
# systems spend digits a DCT does not. 60 dps leaves the delivered error, a
# 1e-14 quantity, about 45 digits of room.
RAT_DPS = 60
RAT_TAIL_FLOOR = mpf("1e-55")
RAT_TERMS = 900

# The degree pairs the scan visits: every numerator/denominator split whose
# stored count is at most RAT_SCAN_MAX, and never more than RAT_K_MAX
# denominator coefficients.
RAT_SCAN_MIN = 8
RAT_SCAN_MAX = 17
RAT_K_MAX = 8

# The certifying grid: RAT_NODES + 1 nodes uniform in t over [-1, 1]. The scan
# reads every RAT_STRIDE-th node of it, so one reference build serves both
# views and the scan's nodes are a subset of the certifying ones.
RAT_NODES = 2000
RAT_STRIDE = 5

# The bar both region-B routes are placed against (the header's 5e-14 target
# for the interval) and the figures the header states.
RAT_BAR = mpf("5e-14")


def rat_fma(a, b, c):
    """a * b + c with the product exact and the sum rounded once.

    Dekker's product split, so the fused step is the same operation on every
    platform and on every Python the CI matrix carries: `math.fma` is 3.13 and
    later, and the generated header is byte-compared on machines that need not
    agree about which Python runs the check.
    """
    p = a * b
    ah = a * 134217729.0  # 2^27 + 1
    ah = ah - (ah - a)
    al = a - ah
    bh = b * 134217729.0
    bh = bh - (bh - b)
    bl = b - bh
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    s = p + c
    bv = s - p
    return s + (((p - (s - bv)) + (c - bv)) + err)


def rat_horner_double(cs, t):
    acc = cs[-1]
    for c in reversed(cs[:-1]):
        acc = rat_fma(acc, t, c)
    return acc


def rat_eval_double(p, q, t):
    """The kernel's evaluation of the route: Horner in double, fused throughout."""
    num = rat_horner_double(p, t)
    den = rat_fma(rat_horner_double(q, t), t, 1.0) if q else 1.0
    return num / den


def clenshaw_split_double(cs, t):
    """ClenshawSplit<ScalarFp64>, operation for operation.

    The shipped header's split Clenshaw, transcribed so the region-B Chebyshev
    route's delivered error is measured in the arithmetic the kernel runs it
    in: the two routes are compared, so an evaluation the kernel does not
    perform would compare the wrong pair.
    """
    deg = len(cs) - 1
    if deg == 0:
        return cs[0]
    if deg == 1:
        return rat_fma(t, cs[1], cs[0])
    v = rat_fma(2.0, t * t, -1.0)
    two_v = v + v
    if deg == 2:
        return rat_fma(t, cs[1], rat_fma(v, cs[2], cs[0]))
    m = deg // 2
    b1 = cs[2 * m]
    b2 = 0.0
    for k in range(m - 1, 0, -1):
        b0 = rat_fma(two_v, b1, cs[2 * k] - b2)
        b2 = b1
        b1 = b0
    even = rat_fma(v, b1, cs[0] - b2)
    o1 = cs[2 * m - 1]
    o2 = 0.0
    for k in range(m - 2, 0, -1):
        o0 = rat_fma(two_v, o1, cs[2 * k + 1] - o2)
        o2 = o1
        o1 = o0
    odd = rat_fma(two_v - 1.0, o1, cs[1] - o2)
    return rat_fma(t, odd, even)


def region_b_seed_chebyshev(cs, x):
    """RegionBSeed(x): the same mapped argument and the same split Clenshaw."""
    t = 2.0 * (float(x) - float(X0)) / (float(X1) - float(X0)) - 1.0
    return clenshaw_split_double(cs, t)


def region_b_seed_rational(p, q, x):
    t = 2.0 * (float(x) - float(X0)) / (float(X1) - float(X0)) - 1.0
    return rat_eval_double(p, q, t)


def rat_nodes():
    """The certifying grid, in t and in x."""
    ts = [-1 + 2 * mpf(i) / RAT_NODES for i in range(RAT_NODES + 1)]
    xs = [X0 + (X1 - X0) * (t + 1) / 2 for t in ts]
    return ts, xs


def rat_reference(xs):
    with mp.workdps(RAT_DPS):
        return [boys_ref(0, x, RAT_TAIL_FLOOR, RAT_TERMS) for x in xs]


def rat_horner(cs, t):
    acc = mpf(0)
    for c in reversed(cs):
        acc = acc * t + c
    return acc


def rat_value(p, q, t):
    """The rational value in mpmath, at the working precision."""
    den = 1 + rat_horner(q, t) * t if q else mpf(1)
    return rat_horner(p, t) / den


def rat_reference_solve(ref, m, k, ts, fs, ws=None):
    """The linearized equioscillation system on a reference set of grid indices.

    p(t_i) - f_i * sum_j q_j t_i^j - (-1)^i E = f_i, with q_0 held at 1: the
    denominator's constant term is the one normalization the family needs, and
    fixing it at 1 keeps the problem linear.

    A weight vector scales the p columns, the q columns and the right-hand side
    of every row, so the solved fit equalizes the *weighted* error: the
    equioscillation the exchange then drives is on w*(f - fit), which is the
    problem a fit weighted against a downstream amplification grows into.
    """
    n = m + k + 1
    a = mp.zeros(n + 1, n + 1)
    rhs = mp.zeros(n + 1, 1)
    for i, index in enumerate(ref):
        t = ts[index]
        fi = fs[index]
        w = ws[index] if ws is not None else mpf(1)
        for j in range(m + 1):
            a[i, j] = w * t ** j
        for j in range(1, k + 1):
            a[i, m + j] = -w * fi * t ** j
        a[i, n] = -mpf((-1) ** i)
        rhs[i] = w * fi
    sol = mp.lu_solve(a, rhs)
    return ([sol[j] for j in range(m + 1)], [sol[m + j] for j in range(1, k + 1)], sol[n])


def rat_extrema_indices(errs):
    """Grid indices where the error is a local extremum, endpoints included."""
    out = [0, len(errs) - 1]
    for i in range(1, len(errs) - 1):
        if (errs[i] >= errs[i - 1] and errs[i] >= errs[i + 1]) or \
           (errs[i] <= errs[i - 1] and errs[i] <= errs[i + 1]):
            out.append(i)
    return sorted(set(out))


def rat_alternating(idxs, errs, need):
    """The largest-magnitude subset of `need` extrema whose signs alternate.

    Greedy from the largest: the reference set of a rational exchange has to
    alternate, and a set that does not is not a reference set at all.
    """
    chosen = []
    for i in sorted(idxs, key=lambda i: -abs(errs[i])):
        if len(chosen) == need:
            break
        cand = sorted(chosen + [i])
        signs = [1 if errs[j] > 0 else -1 for j in cand]
        if all(signs[c] != signs[c + 1] for c in range(len(signs) - 1)):
            chosen = cand
    return sorted(chosen)


def rat_remez(m, k, indices, ts, fs, ws=None, iters=60):
    """The remez exchange over a fixed grid; returns (worst, p, q) or None."""
    n = m + k + 1
    grid = [i for i in indices]
    ref = sorted({min(grid, key=lambda i: abs(ts[i] - mp.cos(mp.pi * j / n)))
                  for j in range(n + 1)})
    if len(ref) != n + 1:
        return None
    best = None
    stall = 0
    for _ in range(iters):
        try:
            p, q, _ = rat_reference_solve(ref, m, k, ts, fs, ws)
        except (ZeroDivisionError, ValueError):
            break
        errs = [(ws[i] if ws is not None else mpf(1)) * (fs[i] - rat_value(p, q, ts[i]))
                for i in grid]
        # A denominator that comes near zero inside the interval is not a fit,
        # whatever its residual on the reference set: the kernel would divide
        # by it.
        if q:
            dens = [abs(1 + rat_horner(q, ts[i]) * ts[i]) for i in grid]
            if min(dens) < max(dens) * mpf("1e-3"):
                break
        mx = max(abs(e) for e in errs)
        if best is None or mx < best[0]:
            best = (mx, p, q)
            stall = 0
        else:
            stall += 1
            if stall > 3:
                break
        sel = rat_alternating(rat_extrema_indices(errs), errs, n + 1)
        if len(sel) != n + 1:
            break
        if [grid[i] for i in sel] == ref:
            break
        ref = [grid[i] for i in sel]
    return best


def rat_lawson(m, k, indices, ts, fs, ws=None, outer=300):
    """Lawson's algorithm with the Sanathanan-Koerner denominator weight.

    The weight carried in `w` is both the target weight, seeded from `ws`, and
    the SK factor the iteration multiplies in: the normal equations are built
    from its square, so seeding it with a target weight is exactly what makes
    the least-squares problem the weighted one.
    """
    nunk = (m + 1) + k
    w = [(ws[i] if ws is not None else mpf(1)) for i in indices]
    last = None
    for _ in range(outer):
        a = mp.zeros(nunk, nunk)
        rhs = mp.zeros(nunk, 1)
        for row, index in enumerate(indices):
            t = ts[index]
            fi = fs[index]
            base = [t ** j for j in range(m + 1)] + [-fi * t ** j for j in range(1, k + 1)]
            w2 = w[row] * w[row]
            for r in range(nunk):
                rhs[r] += w2 * base[r] * fi
                for c in range(r, nunk):
                    a[r, c] += w2 * base[r] * base[c]
        for r in range(nunk):
            for c in range(r):
                a[r, c] = a[c, r]
        try:
            sol = mp.lu_solve(a, rhs)
        except (ZeroDivisionError, ValueError):
            return last
        p = [sol[j] for j in range(m + 1)]
        q = [sol[m + j] for j in range(1, k + 1)]
        errs = []
        for row, index in enumerate(indices):
            t = ts[index]
            den = 1 + rat_horner(q, t) * t
            e = fs[index] - rat_horner(p, t) / den
            if ws is not None:
                e = ws[index] * e
            errs.append(e)
            w[row] = w[row] / abs(den)
        mx = max(abs(e) for e in errs)
        if mx == 0:
            return last
        w = [wi * (abs(e) / mx) for wi, e in zip(w, errs)]
        nrm = max(w)
        w = [wi / nrm for wi in w]
        last = (mx, p, q)
    return last


def rat_delivered_error(p, q, xs, ref):
    """The route's delivered worst |F0 - fit|, as the kernel evaluates it."""
    worst = mpf(0)
    at = None
    for x, f in zip(xs, ref):
        e = abs(mpf(region_b_seed_rational(p, q, x)) - f)
        if e > worst:
            worst, at = e, x
    return worst, at


def cheb_delivered_error(cs, xs, ref):
    worst = mpf(0)
    at = None
    for x, f in zip(xs, ref):
        e = abs(mpf(region_b_seed_chebyshev(cs, x)) - f)
        if e > worst:
            worst, at = e, x
    return worst, at


def rat_delivered_on(p, q, indices, xs, ref):
    """The route's delivered worst |F0 - fit| over `indices`, kernel arithmetic."""
    worst = mpf(0)
    at = None
    for i in indices:
        e = abs(mpf(region_b_seed_rational(p, q, xs[i])) - ref[i])
        if e > worst:
            worst, at = e, xs[i]
    return worst, at


def rat_best(m, k, indices, ts, xs, ref, ws=None):
    """Both routes to the same problem: the better fit, and every figure.

    The exchange and Lawson share no machinery - the exchange solves the
    equioscillation system on an alternating reference, Lawson reweights a
    linear least-squares problem and iterates - and either can stall on a
    given degree pair. So both run, both are reported, and the better fit is
    the one offered for that pair. Returns (best, found), each found entry a
    tuple of the route's name, its delivered error, the argument, and p, q.
    """
    found = []
    r = rat_remez(m, k, indices, ts, ref, ws)
    if r is not None:
        _, p, q = r
        d, at = rat_delivered_on(p, q, indices, xs, ref)
        found.append(("Remez exchange", d, at, p, q))
    l = rat_lawson(m, k, indices, ts, ref, ws)
    if l is not None:
        _, p, q = l
        d, at = rat_delivered_on(p, q, indices, xs, ref)
        found.append(("Lawson/SK", d, at, p, q))
    if not found:
        return None, []
    return min(found, key=lambda f: f[1]), found


def fit_region_b_rational(b_cheb):
    with mp.workdps(RAT_DPS):
        return _fit_region_b_rational(b_cheb)


def _fit_region_b_rational(b_cheb):
    """The rational route over the Chebyshev route's own interval.

    Scans the degree pairs, refines the reaching candidates on the certifying
    grid, and returns the smallest-stored one, with the Chebyshev route's own
    delivered error measured beside it on the same grid against the same
    reference.
    """
    ts, xs = rat_nodes()
    ref = rat_reference(xs)
    coarse = list(range(0, RAT_NODES + 1, RAT_STRIDE))

    print(f"fitting the rational region-B route (both routes, {RAT_DPS} dps, "
          f"bar {mp.nstr(RAT_BAR, 2)}) ...")
    reaching = None
    for stored in range(RAT_SCAN_MIN, RAT_SCAN_MAX + 1):
        row = []
        for k in range(1, min(RAT_K_MAX, stored - 2) + 1):
            m = stored - 1 - k
            best, _ = rat_best(m, k, coarse, ts, xs, ref)
            if best is None:
                row.append(f"[{m}/{k}]-")
                continue
            _, d, _, _, _ = best
            row.append(f"[{m}/{k}]{mp.nstr(d, 3)}")
            if d <= RAT_BAR and (reaching is None or stored < reaching[0]):
                reaching = (stored, m, k)
        print(f"  {stored:>2} stored: " + "  ".join(row))
        if reaching is not None:
            # The scan's job is the knee: the smallest count that reaches, and
            # the count below it that does not. Everything above the knee is
            # the question the refinement answers on a finer grid, so the scan
            # stops here rather than sweeping counts nothing will use.
            break
    if reaching is None:
        raise RuntimeError(f"rational route: no degree pair up to {RAT_SCAN_MAX} "
                           f"stored coefficients reaches {mp.nstr(RAT_BAR, 2)} over "
                           f"[{mp.nstr(X0, 17)}, {mp.nstr(X1, 17)}]")

    stored, m, k = reaching
    # The pairs at the reaching count, and the best pair one count below it,
    # refined on the certifying grid. The one-below row is the knee itself: it
    # is what says the count above is the smallest that reaches rather than the
    # first the scan happened to try.
    candidates = [(stored, stored - 1 - kk, kk)
                  for kk in range(1, min(RAT_K_MAX, stored - 2) + 1)]
    below = min(RAT_SCAN_MIN, stored - 1)
    candidates += [(below, below - 1 - kk, kk)
                   for kk in range(1, min(RAT_K_MAX, below - 2) + 1)]
    full = list(range(RAT_NODES + 1))
    rows = []
    for total, mm, kk in candidates:
        best, _ = rat_best(mm, kk, full, ts, xs, ref)
        if best is None:
            continue
        who, d, at, p, q = best
        rows.append((total, mm, kk, d, at, p, q, who))
    if not rows:
        raise RuntimeError("rational route: the refined fit never converged")

    print(f"  refined on {RAT_NODES + 1} nodes against boys_ref at {RAT_DPS} digits, "
          f"the better of the two routes taken per pair:")
    for total, mm, kk, d, at, _, _, who in sorted(rows, key=lambda r: (r[0], r[3])):
        print(f"    [{mm}/{kk}] {total} stored via {who}: delivered {mp.nstr(d, 6)} at "
              f"x = {mp.nstr(at, 12)}"
              f"{'  <= bar' if d <= RAT_BAR else '  OVER THE BAR'}")

    ok = [r for r in rows if r[0] == stored and r[3] <= RAT_BAR]
    if not ok:
        raise RuntimeError(f"rational route: no pair at {stored} stored coefficients "
                           f"reaches {mp.nstr(RAT_BAR, 2)} on the certifying grid")
    if any(r[0] < stored and r[3] <= RAT_BAR for r in rows):
        raise RuntimeError(f"rational route: a pair below {stored} stored coefficients "
                           f"reaches {mp.nstr(RAT_BAR, 2)}, so the scan's knee is wrong")
    total, m, k, delivered, at, p, q, who = min(ok, key=lambda r: r[3])

    # The two routes' figures at the pair that ships, either one of which may
    # be the one offered above: the gap is what says how close to the best fit
    # for this pair the shipped one is, and a wide gap is a reason to distrust
    # the pair rather than the route that won it.
    _, both = rat_best(m, k, full, ts, xs, ref)
    print(f"  both routes at the shipped pair [{m}/{k}]:")
    for name, d, a, _, _ in both:
        print(f"    {name:<16}: delivered {mp.nstr(d, 6)} at x = {mp.nstr(a, 12)}"
              f"{'   <- shipped' if name == who else ''}")

    bcs = b_cheb[1]
    cheb_delivered, cheb_at = cheb_delivered_error(bcs, xs, ref)
    print(f"  the two routes over the same interval and reference:")
    print(f"    Chebyshev deg {b_cheb[0]} : {len(bcs):>2} stored, delivered "
          f"{mp.nstr(cheb_delivered, 6)} at x = {mp.nstr(cheb_at, 12)}")
    print(f"    rational [{m}/{k}]  : {total:>2} stored, delivered "
          f"{mp.nstr(delivered, 6)} at x = {mp.nstr(at, 12)}")

    return {
        "m": m, "k": k, "p": p, "q": q, "stored": total,
        "delivered": delivered, "cheb_delivered": cheb_delivered,
        "cheb_stored": len(bcs), "bar": RAT_BAR,
    }


# ---------------------------------------------------------------------------
# The region-A rational route
# ---------------------------------------------------------------------------
# Region A's shipped pieces are fitted weighted against the downward-recursion
# amplification, and the route's pieces are fitted against the same weight: a
# rational fitted to the unweighted problem would look leaner and stop being
# the same problem. The criterion is what a consumer sees - the plain delivered
# error in the kernel's arithmetic against the 60-digit reference - rather than
# the weighted acceptance budget the truncation argument uses.
#
# The route's pieces share the shipped pieces' intervals, and its selector takes
# them over per order rather than from zero: below an order's own region-A end
# the shipped lane is documented at 1e-15, and a rational holding 3e-14 is not a
# fit for it, so the route hands that order over at that argument and states the
# lowest of those arguments instead of being stretched to share a domain it does
# not serve.
#
# The bar a piece is accepted against is deliberately below the bar the row
# publishes. The acceptance sweep and the check that certifies the row run on
# different argument grids, so a piece accepted a hair under a published bar
# could measure a hair over it on the certifying grid; the gap is what keeps the
# published figure a promise rather than a coincidence of one grid.
RAT_A_BOUND = mpf("3e-14")
RAT_A_ACCEPT = mpf("2.5e-14")
RAT_A_GRID = 400
RAT_A_SCAN_STRIDE = 5
RAT_A_SCAN_MAX = 22


def rat_a_nodes(a, b, npts):
    """The piece's grid, in the kernel's own double-mapped argument.

    The kernel maps x to t in binary64 and then evaluates there, so the grid's
    arguments, its mapped values and the reference are all binary64: a fit
    certified against an exactly-mapped t would be certified for an evaluation
    the kernel does not perform.
    """
    ad, bd = float(a), float(b)
    xs = [ad + (bd - ad) * i / npts for i in range(npts + 1)]
    ts = [mpf(2.0 * (x - ad) / (bd - ad) - 1.0) for x in xs]
    return ts, xs


def rat_a_delivered_on(p, q, ts, ref, indices):
    """Delivered worst |F_n - fit| over `indices`, in the kernel's arithmetic."""
    pd = [float(c) for c in p]
    qd = [float(c) for c in q]
    worst = mpf(0)
    at = None
    for i in indices:
        e = abs(mpf(rat_eval_double(pd, qd, float(ts[i]))) - ref[i])
        if e > worst:
            worst, at = e, ts[i]
    return worst, at


def rat_a_best_at(ts, ref, ws, indices, total):
    """The best pair at `total` stored over `indices`, by plain delivered error."""
    best = None
    for k in range(1, min(RAT_K_MAX, total - 2) + 1):
        m = total - 1 - k
        if m < 1:
            continue
        r = rat_remez(m, k, indices, ts, ref, ws)
        if r is None:
            continue
        _, p, q = r
        d, at = rat_a_delivered_on(p, q, ts, ref, indices)
        if best is None or d < best[0]:
            best = (d, m, k, p, q, at)
    return best


def rat_a_fit_piece(n, a, b, cs):
    """The smallest stored pair holding RAT_A_ACCEPT on [a, b), and the control.

    The count is first found on a stride subsample and then re-found on the
    denser grid, because a count the subsample accepts and the dense grid
    rejects is the subsample's error rather than the fit's; the search resumes
    above such a count instead of taking it. Both the subsample and the dense
    grid accept against the same tolerance, below the bar the row publishes.
    """
    ts, xs = rat_a_nodes(a, b, RAT_A_GRID)
    with mp.workdps(RAT_DPS):
        ref = [boys_ref(n, x, RAT_TAIL_FLOOR, RAT_TERMS) for x in xs]
    ws = [seed_weight(n, x) for x in xs]
    coarse = list(range(0, RAT_A_GRID + 1, RAT_A_SCAN_STRIDE))
    full = list(range(RAT_A_GRID + 1))

    # The shipped Chebyshev piece beside it: same interval, same mapped
    # argument, same reference, same arithmetic, so the two stored counts and
    # the two delivered errors are a comparison rather than two measurements.
    # Each piece's error is also carried through the downward recursion's gain,
    # which is the criterion the shipped pieces are fitted under and the
    # rational pieces are not: the two weighted figures are what says whether a
    # table could seed that recursion.
    cds = [float(c) for c in cs]
    cheb_worst = mpf(0)
    cheb_at = None
    cheb_gain_worst = mpf(0)
    for i in full:
        e = abs(mpf(clenshaw_split_double(cds, float(ts[i]))) - ref[i])
        if e > cheb_worst:
            cheb_worst, cheb_at = e, ts[i]
        cheb_gain_worst = max(cheb_gain_worst, e * ws[i])

    total = 6
    while total <= RAT_A_SCAN_MAX:
        accepted = None
        for t in range(total, RAT_A_SCAN_MAX + 1):
            r = rat_a_best_at(ts, ref, ws, coarse, t)
            if r is not None and r[0] <= RAT_A_ACCEPT:
                accepted = t
                break
        if accepted is None:
            break
        r = rat_a_best_at(ts, ref, ws, full, accepted)
        if r is not None and r[0] <= RAT_A_ACCEPT:
            d, m, k, p, q, at = r
            pd = [float(c) for c in p]
            qd = [float(c) for c in q]
            gain_worst = mpf(0)
            for i in full:
                e = abs(mpf(rat_eval_double(pd, qd, float(ts[i]))) - ref[i]) * ws[i]
                gain_worst = max(gain_worst, e)
            return ({"m": m, "k": k, "p": p, "q": q, "stored": m + 1 + k,
                     "delivered": d, "at": at, "gain": gain_worst},
                    cheb_worst, cheb_at, cheb_gain_worst, len(cds))
        total = accepted + 1
    return None, cheb_worst, cheb_at, cheb_gain_worst, len(cds)


def fit_region_a_rational(double_orders):
    """The rational route's pieces, in the shipped table's own order."""
    print(f"fitting the rational region-A route ({RAT_DPS} dps, accepted at "
          f"{mp.nstr(RAT_A_ACCEPT, 2)} plain, bar {mp.nstr(RAT_A_BOUND, 2)}, "
          f"fitted under the shipped weighting) ...")
    pieces = []
    stored = 0
    delivered = mpf(0)
    at = None
    cheb_stored = 0
    cheb_delivered = mpf(0)
    cheb_at = None
    gain = mpf(0)
    gain_at = None
    cheb_gain = mpf(0)
    cheb_gain_at = None
    over_gain = 0
    cheb_over_gain = 0

    for n in range(MAX_ORDER + 1):
        row = []
        for (a, b, deg, cs, _mono) in double_orders[n]:
            with mp.workdps(RAT_DPS):
                fit, cw, cat, cg, cst = rat_a_fit_piece(n, mpf(a), mpf(b), cs)
            if fit is None:
                raise RuntimeError(f"rational region-A route: F{n} on [{a}, {b}) "
                                   f"reaches no stored count up to {RAT_A_SCAN_MAX} "
                                   f"holding {mp.nstr(RAT_A_ACCEPT, 2)}")
            pieces.append(fit)
            stored += fit["stored"]
            if fit["delivered"] > delivered:
                delivered, at = fit["delivered"], fit["at"]
            if fit["gain"] > gain:
                gain, gain_at = fit["gain"], (n, mpf(a), mpf(b))
            if fit["gain"] > RAT_A_ACCEPT:
                over_gain += 1
            cheb_stored += cst
            if cw > cheb_delivered:
                cheb_delivered, cheb_at = cw, cat
            if cg > cheb_gain:
                cheb_gain, cheb_gain_at = cg, (n, mpf(a), mpf(b))
            if cg > RAT_A_ACCEPT:
                cheb_over_gain += 1
            row.append(f"F{n} [{mp.nstr(mpf(a), 6)}, {mp.nstr(mpf(b), 6)}) deg {deg} "
                       f"({cst} stored, {mp.nstr(cw, 3)}, gain {mp.nstr(cg, 3)}) -> "
                       f"[{fit['m']}/{fit['k']}] ({fit['stored']} stored, "
                       f"{mp.nstr(fit['delivered'], 3)}, gain {mp.nstr(fit['gain'], 3)})")
        print("  " + "\n  ".join(row))
        sys.stdout.flush()

    print(f"  region A: Chebyshev {cheb_stored} stored, delivered {mp.nstr(cheb_delivered, 6)} "
          f"at x = {mp.nstr(cheb_at, 12)}")
    print(f"  region A: rational  {stored} stored, delivered {mp.nstr(delivered, 6)} "
          f"at x = {mp.nstr(at, 12)}")
    print("  after the downward recursion's gain (max(1, x^n / prod(j+1/2)), the criterion "
          "the shipped pieces are fitted under):")
    print(f"    Chebyshev worst {mp.nstr(cheb_gain, 6)} at F{cheb_gain_at[0]} "
          f"[{mp.nstr(cheb_gain_at[1], 6)}, {mp.nstr(cheb_gain_at[2], 6)}), "
          f"{cheb_over_gain} of {len(pieces)} pieces over {mp.nstr(RAT_A_ACCEPT, 2)}")
    print(f"    rational  worst {mp.nstr(gain, 6)} at F{gain_at[0]} "
          f"[{mp.nstr(gain_at[1], 6)}, {mp.nstr(gain_at[2], 6)}), "
          f"{over_gain} of {len(pieces)} pieces over {mp.nstr(RAT_A_ACCEPT, 2)}")
    print(f"  the pieces cover [0, {mp.nstr(X0, 17)}); the rational selector takes over per "
          f"order, from each order's own region-A end, the lowest of which is "
          f"{mp.nstr(XNEW0, 17)} - above it the shipped lane is documented at 3e-14 and "
          f"below it at 1e-15, a bound these pieces do not hold")
    print(f"  every piece was accepted at {mp.nstr(RAT_A_ACCEPT, 2)} or below, so the "
          f"{mp.nstr(RAT_A_BOUND, 2)} the rows publish carries headroom for the certifying "
          f"sweep to disagree about")

    return {
        "pieces": pieces, "stored": stored, "delivered": delivered, "at": at,
        "cheb_stored": cheb_stored, "cheb_delivered": cheb_delivered,
        "cheb_at": cheb_at, "bound": RAT_A_BOUND,
        "lo": XNEW0, "hi": X0,
    }


def write_header(path, double_orders, float_orders, b_cheb, b_cheb_f32, ext_cheb,
                 scheme_rows, rat_b, rat_a, narrow, narrow_a):
    with open(path, "w", newline="\n") as f:
        f.write("// Generated by tools/gen_boys_coefficients.py - DO NOT EDIT.\n")
        f.write("// Piecewise Chebyshev (split Clenshaw) fits of F_n(x), region A seeds\n")
        f.write("// weighted against downward-recursion amplification; validated against a\n")
        f.write("// 30-digit mpmath reference (definitive check: tests/boys_test.cpp).\n")
        f.write("/// \\cond\n")
        f.write("// Not API: the generated tables the entries are compiled from. The\n")
        f.write("// header ships because the entries' kernels are header-defined; the\n")
        f.write("// API reference documents the entries.\n")
        f.write("#pragma once\n#include <array>\n#include <cstddef>\n\n")
        f.write("namespace boys::detail {\n\n")
        f.write(f"inline constexpr int kMaxOrder = {MAX_ORDER};\n")
        f.write(f"inline constexpr double kX0 = {fmt(X0)};\n")
        f.write(f"inline constexpr double kX1 = {fmt(X1)};\n\n")

        all_coeffs = []
        all_mono = []
        meta = []
        for n in range(MAX_ORDER + 1):
            for (a, b, deg, cs, ms) in double_orders[n]:
                meta.append((n, a, b, deg, len(all_coeffs)))
                all_coeffs.extend(fmt(c) for c in cs)
                all_mono.extend(fmt(c) for c in ms)
        f.write("inline constexpr auto kCoeffs = std::to_array<double>({\n")
        for i in range(0, len(all_coeffs), 6):
            f.write("  " + ", ".join(all_coeffs[i:i + 6]) + ",\n")
        f.write("});\n\n")
        # The monomial form of the same fits, one stored coefficient per
        # Chebyshev coefficient: same pieces, same intervals, same degrees,
        # same offsets, so an evaluation scheme picks a table and not a shape.
        # Converted from the exact Chebyshev coefficients and rounded once, so
        # both tables are the correctly rounded doubles of one ideal
        # polynomial; the Horner route adds no error of its own to the fit.
        f.write("// The same fits in monomial form, ascending, for the Horner\n")
        f.write("// evaluation scheme: same pieces, degrees and offsets as kCoeffs.\n")
        f.write("inline constexpr auto kMonoCoeffs = std::to_array<double>({\n")
        for i in range(0, len(all_mono), 6):
            f.write("  " + ", ".join(all_mono[i:i + 6]) + ",\n")
        f.write("});\n")
        f.write("static_assert(std::size(kMonoCoeffs) == std::size(kCoeffs),\n"
                "              \"the monomial table must parallel the Chebyshev table\");\n\n")
        f.write("struct OrderPiece { double a, b; int deg; int offset; };\n")
        f.write("inline constexpr auto kPieces = std::to_array<OrderPiece>({\n")
        for (n, a, b, deg, off) in meta:
            f.write(f"  {{{fmt(mpf(a))}, {fmt(mpf(b))}, {deg}, {off}}},  // F{n}\n")
        f.write("});\n")
        starts = []
        index = 0
        for n in range(MAX_ORDER + 1):
            starts.append(index)
            index += len(double_orders[n])
        starts.append(index)
        f.write("inline constexpr auto kPieceStart = std::to_array<int>({"
                + ", ".join(map(str, starts)) + "});\n")
        f.write("static_assert(std::size(kPieceStart) == kMaxOrder + 2,\n"
                "              \"piece-start table must cover kMaxOrder\");\n\n")

        # The region-A rational route: one weighted minimax fit per shipped
        # piece, over the piece's own interval and in the piece's own mapped
        # argument, so the two routes are the same partition of the same region
        # fitted two ways. Each piece stores its numerator's coefficients and
        # then its denominator's q_1..q_k with q_0 held at 1, at
        # kRatAOffset[piece]; the degree columns say how many of each.
        rat_pieces = rat_a["pieces"]
        assert len(rat_pieces) == len(meta), "rational region-A piece count"
        a_coeffs = []
        a_offset = []
        a_numdeg = []
        a_dendeg = []
        for fit in rat_pieces:
            a_offset.append(len(a_coeffs))
            a_numdeg.append(fit["m"])
            a_dendeg.append(fit["k"])
            a_coeffs.extend(fmt(float(c)) for c in fit["p"])
            a_coeffs.extend(fmt(float(c)) for c in fit["q"])
        f.write("// The region-A rational route: a weighted minimax fit of each\n"
                "// shipped piece's interval, evaluated as num / (1 + t*horner(q, t))\n"
                "// in the same mapped argument the Chebyshev piece uses. It is an\n"
                "// offered alternative, not a replacement: the Chebyshev tables\n"
                "// above are the default route and are unchanged by its presence.\n")
        f.write("inline constexpr auto kRatACoeffs = std::to_array<double>({\n")
        for i in range(0, len(a_coeffs), 6):
            f.write("  " + ", ".join(a_coeffs[i:i + 6]) + ",\n")
        f.write("});\n")
        f.write("inline constexpr auto kRatAOffset = std::to_array<int>({\n")
        for i in range(0, len(a_offset), 12):
            f.write("  " + ", ".join(str(v) for v in a_offset[i:i + 12]) + ",\n")
        f.write("});\n")
        f.write("inline constexpr auto kRatANumDeg = std::to_array<int>({\n")
        for i in range(0, len(a_numdeg), 12):
            f.write("  " + ", ".join(str(v) for v in a_numdeg[i:i + 12]) + ",\n")
        f.write("});\n")
        f.write("inline constexpr auto kRatADenDeg = std::to_array<int>({\n")
        for i in range(0, len(a_dendeg), 12):
            f.write("  " + ", ".join(str(v) for v in a_dendeg[i:i + 12]) + ",\n")
        f.write("});\n")
        f.write("static_assert(std::size(kRatAOffset) == std::size(kPieces)\n"
                "                  && std::size(kRatANumDeg) == std::size(kPieces)\n"
                "                  && std::size(kRatADenDeg) == std::size(kPieces),\n"
                "              \"the rational region-A route must cover every piece\");\n\n")
        f.write("// Where the route's selector takes over in region A: the lowest of the\n"
                "// per-order ends of the region, which is the boundary the order is read\n"
                "// from the band seed above and from its own fit below. The tables above\n"
                "// still cover every piece from zero; this is the argument from which\n"
                "// naming the route changes any value, and the report says so rather than\n"
                "// claiming the wider domain the fits cover.\n")
        f.write(f"inline constexpr double kRatARouteLo = {fmt(rat_a['lo'])};\n")
        f.write(f"inline constexpr double kRatARouteHi = {fmt(rat_a['hi'])};\n\n")
        deg, cs, mono = b_cheb
        f.write("inline constexpr auto kBcoeffs = std::to_array<double>({"
                + ", ".join(fmt(c) for c in cs) + "});\n")
        f.write("inline constexpr auto kMonoBcoeffs = std::to_array<double>({"
                + ", ".join(fmt(c) for c in mono) + "});\n")
        f.write(f"inline constexpr int kBDeg = {deg};\n")
        f.write("\n")
        # The narrow partitions of the same regions, beside the shipped fits
        # rather than in place of them. See narrow_block_lines.
        for line in narrow_block_lines(narrow_a, narrow):
            f.write(line + "\n")
        f.write("\n")
        ext_deg, ext_cs, ext_mono = ext_cheb
        f.write("// The extended band (the per-range seed design): an F0 fit on\n")
        f.write("// [kExtendedBX0, kX0) evaluated by the same split Clenshaw; the\n")
        f.write("// upward recursion from it is certified per kmax tier - an order n\n")
        f.write("// takes the extended seed exactly when x >= kTierThresholds[n], the\n")
        f.write("// per-order dispatch thresholds (the certified values of the\n")
        f.write("// interval instrument, rounded up to the next double).\n")
        f.write(f"inline constexpr double kExtendedBX0 = {fmt(XNEW0)};\n")
        f.write("inline constexpr auto kExtendedBcoeffs = std::to_array<double>({\n")
        for i in range(0, len(ext_cs), 6):
            f.write("  " + ", ".join(fmt(c) for c in ext_cs[i:i + 6]) + ",\n")
        f.write("});\n")
        f.write("inline constexpr auto kMonoExtendedBcoeffs = std::to_array<double>({\n")
        for i in range(0, len(ext_mono), 6):
            f.write("  " + ", ".join(fmt(c) for c in ext_mono[i:i + 6]) + ",\n")
        f.write("});\n")
        f.write(f"inline constexpr int kExtendedBDeg = {ext_deg};\n")
        # The per-order dispatch threshold table: threshold[n] is the
        # certified boundary (rounded up to the next double) of the smallest
        # kmax row that covers the order n (the rows 4/8/16/32), so the
        # dispatch is a pure function of (n, x): order n takes the extended
        # seed exactly when x >= kTierThresholds[n].
        f.write("inline constexpr auto kTierThresholds = std::to_array<double>({\n")
        thresholds = []
        for n in range(MAX_ORDER + 1):
            tier = 0 if n <= 4 else (1 if n <= 8 else (2 if n <= 16 else 3))
            thresholds.append(fmt(float(TIER_BOUNDARIES_CERTIFIED[tier])))
        for i in range(0, len(thresholds), 6):
            f.write("  " + ", ".join(thresholds[i:i + 6]) + ",\n")
        f.write("});\n")
        f.write("static_assert(kRatARouteLo == kTierThresholds[0],\n"
                "              \"the rational region-A route takes over at the lowest \"\n"
                "              \"per-order end of region A\");\n")
        f.write("\n")

        # The rational region-B route, beside the Chebyshev one it is an
        # alternative to rather than a replacement for: same interval, same
        # mapped argument, both held to the same bar.
        p, q = rat_b["p"], rat_b["q"]
        f.write("// The rational minimax region-B seed: p(t)/q(t) over the same\n")
        f.write("// interval and in the same mapped argument t as kBcoeffs above.\n")
        f.write("// q is stored as q_1..q_k with its constant term held at 1.\n")
        f.write("inline constexpr auto kRatBnum = std::to_array<double>({\n")
        for i in range(0, len(p), 3):
            f.write("  " + ", ".join(fmt(c) for c in p[i:i + 3]) + ",\n")
        f.write("});\n")
        f.write(f"inline constexpr int kRatBnumDeg = {rat_b['m']};\n")
        f.write("inline constexpr auto kRatBden = std::to_array<double>({\n")
        for i in range(0, len(q), 3):
            f.write("  " + ", ".join(fmt(c) for c in q[i:i + 3]) + ",\n")
        f.write("});\n")
        f.write(f"inline constexpr int kRatBdenDeg = {rat_b['k']};\n")
        f.write("\n")
        # The two routes as the generator measures them: the stored count each
        # evaluates and the worst |F0 - fit| each reaches over [kX0, kX1], in
        # the double arithmetic the kernel evaluates them in, against the same
        # high-precision reference the fits themselves are validated against.
        # A swept maximum on a finite grid, not a bound: the bar both are
        # certified against is kRegionBFitBar, and both sit under it.
        f.write("// The two region-B routes as the generator measures them: the stored\n")
        f.write("// coefficients each evaluates and the worst |F0 - fit| each reaches\n")
        f.write("// over [kX0, kX1], in the kernel's double arithmetic and against the\n")
        f.write("// reference the fits are validated against. A swept maximum, not a\n")
        f.write("// bound: the bar both routes are certified against is kRegionBFitBar.\n")
        f.write(f"inline constexpr double kRegionBFitBar = {fmt(float(rat_b['bar']))};\n")
        f.write(f"inline constexpr int kRegionBFitChebStored = {rat_b['cheb_stored']};\n")
        f.write("inline constexpr double kRegionBFitChebDelivered = "
                f"{fmt(float(rat_b['cheb_delivered']))};\n")
        f.write(f"inline constexpr int kRegionBFitRatStored = {rat_b['stored']};\n")
        f.write("inline constexpr double kRegionBFitRatDelivered = "
                f"{fmt(float(rat_b['delivered']))};\n")
        f.write("\n")
        # The region-A route's two columns, measured the same way over the same
        # domain: every shipped piece for the Chebyshev route, every route piece
        # for the rational, against the same reference and in the same
        # arithmetic, so the counts and the errors are one comparison.
        f.write("// The two region-A routes as the generator measures them, over the\n"
                "// domain the two tables cover - every order's own pieces, from zero to\n"
                "// kX0: the stored coefficients each evaluates and the worst\n"
                "// |F_n - fit| each reaches, swept on a stride-1 grid of each piece's own\n"
                "// interval, in the kernel's double arithmetic and against the reference\n"
                "// the fits are validated against. A swept maximum on a finite grid, not\n"
                "// a bound: the bar both routes are certified against is kRegionAFitBar.\n"
                "// The rational route's selector takes over at kRatARouteLo - the lowest\n"
                "// of region A's per-order ends - and per order from that order's own\n"
                "// end, which is where the lane stops reading the order from its own fit\n"
                "// and documents the band's 3e-14 rather than the per-order 1e-15. Below\n"
                "// such an end the lane is documented at 1e-15, which these pieces do not\n"
                "// hold. The rational pieces were accepted below the bar rather than at\n"
                "// it, so the bar survives a certifying sweep on a different grid.\n")
        f.write(f"inline constexpr double kRegionAFitBar = {fmt(float(rat_a['bound']))};\n")
        f.write(f"inline constexpr int kRegionAFitChebStored = {rat_a['cheb_stored']};\n")
        f.write("inline constexpr double kRegionAFitChebDelivered = "
                f"{fmt(float(rat_a['cheb_delivered']))};\n")
        f.write(f"inline constexpr int kRegionAFitRatStored = {rat_a['stored']};\n")
        f.write("inline constexpr double kRegionAFitRatDelivered = "
                f"{fmt(float(rat_a['delivered']))};\n")
        # The scheme report: one row per (evaluation scheme, stored fit), the
        # degree and stored count that fit uses, and the bound each scheme was
        # MEASURED to deliver on it in each multiply-add route. The rows are
        # data: a scheme, a fit, a route or an end-to-end sweep added later is
        # another row here and nothing else.
        #
        # The domain is the fit's own interval (region A: every piece of every
        # order; region B: [kX0, kX1); the extended band: [kExtendedBX0, kX0)).
        # Region C has no stored fit and no row.
        f.write("// The evaluation schemes, one row per (scheme, stored fit): the\n")
        f.write("// degree and stored count the fit uses, and the bound each scheme is\n")
        f.write("// held to on it against the 60-digit reference, in each multiply-add\n")
        f.write("// route. scheme 0 = split Clenshaw, 1 = Horner; lane 0 = region A\n")
        f.write("// (kCoeffs), 1 = region B (kBcoeffs), 2 = the extended band\n")
        f.write("// (kExtendedBcoeffs). Each bound is the worst error a sweep over the\n")
        f.write("// fit's own interval reached, rounded up to the next power of two, so\n")
        f.write("// it is a bound and not the sweep's reading.\n")
        f.write("struct SchemeRow { int scheme, lane, region, deg, stored;\n")
        f.write("                   double fused, separate; };\n")
        f.write("inline constexpr auto kSchemeRows = std::to_array<SchemeRow>({\n")
        for (scheme, lane, region, deg, stored, _fm, _sm, fb, sb) in scheme_rows:
            f.write(f"  {{{scheme}, {lane}, {region}, {deg}, {stored}, {fmt(fb)}, "
                    f"{fmt(sb)}}},\n")
        f.write("});\n")
        f.write("\n}  // namespace boys::detail\n")

        # Float lane.
        f.write("\nnamespace boys::detail::f32 {\n\n")
        all_coeffs = []
        meta = []
        for n in range(MAX_ORDER + 1):
            for (a, b, deg, cs, _ms) in float_orders[n]:
                meta.append((n, a, b, deg, len(all_coeffs)))
                all_coeffs.extend(fmtf(c) for c in cs)
        f.write("inline constexpr auto kCoeffs = std::to_array<float>({\n")
        for i in range(0, len(all_coeffs), 6):
            f.write("  " + ", ".join(all_coeffs[i:i + 6]) + ",\n")
        f.write("});\n\n")
        f.write("struct OrderPiece { float a, b; int deg; int offset; };\n")
        f.write("inline constexpr auto kPieces = std::to_array<OrderPiece>({\n")
        for (n, a, b, deg, off) in meta:
            f.write(f"  {{{fmtf(mpf(a))}, {fmtf(mpf(b))}, {deg}, {off}}},  // F{n}\n")
        f.write("});\n")
        starts = []
        index = 0
        for n in range(MAX_ORDER + 1):
            starts.append(index)
            index += len(float_orders[n])
        starts.append(index)
        f.write("inline constexpr auto kPieceStart = std::to_array<int>({"
                + ", ".join(map(str, starts)) + "});\n")
        f.write("static_assert(std::size(kPieceStart) == kMaxOrder + 2,\n"
                "              \"piece-start table must cover kMaxOrder\");\n\n")
        deg, cs, _ms = b_cheb_f32
        f.write("inline constexpr auto kBcoeffs = std::to_array<float>({"
                + ", ".join(fmtf(c) for c in cs) + "});\n")
        f.write(f"inline constexpr int kBDeg = {deg};\n")
        f.write("\n}  // namespace boys::detail::f32\n")
        f.write("\n/// \\endcond\n")


def write_reference(path):
    import csv
    # The grid is the oracle a reader's long double is compared against, so it
    # is evaluated at GRID_DPS with GRID_TAIL_FLOOR and written with
    # GRID_DIGITS - decisively beyond the widest platform long double (see the
    # constants above), which is what makes the comparison a measurement of
    # the READER rather than of the machine that generated the grid.
    #
    # The argument is quantized to the double every reader parses (strtod:
    # tests read the x column as double) and the value is evaluated AT that
    # double, so argument rounding cannot enter the comparison at all. The
    # grid this replaced evaluated at the full-precision mpf argument while
    # readers used the double: at the high-sensitivity rows (d ln F / d ln x
    # ~ 31 near n = 32, x = 38.31) that alone put ~2.5e-15 relative into the
    # reference and dominated the cross-check on every platform, hiding what
    # the implementation itself contributed. A double is a dyadic rational,
    # so mpf(double) is exact and the argument costs nothing.
    with mp.workdps(GRID_DPS):
        # Dense over [0, 100]: the validated range of the committed grid.
        # Beyond it every F_n decays faster than the asymptotic tail (region C
        # covers it, and the series in boys_ref is only convergent for
        # x <= ~250).
        xgrid = [mpf(0), mpf("1e-12")]
        xgrid += [mpf(10) ** (mpf(-8) + mpf(10) * i / 48) for i in range(49)]  # 1e-8..1e2
        xgrid += [X0, X1, mpf("0.05"), mpf("0.1"), mpf("1"), mpf("2"), mpf("4"), mpf("6"),
                  mpf("8"), mpf("10"), mpf("13"), mpf("26.67"), mpf("29"), mpf("30"),
                  mpf("31"), mpf("33"), mpf("40"), mpf("50"), mpf("100")]
        # The extended band's design edges (the left edge and the certified
        # per-kmax boundaries) plus interior points across the band; the
        # superseded dispatch constants pin the edge of the slice whose
        # dispatch changed hands, whichever way the boundary moved.
        # A retired boundary is never dropped from the grid: each generation
        # of them stays pinned, newest set last.
        extras = [XNEW0] + [v for v in TIER_BOUNDARIES_CERTIFIED]
        extras += [mpf("1.5"), mpf("3"), mpf("5"), mpf("7"), mpf("9"), mpf("11")]
        extras += [mpf("1.857502623467682"), mpf("4.7030889427115925"),
                   mpf("10.655106119385133")]
        extras += [mpf("2.0136053436336927"), mpf("4.895982897243881"),
                   mpf("10.781772313649316")]
        extras += [mpf("2.0170701478602067"), mpf("4.8998472055064735"),
                   mpf("10.785490619744019")]
        # Deduplicate on the PARSED double, not on the mpf: the grid's
        # consumers index rows by exactly that pair (boys_test.cpp fetches the
        # reference for F_k by matching (k, row.x) as doubles), and the mpf
        # spelling of one design point is not the identity of its argument.
        # The committed grid before this change carried the case in the file:
        # the log sweep's top point is 10**2 and the extras' is 100, which the
        # general pow makes 100 + 2^-97 rather than equal to mpf("100"), so
        # x = 100.0 appeared twice for every order (2739 rows for 82 distinct
        # arguments).
        seen = set()
        arguments = []
        for v in xgrid + extras:
            parsed = float(v)
            if parsed not in seen:
                seen.add(parsed)
                arguments.append(mpf(parsed))
        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["n", "x", "value"])
            for n in range(MAX_ORDER + 1):
                for x in arguments:
                    # repr(float) is the shortest spelling that parses back to
                    # this exact double, so the reader's strtod returns the
                    # argument the value below was evaluated at.
                    w.writerow([n, repr(float(x)),
                                mp.nstr(boys_ref(n, x, GRID_TAIL_FLOOR, GRID_TERMS),
                                        GRID_DIGITS)])
    print(f"wrote {path} ({len(arguments) * (MAX_ORDER + 1)} rows)")


def find_clang_format():
    """The formatter behind the committed header's bytes.

    Order: $CLANG_FORMAT (the explicit pin - CI installs a versioned wheel and
    names it here), then PATH, then the Visual Studio install. A missing
    formatter is a hard error rather than a silent skip: the committed header
    IS clang-format output, so regenerating without one writes bytes the
    byte-identity gate then reports as drift, and the "drift" is the missing
    tool rather than the tables. That silent skip is how the macos leg failed
    with a diff full of unwrapped tables while nothing was actually wrong.
    """
    candidates = [os.environ.get("CLANG_FORMAT"), shutil.which("clang-format"),
                  (r"C:\Program Files\Microsoft Visual Studio\18\Community"
                   r"\VC\Tools\Llvm\x64\bin\clang-format.exe")]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "clang-format not found. The byte-identity gate compares against the "
        "committed header, which is clang-format output, so regenerating "
        "without a formatter produces false drift. Set CLANG_FORMAT to a "
        "formatter binary (CI pins clang-format 18.1.3) or put one on PATH.")


def format_header(path):
    """Makes the emitted header clang-format-clean (the byte-identity gate
    compares against the committed, formatted header) and LF (the repo norm).

    The path must live inside the repo tree: clang-format discovers the repo
    .clang-format relative to the file's directory, and a scratch copy outside
    the tree would be formatted with the LLVM default style instead.
    """
    subprocess.run([find_clang_format(), "-i", path], check=True)


def measure_scheme_rows(double_orders, b_cheb, ext_cheb, npts=SCHEME_MEASURE_POINTS):
    """The delivered bound of every (evaluation scheme, stored fit) pair, in
    each multiply-add route, measured against the reference over the fit's own
    interval.

    Rows are (scheme, lane, region, deg, stored, fused, separate, fused_bound,
    separate_bound) with scheme 0 = split Clenshaw and 1 = Horner, lane 0 =
    region A, 1 = region B, 2 = the extended band, and region the
    AccuracyRegion the lane serves. The first pair is the swept maximum over
    the measurement grid and is what the run prints; the second pair is that
    maximum published as a bound and is what the header carries. Region A is
    piecewise: its row carries the largest degree any piece uses and the worst
    value over every piece of every order."""
    worst_a = [[0.0, 0.0], [0.0, 0.0]]
    deg_a = stored_a = 0
    for n in range(MAX_ORDER + 1):
        for (a, b, deg, cs, ms) in double_orders[n]:
            w = fit_delivered(cs, ms, n, a, b, npts)
            for scheme in (0, 1):
                for route in (0, 1):
                    worst_a[scheme][route] = max(worst_a[scheme][route], w[scheme][route])
            if deg > deg_a:
                deg_a, stored_a = deg, len(cs)
    bdeg, bcs, bms = b_cheb
    worst_b = fit_delivered(bcs, bms, 0, X0, X1, npts)
    exdeg, excs, exms, _crossings = ext_cheb
    worst_e = fit_delivered(excs, exms, 0, XNEW0, X0, npts)
    rows = []
    for scheme in (0, 1):
        for (lane, region, deg, cs, w) in ((0, 0, deg_a, None, worst_a),
                                           (1, 1, bdeg, bcs, worst_b),
                                           (2, 0, exdeg, excs, worst_e)):
            stored = stored_a if cs is None else len(cs)
            rows.append((scheme, lane, region, deg, stored,
                         w[scheme][0], w[scheme][1],
                         scheme_bound(w[scheme][0]), scheme_bound(w[scheme][1])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", default="include/boys/boys_coefficients.hpp")
    parser.add_argument("--reference", default="tests/data/boys_reference.csv")
    parser.add_argument("--reference-only", action="store_true",
                        help="regenerate only the reference CSV (skip the fitting)")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed header/CSV are byte-identical to a "
                             "fresh generation (exits nonzero on drift)")
    parser.add_argument("--derive-partition", action="store_true",
                        help="derive a narrow partition from the proved truncation bound at "
                             "the quantum-chemistry target and print it, without fitting "
                             "anything (the design rule the bands are placed by)")
    parser.add_argument("--narrow-only", action="store_true",
                        help="fit only the narrow partition of region B and write its "
                             "header block to --narrow-out, formatted, without fitting "
                             "the rest of the table (the block's byte-identity check "
                             "without the hours-long full generation)")
    parser.add_argument("--narrow-out", default="",
                        help="where --narrow-only writes the block (default a scratch "
                             "name beside the committed header; clang-format must find "
                             "the repo .clang-format, so keep it inside the tree)")
    parser.add_argument("--derive-target", default="1e-14",
                        help="the target --derive-partition derives against (default the "
                             "quantum-chemistry 1e-14)")
    args = parser.parse_args()

    if args.derive_partition:
        return derive_partition_report(mpf(args.derive_target))

    if args.reference_only:
        os.makedirs(os.path.dirname(args.reference) or ".", exist_ok=True)
        write_reference(args.reference)
        return 0

    if args.narrow_only:
        # The narrow partitions' block alone. The full generation fits every
        # order of both lanes and both rational routes and takes hours; this
        # block is a handful of degree-10 fits plus a parallel walk per order,
        # so the block is reproduced on its own and compared against the
        # committed header's own lines without waiting for the rest of the
        # table. narrow_block_lines is shared with write_header, which is what
        # keeps the two from drifting. Region A's walk needs each order's own
        # end, which is where that order's last shipped piece ends, so the double
        # lane's region-A fits are taken here as well - they are the cheap part
        # of the full generation, and the rational routes, the extended band and
        # the scheme sweeps are what take the hours.
        narrow_a = narrow_region_a([fit_order(n) for n in range(MAX_ORDER + 1)])
        narrow = narrow_region_b()
        narrow_out = args.narrow_out or (args.header + ".narrow-tmp.hpp")
        os.makedirs(os.path.dirname(narrow_out) or ".", exist_ok=True)
        with open(narrow_out, "w", newline="\n") as f:
            f.write("#include <array>\n#include <cstddef>\n\n"
                    "namespace boys::detail {\n\n")
            for line in narrow_block_lines(narrow_a, narrow):
                f.write(line + "\n")
            f.write("\n}  // namespace boys::detail\n")
        format_header(narrow_out)
        print(f"wrote {narrow_out}")
        print(f"the narrow partition of region A: {sum(len(p) for p in narrow_a['orders'])} "
              f"pieces over {MAX_ORDER + 1} orders, degree {NARROW_DEG}, "
              f"{sum(len(p) for p in narrow_a['orders']) * (NARROW_DEG + 1)} stored")
        for order, pieces in enumerate(narrow_a["orders"]):
            print(f"  F{order:>2}: {len(pieces):>2} pieces, widths "
                  + " ".join(f"{_b - _a:.3f}" for (_a, _b, _d, _c, _m) in pieces))
        for scheme in range(NARROW_SCHEMES):
            print(f"  {SCHEME_NAMES[scheme]:14s} worst {narrow_a['worst'][scheme][0]:.6e} / "
                  f"{narrow_a['worst'][scheme][1]:.6e} (fused / separate), bound "
                  f"{narrow_a['bounds'][scheme][0]:.6e}")
        print(f"the narrow partition of region B: {len(narrow['pieces'])} pieces, "
              f"degree {narrow['pieces'][0][2]}, "
              f"{len(narrow['pieces']) * (narrow['pieces'][0][2] + 1)} stored")
        for (a, b, _deg, _cs, _ms) in narrow["pieces"]:
            print(f"  [{a!r}, {b!r})  width {b - a:.8f}")
        for scheme in range(NARROW_SCHEMES):
            print(f"  {SCHEME_NAMES[scheme]:14s} worst {narrow['worst'][scheme][0]:.6e} / "
                  f"{narrow['worst'][scheme][1]:.6e} (fused / separate), bound "
                  f"{narrow['bounds'][scheme][0]:.6e}")
        return 0

    print("fitting double lane (weighted region-A seeds, tol 5e-14, deg<=18) ...")
    double_orders = []
    for n in range(MAX_ORDER + 1):
        pieces = fit_order(n)
        double_orders.append(pieces)
        print(f"  F{n:2d}: {len(pieces)} intervals, {sum(len(p[3]) for p in pieces)} coeffs")

    print("fitting float lane (weighted region-A seeds, tol 1e-7, deg<=10) ...")
    float_orders = []
    for n in range(MAX_ORDER + 1):
        pieces = fit_order(n, f32=True)
        float_orders.append(pieces)

    print("fitting region B seeds ...")
    b_cheb = fit_order(0, region_b=True)
    b_cheb_f32 = fit_order(0, region_b=True, f32=True)

    print(f"fitting the narrow partition of region B (derived at "
          f"{mp.nstr(NARROW_TARGET, 3)}, degree {NARROW_DEG}) ...")
    narrow = narrow_region_b()
    for (a, b, _deg, _cs, _ms) in narrow["pieces"]:
        print(f"  [{a!r}, {b!r})  width {b - a:.8f}")
    for scheme in range(NARROW_SCHEMES):
        print(f"  {SCHEME_NAMES[scheme]:14s} worst {narrow['worst'][scheme][0]:.6e} / "
              f"{narrow['worst'][scheme][1]:.6e} (fused / separate), bound "
              f"{narrow['bounds'][scheme][0]:.6e}")

    print(f"walking the narrow partition of region A (both readings held, degree "
          f"{NARROW_DEG}) ...")
    narrow_a = narrow_region_a(double_orders)
    for order, pieces in enumerate(narrow_a["orders"]):
        print(f"  F{order:>2}: {len(pieces):>2} pieces, widths "
              + " ".join(f"{_b - _a:.3f}" for (_a, _b, _d, _c, _m) in pieces))
    for scheme in range(NARROW_SCHEMES):
        print(f"  {SCHEME_NAMES[scheme]:14s} worst {narrow_a['worst'][scheme][0]:.6e} / "
              f"{narrow_a['worst'][scheme][1]:.6e} (fused / separate), bound "
              f"{narrow_a['bounds'][scheme][0]:.6e}")

    # The rational region-B route, over the interval the Chebyshev region-B fit
    # was just given, so the two are compared on the same interval against the
    # same reference.
    rat_b = fit_region_b_rational(b_cheb)

    # The rational region-A route, over the shipped pieces' own intervals, so
    # the two routes over region A are compared piece for piece.
    rat_a = fit_region_a_rational(double_orders)

    print("fitting the extended band seed (F0 on [XNEW0, X0), tol 5e-14; "
          "the fixed-point loop) ...")
    ext_deg, ext_cs, ext_mono, ext_crossings = fit_extended_band(b_cheb=b_cheb)
    print(f"  extended band: deg {ext_deg}, {len(ext_cs)} coeffs")
    print("  generator-side certified crossings (1-ulp-exp model):")
    for k in (4, 8, 16, 32):
        print(f"    kmax {k:2d}: {mp.nstr(ext_crossings[k], 8)}")
    # The cross-check: the shipped boundary constants are the interval
    # instrument's certified values (250-dps interval arithmetic, the
    # authority); the generator's 30-dps design model is deliberately
    # conservative (its seed-width allowance is calibrated above the
    # instrument's measured widths), so each hardcoded certified value must
    # sit within 0.5 (x-units) of the model's converged crossing.
    for k in (4, 8, 16, 32):
        hard = TIER_BOUNDARIES_CERTIFIED[(4, 8, 16, 32).index(k)]
        if not (hard != mpf("inf") and abs(hard - ext_crossings[k]) <= mpf("0.5")):
            raise RuntimeError(f"extended band: kmax {k} hardcoded certified boundary "
                               f"{mp.nstr(hard, 8)} is outside the design model's converged "
                               f"crossing {mp.nstr(ext_crossings[k], 8)}")
    print("  hardcoded certified boundaries consistent with the converged model")

    print("measuring what each evaluation scheme delivers on each stored fit "
          f"({SCHEME_MEASURE_POINTS + 1} arguments per interval, both multiply-add "
          "routes) ...")
    scheme_rows = measure_scheme_rows(double_orders, b_cheb,
                                      (ext_deg, ext_cs, ext_mono, ext_crossings))
    for (scheme, lane, _region, deg, stored, fused, separate, fb, sb) in scheme_rows:
        print(f"  {SCHEME_NAMES[scheme]:>14s}  {LANE_NAMES[lane]:>14s}  deg {deg:2d} "
              f"({stored:2d} stored)  swept fused {fused:.3e} separate {separate:.3e}"
              f"  published fused {fb:.3e} separate {sb:.3e}")

    if args.check:
        import tempfile
        # The scratch header lives next to the committed one: clang-format
        # must discover the repo .clang-format, which it does relative to the
        # file's directory - a foreign temp dir would silently format the
        # scratch copy with the LLVM default style and every check would
        # false-positive on wrapping.
        tmp_header = args.header + ".check-tmp.hpp"
        tmp_reference = os.path.join(tempfile.gettempdir(), "boys_reference-check-tmp.csv")
        try:
            write_header(tmp_header, double_orders, float_orders, b_cheb, b_cheb_f32,
                         (ext_deg, ext_cs, ext_mono), scheme_rows, rat_b, rat_a, narrow, narrow_a)
            format_header(tmp_header)
            write_reference(tmp_reference)
            ok = True
            for generated, committed in ((tmp_header, args.header),
                                         (tmp_reference, args.reference)):
                with open(generated, "rb") as fg, open(committed, "rb") as fc:
                    if fg.read() != fc.read():
                        ok = False
                        print(f"DRIFT: {committed} differs from a fresh generation")
            return 0 if ok else 1
        finally:
            for scratch in (tmp_header, tmp_reference):
                if os.path.exists(scratch):
                    os.remove(scratch)

    os.makedirs(os.path.dirname(args.header) or ".", exist_ok=True)
    write_header(args.header, double_orders, float_orders, b_cheb, b_cheb_f32,
                 (ext_deg, ext_cs, ext_mono), scheme_rows, rat_b, rat_a, narrow, narrow_a)
    format_header(args.header)
    print(f"wrote {args.header}")

    os.makedirs(os.path.dirname(args.reference) or ".", exist_ok=True)
    write_reference(args.reference)

    # Documentation: per-kmax boundaries (the validated configuration uses the
    # fixed kmax = 32 values; the smaller values are the formula (V&S-style)
    # boundaries - the measured ones are reported by the boundary harness).
    print("per-kmax boundaries (documentation):")
    for kmax in (0, 4, 8, 12, 16, 24, 32):
        p = mpf(1)
        for n in range(kmax):
            p *= (n + mpf("0.5"))
        x0k = max(mpf(1), p ** (mpf(1) / max(kmax, 1)))
        print(f"  kmax={kmax:2d}: x0={mp.nstr(x0k, 8)}")


if __name__ == "__main__":
    sys.exit(main())
