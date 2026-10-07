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
    dispatched per kmax tier; the certified per-kmax thresholds kTierThresholds
    are derived below and emitted into the committed header.
  - Region C [x1, inf): asymptotic, no coefficients.
  - Boundaries: fixed kmax=32 values x0 = 11.899848152108484,
    x1 = 28.989337738820740 (the configuration validated end-to-end against
    the reference grid; the per-kmax table is printed for documentation).
    The published per-kmax boundary formula (the table's comparison column)
    is VikhamarSandberg2026, Eqs. 25/13.
  - Float lane: same structure, tolerance 1e-7, degree cap 10.
  - Region B carries a second, certified route beside the Chebyshev one: a
    rational minimax fit, P(t)/Q(t), fitted by the Remez exchange
    ([Remez1934]) over the
    same interval and in the same mapped argument, and cross-checked against
    Lawson's algorithm ([Lawson1964], with the Sanathanan-Koerner denominator
    weight, [SanathananKoerner1963]). The
    region-B routes' stored counts and delivered errors are measured in the
    kernel's double arithmetic against the same reference the fits are
    validated against, and emitted beside the tables.

The DCT normalization is the standard one: c_0 gets 1/(deg+1), all other
coefficients - including the top one - get 2/(deg+1). Halving c_deg as well
makes the fits diverge. The committed
header must match this script's output byte for byte for a given tolerance
configuration.
"""
import argparse
import concurrent.futures
import math
import multiprocessing
import os
import shutil
import struct
import subprocess
import sys
from fractions import Fraction
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

# The extended band: a second F0 fit serving
# [XNEW0, X0) with the region-B-style upward recursion, dispatched per kmax
# tier (4/8/16/32). The boundaries are certified under the band's range-uniform
# a-priori seed bound delta_0' = 1.0641e-15 (R_hat 1.0116e-15 forward
# rounding + tau 5.2538e-17 truncation tail, no x-sampling) and hardcoded
# below as the dispatch constants; the fit tolerance is the same 5e-14
# class as the shipped region-B fit, so the delivered seed width keeps the
# recursion envelope <= 5e-14 from each per-kmax boundary up.
XNEW0 = mpf("1.0855252345349333")
EXTENDED_DEG_LADDER = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72, 96)

# The certified per-kmax boundaries of the extended band (kmax 4/8/16/32),
# as certified under the band's range-uniform
# a-priori seed bound R_hat + tau = 1.0641375040661759464e-15 (R_hat the
# evaluation-rounding sum over the seed's coded roundings, tau the
# polynomial-definition term). tau is computed rather than typed:
# (u/2)*sum_j|c_j| for the stored-coefficient rounding plus Tail_proj for
# truncation and aliasing, Tail_proj the Chebyshev interpolant bound
# 4*M(rho)*rho^-d/(rho-1) minimised over rho (Trefethen2019, ch. 8): the
# coefficients interpolate at Chebyshev-Gauss nodes, where the aliasing is
# at most the size of the tail, so the interpolant constant applies rather
# than the (1 + Lambda_d) form, which charges the aliasing the
# worst-case operator norm a decaying tail never attains and is 2.0059x
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

# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------
# Almost nothing here waits on anything else here: one order's fits, one
# lane's pieces, the degree pairs a rational scan visits and the delivered
# errors a sweep measures are independent jobs, and each is a walk through
# high-precision mpmath - which is why a serial run of this script is an hour
# of clock rather than the second a build step would be. The jobs go to a
# process pool and come back in submission order, so the tables are a function
# of the work and not of the machine's core count: one worker runs the same
# jobs in one process and writes the same bytes.
#
# A worker starts fresh, so it is either told what it needs or carries what it
# needs with it - mpmath's precision is per-process state that a spawn does
# not pass on, and every value that crosses the boundary is a plain number, a
# list of them, or an mpmath value, all of which pickle exactly.

JOBS = None  # --jobs; None means one worker per CPU


def worker_count():
    if JOBS is not None:
        return max(1, JOBS)
    return os.cpu_count() or 1


def run_jobs(fn, jobs):
    """Run fn over jobs; the results come back in submission order.

    That order is part of the answer. Every caller reduces these results with
    a scan - a running maximum, a running total, the first piece that set one
    - and a reduction over completion order would put a different number in
    the tables on a machine with a different core count. map() returns them in
    the order the loop it replaces produced them, and one worker is that loop.
    """
    jobs = list(jobs)
    if len(jobs) <= 1 or worker_count() <= 1:
        return [fn(spec) for spec in jobs]
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(worker_count(), len(jobs))) as pool:
        return list(pool.map(fn, jobs))


def _fit_order_job(spec):
    """One order's fits, in a worker process."""
    n, f32, region_b = spec
    return fit_order(n, f32=f32, region_b=region_b)


def _fit_delivered_job(spec):
    """One piece's delivered error under both schemes and both routes."""
    cs, ms, n, a, b, npts = spec
    return fit_delivered(cs, ms, n, a, b, npts)


def _rat_best_job(spec):
    """One degree pair of a rational scan, in a worker process.

    The grid, the mapped arguments and the reference travel with the job: they
    are the same for every pair a scan visits, and shipping a few hundred
    kilobytes costs a fraction of a second against the minutes a pair's own
    fit costs.
    """
    m, k, indices, ts, xs, ref, ws = spec
    mp.dps = RAT_DPS  # the scan's working precision; a spawn does not carry it
    return rat_best(m, k, indices, ts, xs, ref, ws)


def _rat_a_piece_job(spec):
    """One region-A piece's rational fit, in a worker process.

    The piece's grid, its reference and its weights are built here rather than
    shipped. They are four hundred mpmath values each and they are a function
    of this job's own arguments, so the worker computes exactly what the loop
    it replaces computed at that piece - no other piece shares them, and the
    argument the tables are written from is the one this job was given.
    """
    n, a, b, cs = spec
    with mp.workdps(RAT_DPS):
        return rat_a_fit_piece(n, mpf(a), mpf(b), cs)


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
    sum.

    The fused step is spelled by rat_fma rather than by math.fma, for the same
    reason that function exists: math.fma is Python 3.13 and later, and this
    script's output is byte-compared on machines that need not run the same
    Python. The two are the same operation - rat_fma recovers the product
    exactly and rounds the sum once, which is what a fused multiply-add is.
    """
    return rat_fma(a, b, c) if fused else a * b + c


def route_step32(fused, a, b, c):
    """One multiply-add in binary32 as the kernel's arithmetic spells it: the
    fused step rounds the sum once, the separate step rounds the product and
    then the sum.

    The fused step is spelled by fma32 rather than by a library call, for the
    same reason route_step spells its own with rat_fma: math.fma is Python 3.13
    and later, and this script's output is byte-compared on machines that need
    not run the same Python. The two are the same operation - a*b + c with
    binary32 inputs is exact in binary64, so rounding that sum once to
    binary32 is the correctly rounded fused step and not an approximation of
    it.

    The separate step is two roundings of the same expression, and it is
    spelled here as the two roundings rather than as a single expression the
    host might contract: the product rounds to binary32, then the sum of two
    binary32 values rounds to binary32. Rounding that sum in binary64 first
    cannot change the binary32 result. Where the two addends differ by more
    than 53 binades the binary64 sum is already the larger addend, and an
    addend under half an ulp of the other rounds the sum to the other one in
    binary32 as well; where they do not, the binary64 sum is exact and the
    single rounding that follows is the one the kernel makes.
    """
    return fma32(a, b, c) if fused else r32(r32(a * b) + c)


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


def fit_order(n, region_b=False, f32=False, maxdeg=None):
    tol = TOL_FLOAT if f32 else TOL_DOUBLE
    if maxdeg is None:
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
    print()

    # The uniform grid. It is not a walk the bound placed - one width for every
    # interval is what makes the locate a multiply - so what this report prints
    # is the ladder the width is chosen over, the cell law's degrees on the row
    # that was chosen, and the join the closed form's own error sets.
    for lane in ("double", "float"):
        bound = FLAT_CELL_BOUND[lane]
        budget = bound - FLAT_HALF_ULP[lane]
        cap = FLAT_DEG if lane == "double" else F32_FLAT_DEG
        print(f"the {lane} lane's uniform grid: every cell held to "
              f"{mp.nstr(bound, 3)} less half an ulp of 1 "
              f"({mp.nstr(FLAT_HALF_ULP[lane], 3)}), so the cells are placed at "
              f"{mp.nstr(budget, 4)}, at the read cap {cap}:")
        for width in FLAT_LADDER:
            first = uniform_cell_degree(mpf(0), width, budget, cap)
            print(f"  width {mp.nstr(width, 4):>7}  first cell "
                  + (f"degree {first:2d}" if first is not None
                     else "needs more than the read cap"))
        grid = uniform_grid(lane)
        per_order = sum(d + 1 for d in grid["degs"])
        print(f"  the join: the closed form's own error at order {MAX_ORDER} against "
              f"{mp.nstr(FLAT_CLOSED_BOUND[lane], 3)} is at "
              f"{mp.nstr(grid['join'], 10)}, so the grid is "
              f"{grid['intervals']} intervals of width {mp.nstr(grid['width'], 4)} "
              f"reaching {mp.nstr(grid['hi'], 8)}")
        print(f"  stored {per_order * (MAX_ORDER + 1)} coefficients per table "
              f"({mp.nstr(grid['width'], 4)} x {grid['intervals']} x "
              f"{mp.nstr(mp.mpf(per_order) / grid['intervals'], 3)} mean), read per "
              f"evaluation {min(grid['degs']) + 1}..{max(grid['degs']) + 1}, degrees "
              f"{min(grid['degs'])}..{max(grid['degs'])}")
        print()
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

# The uniform table's read cap: the largest degree any cell of the grid is
# stored at. The cells have their own degrees, derived below (uniform_grid), one
# per interval; this is the ceiling they are placed under, and it is the read
# per evaluation this table publishes, which the derivation does not raise.
#
# A ceiling rather than one degree for the whole table because the table's cost
# is storage, and one degree spends on every interval what only the first one
# needs: F_n is largest and slowest-decaying at the origin, so the proved
# truncation bound only falls as the order or the argument rises.
#
# THE DEGREE MUST BE EVEN, and that is a constraint of the kernel rather than a
# preference of this generator. ClenshawSplit's odd part seeds at c[2m-1] and its
# finalization assumes the m odd coefficients an even degree has, so an odd
# degree drops its leading odd coefficient: the route then evaluates a different,
# lower-degree polynomial, at full accuracy. This table was first fitted at
# degree 7, which is odd, and the route's Clenshaw path asserted in debug and
# returned those wrong values in release while a published bound for it sat in
# the header. Every other table here is emitted even, but that was a convention
# nothing enforced - see the static_assert emitted beside every degree below.
#
# What the cap buys, measured in binary64 over both schemes and both
# multiply-add routes at order 0 on the first interval - the worst cell of the
# grid, being the worst order on the worst interval:
#
#   degree 4    8.373e-11     on [0, 1/7): too coarse by three orders
#   degree 6    1.998e-15     on [0, 1/7): clears 1e-14, under one order of margin
#   degree 7    1.110e-16     on [0, 1/7): at the floor, and odd, so not readable
#   degree 8    at or below the floor, and even; 1.110e-16 on [0, 1/4) too
#
# The decay is not geometric: 4 -> 6 is a factor of 42 000, 6 -> 7 only 18. So
# degree 6 is not the cheap option it looks like - it spends the margin the
# derived partitions are placed to hold - and 8 costs one multiply-add per order
# over 7 for a fit the lane's own arithmetic cannot distinguish from exact.
#
# The grid this caps - the width, the interval count, the join and every cell's
# degree - is derived from the lane's own bound, format and cap by uniform_grid.
FLAT_DEG = 8

# The rule the kernel's split Clenshaw summation imposes on every degree it reads,
# and the check that enforces it where the degrees are published.
#
# ClenshawSplit's odd part seeds at c[2m-1] and its finalization assumes the m odd
# coefficients an even degree has. An odd degree therefore drops its leading odd
# coefficient, and the route returns a different, lower-degree polynomial - at
# full accuracy, so nothing about the value looks wrong. The kernel asserts the
# rule at run time and that assert compiles out under NDEBUG.
#
# The uniform table was first fitted at degree 7, which is odd, and the route's
# Clenshaw path asserted in debug and returned those wrong values in release
# while a published bound for it sat in the header. Every other table here
# happened to be emitted even, but that was a convention and no check held it:
# the generator's own comment claimed "the generator only emits such degrees",
# which was true until it was not. So each degree constant below carries its own
# static_assert, emitted beside it, and a generation that breaks the rule fails
# to compile in every build, host and device alike, rather than shipping a table
# nothing reads correctly.
CLENSHAW_DEGREE_RULE = (
    "a degree the split Clenshaw summation reads must be even: its odd part "
    "seeds at c[2m-1] and assumes the m odd coefficients an even degree has, so "
    "an odd degree would drop its leading odd coefficient and evaluate a "
    "different, lower-degree polynomial accurately"
)


def deg_assert(name):
    """The check emitted beside a degree constant ClenshawSplit reads."""
    return (f"static_assert({name} % 2 == 0,\n"
            f"              \"{CLENSHAW_DEGREE_RULE}\");\n")


# The float lane's uniform table is its own grid, derived by the same three laws
# from the lane's own bound, its own format and its own read cap (uniform_grid).
# Its width, interval count, join and degrees are the double lane's only by
# coincidence: the bound imposed is 1e-7 and not 1e-15, so the grid is shorter
# and its cells are lower-degree.
#
# The worst cell is order 0 on the first interval, at every degree: 0 <= t^(2n)
# <= 1 on the integration range gives |F_n(z)| <= |F_0(z)| for every real z, so
# order 0 dominates, and F_0's derivative is -int t^2 e^(-x t^2) dt, negative
# everywhere, so it decreases and the first interval dominates.
#
# What places this cap is the lane's arithmetic and not its truncation. Half an
# ulp of 1 in binary32 is 5.96e-8 against the lane's 1e-7 bar, so the cell law
# has 4.04e-8 to spend on truncation and the route's own rounding spends the
# rest: on the derived grid's first cell the delivered figure is 7.747e-8 in the
# split Clenshaw route and 3.745e-8 in Horner, against a truncation bound of
# 8.436e-9. The lane's figure is its rounding at every admissible degree, which
# is why the cap is the smallest one the summation reads.
#
# Degree 3 is not admissible, and that is the kernel's property and not a
# preference: ClenshawSplit takes its top odd coefficient at c[2m-1] and asserts
# an even degree of at least 4 (boys_impl.hpp), so a degree-3 fit is not read as a
# degree-3 fit - the reading measured 2.282e-2 on that cell, with c[1]'s term
# carrying what c[3]'s should. The smallest admissible degree is 4, whose first
# neglected coefficient on the double lane's first interval is 8.289e-11: nearly
# three orders below that lane's own arithmetic floor, so what the table delivers
# there is the lane's rounding rather than this truncation. Degree 6 delivers the
# same figure on every cell tried and costs two more coefficients per order and
# two more multiply-adds per evaluation, which is why the cap stops at 4.
F32_FLAT_DEG = 4


# ---------------------------------------------------------------------------
# The uniform grid: the geometry and the per-cell degrees, derived
# ---------------------------------------------------------------------------
# Every other partition here equalises its pieces' error by letting their widths
# grow with the argument. This one cannot: one width for every interval is what
# makes the locate floor(x * (1 / width)) instead of a scan of piece edges, and
# that is the partition's whole point. So the grid has one free parameter, and
# the cell that needs the most degree fixes it. Three laws, and nothing below is
# a choice:
#
#   * the cell law. A cell is read directly - nothing carries its error anywhere
#     - so what a caller receives from it is the cell's own truncation plus the
#     lane's rounding, and the two together must fit inside the bound the cell is
#     held to. The criterion therefore governs the truncation against the bound
#     less half an ulp of 1, and a cell's degree is the smallest admissible even
#     degree whose proved truncation bound (apriori_truncation_bound, the law
#     that places every derived piece here) holds it. This is the criterion a
#     rung moves: the effective-degree machinery spends the dropped-coefficient
#     tail of a cell against (m - 1) times the bound, and a cell's own degree and
#     its own coefficients are what it scans - which is all a cut needs.
#
#   * the width. The widest dyadic interval whose first cell that criterion still
#     carries at the read cap. Widest because for one read cap the storage falls
#     monotonically as the width grows - the interval count falls as the join
#     over the width, while the degree the first cell needs grows far more slowly
#     - and dyadic because the locate multiplies by the width's reciprocal: 1 /
#     2^-p is exact in binary arithmetic, so the interval an argument falls in is
#     a property of the argument and not of a rounding. The kernel asserts that
#     same requirement from the other side, kPerUnit * kFlatHi ==
#     kFlatIntervals, which a dyadic pair satisfies exactly and which 1/7
#     satisfies only by the luck of its reciprocal.
#
#   * the join, where the grid stops and the closed form takes over. The closed
#     form's own error at order n is 0.5 x^-(n+1/2) Gamma(n+1/2, x): largest at
#     the top order, strictly falling in x. The join is the root of that against
#     the bound the closed form is held to, rounded up to a whole number of
#     intervals.
#
# Every figure is a function of the lane's bound, the lane's format and the
# lane's read cap, so a lane with a different target derives its own grid.
#
# The bounds are the contract's, in boys.hpp's table. A grid spans region A, the
# extended band and part of region B at once, and the table is one table for
# every lane that names the partition, so the tightest bound over both binds
# every cell: the double lane answers to the single lane's region A figure. The
# closed form answers to the region it serves above the join.
FLAT_CELL_BOUND = {"double": mpf("1e-15"), "float": TOL_FLOAT}
FLAT_CLOSED_BOUND = {"double": mpf("5.5e-14"), "float": mpf("1.5e-7")}
# Half an ulp of 1 in each lane's format: the floor no fit's arithmetic reaches
# below, and the part of the bound the cell law must leave to it.
FLAT_HALF_ULP = {"double": mpf(2) ** -53, "float": mpf(2) ** -24}
# The widths the grid is derived over, widest first. Dyadic only, for the
# locate's exactness above, and the search takes the first row both laws accept,
# so the ladder's top is above anything a lane asks for and its bottom is below
# anything a lane's bound asks for: a lane whose arithmetic binds needs rows the
# bound alone would never choose, and a table stored on one of them is still a
# table the partition can offer.
FLAT_LADDER = (mpf(8), mpf(4), mpf(2), mpf(1), mpf(1) / 2, mpf(1) / 4,
               mpf(1) / 8, mpf(1) / 16, mpf(1) / 32)
# The degrees a cell may be fitted at, ascending. Even and at least 4: the split
# Clenshaw summation reads the odd coefficients up to c[deg-1], so an odd degree
# drops its leading odd one and evaluates a different polynomial (see
# CLENSHAW_DEGREE_RULE). The top of the ladder is above any cap below, so the
# cell law's answer is the cap and not the ladder whenever the cell is easy.
FLAT_CELL_DEGREES = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)


def closed_form_error(order, x):
    """The closed form's own error: 0.5 x^-(n+1/2) Gamma(n+1/2, x).

    Every route ends in the one-term asymptotic - it starts from 0.5 sqrt(pi/x)
    and never adds the exp(-x)/2 source term of the three-term recurrence - so
    what it omits is exactly that analytic tail. The omission is an absolute
    error and not a rounding one, which is what makes the join a root of an
    equality rather than a stability question.
    """
    return (mpf("0.5") * mpf(x) ** (-(order + mpf("0.5")))
            * mp.gammainc(order + mpf("0.5"), mpf(x), mp.inf))


def uniform_join(bound, order=MAX_ORDER):
    """The smallest x whose closed form is inside `bound` at the top order.

    The tail is largest at the top order and falls in x, so the root of this one
    equation is the join for every order at once.
    """
    lo, hi = mpf(1), mpf(400)
    for _ in range(200):
        mid = (lo + hi) / 2
        if closed_form_error(order, mid) <= bound:
            hi = mid
        else:
            lo = mid
    return hi


def uniform_cell_degree(a, b, budget, cap):
    """The smallest ladder degree the cell [a, b) is carried at under `budget`.

    None when no admissible degree up to the cap holds it, which is what tells a
    width that it cannot be read at all.
    """
    for deg in FLAT_CELL_DEGREES:
        if deg > cap:
            return None
        if apriori_truncation_bound(0, a, b, deg)[0] <= budget:
            return deg
    return None


def uniform_degrees(width, intervals, budget, cap):
    """Every interval's degree over a grid of `intervals` cells of `width`.

    For one width the bound falls as the argument rises, so the degree vector
    falls with the interval index and both searches below are over monotone
    predicates: bisection applies. The walk is linear in the number of degree
    steps and not in the number of intervals - the intervals between two steps
    are settled by one bisection rather than by one bound each, which is what
    keeps a derivation over a hundred cells to a few tens of bound evaluations.
    """
    out, deg, iv = [], cap, 0
    while iv < intervals:
        while deg > FLAT_CELL_DEGREES[0]:
            lower = max(d for d in FLAT_CELL_DEGREES if d < deg)
            a = mpf(iv) * width
            if uniform_cell_degree(a, a + width, budget, lower) is not None:
                deg = lower
            else:
                break
        end = intervals
        if deg > FLAT_CELL_DEGREES[0]:
            lower = max(d for d in FLAT_CELL_DEGREES if d < deg)
            # The first interval the next degree down holds at: this degree's
            # run ends there, and the predicate is monotone in the argument.
            lo, up = iv, intervals
            while up - lo > 1:
                mid = (lo + up) // 2
                a = mpf(mid) * width
                if uniform_cell_degree(a, a + width, budget, lower) is not None:
                    up = mid
                else:
                    lo = mid
            end = up
        out.extend([deg] * (end - iv))
        iv = end
    return out


def uniform_row(lane, width):
    """The grid a lane derives at one width, or None if the cap cannot carry it.

    The bound's majorant is F_0 at the ellipse's leftmost point, F_0 is positive
    and decreasing on the whole real line, and |F_{n+1}| <= |F_n| pointwise, so
    the bound falls as the order or the argument rises: the first cell of order 0
    is the worst cell the *placement* has to satisfy, and a width it cannot carry
    is a width no other cell can save.
    """
    bound = FLAT_CELL_BOUND[lane]
    budget = bound - FLAT_HALF_ULP[lane]
    cap = FLAT_DEG if lane == "double" else F32_FLAT_DEG
    if uniform_cell_degree(mpf(0), width, budget, cap) is None:
        return None
    join = uniform_join(FLAT_CLOSED_BOUND[lane])
    intervals = int(mp.ceil(join / width))
    return {"lane": lane, "bound": bound, "budget": budget, "cap": cap, "join": join,
            "width": width, "intervals": intervals, "hi": intervals * width,
            "degs": uniform_degrees(width, intervals, budget, cap)}


@lru_cache(maxsize=None)
def uniform_grid(lane="double"):
    """The widest grid the lane's read cap carries by the proved bound.

    The placement law's answer and not the whole criterion: what a lane's own
    arithmetic delivers on that grid is a second question, and flat_table asks
    it - a cell placed by the bound can still be delivered outside it where the
    route's summation rounding is what sets the figure, and that rounding is not
    monotone in the argument the way the bound is.
    """
    for width in FLAT_LADDER:
        row = uniform_row(lane, width)
        if row is not None:
            return row
    budget = FLAT_CELL_BOUND[lane] - FLAT_HALF_ULP[lane]
    cap = FLAT_DEG if lane == "double" else F32_FLAT_DEG
    raise RuntimeError(
        f"the uniform grid's ladder carries no width at {mp.nstr(budget, 4)}: every "
        f"dyadic width in {[mp.nstr(w, 3) for w in FLAT_LADDER]} needs a degree above "
        f"the {cap} this lane's table reads")


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
# order 1. For a fixed x the ratio x^n / prod(j + 1/2) rises with n only until n
# passes x - 1/2 and falls after, so below the band edge its largest value at an
# order of 3 or more is order 3's, which at the edge itself is 0.68 and so is
# still short of 1. The walk holds the envelope rather than that figure so that
# no piece's reading depends on which order an entry seeds from.
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
    deg = NARROW_DEG
    lines = [
        f"// The narrow partition of region A: every order's interval cut at degree {deg}, each",
        "// piece as wide as the tighter of its two readings lets it be",
        f"// (region_a_piece_budget). Same fields and lookup as kPieces, but {total} rows",
        f"// to the shipped {shipped_rows}: {total * (deg + 1)} coefficients against {shipped_stored}.",
        f"inline constexpr int kNarrowADeg = {deg};",
        deg_assert("kNarrowADeg"),
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
    lines.append("static_assert(std::size(kNarrowAPieceStart) == kMaxOrder + 2 &&\n"
                 "                  std::size(kNarrowAPieces) ==\n"
                 "                      static_cast<std::size_t>(kNarrowAPieceStart[kMaxOrder + 1]),\n"
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
    lines.append("// The narrow region-A partition's certification rows: the bound each scheme")
    lines.append("// delivers on it in each multiply-add route, worst over its pieces, rounded")
    lines.append("// up to a power of two so it bounds a sweep and not only its own. The")
    lines.append("// pieces are measured at their own size; the recursion gain is budget-bounded.")
    lines.append("struct NarrowARow { int scheme, deg, pieces, stored;")
    lines.append("                    double fused, separate; };")
    lines.append("inline constexpr auto kNarrowARows = std::to_array<NarrowARow>({")
    for scheme in range(NARROW_SCHEMES):
        bounds = narrow_a["bounds"][scheme]
        lines.append(f"  {{{scheme}, {deg}, {total}, {total * (deg + 1)}, "
                     f"{fmt(bounds[0])}, {fmt(bounds[1])}}},")
    lines.append("});")
    return lines


def flat_order(job):
    """One order's uniform-interval fits over [0, kFlatHi).

    The grid travels with the job rather than being read from the module: a
    worker starts fresh, so a grid derived here would be a second derivation
    free to disagree with the one the block writer emits the offsets from. Every
    interval is kFlatWidth wide, so locating the interval an argument falls in
    is one multiply and a truncation, where the derived partitions need a piece
    scan. That is half of what this table is for; the other half is that every
    order is fitted independently, so a ladder is 33 independent polynomials and
    not a recurrence that has to be walked in order. Each cell is fitted at its
    own degree, which is why the interval list carries one.
    """
    n, width, degs = job
    intervals = []
    worst = [[0.0, 0.0] for _ in range(NARROW_SCHEMES)]
    for iv, deg in enumerate(degs):
        a = iv * width
        b = a + width
        cm = cheb_coeffs(n, a, b, deg)
        cs = [float(c) for c in cm]
        ms = [float(c) for c in cheb_to_monomial(cm)]
        intervals.append((float(a), float(b), deg, cs, ms))
        w = fit_delivered(cs, ms, n, a, b, SCHEME_MEASURE_POINTS)
        for scheme in range(NARROW_SCHEMES):
            for route in (0, 1):
                worst[scheme][route] = max(worst[scheme][route], w[scheme][route])
    return n, intervals, worst


def flat_table(lane="double"):
    """The uniform table as stored fits, with its measured bound.

    Every order is walked, measured and returned independently, so the pool
    order cannot reach the result. The bounds are this table's own delivered
    figures per scheme and multiply-add route, rounded up the way every other
    table's are. The grid the fits were placed on travels back with them: the
    block writer emits the degrees that were fitted rather than a second
    derivation of them.

    The ladder is walked from the widest width down and a row is taken when its
    own measured table holds the bound, because the bound places a cell and the
    measurement is what certifies it - and the two are not the same question.
    The figure a cell delivers is its truncation plus its lane's summation
    rounding, the bound is monotone in the argument and the rounding is not: a
    lane whose bar is close to its own arithmetic floor can be delivered outside
    the bound at a cell the bound is comfortable with. A narrower width lowers
    the rounding because it lowers the coefficients it sums, so the ladder is
    the search and this is its acceptance test. A ladder that carries no row at
    all is raised rather than published.
    """
    for width in FLAT_LADDER:
        grid = uniform_row(lane, width)
        if grid is None:
            continue
        jobs = [(n, width, grid["degs"]) for n in range(MAX_ORDER + 1)]
        with multiprocessing.get_context("spawn").Pool(
                min(len(jobs), NARROW_WORKERS)) as pool:
            out = pool.map(flat_order, jobs)
        per_order = [intervals for _n, intervals, _w in out]
        worst = [[0.0, 0.0] for _ in range(NARROW_SCHEMES)]
        for _n, _intervals, w in out:
            for scheme in range(NARROW_SCHEMES):
                for route in (0, 1):
                    worst[scheme][route] = max(worst[scheme][route], w[scheme][route])
        if all(worst[s][r] <= grid["bound"]
               for s in range(NARROW_SCHEMES) for r in (0, 1)):
            bounds = [[scheme_bound(worst[s][r]) for r in (0, 1)]
                      for s in range(NARROW_SCHEMES)]
            return {"orders": per_order, "worst": worst, "bounds": bounds, "grid": grid}
        print(f"    the {lane} uniform table does not hold its bound at width "
              f"{mp.nstr(width, 3)}: worst "
              f"{max(max(r) for r in worst):.6e} against "
              f"{mp.nstr(grid['bound'], 3)}; trying the next width down")
    raise RuntimeError(
        f"the {lane} uniform table holds {mp.nstr(FLAT_CELL_BOUND[lane], 3)} at no "
        f"dyadic width in {[mp.nstr(w, 3) for w in FLAT_LADDER]}: the lane's own "
        f"rounding is above the bound at every grid the ladder carries")


def flat_grid_offsets(grid):
    """Where each interval's block starts, plus the end of the last one.

    The degrees are per interval, so the table has no single stride and an
    interval's block length is (kMaxOrder + 1) * (its degree + 1). The offsets
    are a prefix sum of those lengths; the block writer and the fits are the
    same grid, so the offsets describe the rows that were stored.
    """
    offsets = [0]
    for deg in grid["degs"]:
        offsets.append(offsets[-1] + (MAX_ORDER + 1) * (deg + 1))
    return offsets


def flat_block_lines(flat):
    """The uniform table as the header stores it.

    Interval-major, [interval][order][coefficient], which is the opposite of
    the derived tables' order-major layout and is the point of this one: an
    argument's whole ladder is one contiguous block, so the 33 polynomials a
    batch entry reads for one argument sit in consecutive cache lines instead
    of 33 rows apart. The order is the inner index and the interval the outer
    one, so an interval's block is kMaxOrder + 1 rows of its own degree + 1
    coefficients, and the degrees differ per interval - which is why the block
    writer emits kFlatDegs and kFlatOffsets beside the coefficients and states
    the read rule in a line. A stride the coefficients do not have reads a
    correct table wrongly, and no check of the coefficients alone would catch
    it.

    Two things separate it from the derived partitions, and the option probe
    measures both. The grid is equal-width and derived, so the interval an
    argument falls in is one multiply and a truncation rather than a scan of
    piece edges. And no order is built from another, so a ladder is a set of
    independent polynomials where the derived routes recur upward from a seed,
    a serial dependency chain over the orders that no amount of
    instruction-level parallelism can shorten. What that costs is storage and a
    floor on the work per order: the derived partitions spend coefficients
    where the function needs them and charge one multiply for a high order once
    the seed is paid, while here every order pays its own interval's degree.
    The grid is one width for all orders rather than a per-order walk, and each
    interval's degree is the smallest the proved truncation bound carries under
    the bound the region is held to.
    """
    grid = flat["grid"]
    degs = grid["degs"]
    cap = max(degs)
    intervals = grid["intervals"]
    offsets = flat_grid_offsets(grid)
    stored = offsets[-1]
    per_order = flat["orders"]
    if [iv[2] for iv in per_order[0]] != list(degs):
        raise RuntimeError("the uniform table's fits are not on the grid the block "
                           "writer is emitting: the degrees disagree")
    lines = [
        f"// The uniform table: one derived grid over [0, kFlatHi), width kFlatWidth = {mp.nstr(grid['width'], 8)},",
        f"// every order fitted at its interval's own degree (kFlatDegs), cap kFlatReadCap = {cap}.",
        "// Read at kFlatCoeffs[kFlatOffsets[iv] + order * (kFlatDegs[iv] + 1) + k] and the",
        "// same for kFlatMonoCoeffs, for k <= kFlatDegs[iv].",
        f"inline constexpr int kFlatReadCap = {cap};",
        deg_assert("kFlatReadCap"),
        f"inline constexpr int kFlatIntervals = {intervals};",
        f"inline constexpr double kFlatWidth = {fmt(grid['width'])};",
        f"inline constexpr double kFlatHi = {fmt(grid['hi'])};",
        "// One degree per interval: the smallest the cell law carries there, at",
        "// the read cap or below it, and even in every entry.",
        "inline constexpr auto kFlatDegs = std::to_array<int>({",
    ]
    for i in range(0, len(degs), 12):
        lines.append("  " + ", ".join(str(d) for d in degs[i:i + 12]) + ",")
    lines.append("});")
    lines.append("// Where each interval's block starts, and the end of the last one:")
    lines.append("inline constexpr auto kFlatOffsets = std::to_array<int>({")
    for i in range(0, len(offsets), 12):
        lines.append("  " + ", ".join(str(o) for o in offsets[i:i + 12]) + ",")
    lines.append("});")
    for name, column in (("kFlatCoeffs", 3), ("kFlatMonoCoeffs", 4)):
        values = [fmt(c)
                  for iv in range(intervals)
                  for n in range(MAX_ORDER + 1)
                  for c in per_order[n][iv][column]]
        lines.append(f"inline constexpr auto {name} = std::to_array<double>({{")
        for i in range(0, len(values), 6):
            lines.append("  " + ", ".join(values[i:i + 6]) + ",")
        lines.append("});")
    lines.append(f"static_assert(std::size(kFlatCoeffs) == {stored} &&\n"
                 f"                  std::size(kFlatMonoCoeffs) == {stored},\n"
                 "              \"the uniform table must hold every interval of every order\");")
    lines.append(f"static_assert(std::size(kFlatDegs) == kFlatIntervals &&\n"
                 f"                  std::size(kFlatOffsets) == kFlatIntervals + 1,\n"
                 "              \"the uniform table must carry one degree and one offset per "
                 "interval\");")
    lines.append(f"static_assert(kFlatOffsets[kFlatIntervals] == {stored},\n"
                 "              \"the offsets must reach the end of the uniform table\");")
    lines.append("")
    lines.append("// The uniform table's certification rows, in the derived partitions' form:")
    lines.append("// the bound each scheme delivers in each multiply-add route, worst over the")
    lines.append("// whole table, rounded up to a power of two so it bounds a sweep. No")
    lines.append("// recurrence is read, so no seeding gain. deg is the cap; stored sums the degrees.")
    lines.append("struct FlatRow { int scheme, deg, intervals, stored;")
    lines.append("                 double fused, separate; };")
    lines.append("inline constexpr auto kFlatRows = std::to_array<FlatRow>({")
    for scheme in range(NARROW_SCHEMES):
        bounds = flat["bounds"][scheme]
        lines.append(f"  {{{scheme}, {cap}, {intervals}, {stored}, "
                     f"{fmt(bounds[0])}, {fmt(bounds[1])}}},")
    lines.append("});")
    return lines


# ---------------------------------------------------------------------------
# The rational minimax route over the uniform grid
# ---------------------------------------------------------------------------
# A cell of this grid is an interval, and this file's other rational routes are
# fitted over exactly such intervals - a piece of a derived partition is an
# interval too - so the family has a member over this partition, and what the
# grid's shape costs it is the criterion its pairs are chosen by rather than
# whether it has any.
#
# A derived partition cuts a piece where the proved truncation bound says a piece
# is needed, and the pair over that piece is accepted at the bar the region
# publishes. The grid's intervals are fixed by the width law instead, so there is
# no truncation law to consult for how much each one needs - but each one is
# still an interval, and its pair is chosen the way this file's other rational
# routes choose theirs: the smallest stored pair whose DELIVERED error holds -
# the pair's value in the kernel's own arithmetic, at both multiply-add routes,
# against the high-precision reference, because a fit's own residual is not what
# a caller receives.
#
# The bar is the grid's own and not a second one. A cell of this table is read
# directly - nothing carries its error anywhere - so what a caller receives from
# it is the cell's own truncation plus the lane's rounding, and the cell law
# places the Chebyshev member on the cell against the bound less half an ulp
# (FLAT_CELL_BOUND, FLAT_HALF_ULP). The member here answers to that same bound,
# and its acceptance is half of it, so a pair accepted on one grid and certified
# on another cannot meet the bound from opposite sides.
#
# A pair is stored per interval rather than per cell, and that is the layout's
# requirement rather than a saving: the block is interval-major with the order as
# the inner index (see flat_block_lines), so an interval's rows share one stride.
# Every order of the interval is therefore fitted and measured at the interval's
# own pair, and the pair is the first ladder entry that holds for ALL of that
# interval's orders - not order 0's answer, the worst of them, published for the
# rest.
#
# A cell no ladder entry holds is a raise rather than an emission: a pair that
# does not hold the bar cannot be answered from another table, and a member
# emitted with one would carry a published figure above the bound its own grid
# was derived under.
#
# The exchange is the one the region-B and narrow routes are fitted by (Remez on
# the linearized equioscillation problem, [Remez1934]), run here at the degree
# pairs the ladder names. Its second instrument, Lawson's algorithm with the
# Sanathanan-Koerner denominator weight ([Lawson1964], [SanathananKoerner1963]),
# is the cross-check the narrow route prints beside its fits: it reaches the same
# linear problem from a different direction, and on these cells - where the
# function is analytic and the interval is short, so many degree pairs are far
# more than the cell needs - the two disagree about the pair's SCALING rather
# than about its error, which is why the acceptance is read off the delivered
# value instead of a residual and why a second starting grid is tried before a
# pair is called a failure.

# The bar a cell's pair is accepted under: half of the bound the grid's cells are
# held to. The cell law leaves the bound's other half to the lane's rounding, and
# the acceptance is deliberately below the figure the rows publish, so a pair
# accepted here and certified on the other grid view cannot measure above it.
FLAT_RAT_ACCEPT = FLAT_CELL_BOUND["double"] / 2
# The cell's fitting grid and the certification grid, as two views of the same
# interval: the fits are made at the first and the delivered figure is measured
# at the second, offset by half a step so no point of one is a point of the
# other. A figure taken on the fitting grid alone would be the residual of the
# fit the exchange just made rather than what a caller receives.
FLAT_RAT_GRID = 200
FLAT_RAT_ITERS = 120
# How many stored doubles above the first ladder entry that holds the walk
# considers before it settles: see flat_rat_pick, which is where the figure that
# decides this is measured.
FLAT_RAT_WINDOW = 2
# The certification's working precision: the fit path's own (mp.dps above), which
# is eight orders below the figure being measured and is the precision the other
# tables' delivered figures are taken at. The fits themselves run at RAT_DPS.
FLAT_RAT_CERT_DPS = 30

# The pairs the search visits, in the order it visits them: by the count they
# store, and within one count the smaller denominator first, because the exchange
# returns a better-scaled pair when the denominator has fewer terms to trade
# against the numerator. No entry has a zero denominator: a pair whose
# denominator held only its constant term would be the Chebyshev member's
# polynomial read under the rational route's name, which is the substitution the
# refusal this member removes exists to prevent.
FLAT_RAT_LADDER = ((2, 2), (3, 2), (4, 2), (3, 3), (5, 2), (4, 3), (6, 2), (5, 3),
                   (4, 4), (7, 2), (6, 3), (5, 4), (8, 2), (7, 3), (6, 4), (5, 5),
                   (9, 2), (8, 3), (7, 4), (6, 5), (10, 2), (9, 3), (8, 4), (7, 5),
                   (11, 2), (10, 3), (9, 4), (8, 5), (12, 2), (11, 3), (10, 4),
                   (9, 5), (12, 3), (11, 4), (10, 5), (12, 4))


def flat_rat_arguments(n, iv, grid, half, precise):
    """One cell's x, its mapped argument and its reference.

    The mapped argument is 2 (x * kFlatPerUnit - iv) - 1, spelled exactly as
    FlatLocate spells it, and for a dyadic width that expression is exact: the
    width's reciprocal is a power of two, so the product is the argument with its
    exponent moved and no rounding at all, and the subtraction is exact over the
    interval's own range. A fit made and measured at an exactly-mapped t is a fit
    made and measured at the mapping the kernel performs rather than at a
    rounding of it.
    """
    width = float(grid["width"])
    a = float(iv * grid["width"])
    b = a + width
    step = (b - a) / FLAT_RAT_GRID
    xs = [a + step * (i + (0.5 if half else 0.0)) for i in range(FLAT_RAT_GRID)]
    xs.append(math.nextafter(b, 0.0))
    if precise:
        with mp.workdps(RAT_DPS):
            fs = [boys_ref(n, x, RAT_TAIL_FLOOR, RAT_TERMS) for x in xs]
    else:
        with mp.workdps(FLAT_RAT_CERT_DPS):
            fs = [boys_ref(n, x) for x in xs]
    per = 1.0 / width
    ts = [mpf(2.0 * (x * per - float(iv)) - 1.0) for x in xs]
    return ts, xs, fs


def flat_rat_delivered(p, q, ts, fs):
    """The pair's delivered worst |F_n - fit|, in the kernel's arithmetic.

    Both multiply-add routes, because the kernel's step is std::fma in a default
    build and a bare a * b + c in a BOYS_MULADD_SEPARATE one, and a pair accepted
    on one reading alone would be a pair whose bound was never measured against
    the arithmetic the other build runs (see rat_eval_double). Returns
    [fused, separate].
    """
    pd = [float(c) for c in p]
    qd = [float(c) for c in q]
    worst = [0.0, 0.0]
    for t, f in zip(ts, fs):
        td = float(t)
        for route, fused in ((0, True), (1, False)):
            e = abs(mpf(rat_eval_double(pd, qd, td, fused)) - f)
            if e > worst[route]:
                worst[route] = e
    return worst


def flat_rat_worst(p, q, ts, fs):
    """The delivered error per route and the argument each was reached at."""
    pd = [float(c) for c in p]
    qd = [float(c) for c in q]
    worst = [0.0, 0.0]
    at = [None, None]
    for t, f in zip(ts, fs):
        td = float(t)
        for route, fused in ((0, True), (1, False)):
            e = abs(mpf(rat_eval_double(pd, qd, td, fused)) - f)
            if e > worst[route]:
                worst[route], at[route] = e, td
    return worst, at


def flat_rat_exchange(m, k, ts, fs):
    """The best pair the exchange returns for one cell at one ladder entry.

    Two starting grids, because the exchange's first reference set is built from
    the grid it is handed and a set that collapses - or a denominator that goes
    near zero on the first solve, which rat_remez reports as no fit at all -
    would otherwise be read as the entry having no solution rather than as the
    starting grid having been the wrong one. Of the pairs returned, the one whose
    DELIVERED error is smaller is kept: on a cell this easy the exchange can
    return two pairs of equal residual and very different scaling, and the
    delivered value is the only one of the two a caller sees.
    """
    best = None
    for stride in (1, 2):
        r = rat_remez(m, k, list(range(0, len(ts), stride)), ts, fs, iters=FLAT_RAT_ITERS)
        if r is None:
            continue
        _, p, q = r
        worst = flat_rat_delivered(p, q, ts, fs)
        if best is None or max(worst) < max(best[2]):
            best = (p, q, worst)
    return best


def flat_rat_pick(n, iv, grid, start):
    """The ladder entry that delivers least among those near the first that holds.

    The walk does not stop at the first entry that holds, and that is a measured
    decision rather than a taste: at the cells that set the member's figure the
    fit is at the lane's rounding floor, where the entries differ only by how
    their own coefficients round. On the grid's first interval, order 0, the
    entries that hold deliver between 1.62e-16 and 2.23e-16 - a spread of more
    than a third, set by the coefficients' own last bits and not by any
    approximation error - and the published figure is a power-of-two round-up, so
    which entry is taken decides whether the member publishes the figure the
    Chebyshev member beside it publishes or one power of two above it.

    The window is two stored doubles above the first entry that holds: the member
    buys rounding luck with the smallest amount of storage that buys any, and the
    bound on the search follows from the ladder rather than from an iteration
    count. An entry below the first that holds is never taken - it fails the
    acceptance - and every entry above the window is ignored, so the walk is
    `start`..(the first that holds + two stored doubles).
    """
    entered = None
    best = None
    for index in range(start, len(FLAT_RAT_LADDER)):
        m, k = FLAT_RAT_LADDER[index]
        if entered is not None and (m + 1 + k) > entered + FLAT_RAT_WINDOW:
            break
        ts, _xs, fs = flat_rat_arguments(n, iv, grid, half=False, precise=True)
        r = flat_rat_exchange(m, k, ts, fs)
        if r is None or max(r[2]) > FLAT_RAT_ACCEPT:
            continue
        if entered is None:
            entered = m + 1 + k
        if best is None or max(r[2]) < max(best[1][2]):
            best = (index, r)
    return best


def _flat_rat_interval_job(spec):
    """One interval's pairs, in a worker process."""
    iv, grid = spec
    mp.dps = RAT_DPS  # the fit's working precision; a spawn does not carry it
    return flat_rat_interval(iv, grid)


def flat_rat_interval(iv, grid):
    """One interval of the grid: its pair, its orders' pairs, and its figure.

    The interval's pair is chosen on its own first order - the largest of the
    interval's orders in absolute value, since |F_n| <= F_0 pointwise - and then
    held to every order of the interval, because one pair serves all of them
    (see the section comment). An order that fails the pair the interval was
    chosen at moves the whole interval one ladder entry up and the walk is made
    again: a pair is not chosen per order because the layout has no room for one,
    and it is not reported as the interval's while an order of that interval
    fails it, which is the silent substitution this member's own refusal would
    otherwise have been replaced by.
    """
    index = 0
    while True:
        if index >= len(FLAT_RAT_LADDER):
            raise RuntimeError(
                f"interval {iv} of the uniform grid has no rational pair on the "
                f"ladder: every entry up to {FLAT_RAT_LADDER[-1]} fails either the "
                f"exchange or the {mp.nstr(FLAT_RAT_ACCEPT, 3)} its cells are "
                f"accepted under, so no member of the route can be emitted for it")
        chosen = flat_rat_pick(0, iv, grid, index)
        if chosen is None:
            raise RuntimeError(
                f"interval {iv} of the uniform grid has no rational pair at or "
                f"above {FLAT_RAT_LADDER[index]} on the ladder")
        # The pick resumes above a failing entry, so the pair the orders are
        # held to is read from where the pick settled and not from where the
        # walk entered.
        index, (p0, q0, _worst0) = chosen
        m, k = FLAT_RAT_LADDER[index]
        pairs = []
        worst = [0.0, 0.0]
        at = [None, None]
        failed = None
        for n in range(MAX_ORDER + 1):
            if n == 0:
                p, q = p0, q0
            else:
                ts, _xs, fs = flat_rat_arguments(n, iv, grid, half=False, precise=True)
                r = flat_rat_exchange(m, k, ts, fs)
                if r is None or max(r[2]) > FLAT_RAT_ACCEPT:
                    failed = n
                    break
                p, q = r[0], r[1]
            ts, _xs, fs = flat_rat_arguments(n, iv, grid, half=True, precise=False)
            w, w_at = flat_rat_worst(p, q, ts, fs)
            if max(w) > FLAT_CELL_BOUND["double"]:
                failed = n
                break
            for route in (0, 1):
                if w[route] > worst[route]:
                    worst[route], at[route] = w[route], (n, w_at[route])
            pairs.append((n, p, q))
        if failed is None:
            return {"interval": iv, "m": m, "k": k, "pairs": pairs, "worst": worst,
                    "at": at}
        index += 1


def flat_rat_table(lane="double"):
    """The rational member over the grid's cells, with its measured figure.

    Every interval is walked and measured independently, so the pool order cannot
    reach the result, and the grid the fits were placed on travels back with them
    so the block writer emits the pairs that were fitted rather than a second
    derivation of them.
    """
    grid = uniform_grid(lane)
    jobs = [(iv, grid) for iv in range(grid["intervals"])]
    with multiprocessing.get_context("spawn").Pool(min(len(jobs), NARROW_WORKERS)) as pool:
        out = pool.map(_flat_rat_interval_job, jobs)
    worst = [0.0, 0.0]
    for row in out:
        for route in (0, 1):
            if row["worst"][route] > worst[route]:
                worst[route] = row["worst"][route]
    for route in (0, 1):
        if worst[route] > FLAT_CELL_BOUND["double"]:
            raise RuntimeError(
                f"the {lane} lane's rational member over the uniform grid delivers "
                f"{worst[route]:.6e} in "
                f"{'fused' if route == 0 else 'separate'} multiply-adds, above the "
                f"{mp.nstr(FLAT_CELL_BOUND['double'], 3)} its grid is held to")
    return {"grid": grid, "intervals": out, "worst": worst,
            "stored": sum((MAX_ORDER + 1) * (r["m"] + 1 + r["k"]) for r in out)}


def flat_rat_block_lines(frat):
    """The rational member over the uniform grid, as the header stores it.

    The same interval-major layout as the Chebyshev member beside it, with the
    order as the inner index, so an argument's whole ladder is one contiguous
    block here as it is there. An interval's rows are its own pair's stored
    count, which differs from interval to interval, so the block carries its own
    offsets and its own degree columns and states its read rule in a line - a
    stride the coefficients do not have reads a correct table wrongly, and no
    check of the coefficients alone would catch it.

    The pair is chosen as this file's other rational routes choose theirs: the
    ladder is walked from the smallest stored pair up until one holds half the
    bound the grid's cells are held to, and the entry kept is the one among the
    first few that hold whose delivered error is smallest. At the cells that set
    the member's figure the entries differ only by how their own coefficients
    round, and the published row is a power-of-two round-up, so which entry is
    taken decides which figure is published. Every figure read is the delivered
    error in the kernel's own arithmetic, at both multiply-add routes with the
    worse taken.
    """
    grid = frat["grid"]
    intervals = grid["intervals"]
    rows = sorted(frat["intervals"], key=lambda r: r["interval"])
    if [r["interval"] for r in rows] != list(range(intervals)):
        raise RuntimeError("the rational member's intervals are not the grid's")
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + (MAX_ORDER + 1) * (row["m"] + 1 + row["k"]))
    stored = offsets[-1]
    if stored != frat["stored"]:
        raise RuntimeError("the rational member's stored count is not the sum of its "
                          "intervals' pairs")
    lines = [
        "// The rational minimax family over the uniform grid: one numerator/denominator pair",
        "// per interval at the grid's own cells (the cells the width law derived, not a",
        "// cut this family makes). Row = p_0..p_m then q_1..q_k, q_0 = 1; chosen by the",
        "// ladder, read at kFlatRatCoeffs[kFlatRatOffsets[iv] + order * kFlatRatStored[iv] + j].",
        f"inline constexpr int kFlatRatIntervals = {intervals};",
        f"inline constexpr int kFlatRatStoredTotal = {stored};",
        "inline constexpr auto kFlatRatNumDeg = std::to_array<int>({",
    ]
    for i in range(0, len(rows), 12):
        lines.append("  " + ", ".join(str(r["m"]) for r in rows[i:i + 12]) + ",")
    lines.append("});")
    lines.append("inline constexpr auto kFlatRatDenDeg = std::to_array<int>({")
    for i in range(0, len(rows), 12):
        lines.append("  " + ", ".join(str(r["k"]) for r in rows[i:i + 12]) + ",")
    lines.append("});")
    lines.append("// The stored count of each interval's row, which is its pair plus the")
    lines.append("// denominator's held constant term:")
    lines.append("inline constexpr auto kFlatRatStored = std::to_array<int>({")
    for i in range(0, len(rows), 12):
        lines.append("  " + ", ".join(str(r["m"] + 1 + r["k"]) for r in rows[i:i + 12])
                     + ",")
    lines.append("});")
    lines.append("// Where each interval's block starts, and the end of the last one:")
    lines.append("inline constexpr auto kFlatRatOffsets = std::to_array<int>({")
    for i in range(0, len(offsets), 12):
        lines.append("  " + ", ".join(str(o) for o in offsets[i:i + 12]) + ",")
    lines.append("});")
    values = []
    for row in rows:
        if [p[0] for p in row["pairs"]] != list(range(MAX_ORDER + 1)):
            raise RuntimeError("the rational member's rows do not cover every order")
        for (_n, p, q) in row["pairs"]:
            values.extend(fmt(float(c)) for c in p)
            values.extend(fmt(float(c)) for c in q)
    lines.append("inline constexpr auto kFlatRatCoeffs = std::to_array<double>({")
    for i in range(0, len(values), 6):
        lines.append("  " + ", ".join(values[i:i + 6]) + ",")
    lines.append("});")
    lines.append(f"static_assert(std::size(kFlatRatCoeffs) == {stored},\n"
                 "              \"the rational member must hold every order of every "
                 "interval\");")
    lines.append("static_assert(std::size(kFlatRatNumDeg) == kFlatRatIntervals\n"
                 "                  && std::size(kFlatRatDenDeg) == kFlatRatIntervals\n"
                 "                  && std::size(kFlatRatStored) == kFlatRatIntervals\n"
                 "                  && std::size(kFlatRatOffsets) == kFlatRatIntervals + 1,\n"
                 "              \"the rational member must carry one pair and one offset "
                 "per interval\");")
    lines.append("static_assert(kFlatRatOffsets[kFlatRatIntervals] == "
                 "kFlatRatStoredTotal,\n"
                 "              \"the offsets must reach the end of the rational "
                 "member\");")
    lines.append("static_assert(kFlatRatIntervals == kFlatIntervals,\n"
                 "              \"the rational member is read at the Chebyshev member's "
                 "own grid\");")
    lines.append("")
    lines.append("// What the rational member delivers, at each multiply-add route, against")
    lines.append("// the bound the grid's cells are held to. Measured on each cell's own")
    lines.append("// certification grid - offset half a step from the fit grid - on the stored")
    lines.append("// coefficients, which is what the acceptance above read.")
    lines.append("struct FlatRatRow { int stored; double fused, separate; };")
    lines.append("inline constexpr auto kFlatRatRows = std::to_array<FlatRatRow>({")
    lines.append(f"  {{{stored}, {fmt(frat['worst'][0])}, {fmt(frat['worst'][1])}}},")
    lines.append("});")
    lines.append(f"inline constexpr double kFlatRatBar = "
                 f"{fmt(float(FLAT_CELL_BOUND['double']))};")
    return lines


def narrow_block_lines(narrow_a, narrow_b, narrow_rat_a, narrow_rat_b):
    """The narrow partitions' tables as the header writes them, both regions.

    One function rather than one per region because the block is what is
    reproduced on its own - --narrow-only writes these lines and the block's
    byte-identity is checked without fitting the shipped table - and the order
    the two regions are written in has to be one fact rather than two.
    write_header writes the same lines, so the two cannot drift.
    """
    return (narrow_a_block_lines(narrow_a) + [""] + narrow_b_block_lines(narrow_b)
            + [""] + narrow_rational_block_lines(narrow_rat_a, narrow_rat_b))


def narrow_rational_block_lines(narrow_rat_a, narrow_rat_b):
    """The rational route's fits over the narrow partition, both regions.

    The narrow partition's intervals are the Chebyshev route's - a partition is
    a cut of the region, not a property of a family - so these tables carry the
    pairs only and are read at the narrow pieces' own intervals and mapped
    arguments. Region A's pieces are one per `kNarrowAPieces` row, in that
    order; region B's are one per narrow region-B piece, in that order.
    """
    lines = [
        "// The rational minimax family over the narrow partition: one degree pair per",
        "// narrow piece, read at kNarrowAPieces' intervals and mapped arguments (a",
        "// partition is a cut of the region, not a family's). Row = p_0..p_m then",
        "// q_1..q_k, q_0 = 1; accepted at the shipped 3e-14 bar, worse of both routes.",
    ]
    per_order = narrow_rat_a["orders"]
    a_coeffs = []
    a_offset = []
    a_numdeg = []
    a_dendeg = []
    for fits in per_order:
        for fit in fits:
            a_offset.append(len(a_coeffs))
            a_numdeg.append(fit["m"])
            a_dendeg.append(fit["k"])
            a_coeffs.extend(fmt(float(c)) for c in fit["p"])
            a_coeffs.extend(fmt(float(c)) for c in fit["q"])
    lines.append("inline constexpr auto kNarrowRatACoeffs = std::to_array<double>({")
    for i in range(0, len(a_coeffs), 6):
        lines.append("  " + ", ".join(a_coeffs[i:i + 6]) + ",")
    lines.append("});")
    for name, column in (("kNarrowRatAOffset", a_offset), ("kNarrowRatANumDeg", a_numdeg),
                         ("kNarrowRatADenDeg", a_dendeg)):
        lines.append(f"inline constexpr auto {name} = std::to_array<int>({{")
        for i in range(0, len(column), 12):
            lines.append("  " + ", ".join(str(v) for v in column[i:i + 12]) + ",")
        lines.append("});")
    lines.append("static_assert(std::size(kNarrowRatAOffset) == std::size(kNarrowAPieces)\n"
                 "                  && std::size(kNarrowRatANumDeg) == std::size(kNarrowAPieces)\n"
                 "                  && std::size(kNarrowRatADenDeg) == std::size(kNarrowAPieces),\n"
                 "              \"the narrow rational region-A route must cover every piece\");")

    b_coeffs = []
    b_offset = []
    b_numdeg = []
    b_dendeg = []
    for fit in narrow_rat_b["pieces"]:
        b_offset.append(len(b_coeffs))
        b_numdeg.append(fit["m"])
        b_dendeg.append(fit["k"])
        b_coeffs.extend(fmt(float(c)) for c in fit["p"])
        b_coeffs.extend(fmt(float(c)) for c in fit["q"])
    lines.append("")
    lines.append("// The same family over the narrow partition of region B: one pair")
    lines.append("// per narrow region-B piece, read at that piece's own interval and")
    lines.append("// at the mapped argument the Chebyshev seed on that piece uses, so")
    lines.append("// the two routes over the piece read one t.")
    lines.append("inline constexpr auto kNarrowRatBCoeffs = std::to_array<double>({")
    for i in range(0, len(b_coeffs), 6):
        lines.append("  " + ", ".join(b_coeffs[i:i + 6]) + ",")
    lines.append("});")
    for name, column in (("kNarrowRatBOffset", b_offset), ("kNarrowRatBNumDeg", b_numdeg),
                         ("kNarrowRatBDenDeg", b_dendeg)):
        lines.append(f"inline constexpr auto {name} = std::to_array<int>({{")
        for i in range(0, len(column), 12):
            lines.append("  " + ", ".join(str(v) for v in column[i:i + 12]) + ",")
        lines.append("});")
    lines.append("static_assert(std::size(kNarrowRatBOffset) == kNarrowBPieces\n"
                 "                  && std::size(kNarrowRatBNumDeg) == kNarrowBPieces\n"
                 "                  && std::size(kNarrowRatBDenDeg) == kNarrowBPieces,\n"
                 "              \"the narrow rational region-B route must cover every piece\");")
    lines.append("")
    lines.append("// What the narrow rational route stores and what it delivers, at each")
    lines.append("// multiply-add route, against the bar the shipped rational rows publish.")
    lines.append("// Measured on the narrow pieces' own grids in the kernel's arithmetic on")
    lines.append("// the stored coefficients, which is what the acceptance above read.")
    lines.append(f"inline constexpr double kNarrowRatABound = {fmt(float(narrow_rat_a['bound']))};")
    lines.append(f"inline constexpr int kNarrowRatAStored = {narrow_rat_a['stored']};")
    lines.append("inline constexpr double kNarrowRatADeliveredFused = "
                 f"{fmt(float(narrow_rat_a['worst'][0][0]))};")
    lines.append("inline constexpr double kNarrowRatADeliveredSeparate = "
                 f"{fmt(float(narrow_rat_a['worst'][0][1]))};")
    lines.append(f"inline constexpr double kNarrowRatBBound = {fmt(float(narrow_rat_b['bar']))};")
    lines.append(f"inline constexpr int kNarrowRatBStored = {narrow_rat_b['stored']};")
    lines.append("inline constexpr double kNarrowRatBDeliveredFused = "
                 f"{fmt(float(narrow_rat_b['worst'][0][0]))};")
    lines.append("inline constexpr double kNarrowRatBDeliveredSeparate = "
                 f"{fmt(float(narrow_rat_b['worst'][0][1]))};")
    return lines


def narrow_b_block_lines(narrow):
    """Region B's narrow tables: the derived partition as the header stores it.

    A separate function because the block is reproduced on its own: the full
    generation is hours and this block is seconds, so --narrow-only writes
    these lines and the block's byte-identity is checked without the rest of
    the table. write_header writes the same lines, so the two cannot drift.

    The extended band is the same fit at either granularity, because nothing
    amplifies its seed - the upward recursion it feeds runs the other way, so
    an error in it stays the size it is. Region A's pieces are read the other
    way round and are partitioned separately, by the gain the batch entry's
    downward recursion applies to them rather than by this bound alone. Naming
    this partition leaves the shipped seed's coefficients untouched.
    """
    pieces = narrow["pieces"]
    deg = pieces[0][2]
    lines = [
        f"// The narrow partition of region B: the same interval [kX0, kX1) cut into {len(pieces)}",
        f"// pieces at degree {deg}, each as wide as the proved a-priori truncation bound lets",
        "// it be at the 1e-14 target (see --derive-partition). One degree throughout, so",
        "// no offset table: piece i starts at i * (kNarrowBDeg + 1). The seed is unchanged.",
        f"inline constexpr int kNarrowBDeg = {deg};",
        deg_assert("kNarrowBDeg"),
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
    lines.append("// The narrow partition's certification rows, the same shape and measured")
    lines.append("// the same way as the scheme rows below: the bound each scheme delivers on")
    lines.append("// it in each multiply-add route, worst over its pieces, rounded up to a")
    lines.append("// power of two. Counted apart from the shipped rows: a second partition.")
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
    the seed-bound derivation - the band is certified under
    the same quantity plus the truncation tail tau, delta_0' = R_hat + tau
    = 1.0641e-15)."""
    err = mpf(abs(clenshaw_double(cd, float(x), float(edge), float(X0)) - float(boys_ref(0, x))))
    return err + mpf("1.0116e-15")


def extended_gen_envelope(n, edge, cd, x, gamma):
    """The generator-side closed-form envelope B_n(x) of the extended band
    (the closed-form envelope with the per-argument seed width
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
# Two textbook routes are run. The Remez exchange ([Remez1934]) solves the
# linearized
# equioscillation problem on a reference set and moves the set to the error's
# extrema; Lawson's algorithm ([Lawson1964]) reaches the same linear problem
# through
# iteratively reweighted least squares with the Sanathanan-Koerner denominator
# weight ([SanathananKoerner1963]). They share no machinery, so agreement
# between them is evidence the
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


def rat_horner_double(cs, t, fused=True):
    """Horner in double at a named multiply-add route."""
    acc = cs[-1]
    for c in reversed(cs[:-1]):
        acc = route_step(fused, acc, t, c)
    return acc


def rat_eval_double(p, q, t, fused=True):
    """The kernel's evaluation of the route: Horner in double, at one route.

    The kernel's step is `backend::ScalarFp64::MulAdd`, which is `std::fma` in
    a default build and a bare `a * b + c` in a `BOYS_MULADD_SEPARATE` one, and
    the two round differently. A pair accepted on one reading alone would be a
    pair whose bound was never measured against the arithmetic the other build
    runs, so every acceptance here reads both and takes the worse.
    """
    num = rat_horner_double(p, t, fused)
    den = route_step(fused, rat_horner_double(q, t, fused), t, 1.0) if q else 1.0
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


def region_b_seed_rational(p, q, x, fused=True):
    t = 2.0 * (float(x) - float(X0)) / (float(X1) - float(X0)) - 1.0
    return rat_eval_double(p, q, t, fused)


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
    # A row's powers and the value they are fitted to are the same on every
    # one of the outer iterations - only the weight the iteration carries
    # moves - so they are built once, from the same expressions, and read back
    # three hundred times instead of recomputed three hundred times.
    bases = [[t ** j for j in range(m + 1)] + [-fi * t ** j for j in range(1, k + 1)]
             for t, fi in ((ts[i], fs[i]) for i in indices)]
    last = None
    for _ in range(outer):
        a = [[mpf(0) for _ in range(nunk)] for _ in range(nunk)]
        rhs = mp.zeros(nunk, 1)
        for row, base in enumerate(bases):
            fi = fs[indices[row]]
            w2 = w[row] * w[row]
            # w2 * base[r] is the same for every c it is multiplied into, so
            # it is taken once per row: the products below are the ones the
            # loop made before, with the row's own rounding in the same place.
            g = [w2 * br for br in base]
            for r in range(nunk):
                rhs[r] += g[r] * fi
                arow = a[r]
                for c in range(r, nunk):
                    arow[c] += g[r] * base[c]
        for r in range(nunk):
            arow = a[r]
            for c in range(r):
                arow[c] = a[c][r]
        try:
            sol = mp.lu_solve(mp.matrix(a), rhs)
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


def rat_delivered_error(p, q, xs, ref, fused=True):
    """The route's delivered worst |F0 - fit|, as the kernel evaluates it."""
    worst = mpf(0)
    at = None
    for x, f in zip(xs, ref):
        e = abs(mpf(region_b_seed_rational(p, q, x, fused)) - f)
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
    """The route's delivered worst |F0 - fit| over `indices`."""
    worst = mpf(0)
    at = None
    for i in indices:
        e = abs(mpf(region_b_seed_rational(p, q, xs[i])) - ref[i])
        if e > worst:
            worst, at = e, xs[i]
    return worst, at


def rat_kernel_delivered_on(p, q, indices, xs, ref, fused):
    """The same figure in the arithmetic the kernel runs, at one route.

    `p` and `q` are the solved pairs as stored - every coefficient rounded to
    binary64 before it is read - and the argument is mapped the way
    `region_b_seed_rational` maps it. The region-B acceptance above measures
    the solved pair at the working precision instead; that reading is what its
    own table was certified on and this one does not disturb it, but a bound
    taken from it is not a bound on the evaluation the kernel performs, so the
    narrow table's acceptance reads this one at both routes and takes the worse.
    """
    pd = [float(c) for c in p]
    qd = [float(c) for c in q]
    worst = mpf(0)
    at = None
    for i in indices:
        x = xs[i]
        t = 2.0 * (float(x) - float(X0)) / (float(X1) - float(X0)) - 1.0
        e = abs(mpf(rat_eval_double(pd, qd, t, fused)) - ref[i])
        if e > worst:
            worst, at = e, x
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
        ks = list(range(1, min(RAT_K_MAX, stored - 2) + 1))
        found = run_jobs(_rat_best_job,
                         [(stored - 1 - k, k, coarse, ts, xs, ref, None) for k in ks])
        for k, (best, _) in zip(ks, found):
            m = stored - 1 - k
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
    refined = run_jobs(_rat_best_job,
                       [(mm, kk, full, ts, xs, ref, None) for _, mm, kk in candidates])
    rows = []
    # The pair that ships in the end is one of these candidates, so the figures
    # for it are already in the batch: they are kept here, keyed by the pair,
    # and read back below instead of being computed a second time.
    seen = {}
    for (total, mm, kk), (best, _both) in zip(candidates, refined):
        if best is None:
            continue
        seen[(mm, kk)] = (best, _both)
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
    # the pair rather than the route that won it. Both routes at this pair were
    # run as one of the candidates above, so the second run of the same job
    # would return the same numbers: the batch's own result is read back.
    _, both = seen[(m, k)]
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

    # Every piece of every order is its own fit over its own interval, so the
    # whole lane is one batch of jobs and the walk below reads their results
    # back in order: the same pieces, reduced the same way, from the same
    # order they were computed in when this was a nested loop.
    fitted = iter(run_jobs(
        _rat_a_piece_job,
        [(n, a, b, cs)
         for n in range(MAX_ORDER + 1) for (a, b, _deg, cs, _mono) in double_orders[n]]))

    for n in range(MAX_ORDER + 1):
        row = []
        for (a, b, deg, cs, _mono) in double_orders[n]:
            fit, cw, cat, cg, cst = next(fitted)
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


# ---------------------------------------------------------------------------
# The rational route over the narrow partition
# ---------------------------------------------------------------------------
# The route carried one partition because a rational minimax fit of a whole
# interval is not a piece of anything: the family's region-A pieces follow the
# shipped intervals and its region-B seed is one pair over [X0, X1). The narrow
# partition is a second set of intervals in each region, and the family is
# fitted over those: the narrow region-A pieces the Chebyshev route's walk
# already placed, and the narrow region-B edges it already derived.
#
# **The acceptance is the shipped criterion read in the kernel's arithmetic.**
# The shipped route's pieces were accepted on the delivered error of the stored
# doubles evaluated fused, which is the arithmetic a default build runs and is
# not the arithmetic a BOYS_MULADD_SEPARATE build runs: `ScalarFp64::MulAdd` is
# `std::fma` in one and a bare `a * b + c` in the other, and the second rounds
# twice. A bound taken at one route is not a bound on the other's evaluation,
# so every acceptance below reads both routes and takes the worse,
# and the figure the row publishes is the worse of the two.
#
# The bars are the shipped route's own: RAT_A_ACCEPT for region A's pieces and
# RAT_BAR for region B's seed, the counts the same total-stored search. What
# changes is the partition, not the criterion.
# The search is the shipped route's own: the same scan floor and cap on the
# stored count, the same certifying grid, the same stride for the subsample the
# count is first found on. A narrow piece that needs more than RAT_A_SCAN_MAX
# is reported as that count rather than accepted at a looser bar.
RAT_NARROW_A_SCAN_MIN = 6
RAT_NARROW_A_SCAN_MAX = RAT_A_SCAN_MAX
RAT_NARROW_B_SCAN_MIN = 4
RAT_NARROW_B_SCAN_MAX = 20
RAT_NARROW_GRID = RAT_A_GRID
RAT_NARROW_STRIDE = RAT_A_SCAN_STRIDE


def rat_a_kernel_delivered_on(p, q, ts, ref, indices):
    """Worst |F_n - fit| over `indices`, both routes, the worse taken.

    The pair is read as stored - every coefficient rounded to binary64 first -
    and the argument is the kernel's own mapped t. Returns the figure and the
    route that set it, because a table whose bound is set by one route and
    published for both should say which one.
    """
    pd = [float(c) for c in p]
    qd = [float(c) for c in q]
    worst = mpf(0)
    at = None
    who = None
    for fused in (True, False):
        for i in indices:
            e = abs(mpf(rat_eval_double(pd, qd, float(ts[i]), fused)) - ref[i])
            if e > worst:
                worst, at, who = e, ts[i], "fused" if fused else "separate"
    return worst, at, who


def rat_a_best_at_kernel(ts, ref, ws, indices, total):
    """The best pair at `total` stored over `indices`, both routes held."""
    best = None
    for k in range(1, min(RAT_K_MAX, total - 2) + 1):
        m = total - 1 - k
        if m < 1:
            continue
        r = rat_remez(m, k, indices, ts, ref, ws)
        if r is None:
            continue
        _, p, q = r
        d, at, who = rat_a_kernel_delivered_on(p, q, ts, ref, indices)
        if best is None or d < best[0]:
            best = (d, m, k, p, q, at, who)
    return best


def rat_a_fit_piece_kernel(n, a, b):
    """The smallest stored pair holding RAT_A_ACCEPT on [a, b) at both routes."""
    ts, xs = rat_a_nodes(a, b, RAT_NARROW_GRID)
    with mp.workdps(RAT_DPS):
        ref = [boys_ref(n, x, RAT_TAIL_FLOOR, RAT_TERMS) for x in xs]
    ws = [seed_weight(n, x) for x in xs]
    coarse = list(range(0, RAT_NARROW_GRID + 1, RAT_NARROW_STRIDE))
    full = list(range(RAT_NARROW_GRID + 1))

    # The shipped search's own two stages: the smallest count the subsample
    # accepts, then the same count re-solved on the certifying grid, resuming
    # above a count the subsample accepted and the dense grid rejects.
    total = RAT_NARROW_A_SCAN_MIN
    while total <= RAT_NARROW_A_SCAN_MAX:
        accepted = None
        for t in range(total, RAT_NARROW_A_SCAN_MAX + 1):
            r = rat_a_best_at_kernel(ts, ref, ws, coarse, t)
            if r is not None and r[0] <= RAT_A_ACCEPT:
                accepted = t
                break
        if accepted is None:
            return None
        r = rat_a_best_at_kernel(ts, ref, ws, full, accepted)
        if r is not None and r[0] <= RAT_A_ACCEPT:
            d, m, k, p, q, at, who = r
            wf, wsm = rat_a_routes(p, q, ts, ref, full)
            return {"a": float(a), "b": float(b), "m": m, "k": k, "p": p, "q": q,
                    "stored": m + 1 + k, "delivered": d, "at": at, "route": who,
                    "fused": wf, "separate": wsm}
        total = accepted + 1
    return None


def rat_a_routes(p, q, ts, ref, indices):
    """The accepted pair's delivered figure at each route, separately."""
    pd = [float(c) for c in p]
    qd = [float(c) for c in q]
    out = []
    for fused in (True, False):
        w = mpf(0)
        for i in indices:
            e = abs(mpf(rat_eval_double(pd, qd, float(ts[i]), fused)) - ref[i])
            if e > w:
                w = e
        out.append(w)
    return out[0], out[1]


def _rat_narrow_a_job(job):
    n, pieces = job
    mp.dps = RAT_DPS  # the scan's working precision; a spawn does not carry it
    out = []
    for (a, b, _deg, _cs, _ms) in pieces:
        fit = rat_a_fit_piece_kernel(n, mpf(a), mpf(b))
        if fit is None:
            raise RuntimeError(
                f"rational narrow region-A route: F{n} on [{a}, {b}) reaches no stored "
                f"count up to {RAT_NARROW_A_SCAN_MAX} holding {mp.nstr(RAT_A_ACCEPT, 2)} "
                f"in both multiply-add routes")
        out.append(fit)
    return n, out


def narrow_region_a_rational(narrow_a):
    """The rational route's own fits over the narrow partition's intervals.

    The intervals are the narrow Chebyshev partition's, because a partition is
    a cut of the region and not a property of a family: what the family brings
    to it is its own degree pair per piece. The criterion is the shipped
    rational route's - the plain delivered error of the stored pair in the
    kernel's double arithmetic, accepted at RAT_A_ACCEPT under the 3e-14 bar -
    read at both multiply-add routes with the worse taken.
    """
    print(f"fitting the rational route over the narrow partition of region A "
          f"({RAT_DPS} dps, accepted at {mp.nstr(RAT_A_ACCEPT, 2)} plain at BOTH "
          f"multiply-add routes) ...")
    jobs = [(n, narrow_a["orders"][n]) for n in range(MAX_ORDER + 1)]
    out = run_jobs(_rat_narrow_a_job, jobs)
    per_order = [[] for _ in range(MAX_ORDER + 1)]
    stored = 0
    delivered = mpf(0)
    at = None
    worst_route = set()
    fused = mpf(0)
    separate = mpf(0)
    for n, fits in out:
        for fit in fits:
            per_order[n].append(fit)
            stored += fit["stored"]
            fused = max(fused, fit["fused"])
            separate = max(separate, fit["separate"])
            if fit["delivered"] > delivered:
                delivered, at = fit["delivered"], (n, fit["a"], fit["b"])
                worst_route = {fit["route"]}
            elif fit["delivered"] == delivered:
                worst_route.add(fit["route"])
    cheb = narrow_a
    print(f"  rational {stored} stored, delivered {mp.nstr(delivered, 6)} at F{at[0]} "
          f"[{at[1]:.6f}, {at[2]:.6f}) (set at the {', '.join(sorted(worst_route))} route)")
    print(f"  rational per route: fused {mp.nstr(fused, 6)}, separate {mp.nstr(separate, 6)}")
    print(f"  Chebyshev {sum(len(p) for p in cheb['orders']) * (NARROW_DEG + 1)} stored, "
          f"delivered fused {cheb['worst'][0][0]:.6e} / separate {cheb['worst'][0][1]:.6e}")
    return {"orders": per_order, "stored": stored, "delivered": delivered,
            "at": at, "bound": RAT_A_BOUND,
            "worst": [[float(fused), float(separate)] for _ in range(NARROW_SCHEMES)],
            "route": sorted(worst_route)}


def _rat_narrow_b_job(job):
    a, b, ts, ref, coarse, full = job
    mp.dps = RAT_DPS  # the scan's working precision; a spawn does not carry it
    for total in range(RAT_NARROW_B_SCAN_MIN, RAT_NARROW_B_SCAN_MAX + 1):
        # The scan is on a stride subsample and the pair that ships is re-solved
        # on the whole grid, so a count the subsample accepts and the dense grid
        # rejects is the subsample's error rather than the fit's - the same
        # two-step the shipped region-A search takes.
        best = None
        for k in range(1, min(RAT_K_MAX, total - 2) + 1):
            m = total - 1 - k
            if m < 1:
                continue
            r = rat_remez(m, k, coarse, ts, ref, None)
            if r is None:
                continue
            _, p, q = r
            d, at, who = rat_kernel_delivered_both(p, q, ts, ref, full)
            if best is None or d < best[0]:
                best = (d, m, k, p, q, at, who)
        if best is None or best[0] > RAT_BAR:
            continue
        _, m, k, _, _, _, _ = best
        r = rat_remez(m, k, full, ts, ref, None)
        if r is None:
            continue
        _, p, q = r
        d, at, who = rat_kernel_delivered_both(p, q, ts, ref, full)
        if d > RAT_BAR:
            continue
        wf, ws = rat_a_routes(p, q, ts, ref, full)
        return {"a": a, "b": b, "m": m, "k": k, "p": p, "q": q,
                "stored": m + 1 + k, "delivered": d, "at": at, "route": who,
                "fused": wf, "separate": ws}
    return None


def rat_kernel_delivered_both(p, q, ts, ref, indices):
    """Region B's delivered figure at both routes, the worse taken.

    Region B is a seed read for itself: nothing amplifies its error, so the bar
    is RAT_BAR for the pair whatever route reads it. As in region A, the pair
    is read as stored and the argument is the kernel's own mapped t.
    """
    pd = [float(c) for c in p]
    qd = [float(c) for c in q]
    worst = mpf(0)
    at = None
    who = None
    for fused in (True, False):
        for i in indices:
            e = abs(mpf(rat_eval_double(pd, qd, float(ts[i]), fused)) - ref[i])
            if e > worst:
                worst, at, who = e, ts[i], "fused" if fused else "separate"
    return worst, at, who


def narrow_region_b_rational(narrow_b):
    """The rational route over the narrow partition of region B.

    One pair per narrow piece, over the piece's own interval and in the same
    mapped argument the Chebyshev piece there is read at, accepted at the same
    RAT_BAR the shipped seed is. The interval edges are the narrow Chebyshev
    partition's, for the reason region A's are.
    """
    print(f"fitting the rational route over the narrow partition of region B "
          f"(bar {mp.nstr(RAT_BAR, 2)}, both multiply-add routes) ...")
    pieces = []
    jobs = []
    refs = []
    for (a, b, _deg, _cs, _ms) in narrow_b["pieces"]:
        af, bf = float(a), float(b)
        npts = 96
        ts = []
        xs = []
        for i in range(npts + 1):
            x = af + (bf - af) * i / npts
            if x >= bf:
                x = math.nextafter(bf, 0.0)
            xs.append(mpf(x))
            # The kernel maps x to t in binary64 over the piece's own interval,
            # the expression NarrowRegionBSeed uses, so both routes over this
            # interval read one t.
            ts.append(mpf(2.0 * (x - af) / (bf - af) - 1.0))
        jobs.append((af, bf, ts, list(range(0, npts + 1, 2)), list(range(npts + 1))))
        with mp.workdps(RAT_DPS):
            refs.append([boys_ref(0, x, RAT_TAIL_FLOOR, RAT_TERMS) for x in xs])
    out = run_jobs(_rat_narrow_b_job,
                   [(job[0], job[1], job[2], ref, job[3], job[4])
                    for job, ref in zip(jobs, refs)])
    stored = 0
    delivered = mpf(0)
    at = None
    fused = mpf(0)
    separate = mpf(0)
    for (a, b, _deg, _cs, _ms), fit in zip(narrow_b["pieces"], out):
        if fit is None:
            raise RuntimeError(
                f"rational narrow region-B route: [{a!r}, {b!r}) reaches no stored count "
                f"up to {RAT_NARROW_B_SCAN_MAX} holding {mp.nstr(RAT_BAR, 2)} in both "
                f"multiply-add routes")
        pieces.append(fit)
        stored += fit["stored"]
        fused = max(fused, fit["fused"])
        separate = max(separate, fit["separate"])
        if fit["delivered"] > delivered:
            delivered, at = fit["delivered"], (a, b)
        print(f"  [{a!r}, {b!r}) -> [{fit['m']}/{fit['k']}] {fit['stored']} stored, "
              f"delivered {mp.nstr(fit['delivered'], 6)} at t = {mp.nstr(fit['at'], 12)} "
              f"({fit['route']})")
    print(f"  rational region B: {stored} stored over {len(pieces)} pieces, delivered "
          f"fused {mp.nstr(fused, 6)} / separate {mp.nstr(separate, 6)}, worst "
          f"{mp.nstr(delivered, 6)} at [{at[0]!r}, {at[1]!r})")
    return {"pieces": pieces, "stored": stored, "delivered": delivered, "at": at,
            "bar": RAT_BAR, "worst": [[float(fused), float(separate)]
                                      for _ in range(NARROW_SCHEMES)]}


# ---------------------------------------------------------------------------
# The float lane's narrow partition
# ---------------------------------------------------------------------------
# The float lane's own fits are placed by bisection on a sampled residual:
# fit_interval accepts an interval when a 9-point sweep of the fitted series
# against boys_ref is inside the budget, and the walk splits at the midpoint
# when it is not. That is the placement law, and the degree cap is the lever
# the narrow member of the axis pulls on it: at a lower degree the same law
# accepts a narrower interval, so the partition comes back with more pieces of
# fewer coefficients each - the trade the axis names, and not a saving.
#
# **The acceptance is the shipped criterion in the lane's own arithmetic.** The
# shipped lane's pieces were accepted on a binary64 sweep of the fitted series,
# and the entry does not evaluate in binary64: it evaluates in binary32 with a
# rounding at every step of ClenshawSplit or HornerMono, whose step is
# ScalarFp32::MulAdd - one rounding fused, two separate. Every piece below is
# therefore fitted as the shipped ones are, and then held to the same weighted
# budget TOL_FLOAT / 2 / seed_weight in binary32 at both routes, worse taken.
# A piece that held the budget in binary64 and not in binary32 is a piece the
# entry would deliver outside the bound its row publishes.
F32_NARROW_DEG = 6
F32_NARROW_GRID = 240

# Region B's seed reaches a caller only through the upward recurrence - the
# entry reads the piece's polynomial and steps it up to the order it was asked
# for - so a narrow region-B piece is placed by the figure that comes out of
# that recurrence and not by the seed's own residual. That figure is held to
# half the lane's budget here, the half being the headroom the published figure
# needs: scheme_bound rounds a sweep up to a power of two, and a row publishing
# more than the lane's own 1e-7 would be publishing outside it.
F32_NARROW_B_BUDGET = TOL_FLOAT * mpf("0.5")


def f32_route_delivered(cs, ms, n, a, b, npts):
    """[[scheme][route]] worst |F_n - fit|, binary32, coefficients as stored."""
    a32, b32 = r32(a), r32(b)
    worst = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(npts + 1):
        x = r32(a32 + (b32 - a32) * (i / npts))
        if x >= b32:
            x = math.nextafter(b32, 0.0)
        ref = float(boys_ref(n, mpf(x)))
        t = f32_map(a32, b32, x)
        v = clenshaw_route32(cs, t, True)
        w = clenshaw_route32(cs, t, False)
        h = horner_mono_route32(ms, t, True)
        j = horner_mono_route32(ms, t, False)
        for scheme, pair in ((0, (v, w)), (1, (h, j))):
            for route in (0, 1):
                e = abs(pair[route] - ref)
                if e > worst[scheme][route]:
                    worst[scheme][route] = e
    return worst


def f32_region_b_carried(cs, ms, a, b, npts, max_order=MAX_ORDER):
    """[[scheme][route]] worst |F_n - carried| over orders 0..max_order, binary32.

    The seed as the piece stores it, stepped up by the kernel's own region-B
    recurrence - MulSub rounds the product and then the difference, and the
    divide rounds - to every order the entry can be asked for, against the same
    reference the fits are validated against. The seed's own residual is one
    term of this figure and the recurrence is the rest of it; a piece placed on
    the residual alone is a piece placed on a number no caller receives.
    """
    a32, b32 = r32(a), r32(b)
    worst = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(npts + 1):
        x = r32(a32 + (b32 - a32) * (i / npts))
        if x >= b32:
            x = math.nextafter(b32, 0.0)
        t = f32_map(a32, b32, x)
        expx = r32(0.5 * math.exp(-float(x)))
        seeds = ((0, clenshaw_route32(cs, t, True), clenshaw_route32(cs, t, False)),
                 (1, horner_mono_route32(ms, t, True), horner_mono_route32(ms, t, False)))
        for scheme, fused_seed, separate_seed in seeds:
            for route, seed in ((0, fused_seed), (1, separate_seed)):
                f = seed
                for n in range(max_order + 1):
                    e = abs(f - float(boys_ref(n, mpf(x))))
                    if e > worst[scheme][route]:
                        worst[scheme][route] = e
                    f = r32((r32(r32(float(n) + 0.5) * f) - expx) / x)
    return worst


def _f32_narrow_a_job(order):
    mp.dps = 30  # the fit path's own reference precision; a spawn does not carry it
    pieces = fit_order(order, f32=True, maxdeg=F32_NARROW_DEG)
    worst = [[0.0, 0.0], [0.0, 0.0]]
    budget = 0.0
    at = None
    for (a, b, _deg, cs, ms) in pieces:
        w = f32_route_delivered(cs, ms, order, a, b, F32_NARROW_GRID)
        for scheme in (0, 1):
            for route in (0, 1):
                if w[scheme][route] > worst[scheme][route]:
                    worst[scheme][route] = w[scheme][route]
        # The shipped criterion, in the lane's arithmetic: the piece is held to
        # TOL_FLOAT / 2 / seed_weight at its own right end, which is where the
        # batch entry's downward recursion amplifies it most.
        lim = float(TOL_FLOAT * mpf("0.5") / seed_weight(order, mpf(b)))
        if max(w[0]) > lim and (max(w[0]) > budget):
            budget, at = max(w[0]), (a, b)
    return order, pieces, worst, budget, at


def narrow_region_a_f32():
    """The float lane's narrow region-A partition, as its shipped walk lays it.

    The same fit_interval acceptance and the same midpoint bisection, at
    F32_NARROW_DEG instead of the lane's degree cap, so the partition is the
    lane's own law applied at a narrower degree rather than a second design.
    The delivered figure is then measured in the lane's arithmetic, which is
    what the placement law does not do.
    """
    print(f"walking the float lane's narrow region-A partition (the lane's own "
          f"weighted 1e-7 law, degree {F32_NARROW_DEG}) ...")
    out = run_jobs(_f32_narrow_a_job, list(range(MAX_ORDER + 1)))
    per_order = [[] for _ in range(MAX_ORDER + 1)]
    worst = [[0.0, 0.0], [0.0, 0.0]]
    over = 0
    total = 0
    for order, pieces, w, budget, at in out:
        per_order[order] = pieces
        total += len(pieces)
        for scheme in (0, 1):
            for route in (0, 1):
                if w[scheme][route] > worst[scheme][route]:
                    worst[scheme][route] = w[scheme][route]
        if budget > 0.0:
            over += 1
            print(f"  F{order:>2}: a piece at [{at[0]:.6f}, {at[1]:.6f}) delivers "
                  f"{budget:.6e} in binary32, over its weighted budget")
    stored = sum(len(p) * (F32_NARROW_DEG + 1) for p in per_order)
    print(f"  {total} pieces over {MAX_ORDER + 1} orders, {stored} stored, "
          f"{over} order(s) with a piece over its own budget")
    for scheme, name in enumerate(SCHEME_NAMES):
        print(f"  {name:14s} worst {worst[scheme][0]:.6e} / {worst[scheme][1]:.6e} "
              f"(fused / separate)")
    bounds = [[scheme_bound(worst[s][r]) for r in (0, 1)]
              for s in range(NARROW_SCHEMES)]
    return {"orders": per_order, "stored": stored, "worst": worst, "bounds": bounds,
            "over": over}


def narrow_region_b_f32():
    """The float lane's narrow region-B partition.

    Region B's shipped seed is one fit over [kX0, kX1) at the lane's degree cap,
    because fit_interval's bisection is the region-A walk's and region B has
    none. The narrow member needs one, so the partition is derived by the same
    law: bisect the interval until the lane's own fit_interval accepts a piece
    at F32_NARROW_DEG. What places a piece here is the fit's own residual and,
    beside it, the figure the entry delivers: the seed stepped up the
    recurrence, in the lane's arithmetic, both schemes and both multiply-add
    routes, held to F32_NARROW_B_BUDGET. A piece whose residual passes and
    whose carried figure does not is a piece the entry delivers outside its
    budget, which is a split and not a bound to publish.
    """
    print(f"splitting the float lane's narrow region-B partition (the lane's own "
          f"1e-7 law, degree {F32_NARROW_DEG}) ...")
    pieces = []
    piece_worsts = []
    at_resolution = []
    stack = [((X0), (X1), 0)]
    while stack:
        a, b, depth = stack.pop()
        r = fit_interval(0, a, b, TOL_FLOAT, F32_NARROW_DEG, weighted=False)
        w = None
        if r is not None:
            w = f32_region_b_carried(r[1], r[2], a, b, F32_NARROW_GRID)
        over = r is None or max(max(pair) for pair in w) > F32_NARROW_B_BUDGET
        # The entry reads a piece by mapping x into it in binary32, so a piece
        # the mapping cannot resolve is not a piece: below one float step every
        # x in it maps to the same argument and the carried figure measured
        # there describes no fit. The walk therefore splits on the float grid
        # and stops when no float lies strictly inside the piece.
        a32, b32 = r32(a), r32(b)
        mid = r32((a32 + b32) / 2.0)
        if over and mid > a32 and mid < b32:
            if depth > 40:
                raise RuntimeError(f"float narrow region B: fit never converged on [{a},{b}]")
            stack.append(((mid), b, depth + 1))
            stack.append((a, (mid), depth + 1))
        else:
            if r is None:
                raise RuntimeError(f"float narrow region B: no fit at all on [{a},{b}]")
            deg, cd, mono = r
            pieces.append((float(a), float(b), deg, cd, mono))
            piece_worsts.append(w)
            if over:
                at_resolution.append((float(a), float(b),
                                      max(max(pair) for pair in w)))
    order = sorted(range(len(pieces)), key=lambda i: pieces[i][0])
    pieces = [pieces[i] for i in order]
    piece_worsts = [piece_worsts[i] for i in order]
    worst = [[0.0, 0.0], [0.0, 0.0]]
    for w in piece_worsts:
        for scheme in (0, 1):
            for route in (0, 1):
                if w[scheme][route] > worst[scheme][route]:
                    worst[scheme][route] = w[scheme][route]
    stored = len(pieces) * (F32_NARROW_DEG + 1)
    print(f"  {len(pieces)} pieces, {stored} stored, carried budget "
          f"{mp.nstr(F32_NARROW_B_BUDGET, 2)}:")
    for (a, b, _deg, _cs, _ms), w in zip(pieces, piece_worsts):
        print(f"    [{a!r}, {b!r})  width {b - a:.8f}  carried "
              f"{max(max(pair) for pair in w):.6e}")
    for a, b, w in at_resolution:
        print(f"    [{a!r}, {b!r}) is one float step wide and still delivers {w:.6e}, "
              f"over the carried budget: the lane's own arithmetic sets the figure there")
    for scheme, name in enumerate(SCHEME_NAMES):
        print(f"  {name:14s} worst {worst[scheme][0]:.6e} / {worst[scheme][1]:.6e} "
              f"(fused / separate)")
    bounds = [[scheme_bound(worst[s][r]) for r in (0, 1)]
              for s in range(NARROW_SCHEMES)]
    return {"pieces": pieces, "piece_worsts": piece_worsts, "stored": stored,
            "worst": worst, "bounds": bounds, "at_resolution": at_resolution}


# The float lane's uniform table
# ---------------------------------------------------------------------------
# The double lane's uniform table at the float lane's own bound, on the grid the
# float lane derives for itself: the same independence of every order and the
# same interval-major storage, and not the same width, interval count or
# degrees. What is not the same either is the arithmetic the fit is judged in,
# and that is the whole reason this is a second walk rather than a second table.
#
# The double lane's figure comes from fit_delivered, which sums the fits in
# binary64 at the coefficients the table stores. This lane stores binary32 and
# its entries evaluate in binary32, so a fit placed on a binary64 reading is a
# fit whose published bound was never taken on the arithmetic its caller's
# evaluation makes. The figure here is therefore the lane's own instrument,
# f32_route_delivered: both schemes over both multiply-add routes, the worse of
# the two kept, on the coefficients as they are stored.


def flat_order_f32(job):
    """One order's uniform-interval fits, over the grid the job carries.

    Every order is fitted on its own and read from its own coefficients, exactly
    as flat_order fits the double lane's: the fit is the interpolation of F_n on
    the interval at that interval's degree, converted to the monomial form once,
    and the figure is what the stored coefficients deliver in the lane's
    arithmetic. The grid the figure is read on is the lane's own -
    F32_NARROW_GRID, the one the lane's narrow fits are read on - so a figure
    here and a figure there are extremes over the same number of arguments per
    interval.
    """
    n, width, degs = job
    intervals = []
    worst = [[0.0, 0.0], [0.0, 0.0]]
    for iv, deg in enumerate(degs):
        a = iv * width
        b = a + width
        cm = cheb_coeffs(n, a, b, deg)
        cs = [float(c) for c in cm]
        ms = [float(c) for c in cheb_to_monomial(cm)]
        intervals.append((float(a), float(b), deg, cs, ms))
        w = f32_route_delivered(cs, ms, n, float(a), float(b), F32_NARROW_GRID)
        for scheme in range(NARROW_SCHEMES):
            for route in (0, 1):
                if w[scheme][route] > worst[scheme][route]:
                    worst[scheme][route] = w[scheme][route]
    return n, intervals, worst


def _flat_f32_job(job):
    """One order's uniform fits, in a worker process."""
    mp.dps = 30  # the fit path's own reference precision; a spawn does not carry it
    return flat_order_f32(job)


def flat_table_f32():
    """The float lane's uniform table as stored fits, with its measured bound.

    Every order is walked, measured and returned independently, so the pool
    order cannot reach the result. The bounds are the table's own delivered
    figures per scheme and multiply-add route, rounded up the way every other
    table's are. The per-order figures travel back beside the maxima because a
    maximum that hides its cells hides which order set it, and the orders of this
    table do not all deliver the same figure. The ladder is walked from the
    widest width down and a row is taken when its own measured table holds the
    bound, for the reason flat_table gives - and this lane is the one that needs
    it: its bar is 1e-7 and half an ulp of 1 in its format is 5.96e-8, so what it
    delivers is its rounding and the rounding is not monotone in the argument.
    """
    for width in FLAT_LADDER:
        grid = uniform_row("float", width)
        if grid is None:
            continue
        jobs = [(n, width, grid["degs"]) for n in range(MAX_ORDER + 1)]
        out = run_jobs(_flat_f32_job, jobs)
        per_order = [intervals for _n, intervals, _w in out]
        worst = [[0.0, 0.0], [0.0, 0.0]]
        for _n, _intervals, w in out:
            for scheme in range(NARROW_SCHEMES):
                for route in (0, 1):
                    if w[scheme][route] > worst[scheme][route]:
                        worst[scheme][route] = w[scheme][route]
        if all(worst[s][r] <= grid["bound"]
               for s in range(NARROW_SCHEMES) for r in (0, 1)):
            break
        print(f"    the float uniform table does not hold its bound at width "
              f"{mp.nstr(width, 3)}: worst {max(max(r) for r in worst):.6e} against "
              f"{mp.nstr(grid['bound'], 3)}; trying the next width down")
    else:
        raise RuntimeError(
            f"the float uniform table holds {mp.nstr(FLAT_CELL_BOUND['float'], 3)} at "
            f"no dyadic width in {[mp.nstr(w, 3) for w in FLAT_LADDER]}: the lane's "
            f"own rounding is above the bound at every grid the ladder carries")
    bounds = [[scheme_bound(worst[s][r]) for r in (0, 1)]
              for s in range(NARROW_SCHEMES)]
    return {"orders": per_order, "worst": worst, "bounds": bounds, "grid": grid,
            "per_order": {n: w for n, _intervals, w in out}}


# The float lane's rational route
#
# The double lane's rational construction at the float lane's own target. Two
# things differ from it, and both are forced by the format rather than chosen:
#
# - The lane stores binary32, so the rounding of every solved coefficient is
#   part of the fit a caller receives. At a 1e-7 target that rounding is a
#   material term and not a last-digit detail, so a piece is accepted on the
#   coefficients AS STORED, measured in the lane's own arithmetic, and never
#   on the ones the exchange solved.
# - The lane maps x to t in binary32 too, so the grid is float-representable
#   arguments and the t values they map to, and the residual is read at those
#   t rather than at an exactly-mapped argument.
# - The lane's multiply-add has two routes and a build runs one of them, so
#   the acceptance and the published figure are both the worse of the two. A
#   criterion read at one route certifies a fit the other route delivers
#   worse, which is a figure the lane cannot stand behind.
# - Splitting the interval is not a way out of a target the lane's own
#   arithmetic cannot hold. Where binary32 rounding is what stands between a
#   fit and the target, the rounding does not shrink as the interval does, so a
#   branch can keep splitting without closing and the order is never reported.
#   The cover is therefore bounded by F32_RAT_MAX_DEPTH: an order no cover
#   holds within that bound is raised with both the count and the depth that
#   were tried, rather than walked towards through a tree as deep as the float
#   exponent range.
#
# The count ladder, the split of a count between numerator and denominator and
# the dyadic cover are the double route's: the smallest stored count holding
# the target, every split of it tried, and a cover taken where one piece
# cannot. The target IS the lane's published bound, so the criterion and the
# promise are the same sentence: a piece is accepted exactly when the figure it
# delivers - the lane's own arithmetic, the worse of the two multiply-add
# routes, weighted by the recursion's gain - holds the bar the lane publishes.
# Half the lane's tolerance is not that, and is not a criterion at all: at the
# two lowest orders what stands between the family and a smaller number is
# binary32's own rounding in the Horner evaluation (F0) and the family's
# approximation (F1), in both routes, and neither is removed by any admissible
# pair - so a target below the bar yields a generator that cannot run rather
# than a table that is better.
F32_RAT_DPS = 30
F32_RAT_REF_DPS = 40
F32_RAT_BOUND = mpf("1.5e-7")
F32_RAT_ACCEPT = F32_RAT_BOUND
# The narrow partition's own aim, as against the acceptance above: the aim
# decides how far the walk goes, the bar decides what ships. The narrow pieces
# are aimed at half the lane's tolerance - the margin a relaxed rung's budget
# spends - and kept on F32_RAT_BOUND, which is the figure the lane is
# certified against. The aim is the walk's first target and the bar its second:
# at the two lowest orders the binary32 evaluation cannot round closer than
# half an ulp of a value within 1e-4 of one - 5.96e-8 - so the pieces for those
# orders close at no aim below the bar, the walk returns None at the tighter
# one, and the second ask is the floor rather than a relaxation.
F32_RAT_NARROW_ACCEPT = TOL_FLOAT / 2
F32_RAT_GRID = 240
# The grid a piece's figure is READ on, as against the grid it is FITTED on
# above. The fit grid is uniform over the piece; the reading is taken over
# region A at F32_RAT_DENSE_GRID intervals and AT EVERY CELL THE ACCURACY GATE
# SWEEPS, because the gate judges a route at the arguments of its own reference
# grid - 646 of them below kX0 - and its verdict is what the lane publishes. A
# piece that holds on the fit grid's 241 points can be over the bar at a cell
# of the gate's: the committed table was, at one cell of F16 under the separate
# multiply-add route, and that is the red this route is being fitted against.
# The dense grid is what the reading falls back on between the gate's cells,
# which are logarithmically spaced and reach a spacing of 0.0745 near kX0.
F32_RAT_DENSE_GRID = 8 * F32_RAT_GRID
F32_RAT_SEARCH_GRID = 60
F32_RAT_COUNT_MIN = 3
F32_RAT_COUNT_MAX = 14
F32_RAT_K_MAX = 6
F32_RAT_SCAN_ITERS = 12
F32_RAT_POLISH_ITERS = 60
F32_RAT_MAX_DEPTH = 3
# The gate's committed reference grid, read for its arguments and the values it
# judges them against. Regenerating the float route without it would fit the
# promise on a grid the promise is not read on, so its absence is an error.
F32_RAT_GATE_REFERENCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "tests", "data",
    "boys_accuracy_gate_reference.csv")


def r32(v):
    """The float a stored constant rounds to."""
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def f32_map(a, b, x):
    """The kernel's mapped argument, in the kernel's width.

    The float lane maps x to t in binary32 - 2*(x - a)/(b - a) - 1 with every
    step a single rounding - so a grid of float arguments maps to the t values
    the kernel actually evaluates at.
    """
    span = r32(float(b) - float(a))
    if span == 0.0:
        return 0.0
    u = r32(r32(float(x) - float(a)) / span)
    return r32(2.0 * u - 1.0)


def f32_rat_grid(a, b, npts):
    """Float-representable arguments across [a, b] and the mapped t each."""
    a32, b32 = r32(a), r32(b)
    xs, ts = [], []
    for i in range(npts + 1):
        x = r32(a32 + (b32 - a32) * (i / npts))
        xs.append(x)
        ts.append(mpf(repr(f32_map(a32, b32, x))))
    return xs, ts


def f32_rat_reference(n, xs):
    with mp.workdps(F32_RAT_REF_DPS):
        return [boys_ref(n, mpf(x)) for x in xs]


_F32_RAT_DENSE = {}
_F32_RAT_GATE_CELLS = None
_F32_RAT_GATE = {}
# The regions the float lane's routes cover, and so the regions the acceptance
# grid is laid over: A, then B's single interval.
F32_RAT_DENSE_REGIONS = ((0.0, X0), (X0, X1))


def f32_rat_gate_cells():
    """The accuracy gate's own cells: its arguments and the values it reads.

    Read from the gate's committed reference grid rather than rebuilt, so the
    arguments are the gate's arguments and the reference values are the gate's
    own numbers, at the arguments the gate actually sweeps: the x column as a
    float and the value column evaluated at that float. Both of the float
    lane's regions are kept; a read over a piece slices out the cells that
    piece covers.
    """
    global _F32_RAT_GATE_CELLS
    if _F32_RAT_GATE_CELLS is None:
        import csv
        if not os.path.exists(F32_RAT_GATE_REFERENCE):
            raise RuntimeError(
                f"the float rational route's reading is taken at the accuracy gate's own "
                f"cells, and the gate's reference grid is missing: {F32_RAT_GATE_REFERENCE}")
        order = set()
        values = {}
        x1f = r32(X1)
        with open(F32_RAT_GATE_REFERENCE, newline="") as fh:
            for row in csv.DictReader(fh):
                x = r32(float(row["xf"]))
                if x >= x1f:
                    continue
                n = int(row["n"])
                order.add(x)
                values.setdefault(n, {})[x] = mpf(row["valuef"])
        _F32_RAT_GATE_CELLS = {"xs": sorted(order), "values": values}
    return _F32_RAT_GATE_CELLS


def f32_rat_dense_grid(region, n):
    """The acceptance grid over one region for one order: points, reference, weights.

    The mapped arguments are not held here: the mapping is the piece's, so it
    is taken where the piece is known. What is per order - the reference and
    the recursion's weight - is the expensive part, so it is built once per
    region and order and the pieces slice it.
    """
    if (region, n) not in _F32_RAT_DENSE:
        a, b = F32_RAT_DENSE_REGIONS[region]
        xs, _ts = f32_rat_grid(a, b, F32_RAT_DENSE_GRID)
        _F32_RAT_DENSE[(region, n)] = (xs, f32_rat_reference(n, xs),
                                       [seed_weight(n, mpf(x)) for x in xs])
    return _F32_RAT_DENSE[(region, n)]


def f32_rat_gate_grid(n):
    """The gate's cells for one order, with the reference and the weight.

    The arguments stay binary32 - the gate sweeps the float of its x column,
    and the mapped argument is taken from that float - so a cell and a grid
    point that are the same number are one point and not two.
    """
    if n not in _F32_RAT_GATE:
        cells = f32_rat_gate_cells()
        xs = cells["xs"]
        _F32_RAT_GATE[n] = (list(xs), [cells["values"][n][x] for x in xs],
                            [seed_weight(n, mpf(x)) for x in xs])
    return _F32_RAT_GATE[n]


def f32_rat_read_points(n, a, b, weighted=True):
    """The points a piece's figure is read on, in the piece's own mapping.

    Every acceptance region's grid and the gate's own cells, all sliced to
    [a, b] by the piece's float endpoints - the comparison the gate itself
    makes - plus the piece's own two ends, so a piece is read where it meets
    its neighbours as well as inside. One point per argument: a cell and a grid
    point coincide where the grid lands on the gate's grid, and a duplicate
    would be paid for twice and counted once.
    """
    a32, b32 = r32(a), r32(b)
    points = {}
    for region in range(len(F32_RAT_DENSE_REGIONS)):
        xs, ref, ws = f32_rat_dense_grid(region, n)
        for x, r, w in zip(xs, ref, ws):
            if a32 <= x <= b32 and x not in points:
                points[x] = (r, w)
    xs, ref, ws = f32_rat_gate_grid(n)
    for x, r, w in zip(xs, ref, ws):
        if a32 <= x <= b32 and x not in points:
            points[x] = (r, w)
    with mp.workdps(F32_RAT_REF_DPS):
        for x in (a32, b32):
            if x not in points:
                points[x] = (boys_ref(n, mpf(x)), seed_weight(n, mpf(x)))
    return [(x, f32_map(a32, b32, x), r, w if weighted else mpf(1))
            for x, (r, w) in sorted(points.items())]


def f32_rat_dense_read(n, a, b, p, q, weighted=True):
    """A piece's figure as the promise reads it.

    Both multiply-add routes over every point of f32_rat_read_points, the worse
    of the two taken, weighted by the recursion's gain where the reading is the
    acceptance's. Returns the weighted worst and its argument, the worse
    route's unweighted worst and its argument - the figure the lane publishes -
    and each route's own worst.
    """
    ww, wat = mpf(0), None
    dw, dat = mpf(0), None
    fw, sw = mpf(0), mpf(0)
    for (x, t, r, w) in f32_rat_read_points(n, a, b, weighted):
        ef = abs(mpf(rational_value_float(p, q, float(t), True)) - r)
        es = abs(mpf(rational_value_float(p, q, float(t), False)) - r)
        fw, sw = max(fw, ef), max(sw, es)
        e = max(ef, es)
        if e * w > ww:
            ww, wat = e * w, x
        if e > dw:
            dw, dat = e, x
    return ww, wat, dw, dat, fw, sw


def f32_rat_worst(p, q, t, ref_i):
    """The worse of the two routes' |fit - reference| at one argument."""
    return max(abs(mpf(rational_value_float(p, q, float(t), fused)) - ref_i)
               for fused in (True, False))


def f32_rat_residual(p, q, ts, ref, ws):
    """The weighted residual of the coefficients AS STORED, in the lane's own
    arithmetic, and where it is worst.

    The two multiply-add routes are both measured and the worse of the two is
    the residual: a build runs one of them and the lane publishes one figure,
    so a pair is accepted only where it holds the target whichever route the
    build in force selected."""
    worst, at = mpf(0), None
    for i, t in enumerate(ts):
        e = f32_rat_worst(p, q, t, ref[i]) * ws[i]
        if e > worst:
            worst, at = e, t
    return worst, at


def f32_rat_delivered(p, q, ts, ref):
    """The unweighted error of the stored coefficients in the lane's own
    arithmetic, the figure a route publishes: the worse of the two routes, for
    the same reason the residual takes the worse."""
    worst, at = mpf(0), None
    for i, t in enumerate(ts):
        e = f32_rat_worst(p, q, t, ref[i])
        if e > worst:
            worst, at = e, t
    return worst, at


def f32_rat_try(m, k, indices, ts, ref, ws, iters):
    """The exchange's fit at a degree pair, measured on its stored coefficients."""
    r = rat_remez(m, k, indices, ts, ref, ws, iters=iters)
    if r is None:
        return None
    _, p, q = r
    d, at = f32_rat_residual(p, q, ts, ref, ws)
    return (d, p, q, at)


def f32_rat_piece(n, a, b, weighted=True, accept=None):
    """The smallest stored pair holding the lane's target on [a, b), or None.

    `accept` is the target the walk aims at, and a caller that names none is
    asking for the class's own acceptance. The narrow partition names its own
    and, where nothing closes at it, asks again here at the default: the
    tightest target a piece can hold is a property of that piece's order and
    interval, so a walk with a tighter aim returns the pieces that hold it and
    None for the orders whose binary32 evaluation cannot, and the caller that
    wants the second group fitted anyway asks the second time.

    The count search walks the stored count upward and at each count every
    split of it into numerator and denominator, so the piece returned is the
    cheapest the family offers rather than the cheapest of one split. Two grids
    are in play and only one of them decides. The piece's own F32_RAT_GRID
    points are the FIT grid: the exchange solves on them, at one node in
    F32_RAT_SEARCH_GRID for the count scan and at every node for the polish.
    The acceptance is read on f32_rat_read_points - the acceptance grid and the
    accuracy gate's own cells - because that is where the promise is read: the
    gate judges the route at its own arguments, and a pair that holds on the
    fit grid alone can be over the bar at one of them. Every split the coarse
    scan passes is polished and then read on the acceptance's points, and the
    split that ships at a count is the one whose acceptance reading is lowest,
    so the fit grid can only make a piece cost more than the family's minimum
    and never less: a count whose every split fails the reading resumes the
    search above itself.
    """
    xs, ts = f32_rat_grid(a, b, F32_RAT_GRID)
    ref = f32_rat_reference(n, xs)
    ws = [seed_weight(n, mpf(x)) if weighted else mpf(1) for x in xs]
    sx = list(range(0, F32_RAT_GRID + 1, F32_RAT_GRID // F32_RAT_SEARCH_GRID))
    full = list(range(F32_RAT_GRID + 1))
    target = F32_RAT_ACCEPT if accept is None else accept

    for total in range(F32_RAT_COUNT_MIN, F32_RAT_COUNT_MAX + 1):
        best = None
        chosen = None
        for k in range(1, min(F32_RAT_K_MAX, total - 2) + 1):
            m = total - 1 - k
            if m < 1:
                continue
            r = f32_rat_try(m, k, sx, ts, ref, ws, F32_RAT_SCAN_ITERS)
            if r is None or r[0] > target:
                continue
            cand = f32_rat_try(m, k, full, ts, ref, ws, F32_RAT_POLISH_ITERS)
            if cand is None or cand[0] > target:
                continue
            _, p, q, _ = cand
            read = f32_rat_dense_read(n, a, b, p, q, weighted)
            if read[0] > target:
                continue
            if best is None or read[0] < best:
                best, chosen = read[0], (m, k, p, q) + read
        if chosen is None:
            continue
        m, k, p, q, ww, wat, dw, dat, fw, sw = chosen
        return {"a": r32(a), "b": r32(b), "m": m, "k": k, "p": p, "q": q,
                "count": m + 1 + k, "residual": ww, "residual_at": wat,
                "delivered": dw, "delivered_at": dat, "fused": fw, "separate": sw}
    return None


def f32_rat_cover(n, a, b, depth):
    """The cheapest dyadic cover of [a, b) the family holds the target on.

    The split is bounded by F32_RAT_MAX_DEPTH, and the bound is what makes the
    target a claim the generator can report on rather than one it can only walk
    towards. Splitting is not a way out of a target the lane's own arithmetic
    cannot hold: where a value's binary32 rounding is what stands between the
    fit and the target, the rounding does not shrink as the interval does, so a
    branch can keep splitting without ever closing - unbounded, the walk
    descends past the region's own scale and the order is never reported.
    Bounded, it returns None and the caller says which order and which targets
    were tried.
    """
    options = []
    single = f32_rat_piece(n, a, b)
    if single is not None:
        options.append([single])
    if depth < F32_RAT_MAX_DEPTH:
        mid = r32((r32(a) + r32(b)) / 2.0)
        if r32(a) < mid < r32(b):
            left = f32_rat_cover(n, a, mid, depth + 1)
            right = f32_rat_cover(n, mid, b, depth + 1)
            if left is not None and right is not None:
                options.append(left + right)
    if not options:
        return None
    return min(options, key=lambda cover: sum(pc["count"] for pc in cover))


def fma32(a, b, c):
    """One fused multiply-add in binary32.

    a*b + c with binary32 inputs is exact in binary64 (the product of two
    24-bit significands is 48 bits wide), so rounding that sum once to
    binary32 is the correctly rounded fused step and not an approximation of
    it.
    """
    return r32(float(a) * float(b) + float(c))


def _round32_exact(fr):
    """The binary32 nearest-even rounding of an exact Fraction.

    The separate multiply-add's second rounding is a rounding of an exact sum
    of two binary32 values, and rounding that sum through binary64 first would
    round twice. The sum is therefore formed exactly and rounded once, here, so
    the figure the routed models deliver is the figure the kernel's arithmetic
    delivers rather than a close relative of it.
    """
    if fr == 0:
        return 0.0
    neg = fr < 0
    num, den = (fr.numerator, fr.denominator) if not neg else (-fr.numerator, fr.denominator)
    e = num.bit_length() - den.bit_length()
    if (num < (den << e)) if e >= 0 else ((num << -e) < den):
        e -= 1
    if e < -126:
        raise ValueError("binary32 subnormal: outside the range this model rounds in")
    shift = 23 - e
    if shift >= 0:
        div = den
        q, r = divmod(num << shift, den)
    else:
        div = den << -shift
        q, r = divmod(num, div)
    # The half-way test is against the divisor actually used: past e > 23 the
    # divisor is den scaled up, and testing r against den alone rounds every
    # such tie the wrong way by one unit in the last place.
    if 2 * r > div or (2 * r == div and (q & 1)):
        q += 1
    if q >= (1 << 24):
        q >>= 1
        e += 1
    if e > 127:
        raise ValueError("binary32 overflow")
    # q was scaled by exactly 2^shift = 2^(23-e), so it carries the value
    # (num/den) * 2^(23-e) and the result is q * 2^(e-23): the shift comes back
    # out of the exponent. Dropping it is a factor of 2^23 in every cancellation
    # this path exists to round.
    v = math.ldexp(q, e - 23)
    return -v if neg else v


def add32(a, b):
    """a + b, correctly rounded to binary32 (nearest, ties to even)."""
    fa, fb = float(a), float(b)
    if fa == 0.0 or fb == 0.0:
        return r32(fa + fb)
    # The exact sum of two binary32 values fits in binary64 while their
    # exponents are within 29, so the binary64 sum is exact and one rounding to
    # binary32 is the correctly rounded result. Past that the sum is formed
    # exactly and rounded once, so the model is the kernel's arithmetic rather
    # than a close relative of it.
    if abs(math.frexp(fa)[1] - math.frexp(fb)[1]) <= 29:
        return r32(fa + fb)
    return _round32_exact(Fraction(fa) + Fraction(fb))


def sub32(a, b):
    """a - b, correctly rounded to binary32 (nearest, ties to even)."""
    fb = float(b)
    if fb == 0.0:
        return r32(float(a))
    return add32(a, -fb)


def step32(fused, a, b, c):
    """One multiply-add in binary32 at a named multiply-add route.

    The kernel's step is `ScalarFp32::MulAdd`: one rounding under the fused
    route and two under the separate one. A fit accepted on the fused reading
    alone would be a fit whose bound was never taken on the arithmetic a
    BOYS_MULADD_SEPARATE build runs, which is the defect a table this lane
    shipped and withdrew was withdrawn for; every acceptance of this lane's own
    partitions therefore reads both and takes the worse.
    """
    if fused:
        return fma32(a, b, c)
    return add32(r32(float(a) * float(b)), c)


def clenshaw_route32(cs, t, fused):
    """ClenshawSplit<ScalarFp32>, operation for operation, at one route."""
    c = [r32(v) for v in cs]
    t = r32(t)
    deg = len(c) - 1
    if deg == 0:
        return c[0]
    if deg == 1:
        return step32(fused, t, c[1], c[0])
    v = step32(fused, 2.0, r32(t * t), -1.0)
    two_v = add32(v, v)
    if deg == 2:
        return step32(fused, t, c[1], step32(fused, v, c[2], c[0]))
    m = deg // 2
    b1, b2 = c[2 * m], 0.0
    for k in range(m - 1, 0, -1):
        b0 = step32(fused, two_v, b1, sub32(c[2 * k], b2))
        b2, b1 = b1, b0
    even = step32(fused, v, b1, sub32(c[0], b2))
    o1, o2 = c[2 * m - 1], 0.0
    for k in range(m - 2, 0, -1):
        o0 = step32(fused, two_v, o1, sub32(c[2 * k + 1], o2))
        o2, o1 = o1, o0
    odd = step32(fused, sub32(two_v, 1.0), o1, sub32(c[1], o2))
    return step32(fused, t, odd, even)


def horner_mono_route32(ms, t, fused):
    """HornerMono<ScalarFp32>, operation for operation, at one route."""
    c = [r32(v) for v in ms]
    t = r32(t)
    acc = c[-1]
    for j in range(len(c) - 2, -1, -1):
        acc = step32(fused, acc, t, c[j])
    return acc


def rational_value_route32(p, q, t, fused):
    """The kernel's rational evaluation in binary32, at one multiply-add route.

    `rational_value_float` below is the same evaluation with the step fixed at
    the fused form; this one names the route, which is what an acceptance that
    has to bound both builds' arithmetic reads.
    """
    c = [r32(v) for v in p] + [r32(v) for v in q]
    m = len(p) - 1
    k = len(q)
    t = r32(t)
    num = c[m]
    for j in range(m - 1, -1, -1):
        num = step32(fused, num, t, c[j])
    if k == 0:
        return num
    den = c[m + k]
    for j in range(k - 1, 0, -1):
        den = step32(fused, den, t, c[m + j])
    return r32(num / step32(fused, den, t, 1.0))


def clenshaw_split_float(cs, t, fused=True):
    """ClenshawSplit<ScalarFp32>, operation for operation.

    The shipped split Clenshaw in the lane's own width, at the caller's
    multiply-add route, so a delivered error measured here is the error the
    entry delivers rather than the error a wider evaluation of the same
    coefficients would. Every step the kernel spells as a multiply-add is one
    here, at that route; the steps it spells as a plain add, subtract or
    multiply are the same rounding either way.
    """
    c = [r32(v) for v in cs]
    t = r32(t)
    deg = len(c) - 1
    if deg == 0:
        return c[0]
    if deg == 1:
        return route_step32(fused, t, c[1], c[0])
    v = route_step32(fused, 2.0, r32(t * t), -1.0)
    two_v = r32(v + v)
    if deg == 2:
        return route_step32(fused, t, c[1], route_step32(fused, v, c[2], c[0]))
    m = deg // 2
    b1, b2 = c[2 * m], 0.0
    for k in range(m - 1, 0, -1):
        b0 = route_step32(fused, two_v, b1, r32(c[2 * k] - b2))
        b2, b1 = b1, b0
    even = route_step32(fused, v, b1, r32(c[0] - b2))
    o1, o2 = c[2 * m - 1], 0.0
    for k in range(m - 2, 0, -1):
        o0 = route_step32(fused, two_v, o1, r32(c[2 * k + 1] - o2))
        o2, o1 = o1, o0
    odd = route_step32(fused, r32(two_v - 1.0), o1, r32(c[1] - o2))
    return route_step32(fused, t, odd, even)


def horner_mono_float(ms, t, fused=True):
    """HornerMono<ScalarFp32>, operation for operation.

    The monomial form of the same fit read by the other scheme, in the lane's
    own width and at the caller's multiply-add route, so a delivered error
    measured here is the error the entry delivers under that scheme rather
    than the error the Chebyshev recurrence would have carried.
    """
    c = [r32(v) for v in ms]
    t = r32(t)
    acc = c[-1]
    for j in range(len(c) - 2, -1, -1):
        acc = route_step32(fused, acc, t, c[j])
    return acc


def rational_value_float(p, q, t, fused):
    """The kernel's rational evaluation, in the lane's own arithmetic, at one
    route.

    Every step is the backend's multiply-add at that route, so the value
    returned is the one the entry spells rather than the one a wider
    evaluation of the same coefficients would reach."""
    c = [r32(v) for v in p] + [r32(v) for v in q]
    m = len(p) - 1
    k = len(q)
    t = r32(t)
    num = c[m]
    for j in range(m - 1, -1, -1):
        num = route_step32(fused, num, t, c[j])
    if k == 0:
        return num
    den = c[m + k]
    for j in range(k - 1, 0, -1):
        den = route_step32(fused, den, t, c[m + j])
    return r32(num / route_step32(fused, den, t, 1.0))


def f32_rat_delivered_float(cover, n):
    """The rational route's worst |F_n - fit| over its own pieces, as delivered.

    Both multiply-add routes are measured and the worse is reported, so the
    figure the lane publishes covers the arithmetic of either build. The figure
    is read on the acceptance's points - the acceptance grid and the gate's own
    cells - so what is published is the number the piece was accepted on and
    not a coarser sweep of it.
    """
    worst = mpf(0)
    for pc in cover:
        _, _, dw, _, _, _ = f32_rat_dense_read(n, pc["a"], pc["b"], pc["p"], pc["q"])
        if dw > worst:
            worst = dw
    return worst


def f32_rat_cheb_delivered(float_orders):
    """The shipped float lane's own worst |F_n - fit| over [0, X0).

    Measured on the same construction the rational route is measured on -
    each piece read on the same points, the float-mapped argument, the
    reference the fits are validated against, the lane's own arithmetic - so
    the two routes' delivered figures are a comparison rather than two
    measurements. Both multiply-add routes are measured and the worse is
    reported, for the reason the rational route's figure takes the worse: a
    build runs one of them, and a figure that covers one of the two understates
    the other by more than the gate's shortfall allowance - which is what the
    separate build's reading of this figure was.
    """
    worst = mpf(0)
    for n in range(MAX_ORDER + 1):
        for (a, b, deg, cs, _ms) in float_orders[n]:
            for (_x, t, r, _w) in f32_rat_read_points(n, a, b):
                for fused in (True, False):
                    e = abs(mpf(clenshaw_split_float(cs, float(t), fused)) - r)
                    if e > worst:
                        worst = e
    return worst


def f32_rat_cheb_delivered_horner(float_orders):
    """The same lane, the same points, the same reference, read by Horner.

    Only the summation differs from f32_rat_cheb_delivered above - same
    pieces, same arguments, same coefficients as stored, the same two routes
    with the worse taken - so the two figures are a comparison of the two
    schemes rather than two measurements.
    """
    worst = mpf(0)
    for n in range(MAX_ORDER + 1):
        for (a, b, deg, cs, ms) in float_orders[n]:
            for (_x, t, r, _w) in f32_rat_read_points(n, a, b):
                for fused in (True, False):
                    e = abs(mpf(horner_mono_float(ms, float(t), fused)) - r)
                    if e > worst:
                        worst = e
    return worst


def fit_region_a_rational_f32(float_orders):
    """The float lane's region-A rational cover: every order over [0, X0).

    The pieces are the family's own cover of each order's interval rather than
    the shipped Chebyshev table's pieces: a cover is free to place its breaks
    where the fit needs them, and the shipped table's breaks are where that
    family - a different one - needed its own.
    """
    print(f"fitting the float lane's rational region-A route ({F32_RAT_DPS} dps, the lane's "
          f"own bar {mp.nstr(F32_RAT_BOUND, 2)} as the target: the coefficients as stored, in "
          f"the lane's own arithmetic, both multiply-add routes, the worse taken, fitted "
          f"under the shipped weighting, and read on the region's acceptance grid together "
          f"with every cell the accuracy gate sweeps in it) ...")
    orders = []
    stored = 0
    pieces = 0
    delivered = mpf(0)
    at = None
    residual = mpf(0)
    for n in range(MAX_ORDER + 1):
        cover = f32_rat_cover(n, 0.0, X0, 0)
        if cover is None:
            raise RuntimeError(f"float rational region-A route: F{n} on [0, {X0}) has no "
                               f"cover within {F32_RAT_MAX_DEPTH} level(s) of splitting "
                               f"whose every piece holds {mp.nstr(F32_RAT_ACCEPT, 2)} at "
                               f"a stored count up to {F32_RAT_COUNT_MAX}")
        orders.append(cover)
        stored += sum(pc["count"] for pc in cover)
        pieces += len(cover)
        route_delivered = f32_rat_delivered_float(cover, n)
        if route_delivered > delivered:
            delivered = route_delivered
        for pc in cover:
            if pc["residual"] > residual:
                residual = pc["residual"]
        print("  F{:2d}: {} stored in {} piece(s), delivered {} ({})".format(
            n, sum(pc["count"] for pc in cover), len(cover),
            mp.nstr(route_delivered, 3),
            ", ".join("[{}, {}) {}/{} {}".format(
                mp.nstr(mpf(pc["a"]), 6), mp.nstr(mpf(pc["b"]), 6),
                pc["m"], pc["k"], mp.nstr(pc["delivered"], 3)) for pc in cover)))
        sys.stdout.flush()

    print(f"  region A: rational {stored} stored in {pieces} pieces, delivered "
          f"{mp.nstr(delivered, 6)}, worst weighted residual {mp.nstr(residual, 6)}")
    cheb_delivered = f32_rat_cheb_delivered(float_orders)
    cheb_delivered_horner = f32_rat_cheb_delivered_horner(float_orders)
    cheb_stored = sum(sum(d + 1 for _, _, d, _, _ in float_orders[n])
                      for n in range(MAX_ORDER + 1))
    cheb_pieces = sum(len(float_orders[n]) for n in range(MAX_ORDER + 1))
    print(f"  region A: Chebyshev {cheb_stored} stored in {cheb_pieces} pieces, delivered "
          f"{mp.nstr(cheb_delivered, 6)} (split Clenshaw), "
          f"{mp.nstr(cheb_delivered_horner, 6)} (Horner)")
    return {"orders": orders, "stored": stored, "pieces": pieces,
            "delivered": delivered, "at": at, "residual": residual,
            "cheb_stored": cheb_stored, "cheb_pieces": cheb_pieces,
            "cheb_delivered": cheb_delivered,
            "cheb_delivered_horner": cheb_delivered_horner,
            "bound": F32_RAT_BOUND, "lo": mpf(0), "hi": X0}


def f32_rat_region_b_delivered():
    """The two float region-B seeds' worst |F_0 - fit| over [kX0, kX1).

    The shipped seed first, then the rational one, both read on the same
    points - region B's own acceptance grid and the gate's cells in the
    interval - in the lane's own arithmetic, against the reference the fits are
    validated against, so the two stored counts sit beside a comparison rather
    than beside two measurements. Both multiply-add routes are read and the
    worse is taken, for the reason the region-A figures take the worse.
    """
    deg, cs, mono = fit_order(0, region_b=True, f32=True)
    cheb = mpf(0)
    horner = mpf(0)
    for (_x, t, r, _w) in f32_rat_read_points(0, X0, X1):
        for fused in (True, False):
            e = abs(mpf(clenshaw_split_float(cs, float(t), fused)) - r)
            if e > cheb:
                cheb = e
            e = abs(mpf(horner_mono_float(mono, float(t), fused)) - r)
            if e > horner:
                horner = e
    return deg + 1, cheb, horner


def fit_region_b_rational_f32():
    """The float lane's region-B F_0 seed from the region's own minimax pair.

    The weight is one here: region B reads the seed and recurses upward, so
    there is no amplification between the fit and the value a caller receives
    and the criterion is the lane's own tolerance.
    """
    seed = f32_rat_piece(0, X0, X1, weighted=False)
    if seed is None:
        raise RuntimeError(f"float rational region-B seed: no stored count up to "
                           f"{F32_RAT_COUNT_MAX} holds {mp.nstr(F32_RAT_ACCEPT, 2)} on "
                           f"[{X0}, {X1})")
    delivered = f32_rat_delivered_float([seed], 0)
    cheb_stored, cheb_delivered, cheb_delivered_horner = f32_rat_region_b_delivered()
    seed["delivered"] = delivered
    seed["cheb_stored"] = cheb_stored
    seed["cheb_delivered"] = cheb_delivered
    seed["cheb_delivered_horner"] = cheb_delivered_horner
    return seed


def f32_rat_route_delivered(p, q, n, a, b, npts, weighted):
    """Worst |F_n - fit| in the lane's own binary32, per multiply-add route.

    `f32_rat_residual` above is this figure taken in exact arithmetic on the
    rounded coefficients: the reading the shipped float rational pieces were
    accepted on, and the reading that never bounded the arithmetic the entry
    runs, because the entry rounds at every step of the Horner numerator and
    denominator and rounds twice on the separate multiply-add route. This one
    evaluates the kernel's own chain - `rational_value_route32`, operation for
    operation - so an acceptance that reads it is taken on the delivered figure
    rather than beside it.
    """
    a32, b32 = r32(a), r32(b)
    worst = [0.0, 0.0]
    for i in range(npts + 1):
        x = r32(a32 + (b32 - a32) * (i / npts))
        if x >= b32:
            x = math.nextafter(b32, 0.0)
        ref = float(boys_ref(n, mpf(x)))
        t = f32_map(a32, b32, x)
        w = float(seed_weight(n, mpf(x))) if weighted else 1.0
        for index, fused in ((0, True), (1, False)):
            e = abs(float(rational_value_route32(p, q, t, fused)) - ref) * w
            if e > worst[index]:
                worst[index] = e
    return worst


def f32_narrow_rat_piece(n, a, b, weighted, depth=0):
    """The family's rational pieces over [a, b), in the kernel's arithmetic.

    The family search is the shipped one - `f32_rat_piece` walks the stored
    count upward and every split of it into numerator and denominator, and
    accepts on the lane's own weighted target - and two things are added here:
    the aim the walk is asked for, and the arithmetic the piece is kept on. The
    walk is asked first for F32_RAT_NARROW_ACCEPT, half the lane's tolerance,
    which is the margin the packed lane's rung spends; only where nothing
    closes at it is the same piece asked for again at the lane's own bar.

    That second ask is not a relaxation of the aim but the floor the order has:
    the aim is not reachable at the lowest orders, where what stands between
    the family and a smaller number is binary32's own rounding of F_0 near
    x = 0 - half an ulp of a value within 1e-4 of one, 5.96e-8 - and no
    admissible pair removes it. Measured on this partition, the pieces nearest
    zero come back at 4.9e-8 and the ones that do not are floor failures. Every
    other order closes at the aim: the piece holding the accuracy gate's worst
    rational cell - n = 30 over [5.949924, 8.924886) - walks to count 5 at the
    aim and reads 1.230189e-8 there, against count 4 and 1.470125e-7 when the
    same piece is asked only at the bar, and it is that reading the gate's own
    reference grid sees where the bar-level fit sits 1.38e-7 over the half
    lane's 1e-7.

    Whichever ask returned the piece, it is kept only when the kernel's own
    binary32 evaluation holds the lane's bar too, under both multiply-add
    routes with the worse taken, which is what the lanes are certified against.
    A piece that does not is bisected, since the fit's own error falls with the
    interval even where the rounding's does not; the count that costs is then
    the piece count this returns, which the caller reports.
    """
    piece = f32_rat_piece(n, a, b, weighted=weighted, accept=F32_RAT_NARROW_ACCEPT)
    if piece is None:
        piece = f32_rat_piece(n, a, b, weighted=weighted)
    if piece is not None:
        routes = f32_rat_route_delivered(piece["p"], piece["q"], n, a, b,
                                         F32_NARROW_GRID, weighted)
        piece["fused"], piece["separate"] = routes[0], routes[1]
        if routes[0] <= F32_RAT_BOUND and routes[1] <= F32_RAT_BOUND:
            return [piece]
    if depth >= F32_RAT_MAX_DEPTH:
        return None
    mid = r32((r32(a) + r32(b)) / 2.0)
    if not (r32(a) < mid < r32(b)):
        return None
    left = f32_narrow_rat_piece(n, a, mid, weighted, depth + 1)
    right = f32_narrow_rat_piece(n, mid, b, weighted, depth + 1)
    if left is None or right is None:
        return None
    return left + right


def _f32_narrow_rat_job(job):
    n, pieces, weighted = job
    mp.dps = F32_RAT_DPS  # the scan's working precision; a spawn does not carry it
    out = []
    for (a, b, _deg, _cs, _ms) in pieces:
        cover = f32_narrow_rat_piece(n, mpf(a), mpf(b), weighted)
        if cover is None:
            raise RuntimeError(
                f"float rational narrow region: F{n} on [{a}, {b}) reaches no piece "
                f"holding the lane's {mp.nstr(F32_RAT_BOUND, 2)} bar in its own binary32 "
                f"at both multiply-add routes within {F32_RAT_MAX_DEPTH} bisection(s)")
        out.append(cover)
    return n, out


def fit_narrow_rational_f32(narrow_a_f32, narrow_b_f32):
    """The float lane's rational route over its own narrow partition.

    The partitions are the lane's narrow Chebyshev ones, because a partition is
    a cut of the region and not a property of a family; what the rational
    family brings is its own degree pair per piece, read at that piece's own
    interval and mapped argument. The search is the lane's own rational one -
    the weighted residual of the coefficients as stored, walked down to
    F32_RAT_NARROW_ACCEPT and asked again at the lane's own acceptance where an
    order's floor stands above the aim - and the piece is kept when the figure
    the entry actually delivers holds the lane's own F32_RAT_BOUND bar, read in
    binary32 at both multiply-add routes with the worse drawn. See
    f32_narrow_rat_piece for which orders the aim closes on and which the floor
    holds. Where a piece holds neither, it is bisected and the extra pieces are
    the count the criterion costs; the figure is reported either way.
    """
    print(f"fitting the float lane's rational route over its narrow partition "
          f"({F32_RAT_DPS} dps, the narrow partition's own "
          f"{mp.nstr(F32_RAT_NARROW_ACCEPT, 2)} aim, "
          f"kept at the lane's {mp.nstr(F32_RAT_BOUND, 2)} bar read in binary32 at BOTH "
          f"multiply-add routes) ...")
    jobs = [(n, narrow_a_f32["orders"][n], True) for n in range(MAX_ORDER + 1)]
    out = run_jobs(_f32_narrow_rat_job, jobs)
    orders = [[] for _ in range(MAX_ORDER + 1)]
    stored = 0
    pieces = 0
    fused = 0.0
    separate = 0.0
    for n, covers in out:
        for cover in covers:
            for piece in cover:
                orders[n].append(piece)
                stored += piece["count"]
                pieces += 1
                fused = max(fused, piece["fused"])
                separate = max(separate, piece["separate"])
    n_pieces_a = sum(len(narrow_a_f32["orders"][n]) for n in range(MAX_ORDER + 1))
    print(f"  region A: rational {stored} stored in {pieces} piece(s) over the Chebyshev "
          f"partition's {n_pieces_a}, delivered fused {fused:.6e} / separate {separate:.6e}")

    b_jobs = [(0, [piece], False) for piece in narrow_b_f32["pieces"]]
    b_out = run_jobs(_f32_narrow_rat_job, b_jobs)
    b_pieces = [piece for _, covers in b_out for cover in covers for piece in cover]
    b_stored = sum(piece["count"] for piece in b_pieces)
    b_fused = max(piece["fused"] for piece in b_pieces)
    b_separate = max(piece["separate"] for piece in b_pieces)
    print(f"  region B: rational {b_stored} stored in {len(b_pieces)} piece(s) over the "
          f"partition's {len(narrow_b_f32['pieces'])}, delivered fused {b_fused:.6e} / "
          f"separate {b_separate:.6e}")
    return {"orders": orders, "stored": stored, "pieces": pieces,
            "fused": fused, "separate": separate,
            "b_pieces": b_pieces, "b_stored": b_stored,
            "b_fused": b_fused, "b_separate": b_separate,
            "bound": F32_RAT_BOUND, "accept": F32_RAT_NARROW_ACCEPT}


def narrow_rat_f32_block_lines(f, narrow_rat_f32, narrow_b_f32):
    """The float lane's narrow partition under the rational route.

    The same shape `kRatAPieces`/`kRatACoeffs` and the region-B pair have, at
    the narrow pieces' own intervals: a piece table with its own degree pair
    per row, so a bisected piece is a row like any other.

    Every row was searched at the narrow partition's own aim, half the lane's
    tolerance, and asked again at the lane's own target where an order's
    binary32 floor stands above the aim; it ships only where the lane's own
    weighted target holds, read in the lane's own binary32 at both multiply-add
    routes with the worse taken, which is the arithmetic the entry runs and not
    the exact reading its shipped pieces were accepted on. The figure the block
    carries is that worse reading, swept.
    """
    f.write("\n// The float lane's rational route over its own narrow partition: one degree pair\n"
            "// per piece at each piece's own interval and mapped argument. The partitions\n"
            "// are the narrow Chebyshev ones; a piece it could not hold the target on is\n"
            "// bisected, and the piece table carries it. Accepted on the lane's own binary32.\n")
    a_pieces = [piece for n in range(MAX_ORDER + 1) for piece in narrow_rat_f32["orders"][n]]
    a_coeffs = []
    a_offsets = []
    for piece in a_pieces:
        a_offsets.append(len(a_coeffs))
        a_coeffs.extend(piece["p"])
        a_coeffs.extend(piece["q"])
    f.write("inline constexpr auto kNarrowRatACoeffsF32 = std::to_array<float>({\n")
    for i in range(0, len(a_coeffs), 6):
        f.write("  " + ", ".join(fmtf(v) for v in a_coeffs[i:i + 6]) + ",\n")
    f.write("});\n")
    f.write("inline constexpr auto kNarrowRatAPiecesF32 = std::to_array<RatPiece>({\n")
    starts = []
    index = 0
    for n in range(MAX_ORDER + 1):
        starts.append(index)
        for piece in narrow_rat_f32["orders"][n]:
            f.write(f"  {{{fmtf(piece['a'])}, {fmtf(piece['b'])}, {piece['m']}, "
                    f"{piece['k']}, {a_offsets[index]}}},  // F{n}\n")
            index += 1
    f.write("});\n")
    starts.append(index)
    f.write("inline constexpr auto kNarrowRatAPieceStartF32 = std::to_array<int>({"
            + ", ".join(map(str, starts)) + "});\n")
    f.write(f"inline constexpr int kNarrowRatAStoredF32 = {narrow_rat_f32['stored']};\n")
    f.write(f"inline constexpr double kNarrowRatADeliveredFusedF32 = "
            f"{fmt(mpf(narrow_rat_f32['fused']))};\n")
    f.write(f"inline constexpr double kNarrowRatADeliveredSeparateF32 = "
            f"{fmt(mpf(narrow_rat_f32['separate']))};\n")

    b_pieces = narrow_rat_f32["b_pieces"]
    b_coeffs = []
    b_offsets = []
    for piece in b_pieces:
        b_offsets.append(len(b_coeffs))
        b_coeffs.extend(piece["p"])
        b_coeffs.extend(piece["q"])
    f.write("\n// The same family over the narrow partition of region B, one pair per\n"
            "// piece, read at the piece's own interval and at the mapped argument the\n"
            "// narrow Chebyshev seed on that piece uses, so the two routes over a\n"
            "// piece read one t.\n")
    f.write("inline constexpr auto kNarrowRatBCoeffsF32 = std::to_array<float>({\n")
    for i in range(0, len(b_coeffs), 6):
        f.write("  " + ", ".join(fmtf(v) for v in b_coeffs[i:i + 6]) + ",\n")
    f.write("});\n")
    f.write("inline constexpr auto kNarrowRatBPiecesF32 = std::to_array<RatPiece>({\n")
    for index, piece in enumerate(b_pieces):
        f.write(f"  {{{fmtf(piece['a'])}, {fmtf(piece['b'])}, {piece['m']}, {piece['k']}, "
                f"{b_offsets[index]}}},\n")
    f.write("});\n")
    f.write(f"inline constexpr int kNarrowRatBPiecesCountF32 = {len(b_pieces)};\n")
    f.write(f"inline constexpr int kNarrowRatBStoredF32 = {narrow_rat_f32['b_stored']};\n")
    f.write(f"inline constexpr double kNarrowRatBDeliveredFusedF32 = "
            f"{fmt(mpf(narrow_rat_f32['b_fused']))};\n")
    f.write(f"inline constexpr double kNarrowRatBDeliveredSeparateF32 = "
            f"{fmt(mpf(narrow_rat_f32['b_separate']))};\n")


def narrow_f32_block_lines(f, narrow_a_f32, narrow_b_f32):
    """The float lane's own narrow partition, as the header writes it.

    A function of its own because the block is what --narrow-only
    reproduces on its own: write_f32_namespace calls it and so does that
    path, so the two cannot drift.

    The shipped pieces are untouched by these tables and are the same bytes
    whether or not the partitions are named.
    """
    # The lane's own narrow partition: the same fitted regions at a narrower
    # degree, placed by the lane's own bisection law and measured in the lane's
    # own binary32 arithmetic at both multiply-add routes.
    f.write("\n// The float lane's narrow partition of region A: the lane's own weighted 1e-7 law\n"
            "// and the same midpoint bisection the shipped pieces were placed by, at degree\n"
            f"// {F32_NARROW_DEG} instead of the lane's degree cap. A lower degree accepts a narrower\n"
            "// interval: more pieces of fewer coefficients each, the granularity axis's trade.\n")
    f.write("inline constexpr int kNarrowADegF32 = " + str(F32_NARROW_DEG) + ";\n")
    f.write(deg_assert("kNarrowADegF32"))
    narrow_a32 = []
    narrow_a32_mono = []
    narrow_a32_meta = []
    for n in range(MAX_ORDER + 1):
        for (a, b, deg, cs, ms) in narrow_a_f32["orders"][n]:
            narrow_a32_meta.append((n, a, b, deg, len(narrow_a32)))
            narrow_a32.extend(fmtf(c) for c in cs)
            narrow_a32_mono.extend(fmtf(c) for c in ms)
    f.write("inline constexpr auto kNarrowACoeffsF32 = std::to_array<float>({\n")
    for i in range(0, len(narrow_a32), 6):
        f.write("  " + ", ".join(narrow_a32[i:i + 6]) + ",\n")
    f.write("});\n")
    f.write("inline constexpr auto kNarrowAMonoCoeffsF32 = std::to_array<float>({\n")
    for i in range(0, len(narrow_a32_mono), 6):
        f.write("  " + ", ".join(narrow_a32_mono[i:i + 6]) + ",\n")
    f.write("});\n")
    f.write("static_assert(std::size(kNarrowAMonoCoeffsF32) == std::size(kNarrowACoeffsF32),\n"
            "              \"the monomial table must parallel the Chebyshev table\");\n")
    f.write("inline constexpr auto kNarrowAPiecesF32 = std::to_array<OrderPiece>({\n")
    for (n, a, b, deg, off) in narrow_a32_meta:
        f.write(f"  {{{fmtf(mpf(a))}, {fmtf(mpf(b))}, {deg}, {off}}},  // F{n}\n")
    f.write("});\n")
    nstarts = []
    index = 0
    for n in range(MAX_ORDER + 1):
        nstarts.append(index)
        index += len(narrow_a_f32["orders"][n])
    nstarts.append(index)
    f.write("inline constexpr auto kNarrowAPieceStartF32 = std::to_array<int>({"
            + ", ".join(map(str, nstarts)) + "});\n")
    f.write("static_assert(std::size(kNarrowAPieceStartF32) == kMaxOrder + 2,\n"
            "              \"narrow piece-start table must cover kMaxOrder\");\n")

    f.write("\n// The float lane's narrow partition of region B: [kX0, kX1) cut until the figure\n"
            "// the lane's own recurrence carries is under half its budget. The seed's error\n"
            "// is amplified by the steps that follow it, so a piece is placed by the carried\n"
            "// figure and not its fit residual: the pieces nearest kX0 come out narrowest.\n")
    f.write("inline constexpr int kNarrowBDegF32 = " + str(F32_NARROW_DEG) + ";\n")
    f.write(deg_assert("kNarrowBDegF32"))
    nbp = narrow_b_f32["pieces"]
    f.write("inline constexpr int kNarrowBPiecesF32 = " + str(len(nbp)) + ";\n")
    edges = [fmtf(mpf(nbp[0][0]))] + [fmtf(mpf(p[1])) for p in nbp]
    f.write("inline constexpr auto kNarrowBEdgesF32 = std::to_array<float>({"
            + ", ".join(edges) + "});\n")
    nb_c = [fmtf(c) for p in nbp for c in p[3]]
    nb_m = [fmtf(c) for p in nbp for c in p[4]]
    f.write("inline constexpr auto kNarrowBcoeffsF32 = std::to_array<float>({\n")
    for i in range(0, len(nb_c), 6):
        f.write("  " + ", ".join(nb_c[i:i + 6]) + ",\n")
    f.write("});\n")
    f.write("inline constexpr auto kNarrowBMonoCoeffsF32 = std::to_array<float>({\n")
    for i in range(0, len(nb_m), 6):
        f.write("  " + ", ".join(nb_m[i:i + 6]) + ",\n")
    f.write("});\n")
    f.write("static_assert(std::size(kNarrowBEdgesF32) == kNarrowBPiecesF32 + 1\n"
            "                  && std::size(kNarrowBcoeffsF32) == kNarrowBPiecesF32 * (kNarrowBDegF32 + 1)\n"
            "                  && std::size(kNarrowBMonoCoeffsF32) == std::size(kNarrowBcoeffsF32),\n"
            "              \"the narrow partition's pieces must tile [kX0, kX1)\");\n")


    # What the narrow partition stores and what it delivers, in the lane's own
    # arithmetic at each multiply-add route: the figure a row published for it
    # is the worse of the two.
    f.write("\n// What the lane's narrow partition stores and what it delivers, per multiply-add\n"
            "// route, against the lane's reference: measured on each piece's own grid in\n"
            "// binary32 on the stored coefficients, under both schemes the entries sum by.\n"
            "// A swept maximum, not a bound: the bar is kRegionAFitBar, kRegionBFitBar for B.\n"
            "struct NarrowRowF32 { int scheme, deg, pieces, stored;\n"
            "                      double fused, separate; };\n"
            "inline constexpr auto kNarrowARowsF32 = std::to_array<NarrowRowF32>({\n")
    for scheme in range(NARROW_SCHEMES):
        f.write(f"  {{{scheme}, {F32_NARROW_DEG}, "
                f"{sum(len(p) for p in narrow_a_f32['orders'])}, "
                f"{sum(len(p) for p in narrow_a_f32['orders']) * (F32_NARROW_DEG + 1)}, "
                f"{fmt(narrow_a_f32['bounds'][scheme][0])}, "
                f"{fmt(narrow_a_f32['bounds'][scheme][1])}}},\n")
    f.write("});\n")
    f.write("inline constexpr auto kNarrowBRowsF32 = std::to_array<NarrowRowF32>({\n")
    for scheme in range(NARROW_SCHEMES):
        f.write(f"  {{{scheme}, {F32_NARROW_DEG}, {len(nbp)}, "
                f"{len(nbp) * (F32_NARROW_DEG + 1)}, "
                f"{fmt(narrow_b_f32['bounds'][scheme][0])}, "
                f"{fmt(narrow_b_f32['bounds'][scheme][1])}}},\n")
    f.write("});\n")


def flat_f32_block_lines(f, flat_f32):
    """The float lane's uniform table as the header writes it.

    The double table's shape on the grid the lane derived for itself. Interval-
    major, [interval][order][coefficient], so an argument's whole ladder is one
    contiguous block of kMaxOrder + 1 rows of that interval's own degree + 1
    coefficients each. Every stored coefficient is a binary32 literal and every
    published figure a binary64 one, the width pair the lane's other tables are
    written in, and the names carry the lane's F32 suffix, which is how its added
    tables are spelled beside the shipped ones in the one namespace they share.

    The grid constants are written because they are what the table was fitted on.
    They are not a locator: the single-precision uniform route has no kernel in
    this revision, so the mapping an entry will read the table with is that
    entry's own decision and not something these lines settle.
    """
    grid = flat_f32["grid"]
    degs = grid["degs"]
    cap = max(degs)
    intervals = grid["intervals"]
    offsets = flat_grid_offsets(grid)
    stored = offsets[-1]
    per_order = flat_f32["orders"]
    if [iv[2] for iv in per_order[0]] != list(degs):
        raise RuntimeError("the float uniform table's fits are not on the grid the "
                           "block writer is emitting: the degrees disagree")
    f.write(f"\n// The float lane's uniform table: the lane's own derived grid over [0, kFlatHiF32),\n"
            f"// width kFlatWidthF32 = {mp.nstr(grid['width'], 8)}, every order fitted at its interval's\n"
            f"// own degree (kFlatDegsF32), cap kFlatReadCapF32 = {cap}. Layout is the double table's,\n"
            "// interval-major: kFlatCoeffsF32[kFlatOffsetsF32[iv] + order * (kFlatDegsF32[iv] + 1)].\n")
    f.write(f"inline constexpr int kFlatReadCapF32 = {cap};\n")
    f.write(deg_assert("kFlatReadCapF32"))
    f.write(f"inline constexpr int kFlatIntervalsF32 = {intervals};\n")
    f.write(f"inline constexpr float kFlatWidthF32 = {fmtf(grid['width'])};\n")
    f.write(f"inline constexpr float kFlatHiF32 = {fmtf(grid['hi'])};\n")
    f.write("// One degree per interval: the smallest the cell law carries there, at\n"
            "// the read cap or below it, and even in every entry.\n")
    f.write("inline constexpr auto kFlatDegsF32 = std::to_array<int>({\n")
    for i in range(0, len(degs), 12):
        f.write("  " + ", ".join(str(d) for d in degs[i:i + 12]) + ",\n")
    f.write("});\n")
    f.write("// Where each interval's block starts, and the end of the last one:\n")
    f.write("inline constexpr auto kFlatOffsetsF32 = std::to_array<int>({\n")
    for i in range(0, len(offsets), 12):
        f.write("  " + ", ".join(str(o) for o in offsets[i:i + 12]) + ",\n")
    f.write("});\n")
    for name, column in (("kFlatCoeffsF32", 3), ("kFlatMonoCoeffsF32", 4)):
        values = [fmtf(c)
                  for iv in range(intervals)
                  for n in range(MAX_ORDER + 1)
                  for c in per_order[n][iv][column]]
        f.write(f"inline constexpr auto {name} = std::to_array<float>({{\n")
        for i in range(0, len(values), 6):
            f.write("  " + ", ".join(values[i:i + 6]) + ",\n")
        f.write("});\n")
    f.write(f"static_assert(std::size(kFlatCoeffsF32) == {stored} &&\n"
            f"                  std::size(kFlatMonoCoeffsF32) == {stored},\n"
            "              \"the float uniform table must hold every interval of every "
            "order\");\n")
    f.write(f"static_assert(std::size(kFlatDegsF32) == kFlatIntervalsF32 &&\n"
            f"                  std::size(kFlatOffsetsF32) == kFlatIntervalsF32 + 1,\n"
            "              \"the float uniform table must carry one degree and one offset "
            "per interval\");\n")
    f.write(f"static_assert(kFlatOffsetsF32[kFlatIntervalsF32] == {stored},\n"
            "              \"the offsets must reach the end of the float uniform "
            "table\");\n")
    f.write("\n// The float uniform table's certification rows, in the double table's form: the\n"
            "// bound each scheme delivers in each multiply-add route, worst over the whole\n"
            "// table, rounded up to a power of two so it bounds a sweep. Read in the lane's\n"
            "// own binary32 on the stored coefficients. deg is the cap; stored sums them.\n"
            "struct FlatRowF32 { int scheme, deg, intervals, stored;\n"
            "                    double fused, separate; };\n"
            "inline constexpr auto kFlatRowsF32 = std::to_array<FlatRowF32>({\n")
    for scheme in range(NARROW_SCHEMES):
        bounds = flat_f32["bounds"][scheme]
        f.write(f"  {{{scheme}, {cap}, {intervals}, {stored}, "
                f"{fmt(bounds[0])}, {fmt(bounds[1])}}},\n")
    f.write("});\n")


# ---------------------------------------------------------------------------
# The rational member over the FLOAT lane's uniform grid.
#
# The double lane's construction at the float lane's own grid, bar and
# arithmetic, and a second walk rather than a second reading of the first
# table for the reason the float uniform table beside it is: the double lane's
# pairs are stored in binary64 and are read on the double lane's intervals, and
# this lane's grid has a different width, a different interval count and
# different cells. A pair over one grid is not a pair over the other.
#
# What differs from the double walk, and what forces each difference:
#
# - The acceptance is the LANE's own bar and it is read on the SAME view the
#   figure is published from. The double member sets its acceptance at half the
#   bound its cells are held to, and the margin that buys is what lets it accept
#   on its fitting grid and publish a figure measured on another: the two views
#   agree to well inside the margin, so which one the acceptance was read on
#   cannot change the answer. This lane has no such margin. Half its cells'
#   bound (5e-8) is below what its own binary32 arithmetic delivers at the
#   values F_0 reaches on the first cells - the numerator's Horner sum, the
#   denominator's and the division round - and the first-holding entry on
#   interval 1 is 9.75e-8 on the fitting grid against 1.17e-7 on the cell's own
#   half-step view, a gap of a fifth. So the acceptance here is the lane's
#   published bar (F32_RAT_BOUND, which is the bar the lane's own rational
#   route states: "the target IS the lane's published bound, so the criterion
#   and the promise are the same sentence") and it is read on the points the
#   promise is read on - the cell's half-step grid, the lane's acceptance grid
#   and the accuracy gate's own cells. A pair is accepted exactly where it is
#   certified, which is what removes the gap rather than spending a margin on
#   it.
# - The mapped argument is the FLOAT locator's, not an exact mapping. The
#   kernel maps x to t in binary32 - one multiply by the width's reciprocal,
#   one truncation, one multiply-add - so the argument a cell is read at is a
#   rounding of the exact one, and a pair fitted on the exact mapping would be
#   fitted on arguments the kernel does not produce. f32_map is that mapping
#   transcribed, and every point here is taken through it.
# - The arguments are float-representable for the same reason: the table is
#   read at a binary32 x, so a fit made at an argument the format cannot hold
#   is a fit made between two of the readings it will be judged on.
# - The certification view carries the accuracy gate's own cells as well as
#   the cell's own grid. This lane's narrow rational route was found over the
#   bar at one cell of the gate's under the separate multiply-add route, on a
#   piece that held on its own fit grid - which is why that route reads the
#   gate's cells, and this member answers to the same gate over the same
#   domain, so it reads them too (f32_rat_read_points is that instrument).
#
# The bar is a raise and not a relaxation. An interval no ladder entry holds is
# a RuntimeError naming the interval and the bar, because a member emitted with
# a cell above the bar its route is certified against would publish a figure
# the lane does not stand behind.
FLAT_RAT_F32_ACCEPT = F32_RAT_BOUND
# The cell's fitting grid: the points the exchange solves on. The figure the
# member publishes is NOT read here - it is read on flat_rat_f32_view below,
# which is the acceptance's own view, because on this lane the two do not agree
# to within any margin the bar could leave.
FLAT_RAT_F32_GRID = 200
FLAT_RAT_F32_ITERS = FLAT_RAT_ITERS
# How many stored floats above the first ladder entry that holds the walk
# considers before it settles, on the reading flat_rat_pick gives.
FLAT_RAT_F32_WINDOW = FLAT_RAT_WINDOW
# The certification's working precision, the double member's: eight orders
# below the figure being measured and below the lane's own format by far more.
FLAT_RAT_F32_CERT_DPS = FLAT_RAT_CERT_DPS
# The pairs the search visits, in the double walk's order: by the count they
# store, and within one count the smaller denominator first. No entry has a
# zero denominator, for the reason the double ladder's last comment gives: a
# pair whose denominator held only its constant term would be the Chebyshev
# member's polynomial read under the rational route's name.
FLAT_RAT_F32_LADDER = FLAT_RAT_LADDER


def r32_below(v):
    """The float immediately below `v`, for the top of a cell's open end."""
    bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]

    return struct.unpack("<f", struct.pack("<I", bits - 1))[0]


def flat_rat_f32_cell(iv, grid):
    """A cell's float endpoints: a = iv * width and b = a + width.

    Spelled as the emitted constants are, so the cell a pair was fitted on is
    the cell the locator reaches: kFlatPerUnitF32 is the width's reciprocal
    exactly, so iv * width is the interval's own left end and not a rounding
    of it.
    """
    width = r32(float(grid["width"]))
    a = r32(float(iv) * float(grid["width"]))

    return a, r32(a + width)


@lru_cache(maxsize=None)
def flat_rat_f32_fit_points(n, iv, width, intervals):
    """The grid a cell's pair is solved on: float arguments across the cell and
    the mapped argument each, with the reference.

    The top of the cell is the float just below b rather than b itself, as the
    double walk's is: b belongs to the interval the locator puts it in, and a
    fit that reached it would be fitted one argument outside the cell it is
    stored for.

    Cached per (order, interval): the walk asks for the same cell at the same
    order once per ladder entry it visits, and rebuilding the reference at 40
    digits each time would be most of the run.
    """
    grid = {"width": mpf(width), "intervals": intervals}
    a, b = flat_rat_f32_cell(iv, grid)
    xs, ts = [], []
    for i in range(FLAT_RAT_F32_GRID):
        x = r32(a + (b - a) * (i / FLAT_RAT_F32_GRID))
        xs.append(x)
        ts.append(mpf(repr(f32_map(a, b, x))))
    x = r32_below(b)
    xs.append(x)
    ts.append(mpf(repr(f32_map(a, b, x))))

    return ts, xs, f32_rat_reference(n, xs)


def flat_rat_f32_delivered(p, q, ts, fs):
    """The pair's delivered worst |F_n - fit|, in the lane's own arithmetic.

    Both multiply-add routes, because a build runs one of them and the lane
    publishes one figure: a pair accepted on one reading alone would be a pair
    whose bound was never measured against the arithmetic the other build runs
    (see rational_value_float). Returns [fused, separate].
    """
    worst = [mpf(0), mpf(0)]
    for t, f in zip(ts, fs):
        td = float(t)
        for route, fused in ((0, True), (1, False)):
            e = abs(mpf(rational_value_float(p, q, td, fused)) - f)
            if e > worst[route]:
                worst[route] = e

    return worst


@lru_cache(maxsize=None)
def flat_rat_f32_view(n, iv, width, intervals):
    """Every point a cell's pair is certified on, one point per argument.

    Three views of one cell: the fit grid offset half a step, so no point of
    the two coincides; the points f32_rat_read_points builds for the cell,
    which are the lane's own acceptance grid and the accuracy gate's cells; and
    the cell's left end, where the mapping is exact and t is -1. A cell and a
    view point that are the same float are one point and not two.
    """
    grid = {"width": mpf(width), "intervals": intervals}
    a, b = flat_rat_f32_cell(iv, grid)
    points = {}
    for i in range(FLAT_RAT_F32_GRID):
        x = r32(a + (b - a) * ((i + 0.5) / FLAT_RAT_F32_GRID))
        if a <= x < b:
            points[x] = None
    for (x, _t, _r, _w) in f32_rat_read_points(n, a, b, weighted=False):
        points.setdefault(x, None)
    points.setdefault(a, None)
    with mp.workdps(FLAT_RAT_F32_CERT_DPS):
        return [(x, mpf(repr(f32_map(a, b, x))), boys_ref(n, mpf(x)))
                for x in sorted(points)]


def flat_rat_f32_certify(n, iv, grid, p, q):
    """The pair read on the certification view, with the argument each worst was
    reached at. Returns the per-route worsts and the arguments."""
    worst = [mpf(0), mpf(0)]
    at = [None, None]
    for (_x, t, ref) in flat_rat_f32_view(n, iv, float(grid["width"]),
                                          grid["intervals"]):
        td = float(t)
        for route, fused in ((0, True), (1, False)):
            e = abs(mpf(rational_value_float(p, q, td, fused)) - ref)
            if e > worst[route]:
                worst[route], at[route] = e, td

    return worst, at


def flat_rat_f32_exchange(m, k, ts, fs):
    """The best pair the exchange returns for one cell at one ladder entry.

    Two starting grids and the delivered reading, for the reasons
    flat_rat_exchange gives: a starting set that collapses is not the entry
    having no solution, and of two pairs of equal residual the one a caller
    receives less from is the one kept.
    """
    best = None
    for stride in (1, 2):
        r = rat_remez(m, k, list(range(0, len(ts), stride)), ts, fs,
                      iters=FLAT_RAT_F32_ITERS)
        if r is None:
            continue
        _, p, q = r
        worst = flat_rat_f32_delivered(p, q, ts, fs)
        if best is None or max(worst) < max(best[2]):
            best = (p, q, worst)

    return best


def flat_rat_f32_try(n, iv, grid, m, k):
    """One ladder entry at one cell and order, accepted where it is certified.

    The exchange solves on the fitting grid; the acceptance is then read on
    flat_rat_f32_view, which is the view the figure is published from. An entry
    the exchange cannot solve, or whose certified reading is above the bar, is
    not this cell's - and is not a smaller reading of some other entry either,
    which is why the walk continues rather than stopping.

    Returns (p, q, certified_worsts, certified_arguments) or None.
    """
    ts, _xs, fs = flat_rat_f32_fit_points(n, iv, float(grid["width"]),
                                          grid["intervals"])
    r = flat_rat_f32_exchange(m, k, ts, fs)
    if r is None:
        return None
    p, q = r[0], r[1]
    w, at = flat_rat_f32_certify(n, iv, grid, p, q)
    if max(w) > FLAT_RAT_F32_ACCEPT:
        return None

    return p, q, w, at


def flat_rat_f32_pick(n, iv, grid, start):
    """The ladder entry that certifies least among those near the first that
    holds, on the reading flat_rat_pick gives: the entries at a cell set by the
    lane's rounding differ only by how their own coefficients round, so which
    one is taken decides which figure is published.

    The window is the double walk's, in stored coefficients above the first
    entry that holds: the member buys rounding luck with the smallest amount of
    storage that buys any, and the bound on the search follows from the ladder
    rather than from an iteration count.
    """
    entered = None
    best = None
    for index in range(start, len(FLAT_RAT_F32_LADDER)):
        m, k = FLAT_RAT_F32_LADDER[index]
        if entered is not None and (m + 1 + k) > entered + FLAT_RAT_F32_WINDOW:
            break
        r = flat_rat_f32_try(n, iv, grid, m, k)
        if r is None:
            continue
        if entered is None:
            entered = m + 1 + k
        if best is None or max(r[2]) < max(best[1][2]):
            best = (index, r)

    return best


def _flat_rat_f32_interval_job(spec):
    """One interval's pairs, in a worker process."""
    iv, grid = spec
    mp.dps = RAT_DPS  # the fit's working precision; a spawn does not carry it

    return flat_rat_f32_interval(iv, grid)


def flat_rat_f32_interval(iv, grid):
    """One interval of the float grid: its pair, its orders' pairs, its figure.

    The interval's pair is chosen on its own first order - the largest of the
    interval's orders in absolute value, since |F_n| <= F_0 pointwise - and
    then held to every order of the interval, because one pair serves all of
    them. An order that fails the pair the interval was chosen at moves the
    whole interval one ladder entry up and the walk is made again, which is the
    double walk's rule and is what keeps the layout's one-pair-per-interval
    from becoming a per-order pair reported as the interval's.
    """
    index = 0
    while True:
        if index >= len(FLAT_RAT_F32_LADDER):
            raise RuntimeError(
                f"interval {iv} of the float uniform grid has no rational pair on "
                f"the ladder: every entry up to {FLAT_RAT_F32_LADDER[-1]} fails "
                f"either the exchange or the {mp.nstr(FLAT_RAT_F32_ACCEPT, 3)} its "
                f"cells are accepted under, so no member of the route can be "
                f"emitted for it")
        chosen = flat_rat_f32_pick(0, iv, grid, index)
        if chosen is None:
            raise RuntimeError(
                f"interval {iv} of the float uniform grid has no rational pair at "
                f"or above {FLAT_RAT_F32_LADDER[index]} on the ladder")
        index, (p0, q0, w0, w_at0) = chosen
        m, k = FLAT_RAT_F32_LADDER[index]
        pairs = []
        worst = [mpf(0), mpf(0)]
        at = [None, None]
        failed = None
        for n in range(MAX_ORDER + 1):
            if n == 0:
                p, q, w, w_at = p0, q0, w0, w_at0
            else:
                r = flat_rat_f32_try(n, iv, grid, m, k)
                if r is None:
                    failed = n
                    break
                p, q, w, w_at = r
            for route in (0, 1):
                if w[route] > worst[route]:
                    worst[route], at[route] = w[route], (n, w_at[route])
            pairs.append((n, p, q))
        if failed is None:
            return {"interval": iv, "m": m, "k": k, "pairs": pairs, "worst": worst,
                    "at": at}
        index += 1


def flat_rat_f32_table(grid):
    """The rational member over the float grid's cells, with its measured figure.

    The grid travels in rather than being derived here, because the member and
    the Chebyshev table beside it are one partition read through one locator:
    the grid the float uniform table settled on is the grid this member's
    intervals are, and a member fitted on the placement law's answer instead
    would be a second grid under the first one's name.
    """
    jobs = [(iv, grid) for iv in range(grid["intervals"])]
    with multiprocessing.get_context("spawn").Pool(
            min(len(jobs), NARROW_WORKERS)) as pool:
        out = pool.map(_flat_rat_f32_interval_job, jobs)
    worst = [mpf(0), mpf(0)]
    for row in out:
        for route in (0, 1):
            if row["worst"][route] > worst[route]:
                worst[route] = row["worst"][route]
    for route in (0, 1):
        if worst[route] > FLAT_RAT_F32_ACCEPT:
            raise RuntimeError(
                f"the float lane's rational member over the uniform grid delivers "
                f"{float(worst[route]):.6e} in "
                f"{'fused' if route == 0 else 'separate'} multiply-adds, above the "
                f"{mp.nstr(FLAT_RAT_F32_ACCEPT, 3)} its route is certified against")
    return {"grid": grid, "intervals": out, "worst": worst,
            "stored": sum((MAX_ORDER + 1) * (r["m"] + 1 + r["k"]) for r in out)}


def flat_rat_f32_block_lines(frat):
    """The float lane's rational member over its uniform grid, as the header
    stores it.

    The double member's layout, names and read rule at this lane's suffix and
    width: interval-major with the order as the inner index, the offsets,
    stored counts and degree columns emitted because the intervals carry a pair
    of their own rather than the table one stride, and the coefficients stored
    as binary32 literals beside the binary64 figures the row publishes.
    """
    grid = frat["grid"]
    intervals = grid["intervals"]
    rows = sorted(frat["intervals"], key=lambda r: r["interval"])
    if [r["interval"] for r in rows] != list(range(intervals)):
        raise RuntimeError("the float rational member's intervals are not the grid's")
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + (MAX_ORDER + 1) * (row["m"] + 1 + row["k"]))
    stored = offsets[-1]
    if stored != frat["stored"]:
        raise RuntimeError("the float rational member's stored count is not the sum "
                          "of its intervals' pairs")
    lines = [
        "// The rational minimax family over the float lane's uniform grid: one numerator/",
        "// denominator pair per interval at the grid's own cells, read at the mapped",
        "// argument the float locator builds. Row = p_0..p_m then q_1..q_k, q_0 = 1;",
        "// read at kFlatRatCoeffsF32[kFlatRatOffsetsF32[iv] + order * kFlatRatStoredF32[iv] + j].",
        f"inline constexpr int kFlatRatIntervalsF32 = {intervals};",
        f"inline constexpr int kFlatRatStoredTotalF32 = {stored};",
        "inline constexpr auto kFlatRatNumDegF32 = std::to_array<int>({",
    ]
    for i in range(0, len(rows), 12):
        lines.append("  " + ", ".join(str(r["m"]) for r in rows[i:i + 12]) + ",")
    lines.append("});")
    lines.append("inline constexpr auto kFlatRatDenDegF32 = std::to_array<int>({")
    for i in range(0, len(rows), 12):
        lines.append("  " + ", ".join(str(r["k"]) for r in rows[i:i + 12]) + ",")
    lines.append("});")
    lines.append("// The stored count of each interval's row, which is its pair plus the")
    lines.append("// denominator's held constant term:")
    lines.append("inline constexpr auto kFlatRatStoredF32 = std::to_array<int>({")
    for i in range(0, len(rows), 12):
        lines.append("  " + ", ".join(str(r["m"] + 1 + r["k"]) for r in rows[i:i + 12])
                     + ",")
    lines.append("});")
    lines.append("// Where each interval's block starts, and the end of the last one:")
    lines.append("inline constexpr auto kFlatRatOffsetsF32 = std::to_array<int>({")
    for i in range(0, len(offsets), 12):
        lines.append("  " + ", ".join(str(o) for o in offsets[i:i + 12]) + ",")
    lines.append("});")
    values = []
    for row in rows:
        if [p[0] for p in row["pairs"]] != list(range(MAX_ORDER + 1)):
            raise RuntimeError("the float rational member's rows do not cover every "
                              "order")
        for (_n, p, q) in row["pairs"]:
            values.extend(fmtf(float(c)) for c in p)
            values.extend(fmtf(float(c)) for c in q)
    lines.append("inline constexpr auto kFlatRatCoeffsF32 = std::to_array<float>({")
    for i in range(0, len(values), 6):
        lines.append("  " + ", ".join(values[i:i + 6]) + ",")
    lines.append("});")
    lines.append(f"static_assert(std::size(kFlatRatCoeffsF32) == {stored},\n"
                 "              \"the float rational member must hold every order of "
                 "every interval\");")
    lines.append("static_assert(std::size(kFlatRatNumDegF32) == kFlatRatIntervalsF32\n"
                 "                  && std::size(kFlatRatDenDegF32) == "
                 "kFlatRatIntervalsF32\n"
                 "                  && std::size(kFlatRatStoredF32) == "
                 "kFlatRatIntervalsF32\n"
                 "                  && std::size(kFlatRatOffsetsF32) == "
                 "kFlatRatIntervalsF32 + 1,\n"
                 "              \"the float rational member must carry one pair and one "
                 "offset per interval\");")
    lines.append("static_assert(kFlatRatOffsetsF32[kFlatRatIntervalsF32] == "
                 "kFlatRatStoredTotalF32,\n"
                 "              \"the offsets must reach the end of the float rational "
                 "member\");")
    lines.append("static_assert(kFlatRatIntervalsF32 == kFlatIntervalsF32,\n"
                 "              \"the float rational member is read at the float "
                 "Chebyshev member's own grid\");")
    lines.append("")
    lines.append("// What the float rational member delivers, at each multiply-add route,")
    lines.append("// against the bound the grid's cells are held to. Measured on each cell's own")
    lines.append("// certification view - the fit grid offset half a step, the acceptance grid")
    lines.append("// and the gate's cells - on the stored coefficients, as the acceptance read.")
    lines.append("struct FlatRatRowF32 { int stored; double fused, separate; };")
    lines.append("inline constexpr auto kFlatRatRowsF32 = "
                 "std::to_array<FlatRatRowF32>({")
    lines.append(f"  {{{stored}, {fmt(frat['worst'][0])}, {fmt(frat['worst'][1])}}},")
    lines.append("});")
    # The bar is the LANE's and not the grid's cell law, which is the one place
    # this member differs from the double one in what its row says. The double
    # member's cells are held to FLAT_CELL_BOUND and it delivers two orders
    # below that, so its bar is the grid's bound and the two readings agree.
    # This member cannot be: binary32 delivers ~1.1e-7 on the first cells, above
    # the 1e-7 cell law by construction and below the lane's 1.5e-7 contract, so
    # publishing the cell law here would publish a bar the member is over - a
    # documented claim it does not meet. What it is certified against is the
    # lane's own bar, which is the same target the lane's narrow rational route
    # is accepted at, and that is what this line states.
    lines.append(f"inline constexpr double kFlatRatBarF32 = "
                 f"{fmt(float(FLAT_RAT_F32_ACCEPT))};")

    return lines


def write_f32_namespace(f, float_orders, b_cheb_f32, rat_a_f32, rat_b_f32,
                        narrow_a_f32, narrow_b_f32, narrow_rat_f32, flat_f32,
                        flat_rat_f32):
    """The float lane's tables: the shipped lane, the rational route, the narrow
    partition under both routes, the uniform table.

    The region-A figures cover the lane's two routes over [0, kX0): the
    Chebyshev table is stored once and read by both schemes, so it carries one
    delivered figure for each and one stored count and piece list for the two.
    The rational route's selector takes over at kRatARouteLo and, per order,
    from that order's own end, where the lane stops reading the order from its
    own fit and documents the band's 3e-14 rather than the per-order 1e-15;
    below such an end the lane is documented at 1e-15, which these pieces do
    not hold. The rational pieces were accepted below the bar rather than at
    it, so the bar survives a certifying sweep on a different grid.
    """
    f.write("\nnamespace boys::detail::f32 {\n\n")
    all_coeffs = []
    all_mono = []
    meta = []
    for n in range(MAX_ORDER + 1):
        for (a, b, deg, cs, ms) in float_orders[n]:
            meta.append((n, a, b, deg, len(all_coeffs)))
            all_coeffs.extend(fmtf(c) for c in cs)
            all_mono.extend(fmtf(c) for c in ms)
    f.write("inline constexpr auto kCoeffs = std::to_array<float>({\n")
    for i in range(0, len(all_coeffs), 6):
        f.write("  " + ", ".join(all_coeffs[i:i + 6]) + ",\n")
    f.write("});\n\n")
    # The monomial form of the same fits, one stored coefficient per Chebyshev
    # coefficient: same pieces, same intervals, same degrees, same offsets, so
    # an evaluation scheme picks a table and not a shape. Converted from the
    # exact Chebyshev coefficients and rounded once, in the lane's own width.
    f.write("// The same fits in monomial form, ascending, for the Horner\n"
            "// evaluation scheme: same pieces, degrees and offsets as kCoeffs.\n")
    f.write("inline constexpr auto kMonoCoeffs = std::to_array<float>({\n")
    for i in range(0, len(all_mono), 6):
        f.write("  " + ", ".join(all_mono[i:i + 6]) + ",\n")
    f.write("});\n")
    f.write("static_assert(std::size(kMonoCoeffs) == std::size(kCoeffs),\n"
            "              \"the monomial table must parallel the Chebyshev table\");\n\n")
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
    deg, cs, ms = b_cheb_f32
    f.write("inline constexpr auto kBcoeffs = std::to_array<float>({"
            + ", ".join(fmtf(c) for c in cs) + "});\n")
    f.write("inline constexpr auto kMonoBcoeffs = std::to_array<float>({"
            + ", ".join(fmtf(c) for c in ms) + "});\n")
    f.write(f"inline constexpr int kBDeg = {deg};\n")
    f.write(deg_assert("kBDeg"))

    # The rational route. Its pieces are the family's own dyadic cover of each
    # order's interval rather than the Chebyshev table's breaks, because a
    # cover places its breaks where its own fit needs them; the table is
    # flattened the same way, with the per-order ranges in kRatAPieceStart.
    pieces = []
    stored = []
    for n in range(MAX_ORDER + 1):
        for pc in rat_a_f32["orders"][n]:
            pieces.append((n, pc))
            stored.extend(fmtf(v) for v in pc["p"])
            stored.extend(fmtf(v) for v in pc["q"])
    f.write("\n// The float lane's rational route: a weighted minimax fit of each order's\n"
            "// interval, evaluated as num / (1 + t*horner(q, t)) in the same mapped\n"
            "// argument the Chebyshev piece uses, read in the lane's own precision. An\n"
            "// offered alternative, not a replacement: the Chebyshev tables are the default.\n")
    f.write("inline constexpr auto kRatACoeffs = std::to_array<float>({\n")
    for i in range(0, len(stored), 6):
        f.write("  " + ", ".join(stored[i:i + 6]) + ",\n")
    f.write("});\n\n")
    f.write("struct RatPiece { float a, b; int numdeg; int dendeg; int offset; };\n")
    f.write("inline constexpr auto kRatAPieces = std::to_array<RatPiece>({\n")
    index = 0
    for (n, pc) in pieces:
        f.write(f"  {{{fmtf(mpf(pc['a']))}, {fmtf(mpf(pc['b']))}, {pc['m']}, {pc['k']}, "
                f"{index}}},  // F{n}\n")
        index += pc["count"]
    f.write("});\n")
    rstart = []
    index = 0
    for n in range(MAX_ORDER + 1):
        rstart.append(index)
        index += len(rat_a_f32["orders"][n])
    rstart.append(index)
    f.write("inline constexpr auto kRatAPieceStart = std::to_array<int>({"
            + ", ".join(map(str, rstart)) + "});\n")
    f.write("static_assert(std::size(kRatAPieceStart) == kMaxOrder + 2,\n"
            "              \"rational piece-start table must cover kMaxOrder\");\n")
    f.write("static_assert(std::size(kRatACoeffs) == "
            + str(sum(pc["count"] for _, pc in pieces)) + ",\n"
            "              \"the rational tables must account for every stored coefficient\");\n")

    f.write("\n// The rational region-B seed: p(t)/q(t) over the same interval and\n"
            "// in the same mapped argument as kBcoeffs above. q is stored as\n"
            "// q_1..q_k with its constant term held at 1.\n")
    f.write("inline constexpr auto kRatBnum = std::to_array<float>({"
            + ", ".join(fmtf(v) for v in rat_b_f32["p"]) + "});\n")
    f.write(f"inline constexpr int kRatBnumDeg = {rat_b_f32['m']};\n")
    f.write("inline constexpr auto kRatBden = std::to_array<float>({"
            + ", ".join(fmtf(v) for v in rat_b_f32["q"]) + "});\n")
    f.write(f"inline constexpr int kRatBdenDeg = {rat_b_f32['k']};\n")

    # The two routes as the generator measures them: the stored coefficients
    # each evaluates over region A, and the worst |F_n - fit| each reaches over
    # each order's own intervals at the lane's own precision, on the
    # coefficients as stored. Every piece is read at every argument of the
    # region's own grid and at every cell the accuracy gate sweeps in the
    # region - the arguments the published bar is read at - so the figures are
    # a reading of the same promise the gate holds them to. A swept maximum on
    # a finite grid, not a bound: the bar both routes are certified against is
    # kRegionAFitBar. Every figure here is the worse of the two multiply-add
    # routes, because a build runs one of them and a figure that covers one
    # understates the other.
    f.write("\n// The two region-A routes as the generator measures them, over [0, kX0), in\n"
            "// binary32 on the stored coefficients, the worse of both multiply-add routes:\n"
            "// the stored count each evaluates and the worst |F_n - fit| each reaches on each\n"
            "// order's intervals, at the acceptance grid and the gate's cells. Bar: kRegionAFitBar.\n"
            f"inline constexpr double kRegionAFitBar = {fmt(rat_a_f32['bound'])};\n"
            "inline constexpr int kRegionAFitChebStored = "
            + str(rat_a_f32["cheb_stored"]) + ";\n"
            "inline constexpr double kRegionAFitChebDelivered = "
            + fmt(rat_a_f32["cheb_delivered"]) + ";\n"
            "inline constexpr double kRegionAFitChebDeliveredHorner = "
            + fmt(rat_a_f32["cheb_delivered_horner"]) + ";\n"
            "inline constexpr int kRegionAFitRatStored = "
            + str(rat_a_f32["stored"]) + ";\n"
            "inline constexpr double kRegionAFitRatDelivered = "
            + fmt(rat_a_f32["delivered"]) + ";\n")

    # The region-B seeds, the two routes over one interval.
    f.write("\n// The two region-B seeds as the generator measures them: the stored count each\n"
            "// evaluates and the worst |F_0 - fit| each reaches over [kX0, kX1), on the\n"
            "// region-A points above, in the same arithmetic, the worse of both multiply-add\n"
            "// routes. A swept maximum, not a bound: the bar is kRegionBFitBar.\n"
            f"inline constexpr double kRegionBFitBar = {fmt(rat_a_f32['bound'])};\n"
            "inline constexpr int kRegionBFitChebStored = "
            + str(rat_b_f32["cheb_stored"]) + ";\n"
            "inline constexpr double kRegionBFitChebDelivered = "
            + fmt(rat_b_f32["cheb_delivered"]) + ";\n"
            "inline constexpr double kRegionBFitChebDeliveredHorner = "
            + fmt(rat_b_f32["cheb_delivered_horner"]) + ";\n"
            "inline constexpr int kRegionBFitRatStored = " + str(rat_b_f32["count"]) + ";\n"
            "inline constexpr double kRegionBFitRatDelivered = "
            + fmt(rat_b_f32["delivered"]) + ";\n")

    narrow_f32_block_lines(f, narrow_a_f32, narrow_b_f32)
    narrow_rat_f32_block_lines(f, narrow_rat_f32, narrow_b_f32)

    # The uniform table, beside every derived partition rather than in place of
    # any of them, as the double lane's is written beside its own.
    flat_f32_block_lines(f, flat_f32)

    # The rational member over that same grid, beside the Chebyshev member it
    # shares the grid with, as the double lane's is written beside its own.
    for line in flat_rat_f32_block_lines(flat_rat_f32):
        f.write(line + "\n")

    f.write("\n}  // namespace boys::detail::f32\n")


def write_header(path, double_orders, float_orders, b_cheb, b_cheb_f32, ext_cheb,
                 scheme_rows, rat_b, rat_a, rat_a_f32, rat_b_f32, narrow, narrow_a,
                 narrow_rat_a, narrow_rat_b, narrow_a_f32, narrow_b_f32, flat,
                 flat_f32, flat_rat, flat_rat_f32):
    with open(path, "w", newline="\n") as f:
        f.write("// Generated by tools/gen_boys_coefficients.py - DO NOT EDIT. Not API: the tables the\n")
        f.write("// entries are compiled from - split-Clenshaw Chebyshev fits of F_n(x): region A seeds\n")
        f.write("// weighted against downward-recursion gain (definitive check tests/boys_test.cpp).\n")
        f.write("/// \\cond\n")
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
        f.write("// The region-A rational route: a weighted minimax fit of each shipped piece,\n"
                "// evaluated as num / (1 + t*horner(q, t)) in the same mapped argument the\n"
                "// Chebyshev piece uses. An offered alternative, not a replacement: the\n"
                "// Chebyshev tables above are the default route and are unchanged.\n")
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
        f.write("// Where the route's selector takes over in region A: the lowest of the per-order\n"
                "// ends, the boundary the order is read from the band seed above and from its own\n"
                "// fit below. The tables still cover every piece from zero; naming the route\n"
                "// changes a value only from here, the domain the report claims.\n")
        f.write(f"inline constexpr double kRatARouteLo = {fmt(rat_a['lo'])};\n")
        f.write(f"inline constexpr double kRatARouteHi = {fmt(rat_a['hi'])};\n\n")
        deg, cs, mono = b_cheb
        f.write("inline constexpr auto kBcoeffs = std::to_array<double>({"
                + ", ".join(fmt(c) for c in cs) + "});\n")
        f.write("inline constexpr auto kMonoBcoeffs = std::to_array<double>({"
                + ", ".join(fmt(c) for c in mono) + "});\n")
        f.write(f"inline constexpr int kBDeg = {deg};\n")
        f.write(deg_assert("kBDeg"))
        f.write("\n")
        # The narrow partitions of the same regions, beside the shipped fits
        # rather than in place of them. See narrow_block_lines.
        for line in narrow_block_lines(narrow_a, narrow, narrow_rat_a, narrow_rat_b):
            f.write(line + "\n")
        f.write("\n")
        # The uniform table, beside every derived partition rather than in
        # place of any of them. See flat_block_lines.
        for line in flat_block_lines(flat):
            f.write(line + "\n")
        f.write("\n")
        # The rational family over that same grid, beside the Chebyshev member
        # rather than in place of it. See flat_rat_block_lines.
        for line in flat_rat_block_lines(flat_rat):
            f.write(line + "\n")
        f.write("\n")
        ext_deg, ext_cs, ext_mono = ext_cheb
        f.write("// The extended band: an F0 fit on [kExtendedBX0, kX0) evaluated by the same\n")
        f.write("// split Clenshaw; the upward recursion from it is certified per kmax tier, so\n")
        f.write("// an order n takes the extended seed exactly when x >= kTierThresholds[n] (the\n")
        f.write("// values the generator certifies, rounded up to the next double).\n")
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
        f.write(deg_assert("kExtendedBDeg"))
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
        f.write("// The two region-B routes as the generator measures them: the stored count\n")
        f.write("// each evaluates and the worst |F0 - fit| each reaches over [kX0, kX1], in the\n")
        f.write("// kernel's double arithmetic against the reference the fits are validated\n")
        f.write("// against. A swept maximum, not a bound: the bar is kRegionBFitBar.\n")
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
        # arithmetic, so the counts and the errors are one comparison. The
        # rational route's selector takes over at kRatARouteLo - the lowest of
        # region A's per-order ends - and per order from that order's own end,
        # which is where the lane stops reading the order from its own fit and
        # documents the band's 3e-14 rather than the per-order 1e-15. Below such
        # an end the lane is documented at 1e-15, which these pieces do not
        # hold. The rational pieces were accepted below the bar rather than at
        # it, so the bar survives a certifying sweep on a different grid.
        f.write("// The two region-A routes as the generator measures them, over the domain the\n"
                "// two tables cover - every order's pieces, from zero to kX0 - swept on a\n"
                "// stride-1 grid of each piece's interval, in the kernel's double arithmetic\n"
                "// against the reference. A swept maximum, not a bound: the bar is kRegionAFitBar.\n")
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
        f.write("// The evaluation schemes, one row per (scheme, stored fit): the degree and stored\n")
        f.write("// count the fit uses, and each scheme's bound against the 60-digit reference, per\n")
        f.write("// multiply-add route. scheme 0 = split Clenshaw, 1 = Horner; lane 0 = kCoeffs,\n")
        f.write("// 1 = kBcoeffs, 2 = kExtendedBcoeffs: the worst sweep, rounded up to a power of two.\n")
        f.write("struct SchemeRow { int scheme, lane, region, deg, stored;\n")
        f.write("                   double fused, separate; };\n")
        f.write("inline constexpr auto kSchemeRows = std::to_array<SchemeRow>({\n")
        for (scheme, lane, region, deg, stored, _fm, _sm, fb, sb) in scheme_rows:
            f.write(f"  {{{scheme}, {lane}, {region}, {deg}, {stored}, {fmt(fb)}, "
                    f"{fmt(sb)}}},\n")
        f.write("});\n")
        f.write("\n}  // namespace boys::detail\n")

        # Float lane.
        narrow_rat_f32 = fit_narrow_rational_f32(narrow_a_f32, narrow_b_f32)
        write_f32_namespace(f, float_orders, b_cheb_f32, rat_a_f32, rat_b_f32,
                        narrow_a_f32, narrow_b_f32, narrow_rat_f32, flat_f32,
                        flat_rat_f32)
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
        # per-kmax boundaries) plus interior points across the band; every
        # dispatch boundary this band has been cut at stays on the grid,
        # whichever way the edge moved, so a change of dispatch constant
        # cannot move the grid under a row it certifies. The generations are
        # pinned newest last.
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
        # The dedup key matters here: the log sweep's top point is 10**2 and
        # the extras' is 100, which the general pow makes 100 + 2^-97 rather
        # than equal to mpf("100"), so without this x = 100.0 would appear
        # twice for every order (2739 rows for 82 distinct arguments).
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
    pieces = [(n, a, b, deg, cs, ms)
              for n in range(MAX_ORDER + 1) for (a, b, deg, cs, ms) in double_orders[n]]
    delivs = run_jobs(_fit_delivered_job,
                      [(cs, ms, n, a, b, npts) for (n, a, b, _deg, cs, ms) in pieces])
    for (n, a, b, deg, cs, _ms), w in zip(pieces, delivs):
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
    parser.add_argument("--jobs", type=int, default=None,
                        help="worker processes for the independent fits (default: one "
                             "per CPU; 1 keeps the whole run in one process, which is "
                             "the same computation and the same tables)")
    parser.add_argument("--derive-partition", action="store_true",
                        help="derive a narrow partition from the proved truncation bound at "
                             "the quantum-chemistry target and print it, without fitting "
                             "anything (the design rule the bands are placed by)")
    parser.add_argument("--narrow-only", action="store_true",
                        help="fit only the narrow partition, in both regions, and write "
                             "its header block to --narrow-out, formatted, without fitting "
                             "the rest of the table (the block's byte-identity check "
                             "without the hours-long full generation)")
    parser.add_argument("--narrow-out", default="",
                        help="where --narrow-only writes the block (default a scratch "
                             "name beside the committed header; clang-format must find "
                             "the repo .clang-format, so keep it inside the tree)")
    parser.add_argument("--flat-rat-only", action="store_true",
                        help="fit only the rational member over the uniform grid and "
                             "print what each interval stores and delivers, without "
                             "fitting the rest of the table; writes its header block to "
                             "--flat-rat-out when that is given and nothing when it is "
                             "not")
    parser.add_argument("--flat-rat-out", default="",
                        help="where --flat-rat-only writes the member's header block "
                             "(default: nowhere, the figures are printed instead)")
    parser.add_argument("--f32-flat-only", action="store_true",
                        help="fit only the float lane's uniform table and print what "
                             "each order delivers, without fitting the rest of the "
                             "table; writes its header block to --f32-flat-out when "
                             "that is given and nothing when it is not")
    parser.add_argument("--f32-flat-out", default="",
                        help="where --f32-flat-only writes the table's header block "
                             "(default: nowhere, the figures are printed instead)")
    parser.add_argument("--f32-flat-rat-only", action="store_true",
                        help="fit only the float lane's rational member over its "
                             "uniform grid and print what each interval stores and "
                             "delivers, without fitting the rest of the table; writes "
                             "its header block to --f32-flat-rat-out when that is given "
                             "and nothing when it is not")
    parser.add_argument("--f32-flat-rat-out", default="",
                        help="where --f32-flat-rat-only writes the member's header "
                             "block (default: nowhere, the figures are printed "
                             "instead)")
    parser.add_argument("--derive-target", default="1e-14",
                        help="the target --derive-partition derives against (default the "
                             "quantum-chemistry 1e-14)")
    args = parser.parse_args()

    global JOBS
    JOBS = args.jobs

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
        narrow_rat_b = narrow_region_b_rational(narrow)
        narrow_rat_a = narrow_region_a_rational(narrow_a)
        narrow_a_f32 = narrow_region_a_f32()
        narrow_b_f32 = narrow_region_b_f32()
        narrow_out = args.narrow_out or (args.header + ".narrow-tmp.hpp")
        os.makedirs(os.path.dirname(narrow_out) or ".", exist_ok=True)
        with open(narrow_out, "w", newline="\n") as f:
            f.write("#include <array>\n#include <cstddef>\n\n"
                    "namespace boys::detail {\n\n")
            for line in narrow_block_lines(narrow_a, narrow, narrow_rat_a, narrow_rat_b):
                f.write(line + "\n")
            f.write("\n}  // namespace boys::detail\n\n")
            f.write("namespace boys::detail::f32 {\n\n")
            narrow_f32_block_lines(f, narrow_a_f32, narrow_b_f32)
            narrow_rat_f32_block_lines(f, fit_narrow_rational_f32(narrow_a_f32, narrow_b_f32),
                                       narrow_b_f32)
            f.write("\n}  // namespace boys::detail::f32\n")
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

    if args.flat_rat_only:
        # The rational member over the uniform grid on its own. Its fits are a
        # function of the grid alone - no derived partition, no scheme sweep and
        # no other rational route is read - so the member and its figure are
        # taken here without the hours the rest of the generation spends, and the
        # block's byte-identity is checked without it. flat_rat_block_lines is
        # shared with write_header, which is what keeps the two from drifting.
        # The interval table is printed rather than only the maximum, because a
        # maximum hides which interval set it and the pairs are per interval.
        grid = uniform_grid("double")
        print(f"fitting the rational member over the uniform grid (read cap "
              f"{grid['cap']}, width {mp.nstr(grid['width'], 6)} over "
              f"[0, {mp.nstr(grid['hi'], 6)}], {grid['intervals']} intervals, "
              f"accepted at {mp.nstr(FLAT_RAT_ACCEPT, 3)} delivered, "
              f"{len(FLAT_RAT_LADDER)} ladder entries, both multiply-add routes) ...")
        flat_rat = flat_rat_table()
        rows = sorted(flat_rat["intervals"], key=lambda r: r["interval"])
        print(f"  intervals {len(rows)}, stored {flat_rat['stored']} doubles, "
              f"distinct pairs "
              f"{sorted({(r['m'], r['k']) for r in rows})}")
        for row in rows:
            print(f"  iv {row['interval']:>3} [{mp.nstr(grid['width'] * row['interval'], 6)}, "
                  f"{mp.nstr(grid['width'] * (row['interval'] + 1), 6)}): pair "
                  f"{row['m']}/{row['k']}, {row['m'] + 1 + row['k']} stored, worst "
                  f"fused {row['worst'][0]:.6e} separate {row['worst'][1]:.6e}")
        print(f"  worst over the member: fused {flat_rat['worst'][0]:.6e} at "
              f"{[r['at'][0] for r in rows if r['worst'][0] == flat_rat['worst'][0]][0]}, "
              f"separate {flat_rat['worst'][1]:.6e} at "
              f"{[r['at'][1] for r in rows if r['worst'][1] == flat_rat['worst'][1]][0]}")
        print(f"  the grid's own bound {mp.nstr(FLAT_CELL_BOUND['double'], 3)}, "
              f"acceptance {mp.nstr(FLAT_RAT_ACCEPT, 3)}, published round-up "
              f"{scheme_bound(max(flat_rat['worst'])):.6e}")
        if args.flat_rat_out:
            os.makedirs(os.path.dirname(args.flat_rat_out) or ".", exist_ok=True)
            with open(args.flat_rat_out, "w", newline="\n") as f:
                f.write("#include <array>\n#include <cstddef>\n\n"
                        "namespace boys::detail {\n\n")
                for line in flat_rat_block_lines(flat_rat):
                    f.write(line + "\n")
                f.write("\n}  // namespace boys::detail\n")
            print(f"wrote {args.flat_rat_out}")
        return 0

    if args.f32_flat_only:
        # The float lane's uniform table on its own. A fixed grid and 33
        # independent walks: nothing else in the generation places it and nothing
        # else has to run to read what it delivers, so the table and its figure
        # are taken here without the hours the derived routes and the scheme
        # sweeps take. Every order's figure is printed rather than only the
        # maximum, because a maximum hides which order set it - and the orders of
        # this table are not all the same figure.
        g32 = uniform_grid("float")
        print(f"fitting the float lane's uniform table (widest row: read cap "
              f"{g32['cap']}, width {mp.nstr(g32['width'], 6)} over "
              f"[0, {mp.nstr(g32['hi'], 6)}], {g32['intervals']} intervals, degrees "
              f"{min(g32['degs'])}..{g32['cap']}, {F32_NARROW_GRID + 1} arguments per "
              f"interval, both multiply-add routes) ...")
        flat_f32 = flat_table_f32()
        g32 = flat_f32["grid"]
        print(f"  the float uniform table: {g32['intervals']} intervals of width "
              f"{mp.nstr(g32['width'], 6)} reaching {mp.nstr(g32['hi'], 8)}, degrees "
              f"{min(g32['degs'])}..{max(g32['degs'])}, "
              f"{sum(d + 1 for d in g32['degs']) * (MAX_ORDER + 1)} stored per table")
        print("per order, worst |F_n - fit| over the grid in the lane's own arithmetic:")
        for n in range(MAX_ORDER + 1):
            w = flat_f32["per_order"][n]
            print(f"  F{n:>2}: {max(w[0][0], w[0][1], w[1][0], w[1][1]):.6e}   "
                  f"clenshaw {w[0][0]:.6e} / {w[0][1]:.6e}   "
                  f"horner {w[1][0]:.6e} / {w[1][1]:.6e}  (fused / separate)")
        for scheme, name in enumerate(SCHEME_NAMES):
            for route, route_name in ((0, "fused"), (1, "separate")):
                order = max(range(MAX_ORDER + 1),
                            key=lambda n: flat_f32["per_order"][n][scheme][route])
                print(f"{name:14s} {route_name:8s} worst "
                      f"{flat_f32['worst'][scheme][route]:.6e} at F{order}, "
                      f"bound {flat_f32['bounds'][scheme][route]:.6e}")
        if args.f32_flat_out:
            os.makedirs(os.path.dirname(args.f32_flat_out) or ".", exist_ok=True)
            with open(args.f32_flat_out, "w", newline="\n") as f:
                f.write("#include <array>\n#include <cstddef>\n\n"
                        "namespace boys::detail::f32 {\n\n")
                flat_f32_block_lines(f, flat_f32)
                f.write("\n}  // namespace boys::detail::f32\n")
            print(f"wrote {args.f32_flat_out}")
        return 0

    if args.f32_flat_rat_only:
        # The rational member over the float lane's uniform grid on its own, the
        # double member's scoped run at this lane. Its fits are a function of the
        # grid and the lane's arithmetic alone - no derived partition and no
        # scheme sweep is read - so the member and its figure are taken here
        # without the hours the rest of the generation spends, and the block's
        # byte-identity is checked without it. The grid is the one the float
        # uniform table settles on and not the placement law's answer, so the
        # table is walked here too: flat_rat_f32_block_lines is shared with
        # write_header, which is what keeps the two from drifting.
        g32i = uniform_grid("float")
        print(f"walking the float lane's uniform grid (widest row: read cap "
              f"{g32i['cap']}, width {mp.nstr(g32i['width'], 6)} over "
              f"[0, {mp.nstr(g32i['hi'], 6)}], {g32i['intervals']} intervals, the "
              f"lane's own arithmetic) ...")
        g32 = flat_table_f32()["grid"]
        print(f"  the grid settled on: {g32['intervals']} intervals of width "
              f"{mp.nstr(g32['width'], 6)} reaching {mp.nstr(g32['hi'], 8)}, degrees "
              f"{min(g32['degs'])}..{max(g32['degs'])}")
        print(f"fitting the rational member over it (accepted at "
              f"{mp.nstr(FLAT_RAT_F32_ACCEPT, 3)} delivered, "
              f"{len(FLAT_RAT_F32_LADDER)} ladder entries, both multiply-add routes, "
              f"the gate's cells in the certification) ...")
        flat_rat_f32 = flat_rat_f32_table(g32)
        rows = sorted(flat_rat_f32["intervals"], key=lambda r: r["interval"])
        print(f"  intervals {len(rows)}, stored {flat_rat_f32['stored']} floats, "
              f"distinct pairs {sorted({(r['m'], r['k']) for r in rows})}")
        for row in rows:
            print(f"  iv {row['interval']:>3} [{mp.nstr(g32['width'] * row['interval'], 6)},"
                  f" {mp.nstr(g32['width'] * (row['interval'] + 1), 6)}): pair "
                  f"{row['m']}/{row['k']}, {row['m'] + 1 + row['k']} stored, worst "
                  f"fused {float(row['worst'][0]):.6e} separate "
                  f"{float(row['worst'][1]):.6e}")
        print(f"  worst over the member: fused "
              f"{float(flat_rat_f32['worst'][0]):.6e} at "
              f"{[r['at'][0] for r in rows if r['worst'][0] == flat_rat_f32['worst'][0]][0]},"
              f" separate {float(flat_rat_f32['worst'][1]):.6e} at "
              f"{[r['at'][1] for r in rows if r['worst'][1] == flat_rat_f32['worst'][1]][0]}")
        print(f"  the grid's own bound {mp.nstr(FLAT_CELL_BOUND['float'], 3)}, "
              f"acceptance {mp.nstr(FLAT_RAT_F32_ACCEPT, 3)}, published round-up "
              f"{scheme_bound(max(flat_rat_f32['worst'])):.6e}")
        if args.f32_flat_rat_out:
            os.makedirs(os.path.dirname(args.f32_flat_rat_out) or ".", exist_ok=True)
            with open(args.f32_flat_rat_out, "w", newline="\n") as f:
                f.write("#include <array>\n#include <cstddef>\n\n"
                        "namespace boys::detail::f32 {\n\n")
                for line in flat_rat_f32_block_lines(flat_rat_f32):
                    f.write(line + "\n")
                f.write("\n}  // namespace boys::detail::f32\n")
            print(f"wrote {args.f32_flat_rat_out}")
        return 0

    print("fitting double lane (weighted region-A seeds, tol 5e-14, deg<=18) ...")
    double_orders = run_jobs(_fit_order_job,
                             [(n, False, False) for n in range(MAX_ORDER + 1)])
    for n, pieces in enumerate(double_orders):
        print(f"  F{n:2d}: {len(pieces)} intervals, {sum(len(p[3]) for p in pieces)} coeffs")

    print("fitting float lane (weighted region-A seeds, tol 1e-7, deg<=10) ...")
    float_orders = run_jobs(_fit_order_job,
                            [(n, True, False) for n in range(MAX_ORDER + 1)])

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

    # The rational route over the narrow partition: the same family and the
    # same bars, fitted over the narrow intervals and accepted in the
    # arithmetic the kernel runs at both multiply-add routes.
    narrow_rat_b = narrow_region_b_rational(narrow)
    narrow_rat_a = narrow_region_a_rational(narrow_a)

    # The uniform table. Independent of every partition above - its grid is
    # equal-width and derived from its own bound, and no order is built from
    # another - so it is fitted here rather than anywhere the derived routes'
    # work would reach it.
    gm = uniform_grid("double")
    print(f"fitting the uniform table (widest row: read cap {gm['cap']}, width "
          f"{mp.nstr(gm['width'], 6)} over [0, {mp.nstr(gm['hi'], 6)}], "
          f"{gm['intervals']} intervals, degrees {min(gm['degs'])}..{gm['cap']}) ...")
    flat = flat_table()
    gm = flat["grid"]
    print(f"  the uniform table: {gm['intervals']} intervals of width "
          f"{mp.nstr(gm['width'], 6)} reaching {mp.nstr(gm['hi'], 8)}, degrees "
          f"{min(gm['degs'])}..{max(gm['degs'])}, "
          f"{sum(d + 1 for d in gm['degs']) * (MAX_ORDER + 1)} stored per table")
    # The rational route over that same grid: one numerator/denominator pair per
    # interval, chosen for that interval under the bound its cells are held to.
    # It is fitted here rather than anywhere the derived routes' work reaches,
    # for the reason the table above is: it is a fixed grid and not a walk.
    print(f"fitting the rational member over the uniform grid (accepted at "
          f"{mp.nstr(FLAT_RAT_ACCEPT, 3)} delivered, ladder of "
          f"{len(FLAT_RAT_LADDER)} pair(s), both multiply-add routes) ...")
    flat_rat = flat_rat_table()
    # key=, because the tuples tie on m + 1 + k: without it Python falls through to
    # comparing the interval dicts and raises. Ties keep their emitted order, which is
    # what the stored layout expects. Same shape as the sort at the rows above.
    pairs = sorted(((r["m"] + 1 + r["k"], r) for r in flat_rat["intervals"]),
                   key=lambda t: t[0])
    print(f"  the rational member: {flat_rat['grid']['intervals']} intervals, "
          f"{len({(r['m'], r['k']) for r in flat_rat['intervals']})} distinct pair(s), "
          f"stored counts {pairs[0][0]}..{pairs[-1][0]}, {flat_rat['stored']} stored")
    print(f"  worst delivered {flat_rat['worst'][0]:.6e} / {flat_rat['worst'][1]:.6e} "
          f"(fused / separate)")

    for scheme in range(NARROW_SCHEMES):
        print(f"  {SCHEME_NAMES[scheme]:14s} worst {flat['worst'][scheme][0]:.6e} / "
              f"{flat['worst'][scheme][1]:.6e} (fused / separate), bound "
              f"{flat['bounds'][scheme][0]:.6e}")

    # The same table for the float lane: the lane's own grid, derived from the
    # lane's own bound, format and read cap, and its figure read in the lane's
    # own arithmetic. Beside the double table rather than instead of it, as the
    # lane's other added fits are.
    g32 = uniform_grid("float")
    print(f"fitting the float lane's uniform table (widest row: read cap {g32['cap']}, "
          f"width {mp.nstr(g32['width'], 6)} over [0, {mp.nstr(g32['hi'], 6)}], "
          f"{g32['intervals']} intervals, degrees {min(g32['degs'])}..{g32['cap']}, "
          f"the lane's own arithmetic) ...")
    flat_f32 = flat_table_f32()
    g32 = flat_f32["grid"]
    print(f"  the float uniform table: {g32['intervals']} intervals of width "
          f"{mp.nstr(g32['width'], 6)} reaching {mp.nstr(g32['hi'], 8)}, degrees "
          f"{min(g32['degs'])}..{max(g32['degs'])}, "
          f"{sum(d + 1 for d in g32['degs']) * (MAX_ORDER + 1)} stored per table")
    for scheme in range(NARROW_SCHEMES):
        print(f"  {SCHEME_NAMES[scheme]:14s} worst {flat_f32['worst'][scheme][0]:.6e} / "
              f"{flat_f32['worst'][scheme][1]:.6e} (fused / separate), bound "
              f"{flat_f32['bounds'][scheme][0]:.6e}")

    # The rational member over that same grid: one numerator/denominator pair
    # per interval, chosen for that interval under the bound the grid's cells
    # are held to, in the lane's own arithmetic. It is fitted on the grid the
    # table above settled on rather than on the placement law's answer, so the
    # two members are one partition read through one locator.
    print(f"fitting the float lane's rational member over its uniform grid "
          f"(accepted at {mp.nstr(FLAT_RAT_F32_ACCEPT, 3)} delivered, ladder of "
          f"{len(FLAT_RAT_F32_LADDER)} pair(s), both multiply-add routes) ...")
    flat_rat_f32 = flat_rat_f32_table(g32)
    print(f"  the float rational member: {g32['intervals']} intervals, "
          f"{len({(r['m'], r['k']) for r in flat_rat_f32['intervals']})} distinct "
          f"pair(s), {flat_rat_f32['stored']} stored, worst delivered "
          f"{float(flat_rat_f32['worst'][0]):.6e} / "
          f"{float(flat_rat_f32['worst'][1]):.6e} (fused / separate), bound "
          f"{float(FLAT_CELL_BOUND['float']):.6e}")

    # The rational region-B route, over the interval the Chebyshev region-B fit
    # was just given, so the two are compared on the same interval against the
    # same reference.
    rat_b = fit_region_b_rational(b_cheb)

    # The rational region-A route, over the shipped pieces' own intervals, so
    # the two routes over region A are compared piece for piece.
    rat_a = fit_region_a_rational(double_orders)

    # The float lane's rational route, the double one's construction at that
    # lane's target and in that lane's width.
    rat_a_f32 = fit_region_a_rational_f32(float_orders)
    rat_b_f32 = fit_region_b_rational_f32()
    print(f"  float lane region B: rational {rat_b_f32['m']}/{rat_b_f32['k']} "
          f"{rat_b_f32['count']} stored, delivered {mp.nstr(rat_b_f32['delivered'], 6)}")

    # The float lane's own narrow partition: the lane's fitted regions at the
    # narrow degree, accepted in the lane's binary32 arithmetic at both routes.
    narrow_a_f32 = narrow_region_a_f32()
    narrow_b_f32 = narrow_region_b_f32()

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
        # Both scratch copies live beside the file they are compared against.
        # The header must: clang-format discovers the repo .clang-format
        # relative to the file's directory, and a foreign temp dir would
        # silently format the scratch copy with the LLVM default style, so
        # every check would false-positive on wrapping. The reference must for
        # a different reason: one fixed path in the system temp directory is
        # shared by every checkout on the machine, so two runs at once
        # overwrite each other's grid and report drift in one of the two files
        # they were pointed at.
        tmp_header = args.header + ".check-tmp.hpp"
        tmp_reference = args.reference + ".check-tmp.csv"
        try:
            write_header(tmp_header, double_orders, float_orders, b_cheb, b_cheb_f32,
                         (ext_deg, ext_cs, ext_mono), scheme_rows, rat_b, rat_a,
                         rat_a_f32, rat_b_f32, narrow, narrow_a,
                         narrow_rat_a, narrow_rat_b, narrow_a_f32, narrow_b_f32, flat,
                         flat_f32, flat_rat, flat_rat_f32)
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
                 (ext_deg, ext_cs, ext_mono), scheme_rows, rat_b, rat_a,
                 rat_a_f32, rat_b_f32, narrow, narrow_a,
                 narrow_rat_a, narrow_rat_b, narrow_a_f32, narrow_b_f32, flat,
                 flat_f32, flat_rat, flat_rat_f32)
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
