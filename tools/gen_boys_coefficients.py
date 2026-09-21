#!/usr/bin/env python3
"""Generate the Boys-function Chebyshev coefficient tables and test reference data.

Reproduces src/boys_coefficients.hpp (double and float lanes) and
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
    (VikhamarSandberg2025, eqs. 18-23), so the
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
    is VikhamarSandberg2025, Eqs. 25/13.
  - Float lane: same structure, tolerance 1e-7, degree cap 10.

The DCT normalization is the standard one: c_0 gets 1/(deg+1), all other
coefficients - including the top one - get 2/(deg+1). An earlier normalization
bug (halving c_deg too) made fits diverge - see the design doc. The committed
header must match this script's output byte for byte for a given tolerance
configuration.
"""
import argparse
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
# a-priori seed bound delta_0' = 1.0527e-15 (R_hat 1.0116e-15 forward
# rounding + tau 4.112e-17 truncation tail, no x-sampling) and hardcoded
# below as the dispatch constants; the fit tolerance is the same 5e-14
# class as the shipped region-B fit, so the delivered seed width keeps the
# recursion envelope <= 5e-14 from each per-kmax boundary up.
XNEW0 = mpf("1.0855252345349333")
EXTENDED_DEG_LADDER = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72, 96)

# The certified per-kmax boundaries of the extended band (kmax 4/8/16/32),
# as certified by the interval instrument under the band's range-uniform
# a-priori seed bound R_hat + tau = 1.0761680238880251877e-15 (R_hat the
# evaluation-rounding sum over the seed's coded roundings, tau the
# polynomial-definition term). tau is computed rather than typed:
# (u/2)*sum_j|c_j| for the stored-coefficient rounding plus
# (1 + Lambda_24)*Tail_proj for truncation and aliasing, with Lambda_24 =
# 3.0117926123493714563 the Chebyshev Lebesgue constant of the 25
# interpolation nodes. The bound covers the SEED's own definition and
# evaluation error, NOT the error a caller receives: the crossing condition
# charges the upward recursion and the asymptotic tail separately.
# The stored values ARE the kernel's dispatch constants: the next doubles
# above the certified crossings, so the dispatched region is a subset of the
# certified region. --check reproduces them exactly.
TIER_BOUNDARIES_CERTIFIED = [
    mpf("1.0855252345349333"),  # kmax 4: the band's left edge (the crossing clamps there;
                                # the true failure boundary is at or below the fit's
                                # lower edge - a conservative certified lower bound)
    mpf("2.0170701478602067"),  # kmax 8: the 1-ulp-exp certified crossing
    mpf("4.8998472055064735"),  # kmax 16: the 1-ulp-exp certified crossing
    mpf("10.785490619744019"),  # kmax 32: the 1-ulp-exp certified crossing
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


def seed_weight(n, x):
    """Downward-recursion amplification bound: max(1, x^n / prod(j+1/2))."""
    p = mpf(1)
    for j in range(n):
        p *= x / (j + mpf("0.5"))
    return max(mpf(1), p)


def fit_interval(n, a, b, tol, maxdeg, weighted, degs=None):
    """Returns (deg, [float coeffs]) or None; weighted=True fits region-A seeds.
    degs overrides the degree ladder (the extended band's fit walks its own)."""
    fmax = max(abs(boys_ref(n, x)) for x in grid(a, b, 8))
    if weighted:
        if max(fmax * seed_weight(n, x) for x in grid(a, b, 8)) < tol:
            return (0, [0.0])
    elif fmax < tol:
        return (0, [0.0])
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
            return (deg, cd)
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
            deg, cd = r
            pieces.append((float(a), float(b), deg, cd))
    pieces.sort(key=lambda p: p[0])
    return pieces


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
    = 1.0527e-15)."""
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
        deg, cd = r
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
        bdeg, bcs = b_cheb
        junction = abs(clenshaw_double(cd, float(X0), float(edge), float(X0))
                       - clenshaw_double(bcs, float(X0), float(X0), float(X1)))
        if junction > mpf("2e-14"):
            raise RuntimeError(f"extended band: kX0 junction check failed "
                               f"({junction} > 2e-14)")
    return deg, cd, crossings


def write_header(path, double_orders, float_orders, b_cheb, b_cheb_f32, ext_cheb):
    with open(path, "w", newline="\n") as f:
        f.write("// Generated by tools/gen_boys_coefficients.py - DO NOT EDIT.\n")
        f.write("// Piecewise Chebyshev (split Clenshaw) fits of F_n(x), region A seeds\n")
        f.write("// weighted against downward-recursion amplification; validated against a\n")
        f.write("// 30-digit mpmath reference (definitive check: tests/boys_test.cpp).\n")
        f.write("#pragma once\n#include <array>\n#include <cstddef>\n\n")
        f.write("namespace boys::detail {\n\n")
        f.write(f"inline constexpr int kMaxOrder = {MAX_ORDER};\n")
        f.write(f"inline constexpr double kX0 = {fmt(X0)};\n")
        f.write(f"inline constexpr double kX1 = {fmt(X1)};\n\n")

        all_coeffs = []
        meta = []
        for n in range(MAX_ORDER + 1):
            for (a, b, deg, cs) in double_orders[n]:
                meta.append((n, a, b, deg, len(all_coeffs)))
                all_coeffs.extend(fmt(c) for c in cs)
        f.write("inline constexpr auto kCoeffs = std::to_array<double>({\n")
        for i in range(0, len(all_coeffs), 6):
            f.write("  " + ", ".join(all_coeffs[i:i + 6]) + ",\n")
        f.write("});\n\n")
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
        deg, cs = b_cheb
        f.write("inline constexpr auto kBcoeffs = std::to_array<double>({"
                + ", ".join(fmt(c) for c in cs) + "});\n")
        f.write(f"inline constexpr int kBDeg = {deg};\n")
        f.write("\n")
        ext_deg, ext_cs = ext_cheb
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
        f.write("\n}  // namespace boys::detail\n")

        # Float lane.
        f.write("\nnamespace boys::detail::f32 {\n\n")
        all_coeffs = []
        meta = []
        for n in range(MAX_ORDER + 1):
            for (a, b, deg, cs) in float_orders[n]:
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
        deg, cs = b_cheb_f32
        f.write("inline constexpr auto kBcoeffs = std::to_array<float>({"
                + ", ".join(fmtf(c) for c in cs) + "});\n")
        f.write(f"inline constexpr int kBDeg = {deg};\n")
        f.write("\n}  // namespace boys::detail::f32\n")


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
        # superseded dispatch constants pin the vacated slices' edges (the
        # dispatch shift: [old, new) returns to the region-A path there).
        # A retired boundary is never dropped from the grid: each generation
        # of them stays pinned, newest set last.
        extras = [XNEW0] + [v for v in TIER_BOUNDARIES_CERTIFIED]
        extras += [mpf("1.5"), mpf("3"), mpf("5"), mpf("7"), mpf("9"), mpf("11")]
        extras += [mpf("1.857502623467682"), mpf("4.7030889427115925"),
                   mpf("10.655106119385133")]
        extras += [mpf("2.0136053436336927"), mpf("4.895982897243881"),
                   mpf("10.781772313649316")]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", default="src/boys_coefficients.hpp")
    parser.add_argument("--reference", default="tests/data/boys_reference.csv")
    parser.add_argument("--reference-only", action="store_true",
                        help="regenerate only the reference CSV (skip the fitting)")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed header/CSV are byte-identical to a "
                             "fresh generation (exits nonzero on drift)")
    args = parser.parse_args()

    if args.reference_only:
        os.makedirs(os.path.dirname(args.reference) or ".", exist_ok=True)
        write_reference(args.reference)
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

    print("fitting the extended band seed (F0 on [XNEW0, X0), tol 5e-14; "
          "the fixed-point loop) ...")
    ext_deg, ext_cs, ext_crossings = fit_extended_band(b_cheb=b_cheb)
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
                         (ext_deg, ext_cs))
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
                 (ext_deg, ext_cs))
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
