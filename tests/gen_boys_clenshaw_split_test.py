#!/usr/bin/env python3
"""Hold the generator's split-Clenshaw evaluator to an independent Chebyshev sum.

`tools/gen_boys_coefficients.py` carries `clenshaw_split_double` as a
transcription of the kernel's `ClenshawSplit`, and the region-B Chebyshev
route's delivered error is what that transcription measures. The kernel reads
both parities: its even part seeds at `c[2 * m]` with `m = deg / 2`, and its
odd part at `c[2 * mo - 1]` with `mo = (deg + 1) / 2`, so an odd degree's top
coefficient `c[deg]` is read. A transcription that seeded the odd part at
`c[2 * m - 1]` would be one odd term short at every odd degree and would still
return a plausible number, which is why the parity is pinned here against a
sum the recurrence does not spell.

    python tests/gen_boys_clenshaw_split_test.py

Odd and even degrees, 0 through 64 including 3, 5, 13, 25 and 63, at random
arguments in [-1, 1] and at -1, 0 and 1. The reference is the direct Chebyshev
sum, sum_k c_k T_k(t) with T_0 = 1, T_1 = t and T_{k+1} = 2 t T_k - T_{k-1},
at 60 significant digits. The tolerance is 1e-12 relative to the polynomial's
coefficient norm, sum_k |c_k|, which bounds |p(t)| on [-1, 1]; a polynomial can
sit on a near-zero at a sampled argument, where an error of a few ulp of that
norm is a large ratio to the value and says nothing about the evaluator. At
every point whose value is within two orders of the norm the strict ratio
|error| / |value| is held to 1e-12 as well.

Exit status: 0 when every point agrees, 1 with the disagreeing points named,
and 77 when mpmath is absent - the generator's own skip for a machine without
its pinned dependency.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "tools" / "gen_boys_coefficients.py"

DEGREES = (0, 1, 2, 3, 4, 5, 6, 13, 14, 25, 26, 63, 64)
TRIALS = 8
POINTS = 8
SEED = 20251025
TOLERANCE = "1e-12"
STRICT_FILTER = "1e-2"  # points with |value| at least 1 % of the term scale


def load_generator():
    """The generator by path, so the test reads the file it pins."""
    spec = importlib.util.spec_from_file_location("gen_boys_coefficients", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chebyshev_sum(mp, cs, t):
    """(value, norm): the direct Chebyshev sum and the coefficient 1-norm.

    T_0 = 1, T_1 = t, T_{k+1} = 2 t T_k - T_{k-1} at the working precision.
    This is the polynomial summed term by term, not the split recurrence the
    test pins; its exact zeros stay exact. The norm bounds |p(t)| on the
    interval and stands in for the evaluation's scale when the value is a
    near-zero.
    """
    tm = mp.mpf(t)
    terms = [mp.mpf(cs[0])]
    if len(cs) > 1:
        t_prev, t_cur = mp.mpf(1), tm
        terms.append(mp.mpf(cs[1]) * t_cur)
        for k in range(2, len(cs)):
            t_prev, t_cur = t_cur, 2 * tm * t_cur - t_prev
            terms.append(mp.mpf(cs[k]) * t_cur)
    return mp.fsum(terms), mp.fsum([abs(mp.mpf(c)) for c in cs])


def main() -> int:
    try:
        import mpmath as mp
    except ImportError:
        print("SKIP: mpmath is not installed (the generator's pinned dependency)")
        return 77

    generator = load_generator()
    # The generator's import sets the working precision to 30; the reference
    # below is read at 60 so its own residue is far under the double floor.
    mp.mp.dps = 60
    tolerance = mp.mpf(TOLERANCE)
    filter_scale = mp.mpf(STRICT_FILTER)
    rng = random.Random(SEED)

    findings: list[str] = []
    points = 0
    strict_points = 0
    worst_norm = mp.mpf(0)
    worst_strict = mp.mpf(0)
    worst_norm_at = None
    worst_strict_at = None

    for degree in DEGREES:
        spectra = (
            ("random", [rng.uniform(-1.0, 1.0) for _ in range(degree + 1)]),
            # Only the top coefficient: the term an odd-degree seed has to read.
            ("top-term", [0.0] * degree + [1.0]),
        )
        for name, coefficients in spectra:
            for _ in range(TRIALS):
                arguments = [-1.0, 0.0, 1.0]
                arguments += [rng.uniform(-1.0, 1.0) for _ in range(POINTS)]
                for t in arguments:
                    value, norm = chebyshev_sum(mp, coefficients, t)
                    if norm == 0:
                        continue
                    points += 1
                    got = mp.mpf(generator.clenshaw_split_double(coefficients, t))
                    error = abs(got - value)
                    ratio = error / norm
                    if ratio > worst_norm:
                        worst_norm, worst_norm_at = ratio, (degree, name, t)
                    if ratio > tolerance:
                        findings.append(
                            f"degree {degree} ({name} coefficients) at t={t!r}: "
                            f"|error|/norm = {mp.nstr(ratio, 6)} > {TOLERANCE}; "
                            f"got {mp.nstr(got, 20)}, expected {mp.nstr(value, 20)}"
                        )
                    if abs(value) >= filter_scale * norm:
                        strict_points += 1
                        strict = error / abs(value)
                        if strict > worst_strict:
                            worst_strict, worst_strict_at = strict, (degree, name, t)
                        if strict > tolerance:
                            findings.append(
                                f"degree {degree} ({name} coefficients) at t={t!r}: "
                                f"|error|/|value| = {mp.nstr(strict, 6)} > {TOLERANCE}; "
                                f"got {mp.nstr(got, 20)}, expected {mp.nstr(value, 20)}"
                            )

    if strict_points == 0:
        findings.append(
            "no sampled point reached the strict relative reading; the test is vacuous"
        )

    if findings:
        print(f"split-Clenshaw evaluator: {len(findings)} finding(s)", file=sys.stderr)
        for finding in findings[:20]:
            print(f"  - {finding}", file=sys.stderr)
        if len(findings) > 20:
            print(f"  - ... and {len(findings) - 20} more", file=sys.stderr)
        return 1

    print(
        f"split-Clenshaw evaluator: {len(DEGREES)} degrees, {points} points; "
        f"worst |error|/norm {mp.nstr(worst_norm, 4)} at degree "
        f"{worst_norm_at[0]} t={worst_norm_at[2]!r}; {strict_points} points at the "
        f"strict ratio, worst |error|/|value| {mp.nstr(worst_strict, 4)} at degree "
        f"{worst_strict_at[0]} t={worst_strict_at[2]!r}; all within {TOLERANCE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
