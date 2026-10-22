#!/usr/bin/env python3
"""Reduce the second sweep: the measured amplification and the model test.

The raw path file is written by sweep2.cpp.  Every path value is a double;
the reference is the same 60-digit gamma reference as the first sweep.  This
script applies the reference pivot to the canonical path (degree chosen by
largest resolved seed error), compares the result with the closed form
A_n(x) = (2n-1)!!/(2x)^n, measures the exact-perturbation transfer that the
C++ wrote, and scores the crossing model.

Reported, under results/:

    transfer_cells.csv     one row per argument and order
    transfer_summary.txt   deviation from the closed form, x-dependence,
                           the n^p q^n comparison, argmin and neighbour gaps
    prediction.csv         one row per band, metric and order at the scored margin
    prediction_scan.csv    the score over a pre-registered margin ladder
    band_summary.csv       the second sweep's own ship/product comparison
    per_order.csv          worst error per band and order on this sweep
    summary.txt            counts, criteria, reference validation, deviations

Two parameters of the original prediction are not recoverable from the
surviving records: the abstention threshold and the default winner.  This
reconstruction replaces both with the model's own accumulated-arithmetic
term: the model abstains when the two predicted bounds differ by less than
the arithmetic the recursed bound already carries, so no fitted threshold
enters.  The margin ladder is printed so the sensitivity is visible.

mpmath lives in the user site.  Do not run with `python -I`.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

if sys.flags.isolated:
    print("SKIP: isolated mode hides user-site mpmath; run without python -I")
    sys.exit(77)

try:
    from mpmath import mp, mpf
except ImportError:  # pragma: no cover
    print("SKIP: mpmath is not installed (`pip install mpmath`)")
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import boysprovenance  # noqa: E402

DEFAULT_TREE = boysprovenance.DEFAULT_TREE
NMAX = 32
SEED_ORDERS = [0, 2, 4, 8, 16, 24, 28, 32]
DEGREES = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
MARGINS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
EPS = 2.220446049250313e-16


def text(value):
    return mp.nstr(value, 17)


def csv_float(text):
    """float(), tolerating MSVC's spelling of a non-finite value.

    The harness prints through MSVC's %g, which spells a quiet NaN as
    ``-nan(ind)``.  Python's float() rejects that spelling, so normalise it
    before parsing.  Non-finite path rows are expected at the small arguments
    where the upward recurrence overflows, and the analysis marks them
    unmeasurable rather than failing on them.
    """
    return float(text.strip().replace("nan(ind)", "nan"))


def amplification(n, x):
    if n == 0:
        return mpf(1)
    odd = math.factorial(2 * n) // (2**n * math.factorial(n))
    return mpf(odd) / (2 * x) ** n


def amplification_ratio(n, l, x):
    """The gain from step l's rounding to order n: A_n(x) / A_l(x)."""
    odd_n = math.factorial(2 * n) // (2**n * math.factorial(n))
    odd_l = math.factorial(2 * l) // (2**l * math.factorial(l))
    return mpf(odd_n) / mpf(odd_l) / (2 * x) ** (n - l)


def recurrence_arithmetic(n, x, values):
    """The paper's accumulated-arithmetic charge, absolute.

    Three roundings at 2^-53 per upward step, each carried to order n by the
    recurrence's own gain A_n / A_l.  The step's output F_l sets the size.
    """
    total = mpf(0)
    for l in range(1, n + 1):
        total += 3 * mpf(2) ** -53 * values[l] * amplification_ratio(n, l, x)
    return total


def amplification_gamma(n, x):
    return mp.gamma(n + mpf("0.5")) / (mp.sqrt(mp.pi) * mpf(x) ** n)


def read_reference(path):
    reference = {}
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

        if header[:2] != ["band", "x"]:
            raise SystemExit("unexpected reference header: %r" % (header[:2],))

        for row in reader:
            reference[(row[0], float(row[1]))] = [mpf(value) for value in row[2:]]
    return reference


def read_cells(path):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    row["band"],
                    row["x"],
                    float(row["x"]),
                    int(row["n"]),
                    float(row["recursed"]),
                    float(row["direct"]),
                )
            )
    return rows


def read_grid(path):
    counts = {"A1": 0, "A2": 0}
    tags = {"G": 0, "V": 0, "other": 0}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            band = row["band"]
            counts[band] = counts.get(band, 0) + 1
            for tag in row.get("sets", "").split("+"):
                if tag in ("G", "V"):
                    tags[tag] += 1
                elif tag:
                    tags["other"] += 1
    return counts, tags


def read_admission(path):
    directions = {}
    totals = {}
    mismatches = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            directions[(row["band"], float(row["x"]))] = row["direction"]
            entry = totals.setdefault(row["direction"], [0, 0, 0])
            entry[0] += 1
            entry[1] += int(row["checked"])
            entry[2] += int(row["matched"])

            if row["checked"] != row["matched"]:
                mismatches.append((row["band"], row["x"], row["direction"]))
    return directions, totals, mismatches


def update(store, key, value, x_text):
    current = store.get(key)

    if current is None or value > current[0]:
        store[key] = (value, x_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boys-tree", default=os.environ.get("BOYS_TREE", DEFAULT_TREE))
    parser.add_argument("--tree", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--validation", default=None)
    parser.add_argument("--margin", type=float, default=1.0, help="abstention threshold as a multiple of the recursed arithmetic term")
    parser.add_argument("--saturation", type=float, default=1e300)
    parser.add_argument("--resolve-gap", type=float, default=1e-10)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--ulp-sample", type=int, default=97)
    parser.add_argument("--start-sample", type=int, default=16)
    args = parser.parse_args()

    tree = Path(args.tree).resolve()
    reference_path = tree / "data/reference.csv"
    provenance_path = Path(args.provenance) if args.provenance else tree / "results/provenance.json"

    if provenance_path.is_file():
        with open(provenance_path, encoding="utf-8") as handle:
            constants = json.load(handle)["constants"]
    else:
        constants = boysprovenance.parse_constants(Path(args.boys_tree).resolve())

    mp.dps = 70
    reference = read_reference(reference_path)
    cells = read_cells(tree / "results/cells.csv")
    directions, admission_totals, admission_mismatches = read_admission(
        tree / "results/admission.csv"
    )

    saturation = mpf(args.saturation)

    thresholds = [float(value) for value in constants["kTierThresholds"]]
    if len(thresholds) <= NMAX:
        raise SystemExit("the tree's threshold table stops before order %d" % NMAX)

    # The model's population: the policy's fit answers region A from
    # kTierThresholds[0] upward (boys_fit_chebyshev.hpp, kRegionAFitsFrom).
    # Below it the shipped entry seeds at the top order and recurses downward.
    model_from = thresholds[0]

    # ---- the second sweep's own two-evaluation comparison -----------------
    ship_abs = {}
    ship_rel = {}
    prod_abs = {}
    prod_rel = {}
    band_ship = {"A1": {}, "A2": {}}
    band_prod = {"A1": {}, "A2": {}}
    error0 = {}
    winner = {}

    for band, x_text, x, n, recursed, direct in cells:
        values = reference[(band, x)]
        expected = values[n]
        rec_error = abs(mpf(recursed) - expected)
        dir_error = abs(mpf(direct) - expected)
        rec_relative = rec_error / expected if expected != 0 else mpf("inf")
        dir_relative = dir_error / expected if expected != 0 else mpf("inf")

        update(ship_abs, (band, n), rec_error, x_text)
        update(prod_abs, (band, n), dir_error, x_text)
        update(ship_rel, (band, n), rec_relative, x_text)
        update(prod_rel, (band, n), dir_relative, x_text)

        for metric, value_now in (("absolute", rec_error), ("relative", rec_relative)):
            entry = band_ship[band].get(metric)
            if entry is None or value_now > entry[0]:
                band_ship[band][metric] = (value_now, n, x_text)

        for metric, value_now in (("absolute", dir_error), ("relative", dir_relative)):
            entry = band_prod[band].get(metric)
            if entry is None or value_now > entry[0]:
                band_prod[band][metric] = (value_now, n, x_text)

        if n == 0:
            error0[(band, x)] = rec_error

    for band in ("A1", "A2"):
        for n in range(NMAX + 1):
            for metric, ship_store, prod_store in (
                ("absolute", ship_abs, prod_abs),
                ("relative", ship_rel, prod_rel),
            ):
                ship_value = ship_store[(band, n)][0]
                prod_value = prod_store[(band, n)][0]
                if ship_value < prod_value:
                    state = "ship"
                elif prod_value < ship_value:
                    state = "product"
                else:
                    state = "tie"
                winner[(band, metric, n)] = state

    # ---- the prediction's band-level inputs -------------------------------
    #
    # The model of eq. (model-rec) is written for the band's upward seed.
    # AllOrdersBody reads order 0 from that seed for every argument at or above
    # the fit's own region-A start, and order n >= 1 from it exactly when
    # x >= kTierThresholds[n]; the orders above that prefix are the per-order
    # stored fits, which the direct entry reads as well (boys_impl.hpp).  The
    # population is therefore the arguments at or above kTierThresholds[0].
    # Below it the entry seeds at order 32 and recurses downward, so its
    # order-0 value is not a seed error and A_n does not describe the
    # propagation; those arguments are outside the model.  At an unserved
    # order inside the population both entries read the same fit, so both
    # predicted errors are that fit's error and the model abstains.
    predicted_rec_abs = {"A1": [mpf(0) for _ in range(NMAX + 1)], "A2": [mpf(0) for _ in range(NMAX + 1)]}
    predicted_rec_rel = {"A1": [mpf(0) for _ in range(NMAX + 1)], "A2": [mpf(0) for _ in range(NMAX + 1)]}
    predicted_dir_abs = {"A1": [mpf(0) for _ in range(NMAX + 1)], "A2": [mpf(0) for _ in range(NMAX + 1)]}
    predicted_dir_rel = {"A1": [mpf(0) for _ in range(NMAX + 1)], "A2": [mpf(0) for _ in range(NMAX + 1)]}
    predicted_arith_abs = {"A1": [mpf(0) for _ in range(NMAX + 1)], "A2": [mpf(0) for _ in range(NMAX + 1)]}
    predicted_arith_rel = {"A1": [mpf(0) for _ in range(NMAX + 1)], "A2": [mpf(0) for _ in range(NMAX + 1)]}
    model_arguments = {"A1": 0, "A2": 0}
    model_seen = set()

    for band, x_text, x, n, recursed, direct in cells:
        if x < model_from:
            # x = 0 (the exact branch) and every downward-seeded argument.
            continue
        if (band, x) not in model_seen:
            model_seen.add((band, x))
            model_arguments[band] += 1

        seed = error0.get((band, x))
        if seed is None:
            continue
        expected = reference[(band, x)][n]
        dir_abs = abs(mpf(direct) - expected)
        dir_rel = dir_abs / expected if expected != 0 else mpf("inf")

        if n == 0 or x >= thresholds[n]:
            arith_abs = recurrence_arithmetic(n, mpf(x), reference[(band, x)]) if n else mpf(0)
            rec_abs = seed * amplification(n, mpf(x)) + arith_abs
            rec_rel = rec_abs / expected if expected != 0 else mpf("inf")
            arith_rel = arith_abs / expected if expected != 0 else mpf("inf")
        else:
            arith_abs = mpf(0)
            arith_rel = mpf(0)
            rec_abs = dir_abs
            rec_rel = dir_rel

        if rec_abs > predicted_rec_abs[band][n]:
            predicted_rec_abs[band][n] = rec_abs
        if rec_rel > predicted_rec_rel[band][n]:
            predicted_rec_rel[band][n] = rec_rel
        if dir_abs > predicted_dir_abs[band][n]:
            predicted_dir_abs[band][n] = dir_abs
        if dir_rel > predicted_dir_rel[band][n]:
            predicted_dir_rel[band][n] = dir_rel
        if arith_abs > predicted_arith_abs[band][n]:
            predicted_arith_abs[band][n] = arith_abs
        if arith_rel > predicted_arith_rel[band][n]:
            predicted_arith_rel[band][n] = arith_rel

    def prediction_rows(band, metric, margin):
        rows = []
        if metric == "absolute":
            ship = predicted_rec_abs[band]
            product = predicted_dir_abs[band]
            sigma = predicted_arith_abs[band]
        else:
            ship = predicted_rec_rel[band]
            product = predicted_dir_rel[band]
            sigma = predicted_arith_rel[band]

        for n in range(NMAX + 1):
            ps = ship[n]
            pd = product[n]
            gap = abs(ps - pd)
            if gap <= mpf(margin) * sigma[n]:
                state = "abstain"
                choice = ""
            else:
                state = "ship" if ps < pd else "product"
                choice = state
            measured = winner[(band, metric, n)]
            right = choice == measured if state != "abstain" else False
            rows.append((n, ps, pd, sigma[n], state, measured, choice, right))
        return rows

    def score(rows):
        commits = sum(1 for row in rows if row[4] != "abstain")
        abstains = len(rows) - commits
        right_committed = sum(1 for row in rows if row[4] != "abstain" and row[7])
        return commits, right_committed, abstains

    # ---- the paths --------------------------------------------------------
    predicted_amp = {}

    def amp(n, x):
        key = (n, x)
        if key not in predicted_amp:
            predicted_amp[key] = amplification(n, mpf(x))
        return predicted_amp[key]

    canonical = {}
    seed_cross = {}
    ulp_worst = None
    ulp_worst_where = None
    ulp_cells = 0
    cross_worst = None
    cross_worst_where = None
    cross_cells = 0
    row_index = 0
    argument_index = -1
    current_key = None
    arguments_seen = set()

    transfer_cells = []
    transfer_form_worst = {band: None for band in ("A1", "A2")}
    transfer_form_where = {band: None for band in ("A1", "A2")}
    transfer_counts = {"A1": 0, "A2": 0}
    measurable_counts = {"A1": 0, "A2": 0}
    x_dependence = {"A1": [], "A2": []}
    fit_data = []
    derived_residuals = []
    gamma_form_worst = None
    gamma_form_where = None
    form_pair_worst = None
    form_pair_where = None
    curves = {band: [] for band in ("A1", "A2")}
    curve_text = {}

    def flush(band, x, x_text):
        nonlocal gamma_form_worst, gamma_form_where
        nonlocal form_pair_worst, form_pair_where
        values = canonical.get(0)
        if not values:
            return

        refs = reference[(band, x)]
        best_degree = None
        best_error = mpf(0)

        for degree in DEGREES:
            path = values.get(degree)
            if not path or 0 not in path:
                continue
            seed_error = abs(mpf(path[0][0]) - refs[0])
            if seed_error > best_error:
                best_error = seed_error
                best_degree = degree

        if best_degree is None or best_error == 0:
            return

        path = values[best_degree]
        transfers = [None] * (NMAX + 1)

        for n in range(NMAX + 1):
            if n not in path:
                continue
            value = mpf(path[n][0])
            transfer = (value - refs[n]) / (mpf(path[0][0]) - refs[0])
            measurable = (
                x > 0.0
                and mp.isfinite(transfer)
                and abs(transfer) <= saturation
            )
            transfers[n] = transfer if measurable else None
            transfer_counts[band] += 1
            if measurable:
                measurable_counts[band] += 1

            transfer_cells.append(
                (
                    band,
                    x_text,
                    n,
                    text(transfer) if measurable else "nan",
                    best_degree,
                    text(mpf(path[0][0]) - refs[0]),
                    "1" if measurable else "0",
                    "%.17g" % path[n][1],
                )
            )

            if not measurable:
                continue

            expected = amp(n, x)
            if expected != 0:
                deviation = abs(abs(transfer) - expected) / expected
                current = transfer_form_worst[band]
                if current is None or deviation > current:
                    transfer_form_worst[band] = deviation
                    transfer_form_where[band] = "n=%d x=%s" % (n, x_text)

            gamma_value = amplification_gamma(n, mpf(x))
            if gamma_value != 0:
                gamma_deviation = abs(abs(transfer) - gamma_value) / gamma_value
                if gamma_form_worst is None or gamma_deviation > gamma_form_worst:
                    gamma_form_worst = gamma_deviation
                    gamma_form_where = "%s n=%d x=%s" % (band, n, x_text)
                pair_deviation = abs(expected - gamma_value) / expected if expected != 0 else mpf("inf")
                if form_pair_worst is None or pair_deviation > form_pair_worst:
                    form_pair_worst = pair_deviation
                    form_pair_where = "%s n=%d x=%s" % (band, n, x_text)

            if transfer > 0 and expected > 0:
                residual = mp.log(transfer) - mp.log(expected)
                x_dependence[band].append(residual)
                if n >= 1:
                    derived_residuals.append(float(residual))

            if n >= 1 and transfer > 0:
                fit_data.append((n, float(mp.log(transfer))))

        curves[band].append((x, x_text, transfers))

    with open(tree / "results/paths.csv", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            row_index += 1
            band = row["band"]
            x_text = row["x"]
            x = float(x_text)
            state = row["route"]

            key = (band, x)
            if key != current_key:
                if current_key is not None:
                    flush(current_key[0], current_key[1], curve_text[current_key])
                current_key = key
                argument_index += 1
                arguments_seen.add(key)
                canonical = {}
                seed_cross = {}
                curve_text[key] = x_text

            if state == "seed" or x == 0.0:
                continue

            start = int(row["start"])
            degree = int(row["degree"])
            n = int(row["n"])
            value = csv_float(row["value"])
            pert_delta = csv_float(row["pert_delta"])
            seed_delta = csv_float(row["seed_delta"])
            transfer_ulp = csv_float(row["transfer_ulp"])

            if start == 0:
                # The reference pivot uses the order-0 seed and the upward
                # path; no other start is needed for the band curves.
                canonical.setdefault(start, {}).setdefault(degree, {})[n] = (value, transfer_ulp)

            if args.ulp_sample and row_index % args.ulp_sample == 0 and n != start:
                expected = amp(n, x) / amp(start, x)
                resolved = abs(pert_delta) > 16.0 * EPS * abs(value)
                if expected > 0 and resolved and math.isfinite(transfer_ulp):
                    deviation = abs(mpf(transfer_ulp) - expected) / expected
                    ulp_cells += 1
                    if ulp_worst is None or deviation > ulp_worst:
                        ulp_worst = deviation
                        ulp_worst_where = "%s start=%d degree=%d n=%d x=%s" % (
                            band,
                            start,
                            degree,
                            n,
                            x_text,
                        )

            if (
                args.start_sample
                and start not in (0, 32)
                and argument_index % args.start_sample == 0
            ):
                refs = reference[(band, x)]
                if n == start:
                    seed_cross[(start, degree)] = mpf(value) - refs[start]
                else:
                    seed_error = seed_cross.get((start, degree))
                    if seed_error is not None and seed_error != 0:
                        transfer = (mpf(value) - refs[n]) / seed_error
                        expected = amp(n, x) / amp(start, x)
                        if expected > 0 and mp.isfinite(transfer):
                            deviation = abs(abs(transfer) - expected) / expected
                            cross_cells += 1
                            if cross_worst is None or deviation > cross_worst:
                                cross_worst = deviation
                                cross_worst_where = "%s start=%d degree=%d n=%d x=%s" % (
                                    band,
                                    start,
                                    degree,
                                    n,
                                    x_text,
                                )

    if current_key is not None:
        flush(current_key[0], current_key[1], curve_text[current_key])

    # ---- argmin, neighbour gaps, approximations ---------------------------
    argmin_counts = {"A1": 0, "A2": 0}
    argmin_matches = {"A1": 0, "A2": 0}
    wrong_reading_matches = {"A1": 0, "A2": 0}
    continuous_matches = {"A1": 0, "A2": 0}
    continuous_exceptions = []
    gaps = {"A1": [], "A2": []}
    tie_arguments = {"A1": [], "A2": []}
    unresolved = {"A1": 0, "A2": 0}

    def nearest(value):
        return math.floor(value + 0.5)

    for band in ("A1", "A2"):
        for x, x_text, transfers in curves.get(band, []):
            if any(transfers[n] is None for n in range(NMAX + 1)):
                unresolved[band] += 1
                continue
            magnitudes = [abs(transfers[n]) for n in range(NMAX + 1)]
            m = min(range(0, NMAX + 1), key=lambda n: magnitudes[n])
            nmin = min(NMAX, max(0, math.ceil(x - 0.5)))
            if magnitudes[m] == 0:
                unresolved[band] += 1
                continue
            neighbours = []
            if m > 0:
                neighbours.append(abs(magnitudes[m - 1] - magnitudes[m]) / magnitudes[m])
            if m < NMAX:
                neighbours.append(abs(magnitudes[m + 1] - magnitudes[m]) / magnitudes[m])
            gap = min(neighbours)
            tie_low = m > 0 and abs(magnitudes[m - 1] - magnitudes[m]) / magnitudes[m] <= args.tie_tolerance
            tie_high = m < NMAX and abs(magnitudes[m + 1] - magnitudes[m]) / magnitudes[m] <= args.tie_tolerance
            tie = tie_low or tie_high
            tied_orders = [m - 1, m] if tie_low else ([m, m + 1] if tie_high else [m])
            if gap <= args.resolve_gap and not tie:
                unresolved[band] += 1
                continue
            argmin_counts[band] += 1
            if m == nmin or (tie and nmin in tied_orders):
                argmin_matches[band] += 1
            if tie:
                tie_arguments[band].append(x_text)
            if nearest(x + 0.5) == nmin:
                wrong_reading_matches[band] += 1
            if nearest(x - 0.5) == nmin:
                continuous_matches[band] += 1
            else:
                continuous_exceptions.append("%s x=%s" % (band, x_text))
            gaps[band].append(gap)

    def gap_stats(band):
        values = sorted(gaps[band])
        if not values:
            return "none"
        median = values[len(values) // 2]
        return "median %.4e, maximum %.4e over %d arguments" % (median, values[-1], len(values))

    # ---- the n^p q^n comparison -------------------------------------------
    def least_squares_two(data):
        s11 = s12 = s22 = t1 = t2 = 0.0
        for n, y in data:
            ln = math.log(n)
            s11 += ln * ln
            s12 += ln * n
            s22 += float(n) * n
            t1 += ln * y
            t2 += n * y
        determinant = s11 * s22 - s12 * s12
        if determinant == 0.0:
            return None
        p = (t1 * s22 - t2 * s12) / determinant
        q = (s11 * t2 - s12 * t1) / determinant
        return p, q

    unfit = None
    if fit_data:
        solution = least_squares_two(fit_data)
        if solution is not None:
            p, q = solution
            residuals = [y - (p * math.log(n) + q * n) for n, y in fit_data]
            rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
            unfit = (max(abs(r) for r in residuals), rms, p, q, len(fit_data))

    derived_max = max((abs(r) for r in derived_residuals), default=0.0)
    derived_rms = (
        math.sqrt(sum(r * r for r in derived_residuals) / len(derived_residuals))
        if derived_residuals
        else 0.0
    )

    def residual_stats(band):
        values = x_dependence[band]
        if not values:
            return "none"
        rms = mp.sqrt(sum(value * value for value in values) / len(values))
        worst = max(abs(value) for value in values)
        return "rms %s, maximum %s over %d cells" % (text(rms), text(worst), len(values))

    # ---- the bracket ------------------------------------------------------
    # The amplified seed term against the measured recursed error, upward
    # only: below kTierThresholds[0] the entry seeds at the top order and
    # recurses downward, and the model of eq. (model-rec) does not describe
    # that path.  The full bound adds the modelled arithmetic term.
    extremes = {"upward": None}
    ratios = []

    for band, x_text, x, n, recursed, direct in cells:
        direction = directions.get((band, x))
        values = reference[(band, x)]
        rec_error = abs(mpf(recursed) - values[n])
        if rec_error == 0:
            continue

        if direction == "upward":
            seed = error0.get((band, x))
            if seed is None or n == 0:
                continue
            if x < thresholds[n]:
                # The entry does not recurse at this order; it reads the fit.
                continue
            ratio = seed * amplification(n, mpf(x)) / rec_error
        else:
            continue

        ratios.append(ratio)
        entry = extremes[direction]
        if entry is None:
            extremes[direction] = [ratio, ratio, band, n, x_text]
        else:
            if ratio < entry[0]:
                entry[0] = ratio
                entry[2] = band
                entry[3] = n
                entry[4] = x_text
            if ratio > entry[1]:
                entry[1] = ratio

    ratios.sort()
    bracket_median = ratios[len(ratios) // 2] if ratios else None
    bracket_below = sum(1 for ratio in ratios if ratio < 1)
    full_bound = {}

    for band in ("A1", "A2"):
        value, n, x_text = band_ship[band]["absolute"]
        x = float(x_text)
        values = reference[(band, x)]
        seed = error0.get((band, x))
        if seed is None:
            continue
        term = seed * amplification(n, mpf(x))
        arith = recurrence_arithmetic(n, mpf(x), values) if n else mpf(0)
        full_bound[band] = ((term + arith) / value, value, term, arith, n, x_text)

    # ---- write the tables -------------------------------------------------
    results = tree / "results"
    results.mkdir(parents=True, exist_ok=True)

    with open(results / "per_order.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band", "n", "method", "worst_abs", "at_x_abs", "worst_rel", "at_x_rel"])
        for band in ("A1", "A2"):
            for n in range(NMAX + 1):
                for method, store_abs, store_rel in (
                    ("ship", ship_abs, ship_rel),
                    ("product", prod_abs, prod_rel),
                ):
                    worst_abs, at_abs = store_abs[(band, n)]
                    worst_rel, at_rel = store_rel[(band, n)]
                    writer.writerow([band, n, method, text(worst_abs), at_abs, text(worst_rel), at_rel])

    with open(results / "band_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["band", "metric", "ship_worst", "ship_at_n", "ship_at_x", "prod_worst", "prod_at_n", "prod_at_x", "ratio_of_band_worsts"]
        )
        for band in ("A1", "A2"):
            for metric in ("absolute", "relative"):
                ship_worst, ship_n, ship_x = band_ship[band][metric]
                prod_worst, prod_n, prod_x = band_prod[band][metric]
                ratio = ship_worst / prod_worst if prod_worst > 0 else mpf("inf")
                writer.writerow([band, metric, text(ship_worst), ship_n, ship_x, text(prod_worst), prod_n, prod_x, text(ratio)])

    with open(results / "transfer_cells.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band", "x", "n", "transfer", "degree", "seed_error", "measurable", "ulp_transfer"])
        writer.writerows(transfer_cells)

    with open(results / "prediction.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["band", "metric", "order", "predicted_ship", "predicted_product", "arithmetic", "state", "measured_winner", "choice", "right_where_committed"]
        )
        for band in ("A1", "A2"):
            for metric in ("absolute", "relative"):
                for n, ps, pd, sigma, state, measured, choice, right in prediction_rows(band, metric, args.margin):
                    writer.writerow([band, metric, n, text(ps), text(pd), text(sigma), state, measured, choice, "1" if right else "0"])

    with open(results / "prediction_scan.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["margin", "band", "metric", "commits", "right_where_committed", "wrong_where_committed", "abstains", "orders"])
        for margin in MARGINS:
            for band in ("A1", "A2"):
                for metric in ("absolute", "relative"):
                    rows = prediction_rows(band, metric, margin)
                    commits, right_committed, abstains = score(rows)
                    writer.writerow([margin, band, metric, commits, right_committed, commits - right_committed, abstains, len(rows)])

    # ---- summary ----------------------------------------------------------
    validation_lines = []
    validation_path = Path(args.validation) if args.validation else tree / "data/reference_validation.txt"
    if validation_path.is_file():
        validation_lines = validation_path.read_text(encoding="utf-8").splitlines()

    grid_path = tree / "data/grid.csv"
    if grid_path.is_file():
        grid_counts, grid_tags = read_grid(grid_path)
        grid_total = grid_counts["A1"] + grid_counts["A2"]
        grid_line = (
            "grid: %d arguments (A1 %d, A2 %d; G tags %d, V tags %d, overlap %d)"
            % (
                grid_total,
                grid_counts["A1"],
                grid_counts["A2"],
                grid_tags["G"],
                grid_tags["V"],
                grid_tags["G"] + grid_tags["V"] - grid_total,
            )
        )
        if grid_tags["other"]:
            grid_line += ", %d other tags" % grid_tags["other"]
    else:
        grid_total = None
        grid_line = "grid: %s not found" % grid_path

    summary = [
        "second sweep summary",
        "reference: %s" % reference_path.resolve(),
        grid_line,
        "arguments: %d seen in the path file, %d with a canonical upward curve"
        % (len(arguments_seen), len(curves.get("A1", [])) + len(curves.get("A2", []))),
        "cells: %d" % len(cells),
        "model population: arguments at or above kTierThresholds[0] = %r: A1 %d, A2 %d"
        % (model_from, model_arguments["A1"], model_arguments["A2"]),
        "  below it (x = 0 and every downward-seeded argument) the order-0 value is not a",
        "  seed error and the upward amplification model does not apply; those cells are",
        "  outside the model.",
        "  inside the population an order above the served prefix is fit-vs-fit: both",
        "  entries read the same stored fit, both predicted errors are that fit's error,",
        "  and the model abstains by construction.",
        "",
        "the argument x = 0 is in the grid and has no finite upward amplification; its seed",
        "rows are in paths.csv, and it is outside the transfer analysis as well.",
        "",
        "admission:",
    ]
    for direction in sorted(admission_totals):
        entries, checked, matched = admission_totals[direction]
        summary.append(
            "  %-9s arguments %4d, cells %6d, bit-identical %6d%s"
            % (direction, entries, checked, matched, "" if checked == matched else "  MISMATCH")
        )
    for mismatch in admission_mismatches:
        summary.append("  mismatch: %s" % (mismatch,))
    summary += [
        "",
        "transfer, canonical path (start 0 upward, degree of largest resolved seed error):",
        "  A1 %d measurable of %d cells, A2 %d of %d"
        % (measurable_counts["A1"], transfer_counts["A1"], measurable_counts["A2"], transfer_counts["A2"]),
        "  worst relative deviation from the closed form: A1 %s (%s), A2 %s (%s)"
        % (
            text(transfer_form_worst["A1"]) if transfer_form_worst["A1"] is not None else "none",
            transfer_form_where["A1"],
            text(transfer_form_worst["A2"]) if transfer_form_worst["A2"] is not None else "none",
            transfer_form_where["A2"],
        ),
        "  the two closed forms A_n = (2n-1)!!/(2x)^n and Gamma(n+1/2)/(sqrt(pi) x^n)",
        "  agree to %s at %s"
        % (text(form_pair_worst) if form_pair_worst is not None else "none", form_pair_where),
        "  the measured transfer against the gamma form deviates by at most %s at %s"
        % (text(gamma_form_worst) if gamma_form_worst is not None else "none", gamma_form_where),
        "",
        "x-dependence of log A_n, measured minus predicted:",
        "  A1 %s" % residual_stats("A1"),
        "  A2 %s" % residual_stats("A2"),
        "",
        "n^p q^n with two free parameters, over %d cells:" % (unfit[4] if unfit else 0),
    ]
    if unfit is not None:
        summary.append(
            "  maximum %s, rms %s (p = %s, q = %s)"
            % (unfit[0], unfit[1], unfit[2], unfit[3])
        )
    else:
        summary.append("  not defined on the measured cells")
    summary.append(
        "  derived form on the same cells: maximum %.3e, rms %.3e"
        % (derived_max, derived_rms)
    )
    summary += [
        "",
        "argmin:",
        "  A1 %d of %d resolved arguments match the discrete rule; A2 %d of %d"
        % (argmin_matches["A1"], argmin_counts["A1"], argmin_matches["A2"], argmin_counts["A2"]),
        "  the x + 1/2 reading matches A1 %d, A2 %d of the same arguments"
        % (wrong_reading_matches["A1"], wrong_reading_matches["A2"]),
        "  the continuous x - 1/2 reading matches A1 %d, A2 %d"
        % (continuous_matches["A1"], continuous_matches["A2"]),
        "  continuous exceptions: %s" % (", ".join(continuous_exceptions) if continuous_exceptions else "none"),
        "  neighbour gap A1: %s" % gap_stats("A1"),
        "  neighbour gap A2: %s" % gap_stats("A2"),
        "  half-integer ties: A1 %s, A2 %s"
        % (tie_arguments["A1"] or "none", tie_arguments["A2"] or "none"),
        "  unresolved arguments: A1 %d, A2 %d" % (unresolved["A1"], unresolved["A2"]),
        "",
        "exact-perturbation transfer (sampled every %d rows):" % args.ulp_sample,
        "  worst relative deviation %s at %s over %d cells"
        % (text(ulp_worst) if ulp_worst is not None else "none", ulp_worst_where, ulp_cells),
        "start-independence (non-canonical starts, every %dth argument):" % args.start_sample,
        "  worst relative deviation %s at %s over %d cells"
        % (text(cross_worst) if cross_worst is not None else "none", cross_worst_where, cross_cells),
        "",
        "crossing model at margin = %s x its own arithmetic term:" % args.margin,
    ]
    for band in ("A1", "A2"):
        for metric in ("absolute", "relative"):
            rows = prediction_rows(band, metric, args.margin)
            commits, right_committed, abstains = score(rows)

            if metric == "absolute":
                ship_store, prod_store = ship_abs, prod_abs
            else:
                ship_store, prod_store = ship_rel, prod_rel

            chosen = []
            margins_measured = []

            for row in rows:
                n, ps, pd, sigma, state, measured, choice, right = row
                if state == "abstain":
                    continue
                ship_value = ship_store[(band, n)][0]
                prod_value = prod_store[(band, n)][0]
                chosen.append(ship_value if choice == "ship" else prod_value)
                low, high = min(ship_value, prod_value), max(ship_value, prod_value)
                margins_measured.append(high / low if low > 0 else mpf("inf"))

            if chosen:
                selected_worst = max(chosen)
                single_ship = max(ship_store[(band, row[0])][0] for row in rows if row[4] != "abstain")
                single_prod = max(prod_store[(band, row[0])][0] for row in rows if row[4] != "abstain")
                better_single = min(single_ship, single_prod)
                improvement = better_single / selected_worst if selected_worst > 0 else mpf("inf")
                smallest_margin = min(margins_measured)
            else:
                selected_worst = single_ship = single_prod = better_single = mpf(0)
                improvement = mpf("inf")
                smallest_margin = mpf("inf")

            summary.append(
                "  %s %-8s commits %d, right where committed %d, abstains %d; "
                "selection improvement %s; smallest measured margin %s"
                % (
                    band,
                    metric,
                    commits,
                    right_committed,
                    abstains,
                    text(improvement),
                    text(smallest_margin),
                )
            )

    summary += [
        "  the abstention threshold is the recursed bound's own accumulated arithmetic term;",
        "  prediction_scan.csv reports the score over a pre-registered margin ladder.",
        "",
        "model bracket (amplified seed term over measured recursed error, upward cells only):",
    ]
    entry = extremes["upward"]
    if entry is None:
        summary.append("  no cells")
    else:
        summary.append(
            "  seed term alone: from %s to %s (extreme at %s n=%d x=%s)"
            % (text(entry[0]), text(entry[1]), entry[2], entry[3], entry[4])
        )
        summary.append(
            "  below one at %d of %d cells; median %s"
            % (bracket_below, len(ratios), text(bracket_median))
        )
    for band in ("A1", "A2"):
        if band in full_bound:
            ratio, value, term, arith, n, x_text = full_bound[band]
            summary.append(
                "  full bound at the measured band worst %s: %s x measured "
                "(seed %s + arithmetic %s over error %s, n=%d x=%s)"
                % (band, text(ratio), text(term), text(arith), text(value), n, x_text)
            )
    summary += [
        "",
        "point-count note: the manuscript's second experiment has 836 arguments;",
        "this grid has %s (cell total %s x 33 = %s)."
        % (
            grid_total if grid_total is not None else "no grid file",
            grid_total if grid_total is not None else "?",
            grid_total * 33 if grid_total is not None else "?",
        ),
        "",
        "reference validation:",
    ]
    summary += ["  " + line for line in validation_lines]

    with open(results / "summary.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary) + "\n")

    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
