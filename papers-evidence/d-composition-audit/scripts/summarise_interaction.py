#!/usr/bin/env python3
"""Summarise the composed-bounds interaction run.

Reads the per-(cell, n, x) CSV written by measure-interaction and computes:

  * the per-cell responses E = max |err|/F_0 and R = max |err|/limit, where
    limit is the lane row's whole statement: the accessor's figure plus, on the
    half lanes, the value-dependent half digit the row states beside it;
  * the safety test R <= 1 (a violation is a bound bug, not a result), and,
    separately, the accessor ratio: on a half lane it exceeds 1 wherever the
    digit binds, which is the contract's shape and not a violation;
  * for every pair of axes, the difference-in-differences interaction on E,
    over every completion of the other axes, with materiality at the
    pre-registered 10% and sensitivity at 5% and 20%;
  * the interaction as a function of the argument: for every selected
    contrast, the interaction at each nominal argument, split exact/generic;
  * a full-factorial multi-way interaction decomposition over the six
    within-lane axes (route, scheme, partition, pack, form, exp), including
    the three-level partition and form contrasts, every higher-order term and
    the main effects, plus a cross-lane decomposition (precision included) on
    exact arguments;
  * an explicit liveness report: where each axis changes a returned value, so
    a zero interaction is not read as evidence of additivity where the axis is
    dead;
  * a verdict at each materiality margin.

The interaction response is taken on the resolving lanes only (fp64, fp32).
The half lanes' published figure is base + axis terms + half of the last
representable digit of the returned value (boys src/boys.cpp:634-644,
:710-725). That digit is three to four orders above the axis terms, so a
factorial response built from their returned values measures the store quantum
and its interaction with x and n, not the bound algebra. The half lanes are
measured, safety-checked against their whole row statement, and reported; they
are not evidence about whether the axis terms compose.

The multi-way decomposition uses orthonormal contrasts: mean and, for a factor
with L levels, L-1 orthogonal zero-sum components (linear/quadratic for 3
levels, linear/quadratic/cubic for 4). Its materiality measure is a component's
or an order's share of the point's centred total sum of squares (mean excluded),
not the orthonormal coefficient: the coefficient's scale would grow with the
number of factor levels and cells. The pairwise measure stays in response
units, |I| over the largest of the four responses. Because the design is
balanced and saturated, the highest-order components absorb one-observation
residual; the summary reports shares and never claims a significance test.

Nothing here re-runs the driver.

Usage:
    python summarise_interaction.py --cells results/cells.csv --out results/summary.json
"""

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict

AXES = ["precision", "route", "scheme", "partition", "pack", "form", "exp"]

WITHIN = ["route", "scheme", "partition", "pack", "form", "exp"]

LEVELS = {
    "precision": ["fp64", "fp32", "fp16", "bf16"],
    "route": ["chebyshev", "rational-minimax"],
    "scheme": ["split-clenshaw", "horner"],
    "partition": ["shipped", "narrow", "uniform"],
    "pack": ["arguments", "orders"],
    "form": ["exact-division", "plain-reciprocal", "refined-reciprocal"],
    "exp": ["accurate", "fast"],
}

ALPHAS = [0.05, 0.10, 0.20]
PRIMARY_ALPHA = 0.10

# The lanes whose published figure is a sum of axis terms and whose returned
# value resolves them. The half lanes store at 2^-11 / 2^-8 of the value's
# binade; that quantum is the dominant term of their row and leaves no
# resolution for a 1e-7 axis contrast. Keeping them out of the interaction
# verdict is a statement about instrument resolution, not an omitted option:
# every cell is measured and every safety row is judged.
RESOLVING_LANES = ["fp64", "fp32"]
STORE_DOMINATED_LANES = ["fp16", "bf16"]

# Orthonormal contrast rows. Row 0 is the mean; the remaining rows are
# orthogonal zero-sum contrasts, normalized at use by their norm.
CONTRASTS = {
    2: [[1.0, 1.0], [1.0, -1.0]],
    3: [[1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, -2.0, 1.0]],
    4: [
        [1.0, 1.0, 1.0, 1.0],
        [-3.0, -1.0, 1.0, 3.0],
        [1.0, -1.0, -1.0, 1.0],
        [-1.0, 3.0, -3.0, 1.0],
    ],
}

CHOICE_NAMES = {
    2: ["mean", "contrast"],
    3: ["mean", "linear", "quadratic"],
    4: ["mean", "linear", "quadratic", "cubic"],
}

FLOAT_FIELDS = [
    "x",
    "value",
    "ref_hi",
    "ref_lo",
    "abs_err",
    "f0",
    "fn",
    "figure",
    "half_digit",
    "limit",
    "accessor_ratio",
    "figure_ratio",
]

INT_FIELDS = ["n", "pack_live", "partition_in_domain"]


def read_cells(path):
    cells = defaultdict(dict)
    refusals = []
    total = 0
    kind_counts = defaultdict(int)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            total += 1
            key = tuple(raw[a] for a in AXES)
            nominal = float(raw["x_nominal"])
            point = (int(raw["n"]), nominal)
            kind = raw.get("argument_kind", "unknown")
            kind_counts[kind] += 1

            if raw["status"] != "ok":
                refusals.append({"cell": key, "point": point, "status": raw["status"]})
                continue

            rec = {}
            for field in FLOAT_FIELDS:
                text = raw.get(field, "")
                rec[field] = float(text) if text not in ("", None) else None
            # Older CSVs predate the half-lane limit columns; treat the
            # accessor figure as the whole statement there.
            if rec["half_digit"] is None:
                rec["half_digit"] = 0.0
            if rec["limit"] is None:
                rec["limit"] = rec["figure"]
            if rec["accessor_ratio"] is None:
                rec["accessor_ratio"] = rec["figure_ratio"]
            rec["f0_ratio"] = (
                rec["abs_err"] / rec["f0"] if rec["f0"] and rec["f0"] > 0.0 else 0.0
            )
            for field in INT_FIELDS:
                rec[field] = int(raw[field])
            rec["status"] = raw["status"]
            rec["kind"] = kind
            cells[key][point] = rec

    return cells, refusals, total, dict(kind_counts)


def domain_ok(rec, pair):
    if "pack" in pair and not rec["pack_live"]:
        return False
    if "partition" in pair and not rec["partition_in_domain"]:
        return False
    return True


def response(points, pair):
    best = None
    best_point = None
    for point, rec in points.items():
        if not domain_ok(rec, pair):
            continue
        if best is None or rec["f0_ratio"] > best:
            best = rec["f0_ratio"]
            best_point = point
    return best, best_point


def pair_contrasts(cells, pair):
    """Difference in differences for every completion of the other axes."""
    a, b = pair
    other = [axis for axis in AXES if axis not in pair]
    contrasts = []

    for completion in itertools.product(*(LEVELS[axis] for axis in other)):
        base = dict(zip(other, completion))
        levels_a = LEVELS[a]
        levels_b = LEVELS[b]
        level_a0, level_b0 = levels_a[0], levels_b[0]

        def key_for(level_a, level_b):
            return tuple(
                dict(base, **{a: level_a, b: level_b})[axis] for axis in AXES
            )

        def get(level_a, level_b):
            points = cells.get(key_for(level_a, level_b))
            if not points:
                return None, None
            return response(points, pair)

        e00, at00 = get(level_a0, level_b0)
        if e00 is None:
            continue

        for level_a in levels_a[1:]:
            e10, at10 = get(level_a, level_b0)
            if e10 is None:
                continue
            for level_b in levels_b[1:]:
                e01, at01 = get(level_a0, level_b)
                e11, at11 = get(level_a, level_b)
                if e01 is None or e11 is None:
                    continue

                m_a = e10 - e00
                m_b = e01 - e00
                interaction = e11 - e10 - e01 + e00
                scale = max(abs(e00), abs(e10), abs(e01), abs(e11))
                informative = (abs(m_a) + abs(m_b)) > 0.0
                relative = abs(interaction) / scale if scale > 0.0 else None
                material_at = {
                    str(alpha): bool(
                        informative
                        and scale > 0.0
                        and abs(interaction) > alpha * scale
                    )
                    for alpha in ALPHAS
                }

                record = {
                    "completion": base,
                    "levels": {a: [level_a0, level_a], b: [level_b0, level_b]},
                    "E00": e00,
                    "E10": e10,
                    "E01": e01,
                    "E11": e11,
                    "E_at": {"00": at00, "10": at10, "01": at01, "11": at11},
                    "marginal_a": m_a,
                    "marginal_b": m_b,
                    "interaction": interaction,
                    "relative": relative,
                    "informative": informative,
                    "material_at": material_at,
                    "material": material_at[str(PRIMARY_ALPHA)],
                    "keys": {
                        "00": key_for(level_a0, level_b0),
                        "10": key_for(level_a, level_b0),
                        "01": key_for(level_a0, level_b),
                        "11": key_for(level_a, level_b),
                    },
                }

                # Pointwise interaction for this four-cell block, so a maximum
                # that merely moved to another point is visible.
                pointwise = None
                for point, r00 in cells[record["keys"]["00"]].items():
                    r10 = cells[record["keys"]["10"]].get(point)
                    r01 = cells[record["keys"]["01"]].get(point)
                    r11 = cells[record["keys"]["11"]].get(point)
                    if r10 is None or r01 is None or r11 is None:
                        continue
                    if not all(domain_ok(r, pair) for r in (r00, r10, r01, r11)):
                        continue
                    pa = r10["f0_ratio"] - r00["f0_ratio"]
                    pb = r01["f0_ratio"] - r00["f0_ratio"]
                    if abs(pa) + abs(pb) == 0.0:
                        continue
                    pi = (
                        r11["f0_ratio"]
                        - r10["f0_ratio"]
                        - r01["f0_ratio"]
                        + r00["f0_ratio"]
                    )
                    if pointwise is None or abs(pi) > pointwise:
                        pointwise = abs(pi)

                if pointwise is not None:
                    record["pointwise_max_abs_interaction"] = pointwise

                contrasts.append(record)

    return contrasts


def block_profile(cells, record, pair):
    """Interaction along the argument for one four-cell contrast."""
    by_x = defaultdict(
        lambda: {
            "max_abs": 0.0,
            "interaction": 0.0,
            "n_at_max": None,
            "scale": 0.0,
            "relative": None,
            "kind": "unknown",
        }
    )

    r00s = cells[record["keys"]["00"]]
    for point, r00 in r00s.items():
        r10 = cells[record["keys"]["10"]].get(point)
        r01 = cells[record["keys"]["01"]].get(point)
        r11 = cells[record["keys"]["11"]].get(point)
        if r10 is None or r01 is None or r11 is None:
            continue
        if not all(domain_ok(r, pair) for r in (r00, r10, r01, r11)):
            continue
        pa = r10["f0_ratio"] - r00["f0_ratio"]
        pb = r01["f0_ratio"] - r00["f0_ratio"]
        if abs(pa) + abs(pb) == 0.0:
            continue

        n, x = point
        pi = r11["f0_ratio"] - r10["f0_ratio"] - r01["f0_ratio"] + r00["f0_ratio"]
        scale = max(
            abs(r00["f0_ratio"]),
            abs(r10["f0_ratio"]),
            abs(r01["f0_ratio"]),
            abs(r11["f0_ratio"]),
        )
        entry = by_x[x]
        if abs(pi) > entry["max_abs"]:
            entry["max_abs"] = abs(pi)
            entry["interaction"] = pi
            entry["n_at_max"] = n
            entry["scale"] = scale
            entry["relative"] = abs(pi) / scale if scale > 0.0 else None
            entry["kind"] = r00["kind"]

    return [
        {
            "x": x,
            "kind": entry["kind"],
            "interaction": entry["interaction"],
            "abs_interaction": entry["max_abs"],
            "relative": entry["relative"],
            "n_at_max": entry["n_at_max"],
            "scale": entry["scale"],
        }
        for x, entry in sorted(by_x.items())
    ]


def select_for_profiles(contrasts):
    """All material contrasts at the primary alpha, plus the top five."""
    ranked = sorted(
        (c for c in contrasts if c["relative"] is not None),
        key=lambda c: c["relative"],
        reverse=True,
    )
    selected = [c for c in contrasts if c["material_at"][str(PRIMARY_ALPHA)]]
    for candidate in ranked[:5]:
        if candidate not in selected:
            selected.append(candidate)
    return selected[:20]


def transform_in_place(values, levels):
    """Apply the orthonormal contrast transform along every factor axis."""
    strides = []
    stride = 1
    for level_count in reversed(levels):
        strides.append(stride)
        stride *= level_count
    strides.reverse()

    for level_count, axis_stride in zip(levels, strides):
        matrix = CONTRASTS[level_count]
        block = axis_stride * level_count
        for start in range(0, len(values), block):
            for offset in range(axis_stride):
                source = [values[start + c * axis_stride + offset] for c in range(level_count)]
                for row in range(level_count):
                    values[start + row * axis_stride + offset] = sum(
                        matrix[row][c] * source[c] for c in range(level_count)
                    )

    return strides


def component_metadata(levels):
    """Choice, coefficient norm, factor support and interaction order per component."""
    strides = []
    stride = 1
    for level_count in reversed(levels):
        strides.append(stride)
        stride *= level_count
    strides.reverse()

    norms = [
        [math.sqrt(sum(c * c for c in row)) for row in CONTRASTS[level_count]]
        for level_count in levels
    ]

    metadata = []
    total = math.prod(levels)
    for index in range(total):
        choices = tuple((index // axis_stride) % level_count
                        for level_count, axis_stride in zip(levels, strides))
        norm = 1.0
        support = []
        for axis, (choice, level_count) in enumerate(zip(choices, levels)):
            norm *= norms[axis][choice]
            if choice != 0:
                support.append(axis)
        metadata.append(
            {
                "choices": choices,
                "norm": norm,
                "support": tuple(support),
                "order": len(support),
            }
        )

    return metadata


def levels_of(axis, level_override=None):
    if level_override and axis in level_override:
        return level_override[axis]
    return LEVELS[axis]


def analysis_points(cells, fixed, factors, level_override=None):
    """Common points of a complete factorial, or the cells that break it."""
    keys = []
    missing = []
    for combo in itertools.product(*(levels_of(axis, level_override) for axis in factors)):
        key = tuple(dict(fixed, **dict(zip(factors, combo)))[axis] for axis in AXES)
        if key not in cells:
            missing.append(key)
            continue
        keys.append(key)

    if missing or not keys:
        return [], keys, missing

    common = set(cells[keys[0]])
    for key in keys[1:]:
        common &= set(cells[key])

    return sorted(common), keys, []


def decompose_values(raw, levels):
    values = list(raw)
    transform_in_place(values, levels)
    return values


def multiway(cells, fixed, factors, alphas, level_override=None):
    """Full-factorial interaction decomposition for one block."""
    level_lists = [levels_of(axis, level_override) for axis in factors]
    levels = [len(level_list) for level_list in level_lists]
    metadata = component_metadata(levels)
    points, keys, missing = analysis_points(cells, fixed, factors, level_override)

    if not points:
        problem = {
            "factors": factors,
            "points": 0,
        }
        if missing:
            problem["missing_count"] = len(missing)
            problem["missing_cells"] = [
                dict(zip(AXES, key)) for key in missing[:12]
            ]
            problem["note"] = (
                "the factorial is incomplete: a structurally refused or unserved "
                "cell breaks the balanced design, so no multi-way decomposition is "
                "reported for this block"
            )
        else:
            problem["note"] = "no point is common to every cell of the factorial"
        return problem

    kind_of = {p: cells[keys[0]][p]["kind"] for p in points}

    order_stats = {
        order: {
            "max_abs": 0.0,
            "max_abs_at": None,
            "max_abs_component": None,
            "max_component_share": 0.0,
            "max_component_share_at": None,
            "max_component_share_component": None,
            "max_order_share": 0.0,
            "max_order_share_at": None,
            "ss_sum": 0.0,
            "material_points": {str(a): {"all": 0, "exact": 0, "generic": 0}
                                for a in alphas},
        }
        for order in range(len(factors) + 1)
    }
    set_stats = {}
    component_best = [
        {
            "max_abs": 0.0,
            "max_rel": 0.0,
            "max_share": 0.0,
            "at": None,
            "share_at": None,
            "scale": 0.0,
            "centered_ss": 0.0,
        }
        for _ in metadata
    ]

    centered_ss_sum = 0.0

    for point in points:
        raw = [cells[key][point]["f0_ratio"] for key in keys]
        scale = max(raw) if raw else 0.0
        effects = decompose_values(raw, levels)
        coefficients = [
            effects[index] / meta["norm"] for index, meta in enumerate(metadata)
        ]

        # The transform rows are orthogonal and the normalised coefficients are
        # orthonormal projections, so their squares are the point's sum-of-squares
        # decomposition. The mean (order 0) is not part of the centred total.
        centered_ss = sum(
            coefficient * coefficient
            for coefficient, meta in zip(coefficients, metadata)
            if meta["order"] >= 1
        )
        centered_ss_sum += centered_ss

        order_point_ss = defaultdict(float)
        set_point_ss = defaultdict(float)

        for index, meta in enumerate(metadata):
            coefficient = coefficients[index]
            order = meta["order"]
            abs_effect = abs(coefficient)
            ss = coefficient * coefficient

            if abs_effect > order_stats[order]["max_abs"]:
                order_stats[order]["max_abs"] = abs_effect
                order_stats[order]["max_abs_at"] = point
                order_stats[order]["max_abs_component"] = index

            share = None
            if order >= 1:
                order_stats[order]["ss_sum"] += ss
                order_point_ss[order] += ss
                if centered_ss > 0.0:
                    share = ss / centered_ss
                    if share > order_stats[order]["max_component_share"]:
                        order_stats[order]["max_component_share"] = share
                        order_stats[order]["max_component_share_at"] = point
                        order_stats[order]["max_component_share_component"] = index

            if order >= 1:
                support = tuple(factors[i] for i in meta["support"])
                stat = set_stats.setdefault(
                    support,
                    {
                        "factors": list(support),
                        "max_abs": 0.0,
                        "max_rel": 0.0,
                        "max_component_share": 0.0,
                        "max_set_share": 0.0,
                        "max_abs_at": None,
                        "max_component_share_at": None,
                        "max_set_share_at": None,
                        "ss_sum": 0.0,
                        "material_points": {str(a): 0 for a in alphas},
                    },
                )
                stat["ss_sum"] += ss
                if abs_effect > stat["max_abs"]:
                    stat["max_abs"] = abs_effect
                    stat["max_abs_at"] = point
                    stat["max_rel"] = abs_effect / scale if scale > 0.0 else 0.0
                if share is not None and share > stat["max_component_share"]:
                    stat["max_component_share"] = share
                    stat["max_component_share_at"] = point
                set_point_ss[support] += ss

            best = component_best[index]
            if abs_effect > best["max_abs"]:
                best["max_abs"] = abs_effect
                best["max_rel"] = abs_effect / scale if scale > 0.0 else 0.0
                best["at"] = point
                best["scale"] = scale
            if share is not None and share > best["max_share"]:
                best["max_share"] = share
                best["share_at"] = point
                best["centered_ss"] = centered_ss

        for order, point_ss in order_point_ss.items():
            if order < 1 or centered_ss <= 0.0:
                continue
            order_share = point_ss / centered_ss
            if order_share > order_stats[order]["max_order_share"]:
                order_stats[order]["max_order_share"] = order_share
                order_stats[order]["max_order_share_at"] = point
            if order < 1:
                continue
            for alpha in alphas:
                if order_share > alpha:
                    order_stats[order]["material_points"][str(alpha)]["all"] += 1
                    order_stats[order]["material_points"][str(alpha)][kind_of[point]] += 1

        for support, point_ss in set_point_ss.items():
            if centered_ss <= 0.0:
                continue
            set_share = point_ss / centered_ss
            stat = set_stats[support]
            if set_share > stat["max_set_share"]:
                stat["max_set_share"] = set_share
                stat["max_set_share_at"] = point
            for alpha in alphas:
                if set_share > alpha:
                    stat["material_points"][str(alpha)] += 1

    orders = {}
    for order, stat in order_stats.items():
        stat["ss_share"] = (
            stat["ss_sum"] / centered_ss_sum
            if order >= 1 and centered_ss_sum > 0.0
            else None
        )
        stat.pop("ss_sum", None)
        orders[str(order)] = stat

    ranked_sets = sorted(
        set_stats.values(), key=lambda s: s["max_set_share"], reverse=True
    )
    for stat in ranked_sets:
        stat["ss_share"] = (
            stat["ss_sum"] / centered_ss_sum if centered_ss_sum > 0.0 else None
        )
        stat.pop("ss_sum", None)

    top = []
    for index in sorted(
        (i for i, meta in enumerate(metadata)
         if meta["order"] >= 2 and component_best[i]["max_abs"] > 0.0),
        key=lambda i: component_best[i]["max_share"],
        reverse=True,
    )[:8]:
        meta = metadata[index]
        top.append(
            {
                "component_index": index,
                "factors": [factors[i] for i in meta["support"]],
                "choices": {
                    factors[i]: CHOICE_NAMES[levels[i]][meta["choices"][i]]
                    for i in meta["support"]
                },
                "order": meta["order"],
                "max_abs": component_best[index]["max_abs"],
                "max_rel": component_best[index]["max_rel"],
                "max_share": component_best[index]["max_share"],
                "at": component_best[index]["at"],
                "share_at": component_best[index]["share_at"],
            }
        )

    main_effects = []
    for index, meta in enumerate(metadata):
        if meta["order"] != 1:
            continue
        factor_index = meta["support"][0]
        main_effects.append(
            {
                "component_index": index,
                "factor": factors[factor_index],
                "choice": CHOICE_NAMES[levels[factor_index]][meta["choices"][factor_index]],
                "max_abs": component_best[index]["max_abs"],
                "max_rel": component_best[index]["max_rel"],
                "max_share": component_best[index]["max_share"],
                "at": component_best[index]["at"],
                "share_at": component_best[index]["share_at"],
            }
        )

    return {
        "factors": factors,
        "levels": {axis: level_lists[i] for i, axis in enumerate(factors)},
        "points": len(points),
        "point_kinds": {
            "exact": sum(1 for p in points if kind_of[p] == "exact"),
            "generic": sum(1 for p in points if kind_of[p] == "generic"),
        },
        "orders": orders,
        "sets": ranked_sets,
        "main_effects": main_effects,
        "top_components": top,
        "_metadata": metadata,
        "_points": points,
        "_keys": keys,
    }


def component_profiles(cells, block, targets):
    """Interaction along the argument for selected multi-way components."""
    factors = block["factors"]
    levels = [len(block["levels"][axis]) for axis in factors]
    metadata = block["_metadata"]
    keys = block["_keys"]
    points = block["_points"]

    series = {
        target["component_index"]: defaultdict(
            lambda: {
                "max_abs": 0.0,
                "effect": 0.0,
                "n_at_max": None,
                "scale": 0.0,
                "relative": None,
                "share": None,
                "kind": "unknown",
            }
        )
        for target in targets
    }

    for point in points:
        raw = [cells[key][point]["f0_ratio"] for key in keys]
        scale = max(raw) if raw else 0.0
        effects = decompose_values(raw, levels)
        centered_ss = sum(
            (effects[i] / meta["norm"]) ** 2
            for i, meta in enumerate(metadata)
            if meta["order"] >= 1
        )
        n, x = point
        kind = cells[keys[0]][point]["kind"]

        for target in targets:
            index = target["component_index"]
            effect = effects[index] / metadata[index]["norm"]
            entry = series[index][x]
            if abs(effect) > entry["max_abs"]:
                entry["max_abs"] = abs(effect)
                entry["effect"] = effect
                entry["n_at_max"] = n
                entry["scale"] = scale
                entry["relative"] = abs(effect) / scale if scale > 0.0 else None
                entry["share"] = (
                    effect * effect / centered_ss if centered_ss > 0.0 else None
                )
                entry["kind"] = kind

    output = {}
    for index, by_x in series.items():
        profile = [
            {
                "x": x,
                "kind": entry["kind"],
                "effect": entry["effect"],
                "abs_effect": entry["max_abs"],
                "relative": entry["relative"],
                "share": entry["share"],
                "n_at_max": entry["n_at_max"],
                "scale": entry["scale"],
            }
            for x, entry in sorted(by_x.items())
        ]
        output[str(index)] = profile
    return output


def profile_summary(profile):
    """Sign changes, exact/generic split and variance share for one profile."""
    relatives = [p["relative"] for p in profile if p["relative"] is not None]
    shares = [p["share"] for p in profile if p.get("share") is not None]
    signed = [p["effect"] if "effect" in p else p["interaction"] for p in profile]
    positive = any(value > 0.0 for value in signed)
    negative = any(value < 0.0 for value in signed)

    exact = [p["relative"] for p in profile
             if p["relative"] is not None and p["kind"] == "exact"]
    generic = [p["relative"] for p in profile
               if p["relative"] is not None and p["kind"] == "generic"]
    exact_shares = [p["share"] for p in profile
                    if p.get("share") is not None and p["kind"] == "exact"]
    generic_shares = [p["share"] for p in profile
                      if p.get("share") is not None and p["kind"] == "generic"]

    return {
        "points": len(profile),
        "sign_changes": bool(positive and negative),
        "min_relative": min(relatives) if relatives else None,
        "max_relative": max(relatives) if relatives else None,
        "max_relative_at": (
            max(profile, key=lambda p: p["relative"] if p["relative"] is not None else -1.0)["x"]
            if relatives
            else None
        ),
        "max_relative_exact": max(exact) if exact else None,
        "max_relative_generic": max(generic) if generic else None,
        "max_share": max(shares) if shares else None,
        "max_share_at": (
            max(profile, key=lambda p: p["share"] if p.get("share") is not None else -1.0)["x"]
            if shares
            else None
        ),
        "max_share_exact": max(exact_shares) if exact_shares else None,
        "max_share_generic": max(generic_shares) if generic_shares else None,
    }


def flag_liveness(cells):
    """Rows where the pack and partition domain flags say the axis can act."""
    pack = {}
    partition = {}
    for key, points in cells.items():
        cell = dict(zip(AXES, key))
        for rec in points.values():
            entry = pack.setdefault(cell["precision"], {"rows": 0, "live": 0})
            entry["rows"] += 1
            entry["live"] += 1 if rec["pack_live"] else 0
            entry = partition.setdefault(
                cell["partition"], {"rows": 0, "live": 0}
            )
            entry["rows"] += 1
            entry["live"] += 1 if rec["partition_in_domain"] else 0
    return {"pack_by_lane": pack, "partition_domain_by_level": partition}


def axis_liveness(cells, lanes):
    """Where each axis changes a returned value, per lane and argument kind.

    A zero interaction on a dead axis is a liveness fact, not evidence of
    additivity. The witness is the returned value: if every level of the axis
    returns the same bits at a point, the axis did nothing there.
    """
    report = {}
    for axis in WITHIN:
        other = [a for a in WITHIN if a != axis]
        by_lane = {}
        for lane in lanes:
            stats = {
                "blocks": 0,
                "points": 0,
                "live_points": 0,
                "live_exact": 0,
                "live_generic": 0,
                "max_spread": 0.0,
                "max_spread_at": None,
            }
            for completion in itertools.product(*(LEVELS[a] for a in other)):
                base = dict(zip(other, completion))
                keys = []
                complete = True
                for level in LEVELS[axis]:
                    key = tuple(
                        dict(base, **{"precision": lane, axis: level})[a]
                        for a in AXES
                    )
                    if key not in cells:
                        complete = False
                        break
                    keys.append(key)
                if not complete:
                    continue
                common = set(cells[keys[0]])
                for key in keys[1:]:
                    common &= set(cells[key])
                if not common:
                    continue
                stats["blocks"] += 1
                for point in common:
                    values = [cells[key][point]["value"] for key in keys]
                    spread = max(values) - min(values)
                    stats["points"] += 1
                    if spread > 0.0:
                        stats["live_points"] += 1
                        kind = cells[keys[0]][point]["kind"]
                        if kind == "exact":
                            stats["live_exact"] += 1
                        elif kind == "generic":
                            stats["live_generic"] += 1
                    if spread > stats["max_spread"]:
                        stats["max_spread"] = spread
                        stats["max_spread_at"] = point
            by_lane[lane] = stats
        report[axis] = by_lane
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cells, refusals, total_rows, kind_counts = read_cells(args.cells)

    if not cells:
        print("FAIL: no usable rows in the cells CSV", file=sys.stderr)
        return 1

    design_cells = math.prod(len(LEVELS[axis]) for axis in AXES)
    precisions_present = sorted({key[0] for key in cells})
    expected_cells = len(precisions_present) * math.prod(
        len(LEVELS[axis]) for axis in WITHIN
    )
    resolving_lanes = [p for p in precisions_present if p in RESOLVING_LANES]
    store_lanes = [p for p in precisions_present if p in STORE_DOMINATED_LANES]
    resolving_cells = {k: v for k, v in cells.items() if k[0] in RESOLVING_LANES}
    store_cells = {k: v for k, v in cells.items() if k[0] in STORE_DOMINATED_LANES}

    cell_list = []
    violations = []
    max_ratio = 0.0
    max_ratio_at = None
    max_accessor_ratio = 0.0
    max_accessor_ratio_at = None
    accessor_only_exceeded = 0
    lane_safety = {
        lane: {
            "max_figure_ratio": 0.0,
            "max_figure_ratio_at": None,
            "max_accessor_ratio": 0.0,
            "max_accessor_ratio_at": None,
            "accessor_only_exceeded": 0,
            "violations": 0,
        }
        for lane in precisions_present
    }

    for key in sorted(cells):
        points = cells[key]
        e, e_at = response(points, ())
        r = None
        r_at = None
        lane_stats = lane_safety[key[0]]
        for point, rec in points.items():
            if rec["figure_ratio"] >= 0.0 and (r is None or rec["figure_ratio"] > r):
                r = rec["figure_ratio"]
                r_at = point
            if rec["accessor_ratio"] > lane_stats["max_accessor_ratio"]:
                lane_stats["max_accessor_ratio"] = rec["accessor_ratio"]
                lane_stats["max_accessor_ratio_at"] = point
            if rec["figure_ratio"] > lane_stats["max_figure_ratio"]:
                lane_stats["max_figure_ratio"] = rec["figure_ratio"]
                lane_stats["max_figure_ratio_at"] = point
            if rec["figure_ratio"] > 1.0 + 1e-9:
                lane_stats["violations"] += 1
                violations.append(
                    {
                        "cell": dict(zip(AXES, key)),
                        "point": point,
                        "figure_ratio": rec["figure_ratio"],
                        "accessor_ratio": rec["accessor_ratio"],
                    }
                )
            elif rec["accessor_ratio"] > 1.0 + 1e-9:
                # Expected on a half lane wherever the row's digit binds: the
                # accessor figure alone is not the row's whole statement.
                lane_stats["accessor_only_exceeded"] += 1
                accessor_only_exceeded += 1
            if rec["figure_ratio"] > max_ratio:
                max_ratio = rec["figure_ratio"]
                max_ratio_at = {"cell": dict(zip(AXES, key)), "point": point}
            if rec["accessor_ratio"] > max_accessor_ratio:
                max_accessor_ratio = rec["accessor_ratio"]
                max_accessor_ratio_at = {
                    "cell": dict(zip(AXES, key)),
                    "point": point,
                }

        worst = max(points.values(), key=lambda rec: rec["abs_err"])
        cell_list.append(
            {
                "cell": dict(zip(AXES, key)),
                "E_f0": e,
                "E_at": e_at,
                "R_figure": r,
                "R_at": r_at,
                "worst_abs_err": worst["abs_err"],
                "worst_point": [worst["n"], worst["x"]],
            }
        )

    pairs = []
    material_total = {str(alpha): 0 for alpha in ALPHAS}
    informative_total = 0

    for pair in itertools.combinations(AXES, 2):
        contrasts = pair_contrasts(resolving_cells, pair)
        informative = sum(1 for c in contrasts if c["informative"])
        material = {
            str(alpha): sum(1 for c in contrasts if c["material_at"][str(alpha)])
            for alpha in ALPHAS
        }
        for alpha in ALPHAS:
            material_total[str(alpha)] += material[str(alpha)]
        informative_total += informative

        selected = select_for_profiles(contrasts)
        selected_ids = set(id(c) for c in selected)
        for contrast in contrasts:
            if id(contrast) in selected_ids:
                contrast["by_x"] = block_profile(cells, contrast, pair)
                contrast["argument_dependence"] = profile_summary(contrast["by_x"])
                contrast["selected_for_by_x"] = True
            contrast.pop("keys", None)

        pointwise = [
            c["pointwise_max_abs_interaction"]
            for c in contrasts
            if "pointwise_max_abs_interaction" in c
        ]
        rel = [c["relative"] for c in contrasts if c["relative"] is not None]
        sign_changes = sum(
            1 for c in selected if c.get("argument_dependence", {}).get("sign_changes")
        )

        pairs.append(
            {
                "pair": list(pair),
                "contrasts": contrasts,
                "informative": informative,
                "material": material,
                "material_at_primary": material[str(PRIMARY_ALPHA)],
                "max_abs_interaction": max(
                    (abs(c["interaction"]) for c in contrasts), default=0.0
                ),
                "max_relative": max(rel, default=0.0),
                "pointwise_max_abs_interaction": max(pointwise, default=0.0),
                "profiled_contrasts": len(selected),
                "profiled_sign_changes": sign_changes,
            }
        )

    # Secondary: the same contrasts on the store-dominated lanes, labelled and
    # kept out of the verdict. A material contrast here is the store quantum
    # interacting with the axes and the argument, not the axis terms failing to
    # compose; the half lanes have no resolution for a 1e-7 contrast.
    store_pairs = []
    store_material_total = {str(alpha): 0 for alpha in ALPHAS}
    store_informative_total = 0
    if store_cells:
        for pair in itertools.combinations(AXES, 2):
            contrasts = pair_contrasts(store_cells, pair)
            informative = sum(1 for c in contrasts if c["informative"])
            material = {
                str(alpha): sum(
                    1 for c in contrasts if c["material_at"][str(alpha)]
                )
                for alpha in ALPHAS
            }
            for alpha in ALPHAS:
                store_material_total[str(alpha)] += material[str(alpha)]
            store_informative_total += informative
            rel = [c["relative"] for c in contrasts if c["relative"] is not None]
            store_pairs.append(
                {
                    "pair": list(pair),
                    "informative": informative,
                    "material": material,
                    "max_relative": max(rel, default=0.0),
                }
            )

    # Multi-way: within each host lane over the six call axes. Every lane is
    # decomposed and reported; only the resolving lanes feed the verdict.
    within = {}
    for precision in LEVELS["precision"]:
        block = multiway(cells, {"precision": precision}, WITHIN, ALPHAS)
        block["used_in_verdict"] = precision in RESOLVING_LANES
        block["role"] = (
            "resolving: the published figure is the axis sum and the returned "
            "value resolves it"
            if precision in RESOLVING_LANES
            else "store-dominated: the published figure includes the value's "
            "half digit, which is the dominant term and leaves no resolution "
            "for an axis contrast"
        )
        if block["points"] == 0:
            within[precision] = block
            continue
        targets = block["top_components"]
        profiles = component_profiles(cells, block, targets)
        for target in targets:
            target["by_x"] = profiles.get(str(target["component_index"]), [])
            target["argument_dependence"] = profile_summary(target["by_x"])
        for field in ("_metadata", "_points", "_keys"):
            block.pop(field, None)
        within[precision] = block

    # Multi-way across lanes on exact arguments. The primary block carries the
    # two resolving lanes; the four-lane block is reported beside it and kept
    # out of the verdict, because precision then changes the store format too
    # and every precision contrast is a change of instrument.
    exact_cells = {}
    for key, points in cells.items():
        exact_points = {p: r for p, r in points.items() if r["kind"] == "exact"}
        if exact_points:
            exact_cells[key] = exact_points
    cross = multiway(
        exact_cells, {}, AXES, ALPHAS, {"precision": RESOLVING_LANES}
    )
    cross["used_in_verdict"] = True
    cross["role"] = "resolving lanes, exact arguments"
    for field in ("_metadata", "_points", "_keys"):
        cross.pop(field, None)

    cross_all = multiway(exact_cells, {}, AXES, ALPHAS)
    cross_all["used_in_verdict"] = False
    cross_all["role"] = (
        "all four lanes: the half lanes' store quantum is in the response, and "
        "a precision contrast changes the store format by construction"
    )
    for field in ("_metadata", "_points", "_keys"):
        cross_all.pop(field, None)

    def interaction_material(block):
        totals = {str(alpha): 0 for alpha in ALPHAS}
        if block.get("points", 0) == 0:
            return totals
        for order, stat in block["orders"].items():
            if int(order) < 2:
                continue
            for alpha in ALPHAS:
                totals[str(alpha)] += stat["material_points"][str(alpha)]["all"]
        return totals

    material_multway_by_alpha = {str(alpha): 0 for alpha in ALPHAS}
    store_material_multway = {str(alpha): 0 for alpha in ALPHAS}
    for precision, block in within.items():
        totals = interaction_material(block)
        target = (
            material_multway_by_alpha
            if block.get("used_in_verdict")
            else store_material_multway
        )
        for alpha in ALPHAS:
            target[str(alpha)] += totals[str(alpha)]
    totals = interaction_material(cross)
    for alpha in ALPHAS:
        material_multway_by_alpha[str(alpha)] += totals[str(alpha)]
    # The all-lane cross block includes the resolving lanes too, so it is
    # reported on its own and not added to the store-dominated totals.
    cross_all_material = interaction_material(cross_all)

    material_multway = material_multway_by_alpha[str(PRIMARY_ALPHA)]

    verdict_by_alpha = {}
    for alpha in ALPHAS:
        key_alpha = str(alpha)
        if violations:
            value = "bound-violated"
        elif material_total[key_alpha] > 0 or material_multway_by_alpha[key_alpha] > 0:
            value = "interaction-material"
        elif informative_total == 0:
            value = "inconclusive"
        else:
            value = f"additive-within-{int(round(alpha * 100))}pct"
        verdict_by_alpha[key_alpha] = value
    verdict = verdict_by_alpha[str(PRIMARY_ALPHA)]

    liveness = flag_liveness(cells)
    liveness["axis_value_changes"] = axis_liveness(cells, precisions_present)

    summary = {
        "schema": 3,
        "materiality": {
            "primary_alpha": PRIMARY_ALPHA,
            "alphas": ALPHAS,
            "pairwise_definition": (
                "|I| > alpha * max(E00,E10,E01,E11), informative contrasts only"
            ),
            "multiway_definition": (
                "a component's or an interaction order's share of the point's "
                "centred total sum of squares > alpha; the mean (order 0) is "
                "excluded from the centred total"
            ),
            "definition": (
                "pairwise: |I| > alpha * max(E00,E10,E01,E11), informative only; "
                "multi-way: share of the point's centred total sum of squares > "
                "alpha, with the mean excluded"
            ),
            "rationale": (
                "10% is a pre-registered engineering materiality margin, not a "
                "significance level. The pairwise margin is in response units: "
                "an interaction of one tenth of the largest response in the "
                "contrast, far above the double-double reference's resolution. "
                "The multi-way margin is a variance share, bounded by 1 and not "
                "inflated by the number of factor levels or cells; the "
                "orthonormal coefficient is reported as a descriptive effect but "
                "is not the materiality measure. The 5% and 20% columns make "
                "both choices transparent."
            ),
        },
        "counts": {
            "data_rows": total_rows,
            "cells": len(cells),
            "expected_cells": expected_cells,
            "design_cells": design_cells,
            "precisions_present": precisions_present,
            "resolving_lanes": resolving_lanes,
            "store_dominated_lanes": store_lanes,
            "resolving_cells": len(resolving_cells),
            "store_dominated_cells": len(store_cells),
            "refusals": len(refusals),
            "informative_contrasts": informative_total,
            "material_contrasts": material_total[str(PRIMARY_ALPHA)],
            "material_contrasts_by_alpha": material_total,
            "material_multway_readings": material_multway,
            "material_multway_readings_by_alpha": material_multway_by_alpha,
            "rows_by_argument_kind": kind_counts,
        },
        "refusals": refusals[:20],
        "safety": {
            "definition": (
                "figure_ratio = |err| / limit, where limit is the lane row's "
                "whole statement: the accessor's figure plus, on the half "
                "lanes, half of the last representable digit of the returned "
                "value. accessor_ratio = |err| / figure, reported apart: on a "
                "half lane it exceeds 1 wherever the digit binds, which is the "
                "row's shape and not a violation"
            ),
            "max_figure_ratio": max_ratio,
            "max_figure_ratio_at": max_ratio_at,
            "max_accessor_ratio": max_accessor_ratio,
            "max_accessor_ratio_at": max_accessor_ratio_at,
            "accessor_only_exceeded": accessor_only_exceeded,
            "by_lane": lane_safety,
            "violations": violations[:20],
            "violation_count": len(violations),
        },
        "cells": cell_list,
        "pairs": pairs,
        "store_dominated": {
            "lanes": store_lanes,
            "informative_contrasts": store_informative_total,
            "material_contrasts_by_alpha": store_material_total,
            "material_multway_readings_by_alpha": store_material_multway,
            "pairs": store_pairs,
            "note": (
                "reported, not verdict-bearing. The half lanes' returned value "
                "carries a store quantum three to four orders above the axis "
                "terms; a contrast here measures that quantum's interaction "
                "with the axes and the argument, not the composition of the "
                "axis bounds"
            ),
        },
        "liveness": liveness,
        "multiway": {
            "within_lane": within,
            "cross_lane_exact": cross,
            "cross_lane_exact_all_lanes": cross_all,
            "cross_lane_exact_all_lanes_material_by_alpha": cross_all_material,
        },
        "verdict_by_alpha": verdict_by_alpha,
        "verdict": verdict,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"cells: {len(cells)} of {expected_cells} for lanes "
          f"{','.join(precisions_present)} (design {design_cells}), rows: {total_rows}, "
          f"refusals: {len(refusals)}")
    print(f"argument kinds: {kind_counts}")
    print(f"max figure ratio: {max_ratio:.6g} at {max_ratio_at}")
    print(f"max accessor ratio: {max_accessor_ratio:.6g} at {max_accessor_ratio_at}; "
          f"{accessor_only_exceeded} rows exceed the accessor figure alone and are "
          f"inside the lane row's limit")
    print(f"informative pairwise contrasts: {informative_total}")
    for alpha in ALPHAS:
        print(f"  material at {int(alpha * 100)}%: {material_total[str(alpha)]} "
              f"(store-dominated lanes: {store_material_total[str(alpha)]})")
    print(f"multi-way material readings at 10%: {material_multway} "
          f"(store-dominated lanes: {store_material_multway[str(PRIMARY_ALPHA)]})")
    print(f"verdict by margin: "
          + ", ".join(f"{round(alpha * 100)}% {verdict_by_alpha[str(alpha)]}"
                      for alpha in ALPHAS))
    print(f"verdict: {verdict}")
    for precision, block in within.items():
        if block.get("points", 0) == 0:
            continue
        role = "" if block.get("used_in_verdict") else " [store-dominated]"
        print(f"  {precision}: {block['points']} points, order shares "
              + ", ".join(
                  f"{order}: {stat['ss_share']:.3g}"
                  for order, stat in block["orders"].items()
                  if int(order) >= 2 and stat["ss_share"] is not None
              ) + role)
    if cross.get("points", 0) > 0:
        print(f"  cross-lane exact: {cross['points']} points, order shares "
              + ", ".join(
                  f"{order}: {stat['ss_share']:.3g}"
                  for order, stat in cross["orders"].items()
                  if int(order) >= 2 and stat["ss_share"] is not None
              ))
    print(f"verdict: {verdict}")
    print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
