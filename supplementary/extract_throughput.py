#!/usr/bin/env python3
"""tab:throughput cells from the recorded runs logs.

Extracts the recorded median items/s per kernel from the two runs logs
(runs/boys-cpu-*.txt, runs/boys-gpu-*.txt) and checks every cell and every
caption ratio of the accompanying manuscript's tab:throughput against the
recorded rows. The paper cell is the median of the three recorded passes;
the comparison is at the cell's stated precision (2 significant digits).
The logs report M/s; the manuscript cells are items/s — the script
converts (1 M/s = 1e6 items/s).

Usage:  python extract_throughput.py
Requires: Python 3 only.
"""

import glob
import math
import os
import re


def load_medians():
    """{(file, kernel, workload): median_Mvals_per_s} from the runs logs."""
    medians = {}
    pattern = re.compile(
        r"^kernel: ([^ ]+) \| workload: ([^|]+?)\s*\|.*median_Mvals_per_s: ([\d.]+)")
    for path in sorted(glob.glob(os.path.join("runs", "boys-*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    kernel, workload, value = m.groups()
                    medians[(os.path.basename(path), kernel, workload.strip())] = float(value)
    return medians


def round_n(value, nsig):
    """Round to nsig significant digits (the cells' stated precision)."""
    if value == 0:
        return 0.0
    exp = math.floor(math.log10(abs(value)))
    scaled = value / 10 ** (exp - nsig + 1)
    return round(scaled) * 10 ** (exp - nsig + 1)


# (device, lane, (uniform cell M/s, nsig), (molecular cell M/s, nsig),
#  uniform (log, kernel, workload), molecular (log, kernel, workload))
# Cells are the manuscript's tab:throughput values in items/s converted to
# M/s (9.8e6 items/s = 9.8 M/s), with each cell's significant-figure count
# as stated in the table (e.g. 1.34e8 = 3 sig figs). None = no cell in
# that column. The GPU "double (best)" row is the gridded-LUT kernel per
# the table caption.
CHECKS = [
    ("CPU", "scalar double", (9.8, 2), (22.0, 2),
     ("boys-cpu-i7-9850h-2026-08-29.txt", "scalar-double", "uniform-n32-x40"),
     ("boys-cpu-i7-9850h-2026-08-29.txt", "scalar-double", "molecular")),
    ("CPU", "scalar float", (11.0, 2), None,
     ("boys-cpu-i7-9850h-2026-08-29.txt", "scalar-f32", "uniform-n32-x40"), None),
    ("CPU", "AVX2 region-sorted", (134.0, 3), None,
     ("boys-cpu-i7-9850h-2026-08-29.txt", "cheb-simd-sorted-n8", "uniform-x40-n8 (region-sorted)"),
     None),
    ("CPU", "AVX2 unsorted", (42.0, 2), None,
     ("boys-cpu-i7-9850h-2026-08-29.txt", "cheb-simd-unsorted-n8", "uniform-x40-n8"), None),
    ("GPU", "double (best)", (96.0, 2), None,
     ("boys-gpu-t1000-2026-08-29.txt", "lut-f64", "uniform-n32-x40"), None),
    ("GPU", "float", (420.0, 2), None,
     ("boys-gpu-t1000-2026-08-29.txt", "cheb-f32", "uniform-n32-x40"), None),
    ("GPU", "fp16 (I/O only)", (180.0, 2), None,
     ("boys-gpu-t1000-2026-08-29.txt", "fp16-single", "uniform-n32-x40"), None),
]

# Caption ratios: (name, label, (caption value, nsig), numerator pin,
# denominator pin)
RATIOS = [
    ("GPU float / GPU double (best)", "4.4x", (4.4, 2),
     ("boys-gpu-t1000-2026-08-29.txt", "cheb-f32"),
     ("boys-gpu-t1000-2026-08-29.txt", "lut-f64")),
    ("CPU AVX2 sorted / CPU scalar double", "13.8x", (13.8, 3),
     ("boys-cpu-i7-9850h-2026-08-29.txt", "cheb-simd-sorted-n8"),
     ("boys-cpu-i7-9850h-2026-08-29.txt", "scalar-double")),
    ("CPU AVX2 sorted / unsorted", "3.2x", (3.2, 2),
     ("boys-cpu-i7-9850h-2026-08-29.txt", "cheb-simd-sorted-n8"),
     ("boys-cpu-i7-9850h-2026-08-29.txt", "cheb-simd-unsorted-n8")),
]


def find(medians, file, kernel):
    """First row of the given (file, kernel) regardless of workload."""
    for (f, k, w), v in medians.items():
        if f == file and k == kernel:
            return (w, v)
    return (None, None)


def main():
    medians = load_medians()
    if len(medians) < 8:
        print("ERROR: expected both runs logs in runs/ "
              "(boys-cpu-i7-9850h-2026-08-29.txt, boys-gpu-t1000-2026-08-29.txt)")
        return 1

    print("tab:throughput regeneration check (recorded medians vs paper cells)")
    print("  recorded rows loaded: %d" % len(medians))
    print()
    failures = 0

    for device, lane, uniform_cell, molecular_cell, uniform_pin, molecular_pin in CHECKS:
        ucell, unsig = uniform_cell
        uvalue = medians.get(uniform_pin)
        ucheck = "PASS" if uvalue is not None and round_n(uvalue, unsig) == ucell else "FAIL"
        if ucheck == "FAIL":
            failures += 1
        line = ("  %s %-18s uniform cell %7.1f M/s -> recorded %7.1f M/s (%s) [%s]"
                % (device, lane, ucell, uvalue if uvalue else float("nan"),
                   uniform_pin[2] if uvalue is not None else "row missing", ucheck))
        if molecular_cell is not None:
            mcell, msig = molecular_cell
            mvalue = medians.get(molecular_pin)
            mcheck = "PASS" if mvalue is not None and round_n(mvalue, msig) == mcell else "FAIL"
            if mcheck == "FAIL":
                failures += 1
            line += (" | molecular cell %7.1f M/s -> recorded %7.1f M/s (%s) [%s]"
                     % (mcell, mvalue if mvalue else float("nan"),
                        molecular_pin[2] if mvalue is not None else "row missing", mcheck))
        print(line)

    print()
    for name, label, caption, num_pin, den_pin in RATIOS:
        cvalue, csig = caption
        _, nvalue = find(medians, num_pin[0], num_pin[1])
        _, dvalue = find(medians, den_pin[0], den_pin[1])
        if nvalue and dvalue:
            ratio = nvalue / dvalue
            check = "PASS" if round_n(ratio, csig) == cvalue else "FAIL"
            if check == "FAIL":
                failures += 1
            print("  caption %-35s %s: recorded %.3f -> [%s]"
                  % (name, label, ratio, check))
        else:
            failures += 1
            print("  caption %-35s %s: recorded rows missing -> [FAIL]"
                  % (name, label))

    print()
    print("RESULT: %s" % ("ALL CELLS MATCH" if failures == 0 else
                          "%d check(s) FAILED" % failures))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
