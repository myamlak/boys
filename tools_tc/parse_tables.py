"""Parse the shipped Chebyshev tables out of the generated coefficients header.

Reads the generated header as data (never imports the library). Returns the
region-A piece structure for the double and float lanes: per order, the list
of (a, b, deg, coeffs).
"""
import os
import re

HEADER = "include/boys/boys_coefficients.hpp"
DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, HEADER)


def _num_list(text):
    return [float(tok.rstrip("f")) for tok in re.findall(r"[-+0-9.eE]+f?", text)]


def _split_namespaces(src):
    """Splits the header at the float-lane namespace, returning (double_src, float_src)."""
    marker = "namespace boys::detail::f32"
    i = src.index(marker)
    return src[:i], src[i:]


def _parse_lane(src):
    coeffs = _num_list(re.search(r"kCoeffs\s*=\s*std::to_array<[^>]+>\(\{(.*?)\}\);", src, re.S).group(1))
    pieces_raw = re.search(r"kPieces\s*=\s*std::to_array<OrderPiece>\(\{(.*?)\}\);", src, re.S).group(1)
    pieces = []
    for line in pieces_raw.splitlines():
        m = re.match(r"\s*\{([^,]+),([^,]+),([^,]+),([^}]+)\}", line)
        if not m:
            continue
        a, b, deg, off = m.groups()
        pieces.append((float(a.rstrip("f")), float(b.rstrip("f")), int(deg), int(off.rstrip("f"))))
    starts = [int(t) for t in re.findall(r"[-0-9]+", re.search(r"kPieceStart\s*=\s*std::to_array<int>\(\{(.*?)\}\);", src, re.S).group(1))]
    bcoeffs = _num_list(re.search(r"kBcoeffs\s*=\s*std::to_array<[^>]+>\(\{(.*?)\}\);", src, re.S).group(1))
    bdeg = int(re.search(r"kBDeg = (\d+);", src).group(1))
    m = re.search(r"kExtendedBcoeffs\s*=\s*std::to_array<[^>]+>\(\{(.*?)\}\);", src, re.S)
    ext = _num_list(m.group(1)) if m else []
    m = re.search(r"kExtendedBDeg = (\d+);", src)
    extdeg = int(m.group(1)) if m else -1
    m = re.search(r"kExtendedBX0 = ([0-9.eE+-]+);", src)
    extx0 = float(m.group(1)) if m else float("nan")
    m = re.search(r"kTierThresholds\s*=\s*std::to_array<double>\(\{(.*?)\}\);", src, re.S)
    thresholds = _num_list(m.group(1)) if m else []
    orders = []
    for n in range(len(starts) - 1):
        ps = []
        for (a, b, deg, off) in pieces[starts[n]:starts[n + 1]]:
            ps.append((a, b, deg, coeffs[off:off + deg + 1]))
        orders.append(ps)
    kx0 = float(re.search(r"kX0 = ([0-9.eE+-]+);", src).group(1)) if re.search(r"kX0 = ([0-9.eE+-]+);", src) else 11.899848152108484
    kx1 = float(re.search(r"kX1 = ([0-9.eE+-]+);", src).group(1)) if re.search(r"kX1 = ([0-9.eE+-]+);", src) else 28.989337738820740
    return dict(orders=orders, B=(bdeg, bcoeffs), ext=(extdeg, ext, extx0),
                thresholds=thresholds, kx0=kx0, kx1=kx1)


def load(path=DEFAULT):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    d_src, f_src = _split_namespaces(src)
    return {"double": _parse_lane(d_src), "float": _parse_lane(f_src)}


if __name__ == "__main__":
    t = load()
    for lane in ("double", "float"):
        L = t[lane]
        print(f"=== {lane} lane: kX0={L['kx0']} kX1={L['kx1']} "
              f"regionB deg={L['B'][0]} extended deg={L['ext'][0]} "
              f"extended x0={L['ext'][2]}")
        degs = {}
        for n, ps in enumerate(L["orders"]):
            degs.setdefault(len(ps), []).append(n)
        print(f"  piece counts per order: " +
              ", ".join(f"{k} pieces: {len(v)} orders ({v[0]}..{v[-1]})"
                        for k, v in sorted(degs.items())))
        n0 = 0
        print(f"  F0 pieces: " + ", ".join(f"[{a:.6f},{b:.6f}] deg{d}" for a, b, d, _ in L["orders"][0]))
        print(f"  F32 pieces: " + ", ".join(f"[{a:.6f},{b:.6f}] deg{d}" for a, b, d, _ in L["orders"][32]))
        from collections import Counter
        alldeg = [d for ps in L["orders"] for (_, _, d, _) in ps]
        print(f"  degree histogram (all region-A pieces): {dict(sorted(Counter(alldeg).items()))}")
        print(f"  region-A piece count: {len(alldeg)}")
        print(f"  total region-A coefficients per order: " +
              str([sum(len(c) for _, _, _, c in ps) for ps in L['orders']]))
