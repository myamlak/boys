"""CPU simulation of tensor-core arithmetic modes.

A mode is fixed by two things and nothing else: the format each operand is
rounded to on entry, and the precision the products are accumulated in. With
no tensor-core hardware available, that pair is simulated exactly:

  * operands are rounded to the mode's input format by round-to-nearest-even
    on the significand, with the format's exponent range enforced (fp16's
    subnormals and overflow are real, so fp16 goes through numpy's own cast);
  * the product of two such operands is computed exactly and the accumulation
    rounds once per term into the accumulate format - a fused multiply-add,
    which is what a tensor core's datapath does.

Rounding a significand to ``sig`` bits is exact and correctly rounded: frexp
splits the value into mantissa and exponent, scaling by a power of two is
exact, rint is round-half-to-even, and ldexp is exact.

Significand widths (implicit bit included), which are the standard ones:

  fp64  53     tf32  11 (8-bit exponent, fp32's range)
  fp32  24     bf16   8 (8-bit exponent, fp32's range)
  fp16  11 (5-bit exponent: subnormals below 2^-14, overflow above 65504)
"""
import numpy as np

# ---------------------------------------------------------------------------
# format rounding
# ---------------------------------------------------------------------------
FMT = {
    "fp64": dict(sig=53, kind="wide", u=2.0 ** -53),
    "fp32": dict(sig=24, kind="wide", u=2.0 ** -24),
    "tf32": dict(sig=11, kind="wide", u=2.0 ** -11),
    "bf16": dict(sig=8, kind="wide", u=2.0 ** -8),
    "fp16": dict(sig=11, kind="half", u=2.0 ** -11),
}


def round_sig(x, sig):
    """Round to ``sig`` significant bits, round-half-to-even."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    scaled = np.rint(m * float(1 << sig)) / float(1 << sig)
    return np.ldexp(scaled, e)


def to_format(x, fmt):
    f = FMT[fmt]
    if f["kind"] == "half":
        return np.asarray(np.asarray(x, dtype=np.float64).astype(np.float16), dtype=np.float64)
    return round_sig(x, f["sig"])


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
# split d = 1 -> one operand part per matrix (a plain low-precision GEMM);
# split d > 1 -> each operand is carried as d parts, and every product whose
# part indices sum to <= d-1 is accumulated (d=2 keeps 3 of the 4 products,
# which is the standard 3xTF32 emulation of an fp32 GEMM).
MODES = {
    # --- the shipped lanes' own transform arithmetic, as calibration ---
    "fp64":        dict(fmt="fp64", acc="fp64", split=1,
                        note="fp64 operands, fp64 accumulate (A100/H100 class DMMA)"),
    "fp32":        dict(fmt="fp32", acc="fp32", split=1,
                        note="fp32 operands, fp32 accumulate (the float lane's own arithmetic)"),
    # --- data-centre modes, in the order the question names them ---
    "tf32":        dict(fmt="tf32", acc="fp32", split=1,
                        note="tf32 operands (11-bit significand), fp32 accumulate"),
    "tf32x3":      dict(fmt="tf32", acc="fp32", split=2, pre="fp32",
                        note="3xTF32: fp32 operands split hi+lo in tf32, 3 products, fp32 accumulate"),
    "bf16":        dict(fmt="bf16", acc="fp32", split=1,
                        note="bf16 operands (8-bit significand), fp32 accumulate"),
    "bf16x3":      dict(fmt="bf16", acc="fp32", split=2, pre="fp32",
                        note="bf16 2-part split of an fp32 operand, 3 products, fp32 accumulate"),
    "bf16x6":      dict(fmt="bf16", acc="fp32", split=3, pre="fp32",
                        note="bf16 3-part split of an fp32 operand, 6 products, fp32 accumulate"),
    "fp16":        dict(fmt="fp16", acc="fp32", split=1,
                        note="fp16 operands (11-bit significand), fp32 accumulate"),
    "fp16acc16":   dict(fmt="fp16", acc="fp16", split=1,
                        note="fp16 operands, fp16 accumulate (consumer mode)"),
}
# Deeper splits of an fp32 operand: the same emulation pushed past the point
# where the operand's own 24 bits, not the split depth, is the limit.
for _d in (3, 4):
    MODES[f"tf32x{2 * _d - 1}"] = dict(fmt="tf32", acc="fp32", split=_d, pre="fp32", note="")
    MODES[f"tf32x{2 * _d - 1}_fp64acc"] = dict(fmt="tf32", acc="fp64", split=_d, pre="fp32", note="")
MODES["tf32x3_fp64acc"] = dict(fmt="tf32", acc="fp64", split=2, pre="fp32", note="")
# Splits of an fp64 operand (d parts of 11 bits => 11d bits of the coefficient
# matrix): the test of whether a sub-fp64 mode can carry a double-precision
# coefficient table at all, held against each accumulate format.
for _d in (2, 3, 4, 5, 6):
    MODES[f"tf32_{_d}part_fp64in_fp64acc"] = dict(fmt="tf32", acc="fp64", split=_d, note="")
    MODES[f"tf32_{_d}part_fp64in_fp32acc"] = dict(fmt="tf32", acc="fp32", split=_d, note="")
# Coefficient-storage probes: the coefficient matrix alone is rounded to the
# format, the basis matrix and the accumulation stay at fp64 - so the number
# is what storing the coefficients at that precision costs, and nothing else.
for _f in ("tf32", "bf16", "fp16", "fp32"):
    MODES[f"coeffonly_{_f}"] = dict(fmt=_f, acc="fp64", split=1, rnd="C", note="")


def split_parts(x, d, fmt, pre=None):
    """d parts, each in ``fmt``: p0 = rnd(x), p1 = rnd(x - p0), ...

    ``pre`` names a format the value passes through before the split: an fp32
    GEMM emulated by 3xTF32 splits fp32 operands, not fp64 ones."""
    parts = []
    rest = to_format(x, pre) if pre else np.asarray(x, dtype=np.float64)
    for _ in range(d):
        p = to_format(rest, fmt)
        parts.append(p)
        rest = rest - p
    return parts


def gemm(C, T, mode_name, reduction="running"):
    """C (orders x K) times T (K x S) in the named mode's arithmetic.

    ``rnd`` selects which operand is rounded to the mode's format: "both" is
    the hardware, "C" rounds the coefficient matrix alone (which isolates what
    storing the coefficients at that precision costs, with the basis and the
    accumulation held at fp64).

    ``reduction`` is the degree loop's order and is a layout decision rather
    than a property of a card: "running" sums the degrees into one running
    total from the constant term up, "pairwise" lays the degree out high to
    low and reduces it pairwise, folding one pass per operand part pairing
    into the total. The shipped kernel is "pairwise"; the two differ by about
    three times at fp64 and at the split modes' formats."""
    m = MODES[mode_name]
    fmt, accd, d = m["fmt"], m["acc"], m["split"]
    rnd, pre = m.get("rnd", "both"), m.get("pre")
    acc = np.zeros((C.shape[0], T.shape[1]), dtype=np.float64)
    acc_t = {"fp64": np.float64, "fp32": np.float32, "fp16": np.float16}[accd]
    cs = split_parts(C, d, fmt, pre) if rnd != "T" else [np.asarray(C, dtype=np.float64)]
    ts = split_parts(T, d, fmt, pre) if rnd != "C" else [np.asarray(T, dtype=np.float64)]
    K = C.shape[1]
    for i, ap in enumerate(cs):
        for j, bp in enumerate(ts):
            if i + j > d - 1:
                continue
            if reduction == "running":
                for k in range(K):
                    prod = ap[:, k, None] * bp[k, None, :]   # exact in fp64 for sig <= 11
                    acc = (acc + prod).astype(acc_t).astype(np.float64)
                continue
            # pairwise: one pass per pairing, the degree laid out from the
            # highest down, so the low coefficients are summed among themselves
            # before any of them meets the scale of the high ones.
            layer = [(ap[:, K - 1 - k, None] * bp[K - 1 - k, None, :]).astype(acc_t).astype(np.float64)
                     for k in range(K)]
            width = K
            while width > 1:
                for k in range(width // 2):
                    layer[k] = (layer[2 * k] + layer[2 * k + 1]).astype(acc_t).astype(np.float64)
                if width % 2 != 0:
                    layer[width // 2] = layer[width - 1]
                width = (width + 1) // 2
            acc = (acc + layer[0]).astype(acc_t).astype(np.float64)
    return acc


def product_count(mode_name, K):
    """How many scalar products the mode performs per (order, argument)."""
    d = MODES[mode_name]["split"]
    return K * sum(1 for i in range(d) for j in range(d) if i + j <= d - 1)
