# Per-lane accuracy and limitations

Each lane's budget is tabulated in [the README](https://github.com/myamlak/boys#accuracy-contract)
and in the header preamble. This page gives what a table of budgets cannot: **which arguments each
bound holds over, where each lane stops, and what sets its accuracy ceiling.**

A lane is a range, not a number. Every lane carries a compile-time or per-call multiplier `m`, from
1 to 65536. Raising it loosens the budget and reduces the work together.

## double — scalar and batch

**Bounds.** 1e-15 on region A, 3e-14 on the extended band and region B, and 5.5e-14 on region C for
the scalar entry. The batch entry is 5.5e-14 in every region.

**Ceiling: coefficient rounding.** The stored coefficients are correctly rounded doubles. The fit's
own truncation error is 5e-19, while the doubles holding it carry 1e-16 to 2e-16. The polynomial is
sixty times better than the format that stores it, and nothing below that floor is reachable. **A
caller holding the `m = 1` result holds the most accurate double this library produces.**

## float — single and batch

**Bounds.** 1.5e-7 absolute at `m = 1`, the same in every region, with the same multipliers. The
batch entry seeds in double and recurses in single.

**Ceiling: the format.** A 24-bit significand resolves about 6e-8 relative. The `m = 1` budget is
1.5e-7 absolute. For values of order one those two are within a factor of a few of each other, so
the format has no rung left below the budget.

## half — fp16 and bf16, storing half and computing wide

**Bounds.** `m·1e-7 + ½ ULP`. Half takes no part in the arithmetic. The argument is rounded to the
half type, the float engine evaluates at that value, and the result is rounded back.

**The margin is a fragility, not a cushion.** The worst cell sits at **0.9999 of budget**: a margin
of thousandths of a per cent, not a per cent. At that distance the ½ ULP representation term is 99%
of the budget. One half-quantum of extra error anywhere in the lane therefore takes a cell over it —
a differently rounded argument, or an engine one rounding worse. Treat this lane as inside its bound
with nothing to spare.

**Ceiling: the representation term.** It is independent of `m`. Eleven significand bits resolve about
five parts in ten thousand, and the budget already spends half of one such step on representing the
result. Raising `m` loosens only the engine's share.

**Past the ceiling the lane claims nothing.** With `u` the format's quantum at the returned value,
the bound holds where `|F_n(x)| > m·1e-7 + ½u`, and nowhere else. Below that the return is subnormal,
then exactly zero.

In fp16 the ceiling and the format's floor coincide. **From order 3 upward on region C, every
argument of the branch is past it.** At high orders and large arguments this lane therefore returns
zero rather than a value. A caller who needs those arguments wants the float or double lane. bf16's
exponent range matches float's, so bf16 keeps returning usable values down to about 1e-38.

## packed half — the native lane

**Bounds.** Region C only, from x = 28.984375 upward, returning `2^15·F_k(x)`: **within 8 ULP of the
returned value**, worst measured 4.243 ULP. The bound is in ULP rather than a region budget, because
ULP is the shape of this lane's error. There is no multiplier.

**It is not a drop-in for the half lane above.** It rounds once per operation, where that lane rounds
once per value. It therefore spends about 7.7× the half-quantum of representation that lane's budget
leaves it. That is a floor set by eleven mantissa bits, not by the algorithm. The steps that pay it
are the low orders callers actually use.

**Its arithmetic is portable, not instruction-backed.** Each operation is evaluated in binary64 and
rounded once to binary16. That gives the same correctly rounded half a packed-half instruction would
produce. This library carries no packed-half backend, so what the lane delivers is half arithmetic's
*numbers* rather than its *speed*. **No timing is claimed for it.**

**Ceiling: the exponent span.** The bound holds where the returned value is a normal half,
`F_k(x) ≥ 2^-29`. Above that the return is subnormal, then exactly zero, by design. Order 8 reaches
that crossing at about x = 30, order 4 at about x = 128, and order 3 at about x = 359. At orders 0
and 1 it is the whole of binary16's argument range.

A per-order rescale buys range and no accuracy. The rescale is exact, so the scaled lane is
bit-identical to the unscaled one. No scale reaches past the wall either, because the ladder's own
span grows like x to the n against a finite exponent range.

## Region C, and where the boundary sits

Region C is a single closed form with no coefficients to truncate. **No multiplier relaxes it**, and
its budget holds with slack at every rung.

Its accuracy is set by its lower boundary. Moving that boundary is an accuracy knob but **not a
performance tier**: the branch costs the same wherever it starts, so moving it saves no work.

Below about x = 16 the branch sits outside its domain of validity. There the error grows smoothly,
with no threshold at the edge — consecutive arguments differ by at most 0.6 of an order. Its **size**
is real: relative error reaches 5.6e4 at order 32 just below the edge, five to eight orders past the
branch's own 5.5e-14 budget. The branch is not wrong to be there. The boundary is simply where its
validity ends, and a caller evaluating below it should expect that error.

## What is not claimed

**Throughput.** No timing taken so far supports a performance claim for any lane. Every one comes
from a loaded machine with no tensor cores, and with a double-to-single throughput ratio near 1:32.
**The performance question is open.** Answering it needs a timed pass on an unloaded machine and on
a card chosen to show the difference; the packed-half advantage in particular is a property of the
card. Until such a pass exists, no lane's cost may be assumed better than another's.

**Tensor cores.** Not reachable from the packed lane, and the reason is shape rather than precision.
Its work is a rank-one recurrence, one multiply and one divide per order, so there is no product of
two matrices for a tensor core to take. The one matrix-shaped step, the region-A transform, does not
carry the accuracy lanes with reduced-precision operands. **The ceiling of this technique is packed
half arithmetic's, and no tensor-core path routes around it.**

Every accuracy figure on this page was measured from the code in this repository.
