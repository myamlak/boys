# Accuracy: what each lane guarantees

This library evaluates the Boys function `F_n(x) = ∫₀¹ t^(2n) e^(−x t²) dt`, for orders n = 0 to 32
and arguments x ≥ 0.

It offers several lanes, which differ in precision and in cost. This page states each lane's error
bound, the arguments the bound covers, and where the lane stops being usable.

## How the argument affects the bound

The library evaluates the function differently at different argument sizes, and the error bound
differs with it:

| Arguments | How they are evaluated |
|---|---|
| x < 11.899848152108484 | stored polynomial fits, one per order |
| 11.899848152108484 ≤ x < 28.98933773882074 | a stored fit at the lowest order, then upward recursion |
| x ≥ 28.98933773882074 | a closed-form asymptotic result |

Below about x = 1.0855 all orders are inside their own fits. Between that and x = 11.8998, the
higher orders are served by recursion from a single fit, which is slightly less accurate.

Every bound in this document holds for **all** x ≥ 0. Where a lane is tighter over part of the
range, its entry says so.

## double

**At most 5.5e-14 everywhere.** At most 3e-14 for x below 11.899848152108484, and at most 1e-15 for
x below about 1.0855.

A multiplier, set at compile time or per call, runs from 1 to 65536. Raising it loosens the bound
and reduces the work.

**This is the most accurate double the library produces.** The stored coefficients are rounded to
double precision, and they hold only the 16 or so significant digits that a double can hold. The
mathematical fit behind them is about sixty times more accurate than the numbers used to store it,
so no accuracy below roughly 1e-16 is reachable, however the evaluation is arranged.

## float

**At most 1.5e-7 everywhere.**

The same multiplier applies. The batch entry computes a starting value in double precision and then
recurses in single.

**The limit is the number format.** A 32-bit float carries about seven significant digits, which is
about 6e-8 relative. The bound is 1.5e-7 absolute, so for values of order one the two are within a
factor of a few of each other. The format has no room left to give.

## half — storing 16-bit values and computing in 32-bit

**At most `m·1e-7` plus half of the last representable digit of the result**, where `m` is the same
multiplier as above.

Half precision takes no part in the arithmetic. The argument is rounded to 16 bits, the 32-bit
evaluation runs at that value, and the result is rounded back to 16 bits.

**The margin is thin, and that is a warning rather than a cushion.** The worst case sits at 0.9999 of
the bound — a margin of thousandths of a per cent, not a per cent. At that distance, half of one
representable digit of the result is 99% of the whole bound. One extra step of error anywhere in the
lane therefore pushes a result over its bound. Treat this lane as inside its bound with nothing to
spare.

**The limit is the 16-bit format, and it does not improve with the multiplier.** Sixteen bits carry
about eleven significant bits, or five parts in ten thousand, and the bound already spends half of
one such step on representing the result. Raising the multiplier loosens only the part that comes
from the 32-bit arithmetic.

**Below a certain size the lane returns nothing usable.** The bound holds only where the result is
larger than `m·1e-7` plus half a representable digit. Smaller than that, the returned value becomes a
subnormal 16-bit number, and then exactly zero.

For 16-bit floats the two thresholds coincide, and **from order 3 upward, at arguments at or above
x = 28.98933773882074, every argument is past it.** At high orders and large arguments this lane
therefore returns zero rather than a value, and a caller who needs those arguments wants the float or
double lane. The 16-bit brain-float format has the same exponent range as a 32-bit float, so it keeps
returning usable values down to about 1e-38.

## packed half — the native lane

**Within 8 of the last representable digits of the returned value**, worst measured 4.243, for
arguments at or above x = 28.984375. The bound is stated in representable digits rather than as a
size, because that is the shape of this lane's error. There is no multiplier.

**It is not a drop-in for the half lane above.** It rounds once per calculation step, where that lane
rounds once per value. It therefore loses about 7.7 times as much as the half lane's bound allows for
representing its result — a floor set by having only eleven significant bits, not by the algorithm.
The steps that pay for it are the low orders, which are the ones callers use most.

**The arithmetic is portable rather than hardware-backed.** Each step is evaluated in double
precision and rounded once to 16 bits, which gives the same correctly rounded result a native 16-bit
instruction would. This library carries no native 16-bit backend, so what the lane delivers is 16-bit
arithmetic's *results* rather than its *speed*. **No timing is claimed for it.**

**The limit is the exponent range.** The bound holds where the returned value is a normal 16-bit
number, that is where it is at least 2^-29. Smaller than that, the return is a subnormal, then
exactly zero, by design. Order 8 reaches that point at about x = 30, order 4 at about x = 128, and
order 3 at about x = 359. At orders 0 and 1 it is the whole of the 16-bit argument range.

Rescaling each order buys range and no accuracy. The rescaling is exact, so a rescaled lane returns
bit-identical results to an unscaled one. No rescaling reaches past that limit either, because the
range the recursion spans grows like x to the power n, against a fixed exponent range.

## Large arguments, and the boundary at x = 11.8998 and x = 28.9893

For x at or above 28.98933773882074 the library uses a single closed-form asymptotic result, with no
stored coefficients. **No multiplier relaxes it**, and its bound holds with room to spare at every
setting.

Its accuracy is set by where the asymptotic form starts being used. Moving that start point trades
accuracy against nothing: the evaluation costs the same wherever the boundary sits, so moving it
saves no work.

Below about x = 16 the asymptotic form is outside the range where it is valid, and the error grows
smoothly there, with no sudden jump. Consecutive arguments differ by at most 0.6 of an order of
magnitude. But the size of the error is real: just below x = 16, at order 32, the relative error
reaches 5.6e4, which is five to eight orders of magnitude past the 5.5e-14 bound. The form is not
wrong to be used there; that is simply where its validity ends, and a caller evaluating below x = 16
should expect that error.

## What is not claimed

**Speed.** No timing taken so far supports a speed claim for any lane. Every timing available was
taken on a loaded machine, on a card with no tensor cores, and with a double-to-single throughput
ratio near 1 to 32. **Performance is an open question.** Answering it needs a timed run on an
unloaded machine and on a card chosen to show the difference; the packed-half lane's advantage in
particular is a property of the card. Until then, no lane's cost may be assumed better than
another's.

**Tensor cores.** The packed-half lane cannot use them, and the reason is its shape rather than its
precision. Its work is a chain of one multiply and one divide per order, so there is no product of
two matrices for a tensor core to compute. The one step that *is* matrix-shaped, the stored
polynomial fits, does not stay accurate with reduced-precision inputs. **The limit of this lane is
native 16-bit arithmetic's, and no tensor-core path avoids it.**

Every accuracy figure on this page was measured from the code in this repository.
