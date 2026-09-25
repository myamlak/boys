# Choosing a lane

This note is written for a reader who has not built the library yet and has not chosen anything yet:
the words it needs — *lane*, *region*, *rung* — are defined on [the API reference's landing
page](mainpage.md), and what follows is about the choice rather than about the code.

`docs/lane-contract.md` states what each lane guarantees and where it stops. This note is about the
other half of the choice: how much accuracy your calculation actually needs, and where reduced
precision earns its place.

## Your calculation probably does not need this function's accuracy

This function is one part of an integral evaluation, and for almost every calculation it is not the
binding constraint. In production codes:

- integral screening thresholds sit at 1e-10 to 1e-12;
- density fitting carries errors near 1e-6;
- a density-functional grid carries 1e-6 to 1e-8;
- coupled-perturbed convergence has thresholds of its own.

All of those are larger than a 1e-14 error here. **Tightening this function past 1e-14 buys nothing
unless those move too.** If your calculation is limited by something else — and it almost certainly
is — the double lane's default accuracy is already more than enough, and the multiplier is a way to
buy speed rather than a way to buy accuracy you need.

## When you do need the double lane

Reach for it when this function is *recomputed* and its error compounds: a coupled-perturbed solve,
a geometry optimisation run to tight thresholds, or a property that differentiates the energy. Those
are the cases where an error here stays in the answer.

## Where reduced precision earns its place

Half precision cannot serve the accurate path. That is arithmetic rather than opinion: sixteen bits
carry about five parts in ten thousand, and the accuracy these integrals are usually asked for is
about one part in a hundred trillion.

What it *can* serve is work that does not need accuracy.

- **Screening.** Groups of integrals are bounded before they are computed, and the screened-out set
  is far larger than the computed one. A bound wrong by a part in a thousand is harmless if it stays
  conservative.
- **Early iterations of a self-consistent field.** The density is far from converged for the first
  several cycles, so an error of a part in a thousand in the integrals cannot reach the answer.
- **Anything recomputed in double precision afterwards** — a preconditioner, a trial step, a guess.

The shipped half lanes convert each value to sixteen bits on the way in and back on the way out, so
the conversion is their cost. If you are calling them one value at a time, that conversion is what
you are paying for and the arithmetic is not the bottleneck.

**If your card does sixteen-bit arithmetic on two values per instruction**, that instruction carries
twice the values a 32-bit one does, which is a factor of two in throughput for the same number of
instructions. **That is arithmetic on an instruction's shape, not a measurement of any card**, and
no timing here supports it. Whether a real code sees it depends on the card and on what else the
loop is doing.

**These lanes do not cover every order.** The packed-half lane returns values for orders 0 to 8 only
— at order 9 and above the function is already too small for that format to represent, and no
argument makes it usable. The 16-bit-storage lanes cover all orders but return nothing usable once
the result drops below their own bound, which happens at arguments that depend steeply on the order.
Before putting a reduced-precision lane in a loop, check what it covers for the orders that loop
uses.

## If your arguments are large

Arguments at or above x = 28.984375 have a third option: the packed-half lane, which holds its own
rounding to within 8 of the last representable digits of the result, for orders 0 to 8. It is the
most accurate of the 16-bit lanes where it applies, and it is also the narrowest.

The region-A transform lane is the other alternative direction: it computes a batch of arguments at
once and lets you choose the arithmetic, including a 32-bit mode for cards that have no 64-bit
arithmetic. Its section in `docs/lane-contract.md` states which modes reach which accuracy, and
which orders it may be asked to seed a recursion at.

## What is not claimed

No speed claim is made for any lane. See the end of `docs/lane-contract.md` for why, and for what a
measurement would need to look like.
