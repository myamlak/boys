# Choosing a lane

This note is written for a reader who has not built the library yet and has not chosen anything yet.
The words it needs — *lane*, *region*, *rung* — are defined on [the API reference's landing
page](mainpage.md). What follows is about the choice rather than about the code.

`docs/lane-contract.md` states what each lane guarantees and where it stops. This note is about the
other half of the choice: how much accuracy your calculation actually needs, and where reduced
precision earns its place.

## Your calculation probably does not need this function's accuracy

This function is one part of an integral evaluation. For almost every calculation it is not the
binding constraint. In production codes:

- integral screening thresholds sit at 1e-10 to 1e-12;
- density fitting carries errors near 1e-6;
- a density-functional grid carries 1e-6 to 1e-8;
- coupled-perturbed convergence has thresholds of its own.

All of those are larger than a 1e-14 error here. **Tightening this function past 1e-14 buys nothing
unless those move too.** Your calculation is almost certainly limited by something else. If so, the
double lane's default accuracy is already more than enough. The multiplier is then a way to buy speed,
not a way to buy accuracy you need.

## When you do need the double lane

Reach for it when this function is *recomputed* and its error compounds: a coupled-perturbed solve,
a geometry optimisation run to tight thresholds, or a property that differentiates the energy. Those
are the cases where an error here stays in the answer.

## Where reduced precision earns its place

Half precision cannot serve the accurate path. That is arithmetic rather than opinion: sixteen bits
carry about five parts in ten thousand. The accuracy these integrals are usually asked for is about
one part in a hundred trillion.

What it *can* serve is work that does not need accuracy.

- **Screening.** Groups of integrals are bounded before they are computed, and the screened-out set
  is far larger than the computed one. A bound wrong by a part in a thousand is harmless if it stays
  conservative.
- **Early iterations of a self-consistent field.** The density is far from converged for the first
  several cycles, so an error of a part in a thousand in the integrals cannot reach the answer.
- **Anything recomputed in double precision afterwards** — a preconditioner, a trial step, a guess.

The shipped half lanes convert each value to sixteen bits on the way in and back on the way out, so
the conversion is their cost. If you are calling them one value at a time, that conversion is what
you are paying for. The arithmetic is not the bottleneck.

**If your card does sixteen-bit arithmetic on two values per instruction**, that instruction carries
twice the values a 32-bit one does, which is a factor of two in throughput for the same number of
instructions. **That is arithmetic on an instruction's shape, not a measurement of any card**, and
no timing here supports it. Whether a real code sees it depends on the card and on what else the
loop is doing.

**These lanes do not cover every order.** The packed-half lane returns values for orders 0 to 8 only.
At order 9 and above the function is already too small for that format to represent, and no argument
makes it usable. The 16-bit-storage lanes cover all orders but return nothing usable once the result
drops below their own bound. That happens at arguments that depend steeply on the order. Before
putting a reduced-precision lane in a loop, check what it covers for the orders that loop uses.

## If your arguments are large

Arguments at or above x = 28.984375 have a third option: the packed-half lane. It holds its own
rounding to within 8 of the last representable digits of the result, for orders 0 to 8. It is the
most accurate of the 16-bit lanes where it applies, and it is also the narrowest.

The region-A transform lane is the other alternative direction. It computes a batch of arguments at
once and lets you choose the arithmetic, including a 32-bit mode for cards that have no 64-bit
arithmetic. Its section in `docs/lane-contract.md` states which modes reach which accuracy, and
which orders it may be asked to seed a recursion at.

## Choosing inside the lane you picked

Picking a lane is not the whole of the choice. Inside one lane the library offers the product of six
axes. Each of them is a decision a caller can make:

- **the fit route** — the stored fits that serve a region are either the Chebyshev ones the
  certified lanes are defined by, or a rational minimax alternative that holds the same bar by
  storing different things;
- **the evaluation scheme** — a fit's coefficients are summed either by the split Clenshaw
  recurrence on its Chebyshev form, or by Horner's rule on the monomial form of the same fit;
- **the interval partition** — how the fitted intervals are cut: as the shipped tables cut them; more
  narrowly, into pieces the proved truncation bound places where the function needs them, holding the
  bar at a lower degree; or into a fixed grid of equal cells over the whole fitted domain, every
  order fitted independently at one degree;
- **the packing axis** — which of a call's values share a vector register: four arguments at one
  order, or four orders at one argument;
- **the division form** — how each step of the recurrence divides: exactly, by a plain reciprocal, or
  by a reciprocal refined back to the correctly-rounded quotient. That last one is bit-identical to
  dividing and costs two fused multiply-adds to get there;
- **the accuracy rung** — how far the stored fit is cut for this call, which is what the accuracy
  multiplier names.

**Five of the six are names resolved where the code is written; the sixth is the call's own
argument.** The route, the scheme, the partition, the packing axis and the division form are written
into the call site. The rung is the one a caller may need to decide at run time. The split is
deliberate. How much accuracy you can afford may only be known at run time: from the size of the
system you were handed, or from how many self-consistent-field cycles you are prepared to spend. The
rung therefore has to be something you can compute there. Which fits to read and how to sum them is a
decision made once, while the code is being written. A call site that resolved it on every call would
pay, every time, to rediscover what its author already knew. So those five are resolved where they
are named and nowhere else: there is no combination to look up at the call, no name to match and no
registry to consult. A call costs nothing beyond the arithmetic it asked for.

**The library defaults what a caller has not decided, and the line between the two is the one the
library draws.** Precision, accuracy rung and the shape of the question are the caller's, because
they change what comes back: which arithmetic is used, how much error it is allowed, and whether the
call answers for one order or for a ladder of them, at one argument or across an array of arguments.
The five structural axes are the library's to default. Each precision has one setting that a call
naming nothing receives. A caller who cares about one of them may name it; a caller who does not is
not asked to.

**Where the recommendation comes from: a measurement taken where you deploy.** Which combination is
cheapest depends on the machine, on whether your compiler fuses a product-plus-add into one
rounding, and on how your arguments arrive. So the library ships the measurement rather than a
recommendation. The probe in this tree (`boys-option-probe`) ranks the combinations your build
offers and prints the accuracy each one delivered beside its cost. You can then see whether a faster
row was faster at the same accuracy or at a lower one. Choosing is a development-time act: run it
once, read the row it puts first for your precision, your rung and the shape of your question, and
write that combination's name into the call site. **The figures belong to the host they were taken
on** — a ranking taken somewhere else is not evidence about your machine. The device lane ships a
probe of the same kind for a card (`boys-device-probe`), because which entry is cheapest on a card
is a property of the card.

**What you get by naming nothing is a tie rather than a measured win.** The route and the packing
axis carry the settings the library has always shipped. The scheme and the partition were set from
the probe's own runs, and those runs did not separate the rows of either axis. The three leading rows
of the double lane's full-accuracy all-orders ranking came out closer together than any one of them
moves between two runs of the same probe. The settings this library ships are one of those three,
not the cheapest of them. That costs nothing in accuracy — the two schemes sum one fit and the
shipped and narrow partitions cut one fit, each certified to a bound of its own — but it is what the
default is. A reader told the default was a measured win would have been told something the
measurement does not say.

**If you have a number to stay under rather than a ranking to read**, the library answers that
question directly. Name a combination and the absolute error your calculation needs; you get a
verdict beside the two figures it was made on. A combination whose guaranteed bound is at or below
your requirement is inside it, and that guarantee is the state a calculation's safety can rest on. A
combination whose bound is above the requirement while the error it was measured to deliver is at or
below it is at your target in what it delivers and outside its guarantee. Read that state as what it
is, not as a promise. A combination that is neither is outside. The two figures are also what to rank
two combinations by, and they answer different questions: one is what the lane promises, the other is
what it was measured to do.

## What is not claimed

No speed claim is made for any lane. See the end of `docs/lane-contract.md` for why, and for what a
measurement would need to look like.
