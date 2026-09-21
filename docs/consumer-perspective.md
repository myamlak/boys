# Design notes: the library seen from its consumer's side

This note is about what the library is *for*, as distinct from what it does. Its numbers —
evaluations per second, accuracy per lane, bytes per table — describe the library. A consumer needs
numbers about the calculation the library sits inside, and those have not been measured.

It records reasoning and open questions, not settled design.

## The question that comes first — now measured

**What share of an integral evaluation is this function at all?** Nothing in this
repository can answer that, because it is a measurement inside a consumer. **It has now been
taken in one.**

**The answer is about ten percent** — 10.3 percent of the integral span as the best estimate,
between 4.9 and 10.3 percent across runs, on a 586-basis-function fixture of 74 atoms. The
figure is a **floor, not a ceiling**, because the estimator replays a tight loop rather than
instrumenting the engine. **Machine load moved it by a third between two runs half an hour
apart**, so ten percent is the right order and not three significant figures.

**Three findings beside it matter more than the share.**

**The library supports orders to 32; production never goes past 8.** 95.2 percent of the
965 million calls measured sit at order 3 or below, mean 1.47. **Everything above 8 is
supported, maintained and instantiated for nothing.**

**The asymptotic branch serves 84 percent of calls; the Chebyshev tables serve 9.2 percent.**
That is worth weighing before any further work on the table — **its degrees, its band
structure and the partition that chooses them are on the path for under a tenth of the
calls**, and those calls are a tenth of the integral evaluation.

**And the ladder's contraction runs at roughly nine times the cost of evaluating the
function that feeds it.** The transforms and the block zeroing are the work; this function is
upstream of it.

**So the ceiling on anything done here is about ten percent of an integral evaluation**, and
most of that ten percent is on a branch that does not read the tables at all.

## Which end of the recurrence carries the requirement

For one shell quartet the consumer needs `F_0 … F_L` at a single argument, where `L` is the
total angular momentum of the quartet. There are two ways to get them, and **they put the
accuracy requirement at opposite ends of the recurrence**:

- **upward from `F_0`** — the seed is the lowest order, and each step multiplies its error by
  about `(n + ½)/x`, so the requirement is hardest at the *highest* order;
- **downward from `F_L`** — the recurrence is stable, so the requirement is hardest at the
  seed itself, and the low orders come out accurate as a by-product.

**This question was put, and the answer refutes the worry that prompted it.** Reading the
implementation rather than reasoning about it: **the library already uses both directions, and
chooses between them deliberately per lane.**

- **The double batch entry is hybrid.** Below `X0` it serves a *prefix* of the orders by an
  upward recursion from the extended-band seed — exactly those orders whose tier threshold the
  argument has reached — and takes the remaining orders from the per-order region-A fits. The
  per-order result is bit-identical to the single-order entry, which applies the same
  per-argument rule.
- **The single-precision batch entry is downward**, from a double-precision seed at the top
  order, and the reason is written where the code is: the downward recursion amplifies a float
  seed by up to `X0^n / prod(j + 1/2)`, about `5e4` at eight orders, far beyond the
  single-precision budget. One double evaluation per batch is negligible; the recursion itself
  stays in single precision.

**So the accuracy analysis is aimed at the right end after all, for the path a
double-precision consumer actually uses.** The question was still worth asking — the answer
was not derivable from the contract, only from the implementation — but the expected
consequence did not materialise, and **it should not be carried forward as though it had.**

## What actually limits the delivered error

Accuracy work needs to know which term is binding, and the answer is not the certified one.

For the shipped `F_0` table over `[0, X0)`, the truncation error **of the fit** is around
`5e-19`, while **rounding the coefficients to double contributes about `3e-17`** — the
half-ulp of the leading coefficient alone. Every band is rounding-limited at roughly
`1e-16` to `2e-16`. Meanwhile the generator's own fit bar for the region is `2.5e-14`.

**So the delivered error is set by coefficient rounding, and neither the truncation bound
nor the fit bar is what determines it.** Any change aimed at accuracy should start from that.

## The per-order question: asked, checked, and refuted

**A plausible argument that the high-order fits could be coarser was put forward and then
tested properly, and it is wrong.** It is recorded here because the *reason* it is wrong is the
useful part, and because a later reader will find the argument attractive for the same reasons.

**The argument was this.** Each order enters an integral multiplied by a coefficient decaying
like `(1/(2p))^n / (2n − 1)!!`, so high orders are damped hard and their fits should be allowed
to be less accurate — the opposite of what this library's amplification machinery asks for.

**The coefficient model is wrong.** The contraction coefficients were computed exactly, by
differentiating the closed form of the simplest two-electron integral and validating the
result three ways — against a hand-derived case, against translation invariance, and against
numerical differentiation. **The ratio is not the one assumed: it goes like `R^n`, the
internuclear separation to the order, and carries no double factorial at all.** The assumed
model contains no `R`, so it understates the high-order coefficients — by five orders of
magnitude at four orders and by eight at six, in one realistic case. **The angular-momentum
factors are what does this, and they dominate the double factorial completely.**

**And the amplification does not grow either.** Region B's amplification factor is below one
for every supported order — the band edge is chosen that way on purpose, so that recursion
*contracts* — and region A's returns to one by the highest order. So the two effects the
argument set against each other do not pull in opposite directions; neither grows.

**What is actually there, measured:** the slack per order runs between about five hundred and
twenty thousand, **flat across the orders rather than growing**, and the coefficient tails vary
by no more than fifteen times over the whole range. **Coarsening the high-order fits would buy
nine to eleven percent of a ladder up to eight orders, and only in the two argument bands where
per-order fits are used at all — which is below about two. Everywhere else it buys nothing**,
because most ladders are one seed plus a recursion rather than a set of per-order fits.

**So this is not the lever, and the table should not be reorganised around it.** The measured
cost of the change would be a re-derived table for a return of roughly a tenth of one code
path.

**One thing the same source is clear about, and it is worth writing down:** for almost every
calculation the binding constraint is *not* this function. Production screening thresholds sit
at `1e-10` to `1e-12`, density fitting carries errors around `1e-6`, a density-functional grid
`1e-6` to `1e-8`, and the coupled-perturbed convergence has its own. **All are larger than a
`1e-14` error here.** Tightening this function past `1e-14` buys nothing unless those move too,
and **the accuracy that is genuinely free to give up is in the other thresholds rather than in
this function.**

## Reduced precision: what it is for, and what it cannot do

**It cannot serve the accurate path, and that is arithmetic rather than opinion.** Half
precision carries about eleven bits of mantissa — roughly five parts in ten thousand —
against the one part in a hundred trillion that quantum chemistry asks of this function.

**But the current reduced-precision lanes are slow for a reason that has nothing to do with
half precision:** they convert single to half and back *around* each call, so the conversion
is the cost. **A batch that lives in half precision from end to end pays no conversion.**

**What pays on a GPU is packed arithmetic** — two values per register, one instruction doing
the work of two — worth roughly a factor of two over single precision on any card that
supports it. **Tensor cores buy nothing here**, because a Chebyshev recurrence is scalar:
there is no matrix to multiply. Framing these lanes as a tensor-core path misdescribes them.

**Where half precision genuinely earns its place** is in the work that does not need
accuracy:

- **Screening.** Quartets are bounded before they are computed, and the screened pool is far
  larger than the computed one. A bound wrong by a part in a thousand is harmless if it stays
  conservative.
- **Early iterations of a self-consistent field.** The density is far from converged for the
  first several cycles, so an error of a part in a thousand in the integrals cannot reach the
  answer.
- **Anything recomputed in double afterwards** — a preconditioner, a trial step, a guess.

**So these lanes are not a less accurate imitation of the accurate one.** They are the pass
that runs over everything, where the accurate lane runs over a small part of everything — and
the contract they carry should be the one their own use needs.

**A verdict on their speed belongs on hardware chosen to show it.** The advantage is a
function of the card, so a measurement on a small laptop part is not a measurement of the
lane.

## The first band's degree

The first region-A band, `[0, 5.94992407605424223]`, carried an a-priori truncation bound of
`6.47e-16` at degree 18 — about **sixteen times the `4.1e-17` the seed design targets**. **Every
order on the band was over the target**, `2.18e-16` (`F_32`) to `6.47e-16` (`F_0`), so the
degree is a property of **the band**, not of `F_0`. **It is fitted at degree 20 for every order
now**, which puts the whole family at `1.14e-18`..`3.14e-18`. Degree 19 was never a candidate:
the split Clenshaw evaluates the odd coefficients only up to `c[deg-1]`, so it requires an even
degree. The other two fitted bands meet the target as shipped (`2.58e-18`, `5.98e-18`).

**The fix moves the certificate, not the delivered error** — the point of the section above.
The band's polynomial sits `2.20e-16` from a high-precision `F_0` where it sat `2.43e-16`
before: coefficient rounding, not the truncation tail, sets both.

## What to do, in order

1. **Measure the function's share of a real integral evaluation**, in a consumer. Everything
   else is ranked by this number.
2. **Establish which recursion direction the consumer uses, per region** — one line of code
   to read, large consequence.
3. **Fix the first band's degree** — **done**: the band is fitted at degree 20 for every
   order, which puts its a-priori bound under the seed design's tail.
4. **Offer a selectable accuracy tier** matching what the chemistry needs, derived from the
   truncation bound rather than searched.
5. **Re-aim the reduced-precision lanes** at one of the uses above, and measure them on
   hardware that can show the difference.
