# The library from its consumer's side

This note is about what the library is *for*, not what it does. The library's own numbers describe
the library. A consumer needs numbers about the calculation the library sits inside.

## How much of a calculation is this function?

About **ten percent** of an integral evaluation, measured on a 586-basis-function fixture of 74
atoms.

Treat that as a floor, not a ceiling. The measurement replays a tight loop rather than instrumenting
the engine, so it understates the share. Machine load moved it by a third between two runs half an
hour apart. Ten percent is the right order of magnitude, not three significant figures.

Three findings beside it matter more than the share itself.

**The library supports orders to 32. Production rarely goes past 8.** The call distribution was
measured in one consumer code and is not reproduced here, so read it as an indication rather than a
number.

**The asymptotic branch serves most calls. The Chebyshev tables serve a small minority.** That
matters before any further work on the tables. Their degrees, their band structure and the rule that
chooses the bands sit on a small share of the calls.

**The ladder's contraction costs several times what evaluating the function costs.** The transforms
and the block zeroing are the work. This function is upstream of them.

So the ceiling on anything done here is about a tenth of an integral evaluation.

## Which end of the recurrence carries the requirement

A consumer needs `F_0 … F_L` at one argument, where `L` is the total angular momentum of a shell
quartet. There are two ways to get them, and they put the accuracy requirement at opposite ends.

- Upward from `F_0`: the seed is the lowest order, and each step multiplies its error by about
  `(n + ½)/x`. The requirement is hardest at the highest order.
- Downward from `F_L`: the recurrence is stable. The requirement is hardest at the seed, and the low
  orders come out accurate as a by-product.

The library uses both, and chooses per lane.

- The double batch entry is hybrid. Below `X0` it serves a prefix of the orders by upward recursion
  from the extended-band seed. It takes the rest from the per-order region-A fits.
- The single-precision batch entry recurses downward from a double-precision seed. A float seed
  would be amplified past the single-precision budget, by about 5e4 at eight orders.

## What limits the delivered error

The fits are better than the format that stores them. Over `[0, X0)`, the shipped `F_0` table's own
truncation error is about 5e-19. Rounding its coefficients to double costs about 5.551e-17. That is
the half-ulp of the leading coefficient, which sits in the binade where a half-ulp is exactly that.

Every band is rounding-limited, at roughly 1e-16 to 2e-16. The generator's own fit bar for the region
is 2.5e-14.

**So the delivered error is set by coefficient rounding.** Neither the truncation bound nor the fit
bar determines it. Any work aimed at accuracy should start from that.

## Could the high-order fits be coarser?

No. It is worth recording why, because the argument for it is attractive.

The argument runs like this. Each order enters an integral multiplied by a coefficient that decays
sharply. High orders are therefore damped, so their fits could be less accurate.

The argument fails because the coefficient model is wrong. The contraction coefficients decay like
`R^n`, the internuclear separation to the order. They carry no double factorial. The model they were
assumed to follow has no `R` in it at all, so it understates them by orders of magnitude. The
angular-momentum factors dominate the double factorial completely.

The amplification does not grow either. Region B's amplification is one to within a last-digit
excess, `1 + 1.846e-17` at order 32. Region A's returns to one by the highest order.

What is actually there: the slack per order is flat across the orders rather than growing.
Coarsening the high-order fits would buy about ten percent of one code path, and only below x = 2.
Everywhere else it buys nothing. Most ladders are one seed plus a recursion, not a set of per-order
fits. **This is not the lever.**

## What the other thresholds already allow

For almost every calculation, the binding constraint is not this function. Production screening
thresholds sit at 1e-10 to 1e-12. Density fitting carries errors near 1e-6. A density-functional grid
carries 1e-6 to 1e-8. Coupled-perturbed convergence has its own.

All of those are larger than a 1e-14 error here. Tightening this function past 1e-14 buys nothing
unless they move too. The accuracy that is genuinely free to give up is in the other thresholds, not
in this function.

## Reduced precision

Half precision cannot serve the accurate path. That is arithmetic, not opinion. Eleven bits of
mantissa is about five parts in ten thousand. Quantum chemistry asks this function for one part in a
hundred trillion.

The shipped half lanes are slow for a reason that has nothing to do with half precision. They convert
single to half and back around each call, so the conversion is the cost. A batch that lives in half
precision from end to end pays none of it.

What pays on a GPU is packed arithmetic: two values per register, one instruction doing the work of
two. That is worth roughly a factor of two over single precision on a card that supports it. **Tensor
cores buy nothing here.** A Chebyshev recurrence is scalar, so there is no matrix to multiply.
Framing these lanes as a tensor-core path misdescribes them.

Where half precision earns its place is in work that does not need accuracy.

- **Screening.** Quartets are bounded before they are computed, and the screened pool is far larger
  than the computed one. A bound wrong by a part in a thousand is harmless if it stays conservative.
- **Early self-consistent-field iterations.** The density is far from converged for the first several
  cycles. An error of a part in a thousand in the integrals cannot reach the answer.
- **Anything recomputed in double afterwards** — a preconditioner, a trial step, a guess.

So these lanes are not a less accurate imitation of the accurate one. They are the pass that runs
over everything. The accurate lane runs over a small part of everything.

A verdict on their speed belongs on hardware chosen to show it. The advantage is a property of the
card. A measurement on a small laptop part is not a measurement of the lane.

## The first band's degree

The first region-A band, `[0, 5.94992407605424223]`, carried an a-priori truncation bound of 6.47e-16
at degree 18. That is about sixteen times the 4.1e-17 the seed design targets. Every order on the
band was over the target, so the degree is a property of the band rather than of `F_0`.

It is now fitted at degree 20 for every order, which brings the family to 1.14e-18 to 3.14e-18.
Degree 19 was never a candidate. The split Clenshaw evaluates odd coefficients only up to
`c[deg-1]`, so the degree must be even. The other two fitted bands meet the target as shipped.

The change moves the certificate, not the delivered error. Coefficient rounding sets both.
