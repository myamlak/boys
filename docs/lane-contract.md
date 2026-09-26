# Accuracy: what each lane guarantees

This library evaluates the Boys function `F_n(x) = ∫₀¹ t^(2n) e^(−x t²) dt`, for orders n = 0 to 32
and arguments x ≥ 0.

It offers several lanes, which differ in precision and in cost. This page states each lane's error
bound, the arguments the bound covers, and where the lane stops being usable.

Every bound here is stated as an **absolute** error — |computed − true| — except where an entry says
otherwise. The packed-half lane's bound is relative, and the half lanes' bound is absolute plus a
term that depends on the size of the result.

## How the argument affects the bound

The library evaluates the function differently at different argument sizes, and the error bound
differs with it:

| Arguments | How they are evaluated |
|---|---|
| x < 11.899848152108484 | stored polynomial fits |
| 11.899848152108484 ≤ x < 28.98933773882074 | a stored fit at the lowest order, then upward recursion |
| x ≥ 28.98933773882074 | a closed-form asymptotic result |

Below about x = 1.0855 the lowest orders are inside their own fits. The argument at which an
order stops being read from its own fit rises with the order — 1.0855252345349333 for orders 0
to 4, 2.015297705335114 for 5 to 8, 4.897870299825657 for 9 to 16 and 10.783587858916762 for 17
to 32 — and past its own boundary an order is served by recursion from a single fit, the band
seed, which is slightly less accurate than the fits themselves.

**Two lanes do not hold over the whole range**, and their entries say where they stop: the half
lanes return nothing usable once the result falls below their bound, and the packed-half lane covers
only the largest arguments.

## double

**At most 5.5e-14 everywhere.** At most 3e-14 for x below 11.899848152108484, and at most 1e-15 for
x below about 1.0855. That last figure belongs to the single-argument entry; the batch entry is at
most 5.5e-14 throughout, so a caller using the batch form should read 5.5e-14.

A multiplier, set at compile time or per call, is any value at or above 1, with no upper end.
Raising it loosens the bound and reduces the work **on the default route**, which pays for the
looser bound with fewer stored coefficients to sum. The rational route is not relaxed by the six
multipliers this library names, and its rung section below states why and with what figure: a
caller who needs the rational route's fits to cost less has to ask for a different pair, which is
a fit this revision does not carry.

**This is the most accurate double the library produces.** The stored coefficients are rounded to
double precision and hold only the sixteen or so significant digits a double can hold, so nothing
below about 1e-16 is reachable however the evaluation is arranged. The mathematics behind the fits is
more accurate than the numbers storing them, but by how much depends on the piece: against the
shipped coefficients the two are of the same order, the storage error usually a few times the
truncation and on some pieces smaller than it.

## The two fit routes

The double lane's fits come in two routes, chosen per call with `BoysAllOrdersWithRoute` and
reported by `BoysFitRoutes()`. Neither is a rung: a route is a way of serving a region, and the two
hold the same bar over the same interval.

| Route | Region | Fit covers | Selector serves from | Stored | Measured | Bar |
|---|---|---|---|---|---|---|
| chebyshev (default) | A | 0 | 0 | 1320 | 1.74e-16 | 3e-14 |
| rational minimax | A | 0 | 1.0855252345349333 | 891 | 2.46e-14 | 3e-14 |
| chebyshev (default) | B | 11.899848152108484 | 11.899848152108484 | 19 | 9.92e-15 | 5e-14 |
| rational minimax | B | 11.899848152108484 | 11.899848152108484 | 12 | 4.46e-14 | 5e-14 |

Region A's rows count the whole per-order table: one piece for each order over each of the region's
two bands, so 66 pieces, and "stored" is their coefficients summed. Region B's rows are the one seed
each route evaluates there, carried to higher orders by the region's own upward recursion.

**The rational route in region A does not take over from zero, and it takes over per order.** Its
pieces span the same intervals as the Chebyshev ones, but the lane stops reading an order from that
order's own fit once the argument passes the order's own region-A end, and reaches it from the band
seed instead, where the lane's figure is the band's 3e-14 rather than the order's 1e-15. So the route
hands over each order at that order's own argument — 1.0855252345349333 for orders 0 to 4,
2.015297705335114 for 5 to 8, 4.897870299825657 for 9 to 16 and 10.783587858916762 for 17 to 32 — and
the row states the lowest of them, because that is the argument from which naming the route changes
any value. Below its own boundary an order keeps the default lane's value exactly, so the tighter
figure the lane documents there is untouched; above it the route holds the 3e-14 the lane documents
there rather than a bar of its own.

**The two region-A tables are not interchangeable, and their stored counts are not each other's
substitute.** The default table's pieces do a second job: the batch entry's relaxed path seeds its
downward recursion from the top order's piece over the whole of the region, and that recursion
carries the seed's error down to F_0 with a gain of max(1, b^n / ∏(j+½)) at the piece's right end b —
up to 1.04e5, at order 12 and the region's right end. The default pieces are fitted under a budget
divided by that gain, which is why the table stores more than the plain figure its row reports would
need, and why its 1320 is not what the same published bar would cost on its own. The rational pieces
are read one order at a time, are held to the published bar and nothing tighter, and a piece fitted
for the values alone does not survive that recursion: against the same reference and in the same
arithmetic the rational table's worst error after the gain is 6.01e-12, where the default table's is
1.04e-14, and 18 of the region's 66 rational pieces sit over the 2.5e-14 the shipped ones are fitted
under against none of the default pieces. **The rational route buys the interval's values, not a
seed.**

**The comparison is not "12 against 19" in region A.** Region B's two rows are two seeds for one
interval, so their stored counts sit directly against each other. In region A the rows count whole
tables — 1320 coefficients across the region's 66 pieces against 891 — and what the pair says is
that over the same intervals, against the same reference and in the same arithmetic, the default
route's table holds 1.74e-16 while spending 1320, and the rational route holds 2.46e-14 with 891. A reader
comparing the two counts is comparing a table fitted for the recursion's gain against one fitted for
the values alone, and the honest reading is the one the two rows state between them: both hold the
bar, and the rational route reaches it with fewer coefficients because it is asked for less.

**A rational fit costs one division per order**, where the split-Clenshaw Chebyshev form is
division-free. That is a real difference in the work, and which side of it a machine lands on is its
divide-to-multiply throughput — so no speed is claimed for either route here.

### The rung, per route

The multiplier is a third selector, and it acts on the route the call names rather than on the
library: a rung cuts the named route's own stored fit to the degrees a criterion certifies for that
multiplier. `BoysAllOrdersAtTier(tier, route, scheme, ...)` is where a caller names all three at run
time. The two routes' criteria read different tables, and the difference is not a detail of the
implementation:

- **The Chebyshev route's rung** cuts the stored coefficient series by one degree `d'`, and the
  dropped tail is `Δ(d') = Σ_{k>d'} |c_k|`, which bounds the truncation because every basis function
  is at most one in modulus on the mapped interval. The table is `RegionADegrees`/`RegionBDegrees` in
  `boys_effective_degrees.hpp`, one per (multiplier, lane role, coefficient basis).
- **The rational route's rung** cuts the stored numerator and denominator together, to
  `[m', k'] = [min(d', m), min(d', k)]`. A quotient's perturbation is not a coefficient sum, so the
  measure is the two terms it actually has. With `δP = P − P'` and `δQ = Q − Q'`,

  ```
  R − R' = P/Q − P'/Q' = δP/Q − R'·δQ/Q
  ```

  exactly, and with `ΔP(d') = Σ_{j>m'} |p_j|`, `ΔQ(d') = Σ_{j>k'} |q_j|`, `SP = Σ_j |p_j|` and `Qlo`
  a lower bound on `|Q|` over the piece,

  ```
  Δ_pair(d') = ( ΔP(d') + (SP / (Qlo − ΔQ(d'))) · ΔQ(d') ) / Qlo   ≥   sup |R − R'|
  ```

  — the numerator's dropped tail against the denominator's floor, plus the denominator's dropped
  tail carried by the pair's value scale. `Qlo` is taken from `Q`'s own values on a 64-interval grid
  less what `Q` can move between grid points (`Σ_j j|q_j| · h/2`), because `Σ_j |q_j|` exceeds 1 on
  these pieces and the triangle bound says nothing. A cut whose floor does not clear its own `ΔQ` is
  refused rather than admitted: nothing bounds the pair's scale there. The table is
  `RationalRegionADegrees`/`RationalRegionBDegrees`, beside the polynomial ones.

Both scans share one criterion — `Δ · A ≤ (m − 1) · B_region`, the smallest admissible `d'` in the
evaluator's domain `{0,1,2,4,6,…}` — and both fall back to the untruncated fit when nothing smaller
is admissible. For the polynomial family the fallback is safe because `Δ(deg) = 0`. The rational
family's equivalent is a pair cut to itself: `δP = δQ = 0`, so `R − R' = 0` exactly, and the rung
returns the route's stored pair bit for bit. A rung that cannot reach its budget therefore costs
nothing and claims nothing, and no rung is left without an admissible cut.

**The rational rungs do not relax the rational route at this revision, and this is a measured
result rather than a fallback.** The smallest pairwise tail any truncating cut of any of the 66
region-A pieces reaches is **5.644946e-09** (F24's second piece, cut 6 of its stored `7/5`); the
smallest for the region-B seed is **6.843273e-06** (one numerator order dropped). The criterion
admits a cut when `Δ_pair · A ≤ (m − 1)·5.5e-14`, so the first multiplier that carries the cheapest
region-A cut is `m ≈ 102636` and the first that carries the region-B one is `m ≈ 1.24e8`. The
largest multiplier `AccuracyTier` names is 65536, whose budget is 3.604425e-09. Every one of the
twelve (route, scheme, rung) combinations above m = 1 therefore evaluates the stored pair, and the
gate measures it doing so: the delivered figures are the uncut route's own, and the bound published
for each rung is `m · 5.5e-14`.

The region-A amplification is 1 here rather than the batch role's `w(b)`. The route reads one piece
per order — the pieces are fitted for the values alone and do not survive the batch downward
recursion's gain — so a cut's error reaches the value it was read for with nothing in between.
Region B reads its seed at order 0 and carries it up, so its cut pays the same `A_B(n)` the shipped
seed's does.
## Interval granularity: how narrowly the fitted domain is cut

A stored fit is a polynomial over one interval, and the interval's width is the lever on its degree:
the truncation bound carries the half-width as roughly `(h / 2d)^d`, so **halving a piece buys about
`2^d`**, while raising the degree at a fixed width buys far less, because the optimal ellipse
parameter falls as the degree rises and partly cancels the gain. **Splitting is the design move;
more degree is not.** The library offers two partitions and no spectrum between them, and the
shipped one is the default.

**What narrower pieces buy, and what they cost.** They cut the number of coefficients an evaluation
reads, not the size of the table: a narrower piece needs a lower degree, and there are more of them.
The choice is therefore between work per call and storage plus a piece lookup per call, and neither
partition is a rung of the other.

**The shipped partition.** Region A: two equal-width bands per order, at degree 20 and degree 18 —
40 stored coefficients per order, 1320 across the whole table. Region B: one seed fit, degree 18,
19 stored. The extended band is its own fit at degree 24 and is not a choice either member makes.
Read against the a-priori truncation bound below at the quantum-chemistry target of
1e-14: region A's two bands hold 3.14e-18 and 2.58e-18, and the extended-band seed holds 5.98e-18 —
all far inside the target. **The region-B seed is the one piece the bound does not cover: 1.36e-13,
above the target**, and the piece's *measured* truncation is 9.92e-15, inside it. So the shipped
seed meets the target by measurement and misses it by the bound, and that gap is the honest
statement of where the shipped design stands against a 1e-14 target.

**Region A's pieces are held to a second reading, and it is not the bound above.** A region-A piece
is read two ways. The single-order lane reads it at its own size, where region A is documented at
1e-15. The batch entry reads the top order's piece *and recurses downward from it to F_0*, so that
piece's error arrives at F_0 multiplied by

    w(n, b) = max(1, b^n / ∏(j + ½))

at the piece's right end `b`, the widest point of the gain. The shipped pieces were placed under that
reading already — the law the shipped fits are weighted by is `5e-14 / 2 / w(n, x)` — so the narrow
walk uses one criterion and not two average readings:

    budget(order, b) = min(1e-15, 2.5e-14 / w(n, b))

whose second term is the seed reading and whose minimum is the tighter of them. The gain binds only
above `w = 25`, which over region A is a trailing run of each order's pieces rather than the whole
region. Its size across the orders, at the region's right edge:

| order | `w(n, X0)` | order | `w(n, X0)` |
| --- | --- | --- | --- |
| 0 | 1 | 16 | 5.522e4 |
| 4 | 3056 | 20 | 1.063e4 |
| 8 | 5.078e4 | 24 | 914.6 |
| 12 | 1.04436e5 | 28 | 40.28 |
| | | 32 | 1 |

The envelope peaks at 1.04436e5, at order 12 and the region's right edge, and it tightens 127 of the
walk's 311 pieces — the trailing run of 27 of the 33 orders, from where the weight crosses 25 to the
piece that ends at the region's right edge. The tightest budget any piece is held to is 2.394e-19,
at order 12's last piece.

**The gain the walk holds is the envelope; the most a call in this revision reaches is 2.171050.** The
seeding fallback is taken only below the band's left edge, `x < 1.08552523453493330`, and what a call
pays there is its own top order's gain rather than the region's worst. For a fixed `x` the ratio
`x^n / ∏(j + ½)` rises with the order only until the order passes `x − ½` and falls after, so below
the band edge the largest value it has at an order of 3 or more is order 3's 0.682, and a call whose
top order is 3 or more seeds at gain 1. The two orders below that do amplify: 1.571153 when the top
order is 2, and 2.171050 when it is 1 — the latter only as `x` comes up to the band's edge, where it
is exactly `2x`. Against the envelope's 1.04436e5 that is a factor of 4.8e4.

Holding the envelope rather than the reachable figure is deliberate: a piece whose correctness
depended on which order an entry happened to seed from would be a fit that is sound only until the
dispatch changes, and the internal thresholds that choose the seed are not part of the contract. The
price it charges shows up as narrower pieces at the high orders' right ends rather than as accuracy.

**Where the bound comes from.** For F_n the analytic continuation carries an exact majorant:
`|F_n(z)| ≤ F_n(Re z)` by the triangle inequality applied to the integral representation, and F_n
decreases in its real argument, so on the Bernstein ellipse of `[a, b]` at parameter `rho` the
maximum modulus is at most `F_n(c − (h/2)(rho + 1/rho))` — the ellipse's leftmost point, with
`c = (a + b)/2` and `h = (b − a)/2`. Trefethen's interpolant bound then gives

    E(n, a, b, d) = min over rho > 1 of 2 · F_n(c − (h/2)(rho + 1/rho)) · rho^(−d) / (rho − 1),

and **nothing in it samples the function**. The script that generates the tables computes it
(`tools/gen_boys_coefficients.py --derive-partition`), and the anchor that says the computation is
right is that it reproduces, to five significant figures, the two figures the script's own comments
carry from an independent earlier instrument: 6.4676e-16 for F_0 on `[0, 5.94992407605424223]` at
degree 18, and 3.14e-18 at degree 20. Its series also agrees with the fit path's own stable series
to a worst relative difference of 2.4e-26.

**The derived narrower partition, region B.** Equalising the bound at 1e-14 and walking `[11.8998481521,
28.9893377388)` left to right gives, per degree:

| degree per piece | pieces | stored in total | read per evaluation |
| --- | --- | --- | --- |
| 10 | 5 | 55 | 11 |
| 12 | 3 | 39 | 13 |
| 14 | 2 | 30 | 15 |
| 16 | 2 | 34 | 17 |
| 18 | 2 | 38 | 19 |
| 20 | 1 | 21 | 21 |

The narrow member is the first row: five pieces of width 2.86, 3.41, 4.09, 4.95 and 1.79, each
holding 1e-14 by the bound, **11 stored against the shipped seed's 19** — the coefficient count an
evaluation reads falls by 8, and the table grows from 19 to 55.

**The derived narrower partition, region A.** Every order's interval is walked from zero to where
that order's last shipped piece ends — so the narrow member serves the whole of what it replaces and
nothing beyond it — at one degree, 10, under the budget above. The walk produces 311 pieces, 3421
stored coefficients, and its widths show both readings at work: F_0's pieces grow to 1.78 at the
widest, since the gain is 1 at order 0, while a high order's last pieces come back as narrow as
0.057 where the envelope binds. The two partitions as tables:

| | `FitGranularity::kShipped` | `FitGranularity::kNarrow` |
| --- | --- | --- |
| stored coefficients | 1339 (1320 region A + 19 region B) | 3476 (3421 + 55) |
| stored rows | 67 (66 pieces + 1 seed) | 316 (311 + 5) |
| degree | 20 and 18 in region A, 18 in region B | 10 everywhere |
| read per evaluation | 19 to 21 | 11 |

**What the member measures, and why it is not the narrowing that buys it.** Over region B the
committed tables are the first row of the degree ladder, and over the interval their pieces cover the
seed delivers a worst absolute error of 7.21645e-16 at n = 0 against the committed reference —
published as the bound 8.88178e-16, the round-up the stored rows use. The shipped seed's own row
reads 9.9365e-15 against its published 1.42109e-14, so 9.9365e-15 / 7.21645e-16 = 13.8 is how much
smaller the narrow member's error is. **That difference is the derivation and not the split**: the
shipped seed was placed by bisecting on a sampled residual at a 5e-14 tolerance, while the narrow
pieces are placed by the proved bound at 1e-14, and a tighter criterion is what the accuracy
difference measures. What the split buys is reaching that tighter criterion at *eleven* coefficients
an evaluation rather than nineteen — the bound is conservative by a factor near 14 at this width (see
the degree ladder above), so the 1e-14 the pieces are solved for is not the 1e-14 they deliver. Over
region A the narrow pieces read 2.22045e-16, one unit in the last place of a value near 1, against
the region's published 1e-15 bar and the shipped table's own 2.22045e-16 — the same figure at 11
coefficients instead of 19 to 21, which is what the region's redesign buys there. The gain the walk
holds costs nothing measurable at this revision either: the widest it gets on a call the dispatch can
make is 2.171050, and the region reads 2.22045e-16 at both partitions.

**The trade, stated as a trade.** A consumer trades coefficients read per evaluation (19–21 → 11)
against table size (1339 → 3476) and a piece lookup per call. Nothing about it is a saving, and a
consumer whose cost is the table should not take it.

**And a figure that does not survive the check.** A measurement quoted for this axis says that
narrowing the region-B seed from width 17.09 to one seventh of that cuts the requirement from 19
stored to 7. **That is not what the proved bound says, and it is not what a measurement says
either.** At width 2.4413557 — seventeen point zero nine divided by seven — a degree-6 fit of F_0
delivers a worst measured error of 5.15e-11 and its a-priori bound is 8.57e-10, which is four to
five orders worse than 1e-14. The degree that reaches the target at that width is 10, at 11 stored.
The figures that do agree are these. At degree 6 the bound is 8.57068e-10 against a measured
5.14888e-11, a ratio of 16.6; at degree 8 it is 1.39861e-12 against 8.95157e-14, a ratio of 15.6;
at degree 10 it is 2.01396e-15 against 1.41743e-16, a ratio of 14.2. A consistent factor is what a
bound that is conservative rather than vacuous looks like. **7 stored is what a width-2.44 piece
needs at a target near 1e-10, not at 1e-14, and the denominator here is 5.14888e-11 / 8.57068e-10
— the measured truncation over the bound.**

**The option is exposed, certified, and refused where it has no tables.** `FitGranularity::kNarrow`
is a field of `EvalPolicy` and a name the report can print. Its tables are the generated header's,
reachable through the same entries as the shipped partition, and its certification is the accuracy
gate's own block: 32 rows, one per partition, scheme and call shape, each judged against the bar the
published table holds for the cell it ran in, with the worst cell named on the row. The call shapes
are the stored fits read directly, the batch entry below the band (where the seeding fallback is
taken), the batch and plane entries over the band and below, the batch entry over region B, the
single-order entry at region A's cell, and the single-order and plane entries over the whole grid.
Three rows move between the partitions, and every other row is the same figure under either name:

| row, worst over both schemes | shipped | narrow | bar |
| --- | --- | --- | --- |
| batch entry, below band | 1.11022e-16 | 1.23165e-16 (n=32, x=1e-08) | 5.5e-14 |
| stored seed, region B | 9.9365e-15 | 7.21645e-16 (n=0, x=11.8998) | 3e-14 |
| batch entry, region B | 9.9365e-15 | 7.51675e-16 (n=32, x=11.8998) | 5.5e-14 |
| stored fits and single entry, region A | 2.22045e-16 | 2.22045e-16 | 1e-15 |
| batch and plane entry, band and below | 3.21618e-15 (n=16, x=4.89985) | 3.21618e-15 | 5.5e-14 |
| single and plane entry, whole grid | 5e-14 (n=32, x=28.9893) | 5e-14 | per region (m × B_region) |

**All 32 rows are met and the books around them do not move.** The block is counted apart from the
lane book and from the scheme rows, and those read what they read before it existed: 39 of 39 claims,
44 of 44 scheme rows, and the combinations book 28 of 28 with nothing owed. What does move is the
option space, 28 members to 30 — the two new ones are the partitions themselves, each measured over
289444 cells at both schemes with 150049 of them reading differently under the other partition and
none over the bar its row is judged at. The block reports its own carrying fraction, 481700 of 578888
cells (83.2%), as the cells able to discriminate; the other 97188 carry a bound at least as large as
the value itself, so no error can exceed them, and they are not counted in the rows above.

**What is refused is the combinations with no narrow table**, at compile time and where they are
named: the rational minimax route, which is one numerator/denominator pair over the whole interval
and has no partition of it; the relaxed rungs `m > 1`, which truncate the shipped fits to certified
effective degrees and have no counterpart among pieces that are all at one degree, in either region;
and the single-precision lanes, which hold one coefficient set and no narrow one. Each is a static
assertion with the reason, and none of them falls back: the two partitions are different fits of the
same function over the same interval, so a substitution would return the shipped numbers under the
narrow partition's name.

## float

**At most 1.5e-7, absolute and everywhere.**

The same multiplier applies. The batch entry computes a starting value in double precision and then
recurses in single.

**The limit is the number format.** A 32-bit float carries about seven significant digits, which is
about 6e-8 relative. The bound is 1.5e-7 absolute, so for values of order one the two are within a
factor of about two and a half of each other. Much smaller values are bounded loosely, because an
absolute bound says less about a small number than a relative one would.

## the region-A transform lane

A separate entry computes the fits of region A as a matrix product instead of by the fitted
recurrence: one product per band, all orders at once, for a batch of arguments. It is an alternative
to the lanes above and changes none of them. It adds nothing to them either, at the same precision:
its arithmetic mode is what the caller picks, and the mode decides both the bound and what hardware
can run it.

| Mode | Operand | Accumulated in | Bound over region A |
|---|---|---|---|
| fp64 | 64-bit | 64-bit | m·1e-15 |
| 3xTF32 | 32-bit, as two 11-bit parts | 32-bit | m·1e-15 + 2.5e-7 |
| bf16×6 | 32-bit, as three 8-bit parts | 32-bit | m·1e-15 + 2.5e-7 |

The 64-bit mode delivers 1.11e-16, which is the fits' own truncation: the product adds nothing
measurable to what the coefficients already cost. The other two deliver 1.92e-07 and 1.95e-07, which
is their 32-bit accumulator's own floor.

**The limit of the first is the table, and of the other two the accumulator.** Rounding an fp64
operand to 11-bit parts and summing six of them changes nothing while the total is 32-bit, so no
amount of operand precision reaches 64-bit-grade accuracy on a 32-bit accumulator. That is a
property of the card's accumulator, not of the split.

**The multiplier does not improve the two 32-bit modes.** A format's floor does not scale with the
multiplier; the fit's term does, but it cannot reach 1.9e-07 until the multiplier is about 1.9e8,
four orders past the largest value this library offers. So over the whole multiplier range their
bound is that floor. The 64-bit mode's floor is below the 1e-15 budget, so its bound is the same
m·1e-15 as the double lane's and the multiplier lifts it from 2 upward as everywhere else.

**The two 32-bit modes miss the float lane's bound at multiplier 1 and hold it from 2.** Their floor
is 1.92e-7 against the float lane's 1.5e-7, so 1.28 times as much. The float lane's bound is m·1.5e-7,
so at multiplier 2 it is 3e-7 and both modes are comfortably inside it. That is the trade they offer:
a float-lane-grade result from a card with no 64-bit arithmetic, at 1.28 times the float lane's bound
at the same multiplier.

**These two bounds are arithmetic on a model, not a measurement of a card.** They were measured
against the committed reference grid and against a dense sweep of the band, but on a CPU that
emulates the mode's accumulator with ordinary 32-bit additions. A tensor core's fused sum does not
work that way: it truncates and aligns the terms with a limited number of extra bits, which makes it
less accurate than this model, never more. So the model can talk a caller out of a mode and cannot
talk one into it, and the 32-bit rows above are the ones it is least entitled to judge. The 64-bit
row is a claim about an ordinary double sum, a far narrower thing to assume, but it is still
untested on hardware. **No speed is claimed for any mode**: what was verified here is the
arithmetic.

**The domain is the whole of region A for the values themselves.** A caller who takes one order's
value as the starting point of the downward recursion is held to a tighter requirement, because
that recursion carries the order's error up to order 0 with a gain that peaks at 1.04e5 near the
band's right end and is 1 at the highest order. There the fp64 mode may seed at orders 0 to 2 and 25
to 32 and nowhere between — the admissible orders are not a range — and the two 32-bit modes may
seed at no order at all.

## half — storing 16-bit values and computing in 32-bit

**At most `m·1e-7` plus half of the last representable digit of the result**, where `m` is the same
multiplier as above. This is the one bound that mixes kinds: the first term is an absolute error and
the second is proportional to the size of the result.

Half precision takes no part in the arithmetic. **The argument is rounded to 16 bits and the
evaluation runs at that rounded value**, so the bound compares the result against the function at the
rounded argument, not at the argument the caller passed. That distinction matters where the argument
is large: near x = 29 the rounding of the argument is coarser than the representation term in the
bound, so a caller who needs accuracy relative to their own argument should use the float or double
lane.

**The margin is thin, and it is not the same for both formats.** Both figures below were measured at
the default multiplier against a 45-digit reference, over every argument either format can represent
and every order: a 16-bit input is a finite set, so that sweep is exhaustive rather than dense, and it
is the instrument the two figures rest on. A sparser grid understates them — the committed reference
grid reads 0.999 for the 16-bit format and 0.9994 for the brain-float one, and both figures here are
worse. For the 16-bit format the worst case is **0.9993 of the bound**, a margin of 0.07 per cent.
For the 16-bit brain-float format the worst case is **0.99992 of the bound**, a margin of 0.008 per
cent, against the 16-bit format's 0.07. At that distance the representation term is very
nearly the whole bound, and the arithmetic term has almost nothing left to spend. One extra step of
error anywhere in the lane therefore pushes a result over its bound. Treat this lane as inside its
bound with nothing to spare.

**The limit is the 16-bit format, and it does not improve with the multiplier.** Sixteen bits carry
about eleven significant bits, or five parts in ten thousand, and the bound already spends half of
one such step on representing the result. Raising the multiplier loosens only the part that comes
from the 32-bit arithmetic.

**Below a certain size the lane returns nothing usable.** The bound holds only where the result is
larger than `m·1e-7` plus half a representable digit. Smaller than that, the returned value becomes a
subnormal 16-bit number, and then exactly zero. **The bound is claimed over those larger arguments
and over no others.**

For 16-bit floats the point where the bound stops being useful is at arguments well inside the usable
range, and it moves with the order. At the default multiplier, orders 3, 4 and 5 still return usable
values across the whole of the large-argument range; **the claim that every argument is past the
threshold holds from order 6 upward**, and would hold at order 3 only at a multiplier around 126. The
16-bit brain-float format has the same exponent range as a 32-bit float, so it keeps returning usable
values down to about 1e-38.

## packed half — the native lane

**Within 8 of the last representable digits of the returned value**, worst measured 4.243, for
arguments at or above x = 28.984375. The bound is stated in representable digits rather than as a
size, because that is the shape of this lane's error. There is no multiplier.

**It is not a drop-in for the half lane above.** It rounds once per calculation step, where that lane
rounds once per value, so it spends more of the result's precision than that lane's bound allows: the
worst cell this lane's own suite finds is 4.243 of those digits against the half-ULP the other lane
budgets, which is **8.5 times** its representation allowance. That is a floor set by having only
eleven significant bits, not by the algorithm. The steps that pay for it are the low orders, which
are the ones callers use most.

**The arithmetic is portable rather than hardware-backed.** Each step is evaluated in double
precision and rounded once to 16 bits, which gives the same correctly rounded result a native 16-bit
instruction would. This library carries no native 16-bit backend, so what the lane delivers is 16-bit
arithmetic's *results* rather than its *speed*. **No timing is claimed for it.**

**The limit is the exponent range, and the lane covers fewer orders than its argument bound
suggests.** The bound holds where the returned value is a normal 16-bit number, that is where the
result itself is at least 2^-29. The argument at which that stops depends steeply on the order:

| Orders | Arguments served, above x = 28.984375 |
|---|---|
| 0 and 1 | the whole 16-bit range, to 65504 |
| 2 | to about 2636 |
| 3 | to about 360 |
| 4 | to about 129 |
| 5 | to about 70 |
| 6 | to about 47 |
| 7 | to about 36 |
| 8 | to about 30 |
| **9 to 32** | **none** |

**Orders 9 and above have no argument at all at which this lane claims its bound**, because the
function is already below 2^-29 at the smallest argument the lane accepts. Twenty-four of the
thirty-three orders the library supports are in that position. A caller who needs those orders wants
the double or float lane.

Rescaling each order buys range and no accuracy. The rescaling is exact, so a rescaled lane returns
bit-identical results to an unscaled one. No rescaling reaches past that limit either, because the
range the recursion spans grows like x to the power n, against a fixed exponent range.

## The multiply-add route: fused or separate

The fits above are evaluated by a Chebyshev recurrence, and every step of that recurrence is one
multiply-add. A multiply-add can be delivered two ways, and they are not the same arithmetic:

- **fused** — the product is kept exact and the sum rounds once.
- **separate** — the product rounds to the format, and then the sum rounds again: two roundings.

**The two return different bits**, so which one a build runs is part of what every bound on this page
is a claim about. All of them are the fused route's.

**A target that has the fused instruction uses it.** There the fused step is one instruction and the
separate route is two, to a less accurate answer, so the fused route is the default and a caller
building for such a target need read no further.

**A target that does not have it pays a call.** Written as a fused multiply-add, the step compiles to
a call into the platform's math library. **That call is per multiply-add, not per value.** A fit of
degree d costs d + 1 steps, and this library's fits run

| Arguments | Steps per value |
|---|---|
| x < 1.0855252345349333 | 19 or 21, by order |
| 1.0855252345349333 ≤ x < 11.899848152108484 | 25 |
| 11.899848152108484 ≤ x < 28.98933773882074 | 19, once per value: the orders above 0 come from the upward recursion |
| x ≥ 28.98933773882074 | 0 |

for the double lane — the float lane's are 11 at every argument below 11.899848152108484 — so a
plain x86-64 build, which is MSVC's default with no `/arch` flag, pays that many library calls for
every value it evaluates. **No result changes; only the path to it.** A build with the fused
instruction pays nothing extra and is unaffected.

### the alternative, and its own measured bound

    cmake -S . -B build -DBOYS_MULADD_SEPARATE=ON

That build writes the multiply-add as a bare product-plus-add, which costs no call where there is no
fused instruction to call. **It is a different arithmetic and therefore a different error**, so the
figures below were measured at that route rather than carried over from the fused one. They are the
worst cell of each lane's sweep of the committed reference grid, at the default multiplier, against
the same independent reference the fused figures use.

**They were taken on MSVC x64 at its default architecture, Release, and they name that build.** A
target that contracts the bare form never reaches the separate route at all — the request is
answered with the fused arithmetic and the table below does not apply to it. On a target that does
not contract, the measured figures here are the ones the arithmetic gives, and a rebuild of this
library on another compiler is a second measurement rather than a confirmation of this one.

| lane | region | fused | separate | bound |
|---|---|---|---|---|
| double, single | x < 1.0855252345349333 | 2.22e-16 | 2.22e-16 | 1e-15 |
| double, single | 1.0855252345349333 ≤ x < 11.899848152108484 | 3.22e-15 | 3.07e-15 | 3e-14 |
| double, single | 11.899848152108484 ≤ x < 28.98933773882074 | 9.94e-15 | 9.94e-15 | 3e-14 |
| double, single | x ≥ 28.98933773882074 | 5e-14 | 5e-14 | 5.5e-14 |
| float, single | all arguments | 1.06e-07 | 1.24e-07 | 1.5e-07 |
| float, batch | all arguments | 1.08e-07 | 1.08e-07 | 1.5e-07 |

**No lane is held back on the fused route.** Every scalar lane holds its published bound at both
routes, so the alternative is offered everywhere and no lane's figure is withdrawn. The float single
entry is the one that moves furthest, from 0.705 of its bound to 0.824 at the default multiplier: a
rise of 0.12 of a bound that is m·1.5e-7 and nothing else, which leaves 0.176 of it in hand. That is
the tightest margin on this page and the one to watch if the float fits ever change.

**Where a lane does not move, the reason differs.** Above x = 28.98933773882074 the closed form
evaluates no multiply-add at all, so both routes land on the same bits. The float batch entry's worst
cell is on the upward recursion, which is written as bare multiplies, subtractions and one division
that no multiply-add route governs. And the double lane's worst cells at the smallest arguments and
in the asymptotic region are not on a multiply-add either.

**Three lanes are outside the choice entirely.** The half lanes
evaluate in float and round to 16 bits once per value, so their multiply-adds are the float lane's —
but the 16-bit representation term dominates every cell they measure, and the route's difference is
orders below it, so the sweep finds the same worst cell on both routes. The AVX2 packed lanes name
their instruction directly, so there is no route to select and no call to avoid. The packed-half
lane's chain is one multiply and one divide per order, so it takes no fused step either.

**The separate route does not remove every call, and the remainder is deliberate.** The transform
lane's own product contains no fused step at all — its splits are exact by construction and its
reduction is a plain add — but the recurrence that builds its basis matrix is a two-rounding
subtraction by contract, and the published figure for that lane is the one that recurrence is
evaluated at. One call per step of it remains. A build cannot have both those two roundings and no
call on a target with no fused instruction, and the contract is the part that does not move.
Separately, the report answers whether a bare product-plus-add contracts in a given build by
evaluating one — once per report, not per value. A caller who needs every call gone wants a target
that has the instruction.

### what a build reports

`BoysBackends()` prints the route beside the contraction measurement, so a caller can attribute a
value to the arithmetic that produced it without reading the source:

    muladd route : scalar-fp64 fused (bare product-plus-add does not contract here)
    muladd route : scalar-fp32 fused (bare product-plus-add does not contract here)

The second line is the measurement that decides the first. **A build that contracts a bare
product-plus-add reports the fused route even when the separate route was asked for**, because on
that build the bare form *is* the fused step and the two routes deliver the same bits; the separate
route is a name a report prints only where it is really two roundings.

### The evaluation scheme: split Clenshaw or Horner

A stored fit is a polynomial, and the recurrence above is one way to sum it. The double single lane
offers a second: the same fit, at the same degree over the same interval, evaluated by Horner's rule
on the monomial form of the same coefficients. **Both are offered and neither replaces the other.**
A call site that names no scheme is compiled exactly as it was before the second one existed, so the
certified route is the default and its figures on this page are unchanged.

The two schemes sum one polynomial at the full degree, so the stored fits' figures below are the
same reading twice and differ only by the arithmetic that carried the sum.

**They are not interchangeable under a multiplier, and the two schemes' relaxed rungs are different
degree tables.** A rung truncates the table its scheme sums, and what a truncation costs is the
1-norm of the coefficients it drops *in that table*: the split Clenshaw recurrence reads the
Chebyshev coefficients, whose decay is the fit's accuracy, while Horner reads the monomial
coefficients, which are the same fit's Taylor coefficients on the piece and whose high-order end runs
larger by about 2^k. A degree the Chebyshev tail admits therefore drops a monomial tail orders of
magnitude over the rung's budget, so each scheme's rung is certified against the tail of the table it
reads — one criterion, two tables — and a degree table belongs to a basis rather than to a fit. The
monomial table admits the smaller relaxation by a wide margin: at m = 64 it is no truncation at all
on 63 of region A's 66 pieces and on 16 of the region-B seed's 33 per-order entries, and it reaches a
majority of the region-A pieces only at m = 16384. **A rung is less accuracy for less work, so a
smaller relaxation is a smaller error, not a worse one**: read the direction it runs. At m = 65536
the split Clenshaw rung delivers 3.36e-09 at its worst cell and the Horner rung 4.16e-10, both inside
m·5.5e-14, and the Horner figure is the smaller because that rung truncated less. Every rung this
library offers is served and bounded at either scheme; what a looser rung buys at Horner is bounded
by the same derivation, and the delivered figure beside each rung is the one to read.

| scheme | stored fit | degree | stored | bound, fused route | bound, separate route |
| --- | --- | --- | --- | --- | --- |
| split Clenshaw | region A (per-order fits) | 20 | 21 | 4.441e-16 | 4.441e-16 |
| split Clenshaw | region B seed | 18 | 19 | 1.421e-14 | 1.421e-14 |
| split Clenshaw | extended band seed | 24 | 25 | 2.220e-16 | 2.220e-16 |
| Horner | region A (per-order fits) | 20 | 21 | 2.220e-16 | 4.441e-16 |
| Horner | region B seed | 18 | 19 | 1.421e-14 | 1.421e-14 |
| Horner | extended band seed | 24 | 25 | 2.220e-16 | 2.220e-16 |

Each bound is the worst error a sweep over the fit's own interval reached against the 60-digit
reference, rounded up to the next power of two so that it is a bound rather than the sweep's reading.
The monomial conversion is benign for the reason the fits are: the argument never leaves [−1, 1], so
the conversion's conditioning cannot grow.

**What the scheme reaches.** The double single lane at every multiplier: the per-order region-A fits,
the region-B seed and the extended-band seed. **What it does not reach, and why.** Region C is
evaluated by its closed form and stores no fit to sum, so it is the same arithmetic under either
scheme. The half lanes, the packed region-A lane and the CUDA device lane are the split Clenshaw's
and are not offered under the other scheme: the device kernels carry the Chebyshev tables only, and
the packed region-A lane's kernel is bypassed under Horner, which costs the accelerated path and not
the value.

`BoysEvalSchemes()` and `BoysEvalSchemeFits()` answer what exists and what each scheme promises on
each stored fit in the route in force, so a caller can ask without reading the kernel, and the
accuracy gate carries one measured row per scheme and stored fit beside its own rows.

**The scheme and the fit route compose.** A fit route names the fits that serve the regions and a
scheme names the summation a fit's coefficients are read in; a call site selects the pair with one
`EvalPolicy`, and every pair of the two enumerations is carried. The figures in this section are the
Chebyshev family's, at either scheme. A route whose own fit has one stored form — the rational
minimax family's monomial numerator and denominator — evaluates that fit the same way under either
scheme, and the scheme still reaches the parts of a call the route's own fits do not serve, which
are the shipped family's: a rational-route call at the Horner scheme is the rational fits over the
intervals their rows report and the Horner sums outside them. Every argument of a pair is therefore
covered by one of the two sections' measured rows — the route's row inside its served intervals, this
section's row outside — and no pair states a bound that neither row carries.

### The packing axis: which of a call's values share a vector

A packed lane keeps four doubles in a register, and a call has to supply four of something. The two
something this library has are the two axes a call shape has: four **arguments** at one order, and
four **orders** at one argument. Neither replaces the other — they evaluate the same stored fits by
the same arithmetic, and differ in which of a call's values the vector lanes hold.

The batch entry `BoysAllOrders(nmax, x, out)` computes every order at one argument, so an
across-arguments lane has one argument to put in its four lanes. The orders axis is the axis that
entry actually has: `nmax + 1` values wide, and every one of them is a stored fit of its own. Naming
it is a field of the policy every templated double-precision entry takes, `PackAxis::kOrders`, and
the values it returns are the per-order region-A fits read one order at a time rather than reached
from a seed by a recursion. `BoysPackAxes()` reports both members with the interval each one's
packed lane evaluates.

**Which entries carry it, and which call shapes cannot.** The axis needs four orders to fill a lane,
and this library has two call shapes that have them. `BoysAllOrders` is one. `BoysAllN` is the other:
its layout is `out[k * count + i] = F_k(x[i])`, so an argument's whole order vector is what the entry
writes, and the axis is the shape it already has. A plane call naming the axis takes the entry's
per-argument path — the all-orders entry's own body — and returns that entry's values under the axis
**bit for bit**, which the entry's suite asserts with no tolerance; what it gives up is the region
grouping, which exists to feed a lane that packs four *arguments* and has nothing to group when the
call has one argument to pack. The gate measures the axis on both shapes, and both rows read the same
figures because there is one arithmetic between them.

Two calls are refused where they are named, and both are measured by compiling the call in a
configure probe rather than by quoting an assertion. `BoysFixedN` produces exactly one order at every
argument of an array: a packed lane keeps four orders of one argument, the call has one, and no
revision of that entry produces four — the axis cannot be formed on it, which is a property of the
call and not a table nobody built. A relaxed multiplier on the axis is a different kind of refusal: it
is refused because the packed orders lane evaluates every stored fit at its full degree and reads no
effective-degree table, so it carries no rung. That is a table that has not been derived, and it is
owed rather than impossible — the rung is the same dropped-tail criterion the shipped route's rungs
are truncated by, applied at the degree the lane reads.

**The domain is region A, and its bounds are the fits' own.** The orders lane covers `0 <= x < kX0`
and is certified against the per-order region-A bar, **|F̂ − F| ≤ m·1e-15**. At the certified split
Clenshaw scheme its values are the across-arguments lane's values **bit for bit** — one exact
comparison over 3,009 arguments and every order, 99,297 of 99,297 values, with no tolerance, because
a reordered step or a coefficient read one index out would still return a plausible number. Past
`kX0` the entry runs the certified scalar single lane one order at a time, so it is defined for
every argument the library accepts; that path is bit-identical to `BoysSingle`, and the gate measures
it over the whole reference grid rather than assuming it.

**The fetch decides the ranking, and the counter decides the fetch.** Two coefficient fetches
compute the same lane: one AVX2 gather, and four scalar loads joined by `_mm256_set_pd`. The two are
**bit-identical over every scheme and every value** — the lane's own test asserts it — so this is a
choice of instruction and of nothing else. Measured on one machine (Intel Core i7-9850H, Linux perf
under WSL2, `instructions:u` and `uops_retired.retire_slots:u`), 2,048,000 calls of
`BoysAllOrders(32, x, out)` over 1,024 log-uniform arguments in [1e-3, kX0) at 2,000 repetitions,
67,584,000 output values:

| route | instructions | retired slots | slots per call | slots per value |
| --- | --- | --- | --- | --- |
| shipped `BoysAllOrders` (a seed and a recursion) | 6,326,028,230 | 6,687,926,524 | 3,265.6 | 98.96 |
| orders axis, composed fetch | 5,072,522,334 | **4,719,806,211** | 2,304.6 | 69.84 |
| orders axis, gathered fetch | **4,154,092,317** | 9,550,724,437 | 4,662.9 | 141.33 |
| across-arguments lane forced onto this shape | 30,369,673,181 | 27,806,856,380 | 13,577.6 | 411.38 |

Read the two columns against each other. **The gathered variant retires the fewest instructions of
any lane here and the most slots of any lane here**: it is the fastest by instruction count and the
slowest by retired slots, and the shipped route sits between it and the composed one on both. The
same code, the same values, the same accuracy — and the ranking inverts with the counter. Retired
slots is the counter that decides, because a gather on this part of this microarchitecture is a
microcode assist rather than a single retirement, and the composed form is what ships because of it.

**The measured advantages, in the counter that decides.** The orders axis at the composed fetch
retires **4,719,806,211** slots against the shipped entry's **6,687,926,524**, which is a ratio of
**1.42**; against the across-arguments lane's **27,806,856,380** it is a ratio of **5.89**. Both are
a function of the workload's size and are quoted with it: the same pair measured over 64 arguments
per call gives 297,004,529 against 430,755,616, a ratio of 1.45, and over 4,096 it gives
18,872,789,147 against 26,699,353,538, a ratio of 1.41.

**Why, in the same units.** The composed fetch retires **4,719,806,211** slots where the gathered one
retires **9,550,724,437**, a ratio of **2.02**, so the gather costs 2,358 more retired slots per call
over the 2,048,000 calls measured. A call evaluates its 33 orders as 8 vector groups and one scalar
tail, and each group performs one fetch per stored coefficient — 21 at the first band's degree 20 —
which is 169 fetches a call here. So the gather's own excess is 2,358 / 169 = **13.95 retired slots
per fetch**, which is the mechanism rather than the whole figure.

**The re-measurement obligation.** A second packed path is re-measured whenever another axis moves,
because every figure above is a property of one build's code and one machine's microarchitecture — a
µop count is a fact about the CPU it was taken on, and nothing here generalises to another. This
section states one machine, one counter pair and one call shape, and no time: a time taken on a
loaded machine is not a measurement, and the two counters are. The route's own benchmark driver
(`benchmarks/boys_across_orders_benchmark.cpp`, `BUILD_BENCHMARKS=ON`) prints the work it did and the
worst deviation from the shipped entry beside each run, so a count can never be of a broken variant,
and its header carries the exact command that reproduces the two columns.

**What is refused.** The lane evaluates the shipped region-A piece table and reads no other family's
tables, so naming the orders axis with the rational minimax route is refused where it is named, and
the refusal is measured by a configure probe that compiles exactly that call. The plane entry and the
fixed-order entry have one order to fill a vector lane with, so the orders axis cannot be formed on
those at all — that is the entries' signature and not a body nobody built. Each limit names the
combination and what is not certified about it; none falls back silently to another axis.

## Large arguments, and the boundaries at x = 11.8998 and x = 28.9893

For x at or above 28.98933773882074 the library uses a single closed-form asymptotic result. **No
multiplier relaxes it**, and its bound holds with room to spare at every setting.

Its accuracy is set by where the asymptotic form starts being used. Moving that start point trades
accuracy against nothing: the evaluation costs the same wherever the boundary sits, so moving it
saves no work.

Below about x = 16 the asymptotic form is outside the range where it is valid, and the error grows
smoothly there, with no sudden jump. But the size of the error is real, and below that boundary it is
stated **relative to the function's own value**: the largest such error is **5.6e4**, at order 32 and
x = 14.0468, and at x = 15.936 the same quantity is **5.6e3**. The 5.5e-14 above is an absolute
figure, so the two are not on one scale and this page offers no ratio between them. What the pair
says is that inside the sub-16 part of the branch the accuracy is nowhere near the accuracy of the
fits. A caller evaluating there should expect that, and should not read the 5.5e-14 as covering it.

## What is not claimed

**Speed.** No timing taken so far supports a speed claim for any lane. Every timing available was
taken on a loaded machine, on a card with no tensor cores, and with a double-to-single throughput
ratio near 1 to 32. **Performance is an open question.** Answering it needs a timed run on an
unloaded machine and on a card chosen to show the difference; the packed-half lane's advantage in
particular is a property of the card. Until then, no lane's cost may be assumed better than
another's.

**Tensor cores.** The packed-half lane cannot use them, and the reason is its shape rather than its
precision. Its work is a chain of one multiply and one divide per order, so there is no product of
two matrices for a tensor core to compute. The region-A transform lane is matrix-shaped and does use
them where a card offers the mode; that lane's own section states what each mode can carry.
