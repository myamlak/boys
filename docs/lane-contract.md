# Accuracy: what each lane guarantees

This library evaluates the Boys function `F_n(x) = ∫₀¹ t^(2n) e^(−x t²) dt`, for orders n = 0 to 32
and arguments x ≥ 0.

It offers several lanes, which differ in precision and in cost. This page states each lane's error
bound, the arguments the bound covers, and where the lane stops being usable.

Every bound here is stated as an **absolute** error — |computed − true| — except where an entry says
otherwise. The packed-half lane's bound is relative, and the half lanes' bound is absolute plus a
term that depends on the size of the result.

Every figure on this page is measured by a program in this tree, and the command below prints the
comparison for the machine you run it on:

    cmake --build <build> --target boys-accuracy-gate
    <build>/Release/boys-accuracy-gate --strict      # the config directory is your generator's

It sweeps the documented entries over the committed reference grid and prints, lane by lane and
region by region, the worst error the lane delivered beside the bound claimed, ending in
`PASS: every documented claim met at this revision` or in the numbers of the claims that did not hold
and a non-zero status. `--per-order` extends it to every order and `--probe n x` to a single cell.
`ctest` runs the same binary as one of its tests, but a passing `ctest` prints only how long the test
took.

A caller that has chosen a precision and does not want to choose an axis wants one name, and the
section *The default policy, per precision and per device* states, for each precision and for the
device lane, what its default selects and the bound it carries — and where those settings come from.

## How the argument affects the bound

The library evaluates the function differently at different argument sizes, and the error bound
differs with it:

| Arguments | Region | How they are evaluated |
|---|---|---|
| x < 11.899848152108484 | A | stored polynomial fits |
| 11.899848152108484 ≤ x < 28.98933773882074 | B | a stored fit at the lowest order, then upward recursion |
| x ≥ 28.98933773882074 | C | a closed-form asymptotic result |

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

**The route is carried on every entry.** `BoysSingle`, `BoysAllOrders`, `BoysAllN` and `BoysFixedN`
all take the route the policy names, and each answers with the route's own fits rather than with the
shipped ones under its name: the two shapes that reach their values by a path of their own — the
plane entry's region-grouped path and the fixed-order entry's shaped path — are the shipped route's,
and a call naming the rational route is served by the per-argument body instead, which reads its fit
from the policy. The gate measures the carriage as a difference in the values rather than as a
sentence about the surface: naming the route changes what four entries return over the interval the
route's rows cover, and naming the default changes nothing.

**What the carriage delivers.** `BoysAllN` and `BoysFixedN` are swept over the whole committed grid,
every order, and judged against the named route's own bar for the region the argument falls in —
113,388 cells, worst delivered 5e-14 at n = 32, x = 28.98933773882074, no cell over. That bar is the
route's, not the entry's, so the row is the stronger of the two statements: the route's region-A row
promises 3e-14 where the entry promises 5.5e-14, and the entry holds the tighter figure over the
region, including the arguments below the route's own selector where the shipped lane answers.

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
0.057 where the envelope binds. A piece is a cut of the region and not a property of a family, so the
rational route is fitted over these same pieces, and a piece the family's own degree pair could not
hold the target on is bisected: the result is 311 pieces again at 2611 stored coefficients, one
evaluation reading 6 to 9 of them where the Chebyshev fits on the same pieces read 11. The two
partitions as tables, the default route's counts beside them:

| | `FitGranularity::kShipped` | `FitGranularity::kNarrow` |
| --- | --- | --- |
| stored coefficients | 1339 (1320 region A + 19 region B) | 3476 (3421 + 55) |
| stored rows | 67 (66 pieces + 1 seed) | 316 (311 + 5) |
| degree | 20 and 18 in region A, 18 in region B | 10 everywhere |
| read per evaluation | 19 to 21 | 11 |

**The rational route's narrow table, and what it adds to those counts.** The route fitted over the
narrow cut stores 2646 coefficients in the same 316 rows — 2611 over region A and 35 over region B —
against the 891 its shipped region-A cover and its 12-coefficient shipped seed cost, and a region-A
evaluation reads 6 to 9 coefficients from a piece rather than the 11 the Chebyshev fits on the same
pieces read. Its region-A pieces are fitted to the target the shipped rational table was fitted to and
published at that route's 3e-14 bar, of which they deliver 2.42365e-14; its
region-B seed is published at the seed's 5e-14 and delivers 4.09566e-14, and it is five pieces with
their own degree pairs rather than one numerator/denominator pair over the region, because it is the
narrow cut that makes a region-B seed of several pieces possible at all. Every one of those figures
is the worse of the two multiply-add routes, measured in the arithmetic the kernel runs on the
coefficients as they are stored, which is the criterion this lane's fits are accepted under and not
the exact reading the shipped tables were accepted on.

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

**The option is exposed, certified, and refused only where a table or a kernel is still owed.**
`FitGranularity::kNarrow` is a field of `EvalPolicy` and a name the report can print. Its tables are
the generated header's, reachable through the same entries as the shipped partition, and its
certification is the accuracy gate's own block: 252 rows — one per partition, scheme, call shape and
accuracy rung — each judged against the bar the published table holds for the cell it ran in, times
the rung's multiplier, with the worst cell named on the row, and three more for the rational route
over the narrow partition — 255 in all. The call shapes are the stored fits read directly, the batch
entry below the band (where the seeding fallback is taken), the batch and plane entries over the band
and below, the batch entry over region B, the single-order entry at region A's cell, and the
single-order, fixed-order and plane entries over the whole grid. Three rows move between the
partitions, and every other row is the same figure under either name at the reference multiplier:

| row, worst over both schemes | shipped | narrow | bar |
| --- | --- | --- | --- |
| batch entry, below band | 1.11022e-16 | 1.23165e-16 (n=32, x=1e-08) | 5.5e-14 |
| stored seed, region B | 9.9365e-15 | 7.21645e-16 (n=0, x=11.8998) | 3e-14 |
| batch entry, region B | 9.9365e-15 | 7.51675e-16 (n=32, x=11.8998) | 5.5e-14 |
| stored fits and single entry, region A | 2.22045e-16 | 2.22045e-16 | 1e-15 |
| batch and plane entry, band and below | 3.21618e-15 (n=16, x=4.89985) | 3.21618e-15 | 5.5e-14 |
| single, fixed-order and plane entry, whole grid | 5e-14 (n=32, x=28.9893) | 5e-14 | per region (m × B_region) |
| rational route, region-A pieces | 2.46e-14 (shipped cover) | 2.21663e-14 (n=2, x=9.88955) | 3e-14 |
| rational route, region-B seed | 4.46e-14 (shipped seed) | 4.12448e-14 (n=0, x=12) | 5e-14 |
| rational route, batch entry | — | 5e-14 (n=32, x=28.9893) | 5.5e-14 |

**The rungs are served on both partitions, and the narrow rows are derived from the narrow table.**
A relaxed rung is a cut of a fit's stored coefficients: the effective-degree criterion is
`d'(m) = min { d' : Delta(d') * A <= (m - 1) * B_region }` over `Delta(d')` the sum of the dropped
tail's magnitudes, `A` the amplification the region's reading carries and `B_region` the bar the
region is held to. The same criterion, applied to the narrow table's own coefficients, is what a
narrow rung is: a compile-time derivation over numbers the generated header already stores, so
nothing had to be regenerated and the shipped table's bytes did not move. What it delivers is the
gate's own reading, over all seven rungs at both schemes: the worst figure any row comes in at is
0.909 of the bar it promised at `m = 1`, 0.959 at `m = 64`, 0.997 at `m = 256`, 0.978 at `m = 1024`,
0.995 at `m = 4096`, 0.992 at `m = 16384` and 0.991 at `m = 65536`, every rung inside its budget on
every row. A relaxed narrow rung trades accuracy for work exactly as a relaxed shipped rung does: at
`m = 65536` the narrow partition's stored region-A fits deliver 6.49285e-11 against the 6.5536e-11
they promised, where the same row of the shipped partition at that rung delivers 6.44079e-11 against
the same bar, and the degree one evaluation reads at that rung is 1 to 8 over the narrow pieces
against 10 to 14 over the shipped ones, where at `m = 1` they are 10 and 18 to 20.

**All 255 rows are met and the books around them do not move.** The last three rows are the rational
route's own narrow fits, judged against the route's published bars rather than the Chebyshev rows'
figures: the shipped cover of region A delivers 2.46e-14 and the narrow pieces 2.21663e-14, the
shipped seed of region B delivers 4.46e-14 and the five narrow seed pieces 4.12448e-14, and the batch
entry read through the narrow route lands at 5e-14 against the batch lane's 5.5e-14. The block is
counted apart from the lane book and from the scheme rows, and those read what they read before it
existed: 39 of 39 claims,
44 of 44 scheme rows, and the combinations book 28 of 28 with nothing owed. What does move is the
option space, 28 members to 30 — the two new ones are the partitions themselves, each measured over
289444 cells at both schemes with 150049 of them reading differently under the other partition and
none over the bar its row is judged at. The block reports its own carrying fraction, 527857 of 649327
cells (81.3%), as the cells able to discriminate; the other 121470 carry a bound at least as large as
the value itself, so no error can exceed them, and they are not counted in the rows above.

**What is carried, and what is still refused.** The narrow partition is a field of `EvalPolicy` and
a name the report prints, and every double-precision entry this library has reads it: the stored fits
directly, the batch entry's seeding fallback, the single-order entry, the plane entry, and the
across-orders packed lane. The relaxed rungs `m > 1` are carried on it too — the criterion is the
same one, measured against the partition's own pieces rather than against the shipped rows, and its
table is derived over them — so a rung of the narrow partition is a rung of *it* and not the shipped
table truncated. Four combinations are still refused, at compile time and where they are named: the
rational route over the narrow partition on the packing axis, whose lane steps one order's
coefficients to the next at a fixed stride and reads no per-order table of the route's pairs; the
narrow partition on the single-precision lanes past the reference multiplier, which hold one
coefficient set and one degree table and so have no narrow rung table to cut; the same limit on the
rational route's own rung table, which is derived from the shipped pairs; and the single-precision
lanes' across-orders packed entry, which names no partition at all. Each is a static assertion with
the reason, and each names the table or the kernel it would need — a packed kernel over the route's
narrow pairs, a narrow float rung table, a degree table for the narrow rational pairs, a packed
float kernel that reads a partition. None falls back: the two partitions are different fits of the
same function over the same interval, so a substitution would return the shipped numbers under the
narrow partition's name. Each is unbuilt work rather than an unavailable option, and each is named
where it is refused so that it can be counted.
## float

**At most 1.5e-7, absolute and everywhere.**

The same multiplier applies. The batch entry computes a starting value in double precision and then
recurses in single.

**The limit is the number format.** A 32-bit float carries about seven significant digits, which is
about 6e-8 relative. The bound is 1.5e-7 absolute, so for values of order one the two are within a
factor of about two and a half of each other. Much smaller values are bounded loosely, because an
absolute bound says less about a small number than a relative one would.

**The single entry carries two fit routes, and both hold the bound above.** `BoysSingleF32WithRoute`
is `BoysSingleF32` with the fits that supply the lane's region-A seed and region-B seed selected
instead of fixed, and `BoysFitRoutesF32` reports them. The measured column is the gate's own sweep of
the committed reference, every order and every sample of the row's interval.

**The same two routes are open at compile time, on the policy, and there the scheme comes with
them.** The lane's `BoysSingleF32`, `BoysAllOrdersF32` and `BoysAllNF32` take the `EvalPolicy` the
double lane's entries take, so the route is a template argument as well as a run-time selector, and
the scheme — the choice between the Chebyshev table and the monomial form of the same fits — is
offered beside it. At the reference multiplier every pair this lane stores is carried, and the gate
measures the two entries with a policy as ten rows, one per policy for each region the policy's fits
serve — the six the lane shipped, plus the narrow partition's four — 294690 comparison cells with no
row measured over no argument, and no narrow row of them outside its bar in either multiply-add
route's build. The float lane's Horner reading is 7.68e-08 at its worst cell (order 0, x = 0.553691) over
region A and 2.22e-08 (order 32, x = 11.8998) over region B, against the lane's 1.5e-07 bar; the
rational route's and the shipped route's split Clenshaw figures are the table's above, unchanged.

**A relaxed rung keeps both of them.** The rung's degrees are derived from the table the policy
actually reads: under the Chebyshev route each per-order fit is cut from whichever of the two stored
forms the scheme sums, under the rational route from the tail of the pair that route stores — the
numerator and the denominator cut together — and region B's seed from its own table at the budget the
policy names. The gate measures every pair on both entries, at the tier's first rung and at its last
and again at the fp16/bf16 budget, as thirty-six rows judged against `m · 1.5e-7`: 530442 comparison
cells, none of them outside the row's bar and no row measured over no argument, at the fused
multiply-add route and at the separate one alike. The tightest row is the single entry's Horner row at
`m = 65536`, 9.82766e-03 against 9.8304e-03, which is 0.9997 of that rung's bar at order 7,
x = 1e-12; the separate route delivers the same figure, so that is the criterion's own cut —
`(m − 1) · 1.5e-7` aimed at the truncation and the remainder left to the fit's `m = 1` error — and not
a rounding. Every other row is at or below 0.98 of its bar, and the worst of the relaxed pair is
0.977, the single entry's split Clenshaw row over region B at `m = 64`.

| Route | Region | Interval | Stored | Measured | Bar |
|---|---|---|---|---|---|
| chebyshev (default) | A | 0 | 1067 | 1.24e-07 | 1.5e-07 |
| rational minimax | A | 0 | 465 | 1.44e-07 | 1.5e-07 |
| chebyshev (default) | B | 11.899848152108484 | 11 | 3.14e-08 | 1.5e-07 |
| rational minimax | B | 11.899848152108484 | 6 | 7.50e-08 | 1.5e-07 |

Each measured figure is the worse of the lane's two multiply-add routes: a build runs one of the
two, so a figure that covers one of them understates the other. The gate prints each row's published
figure beside the worst it measures itself.

Region A's rows count the lane's whole per-order table over [0, 11.899848152108484), 97 pieces
against the rational route's 47, and both routes cover that interval from zero: this lane reads each
order from its own fit across the region, so it has no band boundary at which a selector would take
over, and the row does not name one.

**Both routes are open at the narrow partition as well**, which is a second cut of the same two
regions at degree 6: 218 pieces over region A and two over region B, with each route storing its own
fit on that cut. The Chebyshev route stores 1526 coefficients either way over region A and 14 over
region B, the rational route 1238 over region A and 11 over region B, and the gate's single-entry
policy rows measure them at 1.02681e-07 and 1.29916e-07 for the Chebyshev pair and 1.00057e-07 and
2.99288e-08 for the rational pair, over region A and region B respectively, against the lane's
1.5e-07 bar. The narrow rows are the same figures under the two multiply-add routes with one
exception, the rational route's region-B row, which reads 2.99288e-08 with the multiply-add fused and
4.43557e-08 with it separate — the figure the generated header publishes for that seed is the worse of
the two, 3.90533e-08 fused and 2.92450e-08 separate, and the gate reads the route the build actually
runs. Region B's seed is where the narrowing costs rather than saves: 14 stored against the shipped
seed's 11 on the Chebyshev route, and 11 against 6 on the rational one, and a call there reads 7
coefficients where the shipped seed reads 11. All four narrow rows are inside their bar under either
route.

**The stored counts are upper bounds, not minima.** Each count is the first the degree scan found
holding the target, not the family's minimum, so a cheaper cover may exist. **The two routes are a
trade and not a ranking**: region A stores 465 coefficients against 1067 and delivers 1.44e-07
against 1.24e-07, and region B stores 6 against 11 and delivers 7.50e-08 against 3.14e-08. Half the
coefficients at more error is worth having on a machine that pays for coefficient fetches and not on
one that does not, and no measurement here ranks the two.

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

### the same route, on a host that contracts

The fused column above is one measurement of one arithmetic, and it is not the whole of the fused
route. The bounds on this page are bounds on the arithmetic; the figure an arithmetic *delivers* is
a second thing, and the two come apart here because not every multiply-add in these fits is written
as a fused step. The recurrence bodies spell the step as a bare product-plus-add where that is what
they mean, and a compiler holding the contraction licence is free to fuse that form into one
rounding — which gcc and clang do by default wherever the target can form the instruction, and MSVC
does not under its default `/fp:precise`. So a build that
contracts evaluates **different arithmetic from one that does not**, on the same route, and a figure
measured on one is not a figure for the other.

**The architecture name does not decide this.** It is a property of the compiler and its flags, so it
is asked rather than assumed: the configure step evaluates a bare product-plus-add in the build and
compiles the claims against that answer, and `BoysBackends()` asks the same question again at run
time so a caller can attribute a value to the arithmetic that produced it (the section below prints
both lines). gcc and AppleClang contract one on arm64 and MSVC does not, on either architecture, so
the MSVC arm64 build delivers the x86-64 figures. Only part of the contract moves, and this is the
whole of it, at the default multiplier, against the same reference the table above uses:

| lane | region | contracted | not contracted | bound |
|---|---|---|---|---|
| double, single | x < 1.0855252345349333 | 2.22e-16 | 2.22e-16 | 1e-15 |
| double, single | 1.0855252345349333 ≤ x < 11.899848152108484 | 3.29e-15 | 3.22e-15 | 3e-14 |
| double, single | 11.899848152108484 ≤ x < 28.98933773882074 | 9.94e-15 | 9.94e-15 | 3e-14 |
| double, single | x ≥ 28.98933773882074 | 5e-14 | 5e-14 | 5.5e-14 |
| float, single | all arguments | 1.29e-07 | 1.06e-07 | 1.5e-07 |
| float, batch | all arguments | 1.29e-07 | 1.08e-07 | 1.5e-07 |

The double lane's first, third and fourth rows do not move at all: the same worst cell, to the digit,
on both arithmetics. What moves is its band between 1.0855 and 11.8998, whose worst cell is n = 16 at
x = 4.89985 without contraction and n = 32 at x = 10.7836 with it, and both float lanes, whose worst
cell is n = 0 at x = 0.072854 without contraction and n = 32 at x = 11.9447 with it. The
not-contracted column was measured on four
configurations — MSVC on arm64, MSVC on x86-64, clang on x86-64 and AppleClang on x86-64 — which
agree to the digit, two architectures and three compilers reaching one figure. The contracted column
was measured on AppleClang on arm64, and reproduced on x86-64 by building with `-mfma`: the same six
figures to the digit, and the same worst cell down to its order and argument, from two compilers on
two architectures. That is the pair worth reading together, because it is what the licence turns on —
gcc and clang contract a bare product-plus-add when the target can form one, and only one of the two
architectures can without being told to.

The one configuration that contributes a measurement but no printed figures is gcc on arm64. It
reports `contract.this-tu.fp64=1` — it runs the contracted arithmetic — and its consumer check parts
company with the single lane over 10 of the same 9,240 comparisons, nine at the second element of a
strided call and one at the first, which is one of the two ways a contracting build has been seen to
take the licence. Its gate is green: `boys-accuracy-gate --strict` is a test in its ctest run and
passes there. What is missing is the printed figure alone — ctest shows a passing test's output only
under `-V`, and that leg does not carry the reading step the other five do — and the column it would
fill is the contracted one, which the two measurements above already give.

**Every bound on this page holds on both.** The contracted arithmetic's 3.29e-15 is 0.11 of its 3e-14
and its 1.29e-07 is 0.859 of its 1.5e-07, and no cell of any lane exceeded its bound on any of the
six configurations. Five of them print `39 of 39` documented claims met; on the sixth, gcc on arm64,
the gate is the `boys-accuracy-gate --strict` test in its ctest run, and that test passes. Its
consumer check prints 55 rule rows and every one reports zero cells above its bound. What a caller
reads — the bound — is the same on either arithmetic, which is why a row is judged against the same
figure either way and a caller who needs to know which arithmetic produced a value asks the report
rather than the architecture.

**The one claim that is not a bound moves with it.** `BoysFixedN` returns `BoysSingle`'s value at the
same order and argument, and the two run the same recurrence from the same source. Where the build
does not contract that source is one arithmetic and the two agree **bit for bit** — asserted, with no
tolerance, over 9,240 comparisons of the committed grid. Contracting does not license the same
arithmetic at two call sites, and a two-argument strided call is not the same generated code as a
one-argument call, so it permits the two to part company in the last place — though it does not
require it. On gcc, on arm64 and reproduced on x86-64 with `-mfma`, **10 of those 9,240 comparisons
differ**: nine at the second element of a strided call and one at the first. On clang with `-mfma` on
x86-64, and on AppleClang on arm64, which contract too, all 9,240 agree. The element-wise shape
agrees at all 8,712 of its comparisons on every one of those builds.

So the exactness is a property the arithmetic guarantees on one side of the licence and only permits
on the other, and the bound is what the entry promises either way. That is why the bound is asserted
on every build and the offsets are asserted where the arithmetic guarantees them, while the counts
are printed on all of them: the consumer entry reports how many comparisons it made and how many
agreed, so a host that takes the licence shows a figure rather than passing quietly, and no host is
asserted against an arithmetic the licence let it avoid.

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

A stored fit is a polynomial, and the recurrence above is one way to sum it. The single lanes offer a
second: the same fit, at the same degree over the same interval, evaluated by Horner's rule
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
the region-B seed and the extended-band seed. It reaches the float lane's single and batch entries as
well, at the reference multiplier: those entries read the same pair of tables through their own
policy, and the float lane's Chebyshev fits are stored in both forms as the double lane's are.
**What it does not reach, and why.** Region C is evaluated by its closed form and stores no fit to
sum, so it is the same arithmetic under either scheme. The half lanes, the packed region-A lane and
the CUDA device lane are the split Clenshaw's and are not offered under the other scheme: the device
kernels carry the Chebyshev tables only, and the packed region-A lane's kernel is bypassed under
Horner, which costs the accelerated path and not the value. On the float lane the scheme reaches
every multiplier: the batch entry's region-A seed is the double lane's fit at the scheme the policy
names, and the single entry's fits and the batch entry's region-B seed are this lane's own, cut from
the monomial form when Horner is named.

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

One call is refused where it is named, and it is measured by compiling the call in a configure probe
rather than by quoting an assertion. `BoysFixedN` produces exactly one order at every argument of an
array: a packed lane keeps four orders of one argument, the call has one, and no revision of that
entry produces four — the axis cannot be formed on it, which is a property of the call and not a table
nobody built.

The single-precision entries are refused for the opposite reason, and the two are not the same debt.
`BoysAllOrdersF32(nmax, x, out)` produces nmax + 1 orders of one argument, which is exactly what a
packed lane would hold, so the axis names real work on that shape; what the lane does not have is a
packed body — the AVX2 tier is double and half only — so a policy naming the axis there would be read
and then ignored, and the entry refuses it at the call site instead. `BoysSingleF32` answers one order
at one argument and has no lane to fill on either axis. The gate's refusal record separates the two
kinds and counts the first as outstanding work rather than as a limit of the call.

A relaxed multiplier on the axis, and a route other than the shipped one, were limits of the other
kind — a table the lane did not carry — and both are built. The effective-degree table the lane reads
at a rung is the same dropped-tail criterion this library's other rungs are truncated by, applied at
the degree the lane reads; and the rational route's region-A fits cover the same per-order intervals
as the shipped piece table, so the lane reads them where it read that one. The gate's packing book
measures every rung the tier enumeration declares, on both routes, through both entries that carry the
axis, each against the figure that combination documents — and every rung again on the narrow
partition, on the shipped route.

**The narrow partition, where the axis had no kernel.** The lane's third limit looked structural and
was not. The shipped fetch steps from one order's coefficients to the next at a fixed stride, which
works because every order's piece in the shipped region-A table shares its interval and its degree
with the ones beside it; the narrow partition's pieces are cut per order, 311 pieces at 10 coefficients
each with their own intervals, so no such stride exists. A stride is one way to fill a register, not a
requirement of the axis: the lane fills it the other way, by fetching each of the four orders it holds
its own piece's start and its own coefficients — one gather per stored coefficient rather than one
stride per group — and summing the four as one vector. The value the row promises is unchanged, and so
is the figure: **|F̂ − F| ≤ 1e-15 over `0 <= x < kX0`** at m = 1, the per-order region-A bar.

**Two lanes, two mappings, and the reference as the third party.** The narrow partition cannot carry
the bit-for-bit row above, and it is worth saying why rather than letting a reader find the gap. That
identity is between two lanes that map the argument the same way — `fma(x - a, 2/(b - a), -1)`. The
per-order narrow lane is the *scalar* entry, whose mapping is `2 (x - a) / (b - a) - 1`, and the two
roundings are not the same bits. So the narrow axis is measured as a difference in values from the
per-order narrow lane, and where the two part by more than 1e-15 the 80-digit reference says which
reading is the right one. Measured over the committed reference grid's 21,285 region-A cells:

| quantity | split Clenshaw | Horner |
| --- | --- | --- |
| axis, worst against the reference | 2.22e-16 | 2.22e-16 |
| per-order narrow lane, worst against the reference | 3.22e-15 | 3.22e-15 |
| cells where the two lanes part by more than 1e-15 | 32 | 30 |
| ... of those, the reference is on the axis's side | 32 | 30 |
| ... of those, the axis is inside its 1e-15 | 32 | 30 |
| ... of those, the per-order lane is outside its own figure | 0 | 0 |

Every parting cell sits in the extended band, where the single lane's own figure is 3e-14 and not
1e-15, and the per-order lane is inside that figure at all of them: the two are held to different
budgets because they are different lanes. The worst of the parting cells, at n = 16 and
x = 4.8998472055064735, has the axis **5.82e-17** from the reference and the per-order lane
**3.22e-15**, and the same reference the axis is measured against is the one this page measures every
other region-A row at.

**The domain is region A, and its bounds are the fits' own.** The orders lane covers `0 <= x < kX0`
and is certified against the per-order region-A bar, **|F̂ − F| ≤ m·1e-15**. At the certified split
Clenshaw scheme its values are the across-arguments lane's values **bit for bit** — one exact
comparison over 3,009 arguments and every order, 99,297 of 99,297 values, with no tolerance, because
a reordered step or a coefficient read one index out would still return a plausible number. That row
is the shipped partition's, for the reason in the paragraph above; on the narrow partition the same
question is answered by the reference table instead. Past
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

**The narrow partition's lane, in the same units.** The narrow partition's orders axis is measured
against the loop of four scalar evaluations it replaces — what the axis would otherwise be at that
partition — over the same call, the same machine and the same counter pair. One variant per process,
so no pair is subtracted across runs:

| variant | instructions | retired slots | slots per call | slots per value |
| --- | --- | --- | --- | --- |
| narrow partition, orders axis, split Clenshaw | 6,061,196,228 | 8,887,109,142 | 4,339.4 | 131.50 |
| the four scalar orders it replaces, same scheme | 19,665,326,588 | 21,004,501,686 | 10,256.1 | 310.79 |
| narrow partition, orders axis, Horner | 5,149,807,654 | 7,963,896,750 | 3,888.6 | 117.84 |
| the four scalar orders it replaces, same scheme | 14,361,758,765 | 15,037,495,121 | 7,342.5 | 222.50 |

The packed lane retires **2.36×** fewer slots and **3.24×** fewer instructions than the loop at split
Clenshaw, and **1.89×** and **2.79×** at Horner: on this machine the gathered fetch is cheaper than
the four scalar calls it replaces, so the price the gather pays against the composed fetch above is
still a price paid against a loop that pays more. It is a vector path by source and not by count
alone — four orders in one `__m256d`, the group's four piece offsets in one `__m128i`, and one
`_mm256_i32gather_pd` per stored coefficient index, or `deg + 1` a group, where `deg` is the largest
of the group's four certified degrees. A call evaluates its 33 orders as 8 vector groups and one
scalar tail, so at the stored degree 10 it performs 11 gathers a group and **88 a call**, against the
shipped partition's 169 — and it retires 623,566,216 fewer slots over the same 2,048,000 calls while
also paying a per-lane piece scan and mapping the shipped lane does not, so those 81 gathers a call
are worth **at least 3.76 slots each**, the lane's extra geometry already inside that difference.

What the two lanes cannot be compared on is identity: the narrow axis reads its fits under the
shipped lane's `fma` mapping and the per-order narrow lane under `2 (x - a) / (b - a) - 1`, so the
narrow axis is held to the reference above instead of to a bit-for-bit row, for the mapping reason
given with it.

**The composed row, re-measured at this revision.** The composed and gathered rows of the fetch table
were taken before the second partition's tables and the effective-degree cuts landed, and the
re-measurement obligation below is this section's own. Measured again over the same calls and the
same shape: the composed variant now retires **6,570,938,098** instructions and **6,845,085,236**
slots, in three runs agreeing to within 232 instructions and 0.02% of slots, where the table says
5,072,522,334 and 4,719,806,211. The gathered row reproduces to within a percent (4,115,182,612 /
9,510,675,358) and the shipped entry's row reproduces (6,326,030,697 instructions, the published
figure to seven digits, and 6,729,655,927 slots, within a percent). The **1.42** and **2.02** ratios
drawn from the composed figure therefore belong to the revision they were measured at, and the
composed row is due a re-derivation at the current tables. The row is left as measured rather than
quietly restated: it is not the figure this change moves, and restating it here would hide the
distance a reader measuring the same variant would find.

**The re-measurement obligation.** A second packed path is re-measured whenever another axis moves,
because every figure above is a property of one build's code and one machine's microarchitecture — a
µop count is a fact about the CPU it was taken on, and nothing here generalises to another. This
section states one machine, one counter pair and one call shape, and no time: a time taken on a
loaded machine is not a measurement, and the two counters are. The route's own benchmark driver
(`benchmarks/boys_across_orders_benchmark.cpp`, `BUILD_BENCHMARKS=ON`) prints the work it did and the
worst deviation from the shipped entry beside each run, so a count can never be of a broken variant,
and its header carries the exact command that reproduces the two columns.

**What the lane reads.** The lane evaluates the stored fit the policy names: the shipped Chebyshev
piece table on the shipped route, the rational route's region-A pairs where that route is named —
their pieces cover the same per-order intervals — and each fit at the degree the multiplier's
effective-degree table cuts it to. `BoysFixedN` has one order to fill a vector lane with, so the
orders axis cannot be formed on it at all — that is the entry's signature and not a body nobody
built. Each limit names the combination and what is not certified about it; none falls back silently
to another axis.

### The same axis on the single-precision engines: eight orders to a register

The double lane's width is set by its arithmetic path — `avx2-fp64`, four doubles to a `__m256` — and
the single-precision path `avx2-fp32` holds eight floats in the same register, so the float lane's
orders axis is **eight wide**. The width is not the interesting difference. The tables are.

**One premise the double lane rests on is false here.** Every one of the double lane's region-A fits
is cut at the same boundaries to the same degree, so one argument selects one piece index, one mapped
argument and one coefficient stride for the whole vector. The float table gives each order its own
cover: order 0 is cut into two pieces and order 14 into three, and a break is not shared between
orders. A fixed argument therefore selects a *different* piece in each of the eight lanes, and the
offset from one lane's coefficients to the next is not a stride at all. The float lane's vector
carries the per-lane geometry instead — one mapped argument and one coefficient base per lane — and
for the same reason it needs no uniform-piece premise to test before it runs.

**What the eight lanes do share is the degree**, because the split Clenshaw's even/odd structure
belongs to the degree and not to a coefficient. The group runs at its lanes' largest degree, and a
lane whose own fit is cut shorter reads zeros above its own cut. That is not an approximation of the
lane's polynomial; it *is* the lane's polynomial, and down the recurrence it is the lane's own
arithmetic: the extra top step has an exact zero for both of its terms, so it hands the lane the
state its own-degree summation would have started from. The claim is therefore the strong one — the
packed value is the per-order value — and it is asserted as bit-identity rather than as a tolerance,
on the same reasoning the double lane's is.

**The direction of the difference between the two lanes is the multiply-add route.** The packed lane
names its own instruction (`vfmadd`) and is contraction-free whatever the build says, so it performs
one rounding where a scalar build with `BOYS_MULADD_SEPARATE=ON` performs two. On the default fused
build the two are the same arithmetic and the values are the same bits — **0 of 999,240** swept values
differ (five policies at the reference multiplier, three rungs, every `nmax` from 0 to 32 and five
arguments past the lane's own interval) — so on that build the agreement is asserted without a
tolerance. On the two-rounding build they part, and a difference between two lanes is then
not a statement about either one's accuracy, so both are measured against the double lane, whose own
error is a double lane's and is a reference at this scale:

| policy at m = 1, two-rounding build | packed, from the double lane | per-order, from the double lane | packed versus per-order | bar |
| --- | --- | --- | --- | --- |
| chebyshev, split Clenshaw | 1.297e-07 | 1.297e-07 | 1.192e-07 | 1.5e-07 |
| chebyshev, Horner | 9.029e-08 | 9.029e-08 | 1.192e-07 | 1.5e-07 |
| rational minimax, split Clenshaw | **1.309e-07** | **1.727e-07** | 1.788e-07 | 1.5e-07 |
| rational minimax, Horner | 1.309e-07 | 1.727e-07 | 1.788e-07 | 1.5e-07 |

Read the middle three columns together. **On the Chebyshev route the two lanes are one number and it
is inside the bar. On the rational route they are not one number, and the lane that is outside the bar
is the per-order one**: 1.727e-07 against the 1.5e-07 that route is certified against, where the
packed lane reads 1.309e-07 and is inside it. The pair difference of 1.788e-07 is the two-rounding
scalar lane's excess and not the packed lane's, which is why the lane's test measures against the
double lane on that build rather than against its sibling: a sibling that is itself over the bar
cannot be the reference a bar is read against. This is the same defect class as the gate's
pre-existing red row on the same build — `float.route.delivered float.policy.single`, 1.56625e-07
against 1.5e-07 at `n = 16, x = 0.0781091`, byte-identical before and after this axis exists — which
is a property of the region-A rational fit under two-rounding arithmetic and not of this lane.

On the fused build, which is the default, the packed lane's own error is 1.297e-07 on the Chebyshev
route and 1.309e-07 on the rational one, both inside the bar, and the per-order lane's is the same
number because the two lanes are one arithmetic there. The gate's float book carries a row for this
axis at m = 1, measured against the committed mpmath reference grid at the lane's own bar: **56,694
cells**, delivered **1.06e-07** against **1.5e-07**, a ratio of **0.705**, with the worst cell at
`n = 0, x = 0.072854` — the same figure on both multiply-add routes. The per-order entry's row over
the same cells delivers 1.08e-07, so the axis moves which lane evaluates and not the figure the entry
meets.

**Outside region A the axis is not what is running.** Past `x = 11.8998` the entry answers from the
certified scalar single lane at the policy the caller named, one order at a time, which is the same
thing the per-order entry answers with — the axis names which lane runs inside its own interval and
nothing beyond it. That fallback is compiled into the library's translation unit while a caller's
per-order reference is compiled into the caller's, and the two agreeing bit for bit is a property of
the arithmetic they compile and not of the dispatch. It did not hold, and the axis's own contract
test is what showed it: the region-B recurrence in the single-order float body was written as a bare
product and difference, which a compiler contracts where it is allowed to and not where it is not, so
the entry's own loop and the identical call beside it — two inlined copies of one entry in one unit —
could return adjacent values. Measured over region B, the entry's fallback and the per-order lane
parted by at most **4.04e-09** at the worst (`x = 12.8164, n = 32`, from 6.4190026e-08 against
6.8225951e-08), which is 2.7% of the lane's bar; in the contract sweep, where the only region-B
argument is the first one, 15 of the 33 orders at `x = 11.8998` differed and no other argument did.
Region C, which reads no coefficient and names no multiply-add, differed in nothing, and order 0 at
every given argument differed in nothing either — it is the region-B seed and takes no recurrence
step. The step is now spelled
as the backend's two-rounding multiply-subtract, which rounds the product and then the difference on
every build, so every unit that computes this recurrence computes the same number. No figure the
library publishes moves: the gate's output is byte-identical before and after the change, and so is
every row of its float book. The same bare spelling survives in the float lane's *batch* recurrences,
which no entry this axis reaches — it is left where it is rather than moved under a figure that is
already published.

**What the packed lane costs, and against what.** Measured on one machine (Intel Core i7-9850H, Linux
perf under WSL2, `instructions:u` and `uops_retired.retire_slots:u`), 2,048,000 calls of
`BoysAllOrdersF32(32, x, out)` over 1,024 log-uniform arguments in [1e-3, kX0) at 2,000 repetitions,
67,584,000 output values, one variant per run:

| route | instructions | retired slots | slots per call | slots per value |
| --- | --- | --- | --- | --- |
| shipped `BoysAllOrdersF32` (a seed and a recursion) | **1,774,847,463** | **1,859,001,682** | 907.7 | 27.51 |
| the per-order loop this axis replaces | 17,004,791,935 | 18,024,497,457 | 8,801.0 | 266.71 |
| orders axis, gathered fetch | 5,521,695,364 | 6,856,442,627 | 3,347.9 | 101.45 |
| orders axis, composed fetch | 8,562,138,514 | 8,982,817,264 | 4,386.1 | 132.90 |
| orders axis, the entry the policy reaches | 5,548,334,305 | 6,897,521,696 | 3,367.9 | 102.06 |

**The axis packs, and it is not the fastest thing on this lane.** Against the per-order loop it
replaces, the entry retires **3.06** times fewer instructions and **2.61** times fewer slots: eight
orders are summed in one register where eight fits were summed one at a time. Against the float lane's
own default entry it retires **3.13** times *more* instructions and **3.71** times more slots, because
that entry is one seed fit and a downward recurrence across all 33 orders — it pays for one fit where
this lane pays for eight groups of them. A caller naming this axis is choosing the per-order lane's
shape rather than the cheapest shape this lane has, and the axis is served because it was named.

**Which fetch the entry takes, and why it is not the double lane's answer.** Both counters put the
gathered fetch ahead of the composed one here — **5,521,695,364** against **8,562,138,514**
instructions and **6,856,442,627** against **8,982,817,264** slots — which is the reverse of the
double lane, where the gather is a microcode assist and loses on slots. Eight floats fill one gather
where four doubles filled a microcoded one. The two fetches are one lane value for value, which the
lane's own test asserts bit for bit over every scheme and the whole sweep, so this is a choice of
instruction and nothing else, and the entry takes the one this lane's own counters prefer.

**Where the entries are.** The float lane's two fetches are `BoysAllOrdersF32Simd` and
`BoysAllOrdersF32SimdComposed`, and the policy a consumer names reaches the first through
`BoysAllOrdersF32`. A relaxed multiplier reaches the lane through the same effective-degree table the
double lane uses at the degree the float fits are stored; it is answered on the shipped route and
scheme alone, because a degree table is certified against one stored table of one fit family. The
float lane carries the computation budget as a fourth parameter of that entry — `BoysBudget::kFloat`
and `BoysBudget::kFp16` are certified against different bars, 1.5e-7 and 1e-7, so a lane that
answered an fp16 policy with the kFloat degrees would deliver the looser figure under the tighter
name.

**Re-measurement.** As for the double lane, every figure above is a property of one build's code and
one machine's microarchitecture, and the counters are re-read with
`benchmarks/boys_across_orders_benchmark.cpp` (`--variant=orders-f32-…`), whose header carries the
command. The entry's own row is `--variant=orders-f32-orders-axis`, which calls the public
`BoysAllOrdersF32` on the orders policy and so measures the fetch the entry actually selects.

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

## Every combination, and the bound each one carries

The lanes above are one axis of six. A call is a lane, a fit route, an evaluation scheme, an
interval partition, a packing axis and an accuracy multiplier, and the library offers the product of
all six: **2 routes × 2 schemes × 2 partitions × 2 axes × 7 rungs, in 4 lanes — 448 combinations.**
Each axis's own section above states what that axis changes. This one states what a combination is
guaranteed, how a program asks the library for the figure, and which members of the space this
revision does not carry.

**The bound of a combination is its lane's figure at its rung, and no other axis moves it.** The
other four axes change what a call *delivers* — which fits it reads, how many coefficients it sums,
whether it packs arguments or orders — and the guarantee each of them publishes is the lane's own
bound times the multiplier. The measured columns of the sections above are what those axes deliver;
the bound is here.

| Lane | One value carries at m = 1 | Beside that figure | Named |
|---|---|---|---|
| double | 5.5e-14 | — | `Precision::kFp64` |
| float | 1.5e-7 | — | `Precision::kFp32` |
| half, fp16 and bfloat16 | 1e-7 | plus half of the last representable digit of the returned value, claimed only where the value exceeds the sum | `Precision::kFp16` |
| float on a device | 1.5e-7 | plus 8e-8 under the fast region-B exponential | `Precision::kFp32Device` |

A combination at multiplier `m` carries **`m · base + additive`** — the same arithmetic the README's
contract table states lane by lane. `BoysLaneContracts()` returns those four rows, so a program
reads the figures the tables are written from rather than transcribing them, and the accuracy gate
reads the same rows to judge a combination against.

**Two accessors, two questions, and their names say which is which.**
`BoysAccuracyGuaranteed(precision, route, scheme, axis, granularity, tier)` answers *may I rely on
this combination being at least this accurate*: its `value` is the bound above, at the rung named.
`BoysAccuracyDelivered(...)` answers *which of these two combinations has been measured to do
better*: its `value` is the worst of the rows the combination names, each of which publishes what it
was measured to deliver. Both return an `AccuracyFigure` whose `reading` field says which of the
two figures it is, so neither can be read as the other.

**The delivered figure is a floor on a whole call's error and not the whole call's figure.** The
rows it maximises over are the *fits'* own figures, and a call adds its recurrences over them, so
the gate measures a whole call at or above it — strictly above it on 14 of the 78 carried rows of
this grid. What a whole call delivers is the gate's measurement, and the gate's combination table is
where that figure lives for every combination.

**A combination this revision does not carry returns no number.** Both accessors return
`available == false`, `value == 0.0` and a `reason` carrying the library's own sentence for the
refusal — the same sentence the gate prints beside the row. A caller cannot mistake a refusal for an
accuracy. The delivered accessor is absent as well for the half lane, whose error is dominated by
the format's quantum at the returned value rather than by the call, and for every rung past the
reference multiplier, because no row publishes a figure measured at one and a scaled guarantee is
not a measurement.

### What this revision carries, and what is owed

The gate crosses the whole space, prints one line per combination — its measured delivered figure
beside the bound its lane publishes — and ends the block with its own arithmetic. A row of that
table reads `fp64, chebyshev, split-clenshaw, shipped, orders, m = 64 | 56694 cells | 0 outside |
1.54485e-12 delivered | 3.52e-12 bound | certified and published`, and a refused one carries no
cells, no figure and the reason. The block's own last lines, from the same run the top of this page
names:

    COMBINATIONS: 78 of 448 member(s) of the option space are certified and published
                  363 refused with the library's own reason and owed
                  7 not runnable on this host, counted apart and not against the library
                  0 offered and covered by no cell of this block
                  0 delivering outside the bound its lane publishes
    the arithmetic: 78 + 363 + 7 + 0 + 0 = 448
                   the space read off the tables a second way: 448 member(s) over 4 lane(s),
                   a route axis of 2 2 2 2 route(s), 2 scheme(s), 2 partition(s),
                   2 axis(es), 7 rung(s)

Those are three states and there is no fourth: certified and published, refused with the library's
own reason and owed, or not runnable on this host. **A combination added to the library and left
uncovered lands in the fourth count and fails the run**, so a hole cannot go quiet; the run is at
revision `623a8e2`. By lane:

| Lane | Certified and published | Refused, reason owed | Not runnable on this host | Members |
|---|---|---|---|---|
| double | 58 | 54 | 0 | 112 |
| float | 10 | 102 | 0 | 112 |
| half | 10 | 102 | 0 | 112 |
| float on a device | 0 | 105 | 7 | 112 |
| the space | 78 | 363 | 7 | 448 |

**The double lane carries 10 of its 16 members at the reference multiplier and 8 at each relaxed
rung, and all 54 refusals are the narrow partition's.** Six of the 54 are at m = 1: the narrow
partition holds no rational table, so its four rational rows are refused, and it has no kernel that
packs a stride into the across-orders axis, which refuses the two chebyshev ones. The other 48 are
narrow, eight at each of the six relaxed rungs, because a relaxed rung cuts a fit by a per-order
effective degree and only the shipped row carries such a degree table.

**The two single-precision lanes carry 10 members each.** At the reference multiplier they carry the
shipped partition on the arguments axis, at both routes and both schemes — four — and past it the
shipped route and scheme alone, which is six more, one at each relaxed rung. The 102 refusals fall
in three parts: 56 are on the narrow partition, which for this lane is a table nobody has generated;
28 are on the across-orders axis, which this lane has no kernel for because the packing axes are the
double lane's; and 18 are a combination the reference rung carries and a relaxed rung does not,
because a relaxed rung cuts a fit by a per-order effective degree and this lane's degree table is
certified against one stored fit family.

**The device lane carries one combination at seven rungs, and this host cannot run any of them.**
Those seven are counted apart and not against the library: a machine with a CUDA device is the
instrument for the lane, the CUDA accuracy gate is what runs there, and the figure this page
publishes for the lane is its documented one rather than a measurement of it.

A refusal is backed by a `static_assert` in the header, named, or by a configure probe that compiles
the call and reports that it does not build — never by a build failure a reader has to guess at. The
gate prints each refusal with the sentence and its backing, and the per-axis sections above give the
same reasons axis by axis.

### The accessor's figure beside the figure the row is judged by

The gate reads both accessors for a combination in every lane, prints what they answer beside the
bound the row is judged against and the figure the call was measured to deliver, and fails if the
guarantee differs from the judged figure in any digit or if the delivered figure sits *above* the
measurement. One combination per lane, from that run:

| Combination | Accessor | Judged at | Measured | The source the accessor read |
|---|---|---|---|---|
| fp64, chebyshev, split-clenshaw, shipped, arguments, m = 1 | 5.5e-14 | 5.5e-14 | 5e-14 | throughout, every region |
| fp32, chebyshev, split-clenshaw, shipped, arguments, m = 1 | 1.5e-07 | 1.5e-07 | 1.08354e-07 | throughout, every region |
| fp16, chebyshev, split-clenshaw, shipped, arguments, m = 1 | 1e-07 | 1e-07 | 1.08354e-07 | plus half of the last representable digit of the returned value |
| fp32-device, chebyshev, split-clenshaw, shipped, arguments, m = 1 | 2.3e-07 | 2.3e-07 | — | documented figure; this host cannot run the lane |
| fp64, chebyshev, split-clenshaw, narrow, orders, m = 1 | no figure | not judged | not measured | refused, and the accessor returns no number |

The first two columns are one number read two ways rather than two numbers that agree today. The
accessor computes its figure inside the library from a `BoysLaneContracts()` row; the gate computes
the figure it judges the row against from that same row by its own arithmetic, and fails the run
when the two differ in any digit. They are two paths over one source, so a change to a lane's figure
moves both or the run goes red. The fp16 row judged at 1e-07 and measured at 1.08354e-07 is the
ceiling's half-ULP term at work: the base figure is what the accessor returns, and the term of the
format is added by the row's own criterion — the gate counts the cells where the returned value
falls at or below the floor rather than passing them as covered.

## The default policy, per precision and per device

A caller that has chosen a precision and nothing else writes one name. Every precision the library
offers has one, each entry of that precision runs it when the call site names no policy — for the half
lanes as a fixed policy, since those entries take no policy argument at all — and this section states
what each name selects and the bound it carries.

**These are the shipped settings, and not a measurement.** No default on this page was chosen against
a timing: the option space is still being completed, and the runs that would set a default per
precision have not been taken. When they are, each name is set from them, and that is one line per
name — the aliases in `backend.hpp`, and the device lane's two in `accuracy.hpp` and
`boys_device_tables.hpp`. Until
then nothing here is a claim about which setting is fastest. *What is not claimed*, at the end of this
page, is the same statement for every lane.

| Precision | Name | Fit route | Scheme | Granularity | Packing axis | Engine budget |
|---|---|---|---|---|---|---|
| double | `boys::DefaultPolicyFp64` | chebyshev | split Clenshaw | shipped | arguments | none — the double lanes read no budget |
| float | `boys::DefaultPolicyFp32` | chebyshev | split Clenshaw | shipped | arguments | `BoysBudget::kFloat` |
| fp16 | `boys::DefaultPolicyFp16` | chebyshev | split Clenshaw | shipped | arguments | `BoysBudget::kFp16` |
| bf16 | `boys::DefaultPolicyBf16` | chebyshev | split Clenshaw | shipped | arguments | `BoysBudget::kFp16` |

The route column is the lane's shipped Chebyshev table; the scheme reads it by the split Clenshaw
recurrence, which is the lane's certified form; the granularity is the shipped partition, region A's
two equal-width bands per order and region B's one seed; and the packing axis is the arguments axis,
the one a call has whether or not anybody names it. What the other member of each axis costs and buys
is in *The two fit routes*, *Interval granularity*, *The evaluation scheme* and *The packing axis*
above.

**The bound each name carries is the lane's own**, stated above and not restated here: the double
single entry at most 1e-15 below x = 1.0855, 3e-14 below x = 11.899848152108484 and 5.5e-14
everywhere, the double batch entries 5.5e-14 throughout, the float lane 1.5e-7 absolute and
everywhere, the half lanes `m·1e-7` plus half of the last representable digit of the result. The
multiplier is the entry's own default, `boys::kBoysFullAccuracyMultiplier`, which is the rung every
figure on this page is stated at. A call that names an axis is judged against that axis's section
rather than against this table.

**Why four names and not one.** The fp16 and bf16 lanes are the float engine under the fp16 budget,
and the budget is the axis that differs: their region-A and region-B degrees are cut for a 1e-7
budget where the float lane's are cut for 1.5e-7, so past the reference multiplier the two budgets
select different rungs and different arithmetic. One default for every precision would be the float
lane's budget imposed on the half lanes. `DefaultPolicyFp16` and `DefaultPolicyBf16` are one policy
type — one lane at one budget — named twice so that a document can cite the format its reader uses.

**The half lanes' entries take no policy argument.** `BoysSingleF16`, `BoysAllOrdersF16`,
`BoysSingleBf16` and `BoysAllOrdersBf16` take the multiplier and nothing else: the budget is the whole
of what their default adds to the float lane's, so they run `DefaultPolicyFp16` and
`DefaultPolicyBf16` as a policy they carry rather than one a call site passes. The name is what a
document cites and what the check below holds them to.

**The device lane's default is two names**: `boys::kBoysFullAccuracyMultiplier`, the multiplier the
CPU entries default to, and `boys::kDefaultRegionBExp`, which is `RegionBExp::kAccurate` — the library
exponential, and the arithmetic the f32 batch bodies have always run. `BoysCuda::SingleF32` and the
device-callable `BoysDeviceSingleF32` are the two entries of the lane that take the second; every
other entry of the lane takes the first alone. The lane takes no fit route, no scheme, no budget, no
packing axis and no partition, so a caller that has chosen a device precision and named neither of
these two has chosen everything the lane has to choose.

The device lane's bounds are the CPU lane's, precision by precision: CUDA fp64 the double lane's
figures, CUDA fp32 at `RegionBExp::kAccurate` the float lane's 1.5e-7 in every region, and CUDA fp32
at `RegionBExp::kFast` that bound plus the corrected seed's own contribution — the region-B
exponential section above states it, and the device gate measures it.

**The run that prints these figures.** The first command prints the identity of each name and the
in-force default as numbers rather than as a checkmark:

    cmake --build <build> --target boys-consumer-defaults
    <build>/Release/boys-consumer-defaults      # the config directory is your generator's

It prints one line per entry: the cells compared between the name and the same entry with no policy,
how many of them differ, and the worst difference between the two — zero on every identity row,
because the name and the in-force default are one call and not two spellings of one arithmetic. Its
last two rows compare a half lane against the float lane's default at m = 2, where the budget decides
the rung, and print the number of cells the two part on; that is the figure behind the four names.
The device lane's two defaults are printed the same way by a check of its own, which a CUDA build
compiles into two translation units because the lane's headers split that way — the batch entry's
header is a host header, and the device header is what a `.cu` may include:

    cmake --build <build> --target boys-consumer-cuda-defaults   # needs -DBUILD_CUDA=ON
    <build>/Release/boys-consumer-cuda-defaults

It holds each default to its name at compile time, where two spellings of one entry compare equal as
function addresses exactly when they are one instantiation, and runs the batch entry and the
device-callable entry on the card in both spellings, each of them beside `RegionBExp::kFast` as well,
so a printed row shows which arithmetic the default selected.

The second command prints the bound each name carries:

    cmake --build <build> --target boys-accuracy-gate
    <build>/Release/boys-accuracy-gate --strict

which sweeps the documented entries and prints, lane by lane and region by region, the worst error
delivered beside the bound claimed. The device lane's figures come from `boys-cuda-accuracy-gate`,
which a CUDA build runs on the card it was compiled for.

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
