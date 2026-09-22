# Per-lane contract and limitations

The bound is |F̂_n(x) − F_n(x)| ≤ m·B_region for every supported n, x, lane and region; the
per-region budgets are tabulated in
[the README](https://github.com/myamlak/boys#accuracy-contract) and in the header preamble. This
page states the part a table of budgets cannot carry: **where each lane's bound holds, where it
stops, and what sets that lane's accuracy ceiling**. Each lane is a range rather than a single
number — a loose end, the multiplier rungs from m = 1 up to m = 65536, and a tight end that each
lane reaches for a different reason.

## double (scalar and batch)

**Count on** the certified budgets at m = 1 — 1e-15 on region A, 3e-14 on the extended band and
region B, 5.5e-14 on region C for the single entry; 5.5e-14 flat for the batch entry — and the
relaxed rungs above them, m = 64 up to 65536, chosen per call or at compile time. Work and budget
both move with m.

**The ceiling is set by coefficient rounding, and it is the library's maximum accuracy, full
stop.** The stored coefficients are correctly rounded doubles — the generator works at 30 digits
and emits round-trip-exact decimals — so the fit carries 5e-19 of truncation while the doubles
holding it carry 1e-16 to 2e-16. The polynomial is sixty times better than the format that stores
it, so no rung goes below that floor and no change of path reaches it: a caller holding the m = 1
result is already holding the most accurate double this library produces.

## float (single and batch)

**Count on** 1.5e-7 absolute at m = 1, the same in every region, and the same rungs to 65536. The
batch form seeds in double and recurses in single.

**The ceiling is the format.** A 24-bit significand resolves about 6e-8 relative, and the m = 1
budget is 1.5e-7 absolute: for values of order one those two numbers are within a factor of a few
of each other, so there is no rung below the budget left in the format to ask for.

## half (fp16 / bf16, as shipped)

**Count on** the certified float engine's value, correctly rounded to the half type: the budget is
m·1e-7 + ½ULP, and half takes no part in the arithmetic — the argument is rounded to the half type,
the float entry evaluates at that value, the result is rounded back. **The lane stays inside its
budget at every order tested, but its margin is a fragility and not a cushion.** The published
worst cell of 0.990 of budget at order 0, x = 721 reproduces exactly (0.98993), and it is not the
worst cell: an independent sweep over this grid reaches **0.999441 of budget at order 0, x =
3.79688 for bf16 (a margin of 0.05585%) and 0.999008 at order 0, x = 9.29688 for fp16 (0.09924%)**,
and a second, denser sweep reaches **0.999915 for bf16 (0.0085%) and 0.999225 for fp16 (0.0775%)**.
So the margin is thousandths of a per cent, not one per cent, and the sentence has to be read as
what it is: at that distance the ½ULP representation term is 99% of the budget, the 1e-7 base is
spent to its last digit, and **one half-quantum of extra error anywhere in the lane — a float
engine one rounding worse underneath, a different argument, a different rounding of the
argument — takes a cell over budget.** The published figure understated the worst cell by a factor
of a hundred; a lane at 0.9999 of budget is one rounding from failing, and that belongs in the
contract rather than in a footnote.

**The ceiling is the representation term, and it is m-independent.** Eleven significand bits
resolve about five parts in ten thousand (2⁻¹¹), and the budget already spends half of one such
step on representing the result; every rung above m = 1 loosens the engine's share only.

**The ceiling is a domain restriction, and past it the lane claims nothing.** With u the format's
quantum at the returned value, the bound above is claimed over the arguments where
|F_n(x)| > m·1e-7 + ½u — there the return is within that distance of the value — and over no
others. Below that ceiling **no accuracy is claimed**: a caller that has to be right at those
orders and arguments wants the float or the double lane, which carry no such floor.

**What the return is past the ceiling differs between the two formats, because their exponent
ranges do.** In fp16 the ceiling and the format's floor coincide: every point of the sweep where
the bound exceeds the value (14952 of 39633 through the single-order entry, 22773 of 47454 through
the batch entry) is a point where the lane returns a subnormal number, then exactly zero, and the
bound is met by that floor rather than by the arithmetic --- the largest value so discarded is
1.295e-07, at order 20, x = 13. bf16's exponent range equals float's, so its 1e-7 base is met by
the arithmetic, not by a floor, down to |F_n(x)| ~ 2^-126; of its 22436 of 47454 bound-covered
points, only 10090 return nothing usable. The points past the ceiling are **counted, not passed**:
the accuracy gate reports them apart from the points the bound binds on and never folds them into
the count of claims met, because a total that did would be reporting the format's floor as
accuracy.

From order 3 upward on region C, that condition, not the bound, is what decides whether a call
returns a number at all: every argument of the branch is past the ceiling there, so the returns are
subnormal then zero --- a caller reading "within budget" at order 8 has been misled by omission, not
by a number that is wrong.

A per-order power-of-two scale would extend the range --- on region C, order 3 has no argument left
in normal range unscaled --- and cannot remove the limit: the ladder's span grows like x to the n
against a finite exponent range, so beyond about x ~ 35-38 the largest argument whose ladder still
fits binary16's normal field is what bounds any scaling at all, at order 8. **What a scale reaches
is bounded by the format and not by the algorithm, which is the re-runnable part of this
paragraph:** the per-order reaches formerly quoted here (x ~ 361 at order 3, 129 at order 4, 30 at
order 8) came from a packed-half prototype that is not in this repository, and they are withdrawn
with it; what this tree can measure is the ceiling any such scale runs into, and the accuracy gate
measures it from the committed reference instead. **The range is not, however, the failure that
binds this lane** --- the mantissa is, below.

## packed half --- the native lane

**Two half values per register, one correctly rounded half operation per ladder step, the seed's
square root and divide included** (`BoysAllOrdersHalf2`, `BoysAllNF16Native`): region C's ladder
itself in binary16, with no unpacking to single anywhere on the path. Its cost is countable and no
timing is claimed for it --- one square root and one divide for the seed, then one packed multiply
and one packed divide per order, two arguments to a register.

**Count on** the bound that arithmetic can hold, stated in ULP of the returned value rather than as
a region budget, because ULP is the shape of its error: **|out[k] - S_k| <= 8 ULP, measured at a
worst of 4.243 ULP (at order 6, a swept maximum rather than a proof bound)**, with S_k = 2^15 F_k(x)
the scaled value the entry returns. That bound is claimed over the arguments whose returned value
is a normal half, and over no others. The accuracy gate sweeps the same lane independently and
finds **3.853 ULP of the returned value at (n = 5, x = 50.6562)** --- a swept maximum too, and
neither number is upgraded or contradicted by the other.

**It is not a drop-in for the store-half lane at equal budget, and that is a floor rather than a
defect.** A packed ladder spends more than the half-quantum of representation a store-half lane's
budget leaves it --- 3.853 ULP against 0.5 ULP is 7.7x that share --- **because eleven mantissa
bits are what they are**, and the I/O lane's budget was never written for a lane that rounds per
operation: the I/O lane rounds once per value, this one once per operation. The steps that pay it
are the ones callers use, the low orders, so **the ULP bound above, not the I/O lane's, is what a
caller of this lane reads.**

**A per-order power-of-two rescale buys no accuracy, and the reason is one line: the rescale is
exact, so the scaled lane is bit-identical to the unscaled one at every order.** What it buys is
range. The shipped lane scales by 2^15 --- the largest power of two binary16 holds --- which keeps
the running value in the format's normal range down to F_k(x) = 2^-29.

**The ceiling is a domain restriction, and past it the lane claims nothing.** Above that value the
entry returns a normal half and the bound above is the claim. Below it the return is a subnormal
half, then exactly zero, **by design** --- at order 8 and x = 400 the entry returns zero --- and
**no accuracy is claimed for those arguments**: a reader should take the ceiling as where the claim
stops, not as a bound the format's floor happens to satisfy, and a caller that has to be right at
those orders and arguments wants the double or float lane instead. The crossings are the whole of
the format's argument range at orders 0 and 1, x ~ 2590 at order 2, and x ~ 359 / 128 / 30 at
orders 3 / 4 / 8. **No scale reaches past the wall**: 2^15 is the largest there is, and at order 8
the ladder's own span F_0/F_8 passes the format's normal range at x ~ 37.6 whatever scale is
chosen, since the span grows like x to the n against a finite exponent range.

**Graded as three claims, each alone.** The lane publishes them in the shape the accuracy gate's
book uses --- one domain, one bound, one verdict each --- and the suite's own book prints the three
verdicts (`ClaimBookGivesEachClaimOneVerdict`, `tests/boys_half_native_test.cpp`):

| Claim | Domain | Bound | Evidence in this tree |
|---|---|---|---|
| `native-half-packed` | every operation, on a packed pair | correctly rounded to binary16 | exhaustive square root, curated and random pairs, and the 47.3%-past-half-an-ULP distribution a widen-compute-round lane cannot produce |
| `native-half-bound` | region C, returned value a normal half | <= 8 ULP of the returned value | the sweep of region C against the committed grid and the certified double lane: worst 4.243 ULP |
| `native-half-ceiling` | the argument ceiling per order | none: the domain ends | the crossing scan and the span wall, both printed by the suite; the points past it are counted, not passed |

**That the packing is real is the lane's own structural claim, and the gate records it as refuted
rather than confirmed.** Every operation is correctly rounded to binary16 once per operation, with
no widening for the sequence as a whole --- but this repository's accuracy oracle sees values and
never the arrangement of the arithmetic that produced them, so the oracle can refute packedness and
cannot confirm it. **The claim is not upgraded**, and no timing is claimed for the lane: this
tree's development machine has no packed-half arithmetic, so a pass there would measure the
emulation rather than the lane.

**No fallback, and one precondition.** The entries cover region C only, so the argument must be at
or above the region-C boundary; there is no route to the table regions, and no accuracy multiplier
to relax (region C carries no truncatable resource).


## region C — the asymptotic branch

**Count on** the branch being m-invariant: it is a single closed form with no coefficients to
truncate, so no rung relaxes it, and its budget holds with slack at every m.

**Its accuracy is set entirely by its lower boundary, and that boundary is an accuracy knob but
not a performance tier** — the branch's cost does not depend on where the boundary sits, so moving
it saves no work.

**Below about x = 16 the branch leaves its domain of validity, and the error grows there smoothly
and without any threshold.** This paragraph used to say the error "jumps five to eight orders" and
that the boundary "is a hard floor, not a taper", and **that was wrong about the shape**: measured
on the form as coded, over the arguments below x = 16 at every order, consecutive arguments differ
in relative error by **at most 3.597 — 0.6 of an order** — so there is no step at the edge and no
pair of regimes to floor between. The error **rises monotonically at 759 of 825 neighbouring pairs
below the edge** (66 breaks, 8.0%, and 2 of 25 pairs at the top order), which is what a smooth curve
sampled on a finite grid looks like and not what a threshold looks like. Its **size** is real and
in the other direction from the old sentence's emphasis: relative to the value, the branch's error
just below the edge reaches **5.647e4 at (n = 32, x = 14.0468)** here and **5.2e3 at n = 32** on
the independent oracle's reading — against a 5.5e-14 budget that is the branch being **five to eight
orders past its bound in the last few units of x below the edge, having never crossed a boundary to
get there.** Crossing the edge, one grid step moves the ratio 4.175 to 17.24 and a two-unit window
8.002 to 68.82 (0.6 to 1.2 orders), and against the branch's in-domain 5e-14 the same cells give
2.533e5 to 2.955e7. **A claim about the shape of an error survives longer than a claim about its
size, because nothing in a bound table ever checks it** — and that is how this one survived: no
budget is violated by a smooth error, so no assertion anywhere contradicted the taper.

## What is not claimed

**Throughput.** No timing taken so far supports a performance claim for any lane: every one comes
from a loaded machine and from a part with no tensor cores and a double-to-single throughput ratio
of about 1:32. **The performance question is open**, and it must be answered by a timed pass on an
unloaded machine and on a card chosen to show the difference — the packed-half advantage in
particular is a function of the card, so a measurement of it on one part is not a measurement of
the lane. Until such a pass exists, no lane's cost may be assumed better than another's.

The native half lane's packed claim is structural rather than timed: its operations are on half
pairs, and the suite pins that no value is widened to single and narrowed back anywhere on the
path. **What two values per register is worth in time is unmeasured**, and it has to be measured on
a part that has packed half arithmetic — this tree's development machine has none, so a pass there
would measure the emulation rather than the lane.

**Tensor cores.** Not reachable from this lane, and the reason is shape rather than precision. The
ladder is a rank-one scalar recurrence — one multiply and one divide per order, applied to a pair
of arguments — so there is no product of two matrices anywhere for a tensor core to take; the
library's dominant work is that recurrence, not a matrix multiply. The one matrix-shaped step, the
region-A Chebyshev transform, does not carry the accuracy lanes with reduced-precision operands:
measured three orders of magnitude outside them. **So the ceiling of this technique is packed half
arithmetic's, and no tensor-core path is available to route around it.**

## Where the numbers come from

Every number above is a measurement. The table names the comparison behind each one and whether a
reader can re-run it from this repository as it stands. Most rest on the committed 45-digit
reference grid (`tests/data/boys_reference.csv`: n = 0..32 over x in [0, 100]) and, past x = 100,
on the accuracy gate's own 25-digit grid (`tests/data/boys_accuracy_gate_reference.csv`, the same
orders over x up to 3.40282e38, regenerated by
`tools/gen_boys_accuracy_gate_reference.py`); **what falls outside both is not re-checkable here,
and the rows below say so rather than looking like the rows that are.** Rows a lane withdrew leave
no number behind to be mistaken for a measurement.

| Statement | Measurement | Re-checkable from this tree |
|---|---|---|
| half lane 0.98993 of budget at n = 0, x = 721 reproduces; the worst cell is 0.999915 (bf16, 0.0085% margin) and 0.999225 (fp16, 0.0775%) | the shipped lane against this gate's own reference grid at the half-rounded argument, per (n, x), on the gate's grid and on a denser sweep | yes --- `tests/data/boys_accuracy_gate_reference.csv` reaches x = 721 for both half formats |
| native half <= 8 ULP of the returned value, worst measured 4.243 ULP (`native-half-bound`); the gate's own sweep of the same lane puts it at 3.853 ULP at (n = 5, x = 50.6562) | the shipped lane against the committed 45-digit grid where it reaches, and against the certified double lane (5.5e-14) across the rest of region C, per order, over the arguments whose return is normal | yes --- `ctest` prints the per-order maxima and the suite's claim book prints the verdict; **a swept maximum, not a proof bound** |
| the ceiling of that claim, per order --- none at orders 0 and 1, x ~ 2590 at order 2, x ~ 359 / 128 / 30 at orders 3 / 4 / 8 (`native-half-ceiling`) | the shipped lane's crossing scan, per order in x from the region-C boundary to the format maximum, with the reference agreeing on the crossing argument | yes --- `ctest` prints the crossings, and counts the points past them as outside the claim |
| no per-order scale reaches order 8 past x ~ 37.6 | the span F_0/F_8 against the format's normal range, at the top order | yes --- `ctest` prints it, and the entry returns zero at order 8, x = 400 |
| the packed lane's operation arithmetic and its rounding distribution (`native-half-packed`) | the operations against exact binary64 decisions --- exhaustive for the square root, curated and random for the four arithmetic operations --- plus the delivered-error distribution over region C | **partly** --- the lane's own structural test prints the counts and the 47.3% / 20.0% shares; the gate's oracle sees values and never the arrangement of arithmetic, so it records the claim as refuted rather than confirmed |
| subnormal then exactly zero from order 3; a per-order scale reaching x ~ 361, 129, 30; no scaling past x ~ 35-38 at order 8 | a packed-half prototype, now withdrawn | **withdrawn with the prototype** --- no number is claimed and none is left here to be read as one |
| the matrix-shaped region-A step, with reduced-precision operands, lands three orders outside the accuracy lanes | a reduced-precision prototype of that step against a high-precision reference the prototype carried, not in this tree | no --- the prototype is not in this tree |
| region C's error grows smoothly below x ~ 16 with no threshold, reaching 5.647e4 relative at (n = 32, x = 14.0468) | the asymptotic form, as coded, against this gate's own reference grid across the boundary: 825 neighbouring pairs, largest ratio between them 3.597, 66 breaks | yes --- the grid spans those arguments and the form is in this tree |
| coefficient rounding 1e-16 to 2e-16 under 5e-19 of truncation; the shipped leading coefficient rounds at 5.551e-17 | the generator's per-band report on the shipped tables, and the exact rounding of the shipped leading literal | yes --- `python3 tools/gen_boys_coefficients.py --check`, and the gate re-rounds the literal |
| the m = 1 budgets and the rung family | the committed reference grid at the documented multipliers | yes — `ctest --test-dir build --output-on-failure` |

## What the reference is worth

The gate's reference is only as good as its own error, so it is pinned on both sides.

**The generator's 30 digits are sufficient, and that was measured, not assumed.** Re-emitting the
2804 double literals of the shipped coefficients at 60 digits changes 86 of them, worst case
1.558e-30; none of the 1,208,526 values the library delivers differs. Thirty digits are enough for
this coefficient set with that much room to spare.

**The reference is not the limiting instrument.** Raising the working precision from 40 to 90
digits moves the compared values by about 1e-41 — twenty-six orders of headroom over the 1e-15
tightest bound the gate checks. A gate that reports a failure at 1e-14 is therefore reporting the
library and not itself.

**The stored grid, not the arithmetic, is the floor under a comparison.**
`tools/gen_boys_accuracy_gate_reference.py` evaluates in mpmath at 80 digits and writes 25
significant digits per value, so every comparison carries the grid's own truncation, about 1e-25
relative — ten orders below the tightest bound this book states, the 1e-15 region-A figure, and
eighteen below the half lanes' 1e-7. That truncation, and not the 1e-41 above, is what a failing
cell has to be read against. The two are the same story told twice: both sit far below every
budget here, so a red row is a statement about the library.

**The independent oracle that checked this revision caught two bugs in itself before it caught
anything in the library** — a quadrature route that understated its own error by 24×, and an
asymptotic route that returned a zero truncation estimate, which is a route reporting perfect
agreement with itself. Both were found because each route's error estimate was compared against
the other's, not because a value looked wrong. That is the standard every row in this book is held
to, and it is why each verdict names what would have to be true for the check to fail rather than
only what it measured.

**The argument grid reaches the top of every format's finite range**, so that no lane is checked
only where its format happens to have a grid point: for each format narrower than binary64 the grid
carries the format's largest finite value, a geometric ladder of that format's values from the
previous grid maximum up to it, and every representable value in the last band where the format's
spacing leaves a band wider than a grid step. `code.coverage.argument_range` reports the counts
(1353 of 1718 arguments cast to a finite binary16, 1316 distinct, largest 65504) and the gates for
the tier and native-half lanes report their own.

**A verdict says what would have to be true for the check to fail.** Every row prints a
`falsified by:` line: a measured quantity with a threshold where the row has a numeric comparison,
the tree command that would show it otherwise where the row is a tree fact, and the row's class
falsifier where neither applies. A row that cannot state its own failure condition is the failure
condition.

**And a green sweep is only as wide as the cells that can discriminate.** A cell whose bound is at
least as large as the value itself passes whatever the lane returns there — a subnormal, a zero, an
error of the same size as the number — so it cannot separate a lane that meets its bound from one
that has stopped having a value to be wrong about. The gate counts them and prints the count on
every run: at this revision it is 1,481,482 comparison cells across all six lanes, of which 754,101
(50.9%) carry such a bound, and **the no-cell-exceeds-its-budget statement is carried by the
remaining 727,381 (49.1%)**. An independent oracle counting one cell per value the library
delivers — 1,208,526 of them, rather than per lane, region and order as this gate does — flags
698,078 (57.8%) as non-discriminating; that count is its own, at the revision it measured. The two
books partition the sweep differently, and both figures mean the same thing — read "no cell over
budget" as a statement about the discriminating cells and no wider — so the number to compare
against a run is the one that run printed, never the one written here.

**How to re-run this, and what makes it fail.** The gate is a ctest target and a standalone binary:

```
cmake -S . -B build && cmake --build build --target boys-accuracy-gate --config Release
ctest --test-dir build --output-on-failure        # or: build/Release/boys-accuracy-gate
```

It exits non-zero while any claim is not met, prints the `NOT MET at this revision:` line naming the
claim ids, and prints one `falsified by:` line per row so that every row states what would have to
be true for it to fail. **Re-run configure before believing the revision it names**: the label is
read with `git rev-parse` at configure time, so a build-only rebuild after a commit keeps the
revised tree's code under the older label. A full pass prints `RESULT: 32 of 32 claims met at this
revision (26 verified outright, 6 met over a stated domain, 0 exceeded, 0 vacuous only, 0 evidence
absent)` and exits 0, **immediately followed inside the same RESULT block by the fraction of cells
that can discriminate** — the printed verdict names how wide its own sweep is, so a quote of the
count cannot shed it; neither `vacuous only` nor `evidence absent` counts as met, and the six
domains are the ones the rows name in their own `domain:` lines.

**A failed probe must not read as an absent feature.** The tier rows are gated on a configure-time
probe for the tier's declarations. Written with `check_cxx_source_compiles`, that probe *compiled*
and failed to **link** — the tier's definitions live in `src/boys.cpp` — so the configure line read
"this revision has no run-time accuracy tier" and three claims came back evidence-absent on a
revision that has the tier: a failed probe reported as the absence of the feature, which is the same
shape as a missing artefact read as a missing feature. The probe now sets
`CMAKE_TRY_COMPILE_TARGET_TYPE` to `STATIC_LIBRARY` (a declaration probe should not link; the target's
own link is what tests the definitions), clears its cached result so a negative from an earlier
configure cannot silently drop the rows again, and this paragraph is here so that whoever adds the
next probe treats "the feature is absent" and "the probe failed" as two different outcomes.
