# Getting started

This library evaluates the Boys function,

    F_n(x) = integral from 0 to 1 of t^(2n) exp(-x t^2) dt,   n = 0..32,  x >= 0

the integral an electronic-structure program needs at every order a shell quartet can ask for. This
page is the task-shaped route in. Each section is a question you would actually ask, the code that
answers it, and what that code prints. Every transcript below is what one of those programs printed,
and the command that reproduces it is beside it. Nothing on this page is a figure you have to take on
trust.

The precise guarantees, the evaluation settings, and the measurements behind them are in
[docs/lane-contract.md](lane-contract.md) and the [API reference](mainpage.md). This page links to
them where a guarantee is involved. It does not restate them.

## Build it, and run the programs below

    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure     # runs every example

Each example is also a program you can compile on its own against an already-built tree:

    c++ -std=c++20 -DBoysFp16=1 -mavx2 -mfma -mf16c -ffp-contract=off \
        -I include examples/00_first_call.cpp -L build -lboys -o first && ./first

The headers need C++20, and the compile line above is all they ask for: `-DBoysFp16=1` and the
tier flags are the ones the build above put on the library, and `docs/build-facts.md` states the
profile they spell. The build in the block before it is the contributor's: it compiles the test
suite as well, which wants C++23. On Windows with the Visual Studio generator the library lands in
`build/Release/`. The CMake route above is the one that finds it without flags.

---

## What does it look like to call this?

`examples/00_first_call.cpp` — four calls cover most of what a program does with this function.

```cpp
double ladder[boys::kMaxBoysOrder + 1] = {};
boys::BoysAllOrders(6, 3.5, ladder);          // F_0..F_6 at x = 3.5

const double x[3] = {0.25, 4.0, 30.0};
double out[3] = {};
boys::BoysFixedN(2, x, out, 3);               // out[i] = F_2(x[i])

const double batch[4] = {0.0, 0.25, 4.0, 30.0};
double grid[4 * 7] = {};
boys::BoysAllN(6, batch, grid, 4);            // grid[k * 4 + i] = F_k(x[i])

boys::BoysSingle(3, 1.25);                    // F_3(1.25)
```

    F_3(1.25)  = 0.055476132923076911
    F_0(3.5)   = 0.46984703520162335
    F_3(3.5)   = 0.011831383583716588
    F_6(3.5)   = 0.0040954447623731041
    F_2(0.25)  = 0.16753319090734875
    F_2(4)     = 0.017525782161992999
    F_2(30)    = 0.00013483513281472914
    F_0(30)    = 0.16180215937964007
    F_6(30)    = 3.6049669032736188e-08

All four are `noexcept`. The requirements on the values are `n` in `[0, 32]` and `x >= 0`, and both
hold for anything a basis set produces. **Outside them the behaviour is undefined, and it is not
benign**: a release build does not refuse the arguments, it answers them — a negative argument or an
out-of-range order comes back as some value with no meaning, and a `NaN` argument comes back `NaN`.
The exact values are the build's own and move with the compiler and the flags; on this project's
MSVC x64 Release build, `BoysSingle(3, -1.0)` returns a tiny positive number, `BoysSingle(33, 1.0)`
a plausible-looking one, and `BoysAllN` over `{1.0, NaN, 2.0}` is a segmentation fault. A build with
assertions live aborts on the argument instead. The three calls that
write into an array ask one thing more: the buffer has to hold the values the call writes. The
`std::span` overloads of these entries (`boys/boys_span.hpp`) compare that extent and refuse a span
that is short, with a line naming the call and both sizes; a `double*` carries no size, so the
pointer entries called above cannot check it and the caller still owns the buffer.

**Next:** if you want one of these three shapes specifically, read on. If you want the whole
catalogue at once, jump to [Which entry do I call?](#which-entry-do-i-call).

---

## I want one value: F_n(x) at a single argument

`examples/01_one_order.cpp` — one call, and the error the library guarantees for it.

```cpp
const double f = boys::BoysSingle(n, x);
```

    F_3(1.25) = 0.055476132923076911
    guaranteed error <= 7e-14  (from throughout, every region)
    BoysAllOrders agrees to 0

The second line is the contract evaluated at the `(n, x)` the program asked about, read out of the
library at run time rather than typed into a document: **the guarantee is the larger of
`1e-13 · F_n(x)` and `1e-13 · F_0(x)`** — a fraction of the value rather than one figure
for the whole range ([the specification](specification.md), "Accuracy contract") — and since F falls
with the order, the arm scaled by the ladder's first rung is the wider one wherever the order's own
value is tiny. At `F_3(1.25)` the bound is `1e-13 · F_0(1.25)` ≈ 7e-14; the phrase in parentheses,
"throughout, every region", is the reach the guarantee is stated with. The lane's runs measured a
worst case of 5.5e-14, which is the reading [what the library
guarantees](#what-the-library-guarantees) keeps as documentation — the number above is the promise
at the cell, and it is the one to judge a calculation by.

**Next:** `BoysSingle` computes each order on its own, so reach for the ladder below if you need more
than one. It is the same call an integral code makes.

---

## I want every order at one argument

`examples/02_all_orders.cpp` — a shell quartet needs F_0 through F_nmax at the same argument. One call
hands over the whole ladder, instead of one call per order.

```cpp
double ladder[boys::kMaxBoysOrder + 1] = {};
boys::BoysAllOrders(nmax, x, ladder);         // ladder[k] = F_k(x)
```

    k   F_6(3.5)
    0   0.46984703520162335
    1   0.062807093111329163
    2   0.022603413701666943
    3   0.011831383583716588
    4   0.0075174716662425286
    5   0.0053514087962663534
    6   0.0040954447623731041
    worst gap against BoysSingle: 0 (guaranteed error <= 4.7e-14)

The heading is the program's own: it names the top order the call asked for and the argument, and
each row below it is one order, from 0 up to that top order.

The entry writes `nmax + 1` doubles. The example declares the full `kMaxBoysOrder + 1`, so the same
buffer serves any later call without a second array.

**Next:** if your arguments are an array and the order is one number, the call below turns the loop
inside out. That is the shape that gets help from the vector units.

---

## I want one order over many arguments

`examples/03_arguments_array.cpp` — the same order at every argument, written where you want it.

```cpp
boys::BoysFixedN(n, x, column, count, stride);   // column[i * stride] = F_n(x[i])
```

    n = 2, stride 2
    x = 0.25    out[ 0] = 0.16753319090734875
    x = 1       out[ 2] = 0.10026879814501641
    x = 4       out[ 4] = 0.017525782161992999
    x = 12.5    out[ 6] = 0.0012030139279729933
    x = 30      out[ 8] = 0.00013483513281472914
    worst gap against BoysSingle: 0 (guaranteed error <= 1.6e-14)

The stride is why the entry exists: the answer can go straight into a column of a larger structure,
with no intermediate copy. Leave it out and it defaults to 1. A container passed in place of the
pointer and count takes its count from the container's own extent and spells the stride
`BoysStride{...}` — a bare number in that position would be a count under the pointer entry's
meaning, so the container overload names it and refuses to compile the ambiguous spelling.

**Next:** every order over many arguments at once — the batch shape — which is the next section.

---

## I want every order over many arguments {#i-want-every-order-over-many-arguments}

`examples/08_shell_quartet_batch.cpp` — the batch shape an integral engine calls: a whole shell
quartet's arguments at once, every order at each of them.

```cpp
boys::BoysAllN(nmax, x, out, count);            // out[k * count + i] = F_k(x[i])
boys::BoysAllNAtOrders(n, x, out, count);       // each argument at its own top order
```

    five arguments, each at its own top order
      F_3(0.5) = 0.097222024416928829
      F_6(2) = 0.014008835839082516
      F_2(7.5) = 0.0042700136194305507
      F_8(0.125) = 0.052602850434880673
      F_4(20) = 8.1278555493588363e-06
    values the per-argument call produces: 28
    values the padded call produces:        45
    produced for nothing:                   17
    cells compared against the padded call:  28, worst difference 0
    guaranteed error per value, tightest over this batch: 1.98e-14  (throughout, every region)
    every value positive and at most one

This is the entry to reach for when the arguments are the batch and the orders are the ladder: it
groups the arguments once for the whole call rather than a ladder at a time, and it hands the planes
back order-major, so nothing has to be transposed afterwards. The arguments may arrive in any order.
[What the batch and the ladder each cost on a measured machine](#what-the-double-lane-costs), and
what that run does and does not separate, is below.

When the arguments do not share a top order — a quartet whose four shells differ — the padded call
pays for cells nobody asked for, 17 of the 45 in the run above. `BoysAllNAtOrders` writes only the
values you asked for, and it is the column the padded call is checked against.

**Next:** if you want a specific evaluation, the next section is how you name it.

---

## I want to name a specific evaluation {#i-want-to-name-a-specific-evaluation}

**First, a warning, because this one is a trap.** One of the settings below is called `kNarrow`. That
name reads as "narrower, therefore more careful". It is not. `kNarrow` and `kCoarsest` cut the fitted
interval into pieces of different widths and carry a different number of pieces; **both meet the same
published error bound.** Choosing the wrong one changes the work, not the accuracy. It costs you
nothing visible: both compile, both run, both are correct, so you would never find out you had picked
backwards. `kNarrow` is the seam's own name — what the policy `SeamPolicy<>` resolves to — and not
the partition a call site that names no policy reads: that one is its class's row, and every host
double row in this build carries `kUniform`. `kCoarsest` is what the program below names. If you have not
measured a preference, name no policy at all and take the default. The bound the library publishes
is the lane's, and no partition moves it.

`examples/05_policy.cpp` — a policy is named in the template argument list, never constructed. The
values below are the program's own ladder, at the one argument and the one top order it sets for
itself at the top of the file.

```cpp
using Named = boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                               boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                               boys::FitGranularity::kCoarsest,
                               boys::DivisionForm::kExactDivision>;

boys::BoysAllOrders<Named>(nmax, x, named_ladder);
```

    k   default policy            named policy
    0   0.54629197178514766      0.54629197178514799
    1   0.092841394632249538     0.092841394632249843
    2   0.039287837054569889     0.039287837054570153
    3   0.022870837329790165     0.022870837329790391
    4   0.015602172536926584     0.015602172536926787
    worst disagreement: 3.3e-16 (guaranteed error <= 5.5e-14)

The two agree here to within 3.3e-16, far inside the 5.5e-14 the contract prints at this argument, and that is expected
rather
than a coincidence: the settings select how the same approximation is summed, not what is
approximated, so they part company only in the last bits — three of the five lines above differ
there. The places to change a setting are the ones where they do **not** agree. Finding those on your
machine is what [the option probe](#i-want-to-know-which-option-is-fastest-on-this-machine) is for.

The six settings the call above writes, in the order they are written: which stored fit serves the
interval, how its coefficients are summed, an internal precision budget, whether vector lanes hold
four arguments or four orders, how finely the fitted interval is cut, and how the recursion's
divisions are performed. The policy carries a seventh — which exponential seeds a region-B ladder —
which the call above leaves at its default. Each is documented beside its type in
[the API reference](mainpage.md), and all seven are listed under [What you can and cannot choose at
run time](#what-you-can-and-cannot-choose-at-run-time).

**Next:** the settings you did not name. The next section is what each of them changes, and where the
figures for it are.

---

## What does each setting change? {#what-does-each-setting-change}

Five of the seven settings the policy above writes are the library's own option axes — the fit route,
the evaluation scheme, the packing axis, the interval granularity and the division form, which
`docs/architecture.md` under "The option axes" names as the five a host policy holds. The policy's
other two — the engine budget and the region-B exponential — are named at the end of this section.

**A call's bound is its lane's; what these axes move is what a call delivers and what it costs.** One
member of one axis states a term beside a lane's base rather than a figure of its own, and the table
says where. **Which member is fastest is a property of your machine, its flags and its card**, so the
library's rule is that the option probe ranks the combinations where they will run: its transcript
is in the section above, and the recorded run of its
full protocol is `tests/data/boys_option_probe_report.txt`. Every row of that report names the member
of each axis it measured. The table says what each axis decides, what the tree has measured about it,
and where the figures are. **It does not rank the members** — the section above says why no page can.

| the axis | what it selects, and its members | what naming another member changes | where the figures are |
|---|---|---|---|
| **fit route** — `FitRoute` | which stored fits serve a region: the Chebyshev fits the certified lanes are defined by (`kChebyshev`, the default), or a rational minimax alternative holding the same bar by storing different things (`kRationalMinimax`) | **Accuracy, storage and work.** Measured against the committed reference, region A reads 1.74e-16 through the Chebyshev table, which stores 1320 coefficients, against 2.46e-14 through the rational one, which stores 891; the two region-B seeds read 5.09e-17 and 1.69e-14, over 25 and 13 stored. The rational route takes over per order from 1.0855252345349333 rather than from the region's left end, and it is not a seed for the batch entry's recursion. A rational fit costs one division per order, where the split-Clenshaw Chebyshev form is division-free. | `docs/lane-contract.md`, "The two fit routes" — the route table itself and the paragraphs on the two tables' reach. `BoysFitRoutes()` reports the rows at run time. The route has **not** been ranked against a timing, and lane-contract's default-policy section says which axes have been and which have not. |
| **evaluation scheme** — `EvalScheme` | the summation a stored fit is read in: the even/odd split Clenshaw recurrence on its Chebyshev form (`kSplitClenshaw`), or Horner's rule on the monomial form of the same fit (`kHorner`, the default) | **Arithmetic and work, not the fit.** Both sum one fit, at one degree, to one bound: region A's per-order fits read 2.220e-16 under Horner and 4.441e-16 under split Clenshaw at the fused multiply-add route, with Horner's own figure at 4.441e-16 on the separate route, and the two region-B seeds read the same figure either way. Region C is a closed form with no fit to sum; the half lanes, the packed region-A lane and the CUDA device lane are split Clenshaw's and are not offered the other member. | `docs/lane-contract.md`, "The evaluation scheme: split Clenshaw or Horner" — one measured row per scheme and stored fit, which the accuracy gate carries beside its own rows. `BoysEvalSchemes()` and `BoysEvalSchemeFits()` answer the rows at run time. Cost: the probe's. The runs behind the default compare both members and do not separate them. |
| **interval granularity** — `FitGranularity` | how narrowly the fitted domain is cut into pieces: the committed cut (`kCoarsest`), the narrower cut the proved truncation bound places (`kNarrow`, the seam's default), or one fixed grid over the whole fitted domain with every order fitted independently (`kUniform`) | **Work per call against storage, not accuracy** — all three cut one fit. The specification's table reads 19 to 25 coefficients per evaluation over 1345 stored coefficients for `kCoarsest`, and 11 over 3476 for `kNarrow`; `kNarrow`'s own measured region-A figure is 5.68e-16 at that region's 1e-15 bar. The grid buys a locate of one multiply and drops the per-order recursion, and pays in stored coefficients: on the double lane 496 intervals of width 1/16 under a read cap of 5, with the float lane carrying its own 77 of width 1/4 under a cap of 4. **The committed run ranks all three**, and on it the uniform grid leads the fp64 single class — 105.75 against 271.54 ns per argument for the fastest narrow row, which is on the shipped Chebyshev route (the fastest rational-route narrow row is 392.66, `single-batch-narrow-rational-minimax-fp64`) — which is the partition every host double row names. | `docs/specification.md`, "Interval granularity", and `docs/lane-contract.md`, "Interval granularity". The gate carries one row per partition, scheme and call shape; `BoysFitGranularities()` answers the rows at run time. Cost: the probe's. |
| **packing axis** — `PackAxis` | which of a call's values share a vector register: four arguments at one order (`kArguments`, the default), or four orders at one argument (`kOrders`) | **Which call shapes can carry it, and which path a call takes.** The two evaluate the same stored fits by the same arithmetic, so this is neither a choice of accuracy nor one of two designs. `BoysFixedN` produces one order and refuses `kOrders` where it is named; the single-precision entries have no packed body and refuse it too. A plane call naming the orders axis takes the all-orders entry's body and returns its values bit for bit, giving up the region grouping that exists to feed an arguments lane. | `docs/lane-contract.md`, "The packing axis: which of a call's values share a vector"; `BoysPackAxes()` reports the members and the interval each one's packed lane evaluates. **Not ranked against a timing**, with the route. |
| **division form** — `DivisionForm` | how each step of a recurrence divides: one division per step (`kExactDivision`), a reciprocal formed once and multiplied through the ladder (`kPlainReciprocal`), or that form with the correctly rounded quotient recovered by one fused multiply-add per step (`kRefinedReciprocal`, the default) | **The last bits and the work, and on two lanes the published figure.** `kRefinedReciprocal` is bit-identical to exact division; the float and half lanes state a term beside their base where `kPlainReciprocal` is named, because that form's extra rounding takes the all-orders entry's region-A downward ladder past it. The axis governs only the divisions inside a recursion, so an order answered by reading its own fit is the same value under all three forms. | `docs/lane-contract.md`, "The division form: how each recursive step divides" — the lanes' figures, the cost the tree recorded on the host it measured, and the command that reproduces it. `BoysDivisionForms()` reports the members at run time, and the probe names the form in every row it prints. |

**The policy's other two settings.** The **engine budget** (`BoysBudget`) is read by the
single-precision and half engines and by no other lane, so on a double policy it is inert: `kFloat` is
the float lane's 1.5e-7 budget and `kFp16` the half lanes' 1e-7 region target. The **region-B
exponential** (`RegionBExp`) names the arithmetic that seeds a region-B ladder: `kAccurate` is the
library routine, and `kFast` is the library's own reduced-argument polynomial, read above the argument
that polynomial's own requirement admits and the library routine below it. Each target has its own
default — the host's is `RegionBExp::kFast` — and the device lane's fast member carries a term of its
own beside its base. Both are listed with the other five settings, and where each may be named, in
[What you can and cannot choose at run time](#what-you-can-and-cannot-choose-at-run-time).

**If you have not measured a preference, name no policy at all**, which is what the section above
advises: a call naming nothing reads its class's own row.

**Next:** you have named a specific evaluation. Before you rely on it, ask what it guarantees.

---

## What the library guarantees {#what-the-library-guarantees}

`examples/06_what_it_guarantees.cpp` — ask with your tolerance and get a verdict, rather than
comparing a table by eye.

```cpp
const boys::CombinationCoverage answer = boys::QueryCombination(
    boys::Precision::kFp64, 0, 0.0, boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner,
    boys::PackAxis::kArguments, boys::FitGranularity::kNarrow,
    tolerance);
```

    the double entry at x = 0, where the contract is widest, every order:
      requested    verdict                  guaranteed   measured
      1e-12        YES - guaranteed       1e-13        2.4e-14
      1e-10        YES - guaranteed       1e-13        2.4e-14
      1e-08        YES - guaranteed       1e-13        2.4e-14
    1e-20 is answered NO

The distinction the verdict makes matters for a calculation. **`YES - guaranteed` means the bound the
library documents for that combination is at or below what you asked for.** The middle state,
`only measured, not guaranteed`, means a figure exists but it is an observation rather than a promise.
It is deliberately not treated as a pass. A request nothing can meet comes back `NO`, not as a
quietly rounded yes. The last line above is that case.

"Is this accurate enough?" and "which of these two is more accurate?" are different questions with
different answers, and the function above answers only the first. For the second, `BoysAccuracyDelivered`
returns the figure a combination was *measured* to deliver. The [API reference](mainpage.md) documents
both accessors and says which reading a calculation's safety may rest on.

**Next:** `YES - guaranteed` tells you the answer is good enough. The option probe tells you which of
the good-enough options is fastest where you are.

---

## I want to know which option is fastest on this machine {#i-want-to-know-which-option-is-fastest-on-this-machine}

`examples/07_option_probe.cpp` — this is a property of the machine and the build flags, so no
document can answer it for you. It has to be measured where it will run.

```cpp
const boys::OptionProbeReport report = boys::RunOptionProbe(options);
```

    measured 2018 options on 12 logical processors

    fp64 all-orders - 72 options ranked
      fastest: uniform-pack-orders-horner-exact-division-fp64 at 44.54 ns/argument
      the run could not separate 2 of them from the leader
        uniform-pack-orders-horner-plain-reciprocal-fp64
        uniform-pack-orders-horner-fp64

    recommended here: uniform-pack-orders-horner-exact-division-fp64

**The line counts are that build's, not this one's.** The transcript above was taken on the build this
document was written against, and the option space has changed size since, so **your counts will
differ from the two headline numbers** — that is the space growing and shrinking, not a fault in your
build. The counts to read are the ones your own run prints.

**The important line is the third one, not the second.** Of the 72 options in that group, the run
could not place 2 behind the leader. The "fastest" row is therefore one of three that are effectively
tied, and that is the real answer. The probe says so rather than inventing a winner from a timing
difference too small to measure. Read the result as *a group at the top and a tail behind it*. Being
on the leader is worth little. Being well behind the group is worth fixing.

**Every figure in that transcript is a property of this host and this run.** The nanosecond figures
are wall-clock measurements. Which options tie follows from them, and the headline counts follow the
build. What carries to your machine is the shape of the answer — a leader, a tied group, and a tail —
not the numbers in it.

That distinction is also why these figures should not be copied. On the machine that wrote this page
the same program was run several times while other work was going on. The leader changed each time,
and so did the length of the tied group. A timing taken on a busy machine measures the machine. Run it
on a machine you are not otherwise using, and re-run it if you change the build flags.

The example runs a deliberately short protocol so it can run inside a build. The command-line tool
`boys-option-probe` runs the full one and prints every group, every figure beside its spread, and the
machine load the rounds were taken under.

Once you have a preference, name it at your call site — [the section
above](#i-want-to-name-a-specific-evaluation) — or point `BOYS_BUILD_DEFAULTS` at a header carrying
your own choices for a whole build: the five a host class falls back to and the device lane's own
two. Every call site that names nothing then compiles the choices
your probe measured. `CONTRIBUTING.md` describes the second route.

---

## What the double lane costs {#what-the-double-lane-costs}

The probe above answers for your machine. This tree also carries one recorded run of its full
protocol — `tests/data/boys_option_probe_report.txt`, the report `boys-option-probe` writes — and
that run answers for the machine, the build and the process it was taken on.

The report's own header says what it is: *this result is about this machine, this build and this
process*, 12 logical processors with the AVX2+FMA tier present, and a build whose four arithmetic
backends — scalar and AVX2, fp64 and fp32 — each leave a bare `a * b + c` as two roundings. It served
every combination its classes hold, 2,592 of them and no refusals, and measured 2,448 on this host.
Its workload is 16,384 arguments, x log-uniform over [1e-3, 4e1], each argument's own highest order
drawn as the sum of two shell angular momenta over 0..16, up to nmax 32.

**Recorded run, not reproduced by `ctest`.** One row per question shape, for the double lane, each
the fastest entry of its class and all of them documented at 5.5e-14:

| the class's question | the fastest measured entry | ns per argument | options the run could not place behind it |
|---|---|---|---|
| F_n(x) for one order at one argument, one call per order — `BoysSingle` | `single-batch-fast-fp64` | 105.75 | 5 |
| F_0..F_n(x) at one argument, one ladder call — `BoysAllOrders` | `batch-plain-reciprocal-accurate-fp64` | 35.66 | 4 |
| F_n(x_i) at one order over the array — `BoysFixedN` | `fixed-n-batch-exact-division-fp64` | 257.47 | 3 |
| F_0..F_nmax(x) at every argument, one call — `BoysAllN` | `all-n-batch-plain-reciprocal-fast-fp64-sorted` | 54.97 | 7 |
| ...each argument at its own top order — `BoysAllNAtOrders` | `all-n-at-orders-fp64` | 89.67 | 8 |

**A nanosecond figure describes the machine, the build and the run it was taken on, and not the
design.** It moves with the processor, the clock, the flags, and with what else the machine was doing
while the round ran — which is why the probe compares paired inside a round rather than across runs.
What carries from one host to another is the shape of the answer: which entries sit together at the
top, and how long the tail behind them is. Read the last column beside the third: 35.66 ns per
argument is the head of a class whose next four entries the run could not separate from it, and
105.75 the head of one whose next five it could not.

**Each class is ranked inside itself, and nothing here is ranked across two of them.** A class is one
precision and one question shape, and different shapes hand back a different number of values, so the
table is five readings of one lane rather than a ranking of the five calls against each other.

**What that leaves unsaid about the batch.** The [section
above](#i-want-every-order-over-many-arguments) says the batch entry groups the arguments once for the
whole call rather than a ladder at a time. The committed run does not separate that grouping: it ranks
the options inside one question shape. So its batch row here — the head of 288 measured combinations,
with 7 the run could not place behind it — is the choice *within* the batch shape, and its ladder row
is the choice inside the ladder shape above. Read the grouping sentence as what the call does, and
take the cost of grouping from a workload of your own rather than from this table.

---

## Which entry do I call? {#which-entry-do-i-call}

By what you want, not by what the library calls things:

| I want | Entry | One clause on when |
|---|---|---|
| F_n(x) for one order, one argument | `BoysSingle` | the simplest call; use it when you have one value |
| F_0..F_nmax at one argument | `BoysAllOrders` | a shell quartet's ladder; this is the common case |
| F_n at every argument of an array | `BoysFixedN` | when the order is fixed and the arguments vary; takes an output stride |
| F_0..F_nmax at every argument of an array | `BoysAllN` | the batch shape of an integral engine; the arguments may arrive in any order |
| ...with each argument's own top order | `BoysAllNAtOrders` | a shell-quartet batch, where no argument is padded to a common order |
| ...when you know the arguments are already sorted | `BoysAllN` with `BoysSortedArgs{}`, pointer form only | skips the internal sort when your loop already produces a non-decreasing array |
| ...when the arguments are not known to be in domain | `BoysSingleChecked`, `BoysAllOrdersChecked`, `BoysFixedNChecked`, `BoysAllNChecked`, `BoysAllNAtOrdersChecked` | the checked twin of every double entry above; it refuses with a status rather than answering |
| any of the above in single precision | `BoysSingleF32`, `BoysAllOrdersF32`, `BoysFixedNF32`, `BoysAllNF32`, `BoysAllNAtOrdersF32` | when the rest of your kernel is `float` |
| half-precision storage | `BoysSingleF16`, `BoysAllOrdersF16`, `BoysFixedNF16`, `BoysAllNF16`, `BoysAllNAtOrdersF16`, `BoysSingleBf16`, `BoysAllOrdersBf16` | 16-bit I/O around the single-precision engine |
| an answer on a GPU | `boys/boys_cuda.hpp` | device arrays; see [below](#on-a-gpu-the-first-call-is-the-slow-one) |
| the same arithmetic inside your own CUDA kernel | `boys/boys_cuda_device.hpp` | when a round trip through global memory would cost more than the evaluation |

The complete list, with every overload and the arithmetic behind it, is the entry-point table in the
[API reference](mainpage.md).

## What you can and cannot choose at run time {#what-you-can-and-cannot-choose-at-run-time}

The settings that select an evaluation fall into two kinds. One can be named by a value the program
computes. The rest can only be named where the call is compiled:

| Setting | Can you name it at run time? | How |
|---|---|---|
| which stored fit serves the interval | no | template argument only - `BoysAllOrdersWithRoute<kRoute>` on the ladder shape, `BoysSingleF32WithRoute<kRoute>` on the float lane's single |
| how the coefficients are summed | **yes** | the scheme-carrying overload of `BoysAllOrdersWithRoute` takes it as a value - the ladder shape; the float lane's single-order entry takes a route and no scheme |
| the internal precision budget | no | template argument only — `SeamPolicy`'s third parameter |
| the packing axis (whether a vector register holds four arguments or four orders) | no | template argument only — `SeamPolicy`'s fourth parameter |
| how finely the fitted interval is cut | no | template argument only — `SeamPolicy`'s fifth parameter |
| how the recursion divides | no | template argument only — `SeamPolicy`'s sixth parameter |
| which exponential seeds a region-B ladder | no | template argument only — `SeamPolicy`'s seventh parameter |

The settings that cannot be named at run time are fixed when the translation unit is compiled.
**There is no run-time entry for them.** A choice that has to be made per input record needs an `if`
over two instantiations at the call site, not a value passed into one.

## On a GPU, the first call is the slow one {#on-a-gpu-the-first-call-is-the-slow-one}

**This section is about the warm-up only.** If you are calling the library on a card, the page
written for you is [Calling this library on a GPU](gpu-guide.md): which of the two device routes
answers which problem, what a call looks like, what you must allocate and pass, what the library
refuses, and how to read what you got. What follows here is the one thing that belongs on this page
because it is about the *first* call rather than about a device call as such.

The GPU entries upload coefficient tables on first use, lazily and idempotently. **The first call on
a given device therefore pays a real one-time cost that every later
call does not.** A caller timing a loop from a cold start measures the upload, not the evaluation.

If you are benchmarking or budgeting latency, warm the path with one throwaway call of the entry
you intend to use, and time everything after it. A warm-up is also what `BoysCuda::InitializeTables`
is for, if you would rather pay the cost where you can see it than inside your first real call.

## What you do not pay for

A call that names no policy is resolved entirely at compile time. It compiles to
that one call — no run-time dispatch, no branch on an axis, nothing to predict. **The body is
inline**, at the row this build's table carries for the entry's class: your translation unit
instantiates it, nothing else is needed, and no archive stands on the link line for the calculation.
An unnamed call therefore costs you no run-time selection, and one instantiation rather than a
choice among many. Naming another policy is what instantiates the version you asked for, in your
translation unit.

**The measured figures come without the archive too, and they are not the contract.**
`boys/boys_accuracy_here.hpp` answers, from the headers and the macros this build pins, what the
build was measured to deliver: each lane's published absolute figure (`BoysLaneFiguresHere()`) and
the figure the arithmetic in force was measured at (`BoysAccuracyHere<Precision, Shape>()`). It is
not `BoysAccuracyGuaranteed`: that accessor evaluates the lane's contract at the `(n, x)` it is
asked about — on the double lane `max(1e-13 · F_n(x), 1e-13 · F_0(x))` — and it is one of the
archive's reporting queries. Ask the header what this build measured; ask the accessor what a
calculation at a cell is promised.

## Where the specification lives

- [The per-lane contract](lane-contract.md) — the bound each lane (one entry together with the
  arithmetic behind it) guarantees, over which interval, and where it stops.
- [Choosing a lane](consumer-perspective.md) — how much accuracy an integral calculation actually
  needs, and which lanes that leaves.
- [The API reference](mainpage.md) — every entry, every type, and the vocabulary this library uses,
  defined.
- [Calling this library on a GPU](gpu-guide.md) — the device half of this page: the two routes, what
  you allocate and pass, what the library refuses, and how to read the bound you got.
- [Build facts](build-facts.md) — what a build of this library is, read out of the build: the
  instruction sets it targets, whether a bare `a * b + c` in it is one rounding, and which arithmetic
  backends it carries.
- [The specification](specification.md) — what each evaluation path guarantees, the settings that
  select one, and the measurements behind the figures, as a page of its own.
