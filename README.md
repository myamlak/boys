# The Boys function kernel

[![CI](https://github.com/myamlak/boys/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/myamlak/boys/actions/workflows/ci.yml)
[![docs](https://github.com/myamlak/boys/actions/workflows/docs.yml/badge.svg)](https://github.com/myamlak/boys/actions/workflows/docs.yml)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![version](https://img.shields.io/github/v/tag/myamlak/boys?label=latest%20release%20tag)](https://github.com/myamlak/boys/tags)

F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt, for n = 0..32 and x ≥ 0 — the integral an electronic-structure
program evaluates at every order a shell quartet can ask for. This is a C++ library that evaluates
it, in double and single precision, on the CPU and optionally on a GPU. It depends on nothing outside
the C++ standard library and the optional CUDA toolkit.

## Your first call

Copy this into `try.cpp`, build it, run it. It prints the numbers below. They are a build that does not
contract the multiply-add; a compiler that does may differ in the last digit of a value and nowhere
else, which is what `docs/build-facts.md` records per CI leg.

```cpp
#include <boys/boys_span.hpp>

#include <array>
#include <cstdio>

int main()
{
    // One order at one argument.
    std::printf("F_3(1.25)  = %.17g\n", boys::BoysSingle(3, 1.25));

    // Every order 0..6 at one argument. A shell quartet wants the whole
    // ladder, and this is the call that hands it over.
    std::array<double, boys::kMaxBoysOrder + 1> ladder{};
    boys::BoysAllOrders(6, 3.5, ladder);
    std::printf("F_0(3.5)   = %.17g\n", ladder[0]);
    std::printf("F_3(3.5)   = %.17g\n", ladder[3]);
    std::printf("F_6(3.5)   = %.17g\n", ladder[6]);

    // One order over an array of arguments. out[i] = F_2(x[i]).
    const std::array<double, 3> x{0.25, 4.0, 30.0};
    std::array<double, 3> out{};
    boys::BoysFixedN(2, x, out);
    std::printf("F_2(0.25)  = %.17g\n", out[0]);
    std::printf("F_2(4)     = %.17g\n", out[1]);
    std::printf("F_2(30)    = %.17g\n", out[2]);

    // Every order over an array of arguments - the batch shape an integral
    // engine actually calls, and the one entry that groups the arguments once
    // for the whole call rather than a ladder at a time. The result is
    // order-major: out[k * count + i] is F_k(x[i]), one plane per order.
    const std::array<double, 4> batch{0.0, 0.25, 4.0, 30.0};
    constexpr int kBatchTop = 6;
    std::array<double, batch.size() * (kBatchTop + 1)> grid{};
    boys::BoysAllN(kBatchTop, batch, grid);
    std::printf("F_0(30)    = %.17g\n", grid[0 * batch.size() + 3]);
    std::printf("F_6(30)    = %.17g\n", grid[6 * batch.size() + 3]);

    // F is positive and falls off with x; a batch of zeros or a negative
    // value would mean the call did not do what it says.
    const bool sane = out[0] > 0.0 && out[0] > out[1] && out[1] > out[2] && ladder[0] > ladder[6] &&
                      grid[0 * batch.size() + 3] > grid[6 * batch.size() + 3];
    if (!sane)
    {
        std::printf("FAIL: F is not positive and decreasing in x\n");
        return 1;
    }
    return 0;
}
```

`boys/boys_span.hpp` is the one header a C++ caller needs beyond `boys/boys.hpp`: it adds an
overload of each many-argument entry taking a `std::span`, so a container goes straight in. It costs
nothing - the overload forwards to the entry and both spellings compile to the same call - and a call
passing a pointer and a count still reaches the pointer entry exactly.

    git clone https://github.com/myamlak/boys.git && cd boys
    cmake -S . -B build -DBOYS_BUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
    cmake --build build --config Release

or, against a tree you have already built:

    c++ -std=c++20 -I include try.cpp -L build -lboys -o try     # -I and -L as your build laid them out
    ./try

```
F_3(1.25)  = 0.055476132923077584
F_0(3.5)   = 0.46984703520162352
F_3(3.5)   = 0.01183138358371668
F_6(3.5)   = 0.0040954447623731674
F_2(0.25)  = 0.16753319090735061
F_2(4)     = 0.017525782161993072
F_2(30)    = 0.00013483513281636802
F_0(30)    = 0.1618021593796416
F_6(30)    = 3.6049670926598394e-08
```

The three `OFF`s are there so this builds the library and nothing else, which needs no compiler past
C++20 — the suite and the benchmark drivers are the targets that want C++23, and each is behind one
of those flags. Drop them to build all three as well; that is the contributor's build.

On Windows with the Visual Studio generator the library lands in `build/Release/`. This program is
[`examples/00_first_call.cpp`](examples/00_first_call.cpp); the contributor's build compiles it and
`ctest` runs it, along with every other program this README and the guide quote, so the transcript
above cannot rot silently.

**[docs/getting-started.md](docs/getting-started.md)** takes these four calls one at a time, then
the questions that follow them: how to name a specific evaluation, how to ask what the library
guarantees before you rely on it, and how to find out which of the available options is fastest on
your machine. Every figure in it is the
output of a program in [`examples/`](examples/), so you can reproduce any of them.

## Which entry do I call?

By what you want, not by what the library calls things:

| I want | Entry | One clause on when |
|---|---|---|
| F_n(x) for one order, one argument | `BoysSingle` | the simplest call; use it when you have one value |
| F_0..F_nmax at one argument | `BoysAllOrders` | a shell quartet's ladder; this is the common case |
| F_n at every argument of an array | `BoysFixedN` | when the order is fixed and the arguments vary; takes an output stride |
| F_0..F_nmax at every argument of an array | `BoysAllN` | the batch shape of an integral engine; the arguments may arrive in any order |
| ...with each argument's own top order | `BoysAllNAtOrders` | a shell-quartet batch, where no argument is padded to a common order |
| ...when you know the arguments are already sorted | `BoysAllN` with `BoysSortedArgs{}` | skips the internal sort when your loop already produces a non-decreasing array |
| any of the above in single precision | `BoysSingleF32`, `BoysAllOrdersF32`, `BoysFixedNF32`, `BoysAllNF32`, `BoysAllNAtOrdersF32` | when the rest of your kernel is `float` |
| half-precision storage | `BoysSingleF16`, `BoysAllOrdersF16`, `BoysFixedNF16`, `BoysAllNF16`, `BoysAllNAtOrdersF16`, `BoysSingleBf16`, `BoysAllOrdersBf16` | 16-bit I/O around the single-precision engine |
| an answer on a GPU | `boys/boys_cuda.hpp` | device arrays; uploads its tables on first use, so warm the path before measuring |
| the same arithmetic inside your own CUDA kernel | `boys/boys_cuda_device.hpp` | when a round trip through global memory would cost more than the evaluation |

The complete list, with every overload, is the entry-point table in the
[API reference](https://myamlak.github.io/boys/).

**Calling it on a GPU?** The last two rows are the two routes onto a card, and
[docs/gpu-guide.md](docs/gpu-guide.md) is written for you: which route answers which problem, what a
device call looks like, what you must allocate and pass, what the library refuses and why that is a
feature, and how to read the bound you got.

**What the bound is, and what it is not.** In double precision the guarantee is **absolute**:
`|F_n(x) − F_n(x)| <= 5.5e-14` for every `n <= 32` and every `x >= 0`. It does not bound the
*relative* error, and where F is small the two part company — at `n = 32, x = 29` the relative error
is **2.55**, with F about 1.9e-14 and the returned value 6.9e-14. **And no entry checks its
arguments**: outside `n` in `[0, 32]` and `x >= 0` the behaviour is undefined, and a release build
faults on `x < 0` or on a `NaN` anywhere in a batch, while `n > 32` answers a wrong number with no
diagnostic. Every entry has a `*Checked` twin — `BoysAllOrdersChecked` and its four siblings — that
refuses with a status instead of faulting. The
[accuracy contract](docs/specification.md#accuracy-contract) carries the relative-error table and
the command that measures the bound on your machine.

**How much accuracy do you need?** [docs/consumer-perspective.md](docs/consumer-perspective.md) —
what integral calculations actually require, and which of the library's evaluation paths that leaves
to choose between.

---

Everything above this line is what a first call needs. The specification — what each evaluation path
guarantees, the settings that select one, and the measurements behind the figures — is
**[docs/specification.md](docs/specification.md)**, and none of it is needed to make your first call.
The words that page uses — *lane*, *region*, *route*, *scheme*, *axis*, *gate* — are defined on the
documentation's landing page, together with the full API reference:
<https://myamlak.github.io/boys/>. **[docs/architecture.md](docs/architecture.md)** is the map for a
reader working in the tree rather than calling it: which file implements what, and which test holds
each thing to its contract.
