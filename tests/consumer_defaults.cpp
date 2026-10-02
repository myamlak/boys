// Consumer check: the named defaults from <boys/boys.hpp> alone, and the figures
// that show each name is the policy its entry already runs. A template parameter's
// zero value cannot be pointed at or cited; these names can.
//
// Two claims are checked, each printing numbers rather than a checkmark. First, the
// name and the empty argument list are one call - for the double and float lanes an
// identity of types, the aliases being `EvalPolicy<>`, which is what the entries'
// parameter already defaults to - and the two spellings are compared on values as
// well, because a type-level argument is not a value-level one. Second, where one
// precision's default is not another's (the half lanes run the fp16 engine budget, the
// is not another's (the half lanes run the fp16 engine budget, the float lane the
// float one), the entry is compared against the alias-composed call bit for bit, and
// beside it against the float lane's default.
//
// The second claim's rows are the one place a difference is expected and is not a
// failure.
//
// Run:  cmake --build <build> --target boys-consumer-defaults
//       <build>/boys-consumer-defaults
//       ctest --test-dir <build> -R boys-consumer-defaults

#include <algorithm>
#include <array>
#include <bit>
#include <boys/boys.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

// The library's private src/ is deliberately NOT on this file's include path: a
// consumer gets include/ and nothing else. Each of the five names below resolves
// only if a directory holding that source file is on the path, so renaming one still
// leaves four.
#if __has_include("boys.cpp") ||                                                                   \
                  __has_include("boys_simd.cpp") ||                                                \
                                __has_include("boys_transform.cpp") ||                             \
                                              __has_include("boys_c.cpp") ||                       \
                                                            __has_include("boys_half_native.cpp")
#error                                                                                             \
    "this consumer check is compiled with src/ on its include path; it does not prove that a consumer can build against the public headers alone"
#endif

namespace {

// The two half types are declared behind the BoysFp16 seam.
#if BoysFp16
using boys::Bf16;
using boys::F16;
#endif // BoysFp16

/// The four names. What each selects, and the bound it carries, are in docs/lane-contract.md.
using Fp64Default = boys::DefaultPolicyFp64;
using Fp32Default = boys::DefaultPolicyFp32;
using Fp16Default = boys::DefaultPolicyFp16;
using Bf16Default = boys::DefaultPolicyBf16;

// The class-keyed names: what an entry's own policy parameter defaults to, for the classes
// whose table row is not the fallback. `boys::DefaultPolicy<Precision, Shape>` is the name
// the library resolves a class to - the row the build's seam carries for it, or the five
// above where it carries none - so an entry is held to its own class's default rather than
// to a name that happens to agree with it at this revision.
using Fp64AllOrders = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>;
using Fp32AllOrders = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllOrders>;
using Fp16Single = boys::DefaultPolicy<boys::Precision::kFp16, boys::Shape::kSingle>;
using Fp16AllOrders = boys::DefaultPolicy<boys::Precision::kFp16, boys::Shape::kAllOrders>;
using Bf16Single = Fp16Single;  // the two half formats are one lane at one budget
using Bf16AllOrders = Fp16AllOrders;

constexpr double kFull = boys::kBoysFullAccuracyMultiplier;

// The name denotes the type the entry's parameter defaults to: `EvalPolicy<>` for the
// double and float lanes; for the half lanes the same axes under the fp16 engine budget.
static_assert(std::is_same_v<Fp64Default, boys::EvalPolicy<>>);
static_assert(std::is_same_v<Fp32Default, boys::EvalPolicy<>>);
static_assert(Fp64Default::kBudget == boys::BoysBudget::kFloat);
static_assert(Fp32Default::kBudget == boys::BoysBudget::kFloat);
static_assert(Fp16Default::kBudget == boys::BoysBudget::kFp16);
static_assert(!std::is_same_v<Fp16Default, Fp64Default>);
static_assert(std::is_same_v<Bf16Default, Fp16Default>);

// The other four axes are the shipped defaults on all four names.
static_assert(Fp64Default::kRoute == boys::kDefaultFitRoute);
static_assert(Fp64Default::kScheme == boys::kDefaultEvalScheme);
static_assert(Fp64Default::kPack == boys::kDefaultPackAxis);
static_assert(Fp64Default::kGranularity == boys::kDefaultFitGranularity);
static_assert(Fp16Default::kRoute == boys::kDefaultFitRoute);
static_assert(Fp16Default::kScheme == boys::kDefaultEvalScheme);
static_assert(Fp16Default::kPack == boys::kDefaultPackAxis);
static_assert(Fp16Default::kGranularity == boys::kDefaultFitGranularity);

// Address identity: two function addresses compare equal in a constant expression only
// when the two names are one instantiation, and two instantiations of one entry share a
// function-pointer type, so these assert that each entry's *own* default is the name -
// not merely that the name is a type the entry could be called with. A revision that
// changed an entry's template default without changing the alias fails them.
//
// Only the yes direction is a constant expression: comparing two distinct function
// addresses is not one under the sanitizer configuration this file is also built in.
// g++ 15 with -fsanitize=address,undefined refuses it with "'(f == g)' is not a constant
// expression", and a function address against a null pointer with "'(f == 0)'", while
// two spellings of one function fold. A variable template's initializer is a
// constant-expression context even where the value is only read at run time, so the
// comparison cannot be written as one; the two controls that can say no compare in main().
template <auto Left, auto Right> constexpr bool SameCall = (Left == Right);

static_assert(SameCall<&boys::BoysSingle<kFull>, &boys::BoysSingle<kFull, Fp64Default>>);
static_assert(SameCall<&boys::BoysAllOrders<kFull>, &boys::BoysAllOrders<kFull, Fp64AllOrders>>);
static_assert(SameCall<&boys::BoysFixedN<kFull>, &boys::BoysFixedN<kFull, Fp64Default>>);
static_assert(
    SameCall<&boys::BoysAllNAtOrders<kFull>, &boys::BoysAllNAtOrders<kFull, Fp64Default>>);
static_assert(SameCall<&boys::BoysSingleF32<kFull>, &boys::BoysSingleF32<kFull, Fp32Default>>);
static_assert(
    SameCall<&boys::BoysAllOrdersF32<kFull>, &boys::BoysAllOrdersF32<kFull, Fp32AllOrders>>);
static_assert(SameCall<&boys::BoysAllNF32<kFull>, &boys::BoysAllNF32<kFull, Fp32Default>>);

// --- the comparison ---------------------------------------------------------

/// One row of the report: cells compared, how many the two spellings answered
/// differently, and the furthest apart.
struct Row {
    const char* name = "";
    std::size_t cells = 0;
    std::size_t differing = 0;
    double worst = 0.0;
    int worstN = -1;
    double worstX = 0.0;
};

// The half types' representation readers, behind the seam with the rows that call them.
#if BoysFp16
std::uint16_t Bits(F16 v) noexcept {
    return std::bit_cast<std::uint16_t>(v);
}

std::uint16_t Bits(Bf16 v) noexcept {
    return std::bit_cast<std::uint16_t>(v);
}
#endif // BoysFp16

std::uint32_t Bits(float v) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    return bits;
}

std::uint64_t Bits(double v) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    return bits;
}

/// Judges one value of a row. Comparison is on the representation, so a difference
/// below a format's resolution still counts: the question is one call, not closeness.
template <class T> void Tally(Row& row, T named, T plain, int n, double x) {
    ++row.cells;

    if (Bits(named) == Bits(plain))
    {
        return;
    }

    ++row.differing;
    const double distance = std::fabs(static_cast<double>(named) - static_cast<double>(plain));

    if (distance >= row.worst)
    {
        row.worst = distance;
        row.worstN = n;
        row.worstX = x;
    }
}

/// A single-order pair. Both callables take (order, argument) and return the same type.
template <class T, class Named, class Plain>
Row SingleOrderRow(const std::vector<double>& xs, Named named, Plain plain) {
    Row row;

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        for (const double x : xs)
        {
            Tally<T>(row, named(n, x), plain(n, x), n, x);
        }
    }

    return row;
}

/// An all-orders pair: both callables take (argument, output) and fill
/// kMaxBoysOrder + 1 values.
template <class T, class Named, class Plain>
Row AllOrdersRow(const std::vector<double>& xs, Named named, Plain plain) {
    std::array<T, boys::kMaxBoysOrder + 1> left{};
    std::array<T, boys::kMaxBoysOrder + 1> right{};
    Row row;

    for (const double x : xs)
    {
        named(x, left.data());
        plain(x, right.data());

        for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
        {
            const auto index = static_cast<std::size_t>(k);
            Tally<T>(row, left[index], right[index], k, x);
        }
    }

    return row;
}

/// A fixed-order pair: both callables take (order, arguments, count, output)
/// and fill one value per argument.
template <class T, class Named, class Plain>
Row FixedOrderRow(const std::vector<double>& xs, Named named, Plain plain) {
    const std::size_t count = xs.size();
    std::vector<T> left(count, T{});
    std::vector<T> right(count, T{});
    Row row;

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        named(n, xs.data(), count, left.data());
        plain(n, xs.data(), count, right.data());

        for (std::size_t i = 0; i < count; ++i)
        {
            Tally<T>(row, left[i], right[i], n, xs[i]);
        }
    }

    return row;
}

/// A many-argument pair: both callables take (top order, arguments, count, output,
/// workspace).
template <class Named, class Plain>
Row ManyArgRow(const std::vector<double>& xs, Named named, Plain plain) {
    const std::size_t count = xs.size();
    const std::size_t stride = static_cast<std::size_t>(boys::kMaxBoysOrder) + 1u;
    std::vector<double> left(count * stride, 0.0);
    std::vector<double> right(count * stride, 0.0);
    Row row;

    named(boys::kMaxBoysOrder, xs.data(), count, left.data(), nullptr);
    plain(boys::kMaxBoysOrder, xs.data(), count, right.data(), nullptr);

    for (std::size_t i = 0; i < count; ++i)
    {
        for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
        {
            const std::size_t index = static_cast<std::size_t>(k) * count + i;
            Tally<double>(row, left[index], right[index], k, xs[i]);
        }
    }

    return row;
}

void Print(const Row& row) {
    std::printf("  %-38s %7zu cells, %zu differing, worst |named - unnamed| = %.3g",
                row.name,
                row.cells,
                row.differing,
                row.worst);

    if (row.differing != 0)
    {
        std::printf(" (order %d, x = %.17g)", row.worstN, row.worstX);
    }

    std::printf("\n");
}

/// The arguments: a sweep of the whole supported range with the boundaries where the
/// lanes change arithmetic included as their own samples. Sorted and deduplicated,
/// because one of the entries compared promises its arguments are in order.
std::vector<double> Arguments() {
    // The three arguments the accuracy page names: the region-A band end, the region-B
    // selector, and where the asymptotic branch takes over.
    constexpr double kBandEnd = 1.0855252345349333;
    constexpr double kRegionB = 11.899848152108484;
    constexpr double kAsymptotic = 28.98933773882074;

    std::vector<double> xs;
    constexpr std::size_t kSamples = 2048;

    for (std::size_t i = 0; i < kSamples; ++i)
    {
        xs.push_back(kAsymptotic * static_cast<double>(i) / static_cast<double>(kSamples - 1));
    }

    xs.push_back(kBandEnd);
    xs.push_back(kRegionB);
    xs.push_back(kAsymptotic);
    xs.push_back(100.0);
    xs.push_back(1.0e6);
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    return xs;
}

} // namespace

int main() {
    const std::vector<double> xs = Arguments();
    std::size_t rows = 0;
    std::size_t failed = 0;
    std::size_t controls = 0;
    constexpr std::size_t kControls = 2;

    std::printf("consumer check of the named defaults through <boys/boys.hpp>: "
                "%zu arguments, orders 0..%d\n",
                xs.size(),
                boys::kMaxBoysOrder);

    // --- double -------------------------------------------------------------
    // BoysSingle and the family entries name the double lane's default.
    {
        Row row = SingleOrderRow<double>(
            xs,
            [](int n, double x) { return boys::BoysSingle<kFull, Fp64Default>(n, x); },
            [](int n, double x) { return boys::BoysSingle<kFull>(n, x); });
        row.name = "fp64 BoysSingle";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        Row row = AllOrdersRow<double>(
            xs,
            [](double x, double* out) {
                boys::BoysAllOrders<kFull, Fp64AllOrders>(boys::kMaxBoysOrder, x, out);
            },
            [](double x, double* out) { boys::BoysAllOrders<kFull>(boys::kMaxBoysOrder, x, out); });
        row.name = "fp64 BoysAllOrders";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        Row row = ManyArgRow(
            xs,
            [](int nmax, const double* x, std::size_t count, double* out, std::size_t* workspace) {
                boys::BoysAllN<kFull, Fp64Default>(nmax, x, out, count, workspace);
            },
            [](int nmax, const double* x, std::size_t count, double* out, std::size_t* workspace) {
                boys::BoysAllN<kFull>(nmax, x, out, count, workspace);
            });
        row.name = "fp64 BoysAllN";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        Row row = ManyArgRow(
            xs,
            [](int nmax, const double* x, std::size_t count, double* out, std::size_t*) {
                boys::BoysAllN<kFull, Fp64Default>(nmax, x, out, count, boys::BoysSortedArgs{});
            },
            [](int nmax, const double* x, std::size_t count, double* out, std::size_t*) {
                boys::BoysAllN<kFull>(nmax, x, out, count, boys::BoysSortedArgs{});
            });
        row.name = "fp64 BoysAllN (sorted tag)";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        Row row = FixedOrderRow<double>(
            xs,
            [](int n, const double* x, std::size_t count, double* out) {
                boys::BoysFixedN<kFull, Fp64Default>(n, x, out, count);
            },
            [](int n, const double* x, std::size_t count, double* out) {
                boys::BoysFixedN<kFull>(n, x, out, count);
            });
        row.name = "fp64 BoysFixedN";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }

    // --- float --------------------------------------------------------------
    {
        Row row = SingleOrderRow<float>(
            xs,
            [](int n, double x) {
                return boys::BoysSingleF32<kFull, Fp32Default>(n, static_cast<float>(x));
            },
            [](int n, double x) { return boys::BoysSingleF32<kFull>(n, static_cast<float>(x)); });
        row.name = "fp32 BoysSingleF32";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        Row row = AllOrdersRow<float>(
            xs,
            [](double x, float* out) {
                boys::BoysAllOrdersF32<kFull, Fp32AllOrders>(
                    boys::kMaxBoysOrder, static_cast<float>(x), out);
            },
            [](double x, float* out) {
                boys::BoysAllOrdersF32<kFull>(boys::kMaxBoysOrder, static_cast<float>(x), out);
            });
        row.name = "fp32 BoysAllOrdersF32";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }

    // --- half ---------------------------------------------------------------
    // The half lanes' entries take no policy, so the entry is compared against the call
    // the name composes to: the documented name is the entry's arithmetic. Those entries
    // sit behind the BoysFp16 seam, so a closed-seam build says so below.
#if BoysFp16
    {
        Row row = SingleOrderRow<F16>(
            xs,
            [](int n, double x) {
                return boys::BoysSingleF16<kFull>(n, F16(static_cast<float>(x)));
            },
            [](int n, double x) {
                return static_cast<F16>(boys::BoysSingleF32<kFull, Fp16Single>(
                    n, static_cast<float>(F16(static_cast<float>(x)))));
            });
        row.name = "fp16 BoysSingleF16";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        std::array<float, boys::kMaxBoysOrder + 1> scratch{};
        Row row = AllOrdersRow<F16>(
            xs,
            [](double x, F16* out) {
                boys::BoysAllOrdersF16<kFull>(boys::kMaxBoysOrder, F16(static_cast<float>(x)), out);
            },
            [&scratch](double x, F16* out) {
                boys::BoysAllOrdersF32<kFull, Fp16AllOrders>(
                    boys::kMaxBoysOrder,
                    static_cast<float>(F16(static_cast<float>(x))),
                    scratch.data());

                for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
                {
                    out[k] = static_cast<F16>(scratch[static_cast<std::size_t>(k)]);
                }
            });
        row.name = "fp16 BoysAllOrdersF16";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        Row row = SingleOrderRow<Bf16>(
            xs,
            [](int n, double x) {
                return boys::BoysSingleBf16<kFull>(n, Bf16(static_cast<float>(x)));
            },
            [](int n, double x) {
                return static_cast<Bf16>(boys::BoysSingleF32<kFull, Bf16Single>(
                    n, static_cast<float>(Bf16(static_cast<float>(x)))));
            });
        row.name = "bf16 BoysSingleBf16";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        std::array<float, boys::kMaxBoysOrder + 1> scratch{};
        Row row = AllOrdersRow<Bf16>(
            xs,
            [](double x, Bf16* out) {
                boys::BoysAllOrdersBf16<kFull>(
                    boys::kMaxBoysOrder, Bf16(static_cast<float>(x)), out);
            },
            [&scratch](double x, Bf16* out) {
                boys::BoysAllOrdersF32<kFull, Bf16AllOrders>(
                    boys::kMaxBoysOrder,
                    static_cast<float>(Bf16(static_cast<float>(x))),
                    scratch.data());

                for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
                {
                    out[k] = static_cast<Bf16>(scratch[static_cast<std::size_t>(k)]);
                }
            });
        row.name = "bf16 BoysAllOrdersBf16";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }

    // --- why the names are not one name --------------------------------------
    // Past the reference multiplier the budget decides the rung, so the half lanes'
    // default and the float lane's are two arithmetics; this row prints how many cells
    // they part on and is not a failure either way. m = 1 is inert and would read zero,
    // so the row is taken at m = 2.
    std::printf("  (rows below compare a half lane against the FLOAT lane's default, "
                "m = 2, where the budget decides the rung)\n");
    {
        Row row = SingleOrderRow<F16>(
            xs,
            [](int n, double x) { return boys::BoysSingleF16<2.0>(n, F16(static_cast<float>(x))); },
            [](int n, double x) {
                return static_cast<F16>(boys::BoysSingleF32<2.0, Fp32Default>(
                    n, static_cast<float>(F16(static_cast<float>(x)))));
            });
        row.name = "fp16 vs the float lane's default";
        Print(row);
    }
    {
        Row row = SingleOrderRow<Bf16>(
            xs,
            [](int n, double x) {
                return boys::BoysSingleBf16<2.0>(n, Bf16(static_cast<float>(x)));
            },
            [](int n, double x) {
                return static_cast<Bf16>(boys::BoysSingleF32<2.0, Fp32Default>(
                    n, static_cast<float>(Bf16(static_cast<float>(x)))));
            });
        row.name = "bf16 vs the float lane's default";
        Print(row);
    }

    // --- the negative controls ------------------------------------------------
    // The identity assertions above fold only when the two names are one instantiation,
    // so they prove nothing on the pair they cannot be made on - the half default and
    // the float lane's, two names that must compare unequal. In main() rather than a
    // static assertion because the sanitizer configuration rejects the comparison in any
    // constant-expression context (see SameCall above).
    {
        const bool singleIsHalf = &boys::BoysSingle<kFull> == &boys::BoysSingle<kFull, Fp16Default>;
        const bool singleF32IsHalf =
            &boys::BoysSingleF32<kFull> == &boys::BoysSingleF32<kFull, Fp16Default>;

        std::printf("  negative control: the fp16 default and the float default are %s on "
                    "BoysSingle, %s on BoysSingleF32\n",
                    singleIsHalf ? "one call" : "two calls",
                    singleF32IsHalf ? "one call" : "two calls");
        controls += singleIsHalf ? 1u : 0u;
        controls += singleF32IsHalf ? 1u : 0u;
    }
#else // BoysFp16

    // Stated, not dropped. With the seam closed this build declares no half entry, so
    // there is no pair of spellings to compare; the four names are still asserted at the
    // top of this file, and what is absent here is the arithmetic they reach.
    std::printf("  fp16/bf16 lanes NOT CARRIED by this build (BoysFp16 = 0): the four "
                "half-precision entries are declared behind the seam, so the rows those "
                "lanes own are absent here rather than compared\n");

#endif // BoysFp16

    std::printf("  %zu of the %zu identity rows differ anywhere; "
                "the shortcut and the entry are one call\n",
                failed,
                rows);
    std::printf("  %zu of the %zu negative controls report one call, which is the answer "
                "the identity assertions above need them not to give\n",
                controls,
                kControls);

    return failed == 0 && controls == 0 ? 0 : 1;
}
