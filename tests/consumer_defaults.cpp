// Consumer check: the shortcut. One name per precision, from <boys/boys.hpp>
// alone, and the figures that show each name is the policy its entry already
// runs.
//
// A default that is only the zero value of a template parameter is not
// something a caller can point at or a document can cite. These names are: a
// caller that has chosen a precision and no axis writes the name, and what the
// call then runs is a sentence with a subject.
//
// Two claims are checked here, and each prints numbers rather than a checkmark:
//
//  * the name and the empty argument list are one call. For the double and
//    float lanes that is an identity of types - the aliases are `EvalPolicy<>`,
//    which is what the entries' parameter already defaults to - and the two
//    spellings are compared on the values as well, over the grid below, because
//    a type-level argument is not a value-level one;
//
//  * where one precision's default is not another's - the half lanes run the
//    fp16 engine budget where the float lane runs the float one - the entry is
//    compared against the alias-composed call bit for bit, and beside it
//    against the float lane's default, so the figure that separates the two
//    budgets is on the page and not in the reasoning.
//
// The second pair of rows is the one place a difference is expected and is not
// counted as a failure: it is the measurement that says the names are not one
// name written four times.
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

// The library's private src/ directory is deliberately NOT on this file's
// include path: a consumer gets include/ and nothing else. Prove it here rather
// than assume it. Each name below resolves only if a directory holding that
// file is on the include path, and those files are the library's sources; five
// names, so that renaming one does not quietly retire the probe.
#if __has_include("boys.cpp") ||                                                                   \
                  __has_include("boys_simd.cpp") ||                                                \
                                __has_include("boys_transform.cpp") ||                             \
                                              __has_include("boys_c.cpp") ||                       \
                                                            __has_include("boys_half_native.cpp")
#error                                                                                             \
    "this consumer check is compiled with src/ on its include path; it no longer proves that a consumer can build against the public headers alone"
#endif

namespace {

using boys::Bf16;
using boys::F16;

/// The four names, one line each. What each selects, and the bound it carries,
/// are in docs/lane-contract.md.
using Fp64Default = boys::DefaultPolicyFp64;
using Fp32Default = boys::DefaultPolicyFp32;
using Fp16Default = boys::DefaultPolicyFp16;
using Bf16Default = boys::DefaultPolicyBf16;

constexpr double kFull = boys::kBoysFullAccuracyMultiplier;

// The name denotes the type the entry's parameter defaults to. For the double
// and the float lane that is `EvalPolicy<>` itself; for the half lanes it is
// the same row of axes under the fp16 engine budget, which is the one axis
// this precision reaches and the float lane does not.
static_assert(std::is_same_v<Fp64Default, boys::EvalPolicy<>>);
static_assert(std::is_same_v<Fp32Default, boys::EvalPolicy<>>);
static_assert(Fp64Default::kBudget == boys::BoysBudget::kFloat);
static_assert(Fp32Default::kBudget == boys::BoysBudget::kFloat);
static_assert(Fp16Default::kBudget == boys::BoysBudget::kFp16);
static_assert(!std::is_same_v<Fp16Default, Fp64Default>);
static_assert(std::is_same_v<Bf16Default, Fp16Default>);

// The other four axes are the shipped defaults on every one of the four names,
// and the half lanes differ from the float lane in the budget alone.
static_assert(Fp64Default::kRoute == boys::kDefaultFitRoute);
static_assert(Fp64Default::kScheme == boys::kDefaultEvalScheme);
static_assert(Fp64Default::kPack == boys::kDefaultPackAxis);
static_assert(Fp64Default::kGranularity == boys::kDefaultFitGranularity);
static_assert(Fp16Default::kRoute == boys::kDefaultFitRoute);
static_assert(Fp16Default::kScheme == boys::kDefaultEvalScheme);
static_assert(Fp16Default::kPack == boys::kDefaultPackAxis);
static_assert(Fp16Default::kGranularity == boys::kDefaultFitGranularity);

// Address identity: two function addresses compare equal in a constant
// expression exactly when the two names are one instantiation, and two
// instantiations of one entry have the same function-pointer type, so these are
// the assertions that each entry's *own* default is the name — not that the
// name is a type the entry could be called with. They are what a revision that
// changed an entry's template default without changing the alias would fail,
// which is the defect the shortcut exists to make visible. The last two are
// negative controls: a comparison that can only say yes proves nothing.
template <auto Left, auto Right> constexpr bool SameCall = (Left == Right);

static_assert(SameCall<&boys::BoysSingle<kFull>, &boys::BoysSingle<kFull, Fp64Default>>);
static_assert(SameCall<&boys::BoysAllOrders<kFull>, &boys::BoysAllOrders<kFull, Fp64Default>>);
static_assert(SameCall<&boys::BoysFixedN<kFull>, &boys::BoysFixedN<kFull, Fp64Default>>);
static_assert(
    SameCall<&boys::BoysAllNAtOrders<kFull>, &boys::BoysAllNAtOrders<kFull, Fp64Default>>);
static_assert(SameCall<&boys::BoysSingleF32<kFull>, &boys::BoysSingleF32<kFull, Fp32Default>>);
static_assert(
    SameCall<&boys::BoysAllOrdersF32<kFull>, &boys::BoysAllOrdersF32<kFull, Fp32Default>>);
static_assert(SameCall<&boys::BoysAllNF32<kFull>, &boys::BoysAllNF32<kFull, Fp32Default>>);

static_assert(!SameCall<&boys::BoysSingle<kFull>, &boys::BoysSingle<kFull, Fp16Default>>);
static_assert(!SameCall<&boys::BoysSingleF32<kFull>, &boys::BoysSingleF32<kFull, Fp16Default>>);

// --- the comparison ---------------------------------------------------------

/// One row of the report: the cells compared, how many of them the two
/// spellings answered differently, and the furthest apart they were.
struct Row {
    const char* name = "";
    std::size_t cells = 0;
    std::size_t differing = 0;
    double worst = 0.0;
    int worstN = -1;
    double worstX = 0.0;
};

std::uint16_t Bits(F16 v) noexcept {
    return std::bit_cast<std::uint16_t>(v);
}

std::uint16_t Bits(Bf16 v) noexcept {
    return std::bit_cast<std::uint16_t>(v);
}

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

/// Judges one value of a row. Comparison is on the representation, so a
/// difference that is below a format's resolution still counts: what is being
/// asked is whether the two spellings are one call, not whether they are
/// close.
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

/// A single-order pair at every order and every argument. Both callables take
/// (order, argument) and return the same type.
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

/// A many-argument pair: both callables take (top order, arguments, count,
/// output, workspace).
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

/// The arguments: a sweep of the whole supported range with the boundaries
/// where the lanes change arithmetic included as their own samples, so a
/// difference in what two spellings select has nowhere to hide between
/// samples. Sorted and deduplicated, because one of the entries compared is
/// the form that promises its arguments are in order.
std::vector<double> Arguments() {
    // The three arguments the accuracy page names: the region-A band end, the
    // region-B selector and the point the asymptotic branch takes over.
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
                boys::BoysAllOrders<kFull, Fp64Default>(boys::kMaxBoysOrder, x, out);
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
                boys::BoysAllOrdersF32<kFull, Fp32Default>(
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
    // The half lanes' entries take no policy: they run their own budget. What
    // is compared here is the entry against the call the name composes to, so
    // that the name in the documentation is the arithmetic in the entry.
    {
        Row row = SingleOrderRow<F16>(
            xs,
            [](int n, double x) {
                return boys::BoysSingleF16<kFull>(n, F16(static_cast<float>(x)));
            },
            [](int n, double x) {
                return static_cast<F16>(boys::BoysSingleF32<kFull, Fp16Default>(
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
                boys::BoysAllOrdersF32<kFull, Fp16Default>(
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
                return static_cast<Bf16>(boys::BoysSingleF32<kFull, Bf16Default>(
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
                boys::BoysAllOrdersF32<kFull, Bf16Default>(
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
    // Past the reference multiplier the budget decides the rung, so the half
    // lanes' default and the float lane's are two arithmetics. This row prints
    // how many cells they part on; it is the figure behind giving each
    // precision its own name, and it is not a failure either way. At m = 1 the
    // budget is inert, so this row would read zero and is taken at m = 2.
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

    std::printf("  %zu of the %zu identity rows differ anywhere; "
                "the shortcut and the entry are one call\n",
                failed,
                rows);

    return failed == 0 ? 0 : 1;
}
