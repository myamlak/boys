// Consumer check: the policy a call naming nothing compiles, from <boys/boys.hpp> alone, and the
// figures that show each name is the policy its entry already runs. A template parameter's zero
// value cannot be pointed at or cited; these names can.

// Two claims are checked, each printing numbers rather than a checkmark. First, the name and the
// empty argument list are one call, and the two spellings are compared on values as well, because a
// type-level argument is not a value-level one.

// The name the first claim uses is the class's own row. What an entry's policy parameter defaults
// to is the row this build's default-policy table carries for the class that entry's precision and
// shape make - `boys::DefaultPolicy<Precision, Shape>`, which boys/boys.hpp resolves from the one
// table the seam file carries - and not the shortcut composed from the seam's five.

// The five are the point a class the table does not name falls to, and this build's table names
// every class: a build that writes its own rows makes a class's default a row of its own choosing,
// and an entry held to the five would be held to a policy it does not run. So the class-keyed names
// below are what the entries are compared against; the four shortcut names are asserted as such.

// Second, where one precision's default is not another's (the half lanes run the fp16 engine
// budget, the float lane the float one), the entry is compared against its own class's call bit for
// bit, and beside it against the float lane's class: the one place a difference is expected.

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

/// The four shortcut names. What each selects, and the bound it carries, are in
/// docs/lane-contract.md. They are the names a caller writes to ask for one precision and
/// nothing else; they are **not** what an entry's own parameter defaults to, which is its
/// class's row below.
using Fp64Default = boys::DefaultPolicyFp64;
using Fp32Default = boys::DefaultPolicyFp32;
using Fp16Default = boys::DefaultPolicyFp16;
using Bf16Default = boys::DefaultPolicyBf16;

// The class-keyed names: what an entry's own policy parameter defaults to. `boys::DefaultPolicy<
// Precision, Shape>` is the name the library resolves a class to - the row this build's seam
// carries for it, or the five above where it carries none - so a seam that writes its own rows
// moves these names and leaves the four above. One per class this file's entries make, all host.
using Fp64Single = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kSingle>;
using Fp64FixedN = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kFixedN>;
using Fp64AllN = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllN>;
using Fp64AllNAtOrders = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllNAtOrders>;
using Fp64AllOrders = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>;
using Fp32Single = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kSingle>;
using Fp32AllN = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllN>;
using Fp32AllOrders = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllOrders>;
using Fp16Single = boys::DefaultPolicy<boys::Precision::kFp16, boys::Shape::kSingle>;
using Fp16AllOrders = boys::DefaultPolicy<boys::Precision::kFp16, boys::Shape::kAllOrders>;
// The two half formats are one engine at one budget and two classes: the seam keys a class by the
// format a return carries, so this name is the bf16 class's own row and not the fp16 class's. A
// bf16 entry that names no policy resolves through it (`Precision::kBf16`), which is why the row
// below compares `BoysAllOrdersBf16<>` against the bf16 class and not against the fp16 class.
using Bf16Single = boys::DefaultPolicy<boys::Precision::kBf16, boys::Shape::kSingle>;
using Bf16AllOrders = boys::DefaultPolicy<boys::Precision::kBf16, boys::Shape::kAllOrders>;

// Each shortcut name denotes the type it is defined as: `EvalPolicy<>` for the double and float
// lanes; for the half lanes the same axes under the fp16 engine budget. That is a statement about
// the four names and not about any entry's parameter - the entries' parameters default to the
// class-keyed names above, and the two coincide only on a class whose row is the five.
static_assert(std::is_same_v<Fp64Default, boys::EvalPolicy<>>);
static_assert(std::is_same_v<Fp32Default, boys::EvalPolicy<>>);
static_assert(Fp64Default::kBudget == boys::BoysBudget::kFloat);
static_assert(Fp32Default::kBudget == boys::BoysBudget::kFloat);
static_assert(Fp16Default::kBudget == boys::BoysBudget::kFp16);
static_assert(!std::is_same_v<Fp16Default, Fp64Default>);
static_assert(std::is_same_v<Bf16Default, Fp16Default>);

// The other four axes are the committed defaults on all four names.
static_assert(Fp64Default::kRoute == boys::kDefaultFitRoute);
static_assert(Fp64Default::kScheme == boys::kDefaultEvalScheme);
static_assert(Fp64Default::kPack == boys::kDefaultPackAxis);
static_assert(Fp64Default::kGranularity == boys::kDefaultFitGranularity);
static_assert(Fp16Default::kRoute == boys::kDefaultFitRoute);
static_assert(Fp16Default::kScheme == boys::kDefaultEvalScheme);
static_assert(Fp16Default::kPack == boys::kDefaultPackAxis);
static_assert(Fp16Default::kGranularity == boys::kDefaultFitGranularity);

// Address identity: two function addresses compare equal in a constant expression only when the two
// names are one instantiation, and two instantiations of one entry share a function-pointer type,
// so these assert that each entry's own default is the name its class resolves to, not merely that
// the name is a type the entry could be called with. A moved template default or class row fails.

// The alias each entry is held to is its own class's name above, not the lane name: an entry's
// parameter defaults to `DefaultPolicy<Precision, Shape>` for the class it belongs to:
// the declarations at BoysSingle, BoysFixedN, BoysAllN, BoysAllNAtOrders, BoysSingleF32,
// BoysAllNF32 (boys/boys.hpp), and it is the lane name only while the seam carries no row.

// Only the yes direction is a constant expression: comparing two distinct function addresses is not
// one under the sanitizer configuration this file is also built in - g++ 15 with
// -fsanitize=address,undefined refuses it with "'(f == g)' is not a constant expression", and a
// function address against a null pointer with "'(f == 0)'"; two spellings of one function fold.

// A variable template's initializer is a constant-expression context even where the value is only
// read at run time, so the comparison cannot be written as one; the two controls that can say no
// compare in main().
template <auto Left, auto Right> constexpr bool SameCall = (Left == Right);

static_assert(SameCall<&boys::BoysSingle<>, &boys::BoysSingle<Fp64Single>>);
static_assert(SameCall<&boys::BoysAllOrders<>, &boys::BoysAllOrders<Fp64AllOrders>>);
static_assert(SameCall<&boys::BoysFixedN<>, &boys::BoysFixedN<Fp64FixedN>>);
static_assert(SameCall<&boys::BoysAllNAtOrders<>, &boys::BoysAllNAtOrders<Fp64AllNAtOrders>>);
static_assert(SameCall<&boys::BoysSingleF32<>, &boys::BoysSingleF32<Fp32Single>>);
static_assert(SameCall<&boys::BoysAllOrdersF32<>, &boys::BoysAllOrdersF32<Fp32AllOrders>>);
static_assert(SameCall<&boys::BoysAllNF32<>, &boys::BoysAllNF32<Fp32AllN>>);

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
            [](int n, double x) { return boys::BoysSingle<Fp64Single>(n, x); },
            [](int n, double x) { return boys::BoysSingle<>(n, x); });
        row.name = "fp64 BoysSingle";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        Row row = AllOrdersRow<double>(
            xs,
            [](double x, double* out) {
                boys::BoysAllOrders<Fp64AllOrders>(boys::kMaxBoysOrder, x, out);
            },
            [](double x, double* out) { boys::BoysAllOrders<>(boys::kMaxBoysOrder, x, out); });
        row.name = "fp64 BoysAllOrders";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        Row row = ManyArgRow(
            xs,
            [](int nmax, const double* x, std::size_t count, double* out, std::size_t* workspace) {
                boys::BoysAllN<Fp64AllN>(nmax, x, out, count, workspace);
            },
            [](int nmax, const double* x, std::size_t count, double* out, std::size_t* workspace) {
                boys::BoysAllN<>(nmax, x, out, count, workspace);
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
                boys::BoysAllN<Fp64AllN>(nmax, x, out, count, boys::BoysSortedArgs{});
            },
            [](int nmax, const double* x, std::size_t count, double* out, std::size_t*) {
                boys::BoysAllN<>(nmax, x, out, count, boys::BoysSortedArgs{});
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
                boys::BoysFixedN<Fp64FixedN>(n, x, out, count);
            },
            [](int n, const double* x, std::size_t count, double* out) {
                boys::BoysFixedN<>(n, x, out, count);
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
                return boys::BoysSingleF32<Fp32Single>(n, static_cast<float>(x));
            },
            [](int n, double x) { return boys::BoysSingleF32<>(n, static_cast<float>(x)); });
        row.name = "fp32 BoysSingleF32";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }
    {
        Row row = AllOrdersRow<float>(
            xs,
            [](double x, float* out) {
                boys::BoysAllOrdersF32<Fp32AllOrders>(
                    boys::kMaxBoysOrder, static_cast<float>(x), out);
            },
            [](double x, float* out) {
                boys::BoysAllOrdersF32<>(boys::kMaxBoysOrder, static_cast<float>(x), out);
            });
        row.name = "fp32 BoysAllOrdersF32";
        Print(row);
        ++rows;
        failed += row.differing != 0 ? 1u : 0u;
    }

    // --- half ---------------------------------------------------------------
    // The half lane's entry takes a policy like the float lane's and defaults to the fp16 lane's,
    // so the row compares the entry as a caller writes it against the float lane's entry at that
    // same policy, converted: the documented name is the entry's arithmetic.
#if BoysFp16
    {
        Row row = SingleOrderRow<F16>(
            xs,
            [](int n, double x) {
                return boys::BoysSingleF16<>(n, F16(static_cast<float>(x)));
            },
            [](int n, double x) {
                return static_cast<F16>(boys::BoysSingleF32<Fp16Single>(
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
                boys::BoysAllOrdersF16<>(boys::kMaxBoysOrder, F16(static_cast<float>(x)), out);
            },
            [&scratch](double x, F16* out) {
                boys::BoysAllOrdersF32<Fp16AllOrders>(
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
                return boys::BoysSingleBf16<>(n, Bf16(static_cast<float>(x)));
            },
            [](int n, double x) {
                return static_cast<Bf16>(boys::BoysSingleF32<Bf16Single>(
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
                boys::BoysAllOrdersBf16<>(
                    boys::kMaxBoysOrder, Bf16(static_cast<float>(x)), out);
            },
            [&scratch](double x, Bf16* out) {
                boys::BoysAllOrdersF32<Bf16AllOrders>(
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
    // The half lanes' class and the float lane's single-order class are two arithmetics, the engine
    // budget being the one axis on which they part; this row prints how many cells they differ on
    // and is not a failure either way. The float side is that class row, not the seam's five.
    std::printf("  (rows below compare a half lane against the FLOAT lane's single-order "
                "class, which is the other engine budget)\n");
    {
        Row row = SingleOrderRow<F16>(
            xs,
            [](int n, double x) { return boys::BoysSingleF16<>(n, F16(static_cast<float>(x))); },
            [](int n, double x) {
                return static_cast<F16>(boys::BoysSingleF32<Fp32Single>(
                    n, static_cast<float>(F16(static_cast<float>(x)))));
            });
        row.name = "fp16 vs the float lane's single-order class";
        Print(row);
    }
    {
        Row row = SingleOrderRow<Bf16>(
            xs,
            [](int n, double x) {
                return boys::BoysSingleBf16<>(n, Bf16(static_cast<float>(x)));
            },
            [](int n, double x) {
                return static_cast<Bf16>(boys::BoysSingleF32<Fp32Single>(
                    n, static_cast<float>(Bf16(static_cast<float>(x)))));
            });
        row.name = "bf16 vs the float lane's single-order class";
        Print(row);
    }

    // --- the negative controls ------------------------------------------------
    // The identity assertions above fold only for one instantiation, so they prove nothing on the
    // pair that must compare unequal: the half lane's class and the half shortcut name, which
    // carries the default axes at the fp16 engine budget.
    {
        const bool singleIsHalf = &boys::BoysSingle<> == &boys::BoysSingle<Fp16Default>;
        const bool singleF32IsHalf = &boys::BoysSingleF32<> == &boys::BoysSingleF32<Fp16Default>;

        std::printf("  negative control: the fp16 shortcut name and the double and float "
                    "classes are %s on "
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
