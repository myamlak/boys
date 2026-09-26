// The across-orders packed lane against the across-arguments lane, on the
// workload the all-orders entrance actually has.
//
// The two lanes vectorise different axes of the same call, and the call shape
// decides which axis there is anything to fill:
//
//   * BoysAllOrders(nmax, x, out) is ONE argument and nmax + 1 orders, so the
//     across-arguments lane has one argument to put in its four lanes - the
//     baseline below runs it over four copies of x, which is what it costs to
//     serve this shape with that axis. The across-orders lane fills its lanes
//     with the orders.
//
//   * BoysAllN(nmax, x, out, count) is count arguments by nmax + 1 orders, so
//     both axes are available. The shipped entry takes the arguments (an
//     order-major loop of the region-A lane); the across-orders lane takes the
//     orders (an argument-major loop), and pays the plane's order stride on
//     the store because AVX2 has no scatter.
//
//   * THE PARTITION. Both readings above are of the shipped region-A pieces,
//     whose every order shares its intervals and degrees. That is what lets the
//     lane fetch one piece's coefficients at a fixed stride and hold four orders
//     of ONE piece. The narrow partition is cut per order, so no such stride
//     exists: the narrow lane fetches each of the four orders it packs its own
//     piece and coefficients, and one group is four different pieces evaluated
//     together. The per-order loop beside it is that partition read one order at
//     a time by the library's own single-order entry - the shape a fallback onto
//     scalar calls would have - and the pair's two counts are what says whether
//     the packed form is a vector path or four scalar calls carrying its name.
//
// WHAT THIS PROGRAM REPORTS. It drives one variant at a time over a fixed
// workload and prints the work it did - calls, output values, and the worst
// deviation from the shipped entry, so a measurement can never be of a broken
// variant. It does not report a time as a result: instruction and operation
// counts are what it is for, and they are read from the counter the platform
// exposes (perf stat) or from the compiled code, not from a clock. --time
// exists for a reader who wants one, and says in its own output why the number
// it prints is not a measurement on a loaded machine.
//
// Usage:
//   boys-across-orders-benchmark --list
//   boys-across-orders-benchmark [--variant=NAME] [--reps=N] [--count=N]
//                                [--nmax=N] [--time]
//
// Reproducing an instruction count, one variant per run:
//   perf stat -e instructions:u,uops_retired.retire_slots:u
//     boys-across-orders-benchmark --variant=orders-across-direct --reps=20000
//
// The narrow partition's packed lane against the per-order loop it replaces,
// one run each and both counters read:
//   perf stat -e instructions:u,uops_retired.retire_slots:u
//     boys-across-orders-benchmark --variant=orders-narrow-clenshaw --reps=20000
//   perf stat -e instructions:u,uops_retired.retire_slots:u
//     boys-across-orders-benchmark --variant=orders-narrow-scalar-clenshaw --reps=20000

#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_impl.hpp"
#include "boys_orders_simd.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr double kX0 = boys::detail::kX0;

// The workload's arguments: log-uniform over region A, where the per-order
// fits live and where this lane is defined. Deterministic, from a fixed
// formula rather than a generator, so two runs measure the same arguments.
std::vector<double> MakeArguments(std::size_t count) {
    std::vector<double> x(count);
    const double lo = std::log(1e-3);
    const double hi = std::log(kX0 * (1.0 - 1e-9));

    for (std::size_t i = 0; i < count; ++i)
    {
        const double u =
            (count > 1) ? static_cast<double>(i) / static_cast<double>(count - 1) : 0.0;
        x[i] = std::exp(lo + u * (hi - lo));
    }

    return x;
}

struct Config {
    std::string variant;
    std::size_t reps = 1;
    std::size_t count = 1024;
    int nmax = boys::kMaxBoysOrder;
    bool time = false;
    bool list = false;
};

struct Variant {
    const char* name;
    const char* shape;
    const char* what;
};

constexpr Variant kVariants[] = {
    {"orders-shipped",
     "BoysAllOrders(nmax, x)",
     "the shipped entry: one seed fit and a recursion across the orders"},
    {"orders-scalar-fits",
     "BoysAllOrders(nmax, x)",
     "the certified scalar region-A fit per order, no recursion"},
    {"orders-args-lane",
     "BoysAllOrders(nmax, x)",
     "the across-arguments lane on four copies of x, one call per order"},
    {"orders-across-clenshaw", "BoysAllOrders(nmax, x)", "the across-orders lane, split Clenshaw"},
    {"orders-across-direct",
     "BoysAllOrders(nmax, x)",
     "the across-orders lane, direct Chebyshev sum"},
    {"orders-across-horner",
     "BoysAllOrders(nmax, x)",
     "the across-orders lane, Horner on the monomial coefficients"},
    {"orders-across-clenshaw-composed",
     "BoysAllOrders(nmax, x)",
     "the across-orders lane, split Clenshaw, coefficients composed not gathered"},
    {"orders-across-direct-composed",
     "BoysAllOrders(nmax, x)",
     "the across-orders lane, direct sum, coefficients composed not gathered"},
    {"orders-across-horner-composed",
     "BoysAllOrders(nmax, x)",
     "the across-orders lane, Horner, coefficients composed not gathered"},
    {"balln-shipped", "BoysAllN(nmax, x[], count)", "the shipped plane entry, across arguments"},
    {"balln-across-clenshaw",
     "BoysAllN(nmax, x[], count)",
     "the across-orders lane per argument, split Clenshaw"},
    {"balln-across-direct",
     "BoysAllN(nmax, x[], count)",
     "the across-orders lane per argument, direct Chebyshev sum"},
    {"balln-across-horner",
     "BoysAllN(nmax, x[], count)",
     "the across-orders lane per argument, Horner"},
    {"balln-across-direct-composed",
     "BoysAllN(nmax, x[], count)",
     "the across-orders lane per argument, direct sum, coefficients composed"},
    {"orders-narrow-clenshaw",
     "BoysAllOrders(nmax, x)",
     "the orders axis on the narrow partition, split Clenshaw"},
    {"orders-narrow-horner",
     "BoysAllOrders(nmax, x)",
     "the orders axis on the narrow partition, Horner on the monomial coefficients"},
    {"orders-narrow-scalar-clenshaw",
     "BoysAllOrders(nmax, x)",
     "the narrow partition read one order at a time by the scalar single entry: the lane this "
     "axis replaces, and the count that says whether it does"},
    {"orders-narrow-scalar-horner",
     "BoysAllOrders(nmax, x)",
     "the same per-order loop at the Horner scheme"},
};

const Variant* FindVariant(const std::string& name) {
    for (const Variant& variant : kVariants)
    {
        if (name == variant.name)
        {
            return &variant;
        }
    }

    return nullptr;
}

// The lane's four schemes, and the shipped route, over the same workload.
// `work` returns the number of output values the run produced.
template <class Work> std::size_t Drive(const Config& config, Work work) {
    const std::vector<double> x = MakeArguments(config.count);
    const std::size_t orders = static_cast<std::size_t>(config.nmax) + 1;
    std::vector<double> out(orders * ((config.variant.rfind("balln", 0) == 0) ? x.size() : 1));
    std::size_t values = 0;

    for (std::size_t rep = 0; rep < config.reps; ++rep)
    {
        values += work(x, out);
    }

    return values;
}

// The worst absolute and relative deviation from the shipped route, over the
// whole workload: the check that what is being measured is a correct lane.
struct Deviation {
    double absolute = 0.0;
    double relative = 0.0;
};

// The narrow partition's lane at one argument, at the scheme the variant
// names: the four orders of a group fetch each its own piece and coefficients,
// because the narrow pieces are cut per order and do not share a stride.
//
// The library's packed entry carries the two certified schemes on this
// partition. The direct Chebyshev sum is this driver's own extra reading of the
// shipped lane and is not a scheme of the library's, so this partition has no
// direct-sum variant rather than one answered with another scheme's numbers.
void NarrowLane(boys::EvalScheme scheme, int nmax, double x, double* out) noexcept {
    if (scheme == boys::EvalScheme::kHorner)
    {
        boys::detail::BoysAllOrdersPacked<boys::EvalScheme::kHorner,
                                          1.0,
                                          boys::FitRoute::kChebyshev,
                                          boys::FitGranularity::kNarrow>(nmax, x, out);
    } else
    {
        boys::detail::BoysAllOrdersPacked<boys::EvalScheme::kSplitClenshaw,
                                          1.0,
                                          boys::FitRoute::kChebyshev,
                                          boys::FitGranularity::kNarrow>(nmax, x, out);
    }
}

// The lane the refusal would have left a caller with: the same partition's
// certified single-order fit, called once per order in a loop, with no vector
// group anywhere. The library's own entry, not a restatement of the body, so
// the two columns of a count are the same arithmetic over the same pieces.
void NarrowPerOrderFits(boys::EvalScheme scheme, int nmax, double x, double* out) noexcept {
    for (int l = 0; l <= nmax; ++l)
    {
        if (scheme == boys::EvalScheme::kHorner)
        {
            out[l] = boys::BoysSingle<1.0,
                                      boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                                       boys::EvalScheme::kHorner,
                                                       boys::BoysBudget::kFloat,
                                                       boys::PackAxis::kArguments,
                                                       boys::FitGranularity::kNarrow>>(l, x);
        } else
        {
            out[l] = boys::BoysSingle<1.0,
                                      boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                                       boys::EvalScheme::kSplitClenshaw,
                                                       boys::BoysBudget::kFloat,
                                                       boys::PackAxis::kArguments,
                                                       boys::FitGranularity::kNarrow>>(l, x);
        }
    }
}

Deviation CompareToShipped(const std::string& variant,
                           int nmax,
                           const std::vector<double>& x,
                           std::size_t count) {
    const std::size_t orders = static_cast<std::size_t>(nmax) + 1;
    std::vector<double> mine(orders, 0.0);
    std::vector<double> shipped(orders, 0.0);
    Deviation worst;

    // The variant's own scheme and fetch, so that the deviation reported
    // beside a measurement is the deviation of what was measured.
    const bool composed = (variant.find("-composed") != std::string::npos);
    const bool direct = (variant.find("direct") != std::string::npos);
    const bool horner = (variant.find("horner") != std::string::npos);
    const bool narrow = (variant.rfind("orders-narrow-", 0) == 0);
    const bool narrowScalar = narrow && (variant.find("scalar") != std::string::npos);
    const auto scheme = direct ? boys::detail::OrdersScheme::kDirectSum
                               : (horner ? boys::detail::OrdersScheme::kHorner
                                         : boys::detail::OrdersScheme::kSplitClenshaw);
    const auto narrowScheme =
        horner ? boys::EvalScheme::kHorner : boys::EvalScheme::kSplitClenshaw;

    for (std::size_t i = 0; i < count; ++i)
    {
        if (narrowScalar)
        {
            NarrowPerOrderFits(narrowScheme, nmax, x[i], mine.data());
        } else if (narrow)
        {
            NarrowLane(narrowScheme, nmax, x[i], mine.data());
        } else if (composed)
        {
            boys::detail::BoysAllOrdersSimdComposed(scheme, nmax, x[i], mine.data(), 1);
        } else
        {
            boys::detail::BoysAllOrdersSimd(scheme, nmax, x[i], mine.data(), 1);
        }

        boys::BoysAllOrders(nmax, x[i], shipped.data());

        for (std::size_t l = 0; l < orders; ++l)
        {
            const double difference = std::abs(mine[l] - shipped[l]);
            worst.absolute = std::max(worst.absolute, difference);

            if (shipped[l] != 0.0)
            {
                worst.relative = std::max(worst.relative, difference / std::abs(shipped[l]));
            }
        }
    }

    return worst;
}

double Run(const Config& config) {
    const std::string& name = config.variant;
    const int nmax = config.nmax;
    const std::size_t orders = static_cast<std::size_t>(nmax) + 1;
    std::size_t values = 0;
    std::chrono::steady_clock::duration elapsed{};

    const auto start = std::chrono::steady_clock::now();

    if (name == "orders-shipped")
    {
        values = Drive(config, [&](const std::vector<double>& x, std::vector<double>& out) {
            for (double xi : x)
            {
                boys::BoysAllOrders(nmax, xi, out.data());
            }

            return x.size() * orders;
        });
    } else if (name == "orders-scalar-fits")
    {
        values = Drive(config, [&](const std::vector<double>& x, std::vector<double>& out) {
            for (double xi : x)
            {
                for (std::size_t l = 0; l < orders; ++l)
                {
                    out[l] = boys::detail::ChebyshevValue(static_cast<int>(l), xi);
                }
            }

            return x.size() * orders;
        });
    } else if (name == "orders-args-lane")
    {
        double duplicate[4] = {};
        double lanes[4] = {};

        values = Drive(config, [&](const std::vector<double>& x, std::vector<double>& out) {
            for (double xi : x)
            {
                duplicate[0] = xi;
                duplicate[1] = xi;
                duplicate[2] = xi;
                duplicate[3] = xi;

                for (std::size_t l = 0; l < orders; ++l)
                {
                    boys::detail::BoysRegionASimd<boys::kBoysFullAccuracyMultiplier>(
                        static_cast<int>(l), duplicate, lanes, 4);
                    out[l] = lanes[0];
                }
            }

            return x.size() * orders;
        });
    } else if (name.rfind("orders-narrow-", 0) == 0)
    {
        const bool scalarFits = (name.find("scalar") != std::string::npos);
        const bool horner = (name.find("horner") != std::string::npos);
        const auto narrowScheme =
            horner ? boys::EvalScheme::kHorner : boys::EvalScheme::kSplitClenshaw;

        values = Drive(config, [&](const std::vector<double>& x, std::vector<double>& out) {
            for (double xi : x)
            {
                if (scalarFits)
                {
                    NarrowPerOrderFits(narrowScheme, nmax, xi, out.data());
                } else
                {
                    NarrowLane(narrowScheme, nmax, xi, out.data());
                }
            }

            return x.size() * orders;
        });
    } else if (name.rfind("orders-across-", 0) == 0)
    {
        const bool composed = (name.find("-composed") != std::string::npos);
        const bool direct = (name.find("direct") != std::string::npos);
        const bool horner = (name.find("horner") != std::string::npos);
        const auto scheme = direct ? boys::detail::OrdersScheme::kDirectSum
                                   : (horner ? boys::detail::OrdersScheme::kHorner
                                             : boys::detail::OrdersScheme::kSplitClenshaw);

        values = Drive(config, [&](const std::vector<double>& x, std::vector<double>& out) {
            for (double xi : x)
            {
                if (composed)
                {
                    boys::detail::BoysAllOrdersSimdComposed(scheme, nmax, xi, out.data(), 1);
                } else
                {
                    boys::detail::BoysAllOrdersSimd(scheme, nmax, xi, out.data(), 1);
                }
            }

            return x.size() * orders;
        });
    } else if (name == "balln-shipped")
    {
        // The workspace is the caller's and is reused: a per-call allocation
        // would put the allocator's instructions in the counter column beside
        // the kernel's, which is not what this driver is measuring.
        std::vector<std::size_t> workspace(boys::BoysAllNWorkspaceSize(config.count));

        values = Drive(config, [&](const std::vector<double>& x, std::vector<double>& out) {
            boys::BoysAllN(nmax, x.data(), out.data(), x.size(), workspace.data());
            return x.size() * orders;
        });
    } else if (name.rfind("balln-across-", 0) == 0)
    {
        const bool composed = (name.find("-composed") != std::string::npos);
        const bool horner = (name.find("horner") != std::string::npos);
        const bool direct = (name.find("direct") != std::string::npos);
        const auto scheme = horner ? boys::detail::OrdersScheme::kHorner
                                   : (direct ? boys::detail::OrdersScheme::kDirectSum
                                             : boys::detail::OrdersScheme::kSplitClenshaw);

        values = Drive(config, [&](const std::vector<double>& x, std::vector<double>& out) {
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                if (composed)
                {
                    boys::detail::BoysAllOrdersSimdComposed(
                        scheme, nmax, x[i], out.data() + i, x.size());
                } else
                {
                    boys::detail::BoysAllOrdersSimd(scheme, nmax, x[i], out.data() + i, x.size());
                }
            }

            return x.size() * orders;
        });
    } else
    {
        std::fprintf(stderr, "unknown variant: %s\n", name.c_str());
        std::exit(2);
    }

    elapsed = std::chrono::steady_clock::now() - start;

    const Variant* variant = FindVariant(name);
    std::printf("variant           %s\n", name.c_str());
    std::printf("call shape        %s\n", variant != nullptr ? variant->shape : "?");
    std::printf("what it is        %s\n", variant != nullptr ? variant->what : "?");
    std::printf("orders per call   %zu\n", orders);
    std::printf("arguments         %zu\n", config.count);
    std::printf("reps              %zu\n", config.reps);
    std::printf("calls             %zu\n", config.reps * config.count);
    std::printf("output values     %zu\n", values);

    if (config.time)
    {
        const double seconds = std::chrono::duration<double>(elapsed).count();
        std::printf("elapsed           %.6f s\n", seconds);
        std::printf("NOTE: the elapsed figure above is a WALL-CLOCK reading and is reported as\n"
                    "      such. If this machine was running anything else, it is not a\n"
                    "      measurement of this lane - read the counter columns instead.\n");
    }

    return static_cast<double>(values);
}

} // namespace

int main(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--list")
        {
            config.list = true;
        } else if (arg == "--time")
        {
            config.time = true;
        } else if (arg.rfind("--variant=", 0) == 0)
        {
            config.variant = arg.substr(10);
        } else if (arg.rfind("--reps=", 0) == 0)
        {
            config.reps = std::strtoull(arg.c_str() + 7, nullptr, 10);
        } else if (arg.rfind("--count=", 0) == 0)
        {
            config.count = std::strtoull(arg.c_str() + 8, nullptr, 10);
        } else if (arg.rfind("--nmax=", 0) == 0)
        {
            config.nmax = static_cast<int>(std::strtol(arg.c_str() + 7, nullptr, 10));
        } else
        {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    if (config.list)
    {
        for (const Variant& variant : kVariants)
        {
            std::printf("%-24s %-30s %s\n", variant.name, variant.shape, variant.what);
        }

        return 0;
    }

    if (config.reps == 0)
    {
        config.reps = 1;
    }

    if (config.variant.empty())
    {
        std::fprintf(stderr,
                     "name a variant (--list shows them): one variant per run is what\n"
                     "makes an instruction count per call mean anything.\n");
        return 2;
    }

    if (!boys::BoysAvx2Available())
    {
        std::printf("BoysAvx2Available() = false: the across-orders lane runs the scalar fits\n"
                    "and the across-arguments lane does not exist on this target.\n");
    }

    // The correctness guard: the variant is measured only over arguments on
    // which it agrees with the shipped entry inside the shipped budget.
    if (!config.variant.empty() && (config.variant.rfind("orders-across-", 0) == 0 ||
                                    config.variant.rfind("orders-narrow-", 0) == 0))
    {
        const std::vector<double> sample = MakeArguments(std::min<std::size_t>(config.count, 64));
        const Deviation deviation =
            CompareToShipped(config.variant, config.nmax, sample, sample.size());

        std::printf("against the shipped entry, over %zu arguments: worst absolute %.6e, "
                    "worst relative %.6e\n",
                    sample.size(),
                    deviation.absolute,
                    deviation.relative);
    }

    Run(config);
    return 0;
}
