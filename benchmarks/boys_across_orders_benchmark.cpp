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
// The single-precision lane's figures are read the same way, over the workload
// they are quoted with, one variant per run:
//   perf stat -e instructions:u,uops_retired.retire_slots:u
//     boys-across-orders-benchmark --variant=orders-f32-orders-axis --reps=2000
//   (and the same line with --variant=orders-f32-shipped,
//    --variant=orders-f32-scalar-fits, --variant=orders-f32-across-clenshaw,
//    --variant=orders-f32-across-composed)

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

    // The single-precision lane, where a register holds eight orders rather
    // than four. The baseline is the per-order fit loop the float lane's own
    // tables are read by one order at a time - the same baseline the double
    // lane is measured against above, so the two widths are read off the same
    // comparison.
    {"orders-f32-shipped",
     "BoysAllOrdersF32(nmax, x)",
     "the shipped float entry: one seed fit and a recursion across the orders"},
    {"orders-f32-scalar-fits",
     "BoysAllOrdersF32(nmax, x)",
     "the certified scalar float region-A fit per order, no recursion"},
    {"orders-f32-across-clenshaw",
     "BoysAllOrdersF32(nmax, x)",
     "the float across-orders lane, eight orders to a register, split Clenshaw"},
    {"orders-f32-across-composed",
     "BoysAllOrdersF32(nmax, x)",
     "the float across-orders lane, eight orders to a register, bases composed not gathered"},
    {"orders-f32-orders-axis",
     "BoysAllOrdersF32(nmax, x)",
     "the shipped entry of the orders axis on the float lane: the policy the public call "
     "names, which is the fetch the entry itself selects"},
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
    const auto scheme = direct ? boys::detail::OrdersScheme::kDirectSum
                               : (horner ? boys::detail::OrdersScheme::kHorner
                                         : boys::detail::OrdersScheme::kSplitClenshaw);

    for (std::size_t i = 0; i < count; ++i)
    {
        if (composed)
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
    } else if (name.rfind("orders-f32-", 0) == 0)
    {
        // The float lane's variants. The buffer is the caller's and is
        // constructed once, outside the rep loop, so the allocator's
        // instructions are not counted beside the kernel's.
        std::vector<float> fout(orders, 0.0f);

        values = Drive(config, [&](const std::vector<double>& x, std::vector<double>& out) {
            (void)out;

            for (double xi : x)
            {
                const float x32 = static_cast<float>(xi);

                if (name == "orders-f32-shipped")
                {
                    boys::BoysAllOrdersF32(nmax, x32, fout.data());
                } else if (name == "orders-f32-scalar-fits")
                {
                    for (int l = 0; l <= nmax; ++l)
                    {
                        fout[static_cast<std::size_t>(l)] =
                            boys::detail::ChebyshevValueF32<boys::EvalScheme::kSplitClenshaw>(l,
                                                                                             x32);
                    }
                } else if (name == "orders-f32-across-composed")
                {
                    boys::detail::BoysAllOrdersF32SimdComposed(
                        boys::detail::OrdersScheme::kSplitClenshaw,
                        boys::FitRoute::kChebyshev,
                        nmax,
                        x32,
                        fout.data());
                } else if (name == "orders-f32-orders-axis")
                {
                    using OrdersAxis =
                        boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                          boys::EvalScheme::kSplitClenshaw,
                                          boys::BoysBudget::kFloat,
                                          boys::PackAxis::kOrders>;

                    boys::BoysAllOrdersF32<1.0, OrdersAxis>(nmax, x32, fout.data());
                } else
                {
                    boys::detail::BoysAllOrdersF32Simd(boys::detail::OrdersScheme::kSplitClenshaw,
                                                       boys::FitRoute::kChebyshev,
                                                       nmax,
                                                       x32,
                                                       fout.data());
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
    if (!config.variant.empty() && config.variant.rfind("orders-across-", 0) == 0)
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

    // The same guard on the float lane, at the bound that lane documents: the
    // packed lane rides fused arithmetic by construction, so on a build whose
    // scalar arithmetic is the two-rounding route the guard reading is the
    // float budget and not a bit comparison.
    if (!config.variant.empty() &&
        (config.variant.rfind("orders-f32-across", 0) == 0 ||
         config.variant == "orders-f32-orders-axis"))
    {
        const std::vector<double> sample = MakeArguments(std::min<std::size_t>(config.count, 64));
        std::vector<float> mine(static_cast<std::size_t>(config.nmax) + 1, 0.0f);
        std::vector<float> shipped(static_cast<std::size_t>(config.nmax) + 1, 0.0f);
        float worst = 0.0f;

        for (double xi : sample)
        {
            const float x32 = static_cast<float>(xi);

            if (config.variant == "orders-f32-across-composed")
            {
                boys::detail::BoysAllOrdersF32SimdComposed(boys::detail::OrdersScheme::kSplitClenshaw,
                                                           boys::FitRoute::kChebyshev,
                                                           config.nmax,
                                                           x32,
                                                           mine.data());
            } else if (config.variant == "orders-f32-orders-axis")
            {
                using OrdersAxis =
                    boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::BoysBudget::kFloat,
                                      boys::PackAxis::kOrders>;

                boys::BoysAllOrdersF32<1.0, OrdersAxis>(config.nmax, x32, mine.data());
            } else
            {
                boys::detail::BoysAllOrdersF32Simd(boys::detail::OrdersScheme::kSplitClenshaw,
                                                   boys::FitRoute::kChebyshev,
                                                   config.nmax,
                                                   x32,
                                                   mine.data());
            }

            boys::BoysAllOrdersF32(config.nmax, x32, shipped.data());

            for (int l = 0; l <= config.nmax; ++l)
            {
                worst = std::max(worst,
                                 std::abs(mine[static_cast<std::size_t>(l)] -
                                          shipped[static_cast<std::size_t>(l)]));
            }
        }

        std::printf("against the shipped entry, over %zu arguments: worst absolute %.6e "
                    "(the float lane documents 1.5e-7)\n",
                    sample.size(),
                    static_cast<double>(worst));
    }

    Run(config);
    return 0;
}
