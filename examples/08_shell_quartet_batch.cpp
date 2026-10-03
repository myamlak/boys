// A shell quartet does not ask for one order. What does the batch it needs cost?
//
//   c++ -std=c++20 -I include examples/08_shell_quartet_batch.cpp -L build -lboys -o quartet
//
// Every argument of a real batch arrives with its own highest order, and they are
// not the same: a quartet's demand runs to la + lb + lc + ld for the argument its
// exponents produce. Padding all of them to the batch's largest is the easy call
// and it is not free. This program asks for both shapes over the same arguments
// and counts the values each one had to produce.
//
// The accuracy question is the other half. The bound and the speed are one
// choice, so the program asks the library what it guarantees before it relies on
// the answer, and checks the answer against that.
#include <boys/boys_span.hpp>

#include <array>
#include <cstdio>

int main()
{
    // Five arguments of one batch and the highest order each needs. The first
    // and the third ask for little, the fourth for a lot.
    const std::size_t count = 5;
    const std::array<int, 5> n{3, 6, 2, 8, 4};
    const std::array<double, 5> x{0.5, 2.0, 7.5, 0.125, 20.0};

    int largest = 0;
    std::size_t demanded = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
        largest = n[i] > largest ? n[i] : largest;
        demanded += static_cast<std::size_t>(n[i]) + 1;
    }

    // out[k * count + i] = F_k(x[i]), for k up to that argument's own n[i].
    std::array<double, 5 * 9> perArgument{};
    boys::BoysAllNAtOrders(n, x, perArgument);

    // The same arguments at one common top order, which is what padding means.
    std::array<double, 5 * 9> padded{};
    boys::BoysAllN(largest, x, padded);

    const std::size_t paddedCells = count * (static_cast<std::size_t>(largest) + 1);

    std::printf("five arguments, each at its own top order\n");
    for (std::size_t i = 0; i < count; ++i)
    {
        std::printf("  F_%d(%.4g) = %.17g\n", n[i], x[i],
                    perArgument[static_cast<std::size_t>(n[i]) * count + i]);
    }
    std::printf("values the per-argument call produces: %zu\n", demanded);
    std::printf("values the padded call produces:        %zu\n", paddedCells);
    std::printf("produced for nothing:                   %zu\n", paddedCells - demanded);

    // Every cell the per-argument call wrote must equal the padded call's, since
    // both evaluate the same function at the same arguments.
    std::size_t compared = 0;
    double worst = 0.0;
    for (std::size_t i = 0; i < count; ++i)
    {
        for (int k = 0; k <= n[i]; ++k)
        {
            const std::size_t at = static_cast<std::size_t>(k) * count + i;
            const double difference = perArgument[at] - padded[at];
            worst = difference > worst ? difference : worst;
            ++compared;
        }
    }
    std::printf("cells compared against the padded call:  %zu, worst difference %.3g\n", compared,
                worst);

    // What the library promises for this call, asked rather than assumed. The
    // default policy is what a call naming no policy compiles.
    const boys::AccuracyFigure guaranteed =
        boys::BoysAccuracyGuaranteed(boys::Precision::kFp64, boys::kDefaultFitRoute,
                                     boys::kDefaultEvalScheme, boys::kDefaultPackAxis,
                                     boys::kDefaultFitGranularity);
    std::printf("guaranteed error per value:             %.3g  (%s)\n", guaranteed.value,
                guaranteed.source);

    // F is positive and falls off with x, and an answer that is not is not an
    // answer. The check is on the value the caller asked for, not on the padding.
    for (std::size_t i = 0; i < count; ++i)
    {
        const double value = perArgument[static_cast<std::size_t>(n[i]) * count + i];
        if (!(value > 0.0) || !(value <= 1.0))
        {
            std::printf("FAIL: F_%d(%.4g) = %.17g is not a Boys value\n", n[i], x[i], value);
            return 1;
        }
    }
    std::printf("every value positive and at most one\n");
    return 0;
}
