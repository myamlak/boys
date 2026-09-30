// Can I trade accuracy for speed, and what does the trade actually buy?
//
//   c++ -std=c++20 -I include examples/04_accuracy_tier.cpp -L build -lboys -o tier && ./tier
//
// Every entry accepts a multiplier on its error. 1 is the default; larger
// values allow a larger error and do less work. The number to read is the
// guaranteed error printed beside each value - that is the whole price list.
#include <boys/boys.hpp>

#include <cmath>
#include <cstdio>

int main()
{
    const int n = 3;
    const double x = 1.25;
    const boys::AccuracyTier tiers[] = {boys::AccuracyTier::kReference,
                                        boys::AccuracyTier::kRelaxed64,
                                        boys::AccuracyTier::kRelaxed1024,
                                        boys::AccuracyTier::kRelaxed65536};

    std::printf("F_%d(%.4g), the same call at four multipliers:\n", n, x);
    const double reference = boys::BoysSingleAtTier(boys::AccuracyTier::kReference, n, x);
    double worst = 0.0;
    for (const boys::AccuracyTier tier : tiers) {
        const double value = boys::BoysSingleAtTier(tier, n, x);
        const double bound = boys::BoysAccuracyGuaranteed(
                                 boys::Precision::kFp64, boys::FitRoute::kChebyshev,
                                 boys::EvalScheme::kHorner, boys::PackAxis::kArguments,
                                 boys::FitGranularity::kNarrow, tier)
                                 .value;
        std::printf("  m = %-7.0f value = %.17g  guaranteed error <= %.2g\n",
                    boys::AccuracyMultiplier(tier), value, bound);

        // A relaxed answer must stay inside the bound its own multiplier
        // promises, measured against the full-accuracy answer.
        worst = std::fmax(worst, std::fabs(value - reference));
        if (!(std::fabs(value - reference) <= bound)) {
            std::printf("FAIL: m = %.0f left its own bound\n", boys::AccuracyMultiplier(tier));
            return 1;
        }
    }
    std::printf("worst departure from the m = 1 answer: %.2g\n", worst);
    return 0;
}
