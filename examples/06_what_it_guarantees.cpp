// Will this entry meet the error my calculation needs?
//
//   c++ -std=c++20 -I include examples/06_what_it_guarantees.cpp -L build -lboys -o check && ./check
//
// Ask before you rely on it. The answer is decided by the guaranteed error, not
// by the error the lane happened to deliver on somebody's test grid, so a "yes"
// here is something a calculation can rest on.
#include <boys/boys_span.hpp>

#include <array>
#include <cstdio>

namespace {

const char* VerdictName(boys::ToleranceVerdict verdict)
{
    switch (verdict)
    {
    case boys::ToleranceVerdict::kGuaranteedInside:
        return "YES - guaranteed";
    case boys::ToleranceVerdict::kDeliveredInside:
        return "only measured, not guaranteed";
    case boys::ToleranceVerdict::kOutside:
        return "NO";
    case boys::ToleranceVerdict::kNotCarried:
        return "not offered by this build";
    }
    return "?";
}

} // namespace

int main()
{
    const std::array<double, 3> wanted{1e-12, 1e-10, 1e-8};

    std::printf("the double entry, every order, as it stands:\n");
    std::printf("  requested    verdict                  guaranteed   measured\n");
    for (const double tolerance : wanted)
    {
        const boys::CombinationCoverage answer =
            boys::QueryCombination(boys::Precision::kFp64, boys::FitRoute::kChebyshev,
                                   boys::EvalScheme::kHorner, boys::PackAxis::kArguments,
                                   boys::FitGranularity::kNarrow, boys::AccuracyTier::kReference,
                                   tolerance);
        std::printf("  %-12.2g %-22s %-12.2g %.2g  [%s]\n", tolerance,
                    VerdictName(answer.verdict), answer.bound, answer.delivered, answer.source);
    }

    // Asking for something impossible must not come back as a "yes".
    const boys::CombinationCoverage absurd =
        boys::QueryCombination(boys::Precision::kFp64, boys::FitRoute::kChebyshev,
                               boys::EvalScheme::kHorner, boys::PackAxis::kArguments,
                               boys::FitGranularity::kNarrow, boys::AccuracyTier::kReference, 1e-20);
    std::printf("1e-20 is answered %s\n", VerdictName(absurd.verdict));
    return (absurd.verdict != boys::ToleranceVerdict::kGuaranteedInside) ? 0 : 1;
}
