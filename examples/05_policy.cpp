// How do I ask for a specific evaluation, once I know which one I want?
//
//   c++ -std=c++20 -I include examples/05_policy.cpp -L build -lboys -o policy && ./policy
//
// A policy is named in the template argument list, never constructed. The four
// settings below are the ones a caller who has measured a preference names; a
// call site that names none gets the build's own defaults, which is the call
// this program compares against.
#include <boys/boys_span.hpp>

#include <array>
#include <cmath>
#include <cstdio>

int main()
{
    const int nmax = 4;
    const double x = 2.5;

    using Named = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                                   boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                                   boys::FitGranularity::kCoarsest,
                                   boys::DivisionForm::kExactDivision>;

    std::array<double, boys::kMaxBoysOrder + 1> default_ladder{};
    std::array<double, boys::kMaxBoysOrder + 1> named_ladder{};
    boys::BoysAllOrders(nmax, x, default_ladder);
    boys::BoysAllOrders<Named>(nmax, x, named_ladder);

    std::printf("k   default policy            named policy\n");
    double worst = 0.0;
    for (int k = 0; k <= nmax; ++k)
    {
        std::printf("%-3d %-24.17g %.17g\n", k, default_ladder[k], named_ladder[k]);
        worst = std::fmax(worst, std::fabs(named_ladder[k] - default_ladder[k]));
    }

    // The two are different sums of the same approximation of F, so they agree
    // to well inside the error the library guarantees for either.
    const double bound = boys::BoysAccuracyGuaranteed(
                             boys::Precision::kFp64, boys::FitRoute::kChebyshev,
                             boys::EvalScheme::kSplitClenshaw, boys::PackAxis::kArguments,
                             boys::FitGranularity::kCoarsest)
                             .value;
    std::printf("worst disagreement: %.2g (guaranteed error <= %.2g)\n", worst, bound);
    return (worst <= bound) ? 0 : 1;
}
