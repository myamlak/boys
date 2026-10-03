// I need F_0(x) through F_nmax(x) at one argument - how do I get the ladder?
//
//   c++ -std=c++20 -I include examples/02_all_orders.cpp -L build -lboys -o f0n && ./f0n
//
// One call fills an array of nmax + 1 values: out[k] = F_k(x). This is the
// shape an integral code wants, because a shell quartet needs every order at
// once and the higher orders are cheaper reached from the lower ones than
// computed one at a time.
#include <boys/boys_span.hpp>

#include <array>
#include <cmath>
#include <cstdio>

int main()
{
    const int nmax = 6;
    const double x = 3.5;

    std::array<double, boys::kMaxBoysOrder + 1> ladder{};
    boys::BoysAllOrders(nmax, x, ladder);

    std::printf("k   F_%d(%.4g)\n", nmax, x);
    for (int k = 0; k <= nmax; ++k)
    {
        std::printf("%-3d %.17g\n", k, ladder[k]);
    }

    // Each order must agree with the single-order entry, inside the bound.
    const boys::AccuracyFigure bound = boys::BoysAccuracyGuaranteed(
        boys::Precision::kFp64, boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner,
        boys::PackAxis::kArguments, boys::FitGranularity::kNarrow);

    double worst = 0.0;
    for (int k = 0; k <= nmax; ++k)
    {
        worst = std::fmax(worst, std::fabs(ladder[k] - boys::BoysSingle(k, x)));
    }
    std::printf("worst gap against BoysSingle: %.2g (guaranteed error <= %.2g)\n", worst, bound.value);
    return (worst <= bound.value) ? 0 : 1;
}
