// I have many arguments but want one order - how do I get F_n(x_i) for all i?
//
//   c++ -std=c++20 -I include examples/03_arguments_array.cpp -L build -lboys -o fn && ./fn
//
// BoysFixedN walks the array and writes every stride-th element, so the answer
// can go straight into a column of a larger structure without a copy. The same
// entry works for F_0 alone, which is the cheapest thing this library does.
#include <boys/boys_span.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

int main()
{
    const int n = 2;
    const std::array<double, 5> x{0.25, 1.0, 4.0, 12.5, 30.0};

    // stride 2: the values land in out[0], out[2], out[4] ... of a wider buffer.
    std::array<double, 10> column{};
    boys::BoysFixedN(n, x, column, 2);

    std::printf("n = %d, stride 2\n", n);
    for (std::size_t i = 0; i < 5; ++i)
    {
        std::printf("x = %-7.4g out[%2zu] = %.17g\n", x[i], i * 2, column[i * 2]);
    }

    // Every one of them must agree with the single-order entry.
    double worst = 0.0;
    for (std::size_t i = 0; i < 5; ++i)
    {
        worst = std::fmax(worst, std::fabs(column[i * 2] - boys::BoysSingle(n, x[i])));
    }
    const double bound = boys::BoysAccuracyGuaranteed(
                             boys::Precision::kFp64, boys::FitRoute::kChebyshev,
                             boys::EvalScheme::kHorner, boys::PackAxis::kArguments,
                             boys::FitGranularity::kNarrow, boys::AccuracyTier::kReference)
                             .value;
    std::printf("worst gap against BoysSingle: %.2g (guaranteed error <= %.2g)\n", worst, bound);
    return (worst <= bound) ? 0 : 1;
}
