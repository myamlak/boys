// How do I evaluate F_n(x) for one order at one argument?
//   c++ -std=c++20 -DBoysFp16=1 -mavx2 -mfma -mf16c -ffp-contract=off -I include examples/01_one_order.cpp -L build -lboys -o f0 && ./f0
// The answer is one call, printed beside the error the library guarantees for it, so the value is
// read with a figure next to it rather than on trust.
#include <boys/boys_span.hpp>

#include <array>
#include <cmath>
#include <cstdio>

int main()
{
    const int n = 3;
    const double x = 1.25;

    const double f = boys::BoysSingle(n, x);

    const boys::AccuracyFigure bound =
        boys::DefaultGuarantee<boys::Precision::kFp64, boys::Shape::kSingle>();

    std::printf("F_%d(%.4g) = %.17g\n", n, x, f);
    std::printf("guaranteed error <= %.2g  (from %s)\n", bound.value, bound.source);

    // The all-orders entry computes the same value by a different route; the
    // two must agree inside the bound printed above.
    std::array<double, boys::kMaxBoysOrder + 1> ladder{};
    boys::BoysAllOrders(n, x, ladder);
    const double gap = std::fabs(ladder[n] - f);

    std::printf("BoysAllOrders agrees to %.2g\n", gap);
    if (!(gap <= bound.value) || !(f > 0.0))
    {
        std::printf("FAIL: entries disagree, or F is not positive\n");
        return 1;
    }
    return 0;
}
