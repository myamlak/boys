// What does it look like to call this library? - the program the README opens with.
//
//   c++ -std=c++20 -I include examples/00_first_call.cpp -L build -lboys -o first && ./first
//
// Three calls cover most of what a program does with this function: one order
// at one argument, every order at one argument, and one order over an array of
// arguments. Everything else in this directory is one of these three, asked
// about more precisely.
#include <boys/boys.hpp>

#include <cstdio>

int main()
{
    // One order at one argument.
    std::printf("F_3(1.25)  = %.17g\n", boys::BoysSingle(3, 1.25));

    // Every order 0..6 at one argument. A shell quartet wants the whole
    // ladder, and this is the call that hands it over.
    double ladder[boys::kMaxBoysOrder + 1] = {};
    boys::BoysAllOrders(6, 3.5, ladder);
    std::printf("F_0(3.5)   = %.17g\n", ladder[0]);
    std::printf("F_3(3.5)   = %.17g\n", ladder[3]);
    std::printf("F_6(3.5)   = %.17g\n", ladder[6]);

    // One order over an array of arguments. out[i] = F_2(x[i]).
    const double x[3] = {0.25, 4.0, 30.0};
    double out[3] = {};
    boys::BoysFixedN(2, x, out, 3);
    std::printf("F_2(0.25)  = %.17g\n", out[0]);
    std::printf("F_2(4)     = %.17g\n", out[1]);
    std::printf("F_2(30)    = %.17g\n", out[2]);

    // F is positive and falls off with x; a batch of zeros or a negative
    // value would mean the call did not do what it says.
    const bool sane = out[0] > 0.0 && out[0] > out[1] && out[1] > out[2] && ladder[0] > ladder[6];
    if (!sane) {
        std::printf("FAIL: F is not positive and decreasing in x\n");
        return 1;
    }
    return 0;
}
