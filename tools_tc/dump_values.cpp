// Cross-check driver: the shipped library's region-A values on a grid, at full
// precision. Reads x values from argv[1] (one per line).
//
// Two jobs: (1) it pins the harness - a Python reimplementation of the shipped
// recursion must reproduce these values bit for bit, and where it cannot (the
// shipped expx is a float expf, which no Python expression reproduces) these
// values are the measurement instead; (2) it gives the recursion's own
// rounding, which is the term a mode's seed has to be added to.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "boys/boys.hpp"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: dump_values <xfile>\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "r");
    if (f == nullptr)
    {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::vector<double> xs;
    double v = 0.0;
    while (std::fscanf(f, "%lf", &v) == 1)
    {
        xs.push_back(v);
    }
    std::fclose(f);

    const int nmaxes[] = {0, 1, 2, 4, 8, 16, 32};
    for (int nmax : nmaxes)
    {
        for (double x : xs)
        {
            double d[33];
            float h[33];
            boys::BoysAllOrders(nmax, x, d);
            boys::BoysAllOrdersF32(nmax, static_cast<float>(x), h);
            for (int n = 0; n <= nmax; ++n)
            {
                std::printf("D %d %.17g %d %.17g\n", nmax, x, n, d[n]);
            }
            for (int n = 0; n <= nmax; ++n)
            {
                std::printf("F %d %.17g %d %.17g\n", nmax, x, n, static_cast<double>(h[n]));
            }
        }
    }
    return 0;
}
