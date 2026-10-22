// First Paper A sweep: the recursed entry against the direct entry on every
// argument of the designed grid, the recurrence replica's admission, and the
// gain the shipped recurrence really pays at every cell it recurses.
//
// Both evaluations are the library's public entries.  The recursed one names
// the policy the paper describes (common/replica.hpp); the direct one is the
// region-A transform in its fp64 mode, called once per band so that the batch
// is the matrix product the paper names.
//
// Outputs:
//   cells.csv      band,x,n,recursed,direct
//   admission.csv  band,x,direction,checked,matched
//   gain.csv       case,band,n,x,gain
//
// The gain cases are `up` (the shipped upward path at every gated cell),
// `down` (the shipped downward path), and `up_ungated` (the upward ladder
// driven past the order the shipped entry stops it, at the first upward
// argument of each band).

#include "../common/replica.hpp"

#include "boys/boys_transform.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

using paper_a::kNmax;
using paper_a::PaperPolicy;

struct Row {
    std::string band;
    double x;
};

struct Band {
    const char* name;
    boys::RegionABand region;
    std::vector<Row> rows;
};

std::string Argument(int argc, char** argv, const char* name, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

int IntegerArgument(int argc, char** argv, const char* name, int fallback) {
    const std::string text = Argument(argc, argv, name, "");
    return text.empty() ? fallback : std::atoi(text.c_str());
}

std::vector<Row> ReadGrid(const std::string& path) {
    std::FILE* file = std::fopen(path.c_str(), "r");

    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }

    std::vector<Row> rows;
    char line[512];
    std::fgets(line, sizeof line, file); // header

    while (std::fgets(line, sizeof line, file)) {
        const std::string text(line);
        const std::size_t first = text.find(',');
        const std::size_t second = text.find(',', first + 1);

        if (first == std::string::npos || second == std::string::npos) {
            continue;
        }

        rows.push_back({text.substr(0, first), std::stod(text.substr(first + 1, second - first - 1))});
    }

    std::fclose(file);
    return rows;
}

std::FILE* Open(const std::string& path, const char* header) {
    std::FILE* file = std::fopen(path.c_str(), "w");

    if (!file) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(2);
    }

    std::fputs(header, file);
    return file;
}

} // namespace

int main(int argc, char** argv) {
    const std::string grid_path = Argument(argc, argv, "--grid", "data/grid.csv");
    const std::string cells_path = Argument(argc, argv, "--cells", "results/cells.csv");
    const std::string admission_path = Argument(argc, argv, "--admission", "results/admission.csv");
    const std::string gain_path = Argument(argc, argv, "--gain", "results/replica_gain.csv");
    const std::string batch = Argument(argc, argv, "--batch", "full");
    const int limit = IntegerArgument(argc, argv, "--max-arguments", 0);

    std::vector<Row> rows = ReadGrid(grid_path);

    if (limit > 0 && static_cast<int>(rows.size()) > limit) {
        rows.resize(static_cast<std::size_t>(limit));
    }

    std::array<Band, 2> bands = {{
        {"A1", boys::RegionABand::kA1, {}},
        {"A2", boys::RegionABand::kA2, {}},
    }};

    for (const Row& row : rows) {
        for (Band& band : bands) {
            if (row.band == band.name) {
                band.rows.push_back(row);
            }
        }
    }

    std::FILE* cells = Open(cells_path, "band,x,n,recursed,direct\n");
    std::FILE* admission = Open(admission_path, "band,x,direction,checked,matched\n");
    std::FILE* gain = Open(gain_path, "case,band,n,x,gain\n");

    for (Band& band : bands) {
        const std::size_t count = band.rows.size();

        if (count == 0) {
            continue;
        }

        std::vector<double> xs(count);
        for (std::size_t i = 0; i < count; ++i) {
            xs[i] = band.rows[i].x;
        }

        std::vector<double> direct(((kNmax + 1) * count), 0.0);

        if (batch == "single") {
            for (std::size_t i = 0; i < count; ++i) {
                boys::BoysRegionAProduct<boys::ProductMode::kFp64>(
                    band.region, kNmax, &xs[i], &direct[static_cast<std::size_t>(kNmax + 1) * i], 1);
            }
        } else {
            boys::BoysRegionAProduct<boys::ProductMode::kFp64>(
                band.region, kNmax, xs.data(), direct.data(), count);
        }

        bool ungated_done = false;

        for (std::size_t i = 0; i < count; ++i) {
            const double x = xs[i];
            std::array<double, kNmax + 1> shipped{};
            boys::BoysAllOrders<PaperPolicy>(kNmax, x, shipped.data());

            const paper_a::Admission admitted = paper_a::AdmitReplica(x, shipped.data());
            std::fprintf(admission, "%s,%.17g,%s,%d,%d\n", band.name, x, admitted.direction,
                         admitted.checked, admitted.matched);

            for (int n = 0; n <= kNmax; ++n) {
                const double value =
                    batch == "single"
                        ? direct[static_cast<std::size_t>(kNmax + 1) * i + static_cast<std::size_t>(n)]
                        : direct[static_cast<std::size_t>(n) * count + i];
                std::fprintf(cells, "%s,%.17g,%d,%.17g,%.17g\n", band.name, x, n, shipped[n], value);
            }

            if (x == 0.0) {
                continue;
            }

            using Fit = typename PaperPolicy::Fit;
            const double half_exp = 0.5 * std::exp(-x);

            if (x >= Fit::kRegionAFitsFrom) {
                int served = 0;

                while (served < kNmax && x >= boys::detail::kTierThresholds[served + 1]) {
                    ++served;
                }

                typename Fit::template BandSource<PaperPolicy::kDivision> source(x);
                double f = source.Next(0, x);
                // A relative seed perturbation far enough above the recurrence's
                // rounding to be measurable.  The homogeneous map keeps the
                // relative size of the perturbation at about 2^-20 of every
                // order's value, where a one-ULP seed rides at the arithmetic's
                // own noise floor and the measured ratio is rounding, not gain.
                double p = f + std::ldexp(f, -20);
                const double delta = p - f;
                const double reciprocal = 1.0 / x;

                for (int l = 1; l <= served; ++l) {
                    f = paper_a::UpwardStep(l, f, x, reciprocal, half_exp);
                    p = paper_a::UpwardStep(l, p, x, reciprocal, half_exp);
                    std::fprintf(gain, "up,%s,%d,%.17g,%.17g\n", band.name, l, x, (p - f) / delta);
                }

                if (!ungated_done) {
                    double base = source.Next(0, x);
                    double pert = base + std::ldexp(base, -20);
                    const double seed_delta = pert - base;

                    for (int l = 1; l <= kNmax; ++l) {
                        base = paper_a::UpwardStep(l, base, x, reciprocal, half_exp);
                        pert = paper_a::UpwardStep(l, pert, x, reciprocal, half_exp);
                    }

                    std::fprintf(gain, "up_ungated,%s,%d,%.17g,%.17g\n", band.name, kNmax, x,
                                 (pert - base) / seed_delta);
                    ungated_done = true;
                }
            } else {
                double f = boys::detail::PolicyRegionAValue<PaperPolicy>(kNmax, x);
                // The same resolvable relative perturbation as the upward ladder.
                double p = f + std::ldexp(f, -20);
                const double delta = p - f;

                for (int l = kNmax - 1; l >= 0; --l) {
                    f = paper_a::DownwardStep(l, f, x, half_exp);
                    p = paper_a::DownwardStep(l, p, x, half_exp);
                    std::fprintf(gain, "down,%s,%d,%.17g,%.17g\n", band.name, l, x, (p - f) / delta);
                }
            }
        }
    }

    std::fclose(cells);
    std::fclose(admission);
    std::fclose(gain);
    return 0;
}
