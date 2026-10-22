// Second Paper A sweep: vary the seed-fit degree and the order at which the
// recurrence starts, independently, and record every path value together
// with an exactly known seed perturbation.
//
// For each argument, each seed order and each degree on the ladder:
//   * evaluate the seed as the double Clenshaw of the rounded fit;
//   * perturb it one ULP upward and run the same recurrence;
//   * emit every value of the base path, the perturbation's increment, and
//     the exact transfer (increment_n / increment_seed).
// That transfer is a property of the recurrence alone and needs no
// reference; the reference pivot is applied in analyze.py to the base path.
//
// The shipped and direct entries are evaluated as in the first sweep, so the
// second sweep's own band-level comparison and the model's prediction can be
// computed from the same cells.  The replica admission is repeated here.
//
// x = 0 (present in the committed gate grid) has no finite upward
// amplification; only its seed rows are emitted, tagged `seed`.
//
// Outputs:
//   paths.csv      route,band,x,start,degree,n,value,pert_delta,seed_delta,transfer_ulp
//   cells.csv      band,x,n,recursed,direct
//   admission.csv  band,x,direction,checked,matched

#include "../common/replica.hpp"

#include "boys/boys_transform.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {

using paper_a::kNmax;
using paper_a::PaperPolicy;

constexpr std::array<int, 8> kSeedOrders = {0, 2, 4, 8, 16, 24, 28, 32};
constexpr std::array<int, 14> kDegrees = {0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
constexpr std::array<int, 6> kUpwardStarts = {0, 2, 4, 8, 16, 24};
constexpr std::array<int, 5> kDownwardStarts = {32, 28, 24, 16, 8};

struct Row {
    std::string band;
    double x;
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

bool Contains(const int* values, std::size_t count, int value) {
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] == value) {
            return true;
        }
    }
    return false;
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

using FitKey = std::tuple<int, int, int>; // band index, seed order, degree

std::map<FitKey, std::vector<double>> ReadFits(const std::string& path) {
    std::FILE* file = std::fopen(path.c_str(), "r");

    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }

    std::map<FitKey, std::vector<double>> fits;
    char line[512];
    std::fgets(line, sizeof line, file); // header

    while (std::fgets(line, sizeof line, file)) {
        char band[16];
        int order = 0;
        int degree = 0;
        int index = 0;
        double value = 0.0;

        if (std::sscanf(line, "%15[^,],%d,%d,%d,%lf", band, &order, &degree, &index, &value) != 5) {
            continue;
        }

        const int band_index = std::string(band) == "A1" ? 0 : 1;
        std::vector<double>& coefficients = fits[{band_index, order, degree}];

        if (static_cast<int>(coefficients.size()) <= index) {
            coefficients.resize(static_cast<std::size_t>(index) + 1, 0.0);
        }

        coefficients[static_cast<std::size_t>(index)] = value;
    }

    std::fclose(file);
    return fits;
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

void Emit(std::FILE* paths, const char* route, const char* band, double x, int start, int degree,
          int n, double value, double pert_delta, double seed_delta) {
    std::fprintf(paths, "%s,%s,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g\n", route, band, x, start,
                 degree, n, value, pert_delta, seed_delta,
                 seed_delta != 0.0 ? pert_delta / seed_delta : 0.0);
}

} // namespace

int main(int argc, char** argv) {
    const std::string grid_path = Argument(argc, argv, "--grid", "data/grid.csv");
    const std::string fits_path = Argument(argc, argv, "--fits", "data/fits.csv");
    const std::string paths_path = Argument(argc, argv, "--paths", "results/paths.csv");
    const std::string cells_path = Argument(argc, argv, "--cells", "results/cells.csv");
    const std::string admission_path = Argument(argc, argv, "--admission", "results/admission.csv");
    const int limit = IntegerArgument(argc, argv, "--max-arguments", 0);

    std::vector<Row> rows = ReadGrid(grid_path);

    if (limit > 0 && static_cast<int>(rows.size()) > limit) {
        rows.resize(static_cast<std::size_t>(limit));
    }

    const std::map<FitKey, std::vector<double>> fits = ReadFits(fits_path);

    struct Band {
        const char* name;
        boys::RegionABand region;
        double left;
        double right;
        std::vector<Row> rows;
    };

    std::array<Band, 2> bands = {{
        {"A1", boys::RegionABand::kA1, 0.0, boys::kRegionA1Edge, {}},
        {"A2", boys::RegionABand::kA2, boys::kRegionA1Edge, boys::kRegionAEnd, {}},
    }};

    for (const Row& row : rows) {
        for (Band& band : bands) {
            if (row.band == band.name) {
                band.rows.push_back(row);
            }
        }
    }

    std::FILE* paths = Open(paths_path, "route,band,x,start,degree,n,value,pert_delta,seed_delta,transfer_ulp\n");
    std::FILE* cells = Open(cells_path, "band,x,n,recursed,direct\n");
    std::FILE* admission = Open(admission_path, "band,x,direction,checked,matched\n");

    for (std::size_t band_index = 0; band_index < bands.size(); ++band_index) {
        Band& band = bands[band_index];
        const std::size_t count = band.rows.size();

        if (count == 0) {
            continue;
        }

        std::vector<double> xs(count);
        for (std::size_t i = 0; i < count; ++i) {
            xs[i] = band.rows[i].x;
        }

        std::vector<double> direct(((kNmax + 1) * count), 0.0);
        boys::BoysRegionAProduct<boys::ProductMode::kFp64>(
            band.region, kNmax, xs.data(), direct.data(), count);

        for (std::size_t i = 0; i < count; ++i) {
            const double x = xs[i];
            std::array<double, kNmax + 1> shipped{};
            boys::BoysAllOrders<PaperPolicy>(kNmax, x, shipped.data());

            const paper_a::Admission admitted = paper_a::AdmitReplica(x, shipped.data());
            std::fprintf(admission, "%s,%.17g,%s,%d,%d\n", band.name, x, admitted.direction,
                         admitted.checked, admitted.matched);

            for (int n = 0; n <= kNmax; ++n) {
                std::fprintf(cells, "%s,%.17g,%d,%.17g,%.17g\n", band.name, x, n, shipped[n],
                             direct[static_cast<std::size_t>(n) * count + i]);
            }

            const double half_exp = 0.5 * std::exp(-x);
            const double reciprocal = x != 0.0 ? 1.0 / x : 0.0;

            for (int start : kSeedOrders) {
                for (int degree : kDegrees) {
                    const auto key = FitKey{static_cast<int>(band_index), start, degree};
                    const auto found = fits.find(key);

                    if (found == fits.end()) {
                        std::fprintf(stderr, "missing fit: %s order %d degree %d\n", band.name, start,
                                     degree);
                        std::exit(2);
                    }

                    const std::vector<double>& coefficients = found->second;

                    if (static_cast<int>(coefficients.size()) != degree + 1) {
                        std::fprintf(stderr, "fit size mismatch: %s order %d degree %d\n", band.name,
                                     start, degree);
                        std::exit(2);
                    }

                    const double seed =
                        paper_a::SeedFit(coefficients.data(), degree, x, band.left, band.right);

                    if (!std::isfinite(seed)) {
                        continue;
                    }

                    const double perturbed = std::nextafter(seed, std::numeric_limits<double>::infinity());
                    const double seed_delta = perturbed - seed;

                    if (x == 0.0) {
                        Emit(paths, "seed", band.name, x, start, degree, start, seed, seed_delta,
                             seed_delta);
                        continue;
                    }

                    if (Contains(kUpwardStarts.data(), kUpwardStarts.size(), start)) {
                        double base = seed;
                        double pert = perturbed;
                        Emit(paths, "up", band.name, x, start, degree, start, base, seed_delta,
                             seed_delta);

                        for (int n = start + 1; n <= kNmax; ++n) {
                            base = paper_a::UpwardStep(n, base, x, reciprocal, half_exp);
                            pert = paper_a::UpwardStep(n, pert, x, reciprocal, half_exp);
                            Emit(paths, "up", band.name, x, start, degree, n, base, pert - base,
                                 seed_delta);
                        }
                    }

                    if (Contains(kDownwardStarts.data(), kDownwardStarts.size(), start)) {
                        double base = seed;
                        double pert = perturbed;
                        Emit(paths, "down", band.name, x, start, degree, start, base, seed_delta,
                             seed_delta);

                        for (int n = start - 1; n >= 0; --n) {
                            base = paper_a::DownwardStep(n, base, x, half_exp);
                            pert = paper_a::DownwardStep(n, pert, x, half_exp);
                            Emit(paths, "down", band.name, x, start, degree, n, base, pert - base,
                                 seed_delta);
                        }
                    }
                }
            }
        }
    }

    std::fclose(paths);
    std::fclose(cells);
    std::fclose(admission);
    return 0;
}
