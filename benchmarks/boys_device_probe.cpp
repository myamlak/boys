// The device option probe's driver. There is no measurement in this file —
// every number printed came out of the library call, including the card's own
// name. The exit status is 0 whether or not a winner was named; a status that
// is not kSuccess, and an option space whose closure does not close, are
// reported and exit 1. The closure is the report's last block and the verdict
// it prints is the one this program returns: a member of the space in no state
// is a failure here and not a paragraph.
#include "boys/boys_cuda_probe.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

namespace {

/// The moment a run started, in UTC, the way the option probe writes it too: a written file
/// belongs to the host and the run that produced it, and this is the run.
std::string Timestamp() {
    const std::time_t now = std::time(nullptr);
    std::array<char, 32> buffer{};
    std::tm utc{};

#ifdef _MSC_VER
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif

    std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return std::string(buffer.data());
}

/// The card and the moment, as the file a run writes states them: a device figure belongs to
/// the card it was taken on, so a reader holding the file has to be able to tell which one.
std::string TakenAt(const boys::DeviceProbeReport& report, const std::string& started) {
    if (report.device.name.empty())
    {
        return started;
    }

    return report.device.name + ", " + started;
}

/// A double as the help prints it: enough to name the value, no more.
std::string Number(double value) {
    char buffer[32];

    std::snprintf(buffer, sizeof buffer, "%g", value);

    return buffer;
}

/// The usage text. Every default in it is read from the options a run starts
/// from, so the text cannot name a default the program does not have.
void Usage(const boys::DeviceProbeOptions& defaults) {
    const std::string device = std::to_string(defaults.device);
    const std::string count = std::to_string(defaults.count);
    const std::string nmax = std::to_string(defaults.nmax);
    const std::string xLo = Number(defaults.xLo);
    const std::string xHi = Number(defaults.xHi);
    const std::string seed = std::to_string(defaults.seed);
    const std::string passes = std::to_string(defaults.passes);
    const std::string rounds = std::to_string(defaults.rounds);
    const std::string reps = std::to_string(defaults.repetitions);
    const std::string pairFactor = std::to_string(defaults.countPairFactor);
    const std::string refineRuns = std::to_string(defaults.refinementRuns);
    const std::string refineFactor = std::to_string(defaults.refinementFactor);

    std::printf("boys device probe — which CUDA entry is cheapest on the card this runs on\n"
                "\n"
                "usage: boys-device-probe [options]\n"
                "\n"
                "  --device=N         ordinal of the device to measure (default %s)\n"
                "  --count=N          arguments per call (default %s)\n"
                "  --nmax=N           highest order any argument carries, 1..32\n"
                "                     (default %s)\n"
                "  --xrange=LO,HI     log-uniform argument range (default %s,%s)\n"
                "  --seed=N           workload generator seed (default %s)\n"
                "  --passes=N         timed passes (default %s)\n"
                "  --rounds=N         rounds per pass (default %s)\n"
                "  --reps=N           launches inside one timed region (default %s)\n"
                "  --pair-factor=N    the second count a row is read at, as a multiple of\n"
                "                     --count; a row's figure is the count-independent cost\n"
                "                     its two readings extrapolate to (default %s)\n"
                "  --refine-runs=N    runs the refinement stage takes of a shape whose\n"
                "                     entries its own rounds could not separate (default %s)\n"
                "  --refine-factor=N  how much longer each refinement run is than the main\n"
                "                     protocol (default %s)\n"
                "  --only=A,B,C       measure only these rows, named as the report\n"
                "                     prints them: an entry's name is its row at the\n"
                "                     build's default division form, and a row at another\n"
                "                     form is that name with the form's segment (default:\n"
                "                     every row this build offers, at every form)\n"
                "  --emit-defaults=F  write this run's own device classes as a replacement\n"
                "                     for the build-defaults seam, in the format its\n"
                "                     BOYS_BUILD_DEFAULT_ROWS consumes, and report which\n"
                "                     classes carry a measured row and which carry none and\n"
                "                     why. The file is what the CMake option\n"
                "                     BOYS_BUILD_DEFAULTS points a build at; a run that\n"
                "                     measured no device class writes no file.\n"
                "  --help             this text\n"
                "\n"
                "Transfer and host submission are outside the timed region by design: the\n"
                "buffers are uploaded once and the launches are amortised over --reps. Every\n"
                "row is read at --count and at --count times --pair-factor arguments, and its\n"
                "figure is the count-independent cost the two readings extrapolate to; both\n"
                "readings are printed beside it. The report says what it could and could not\n"
                "separate on this card, names the card, and names one entry per shape: where a\n"
                "shape's own rounds could not order its entries, or its rows were set aside by\n"
                "the run's own checks, the ones left go to the refinement stage together, the\n"
                "entry that led the most of its runs is the shape's entry, and the report says\n"
                "how the name was reached — never an entry no clock produced.\n",
                device.c_str(),
                count.c_str(),
                nmax.c_str(),
                xLo.c_str(),
                xHi.c_str(),
                seed.c_str(),
                passes.c_str(),
                rounds.c_str(),
                reps.c_str(),
                pairFactor.c_str(),
                refineRuns.c_str(),
                refineFactor.c_str());
}

/// A comma-separated list, split into the names the probe takes.
std::vector<std::string> SplitNames(const char* text) {
    std::vector<std::string> names;
    std::string current;

    for (const char* p = text; *p != '\0'; ++p)
    {
        if (*p == ',')
        {
            if (!current.empty())
            {
                names.push_back(current);
                current.clear();
            }
        } else
        {
            current.push_back(*p);
        }
    }

    if (!current.empty())
    {
        names.push_back(current);
    }

    return names;
}

} // namespace

int main(int argc, char** argv) {
    boys::DeviceProbeOptions options;
    std::string emitDefaults;
    const std::string started = Timestamp();

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--help")
        {
            Usage(options);
            return 0;
        } else if (arg.rfind("--emit-defaults=", 0) == 0)
        {
            emitDefaults = arg.c_str() + 16;
        } else if (arg == "--emit-defaults")
        {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "boys-device-probe: --emit-defaults needs a file name\n");
                return 2;
            }

            emitDefaults = argv[++i];
        } else if (arg.rfind("--device=", 0) == 0)
        {
            options.device = std::atoi(arg.c_str() + 9);
        } else if (arg.rfind("--count=", 0) == 0)
        {
            options.count = static_cast<std::size_t>(std::strtoull(arg.c_str() + 8, nullptr, 10));
        } else if (arg.rfind("--nmax=", 0) == 0)
        {
            options.nmax = std::atoi(arg.c_str() + 7);
        } else if (arg.rfind("--seed=", 0) == 0)
        {
            options.seed = std::strtoull(arg.c_str() + 7, nullptr, 10);
        } else if (arg.rfind("--passes=", 0) == 0)
        {
            options.passes = std::atoi(arg.c_str() + 9);
        } else if (arg.rfind("--rounds=", 0) == 0)
        {
            options.rounds = std::atoi(arg.c_str() + 9);
        } else if (arg.rfind("--reps=", 0) == 0)
        {
            options.repetitions = std::atoi(arg.c_str() + 7);
        } else if (arg.rfind("--pair-factor=", 0) == 0)
        {
            options.countPairFactor = std::atoi(arg.c_str() + 14);
        } else if (arg.rfind("--refine-runs=", 0) == 0)
        {
            options.refinementRuns = std::atoi(arg.c_str() + 14);
        } else if (arg.rfind("--refine-factor=", 0) == 0)
        {
            options.refinementFactor = std::atoi(arg.c_str() + 16);
        } else if (arg.rfind("--only=", 0) == 0)
        {
            options.only = SplitNames(arg.c_str() + 7);
        } else if (arg.rfind("--xrange=", 0) == 0)
        {
            options.xLo = std::strtod(arg.c_str() + 9, nullptr);
            const char* comma = std::strchr(arg.c_str() + 9, ',');

            if (comma != nullptr)
            {
                options.xHi = std::strtod(comma + 1, nullptr);
            } else
            {
                options.xHi = options.xLo;
            }
        } else
        {
            std::fprintf(
                stderr, "boys-device-probe: unknown argument '%s' (try --help)\n", arg.c_str());
            return 2;
        }
    }

    const boys::DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    std::printf("boys device probe | device %d | %s | count %zu | seed %llu | status %d\n",
                options.device,
                report.device.name.empty() ? "(unnamed)" : report.device.name.c_str(),
                report.workloadCount,
                static_cast<unsigned long long>(report.options.seed),
                static_cast<int>(report.status));

    const std::string text = boys::FormatDeviceOptionProbe(report);
    std::fputs(text.c_str(), stdout);

    // The device half of the seam this run implies, where the caller asked for it: the
    // classes it carries a row for and the ones it refuses, then the file itself. A run
    // that measured no device class has nothing to write and the entry says so with an
    // empty text, which is a failure of this command's own contract rather than a result:
    // the file would be a transcription of the seam and not a measurement.
    if (!emitDefaults.empty())
    {
        const boys::DeviceDefaultsEmission emission =
            boys::FormatDeviceBuildDefaults(report, TakenAt(report, started));

        std::printf("\nthe seam rows this run can write - one per (device, precision, shape) class:\n");

        for (const std::string& line : emission.emitted)
        {
            std::printf("  written: %s\n", line.c_str());
        }

        for (const std::string& line : emission.overridden)
        {
            std::printf("  overrode: %s\n", line.c_str());
        }

        for (const std::string& line : emission.refused)
        {
            std::printf("  no row:  %s\n", line.c_str());
        }

        if (emission.text.empty())
        {
            std::fprintf(stderr,
                         "boys-device-probe: no defaults written - this run measured no device "
                         "class, so there is no ranking to write\n");
            return 3;
        }

        // A std::ofstream and not the C stdio the option probe's writer uses: this target
        // carries the CUDA toolkit's include directories, and there the CRT marks fopen
        // deprecated under the tree's own /WX, so the portable C++ stream is what compiles.
        std::ofstream file(emitDefaults, std::ios::binary | std::ios::trunc);

        if (!file)
        {
            std::fprintf(stderr, "boys-device-probe: cannot write '%s'\n", emitDefaults.c_str());
            return 3;
        }

        file.write(emission.text.data(), static_cast<std::streamsize>(emission.text.size()));
        file.close();

        if (!file)
        {
            std::fprintf(stderr, "boys-device-probe: short write to '%s'\n", emitDefaults.c_str());
            return 3;
        }

        std::printf("defaults written: %s | %zu byte(s) | point a build at it with "
                    "-DBOYS_BUILD_DEFAULTS=%s\n",
                    emitDefaults.c_str(), emission.text.size(), emitDefaults.c_str());
    }

    // The report's own last block, read back as the exit status: the closure holds, or
    // a member of the option space is in no state and this run says so with a non-zero
    // status rather than with a line a script has to parse.
    return boys::DeviceOptionSpaceClosure(report).closed ? 0 : 1;
}
