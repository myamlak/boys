// The device option probe's driver. There is no measurement in this file —
// every number printed came out of the library call, including the card's own
// name. The exit status is 0 whether or not a winner was named; a status that
// is not kSuccess, and an option space whose closure does not close, are
// reported and exit 1. The closure is the report's last block and the verdict
// it prints is the one this program returns: a member of the space in no state
// is a failure here and not a paragraph.
#include "boys/boys_cuda_probe.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

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
                "  --only=A,B,C       measure only these entries, named as the report\n"
                "                     prints them (default: every entry this build\n"
                "                     offers)\n"
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

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--help")
        {
            Usage(options);
            return 0;
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

    // The report's own last block, read back as the exit status: the closure holds, or
    // a member of the option space is in no state and this run says so with a non-zero
    // status rather than with a line a script has to parse.
    return boys::DeviceOptionSpaceClosure(report).closed ? 0 : 1;
}
