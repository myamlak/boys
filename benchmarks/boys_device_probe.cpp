// The device option probe's driver: it parses arguments, calls the library
// entry, and prints what came back.
//
// There is no measurement in this file. boys::RunDeviceOptionProbe is the
// surface — it takes the device selector and returns the data — and this tool
// exists so that a person can get the same report at a terminal without writing
// a program first. Every number it prints came out of that call, including the
// card's own name.
//
// The exit status is 0 whether or not a winner was named: a refusal is one of
// this probe's results, not a failure of it. A status that is not kSuccess is
// reported and exits 1, because nothing was measured at all.
#include "boys/boys_cuda_probe.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void Usage() {
    std::fputs("boys device probe — which CUDA entry is cheapest on the card this runs on\n"
               "\n"
               "usage: boys-device-probe [options]\n"
               "\n"
               "  --device=N         ordinal of the device to measure (default 0)\n"
               "  --count=N          arguments per call (default 262144)\n"
               "  --nmax=N           highest order any argument carries, 1..32\n"
               "                     (default 32)\n"
               "  --xrange=LO,HI     log-uniform argument range (default 1e-3,40)\n"
               "  --seed=N           workload generator seed (default 47)\n"
               "  --passes=N         timed passes (default 3)\n"
               "  --rounds=N         rounds per pass (default 4)\n"
               "  --reps=N           launches inside one timed region (default 32)\n"
               "  --only=A,B,C       measure only these entries, named as the report\n"
               "                     prints them (default: every entry this build\n"
               "                     offers)\n"
               "  --help             this text\n"
               "\n"
               "Transfer and host submission are outside the timed region by design: the\n"
               "buffers are uploaded once and the launches are amortised over --reps. The\n"
               "report says what it could and could not separate on this card, and names\n"
               "the card.\n",
               stdout);
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
            Usage();
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
            std::fprintf(stderr, "boys-device-probe: unknown argument '%s' (try --help)\n",
                         arg.c_str());
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

    return report.status == boys::DeviceProbeStatus::kSuccess ? 0 : 1;
}
