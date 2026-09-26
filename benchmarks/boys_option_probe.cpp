// The option probe driver: measures this build's evaluation options on this
// machine and prints what it found, including whether it found enough to name
// one.
//
// One process, one run: the probe warms up, calibrates its load instrument,
// takes the passes, and reports. The exit status is 0 whether or not a winner
// was named — a refusal is one of this tool's results, not a failure of it —
// so a script that wants the verdict reads it from the text.
//
// The knobs move the workload to the caller's own shape (--count, --nmax,
// --xrange), the protocol to the machine's own patience (--passes, --rounds,
// --bg, --cal, --canary-spread), and the measured set to the caller's own
// shortlist (--only, repeatable). Every value that was in force is printed in
// the report, so a figure is never read without the protocol that produced it,
// and a name that is no option of this library is printed as such rather than
// silently measuring nothing.
#include "boys/boys_probe.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

namespace {

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

// A comma-separated list of option names, appended to the set the caller is
// narrowing the measurement to.
void AppendNames(const char* list, std::vector<std::string>& names) {
    const std::string text(list);

    for (std::size_t start = 0; start <= text.size();)
    {
        const std::size_t comma = text.find(',', start);
        const std::size_t end = comma == std::string::npos ? text.size() : comma;

        if (end > start)
        {
            names.push_back(text.substr(start, end - start));
        }

        if (comma == std::string::npos)
        {
            break;
        }

        start = comma + 1;
    }
}

void Usage() {
    std::fputs("boys option probe — measures this build's evaluation options on this machine\n"
               "\n"
               "usage: boys-option-probe [options]\n"
               "\n"
               "  --count=N          arguments per call (default 16384)\n"
               "  --nmax=N           highest order any argument carries (default 32)\n"
               "  --xrange=LO,HI     log-uniform argument range (default 1e-3,40)\n"
               "  --seed=N           workload generator seed (default 47)\n"
               "  --passes=N         timed passes (default 5)\n"
               "  --rounds=N         rounds per pass (default 5)\n"
               "  --bg=SECONDS       background load window (default 1.0)\n"
               "  --cal=SECONDS      load calibration window (default 0.5)\n"
               "  --canary-spread=P  spread percentage of the canary's own runs\n"
               "                     across a pass above which the pass is\n"
               "                     discarded (default 5.0)\n"
               "  --only=A,B         measure only these options, by name, repeatable\n"
               "                     and comma-separated (default: every option this\n"
               "                     build offers). A name that is no option of this\n"
               "                     library, a cell the library refuses, and an option\n"
               "                     this build does not carry are answered apart.\n"
               "  --help             this text\n"
               "\n"
               "The result is about this machine, this build and this process. It is\n"
               "printed with the report rather than left to the reader to remember.\n",
               stdout);
}

} // namespace

int main(int argc, char** argv) {
    boys::ProbeOptions options;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--help")
        {
            Usage();
            return 0;
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
        } else if (arg.rfind("--bg=", 0) == 0)
        {
            options.backgroundWindowSeconds = std::strtod(arg.c_str() + 5, nullptr);
        } else if (arg.rfind("--cal=", 0) == 0)
        {
            options.calibrationSeconds = std::strtod(arg.c_str() + 6, nullptr);
        } else if (arg.rfind("--canary-spread=", 0) == 0)
        {
            options.canarySpreadThreshold = std::strtod(arg.c_str() + 16, nullptr);
        } else if (arg.rfind("--only=", 0) == 0)
        {
            AppendNames(arg.c_str() + 7, options.only);
        } else if (arg.rfind("--xrange=", 0) == 0)
        {
            options.xLo = std::strtod(arg.c_str() + 9, nullptr);
            const char* comma = std::strchr(arg.c_str() + 9, ',');
            options.xHi = (comma != nullptr) ? std::strtod(comma + 1, nullptr) : options.xLo;
        } else
        {
            std::fprintf(stderr, "boys-option-probe: unknown argument '%s' (try --help)\n",
                         arg.c_str());
            return 2;
        }
    }

    const std::string started = Timestamp();
    const boys::OptionProbeReport report = boys::RunOptionProbe(options);

    std::printf("boys option probe | started %s | finished %s | seed %llu | verdict %s\n",
                started.c_str(), Timestamp().c_str(),
                static_cast<unsigned long long>(report.options.seed),
                report.verdict == boys::OptionProbeVerdict::kRecommend ? "RECOMMEND"
                                                                       : "CANNOT DETERMINE");

    const std::string text = boys::FormatOptionProbe(report);
    std::fputs(text.c_str(), stdout);
    return 0;
}
