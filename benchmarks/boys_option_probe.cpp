// The option probe driver: measures this build's evaluation options on this machine and prints what
// it found, verdict included. Exit 0 is a result either way: a refusal is one of them, so a script
// reads the verdict from the text. A measured run exits 1 when its option space does not close (an
// accounting defect, not a result), 3 on a failed --emit-defaults write, 2 otherwise.
#include "boys/boys_probe.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

// CMake passes the revision the report was taken at; a build that bypasses it says so rather
// than naming a revision it did not read.
#ifndef BoysProbeRevision
#define BoysProbeRevision "unknown"
#endif

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

// Appends a comma-separated list of option names to the caller's set.
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
               "  --refine-runs=N    runs the refinement stage takes over the\n"
               "                     options a class left tied. The vote says\n"
               "                     whether the row the figures put first holds\n"
               "                     up over more rounds or whether the two\n"
               "                     cannot be separated\n"
               "                     (default 5)\n"
               "  --refine-factor=N  how much longer each refinement run is than\n"
               "                     one pass protocol. The stage refines a tie by\n"
               "                     asking whether the leader holds up over more\n"
               "                     rounds of the same comparison, so the factor\n"
               "                     lengthens the ROUNDS and leaves the passes\n"
               "                     alone (default 1)\n"
               "  --canary-spread=P  spread percentage of the canary's own runs\n"
               "                     across a pass above which the pass is\n"
               "                     flagged as one that ran on a wandering\n"
               "                     machine. A diagnostic: it discards nothing\n"
               "                     and is reported beside every pass\n"
               "                     (default 5.0)\n"
               "  --reference=NAME   the entry every ratio of this run is formed\n"
               "                     against, by name (default: the library's own\n"
               "                     default double-precision all-orders entry). The\n"
               "                     anchor is a choice of the reporting and not of the\n"
               "                     measurement, so two runs of one binary under two\n"
               "                     anchors measure the same thing; a name this run did\n"
               "                     not measure falls back to the default and the\n"
               "                     report says so.\n"
               "  --only=A,B         measure only these options, by name, repeatable\n"
               "                     and comma-separated (default: every option this\n"
               "                     build offers). A name that is no option of this\n"
               "                     library, a cell the library refuses, and an option\n"
               "                     this build does not carry are answered apart.\n"
               "  --emit-defaults=F  write this run's own rankings as a replacement for\n"
               "                     the build-defaults seam, in the format its\n"
               "                     BOYS_BUILD_DEFAULT_ROWS consumes, and report which\n"
               "                     classes carry a measured row and which carry the\n"
               "                     seam's own names. The file is what the CMake option\n"
               "                     BOYS_BUILD_DEFAULTS points a build at; a run that\n"
               "                     measured no class writes no file.\n"
               "  --help             this text\n"
               "\n"
               "The result is about this machine, this build and this process. It is\n"
               "printed with the report rather than left to the reader to remember.\n",
               stdout);
}

} // namespace

int main(int argc, char** argv) {
    boys::ProbeOptions options;
    std::string emitDefaults;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--help")
        {
            Usage();
            return 0;
        } else if (arg.rfind("--emit-defaults=", 0) == 0)
        {
            emitDefaults = arg.c_str() + 16;
        } else if (arg == "--emit-defaults")
        {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "boys-option-probe: --emit-defaults needs a file name\n");
                return 2;
            }

            emitDefaults = argv[++i];
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
        } else if (arg.rfind("--refine-runs=", 0) == 0)
        {
            options.refinementRuns = std::atoi(arg.c_str() + 14);
        } else if (arg.rfind("--refine-factor=", 0) == 0)
        {
            options.refinementFactor = std::atoi(arg.c_str() + 16);
        } else if (arg.rfind("--canary-spread=", 0) == 0)
        {
            options.canarySpreadAlarm = std::strtod(arg.c_str() + 16, nullptr);
        } else if (arg.rfind("--reference=", 0) == 0)
        {
            options.reference = arg.c_str() + 12;
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

    // The revision follows the timestamp the line already opened with, so a reader keyed on that
    // stamp reads what it read before, and the report dates itself for a check to hold it to.
    std::printf("boys option probe | started %s | finished %s | seed %llu | revision %s | "
                "verdict %s\n",
                started.c_str(), Timestamp().c_str(),
                static_cast<unsigned long long>(report.options.seed), BoysProbeRevision,
                report.verdict == boys::OptionProbeVerdict::kRecommend ? "RECOMMEND"
                                                                       : "CANNOT DETERMINE");

    // The default and how it was reached, on one line a script can read: a
    // majority winner and a pick among equals are different answers.
    std::printf("default %s | reached by %s\n", report.recommended.c_str(),
                boys::OptionProbeDefaultHowName(report.defaultHow).c_str());

    const std::string text = boys::FormatOptionProbe(report);
    std::fputs(text.c_str(), stdout);

    // The seam this run implies, where the caller asked for it: the rows a build pointed at
    // the file compiles. A run that ranked no class has nothing to write and the entry says
    // so with an empty text, which is a failure of this command's own contract rather than a
    // result - the file would be a transcription of the seam and not a measurement.
    if (!emitDefaults.empty())
    {
        const std::string headerText = boys::FormatBuildDefaults(report, started);

        if (headerText.empty())
        {
            std::fprintf(stderr,
                         "boys-option-probe: no defaults written - this run measured no class, "
                         "so there is no ranking to write\n");
            return 3;
        }

        std::ofstream file(emitDefaults, std::ios::binary);

        if (!file)
        {
            std::fprintf(stderr, "boys-option-probe: cannot write '%s'\n", emitDefaults.c_str());
            return 3;
        }

        file << headerText;
        file.close();

        if (!file)
        {
            std::fprintf(stderr, "boys-option-probe: short write to '%s'\n", emitDefaults.c_str());
            return 3;
        }

        std::printf("defaults written: %s | %zu byte(s) | point a build at it with "
                    "-DBOYS_BUILD_DEFAULTS=%s\n",
                    emitDefaults.c_str(), headerText.size(), emitDefaults.c_str());
    }

    return boys::OptionProbeSpaceClosure(report).closed ? 0 : 1;
}
