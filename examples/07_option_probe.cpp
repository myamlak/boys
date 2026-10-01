// Which of the options this build offers is fastest on this machine?
//
//   c++ -std=c++20 -I include examples/07_option_probe.cpp -L build -lboys -o probe && ./probe
//
// The answer is a property of the machine and the build flags, so it cannot be
// read off a table - it has to be measured where it will run. This is that
// measurement, on a protocol short enough to run here; the command-line tool
// boys-option-probe runs the full one and prints the whole report.
#include <boys/boys_span.hpp>

#include <cstddef>
#include <cstdio>
#include <string>

int main()
{
    boys::ProbeOptions options;
    options.count = 1u << 10;
    options.nmax = 8;
    options.passes = 4;
    options.rounds = 4;
    options.calibrationSeconds = 0.0;
    options.backgroundWindowSeconds = 0.0;

    const boys::OptionProbeReport report = boys::RunOptionProbe(options);

    std::printf("measured %zu options on %d logical processors\n", report.measurements.size(),
                report.logicalProcessors);

    // The group a new caller lives in: full accuracy, double precision, the
    // shape that hands back one argument's whole ladder.
    for (const boys::OptionProbeClass& group : report.classes)
    {
        if (group.precision != boys::OptionPrecision::kFp64 ||
            group.tier != boys::AccuracyTier::kReference ||
            group.shape != boys::OptionProbeShape::kAllOrders)
        {
            continue;
        }
        std::printf("\n%s - %zu options ranked\n", group.name.c_str(), group.ranked.size());
        std::printf("  fastest: %s at %.2f ns/argument\n", group.leader.c_str(),
                    group.leaderNsPerArgument);
        std::printf("  the run could not separate %zu of them from the leader\n",
                    group.unplaced.size());
        for (const std::string& name : group.unplaced)
        {
            std::printf("    %s\n", name.c_str());
        }
    }

    std::printf("\nrecommended here: %s\n",
                report.hasDefault ? report.recommended.c_str() : "CANNOT DETERMINE");
    if (!report.hasDefault)
    {
        std::printf("reason: %s\n", report.reason.c_str());
    }
    return report.measurements.empty() ? 1 : 0;
}
