#pragma once

// The accuracy gate's instrument, shared by every gate in this tree: the
// committed high-precision reference grid and the one reading of it, the bound
// vocabulary the narrow formats are measured in, and the accumulator and row
// shape a report prints. The CPU gate (tests/boys_accuracy_gate.cpp) and the
// CUDA device gate (tests/boys_cuda_accuracy_gate.cpp) both include this, so a
// lane measured on either side is measured against one reference format and
// printed in one vocabulary rather than two that can drift apart.
//
// One reading is one instrument and not one per book: MeasureAt fills whatever
// accumulator it is handed, MeasureInto names one of a book by index, and the
// three books the CPU gate keeps - the lanes', the evaluation schemes' and the
// fit routes' - are filled by the same comparisons and the same counters, so a
// row added to one of them cannot move another's totals.
//
// The reference itself is tests/data/boys_accuracy_gate_reference.csv,
// re-derivable with tools/gen_boys_accuracy_gate_reference.py. Why it is
// trusted, and what its routes agree to, is the CPU gate's argument and lives
// in its preamble rather than here.

#include "boys/boys.hpp"
#include "boys/f16.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace boys_gate {

// --- the bound the narrow formats are measured in --------------------------

// README and header: the fp16 and bf16 lanes, every region.
constexpr double kBoundHalfBase = 1e-7;

// IEEE-754 binary16: 10 stored mantissa bits, smallest normal exponent -14.
constexpr int kF16MantissaBits = 10;
constexpr int kF16MinNormalExp = -14;

// --- the accumulators, and the one instrument that fills them ---------------

// How many exceeded points a claim prints before the report's own worst-cell
// line takes over; a sweep that fails everywhere would otherwise bury it.
constexpr std::size_t kMaxReportedExceeded = 5;

// The same measurements again, keyed by order, so a claim can be reported as
// per lane, per region, per order with the delivered value beside the
// documented one - a maximum over a whole sweep cannot be acted on without
// the order it falls at.
struct OrderAccum {
    std::size_t points = 0;
    std::size_t vacuous = 0;
    std::size_t domainPoints = 0; // the bound is tighter than the value here
    std::size_t failures = 0;
    double worstRatio = 0.0;
    double worstErr = 0.0;
    double worstBound = 0.0;
    double worstX = 0.0;
};

struct Accum {
    std::string lane;
    std::string region;
    double baseBound = 0.0;
    // Whether any row of the report carries this slot's verdict. A slot
    // measured for the record - a reading the tree has withdrawn and keeps
    // re-runnable rather than deleting - is judged by no row, so it can be
    // over its bound and still leave the gate green. The flag is what lets
    // the two tables say so instead of leaving a reader to infer it from the
    // RESULT line not counting the slot.
    bool judged = true;
    std::size_t points = 0;
    std::size_t vacuous = 0;     // the bound alone exceeds |F_n(x)|
    std::size_t vacuousZero = 0; // ... and the returned value cannot hold it
    std::size_t failures = 0;
    double worstRatio = 0.0;
    double worstErr = 0.0;
    double worstBound = 0.0;
    int worstN = -1;
    double worstX = 0.0;
    // The largest function magnitude a zero/subnormal return discarded.
    double lostSignal = 0.0;
    int lostSignalN = -1;
    double lostSignalX = 0.0;
    // The domain the bound binds over: the cells where |F_n(x)| exceeds the
    // bound, so no floor and no value's own smallness can meet it. Inside it
    // the lane has to return a value inside the bound; outside it the bound is
    // met by the format's floor rather than by the arithmetic and no accuracy
    // is claimed. Counted per claim, so the two sides are never confused.
    std::size_t domainPoints = 0;
    std::size_t domainSubnormal = 0; // ... and the return is subnormal
    std::size_t domainZero = 0;      // ... and the return is the format's zero
    std::size_t domainOutside = 0;   // ... and the return is outside the bound
    double domainZeroRef = 0.0;      // largest |F_n(x)| a zero return discarded
    std::array<OrderAccum, boys::kMaxBoysOrder + 1> byOrder{};
};

inline std::vector<Accum>& Claims() {
    static std::vector<Accum> claims;
    return claims;
}

inline int AddClaim(const char* lane, const char* region, double bound, bool judged = true) {
    Accum a;
    a.lane = lane;
    a.region = region;
    a.baseBound = bound;
    a.judged = judged;
    Claims().push_back(a);
    return static_cast<int>(Claims().size()) - 1;
}

// One reading into one accumulator. Two accumulations are filled by this one
// instrument - the claim book's and the evaluation-scheme book's, and the fit
// routes' as well - so a row added to any of them is measured by the same
// comparisons and the same counters and none of them can move another's totals.
inline void MeasureAt(Accum& a,
                      int n,
                      double x,
                      double got,
                      double ref,
                      int refDecade,
                      double bound,
                      bool gotUnrepresentable) {
    ++a.points;

    const double err = std::abs(got - ref);
    const bool boundAboveValue = bound >= std::abs(ref);
    OrderAccum& o = a.byOrder[static_cast<std::size_t>(n)];
    ++o.points;

    if (boundAboveValue)
    {
        ++a.vacuous;
        ++o.vacuous;

        if (gotUnrepresentable)
        {
            ++a.vacuousZero;

            if (std::abs(ref) > a.lostSignal)
            {
                a.lostSignal = std::abs(ref);
                a.lostSignalN = n;
                a.lostSignalX = x;
            }
        }
    }

    const double ratio = err / bound;

    if (!boundAboveValue)
    {
        ++a.domainPoints;
        ++o.domainPoints;

        if (gotUnrepresentable)
        {
            ++a.domainSubnormal;
        }

        if (got == 0.0)
        {
            ++a.domainZero;

            if (std::abs(ref) > a.domainZeroRef)
            {
                a.domainZeroRef = std::abs(ref);
            }
        }

        if (ratio > 1.0)
        {
            ++a.domainOutside;
        }
    }

    if (ratio > o.worstRatio)
    {
        o.worstRatio = ratio;
        o.worstErr = err;
        o.worstBound = bound;
        o.worstX = x;
    }

    if (ratio > a.worstRatio)
    {
        a.worstRatio = ratio;
        a.worstErr = err;
        a.worstBound = bound;
        a.worstN = n;
        a.worstX = x;
    }

    if (ratio > 1.0)
    {
        ++a.failures;
        ++o.failures;

        if (a.failures <= kMaxReportedExceeded)
        {
            std::printf("  EXCEEDED%s %s / %s  n=%d x=%.17g  err=%.6g  bound=%.6g  "
                        "ratio=%.4g  ref=%.6g (1e%d)\n",
                        a.judged ? " " : " (record, not judged)",
                        a.lane.c_str(),
                        a.region.c_str(),
                        n,
                        x,
                        err,
                        bound,
                        ratio,
                        ref,
                        refDecade);
        }
    }
}

// The claim book's slot, by the vector and the index its sweeps carry. It is
// the same instrument as MeasureAt above: one count of one reading, whichever
// book the reading is counted in.
inline void MeasureInto(std::vector<Accum>& claims,
                        int claim,
                        int n,
                        double x,
                        double got,
                        double ref,
                        int refDecade,
                        double bound,
                        bool gotUnrepresentable) {
    MeasureAt(claims[static_cast<std::size_t>(claim)],
              n,
              x,
              got,
              ref,
              refDecade,
              bound,
              gotUnrepresentable);
}

inline void Measure(int claim,
                    int n,
                    double x,
                    double got,
                    double ref,
                    int refDecade,
                    double bound,
                    bool gotUnrepresentable) {
    MeasureInto(Claims(), claim, n, x, got, ref, refDecade, bound, gotUnrepresentable);
}

// --- ULP arithmetic, and what "the format cannot hold it" means ------------

// The ULP of a value in a binary format with `mantissaBits` stored bits and
// `minNormalExp` its smallest normal exponent.
inline double UlpOf(double v, int mantissaBits, int minNormalExp) {
    if (v == 0.0 || !std::isfinite(v))
    {
        return std::ldexp(1.0, minNormalExp - mantissaBits);
    }

    int e = std::ilogb(std::fabs(v));

    if (e < minNormalExp)
    {
        e = minNormalExp;
    }

    return std::ldexp(1.0, e - mantissaBits);
}

inline double HalfBound(double got, int mantissaBits, int minNormalExp) {
    return kBoundHalfBase + 0.5 * UlpOf(got, mantissaBits, minNormalExp);
}

inline bool Unrepresentable(double got, int minNormalExp) {
    return got == 0.0 || std::fabs(got) < std::ldexp(1.0, minNormalExp);
}

// --- the reference: a rectangular n x argument table, read once ------------

struct Reference {
    std::vector<double> x;        // the shared argument list
    std::vector<double> v;        // [n * count + i] = F_n(x[i])
    std::vector<int> decade;      // its base-10 magnitude
    std::vector<double> xf;       // the float the F32 lane evaluates at
    std::vector<double> vf;       // F_n at xf
    std::vector<int> decadeF;
    std::vector<double> x16;      // the fp16 value the half lane evaluates at
    std::vector<double> v16;
    std::vector<int> decade16;
    std::vector<double> xb;       // the bf16 value
    std::vector<double> vb;
    std::vector<int> decadeB;
    std::size_t orderCount = 0;

    std::size_t Index(int n, std::size_t i) const {
        return static_cast<std::size_t>(n) * count + i;
    }

    std::size_t count = 0;
};

inline std::vector<std::string> Split(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    std::istringstream stream(line);

    while (std::getline(stream, field, ','))
    {
        out.push_back(field);
    }

    return out;
}

inline Reference LoadReference(const std::string& path) {
    std::ifstream in(path);

    if (!in)
    {
        std::fprintf(stderr, "gate: cannot open the reference grid: %s\n", path.c_str());
        std::exit(2);
    }

    Reference ref;
    std::string line;
    std::getline(in, line); // header

    std::vector<double> xs;
    std::vector<double> vs;
    std::vector<int> ds;
    std::vector<double> xfs;
    std::vector<double> vfs;
    std::vector<int> dfs;
    std::vector<double> x16s;
    std::vector<double> v16s;
    std::vector<int> d16s;
    std::vector<double> xbs;
    std::vector<double> vbs;
    std::vector<int> dbs;
    int previousN = -1;

    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }

        const std::vector<std::string> f = Split(line);

        if (f.size() != 13)
        {
            std::fprintf(stderr, "gate: malformed reference row (%zu fields): %s\n",
                         f.size(),
                         line.c_str());
            std::exit(2);
        }

        const int n = std::atoi(f[0].c_str());

        if (n != previousN)
        {
            if (n != previousN + 1)
            {
                std::fprintf(stderr, "gate: reference is not n-major at row: %s\n", line.c_str());
                std::exit(2);
            }

            previousN = n;
            ++ref.orderCount;
        }

        xs.push_back(std::strtod(f[1].c_str(), nullptr));
        vs.push_back(std::strtod(f[2].c_str(), nullptr));
        ds.push_back(std::atoi(f[3].c_str()));
        xfs.push_back(std::strtod(f[4].c_str(), nullptr));
        vfs.push_back(std::strtod(f[5].c_str(), nullptr));
        dfs.push_back(std::atoi(f[6].c_str()));
        x16s.push_back(std::strtod(f[7].c_str(), nullptr));
        v16s.push_back(std::strtod(f[8].c_str(), nullptr));
        d16s.push_back(std::atoi(f[9].c_str()));
        xbs.push_back(std::strtod(f[10].c_str(), nullptr));
        vbs.push_back(std::strtod(f[11].c_str(), nullptr));
        dbs.push_back(std::atoi(f[12].c_str()));
    }

    if (ref.orderCount == 0)
    {
        std::fprintf(stderr, "gate: empty reference grid\n");
        std::exit(2);
    }

    ref.count = xs.size() / ref.orderCount;

    if (ref.count * ref.orderCount != xs.size())
    {
        std::fprintf(stderr, "gate: reference grid is not rectangular\n");
        std::exit(2);
    }

    // The first block defines the shared argument list; every later block must
    // repeat it, or an element-wise batch check would compare the wrong pair.
    for (std::size_t n = 1; n < ref.orderCount; ++n)
    {
        for (std::size_t i = 0; i < ref.count; ++i)
        {
            if (xs[n * ref.count + i] != xs[i])
            {
                std::fprintf(stderr, "gate: reference blocks disagree at n=%zu i=%zu\n", n, i);
                std::exit(2);
            }
        }
    }

    ref.x.assign(xs.begin(), xs.begin() + static_cast<std::ptrdiff_t>(ref.count));
    ref.xf.assign(xfs.begin(), xfs.begin() + static_cast<std::ptrdiff_t>(ref.count));
    ref.x16.assign(x16s.begin(), x16s.begin() + static_cast<std::ptrdiff_t>(ref.count));
    ref.xb.assign(xbs.begin(), xbs.begin() + static_cast<std::ptrdiff_t>(ref.count));
    ref.v = std::move(vs);
    ref.decade = std::move(ds);
    ref.vf = std::move(vfs);
    ref.decadeF = std::move(dfs);
    ref.v16 = std::move(v16s);
    ref.decade16 = std::move(d16s);
    ref.vb = std::move(vbs);
    ref.decadeB = std::move(dbs);

    // The rounded-argument columns must be the roundings of the same
    // argument: a mismatch would silently measure the lane at one argument
    // and the reference at another.
    for (std::size_t i = 0; i < ref.count; ++i)
    {
        if (static_cast<double>(static_cast<float>(ref.x[i])) != ref.xf[i])
        {
            std::fprintf(stderr, "gate: float argument column disagrees at i=%zu\n", i);
            std::exit(2);
        }

        if (std::isfinite(ref.x16[i]) &&
            static_cast<double>(static_cast<float>(boys::F16(static_cast<float>(ref.x[i])))) != ref.x16[i])
        {
            std::fprintf(stderr, "gate: fp16 argument column disagrees at i=%zu\n", i);
            std::exit(2);
        }

        if (static_cast<double>(static_cast<float>(boys::Bf16(static_cast<float>(ref.x[i])))) != ref.xb[i])
        {
            std::fprintf(stderr, "gate: bf16 argument column disagrees at i=%zu\n", i);
            std::exit(2);
        }
    }

    return ref;
}

// --- the row a report prints -----------------------------------------------

inline void PrintClaim(const Accum& a) {
    const std::size_t real = a.points - a.vacuous;
    char location[96] = "-";
    char delivered[64] = "-";

    if (a.worstN >= 0)
    {
        std::snprintf(location,
                      sizeof(location),
                      "%.3g (n=%d, x=%.6g)",
                      a.worstRatio,
                      a.worstN,
                      a.worstX);
        std::snprintf(delivered, sizeof(delivered), "%.3g / %.3g", a.worstErr, a.worstBound);
    }

    // The trailing field is the one thing a summary row cannot show: whether
    // the row has a verdict at all. A slot kept for the record has none, and
    // its ratio can be above 1.0 in a green run, so the row says so rather
    // than leaving it to be inferred from the RESULT line not counting it.
    std::printf("  %-24s %-9s %8zu %8zu  %-24s %-24s %8zu %8zu%s\n",
                a.lane.c_str(),
                a.region.c_str(),
                a.points,
                real,
                delivered,
                location,
                a.vacuous,
                a.vacuousZero,
                a.judged ? "" : "  record, not judged");
}

} // namespace boys_gate
