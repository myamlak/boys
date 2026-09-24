// The CUDA device gate: the device lane's documented bounds measured against
// the same independent high-precision reference the CPU gate measures against
// (tests/data/boys_accuracy_gate_reference.csv, re-derivable with
// tools/gen_boys_accuracy_gate_reference.py), over the whole of the lane's
// named domain, in the CPU gate's own row shape.
//
// What it adds to tests/boys_cuda_test.cpp. That test compares the device lane
// against the CPU lane - |gpu - cpu| against a cross-lane budget - and the two
// implementations it subtracts can carry error in the same direction, so a
// difference bounds the device's distance from the CPU lane and not its
// distance from F_n(x). This gate removes the middle term: the reference is the
// committed grid of the CPU gate, which shares no code with the library, and
// every device entry is measured against it directly. The device contract's
// numbers are what the rows below hold the sweep to.
//
// What is measured, per entry. Every documented device bound is a claim about
// a (lane, region) cell, and the reference grid is rectangular over orders
// 0..kMaxBoysOrder and one shared argument list, so one launch per entry
// covers every cell: the single entries are handed one element per (order,
// argument) cell, and the batch entries are handed the argument list at
// nmax = kMaxBoysOrder. A cell the bound cannot fail on - the bound is at
// least as large as |F_n(x)| itself, so any return in range passes there - is
// counted in the vacuous column rather than folded into the total, which is
// the CPU gate's rule and the number to read before the green rows.
//
// The card is named rather than assumed. A delivered figure describes the
// arithmetic this device executes; a weaker or stronger double unit changes
// what a bound costs and not what it is, so the bounds transfer between cards
// and the delivered figures do not. The relaxed instantiations are measured
// too, because the header asserts a bound for them: the device has no
// per-call accuracy parameter, so each multiplier is a lane of its own, and
// each sweep launches the instantiation its row names.
//
// One entry carries two certified options rather than one arithmetic: the f32
// single entry's region-B exponential. It is swept once per option, against the
// bound that option documents, and the two returns are measured against each
// other as well - they differ in that one factor, so their difference is the
// contribution the fast option's second bound term has to cover, and the
// wrong-sign cells are the audit for the defect that term exists because of.
// The term itself is derived from the recurrence's condition number rather than
// read off this sweep; the sweep is what confirms it.
//
// Run:  cmake --build <build> --config Release --target boys-cuda-accuracy-gate
//       <build>/Release/boys-cuda-accuracy-gate [--reference <grid.csv>]
// Exits non-zero when any measured cell is over its bound.

#include "boys/boys.hpp"
#include "boys/boys_cuda.hpp"
#include "boys/boys_impl.hpp" // the region boundaries
#include "boys/f16.hpp"
#include "boys_gate_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// The reference grid is located the way the CPU gate locates it; a build that
// bypasses CMake still says which revision it measured rather than naming one
// it did not read.
#ifndef BoysDataDir
#define BoysDataDir "tests/data"
#endif

#ifndef BoysGateRevision
#define BoysGateRevision "unknown"
#endif

// The same reference reader and row shape the CPU gate reports in, so a lane
// measured here is measured against one reference format and printed in one
// vocabulary rather than one apiece.
using namespace boys_gate;

namespace {

// ---------------------------------------------------------------------------
// The documented device bounds, transcribed from README.md and
// include/boys/boys_cuda.hpp.
// ---------------------------------------------------------------------------

// README's accuracy contract: "CUDA fp64 | same m*budgets as the CPU double
// lanes". The CPU double single lane is the one with per-region cells
// (1e-15 below the region-A edge, 3e-14 through the extended band and region
// B, 5.5e-14 in region C); the CPU double batch lane publishes one, 5.5e-14.
constexpr double kBoundSingleA = 1e-15;
constexpr double kBoundSingleBand = 3e-14;
constexpr double kBoundSingleB = 3e-14;
constexpr double kBoundSingleC = 5.5e-14;
constexpr double kBoundDoubleBatch = 5.5e-14;
// README: "CUDA fp32, RegionBExp::kAccurate (the default) | same m*budgets as
// the CPU float lanes" - 1.5e-7, the bound of the single entry's accurate
// region-B exponential, of the batch entries and of the all-n entry.
constexpr double kBoundFloat = 1.5e-7;
// The single entry's fast region-B exponential carries its own bound: the
// lane's m * 1.5e-7 plus the corrected seed's own contribution, which the
// recurrence's amplification caps at this figure. The cap is derived, not
// measured cell by cell: the region-B ladder f_l = ((l - 1/2) f_{l-1} - e)/x
// propagates an error in e to order n with gain G_n(x) = sum_k A_n/(x A_k),
// A_m = prod_{j<=m} (j - 1/2)/x - the ratio of the dominant solution of the
// homogeneous recurrence to the wanted one, 7.6e4 at n = 32 at the region-B
// boundary - so a seed whose relative error is flat at rho ulp contributes at
// most G_n (1/2) e^{-x} rho: 6.1e-8 at rho = 4 ulp, 8e-8 here. The sweep below
// reports the contribution it actually measured, and the audit reports the
// wrong-sign cells (zero at m = 1 for the corrected form).
constexpr double kFastExpContribution = 8e-8;
// The fp16 entries' bound is the header's: m * 1e-7 + 1/2 ULP of the returned
// value, which is what HalfBound computes at m = 1.

// The multipliers the header names as instantiated (a call is fixed at the m
// its instantiation was built with), in the order the rows print.
constexpr const char* kRelaxedNames[] = {"2", "10", "100", "1e4", "1e8"};

// The smallest positive normal float: below it a returned value is the
// format's floor rather than arithmetic, which is what the relative-error side
// of the fast exponential's audit is restricted to.
constexpr double kFloatMinNormal = std::numeric_limits<float>::min();

// What the fast region-B exponential costs, measured against the accurate one
// over the same grid at the same multiplier: the seed substitution's own
// contribution (max|fast - accurate|, which is the term the fast bound carries
// on top of the lane's), and the cells whose return has the opposite sign to
// the function. The second is a tripwire rather than a bound term: a seed error
// the recurrence amplifies is what put the wrong sign there before the
// correction, so a non-zero count at m = 1 means the correction has been lost.
// A wrong sign is at least as large as the value itself in relative terms, so
// that column is never below 1.
struct ExpAudit {
    std::string lane;
    double contributed = 0.0;
    int contributedN = -1;
    double contributedX = 0.0;
    std::size_t relativeCells = 0;
    std::size_t wrongSign = 0;
    double signRelative = 0.0;
    int signRelativeN = -1;
    double signRelativeX = 0.0;
    double signRelativeWant = 0.0;
    double signRelativeGot = 0.0;
};

std::vector<ExpAudit>& ExpAudits() {
    static std::vector<ExpAudit> audits;
    return audits;
}

// The region a double-single argument is dispatched in, by README's interval
// table - the same partition the CPU gate keys its double single rows by.
int SingleClaim(double x) {
    if (x < boys::detail::kExtendedBX0)
    {
        return 0;
    }

    if (x < boys::detail::kX0)
    {
        return 1;
    }

    if (x < boys::detail::kX1)
    {
        return 2;
    }

    return 3;
}

[[noreturn]] void Fail(const char* what, cudaError_t error) {
    std::fprintf(stderr, "cuda gate: %s failed: %s\n", what, cudaGetErrorString(error));
    std::exit(2);
}

void Check(cudaError_t error, const char* what) {
    if (error != cudaSuccess)
    {
        Fail(what, error);
    }
}

void CheckLaunch(boys::BoysStatus status, const char* what) {
    if (status != boys::BoysStatus::kSuccess)
    {
        std::fprintf(stderr, "cuda gate: %s returned status %d\n", what, static_cast<int>(status));
        std::exit(2);
    }

    Check(cudaGetLastError(), what);
}

// A device buffer the gate owns for the length of one launch.
template <typename T> class DevBuf {
public:
    explicit DevBuf(std::size_t count) : mCount(count) {
        Check(cudaMalloc(reinterpret_cast<void**>(&mPtr), count * sizeof(T)), "cudaMalloc");
    }

    ~DevBuf() {
        if (mPtr != nullptr)
        {
            cudaFree(mPtr);
        }
    }

    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;

    T* get() const {
        return mPtr;
    }

    void Upload(const std::vector<T>& src) {
        Check(cudaMemcpy(mPtr,
                         src.data(),
                         mCount * sizeof(T),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy host to device");
    }

    void Download(std::vector<T>& dst) const {
        Check(cudaMemcpy(dst.data(),
                         mPtr,
                         mCount * sizeof(T),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy device to host");
    }

private:
    T* mPtr = nullptr;
    std::size_t mCount = 0;
};

// The swept cells, laid out so that an element index IS the reference's own
// index for the cell it holds: element e = n * count + i carries order n at
// argument i, so one single-element launch over this list covers every cell of
// the grid and the value the entry returns for e is the value the reference
// holds at e.
struct Grid {
    std::vector<int> n;
    std::vector<double> x;
    std::vector<boys::F16> x16;
    std::size_t cells = 0;
};

Grid MakeGrid(const Reference& ref) {
    Grid grid;
    const int nmax = boys::kMaxBoysOrder;
    grid.cells = ref.count * static_cast<std::size_t>(nmax + 1);
    grid.n.resize(grid.cells);
    grid.x.resize(grid.cells);
    grid.x16.resize(grid.cells);

    for (int n = 0; n <= nmax; ++n)
    {
        for (std::size_t i = 0; i < ref.count; ++i)
        {
            const std::size_t e = ref.Index(n, i);
            grid.n[e] = n;
            grid.x[e] = ref.x[i];
            grid.x16[e] = boys::F16(static_cast<float>(ref.x[i]));
        }
    }

    return grid;
}

// The batch entries' precondition (x[i - 1] <= x[i]) as a permutation: the
// reference's argument list in non-decreasing order, with the index each
// argument came from, so a cell measured off a sorted batch is still measured
// against the reference row for its own argument.
struct SortedArgs {
    std::vector<double> x;
    std::vector<std::size_t> order;
};

SortedArgs SortArgs(const Reference& ref) {
    SortedArgs sorted;
    sorted.order.resize(ref.count);
    std::iota(sorted.order.begin(), sorted.order.end(), std::size_t{0});
    std::sort(sorted.order.begin(), sorted.order.end(), [&](std::size_t a, std::size_t b) {
        return ref.x[a] < ref.x[b];
    });
    sorted.x.resize(ref.count);

    for (std::size_t j = 0; j < ref.count; ++j)
    {
        sorted.x[j] = ref.x[sorted.order[j]];
    }

    return sorted;
}

std::string Label(const char* lane, const char* rung) {
    return rung == nullptr ? std::string(lane) : std::string(lane) + " m=" + rung;
}

// ---------------------------------------------------------------------------
// The sweeps, one per precision family, each covering all three shapes.
// ---------------------------------------------------------------------------

// The fp16 bound at a multiplier: the header's m * 1e-7 + 1/2 ULP, in ULP of
// the value the entry returned rather than of the value it should have.
double HalfBoundAt(double got, double multiplier) {
    return multiplier * kBoundHalfBase + 0.5 * UlpOf(got, kF16MantissaBits, kF16MinNormalExp);
}

// Each sweep is measured at one multiplier and launches that multiplier's
// instantiation: the header asserts a bound per multiplier, so the kernel that
// has to meet it is the one the multiplier names.
template <double kMultiplier>
void SweepDouble(const Reference& ref,
                 const Grid& grid,
                 const SortedArgs& sorted,
                 int slotSingleA,
                 int slotSingleBand,
                 int slotSingleB,
                 int slotSingleC,
                 int slotOrders,
                 int slotAllN) {
    const std::size_t count = ref.count;
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t cells = grid.cells;
    const int singleSlots[4] = {slotSingleA, slotSingleBand, slotSingleB, slotSingleC};

    // The single entry: one element per cell of the grid.
    {
        DevBuf<int> dN(cells);
        DevBuf<double> dX(cells);
        DevBuf<double> dOut(cells);
        dN.Upload(grid.n);
        dX.Upload(grid.x);
        CheckLaunch(
            boys::BoysCuda::SingleF64<kMultiplier>(dN.get(), dX.get(), dOut.get(), cells, nullptr),
            "SingleF64");
        std::vector<double> out(cells);
        dOut.Download(out);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t e = ref.Index(n, i);
                const int slot = singleSlots[SingleClaim(ref.x[i])];
                Measure(slot,
                        n,
                        ref.x[i],
                        out[e],
                        ref.v[e],
                        ref.decade[e],
                        Claims()[static_cast<std::size_t>(slot)].baseBound,
                        Unrepresentable(out[e], -1022));
            }
        }
    }

    // The per-element-order batch entry: every argument at the top order, so
    // the whole output family is filled by one launch.
    {
        DevBuf<int> dN(count);
        DevBuf<double> dX(count);
        DevBuf<double> dOut(cells);
        std::vector<int> hostN(count, nmax);
        dN.Upload(hostN);
        dX.Upload(grid.x);
        CheckLaunch(boys::BoysCuda::AllOrdersF64<kMultiplier>(
                        dN.get(), dX.get(), dOut.get(), count, nullptr),
                    "AllOrdersF64");
        std::vector<double> out(cells);
        dOut.Download(out);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const double got = out[ref.Index(n, i)];
                Measure(slotOrders,
                        n,
                        ref.x[i],
                        got,
                        ref.v[ref.Index(n, i)],
                        ref.decade[ref.Index(n, i)],
                        kMultiplier * kBoundDoubleBatch,
                        Unrepresentable(got, -1022));
            }
        }
    }

    // The uniform-order batch entry, on the non-decreasing argument list its
    // contract asks for.
    {
        DevBuf<double> dX(count);
        DevBuf<double> dOut(cells);
        dX.Upload(sorted.x);
        CheckLaunch(
            boys::BoysCuda::AllNF64<kMultiplier>(nmax, dX.get(), dOut.get(), count, nullptr),
            "AllNF64");
        std::vector<double> out(cells);
        dOut.Download(out);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t j = 0; j < count; ++j)
            {
                const std::size_t i = sorted.order[j];
                const std::size_t e = static_cast<std::size_t>(n) * count + j;
                Measure(slotAllN,
                        n,
                        ref.x[i],
                        out[e],
                        ref.v[ref.Index(n, i)],
                        ref.decade[ref.Index(n, i)],
                        kMultiplier * kBoundDoubleBatch,
                        Unrepresentable(out[e], -1022));
            }
        }
    }
}

template <double kMultiplier>
void SweepFloat(const Reference& ref,
                const Grid& grid,
                const SortedArgs& sorted,
                int slotSingle,
                int slotSingleFast,
                int slotOrders,
                int slotAllN,
                const std::string& fastLane) {
    const std::size_t count = ref.count;
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t cells = grid.cells;
    const double bound = kMultiplier * kBoundFloat;
    const double fastBound = kMultiplier * kBoundFloat + kFastExpContribution;

    // The entry the caller hands a double and the kernel narrows itself: the
    // argument measured at is the float it evaluates at, which is the
    // reference's own xf column.
    const auto narrow = [](const float* got) {
        const double asDouble = static_cast<double>(*got);
        return std::make_pair(asDouble,
                              *got == 0.0f ||
                                  std::fabs(asDouble) < std::numeric_limits<float>::min());
    };

    // The single entry's two options in one pass. They share every operation
    // but the exponential, so the difference between them is that exponential's
    // own contribution and nothing else - which is what the fast bound's second
    // term has to cover, and what the audit below reports from the measurement
    // rather than from the constant.
    {
        DevBuf<int> dN(cells);
        DevBuf<double> dX(cells);
        DevBuf<float> dOut(cells);
        DevBuf<float> dOutFast(cells);
        dN.Upload(grid.n);
        dX.Upload(grid.x);
        CheckLaunch(
            boys::BoysCuda::SingleF32<kMultiplier>(dN.get(), dX.get(), dOut.get(), cells, nullptr),
            "SingleF32");
        CheckLaunch(boys::BoysCuda::SingleF32<kMultiplier, boys::RegionBExp::kFast>(
                        dN.get(), dX.get(), dOutFast.get(), cells, nullptr),
                    "SingleF32 fast");
        std::vector<float> out(cells);
        std::vector<float> outFast(cells);
        dOut.Download(out);
        dOutFast.Download(outFast);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        ExpAudit audit;
        audit.lane = fastLane;

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t e = ref.Index(n, i);
                const auto [got, unrepresentable] = narrow(&out[e]);
                Measure(slotSingle,
                        n,
                        ref.xf[i],
                        got,
                        ref.vf[e],
                        ref.decadeF[e],
                        bound,
                        unrepresentable);

                const auto [gotFast, fastUnrepresentable] = narrow(&outFast[e]);
                Measure(slotSingleFast,
                        n,
                        ref.xf[i],
                        gotFast,
                        ref.vf[e],
                        ref.decadeF[e],
                        fastBound,
                        fastUnrepresentable);

                const double contribution = std::abs(gotFast - got);

                if (contribution > audit.contributed)
                {
                    audit.contributed = contribution;
                    audit.contributedN = n;
                    audit.contributedX = ref.xf[i];
                }

                // The sign audit is over the cells whose value the format
                // holds: a subnormal or zero F_n(x) has no relative error to
                // speak of, and the bound's own floor is what covers those
                // cells.
                const double want = ref.vf[e];

                if (want != 0.0 && std::fabs(want) >= kFloatMinNormal)
                {
                    ++audit.relativeCells;

                    if (gotFast != 0.0 && std::signbit(gotFast) != std::signbit(want))
                    {
                        ++audit.wrongSign;
                        const double relative = std::abs(gotFast - want) / std::fabs(want);

                        if (relative > audit.signRelative)
                        {
                            audit.signRelative = relative;
                            audit.signRelativeN = n;
                            audit.signRelativeX = ref.xf[i];
                            audit.signRelativeWant = want;
                            audit.signRelativeGot = gotFast;
                        }
                    }
                }
            }
        }

        ExpAudits().push_back(audit);
    }

    {
        DevBuf<int> dN(count);
        DevBuf<double> dX(count);
        DevBuf<float> dOut(cells);
        std::vector<int> hostN(count, nmax);
        dN.Upload(hostN);
        dX.Upload(grid.x);
        CheckLaunch(boys::BoysCuda::AllOrdersF32<kMultiplier>(
                        dN.get(), dX.get(), dOut.get(), count, nullptr),
                    "AllOrdersF32");
        std::vector<float> out(cells);
        dOut.Download(out);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t e = ref.Index(n, i);
                const auto [got, unrepresentable] = narrow(&out[e]);
                Measure(slotOrders,
                        n,
                        ref.xf[i],
                        got,
                        ref.vf[e],
                        ref.decadeF[e],
                        bound,
                        unrepresentable);
            }
        }
    }

    {
        DevBuf<double> dX(count);
        DevBuf<float> dOut(cells);
        dX.Upload(sorted.x);
        CheckLaunch(
            boys::BoysCuda::AllNF32<kMultiplier>(nmax, dX.get(), dOut.get(), count, nullptr),
            "AllNF32");
        std::vector<float> out(cells);
        dOut.Download(out);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t j = 0; j < count; ++j)
            {
                const std::size_t i = sorted.order[j];
                const std::size_t e = static_cast<std::size_t>(n) * count + j;
                const auto [got, unrepresentable] = narrow(&out[e]);
                Measure(slotAllN,
                        n,
                        ref.xf[i],
                        got,
                        ref.vf[ref.Index(n, i)],
                        ref.decadeF[ref.Index(n, i)],
                        bound,
                        unrepresentable);
            }
        }
    }
}

template <double kMultiplier>
void SweepHalf(const Reference& ref,
               const Grid& grid,
               const SortedArgs& sorted,
               int slotSingle,
               int slotOrders,
               int slotAllN) {
    const std::size_t count = ref.count;
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t cells = grid.cells;

    {
        DevBuf<int> dN(cells);
        DevBuf<boys::F16> dX(cells);
        DevBuf<boys::F16> dOut(cells);
        dN.Upload(grid.n);
        dX.Upload(grid.x16);
        CheckLaunch(
            boys::BoysCuda::SingleF16<kMultiplier>(dN.get(), dX.get(), dOut.get(), cells, nullptr),
            "SingleF16");
        std::vector<boys::F16> out(cells);
        dOut.Download(out);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                // An argument past the fp16 range has no reference value at
                // all (the grid's x16 column is infinite there), so it is not
                // a measured cell of this row.
                if (!std::isfinite(ref.x16[i]))
                {
                    continue;
                }

                const std::size_t e = ref.Index(n, i);
                const double got = static_cast<double>(out[e]);
                Measure(slotSingle,
                        n,
                        ref.x16[i],
                        got,
                        ref.v16[e],
                        ref.decade16[e],
                        HalfBoundAt(got, kMultiplier),
                        Unrepresentable(got, kF16MinNormalExp));
            }
        }
    }

    {
        DevBuf<int> dN(count);
        DevBuf<boys::F16> dX(count);
        DevBuf<boys::F16> dOut(cells);
        std::vector<int> hostN(count, nmax);
        std::vector<boys::F16> hostX(count);

        for (std::size_t i = 0; i < count; ++i)
        {
            hostX[i] = boys::F16(static_cast<float>(ref.x[i]));
        }

        dN.Upload(hostN);
        dX.Upload(hostX);
        CheckLaunch(boys::BoysCuda::AllOrdersF16<kMultiplier>(
                        dN.get(), dX.get(), dOut.get(), count, nullptr),
                    "AllOrdersF16");
        std::vector<boys::F16> out(cells);
        dOut.Download(out);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t e = ref.Index(n, i);
                const double got = static_cast<double>(out[e]);
                Measure(slotOrders,
                        n,
                        ref.x16[i],
                        got,
                        ref.v16[e],
                        ref.decade16[e],
                        HalfBoundAt(got, kMultiplier),
                        Unrepresentable(got, kF16MinNormalExp));
            }
        }
    }

    {
        DevBuf<boys::F16> dX(count);
        DevBuf<boys::F16> dOut(cells);
        std::vector<boys::F16> hostX(count);

        for (std::size_t j = 0; j < count; ++j)
        {
            hostX[j] = boys::F16(static_cast<float>(ref.x[sorted.order[j]]));
        }

        dX.Upload(hostX);
        CheckLaunch(
            boys::BoysCuda::AllNF16<kMultiplier>(nmax, dX.get(), dOut.get(), count, nullptr),
            "AllNF16");
        std::vector<boys::F16> out(cells);
        dOut.Download(out);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t j = 0; j < count; ++j)
            {
                const std::size_t i = sorted.order[j];
                const std::size_t e = static_cast<std::size_t>(n) * count + j;
                const double got = static_cast<double>(out[e]);
                Measure(slotAllN,
                        n,
                        ref.x16[i],
                        got,
                        ref.v16[ref.Index(n, i)],
                        ref.decade16[ref.Index(n, i)],
                        HalfBoundAt(got, kMultiplier),
                        Unrepresentable(got, kF16MinNormalExp));
            }
        }
    }
}

// One relaxed instantiation's rows. The device takes the multiplier as a
// compile-time argument and has no per-call accuracy parameter, so each one is
// the separate lane the header says it is; its asserted bound is the header's
// m * (the m = 1 budget), which is the relaxation the contract states.
template <double kMultiplier>
void SweepRelaxed(const Reference& ref,
                  const Grid& grid,
                  const SortedArgs& sorted,
                  const char* rung) {
    const std::string doubleSingle = Label("cuda single f64", rung);
    const std::string doubleOrders = Label("cuda all-orders f64", rung);
    const std::string doubleAllN = Label("cuda all-n f64", rung);
    const std::string floatSingle = Label("cuda single f32", rung);
    const std::string floatSingleFast = Label("cuda single f32 (fast exp)", rung);
    const std::string floatOrders = Label("cuda all-orders f32", rung);
    const std::string floatAllN = Label("cuda all-n f32", rung);
    const std::string halfSingle = Label("cuda single f16", rung);
    const std::string halfOrders = Label("cuda all-orders f16", rung);
    const std::string halfAllN = Label("cuda all-n f16", rung);

    const int doubleSingleA = AddClaim(doubleSingle.c_str(), "A", kMultiplier * kBoundSingleA);
    const int doubleSingleBand =
        AddClaim(doubleSingle.c_str(), "band", kMultiplier * kBoundSingleBand);
    const int doubleSingleB = AddClaim(doubleSingle.c_str(), "B", kMultiplier * kBoundSingleB);
    const int doubleSingleC = AddClaim(doubleSingle.c_str(), "C", kMultiplier * kBoundSingleC);
    const int doubleOrderSlot =
        AddClaim(doubleOrders.c_str(), "A..C", kMultiplier * kBoundDoubleBatch);
    const int doubleAllNSlot =
        AddClaim(doubleAllN.c_str(), "A..C", kMultiplier * kBoundDoubleBatch);
    const int floatSingleSlot = AddClaim(floatSingle.c_str(), "A..C", kMultiplier * kBoundFloat);
    // The fast option's own bound, which permits the wrong sign the audit
    // reports: the lane's m * 1.5e-7 plus the exponential's contribution.
    const int floatSingleFastSlot =
        AddClaim(floatSingleFast.c_str(), "A..C", kMultiplier * kBoundFloat + kFastExpContribution);
    const int floatOrderSlot = AddClaim(floatOrders.c_str(), "A..C", kMultiplier * kBoundFloat);
    const int floatAllNSlot = AddClaim(floatAllN.c_str(), "A..C", kMultiplier * kBoundFloat);
    const int halfSingleSlot = AddClaim(halfSingle.c_str(), "A..C", kMultiplier * kBoundHalfBase);
    const int halfOrderSlot = AddClaim(halfOrders.c_str(), "A..C", kMultiplier * kBoundHalfBase);
    const int halfAllNSlot = AddClaim(halfAllN.c_str(), "A..C", kMultiplier * kBoundHalfBase);

    SweepDouble<kMultiplier>(ref,
                             grid,
                             sorted,
                             doubleSingleA,
                             doubleSingleBand,
                             doubleSingleB,
                             doubleSingleC,
                             doubleOrderSlot,
                             doubleAllNSlot);
    SweepFloat<kMultiplier>(ref,
                            grid,
                            sorted,
                            floatSingleSlot,
                            floatSingleFastSlot,
                            floatOrderSlot,
                            floatAllNSlot,
                            floatSingleFast);
    SweepHalf<kMultiplier>(ref, grid, sorted, halfSingleSlot, halfOrderSlot, halfAllNSlot);
}

// One cell, every entry: what each device entry returns beside the reference,
// so a reported failure can be reproduced and acted on rather than believed.
// The argument is looked up on the committed grid; an argument that is not on
// it is printed as such rather than measured against the nearest node.
void RunProbe(const Reference& ref, int n, double x) {
    const int nmax = boys::kMaxBoysOrder;
    const int count = 1;
    const double xf = static_cast<double>(static_cast<float>(x));
    const double x16 = static_cast<double>(boys::F16(static_cast<float>(x)));
    const bool finiteX16 = std::isfinite(x16);
    const std::size_t notFound = ref.count * static_cast<std::size_t>(nmax + 1);

    // Each lane is measured at its own rounding of the argument, and the grid
    // carries a column per rounding, so the reference for a lane is the cell
    // whose own column holds the argument that lane evaluates at. A column
    // with no such cell leaves that lane's reference blank rather than
    // measuring it against the nearest node.
    const auto findIn = [&](const std::vector<double>& column, double value) {
        for (std::size_t i = 0; i < ref.count; ++i)
        {
            if (column[i] == value)
            {
                return ref.Index(n, i);
            }
        }

        return notFound;
    };

    const std::size_t atX = findIn(ref.x, x);
    const std::size_t atF = findIn(ref.xf, xf);
    const std::size_t atH = finiteX16 ? findIn(ref.x16, x16) : notFound;
    const bool have64 = atX != notFound;
    const bool have32 = atF != notFound;
    const bool have16 = atH != notFound;

    std::vector<int> hostN(1, n);
    std::vector<double> hostX(1, x);
    std::vector<boys::F16> hostH(1, boys::F16(static_cast<float>(x)));

    DevBuf<int> dN(1);
    DevBuf<double> dX(1);
    DevBuf<boys::F16> dH(1);
    dN.Upload(hostN);
    dX.Upload(hostX);
    dH.Upload(hostH);

    const std::size_t planes = static_cast<std::size_t>(nmax + 1);
    DevBuf<double> d64(planes);
    DevBuf<float> d32(planes);
    DevBuf<boys::F16> d16(planes);

    const auto row = [&](const char* lane, double got, double want, double bound, bool have) {
        if (!have)
        {
            std::printf("  %-22s %.17g   %-24s %-12s %-12s\n", lane, got, "-", "-", "-");
            return;
        }

        const double err = std::abs(got - want);
        std::printf("  %-22s %.17g   %-24.17g %-12.3g %-12.3g\n",
                    lane,
                    got,
                    want,
                    err,
                    err / bound);
    };

    std::printf("  probe n=%d x=%.17g\n", n, x);
    std::printf(
        "  %-22s %-19s   %-24s %-12s %-12s\n", "", "delivered", "reference", "|diff|", "ratio");

    if (!have64 && !have32 && !have16)
    {
        std::printf("  (argument is on no column of the grid: reference blank)\n");
    }

    std::vector<double> out64(planes);
    std::vector<float> out32(planes);
    std::vector<boys::F16> out16(planes);
    const double ref64 = have64 ? ref.v[atX] : 0.0;
    const double ref32 = have32 ? ref.vf[atF] : 0.0;
    const double ref16 = have16 ? ref.v16[atH] : 0.0;


    CheckLaunch(boys::BoysCuda::SingleF64<1.0>(dN.get(), dX.get(), d64.get(), count, nullptr),
                "SingleF64");
    d64.Download(out64);
    row("cuda single f64", out64[0], ref64, kBoundSingleC, have64);

    CheckLaunch(
        boys::BoysCuda::AllOrdersF64<1.0>(dN.get(), dX.get(), d64.get(), count, nullptr),
        "AllOrdersF64");
    d64.Download(out64);
    row("cuda all-orders f64",
        out64[static_cast<std::size_t>(n)],
        ref64,
        kBoundDoubleBatch,
        have64);

    CheckLaunch(boys::BoysCuda::AllNF64<1.0>(n, dX.get(), d64.get(), count, nullptr), "AllNF64");
    d64.Download(out64);
    row("cuda all-n f64", out64[static_cast<std::size_t>(n)], ref64, kBoundDoubleBatch, have64);

    CheckLaunch(boys::BoysCuda::SingleF32<1.0>(dN.get(), dX.get(), d32.get(), count, nullptr),
                "SingleF32");
    d32.Download(out32);
    row("cuda single f32", static_cast<double>(out32[0]), ref32, kBoundFloat, have32);
    std::printf("  %-22s (evaluated at xf=%.17g)\n", "", xf);

    CheckLaunch(boys::BoysCuda::SingleF32<1.0, boys::RegionBExp::kFast>(
                    dN.get(), dX.get(), d32.get(), count, nullptr),
                "SingleF32 fast");
    d32.Download(out32);
    row("cuda single f32 (fast exp)",
        static_cast<double>(out32[0]),
        ref32,
        kBoundFloat + kFastExpContribution,
        have32);

    CheckLaunch(
        boys::BoysCuda::AllOrdersF32<1.0>(dN.get(), dX.get(), d32.get(), count, nullptr),
        "AllOrdersF32");
    d32.Download(out32);
    row("cuda all-orders f32",
        static_cast<double>(out32[static_cast<std::size_t>(n)]),
        ref32,
        kBoundFloat,
        have32);

    CheckLaunch(boys::BoysCuda::AllNF32<1.0>(n, dX.get(), d32.get(), count, nullptr), "AllNF32");
    d32.Download(out32);
    row("cuda all-n f32",
        static_cast<double>(out32[static_cast<std::size_t>(n)]),
        ref32,
        kBoundFloat,
        have32);

    CheckLaunch(boys::BoysCuda::SingleF16<1.0>(dN.get(), dH.get(), d16.get(), count, nullptr),
                "SingleF16");
    d16.Download(out16);
    row("cuda single f16",
        static_cast<double>(out16[0]),
        ref16,
        HalfBoundAt(static_cast<double>(out16[0]), 1.0),
        have16);
    std::printf("  %-22s (evaluated at x16=%.17g)\n", "", x16);

    CheckLaunch(
        boys::BoysCuda::AllOrdersF16<1.0>(dN.get(), dH.get(), d16.get(), count, nullptr),
        "AllOrdersF16");
    d16.Download(out16);
    row("cuda all-orders f16",
        static_cast<double>(out16[static_cast<std::size_t>(n)]),
        ref16,
        HalfBoundAt(static_cast<double>(out16[static_cast<std::size_t>(n)]), 1.0),
        have16);

    CheckLaunch(boys::BoysCuda::AllNF16<1.0>(n, dH.get(), d16.get(), count, nullptr), "AllNF16");
    d16.Download(out16);
    row("cuda all-n f16",
        static_cast<double>(out16[static_cast<std::size_t>(n)]),
        ref16,
        HalfBoundAt(static_cast<double>(out16[static_cast<std::size_t>(n)]), 1.0),
        have16);
}

} // namespace

int main(int argc, char** argv) {
    std::string reference = std::string(BoysDataDir) + "/boys_accuracy_gate_reference.csv";
    int probeN = -1;
    double probeX = 0.0;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--reference" && i + 1 < argc)
        {
            reference = argv[++i];
        }
        else if (arg == "--probe" && i + 2 < argc)
        {
            probeN = std::atoi(argv[++i]);
            probeX = std::strtod(argv[++i], nullptr);
        }
    }

    if (probeN >= 0 && probeN > boys::kMaxBoysOrder)
    {
        std::printf("cuda gate: order %d is outside 0..%d\n", probeN, boys::kMaxBoysOrder);
        return 2;
    }

    std::printf("boys CUDA device accuracy gate\n");

    int deviceCount = 0;
    Check(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");

    if (deviceCount == 0)
    {
        std::printf("device       : none (the CUDA lane cannot be measured here)\n");
        std::printf("  RESULT: evidence absent - no CUDA device on this machine\n");
        return 2;
    }

    cudaDeviceProp prop{};
    Check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    Check(cudaSetDevice(0), "cudaSetDevice");

    const Reference ref = LoadReference(reference);
    const Grid grid = MakeGrid(ref);
    const SortedArgs sorted = SortArgs(ref);
    CheckLaunch(boys::BoysCuda::InitializeTables(), "InitializeTables");

    if (probeN >= 0)
    {
        RunProbe(ref, probeN, probeX);
        return 0;
    }

    const int slotSingleA = AddClaim("cuda single f64", "A", kBoundSingleA);
    const int slotSingleBand = AddClaim("cuda single f64", "band", kBoundSingleBand);
    const int slotSingleB = AddClaim("cuda single f64", "B", kBoundSingleB);
    const int slotSingleC = AddClaim("cuda single f64", "C", kBoundSingleC);
    const int slotDoubleOrders = AddClaim("cuda all-orders f64", "A..C", kBoundDoubleBatch);
    const int slotDoubleAllN = AddClaim("cuda all-n f64", "A..C", kBoundDoubleBatch);
    const int slotFloatSingle = AddClaim("cuda single f32", "A..C", kBoundFloat);
    // The single entry's fast region-B exponential: a second certified option
    // with a bound of its own, not a relaxed rung of the lane's.
    const int slotFloatSingleFast =
        AddClaim("cuda single f32 (fast exp)", "A..C", kBoundFloat + kFastExpContribution);
    const int slotFloatOrders = AddClaim("cuda all-orders f32", "A..C", kBoundFloat);
    const int slotFloatAllN = AddClaim("cuda all-n f32", "A..C", kBoundFloat);
    const int slotHalfSingle = AddClaim("cuda single f16", "A..C", kBoundHalfBase);
    const int slotHalfOrders = AddClaim("cuda all-orders f16", "A..C", kBoundHalfBase);
    const int slotHalfAllN = AddClaim("cuda all-n f16", "A..C", kBoundHalfBase);

    SweepDouble<1.0>(ref,
                     grid,
                     sorted,
                     slotSingleA,
                     slotSingleBand,
                     slotSingleB,
                     slotSingleC,
                     slotDoubleOrders,
                     slotDoubleAllN);
    SweepFloat<1.0>(ref,
                    grid,
                    sorted,
                    slotFloatSingle,
                    slotFloatSingleFast,
                    slotFloatOrders,
                    slotFloatAllN,
                    "cuda single f32 (fast exp)");
    SweepHalf<1.0>(ref, grid, sorted, slotHalfSingle, slotHalfOrders, slotHalfAllN);

    SweepRelaxed<2.0>(ref, grid, sorted, kRelaxedNames[0]);
    SweepRelaxed<10.0>(ref, grid, sorted, kRelaxedNames[1]);
    SweepRelaxed<100.0>(ref, grid, sorted, kRelaxedNames[2]);
    SweepRelaxed<1e4>(ref, grid, sorted, kRelaxedNames[3]);
    SweepRelaxed<1e8>(ref, grid, sorted, kRelaxedNames[4]);

    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("revision     : %s\n", BoysGateRevision);
    std::printf("device       : %s (compute capability %d.%d, %zu MiB)\n",
                prop.name,
                prop.major,
                prop.minor,
                static_cast<std::size_t>(prop.totalGlobalMem >> 20));
    std::printf("reference    : %s\n", reference.c_str());
    std::printf("               %zu orders x %zu arguments = %zu points per entry\n",
                ref.orderCount,
                ref.count,
                ref.orderCount * ref.count);
    std::printf("boundaries   : x_ext=%.17g  x0=%.17g  x1=%.17g\n",
                boys::detail::kExtendedBX0,
                boys::detail::kX0,
                boys::detail::kX1);
    std::printf("bounds       : README \"CUDA fp64 | same m*budgets as the CPU double lanes\",\n"
                "               README \"CUDA fp32, RegionBExp::kAccurate (the default) | same"
                " m*budgets\n"
                "               as the CPU float lanes\" (1.5e-7),\n"
                "               README \"CUDA fp32, RegionBExp::kFast | <= m*1.5e-7 + 8e-8 - the"
                " lane's\n"
                "               budget plus the corrected seed's own contribution\",\n"
                "               boys_cuda.hpp \"m * 1e-7 + 1/2 ULP\" for the fp16 entries\n");

    std::printf("\n  lane                     region   points     real  delivered / bound        "
                "worst cell               vacuous   zeros\n");

    for (const Accum& a : Claims())
    {
        PrintClaim(a);
    }

    // The fast option's row is a bound on the distance from F_n(x) and not on
    // the direction, and the seed it carries is a correction of the hardware
    // approximation rather than the approximation itself. The audit is what
    // keeps both facts checkable: the contribution is the term the fast bound
    // carries on top of the lane's, and the wrong-sign count is what a seed
    // error the recurrence amplifies does to a value that is itself at that
    // level - the defect this correction removes, and a tripwire for its
    // return.
    if (!ExpAudits().empty())
    {
        std::printf(
            "\n  the single entry's fast region-B exponential, measured against the accurate one\n"
            "  on the same grid at the same multiplier. Its bound's second term is the\n"
            "  contribution below, capped by the recurrence's amplification; the wrong-sign\n"
            "  count is the audit for the return of the defect the correction removes.\n");
        std::printf("  %-32s %-26s %-16s %s\n",
                    "lane",
                    "contribution",
                    "wrong sign",
                    "worst wrong-sign relative error");

        for (const ExpAudit& a : ExpAudits())
        {
            char contribution[32];
            char wrong[24];
            char relative[64];
            std::snprintf(contribution,
                          sizeof(contribution),
                          "%.3g (n=%d, x=%.6g)",
                          a.contributed,
                          a.contributedN,
                          a.contributedX);
            std::snprintf(wrong, sizeof(wrong), "%zu / %zu", a.wrongSign, a.relativeCells);

            if (a.signRelativeN >= 0)
            {
                std::snprintf(relative,
                              sizeof(relative),
                              "%.3g (n=%d, x=%.6g: %.6g for %.6g)",
                              a.signRelative,
                              a.signRelativeN,
                              a.signRelativeX,
                              a.signRelativeGot,
                              a.signRelativeWant);
            } else
            {
                std::snprintf(relative, sizeof(relative), "-");
            }

            std::printf("  %-32s %-26s %-16s %s\n", a.lane.c_str(), contribution, wrong, relative);
        }
    }

    std::size_t cells = 0;
    std::size_t nonDiscriminating = 0;
    std::size_t exceeded = 0;

    for (const Accum& a : Claims())
    {
        cells += a.points;
        nonDiscriminating += a.vacuous;
        exceeded += a.failures;
    }

    if (cells > 0)
    {
        const double asDouble = static_cast<double>(cells);
        const double blind = 100.0 * static_cast<double>(nonDiscriminating) / asDouble;
        const double live = 100.0 * static_cast<double>(cells - nonDiscriminating) / asDouble;
        std::printf("\n  cells: %zu comparison cells across every device entry, of which "
                    "%zu (%.1f%%) carry a bound at least as large as the value itself, so any "
                    "return in range passes there and the cell cannot discriminate; the "
                    "no-cell-over-budget statement above is carried by the remaining %zu "
                    "(%.1f%%).\n",
                    cells,
                    nonDiscriminating,
                    blind,
                    cells - nonDiscriminating,
                    live);
    }

    if (exceeded > 0)
    {
        std::printf("\n  RESULT: FAIL - %zu cells over their documented bound on this device "
                    "(exit status 1)\n",
                    exceeded);
        return 1;
    }

    std::printf("\n  RESULT: every documented device bound met at this revision, %zu of %zu "
                "comparison cells able to discriminate (exit status 0)\n",
                cells - nonDiscriminating,
                cells);
    return 0;
}
