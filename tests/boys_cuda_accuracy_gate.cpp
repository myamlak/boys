// The CUDA device gate: the device lane's documented bounds measured against
// the same independent high-precision reference the CPU gate measures against
// (tests/data/boys_accuracy_gate_reference.csv, re-derivable with
// tools/gen_boys_accuracy_gate_reference.py), over the whole of the lane's
// named domain, in the CPU gate's own row shape.
//
// What it adds to tests/boys_cuda_test.cpp: that test compares the device lane
// with the CPU lane, and the two implementations it subtracts can carry error
// in the same direction, so a difference bounds the device's distance from the
// CPU lane and not its distance from F_n(x). This gate removes the middle term
// and measures every device entry against the CPU gate's committed grid, which
// shares no code with the library, directly.
//
// What is measured, per entry. Every documented device bound is a claim about a
// (lane, region) cell, and the reference grid is rectangular over orders
// 0..kMaxBoysOrder and one shared argument list, so one launch per entry covers
// every cell: the single entries are handed one element per (order, argument)
// cell, and the batch entries the argument list at nmax = kMaxBoysOrder. A cell
// the bound cannot fail on - the bound is at least as large as |F_n(x)| itself,
// so any return in range passes there - is counted in the vacuous column rather
// than folded into the total, which is the CPU gate's rule.
//
// The whole launched surface, entry by entry. A surface is not measured by
// measuring the rows somebody remembered: every launched entry the library's
// report carries (BoysDeviceOptions()) has an arm in the table further down,
// launched over the same grid at that row's own documented bound, and an entry
// the report carries and no arm reaches is named and fails the run. An arm the
// library declares and this revision defines no kernel for is written where it
// belongs and commented out with the reason, because arming it would be an
// unresolved external at link time rather than a measurement; the survey counts
// it as a row no arm reaches, so the gap is a number and not a silence.
//
// A launch that did not happen. Every launch in the arms is read back through
// cudaGetLastError() and a non-success stops the gate naming the entry, because a
// launcher whose kernel the driver never issued returns what a launcher that
// agreed with the reference returns. That read is itself shown able to fail
// before it is trusted: the runtime is asked for a refused launch, and the read
// is required to report it or the run stops.
//
// The card is named rather than assumed: a delivered figure describes the
// arithmetic this device executes, so a weaker or stronger double unit changes
// what a bound costs and not what it is - the bounds transfer between cards and
// the delivered figures do not. Every entry is measured at the one accuracy
// this library has: the batch entries are swept once and the device-callable
// entries once.
//
// One entry carries two certified options rather than one arithmetic: the f32
// single entry's region-B exponential, each option measured against the bound
// that option documents. The two returns are measured against each other as
// well - they differ in that one factor, so their difference is the
// contribution the fast option's second bound term has to cover, and the
// wrong-sign cells are the audit for the defect that term exists because of.
// The term itself is derived from the recurrence's condition number rather than
// read off this sweep; the sweep is what confirms it.
//
// The device-callable entries (boys_cuda_device.hpp) are measured beside the
// batch ones, and through the kernels of tests/boys_cuda_device_demo.cu rather
// than through a host wrapper: that file includes the public device header and
// the CUDA runtime and nothing of this library's implementation, so a row for
// one of these entries is a measurement of what a consumer's own kernel
// reaches. Each of its threads forms its own argument from a factor pair the
// gate chose exact, so the value measured is the reference's own. The entries
// that are one body reached through different shapes are additionally compared
// with each other bit for bit, which is a stronger statement than a bound.
//
// Every device entry is also compared bit for bit with the batch entry of the
// same precision, because the two are one arithmetic reached two ways and no
// bound can say so: the same degree tables, the same inlined body, a lane object
// that reads the caller's handle where the batch kernel reads a __constant__
// symbol. The order and the capacity refusals are exercised beside those
// comparisons.
//
// The fp64 device entries are additionally measured against the 45-digit
// reference grid (tests/data/boys_reference.csv), whose arguments are the region
// boundaries and a logarithmic sweep rather than the gate grid's 1718. That grid
// carries one argument column, so only the double entries can be measured on it:
// a float or fp16 row needs the reference at the rounded argument, and the
// rounded columns are the gate grid's and not this one's. Both references are
// named, with their argument counts, in the report.
//
// Run:  cmake --build <build> --config Release --target boys-cuda-accuracy-gate
//       <build>/Release/boys-cuda-accuracy-gate [--reference <grid.csv>]
//                                                      [--digit-reference <grid.csv>]
// The statuses: 0 when every documented bound is met and every entry the report
// carries is armed; 1 when a cell is over its bound, or when a served option has
// no claim; 2 when a launch or a runtime call failed, or an input could not be
// read; 3 when a launched entry the report carries has no arm, which is a
// coverage gap and not an accuracy figure. A run that measured nothing and one
// that measured everything green do not share a status.

#include "boys/boys.hpp"
#include "boys/boys_cuda.hpp"
#include "boys/boys_impl.hpp" // the region boundaries
#include "boys/f16.hpp"
#include "boys_cuda_device_demo.hpp"
#include "boys_gate_reference.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

// The grid is located as the CPU gate locates it, so a build that bypasses CMake
// still says which revision it measured rather than naming one it did not read.
#ifndef BoysDataDir
#define BoysDataDir "tests/data"
#endif

#ifndef BoysGateRevision
#define BoysGateRevision "unknown"
#endif

// The fp16 device entries and the host launchers that reach them are declared
// under the BoysFp16 build-time seam (include/boys/boys_cuda.hpp), and CMake
// pins that seam ON, so a consumer may build this tree with it closed. The
// format types arrive from boys/f16.hpp either way; the entries do not exist in
// a closed build, so calling one is a compile error rather than a wrong number.
// The lane's cells are therefore measured only where this build carries it, and
// the report names every row it could not measure and why, and says so in its
// RESULT line: a table missing a row reads as a row that was measured.
#if BoysFp16
#define BOYS_CUDA_GATE_FP16 1
#endif

// The same reference reader and row shape the CPU gate reports in, so a lane
// measured here is printed in one vocabulary rather than one apiece.
using namespace boys_gate;

namespace {

// ---------------------------------------------------------------------------
// The documented device bounds, and the option rows they are the bounds of.
//
// The fp32 and fp16 figures, and the rows the device cells below are claimed
// for, are read from boys::BoysDeviceOptions(), the library's own report of its
// device option space. A gate that transcribed them would be a second source of
// truth for the same numbers, so a row added to the surface could be certified
// by nothing and a bound could say one thing to a chooser and another to the
// certifier. The fp64 single lane's per-region cells are the exception and are
// transcribed below, because the report states one figure per option and that
// lane's contract is four cells of one option.
// ---------------------------------------------------------------------------

/// The report's row for an option, by entry and axis member.
///
/// A row this gate asks for and the library does not report stops the gate
/// rather than substituting a figure: a gate that can invent a bound is not a
/// certifier.
const boys::DeviceOptionInfo& DeviceRow(boys::DeviceEntry entry,
                                        boys::RegionBExp exp = boys::RegionBExp::kAccurate) {
    for (const boys::DeviceOptionInfo& option : boys::BoysDeviceOptions())
    {
        if (option.entry == entry && option.regionBExp == exp)
        {
            return option;
        }
    }

    std::fprintf(stderr,
                 "gate: the library reports no device option at entry %d, region-B exponential %d\n",
                 static_cast<int>(entry),
                 static_cast<int>(exp));
    std::abort();
}

// README's accuracy contract: "CUDA fp64 | the same budgets as the CPU double
// lanes". The CPU double single lane is the one with per-region cells (1e-15
// below the region-A edge, 3e-14 through the extended band and region B,
// 5.5e-14 in region C); the CPU double batch lane publishes one, 5.5e-14.
// Three of those cells are transcribed below, because the report states one
// figure per option and this lane's contract is four cells of one option. The
// region-C cell and the batch bound are read from the report, as every fp32 and
// fp16 figure below is.
constexpr double kBoundSingleA = 1e-15;
constexpr double kBoundSingleBand = 3e-14;
constexpr double kBoundSingleB = 3e-14;
const double kBoundSingleC = DeviceRow(boys::DeviceEntry::kSingleF64).bound;
const double kBoundDoubleBatch = DeviceRow(boys::DeviceEntry::kAllOrdersF64).bound;
// README: "CUDA fp32, RegionBExp::kAccurate (the default) | the same budgets
// as the CPU float lanes" - 1.5e-7, the bound of the single entry's accurate
// region-B exponential, of the batch entries and of the all-n entry. Read from
// the report, which is the header's own figure for that option.
const double kBoundFloat = DeviceRow(boys::DeviceEntry::kDeviceSingleF32).bound;
// The single entry's fast region-B exponential carries its own bound: the
// lane's 1.5e-7 plus the corrected seed's own contribution, which the
// recurrence's amplification caps at this figure. The cap is derived, not
// measured cell by cell: the region-B ladder f_l = ((l - 1/2) f_{l-1} - e)/x
// propagates an error in e to order n with gain G_n(x) = sum_k A_n/(x A_k),
// A_m = prod_{j<=m} (j - 1/2)/x - the ratio of the dominant solution of the
// homogeneous recurrence to the wanted one, 7.6e4 at n = 32 at the region-B
// boundary - so a seed whose relative error is flat at rho ulp contributes at
// most G_n (1/2) e^{-x} rho: 6.1e-8 at rho = 4 ulp, 8e-8 here. The sweep below
// reports the contribution it actually measured, and the audit reports the
// wrong-sign cells.
//
// It is read as the difference between the two options' documented bounds: the
// fast option's figure is the lane's plus this contribution, and both figures
// are the report's.
const double kFastExpContribution =
    DeviceRow(boys::DeviceEntry::kDeviceSingleF32Fast, boys::RegionBExp::kFast).bound -
                                    DeviceRow(boys::DeviceEntry::kDeviceSingleF32).bound;
// The fp16 entries' bound is the header's: 1e-7 + 1/2 ULP of the returned
// value, which is what HalfBoundAt computes. Its constant part is read from the
// report's fp16 row, and the coverage check at the end of this file compares it
// against kBoundHalfBase, the figure the shared reference book states: two
// books, one number, and a disagreement stops the gate.
const double kBoundHalfRow = DeviceRow(boys::DeviceEntry::kDeviceSingleF16).bound;

// The smallest positive normal float: below it a returned value is the
// format's floor rather than arithmetic, which is what the relative-error side
// of the fast exponential's audit is restricted to.
constexpr double kFloatMinNormal = std::numeric_limits<float>::min();

// What the fast region-B exponential costs, measured against the accurate one
// over the same grid: the seed substitution's own contribution (max|fast -
// accurate|, which is the term the fast bound carries on top of the lane's),
// and the cells whose return has the opposite sign to the function. The second
// is a tripwire rather than a bound term: a seed error the recurrence amplifies
// is what put the wrong sign there before the correction, so a non-zero count
// means the correction has been lost.
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

// The claim rows this build could not measure because the entries behind them
// sit behind a closed fp16 seam, named the way their own claim is registered.
// Empty in a build that carries the lane, and every reader of it prints nothing
// then.
std::vector<std::string>& NotCarriedLanes() {
    static std::vector<std::string> lanes;
    return lanes;
}

// One unmeasured claim row, by its slot: the name comes from the claim itself,
// so the report cannot drift from the table it is talking about. It exists only
// in a build that has rows it cannot measure, which is the same build that
// calls it.
#ifndef BOYS_CUDA_GATE_FP16
void NoteNotCarried(int slot) {
    NotCarriedLanes().push_back(Claims()[static_cast<std::size_t>(slot)].lane);
}
#endif

// That register as the report prints it: one line naming the seam and the
// reason, then the rows, deduplicated because a row is registered by every
// entry that reaches it.
void PrintNotCarried() {
    if (NotCarriedLanes().empty())
    {
        return;
    }

    std::vector<std::string> printed;

    for (const std::string& lane : NotCarriedLanes())
    {
        if (std::find(printed.begin(), printed.end(), lane) != printed.end())
        {
            continue;
        }

        printed.push_back(lane);
    }

    std::printf("\n  fp16 claim rows this build does not carry, so no bound of theirs is measured "
                "here: their\n  entries are declared behind the BoysFp16 seam, which this build "
                "has closed (BoysFp16 = 0),\n  and the %zu rows below name them. They are left in "
                "the tables unmeasured rather than\n  removed, so the count of rows does not move "
                "with the seam:\n",
                printed.size());

    for (const std::string& lane : printed)
    {
        std::printf("    %s\n", lane.c_str());
    }
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
//
// rho and d2 are the factor pair a fused kernel multiplies to form its own
// argument: the device entries take x where it is formed, in a register, so the
// gate hands the demo kernels a product rather than the argument itself. rho is
// a power of two, so x / rho is a shift and the product is the grid's own
// argument to the last bit; a general density factor would only move which
// argument is measured.
struct Grid {
    std::vector<int> n;
    std::vector<double> x;
    std::vector<boys::F16> x16;
    std::vector<double> rho;
    std::vector<double> d2;
    std::size_t cells = 0;
};

Grid MakeGrid(const Reference& ref) {
    Grid grid;
    const int nmax = boys::kMaxBoysOrder;
    grid.cells = ref.count * static_cast<std::size_t>(nmax + 1);
    grid.n.resize(grid.cells);
    grid.x.resize(grid.cells);
    grid.x16.resize(grid.cells);
    grid.rho.resize(grid.cells);
    grid.d2.resize(grid.cells);

    for (int n = 0; n <= nmax; ++n)
    {
        for (std::size_t i = 0; i < ref.count; ++i)
        {
            const std::size_t e = ref.Index(n, i);
            grid.n[e] = n;
            grid.x[e] = ref.x[i];
            grid.x16[e] = boys::F16(static_cast<float>(ref.x[i]));
            // A different power of two per argument, so the product is not
            // accidentally the identity at every thread.
            const double rho = std::ldexp(1.0, static_cast<int>(i % 9) - 4);
            const double d2 = ref.x[i] / rho;
            grid.rho[e] = rho;
            grid.d2[e] = d2;

            if (rho * d2 != ref.x[i])
            {
                std::fprintf(stderr,
                             "cuda gate: the factor pair is not exact at i=%zu: %.17g * %.17g"
                             " != %.17g\n",
                             i,
                             rho,
                             d2,
                             ref.x[i]);
                std::exit(2);
            }
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

// The name a claim is registered under, which is the library's own name for
// the row: the coverage check at the end of this file reads every claim by it,
// so a claim this gate invented would be a claim for no reported option.
std::string Label(const char* lane) {
    return std::string(lane);
}

// The same, on a phrase rather than on a lane name.
std::string ShapeLabel(const char* what) {
    return std::string(what);
}

// A fit route's own name, read from the library's table rather than
// transcribed: a route added to that table is named here the day it lands.
const char* RouteName(boys::FitRoute route) {
    for (const boys::FitRouteInfo& tableRow : boys::BoysFitRoutes()) {
        if (tableRow.route == route) {
            return tableRow.name;
        }
    }

    return "unnamed route";
}

// The committed 45-digit grid (tests/data/boys_reference.csv), the reference the
// CPU tests call definitive: F_n at the double its x column states, to 45 digits.
// Its arguments are the region boundaries and a logarithmic sweep rather than
// the gate grid's 1718, so it is also where the fp64 entries are pointed at the
// boundaries. One argument column means only an fp64 entry can be measured on
// it: a float or fp16 row needs the reference at the argument the entry actually
// evaluated, and the rounded columns belong to the gate grid.
struct DigitGrid {
    std::vector<int> n;
    std::vector<double> x;
    std::vector<double> v;
    std::vector<int> decade;
    std::size_t count = 0;
    std::size_t cells = 0;
};

DigitGrid LoadDigitGrid(const std::string& path) {
    std::ifstream in(path);

    if (!in)
    {
        std::fprintf(stderr, "cuda gate: cannot open the 45-digit grid: %s\n", path.c_str());
        std::exit(2);
    }

    DigitGrid grid;
    std::string line;
    std::getline(in, line); // header

    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }

        const std::vector<std::string> f = Split(line);

        if (f.size() < 3)
        {
            continue;
        }

        const double value = std::strtod(f[2].c_str(), nullptr);
        grid.n.push_back(std::atoi(f[0].c_str()));
        grid.x.push_back(std::strtod(f[1].c_str(), nullptr));
        grid.v.push_back(value);
        grid.decade.push_back(value == 0.0 ? 0
                                           : static_cast<int>(
                                                 std::floor(std::log10(std::abs(value)))));
    }

    grid.cells = grid.n.size();

    if (grid.cells == 0)
    {
        std::fprintf(stderr, "cuda gate: the 45-digit grid is empty: %s\n", path.c_str());
        std::exit(2);
    }

    // The rows are order-major with one shared argument list, which is what lets
    // the element index be the reference's own: the first block defines the
    // arguments and every later block repeats them.
    std::size_t block = 0;

    while (block < grid.cells && grid.n[block] == 0)
    {
        ++block;
    }

    if (block == 0)
    {
        std::fprintf(stderr, "cuda gate: the 45-digit grid has no order-0 block: %s\n",
                     path.c_str());
        std::exit(2);
    }

    grid.count = block;

    if (grid.count * static_cast<std::size_t>(boys::kMaxBoysOrder + 1) != grid.cells)
    {
        std::fprintf(stderr,
                     "cuda gate: the 45-digit grid is not %d orders x %zu arguments: %zu rows\n",
                     boys::kMaxBoysOrder + 1,
                     grid.count,
                     grid.cells);
        std::exit(2);
    }

    for (std::size_t k = 0; k < grid.cells; ++k)
    {
        if (grid.n[k] != static_cast<int>(k / grid.count) || grid.x[k] != grid.x[k % grid.count])
        {
            std::fprintf(stderr, "cuda gate: the 45-digit grid is not rectangular at row %zu\n", k);
            std::exit(2);
        }
    }

    return grid;
}

// ---------------------------------------------------------------------------
// The sweeps, one per precision family, each covering all three shapes.
// ---------------------------------------------------------------------------

// The fp16 bound: the header's 1e-7 + 1/2 ULP, in ULP of the value the entry
// returned rather than of the value it should have.
double HalfBoundAt(double got) {
    return kBoundHalfRow + 0.5 * UlpOf(got, kF16MantissaBits, kF16MinNormalExp);
}

// A float lane's return, widened for the comparison against a double reference,
// with the flag that says the value is the format's floor rather than
// arithmetic: zero, or below the smallest normal float.
std::pair<double, bool> Widen(float got) {
    const double asDouble = static_cast<double>(got);
    return std::make_pair(asDouble,
                          got == 0.0f ||
                              std::fabs(asDouble) < std::numeric_limits<float>::min());
}

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
            boys::BoysCuda::SingleF64(dN.get(), dX.get(), dOut.get(), cells, nullptr),
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
        CheckLaunch(boys::BoysCuda::AllOrdersF64(
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
                        kBoundDoubleBatch,
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
            boys::BoysCuda::AllNF64(nmax, dX.get(), dOut.get(), count, nullptr),
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
                        kBoundDoubleBatch,
                        Unrepresentable(out[e], -1022));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The rows the lane gained when its narrow partition and its orders axis
// stopped being refusals.
// ---------------------------------------------------------------------------

// The division form this gate's device rows are launched at. Every entry of the
// surface takes the form as a trailing argument and the option space crosses each
// of them with all three, but a bound is stated for a lane and a region and not
// for a form (there is no per-form figure in BoysLaneContracts()), so the rows
// below are measured at the arithmetic the library's own default names - the one
// every published figure was measured at. Crossing the forms here would be a
// second option space beside the probe's, and this gate is not that instrument.
constexpr boys::DivisionForm kGateDivisionForm = boys::kDefaultDivisionForm;

// One of those rows, launched once over the whole grid, measured against the
// independent reference as every other row of this gate is. The returns are
// handed back so the same launch can be measured against the host lane as
// well, which is one launch and not three.
std::vector<double> LaunchDeviceChoice(const Reference& ref,
                                       const Grid& grid,
                                       const char* what,
                                       boys::BoysStatus (*launch)(const int*,
                                                                  const double*,
                                                                  double*,
                                                                  std::size_t,
                                                                  void*,
                                                                  boys::DivisionForm),
                                       int slotReference) {
    const std::size_t count = ref.count;
    const std::size_t cells = grid.cells;
    const double bound = kBoundDoubleBatch;

    DevBuf<int> dN(count);
    DevBuf<double> dX(count);
    DevBuf<double> dOut(cells);
    const std::vector<int> tops(count, boys::kMaxBoysOrder);
    dN.Upload(tops);
    dX.Upload(grid.x);
    CheckLaunch(launch(dN.get(), dX.get(), dOut.get(), count, nullptr, kGateDivisionForm), what);
    std::vector<double> out(cells);
    dOut.Download(out);
    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t e = ref.Index(n, i);
            Measure(slotReference,
                    n,
                    ref.x[i],
                    out[e],
                    ref.v[e],
                    ref.decade[e],
                    bound,
                    Unrepresentable(out[e], -1022));
        }
    }

    return out;
}

// The other half of the claim: the two lanes are one question answered twice
// rather than two rows that agree with a third party. The host lane is handed
// the same arguments and its returns are the reference
// the device row's are read against, so what is measured is the distance
// between the two lanes. The host entry is the CPU spelling of this lane's
// batch (boys.hpp says so), so the two calls are one question with two answers.
//
// The distance is held to the sum of the two rows' bounds and not to one of
// them. A row's bound says it is that far from the true value; two rows each
// that far from the true value can be twice that far from each other, so the
// sum is what a comparison between two bounded rows can assert, and one row's
// bound is not. The sum is a statement about the pair, and it neither loosens
// nor tightens either row's own figure: those stay at the single bound, and are
// measured against the 45-digit reference above.
template <typename HostPolicy>
void CompareDeviceWithHost(const Reference& ref,
                           const Grid& grid,
                           const std::vector<double>& out,
                           int slotHost) {
    const std::size_t count = ref.count;
    const std::size_t cells = grid.cells;
    const double bound = 2.0 * kBoundDoubleBatch;
    const std::vector<int> tops(count, boys::kMaxBoysOrder);

    std::vector<double> host(cells, 0.0);
    boys::BoysAllNAtOrders<HostPolicy>(
        tops.data(), grid.x.data(), host.data(), count);

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t e = ref.Index(n, i);
            Measure(slotHost, n, ref.x[i], out[e], host[e], ref.decade[e], bound, false);
        }
    }
}

// The seven rows, each measured against the reference and against the host lane.
//
// Two host comparisons and not one. The lane's shipped batch entry is the host
// entry that answers the same question, so it is what the device row is read
// against; the row is additionally shown against the host's combination of the
// same name - the same route, scheme, partition and packing axis - which is the
// counterpart that says the two lanes agree about the option and not merely
// about the function.
//
// A cross-lane claim is registered under the row's own name followed by the
// phrase naming what it was compared with, because the device-option coverage
// below reads a claim as belonging to an option when it is that option's name
// or that name followed by a phrase, and a cross-lane claim is not the row
// itself.
//
// The entry and the launcher are named together and once, in MeasureDeviceRow:
// the row's claim takes its name from the report's own row for that entry, and
// the launcher is that entry's kernel, so a claim cannot name a row the library
// does not report or an entry no kernel serves.
std::vector<double> MeasureDeviceRow(const Reference& ref,
                                     const Grid& grid,
                                     boys::DeviceEntry entry,
                                     boys::BoysStatus (*launch)(const int*,
                                                                const double*,
                                                                double*,
                                                                std::size_t,
                                                                void*,
                                                                boys::DivisionForm)) {
    const std::string name = Label(DeviceRow(entry).name);
    const double bound = kBoundDoubleBatch;
    const int row = AddClaim(name.c_str(), "A..C", bound);
    const int shipped = AddClaim(
        (std::string(DeviceRow(entry).name) + " vs fp64 host").c_str(),
        "A..C",
        2.0 * bound);

    const std::vector<double> out =
        LaunchDeviceChoice(ref, grid, name.c_str(), launch, row);

    CompareDeviceWithHost<boys::DefaultPolicyFp64>(ref, grid, out, shipped);

    return out;
}

// The single-precision launched rows, measured the way this lane's other float
// entries are: against the reference's own float column and at the float lane's
// bound, which is the figure the row documents. The fp64 sibling above cannot
// certify one of these — its bound is the double batch lane's and its comparison
// is against the fp64 host lane — and a row measured at another precision's bar
// is a claim about arithmetic that row does not run.
//
// One claim, the row's own: the name comes from the report's row for the entry,
// so a name this gate invented could not pass the coverage check at the end of
// this file, and a row the report carries is either certified here or counted
// there.
void MeasureDeviceRowF32(const Reference& ref,
                         const Grid& grid,
                         boys::DeviceEntry entry,
                         boys::BoysStatus (*launch)(const int*,
                                                    const double*,
                                                    float*,
                                                    std::size_t,
                                                    void*,
                                                    boys::DivisionForm)) {
    const std::string name = Label(DeviceRow(entry).name);
    const double bound = kBoundFloat;
    const int row = AddClaim(name.c_str(), "A..C", bound);
    const std::size_t count = ref.count;
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t cells = grid.cells;

    // The launch is the fp64 sibling's, one lane down: `count` arguments and a
    // ladder for each, so the output holds `count * (kMaxBoysOrder + 1)` values.
    // The cell count is the size of that output and is not the batch size — a
    // launch handed the cells as its count would ask every element for a whole
    // ladder over a buffer sized for one value each, which is the out-of-bounds
    // write this helper originally made.
    DevBuf<int> dN(count);
    DevBuf<double> dX(count);
    DevBuf<float> dOut(cells);
    const std::vector<int> tops(count, boys::kMaxBoysOrder);
    dN.Upload(tops);
    dX.Upload(grid.x);
    CheckLaunch(launch(dN.get(), dX.get(), dOut.get(), count, nullptr, kGateDivisionForm),
                name.c_str());
    std::vector<float> out(cells);
    dOut.Download(out);
    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    for (int n = 0; n <= nmax; ++n)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t e = ref.Index(n, i);
            const auto [got, unrepresentable] = Widen(out[e]);

            Measure(row, n, ref.xf[i], got, ref.vf[e], ref.decadeF[e], bound, unrepresentable);
        }
    }
}

#ifdef BOYS_CUDA_GATE_FP16
// The half-precision launched rows, measured the way this lane's other fp16
// entries are: against the reference at the argument the entry actually
// evaluated, at the header's own bound - 1e-7 plus half of the last
// representable digit of the value the entry returned - and with the format's
// floor counted apart rather than judged. The fp32 sibling above cannot certify
// one of these: its bound is the float lane's and its reference is the float
// column, and a row measured at another precision's bar is a claim about
// arithmetic that row does not run.
//
// One claim, the row's own: the name comes from the report's row for the entry,
// so a name this gate invented could not pass the coverage check at the end of
// this file, and a row the report carries is either certified here or counted
// there.
void MeasureDeviceRowF16(const Reference& ref,
                         const Grid& grid,
                         boys::DeviceEntry entry,
                         boys::BoysStatus (*launch)(const int*,
                                                    const boys::F16*,
                                                    boys::F16*,
                                                    std::size_t,
                                                    void*,
                                                    boys::DivisionForm)) {
    const std::string name = Label(DeviceRow(entry).name);
    const int row = AddClaim(name.c_str(), "A..C", kBoundHalfRow);
    const std::size_t count = ref.count;
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t cells = grid.cells;

    // The launch is the fp64 sibling's, one lane down: `count` arguments and a
    // ladder for each, so the output holds `count * (kMaxBoysOrder + 1)` values.
    // The count handed to the entry is the number of arguments and not the size
    // of that output, which is the same distinction the fp32 sibling's comment
    // makes.
    DevBuf<int> dN(count);
    DevBuf<boys::F16> dX(count);
    DevBuf<boys::F16> dOut(cells);
    const std::vector<int> tops(count, nmax);
    std::vector<boys::F16> hostX(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        hostX[i] = boys::F16(static_cast<float>(ref.x[i]));
    }

    dN.Upload(tops);
    dX.Upload(hostX);
    CheckLaunch(launch(dN.get(), dX.get(), dOut.get(), count, nullptr, kGateDivisionForm),
                name.c_str());
    std::vector<boys::F16> out(cells);
    dOut.Download(out);
    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    for (int n = 0; n <= nmax; ++n)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t e = ref.Index(n, i);
            const double got = static_cast<double>(out[e]);

            Measure(row,
                    n,
                    ref.x16[i],
                    got,
                    ref.v16[e],
                    ref.decade16[e],
                    HalfBoundAt(got),
                    Unrepresentable(got, kF16MinNormalExp));
        }
    }
}
#endif // BOYS_CUDA_GATE_FP16

void SweepDeviceChoices(const Reference& ref, const Grid& grid) {
    const double pairBound = 2.0 * kBoundDoubleBatch;

    using NarrowPolicy = boys::EvalPolicy<boys::kDefaultFitRoute,
                                          boys::kDefaultEvalScheme,
                                          boys::BoysBudget::kFloat,
                                          boys::PackAxis::kArguments,
                                          boys::FitGranularity::kNarrow>;
    using OrdersPolicy = boys::EvalPolicy<boys::kDefaultFitRoute,
                                          boys::kDefaultEvalScheme,
                                          boys::BoysBudget::kFloat,
                                          boys::PackAxis::kOrders,
                                          boys::FitGranularity::kCoarsest>;
    using NarrowOrdersPolicy = boys::EvalPolicy<boys::kDefaultFitRoute,
                                                boys::kDefaultEvalScheme,
                                                boys::BoysBudget::kFloat,
                                                boys::PackAxis::kOrders,
                                                boys::FitGranularity::kNarrow>;

    // The same partition and packing axes over the monomial basis: the scheme
    // axis names which of the lane's two stored forms a body sums, so a row of
    // it is read against the host entry that sums the other one.
    using MonoPolicy = boys::EvalPolicy<boys::kDefaultFitRoute,
                                        boys::EvalScheme::kHorner,
                                        boys::BoysBudget::kFloat,
                                        boys::PackAxis::kArguments,
                                        boys::FitGranularity::kCoarsest>;
    using OrdersMonoPolicy = boys::EvalPolicy<boys::kDefaultFitRoute,
                                              boys::EvalScheme::kHorner,
                                              boys::BoysBudget::kFloat,
                                              boys::PackAxis::kOrders,
                                              boys::FitGranularity::kCoarsest>;
    using NarrowMonoPolicy = boys::EvalPolicy<boys::kDefaultFitRoute,
                                              boys::EvalScheme::kHorner,
                                              boys::BoysBudget::kFloat,
                                              boys::PackAxis::kArguments,
                                              boys::FitGranularity::kNarrow>;
    using NarrowOrdersMonoPolicy = boys::EvalPolicy<boys::kDefaultFitRoute,
                                                    boys::EvalScheme::kHorner,
                                                    boys::BoysBudget::kFloat,
                                                    boys::PackAxis::kOrders,
                                                    boys::FitGranularity::kNarrow>;

    // The fit route's rows, in the same four shapes. The route's two scheme
    // names select one arithmetic, so the four policies below are the shapes'
    // four and not eight: a row of the route is read against the host lane at
    // the route, the shape's partition and the shape's packing axis.
    using RatPolicy = boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                                       boys::kDefaultEvalScheme,
                                       boys::BoysBudget::kFloat,
                                       boys::PackAxis::kArguments,
                                       boys::FitGranularity::kCoarsest>;
    using OrdersRatPolicy = boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                                             boys::kDefaultEvalScheme,
                                             boys::BoysBudget::kFloat,
                                             boys::PackAxis::kOrders,
                                             boys::FitGranularity::kCoarsest>;
    using NarrowRatPolicy = boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                                             boys::kDefaultEvalScheme,
                                             boys::BoysBudget::kFloat,
                                             boys::PackAxis::kArguments,
                                             boys::FitGranularity::kNarrow>;
    using NarrowOrdersRatPolicy = boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                                                   boys::kDefaultEvalScheme,
                                                   boys::BoysBudget::kFloat,
                                                   boys::PackAxis::kOrders,
                                                   boys::FitGranularity::kNarrow>;

    const std::vector<double> narrowOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64Narrow,
        &boys::BoysCuda::AllOrdersF64Narrow);
    const std::vector<double> ordersOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64Orders,
        &boys::BoysCuda::AllOrdersF64Orders);
    const std::vector<double> bothOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64NarrowOrders,
        &boys::BoysCuda::AllOrdersF64NarrowOrders);
    const std::vector<double> monoOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64Mono,
        &boys::BoysCuda::AllOrdersF64Mono);
    const std::vector<double> ordersMonoOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64OrdersMono,
        &boys::BoysCuda::AllOrdersF64OrdersMono);
    const std::vector<double> narrowMonoOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64NarrowMono,
        &boys::BoysCuda::AllOrdersF64NarrowMono);
    const std::vector<double> bothMonoOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64NarrowOrdersMono,
        &boys::BoysCuda::AllOrdersF64NarrowOrdersMono);
    // The fit route's rows. Both rows of a scheme pair are measured and each
    // carries its own figure; a pair whose figures differ would be a report that
    // named two arithmetics where the lane has one.
    const std::vector<double> ratOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64Rat,
        &boys::BoysCuda::AllOrdersF64Rat);
    const std::vector<double> ratHornerOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64RatHorner,
        &boys::BoysCuda::AllOrdersF64Rat);
    const std::vector<double> ordersRatOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64OrdersRat,
        &boys::BoysCuda::AllOrdersF64OrdersRat);
    const std::vector<double> ordersRatHornerOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64OrdersRatHorner,
        &boys::BoysCuda::AllOrdersF64OrdersRat);
    const std::vector<double> narrowRatOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64NarrowRat,
        &boys::BoysCuda::AllOrdersF64NarrowRat);
    const std::vector<double> narrowRatHornerOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64NarrowRatHorner,
        &boys::BoysCuda::AllOrdersF64NarrowRat);
    const std::vector<double> bothRatOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64NarrowOrdersRat,
        &boys::BoysCuda::AllOrdersF64NarrowOrdersRat);
    const std::vector<double> bothRatHornerOut = MeasureDeviceRow(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner,
        &boys::BoysCuda::AllOrdersF64NarrowOrdersRat);

    // The uniform route's rows. Its table is stored at one degree for every
    // order and every interval, so no criterion cuts it: the arithmetic is that
    // one degree and the row is the figure it documents. Every row of the route
    // is measured, the two packing-axis rows included.
    {
        const std::vector<double> uniformOut = MeasureDeviceRow(
            ref,
            grid,
            boys::DeviceEntry::kAllOrdersF64Uniform,
            &boys::BoysCuda::AllOrdersF64Uniform);
        const std::vector<double> uniformHornerOut = MeasureDeviceRow(
            ref,
            grid,
            boys::DeviceEntry::kAllOrdersF64UniformHorner,
            &boys::BoysCuda::AllOrdersF64UniformHorner);

            // The same grid on the rational route, the double lane's member.
            MeasureDeviceRow(
                ref,
                grid,
                boys::DeviceEntry::kAllOrdersF64UniformRat,
                &boys::BoysCuda::AllOrdersF64UniformRat);
            MeasureDeviceRow(
                ref,
                grid,
                boys::DeviceEntry::kAllOrdersF64UniformRatHorner,
                &boys::BoysCuda::AllOrdersF64UniformRatHorner);
            MeasureDeviceRow(
                ref,
                grid,
                boys::DeviceEntry::kAllOrdersF64OrdersUniformRat,
                &boys::BoysCuda::AllOrdersF64UniformRat);
            MeasureDeviceRow(
                ref,
                grid,
                boys::DeviceEntry::kAllOrdersF64OrdersUniformRatHorner,
                &boys::BoysCuda::AllOrdersF64UniformRatHorner);
        const std::vector<double> ordersUniformOut = MeasureDeviceRow(
            ref,
            grid,
            boys::DeviceEntry::kAllOrdersF64OrdersUniform,
            &boys::BoysCuda::AllOrdersF64OrdersUniform);
        const std::vector<double> ordersUniformHornerOut = MeasureDeviceRow(
            ref,
            grid,
            boys::DeviceEntry::kAllOrdersF64OrdersUniformHorner,
            &boys::BoysCuda::AllOrdersF64OrdersUniformHorner);
        (void)uniformOut;
        (void)uniformHornerOut;
        (void)ordersUniformOut;
        (void)ordersUniformHornerOut;
    }

    // The float lane's grid, the uniform route at that lane's precision.
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32Uniform,
        &boys::BoysCuda::AllOrdersF32Uniform);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32UniformHorner,
        &boys::BoysCuda::AllOrdersF32UniformHorner);

    // The grid on its rational route, the float lane's member over its own
    // intervals: one pair per interval and no per-order effective-degree column,
    // so nothing to cut.
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32UniformRat,
        &boys::BoysCuda::AllOrdersF32UniformRat);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32UniformRatHorner,
        &boys::BoysCuda::AllOrdersF32UniformRatHorner);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32OrdersUniformRat,
        &boys::BoysCuda::AllOrdersF32UniformRat);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32OrdersUniformRatHorner,
        &boys::BoysCuda::AllOrdersF32UniformRatHorner);

    // The float lane's other packing axis, on the two tables this lane holds:
    // the shipped partition's own cut of the float Chebyshev table, and the
    // uniform grid, whose one degree per interval no criterion cuts. Both are
    // measured at the bound the float lane documents.
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32Orders,
        &boys::BoysCuda::AllOrdersF32Orders);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32OrdersUniform,
        &boys::BoysCuda::AllOrdersF32OrdersUniform);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32OrdersUniformHorner,
        &boys::BoysCuda::AllOrdersF32OrdersUniformHorner);

    // The float lane's narrow partition, in each of the two forms it is stored
    // in, each form's cut being the float lane's own region-B degrees in that
    // form beside the double lane's region-A cut.
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32Narrow,
        &boys::BoysCuda::AllOrdersF32Narrow);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32NarrowMono,
        &boys::BoysCuda::AllOrdersF32NarrowMono);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32NarrowOrders,
        &boys::BoysCuda::AllOrdersF32NarrowOrders);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32NarrowOrdersMono,
        &boys::BoysCuda::AllOrdersF32NarrowOrdersMono);

    // The float lane's rational route on both axes and both partitions: the
    // region-B pair is that lane's own fit, cut by the same criterion over the
    // coefficients the row sums, and region A's seed is the double lane's pair
    // at the same cut. Each of its rows is named twice because the route's pair
    // is stored in one form: the two scheme names of a partition reach one
    // kernel, and the row's own scheme says which name reached it.
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32Rat,
        &boys::BoysCuda::AllOrdersF32Rat);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32RatHorner,
        &boys::BoysCuda::AllOrdersF32RatHorner);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32NarrowRat,
        &boys::BoysCuda::AllOrdersF32NarrowRat);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32NarrowRatHorner,
        &boys::BoysCuda::AllOrdersF32NarrowRatHorner);
    // The same tables and route on the packing axis's other side: the cut this
    // row reads is the per-argument row's own.
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32OrdersRat,
        &boys::BoysCuda::AllOrdersF32OrdersRat);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32OrdersRatHorner,
        &boys::BoysCuda::AllOrdersF32OrdersRatHorner);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32NarrowOrdersRat,
        &boys::BoysCuda::AllOrdersF32NarrowOrdersRat);
    MeasureDeviceRowF32(
        ref,
        grid,
        boys::DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner,
        &boys::BoysCuda::AllOrdersF32NarrowOrdersRatHorner);

    // The two lanes' rows of the same name, read against each other: the
    // device row's distance from the host lane's own combination - the same
    // route, scheme, partition and packing axis - which is the counterpart that
    // says the two lanes agree about the option and not merely about the
    // function.
    {
        const int narrowHost = AddClaim(
            (std::string(DeviceRow(boys::DeviceEntry::kAllOrdersF64Narrow).name) + " vs fp64 host narrow")
                .c_str(),
            "A..C",
            pairBound);
        const int ordersHost = AddClaim(
            (std::string(DeviceRow(boys::DeviceEntry::kAllOrdersF64Orders).name) + " vs fp64 host orders")
                .c_str(),
            "A..C",
            pairBound);
        const int bothHost = AddClaim(
            (std::string(DeviceRow(boys::DeviceEntry::kAllOrdersF64NarrowOrders).name) + " vs fp64 host narrow orders")
                .c_str(),
            "A..C",
            pairBound);
        const int monoHost = AddClaim(
            (std::string(DeviceRow(boys::DeviceEntry::kAllOrdersF64Mono).name) + " vs fp64 host mono")
                .c_str(),
            "A..C",
            pairBound);
        const int ordersMonoHost = AddClaim(
            (std::string(DeviceRow(boys::DeviceEntry::kAllOrdersF64OrdersMono).name) + " vs fp64 host orders mono")
                .c_str(),
            "A..C",
            pairBound);
        const int narrowMonoHost = AddClaim(
            (std::string(DeviceRow(boys::DeviceEntry::kAllOrdersF64NarrowMono).name) + " vs fp64 host narrow mono")
                .c_str(),
            "A..C",
            pairBound);
        const int bothMonoHost = AddClaim(
            (std::string(DeviceRow(boys::DeviceEntry::kAllOrdersF64NarrowOrdersMono).name)
             + " vs fp64 host narrow orders mono")
                .c_str(),
            "A..C",
            pairBound);

        CompareDeviceWithHost<NarrowPolicy>(ref, grid, narrowOut, narrowHost);
        CompareDeviceWithHost<OrdersPolicy>(ref, grid, ordersOut, ordersHost);
        CompareDeviceWithHost<NarrowOrdersPolicy>(ref, grid, bothOut, bothHost);
        CompareDeviceWithHost<MonoPolicy>(ref, grid, monoOut, monoHost);
        CompareDeviceWithHost<OrdersMonoPolicy>(
            ref, grid, ordersMonoOut, ordersMonoHost);
        CompareDeviceWithHost<NarrowMonoPolicy>(
            ref, grid, narrowMonoOut, narrowMonoHost);
        CompareDeviceWithHost<NarrowOrdersMonoPolicy>(
            ref, grid, bothMonoOut, bothMonoHost);

        // The fit route's rows, each shape's two scheme names measured against the
        // host lane at the route and that shape's partition and packing axis.
        // Both names of a shape run one kernel, so the two distances are two
        // measurements of one arithmetic; each is recorded per row, because a
        // row's claim is the row's own.
        const auto routeHostClaim = [&](const char* rowName, const char* shape) {
            return AddClaim((std::string(rowName) + " vs fp64 host " + shape).c_str(),
                            "A..C",
                            pairBound);
        };

        const int ratHost = routeHostClaim(
            DeviceRow(boys::DeviceEntry::kAllOrdersF64Rat).name, "rat");
        const int ratHornerHost = routeHostClaim(
            DeviceRow(boys::DeviceEntry::kAllOrdersF64RatHorner).name, "rat");
        const int ordersRatHost = routeHostClaim(
            DeviceRow(boys::DeviceEntry::kAllOrdersF64OrdersRat).name, "orders rat");
        const int ordersRatHornerHost = routeHostClaim(
            DeviceRow(boys::DeviceEntry::kAllOrdersF64OrdersRatHorner).name, "orders rat");
        const int narrowRatHost = routeHostClaim(
            DeviceRow(boys::DeviceEntry::kAllOrdersF64NarrowRat).name, "narrow rat");
        const int narrowRatHornerHost = routeHostClaim(
            DeviceRow(boys::DeviceEntry::kAllOrdersF64NarrowRatHorner).name, "narrow rat");
        CompareDeviceWithHost<RatPolicy>(ref, grid, ratOut, ratHost);
        CompareDeviceWithHost<RatPolicy>(ref, grid, ratHornerOut, ratHornerHost);
        CompareDeviceWithHost<OrdersRatPolicy>(
            ref, grid, ordersRatOut, ordersRatHost);
        CompareDeviceWithHost<OrdersRatPolicy>(
            ref, grid, ordersRatHornerOut, ordersRatHornerHost);
        CompareDeviceWithHost<NarrowRatPolicy>(
            ref, grid, narrowRatOut, narrowRatHost);
        CompareDeviceWithHost<NarrowRatPolicy>(
            ref, grid, narrowRatHornerOut, narrowRatHornerHost);

        // The narrow partition under the rational route on the orders axis, the
        // last of the route's four shapes. The host lane carries it: its packed
        // entry takes the route and the partition as template arguments, and
        // boys_orders_simd.cpp instantiates that pair for both schemes
        // (BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS). The rows are read
        // against the host lane at their own policy like every other shape of
        // the route.
        const int bothRatHost = routeHostClaim(
            DeviceRow(boys::DeviceEntry::kAllOrdersF64NarrowOrdersRat).name, "narrow orders rat");
        const int bothRatHornerHost = routeHostClaim(
            DeviceRow(boys::DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner).name,
            "narrow orders rat");
        CompareDeviceWithHost<NarrowOrdersRatPolicy>(
            ref, grid, bothRatOut, bothRatHost);
        CompareDeviceWithHost<NarrowOrdersRatPolicy>(
            ref, grid, bothRatHornerOut, bothRatHornerHost);
    }
}

// The half lane's partition, route and packing axes, measured row by row the way
// the float lane's are. The names are the report's, so the two sweeps differ in
// the format and in nothing else: every member of the half device lane's option
// space is named here, and a member added to that surface is certified by
// nothing until its row appears below.
void SweepHalfChoices(const Reference& ref, const Grid& grid) {
#ifdef BOYS_CUDA_GATE_FP16
    // The shipped partition's two other shapes: the narrow pieces, in the
    // Clenshaw basis and in the monomial basis the Horner scheme name sums.
    MeasureDeviceRowF16(
        ref, grid, boys::DeviceEntry::kAllOrdersF16Narrow, &boys::BoysCuda::AllOrdersF16Narrow);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16NarrowMono,
                        &boys::BoysCuda::AllOrdersF16NarrowMono);
    // The uniform grid, its two scheme names, and the route's pair over it.
    MeasureDeviceRowF16(
        ref, grid, boys::DeviceEntry::kAllOrdersF16Uniform, &boys::BoysCuda::AllOrdersF16Uniform);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16UniformHorner,
                        &boys::BoysCuda::AllOrdersF16UniformHorner);
    MeasureDeviceRowF16(
        ref, grid, boys::DeviceEntry::kAllOrdersF16Rat, &boys::BoysCuda::AllOrdersF16Rat);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16RatHorner,
                        &boys::BoysCuda::AllOrdersF16RatHorner);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16NarrowRat,
                        &boys::BoysCuda::AllOrdersF16NarrowRat);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16NarrowRatHorner,
                        &boys::BoysCuda::AllOrdersF16NarrowRatHorner);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16UniformRat,
                        &boys::BoysCuda::AllOrdersF16UniformRat);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16UniformRatHorner,
                        &boys::BoysCuda::AllOrdersF16UniformRatHorner);
    // The same tables and route on the packing axis's other side: the cut this
    // row reads is the per-argument row's own.
    MeasureDeviceRowF16(
        ref, grid, boys::DeviceEntry::kAllOrdersF16Orders, &boys::BoysCuda::AllOrdersF16Orders);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16NarrowOrders,
                        &boys::BoysCuda::AllOrdersF16NarrowOrders);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16NarrowOrdersMono,
                        &boys::BoysCuda::AllOrdersF16NarrowOrdersMono);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16OrdersRat,
                        &boys::BoysCuda::AllOrdersF16OrdersRat);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16OrdersRatHorner,
                        &boys::BoysCuda::AllOrdersF16OrdersRatHorner);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16NarrowOrdersRat,
                        &boys::BoysCuda::AllOrdersF16NarrowOrdersRat);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16NarrowOrdersRatHorner,
                        &boys::BoysCuda::AllOrdersF16NarrowOrdersRatHorner);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16OrdersUniform,
                        &boys::BoysCuda::AllOrdersF16OrdersUniform);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16OrdersUniformHorner,
                        &boys::BoysCuda::AllOrdersF16OrdersUniformHorner);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16OrdersUniformRat,
                        &boys::BoysCuda::AllOrdersF16OrdersUniformRat);
    MeasureDeviceRowF16(ref,
                        grid,
                        boys::DeviceEntry::kAllOrdersF16OrdersUniformRatHorner,
                        &boys::BoysCuda::AllOrdersF16OrdersUniformRatHorner);
#else
    // No entry to call: this build's fp16 seam is closed, and every fp16 device
    // option is reported unbuilt, so the coverage check at the end of this file
    // skips them rather than counting them against a surface this build has not
    // got.
    (void)ref;
    (void)grid;
#endif // BOYS_CUDA_GATE_FP16
}

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
    const double bound = kBoundFloat;
    const double fastBound = kBoundFloat + kFastExpContribution;

    // The entry the caller hands a double and the kernel narrows itself: the
    // argument measured at is the float it evaluates at, which is the
    // reference's own xf column.
    const auto narrow = [](const float* got) { return Widen(*got); };

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
            boys::BoysCuda::SingleF32(dN.get(), dX.get(), dOut.get(), cells, nullptr),
            "SingleF32");
        CheckLaunch(boys::BoysCuda::SingleF32<boys::RegionBExp::kFast>(
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
                // speak of, and the bound's own floor covers those cells.
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
        CheckLaunch(boys::BoysCuda::AllOrdersF32(
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
            boys::BoysCuda::AllNF32(nmax, dX.get(), dOut.get(), count, nullptr),
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

void SweepHalf(const Reference& ref,
               const Grid& grid,
               const SortedArgs& sorted,
               int slotSingle,
               int slotOrders,
               int slotAllN) {
#ifdef BOYS_CUDA_GATE_FP16
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
            boys::BoysCuda::SingleF16(dN.get(), dX.get(), dOut.get(), cells, nullptr),
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
                        HalfBoundAt(got),
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
        CheckLaunch(boys::BoysCuda::AllOrdersF16(
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
                        HalfBoundAt(got),
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
            boys::BoysCuda::AllNF16(nmax, dX.get(), dOut.get(), count, nullptr),
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
                        HalfBoundAt(got),
                        Unrepresentable(got, kF16MinNormalExp));
            }
        }
    }
#else
    // No entry to call: this build's fp16 seam is closed, so the three rows
    // this sweep fills are left unmeasured, and the report names them by their
    // claims.
    (void)ref;
    (void)grid;
    (void)sorted;
    NoteNotCarried(slotSingle);
    NoteNotCarried(slotOrders);
    NoteNotCarried(slotAllN);
#endif // BOYS_CUDA_GATE_FP16
}

// The device-callable entries, reached from a consumer's own kernel.
// ---------------------------------------------------------------------------

// The slot a family entry wrote for one element: a family call writes
// kMaxBoysOrder + 1 values per element, in the element's own block.
std::size_t FamilySlot(std::size_t element, int order) {
    return element * (static_cast<std::size_t>(boys::kMaxBoysOrder) + 1) +
           static_cast<std::size_t>(order);
}

// Two shapes of one body, compared bit for bit. A bound says how close a return
// is to F_n(x); this says two entries returned the same bits. Only the slots a
// call writes are compared: the slots above an element's top order would only
// count agreements that say nothing.
struct ShapeAgreement {
    std::string what;
    std::size_t identical = 0;
    std::size_t slots = 0;
};

std::vector<ShapeAgreement>& ShapeAgreements() {
    static std::vector<ShapeAgreement> agreements;
    return agreements;
}

template <typename T>
void Agree(const std::string& what,
           const std::vector<T>& left,
           const std::vector<T>& right,
           const Reference& ref,
           int nmax) {
    ShapeAgreement agreement;
    agreement.what = what;

    for (int n = 0; n <= nmax; ++n)
    {
        for (std::size_t i = 0; i < ref.count; ++i)
        {
            const std::size_t e = ref.Index(n, i);

            for (int l = 0; l <= n; ++l)
            {
                const std::size_t slot = FamilySlot(e, l);
                ++agreement.slots;

                if (left[slot] == right[slot])
                {
                    ++agreement.identical;
                    continue;
                }

                std::fprintf(stderr,
                             "cuda gate: %s disagree at n=%d x=%.17g order %d: %.17g against"
                             " %.17g\n",
                             what.c_str(),
                             n,
                             ref.x[i],
                             l,
                             static_cast<double>(left[slot]),
                             static_cast<double>(right[slot]));
                std::exit(2);
            }
        }
    }

    ShapeAgreements().push_back(agreement);
}

// The compile-time-top-order entry against the runtime-top-order one, at the
// element whose own top order is kMaxBoysOrder. Both are one call to one body
// with one top order, so the claim is bit identity; a call at a lower order
// descends from that lower order, so comparing there would be comparing two
// arithmetic paths and not two entries.
template <typename T>
void AgreeFixedTopOrder(const std::string& what,
                        const std::vector<T>& fixed,
                        const std::vector<T>& runtime,
                        const Reference& ref,
                        int nmax) {
    ShapeAgreement agreement;
    agreement.what = what;

    for (std::size_t i = 0; i < ref.count; ++i)
    {
        const std::size_t top = ref.Index(nmax, i);

        for (int l = 0; l <= nmax; ++l)
        {
            const std::size_t slot = FamilySlot(top, l);
            ++agreement.slots;

            if (fixed[slot] == runtime[slot])
            {
                ++agreement.identical;
                continue;
            }

            std::fprintf(stderr,
                         "cuda gate: %s disagree at x=%.17g order %d: %.17g against %.17g\n",
                         what.c_str(),
                         ref.x[i],
                         l,
                         static_cast<double>(fixed[slot]),
                         static_cast<double>(runtime[slot]));
            std::exit(2);
        }
    }

    ShapeAgreements().push_back(agreement);
}

// Every element's status, required to be what the entry should have returned.
// The entries report through their enum and never throw.
void ExpectStatus(const std::vector<int>& status,
                  std::size_t count,
                  int want,
                  const char* what) {
    for (std::size_t e = 0; e < count; ++e)
    {
        if (status[e] != want)
        {
            std::fprintf(stderr,
                         "cuda gate: %s returned status %d for element %zu, expected %d\n",
                         what,
                         status[e],
                         e,
                         want);
            std::exit(2);
        }
    }
}

// What a caller reads back after a call that was refused, and after the same
// call repaired: a sentinel that survives every call is the other way this
// check can lie, so both directions are required.
template <typename T>
void ReportSentinel(const std::vector<T>& got,
                    const T& sentinel,
                    bool wrote,
                    const char* what) {
    const bool untouched = std::all_of(got.begin(), got.end(), [&](const T& v) {
        return v == sentinel;
    });

    if (wrote && untouched)
    {
        std::fprintf(stderr, "cuda gate: %s wrote nothing on a call it accepted\n", what);
        std::exit(2);
    }

    if (!wrote && !untouched)
    {
        std::fprintf(stderr, "cuda gate: %s wrote on a call it refused\n", what);
        std::exit(2);
    }
}

void CheckDemo(int error, const char* what) {
    Check(static_cast<cudaError_t>(error), what);
}

// Which row each device entry's cells are measured into. One entry per row
// except the double single lane, whose bound is published per region.
struct DeviceSlots {
    int singleA = -1;
    int singleBand = -1;
    int singleB = -1;
    int singleC = -1;
    int orders64 = -1;
    int allN64 = -1;
    int each64 = -1;
    int single32 = -1;
    int single32Fast = -1;
    int orders32 = -1;
    int allN32 = -1;
    int each32 = -1;
    int single16 = -1;
    int orders16 = -1;
    int allN16 = -1;
    int each16 = -1;

    // The partition and route axes of the ladder shape. Every row of them - the
    // float lane's narrow and rational rows included - is served, so
    // DeviceClaimSet below fills every slot here; the -1 is the initial value
    // and not a slot the library leaves unclaimed.
    int narrow64 = -1;
    int narrowMono64 = -1;
    int rat64 = -1;
    int ratHorner64 = -1;
    int narrowRat64 = -1;
    int narrowRatHorner64 = -1;
    int uniform64 = -1;
    int uniformHorner64 = -1;
    int narrow32 = -1;
    int narrowMono32 = -1;
    int rat32 = -1;
    int ratHorner32 = -1;
    int narrowRat32 = -1;
    int narrowRatHorner32 = -1;
    int uniform32 = -1;
    int uniformHorner32 = -1;
    int uniformRat64 = -1;
    int uniformRatHorner64 = -1;
    int uniformRat32 = -1;
    int uniformRatHorner32 = -1;
};

// The rows the device cells are measured into, each at the bound its own
// reported row documents.
DeviceSlots DeviceClaimSet() {
    DeviceSlots slots;
    const std::string single = Label(DeviceRow(boys::DeviceEntry::kDeviceSingleF64).name);
    const std::string orders64 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64).name);
    const std::string allN64 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllNF64).name);
    const std::string each64 = Label(DeviceRow(boys::DeviceEntry::kDeviceEachOrderF64).name);
    const std::string single32 = Label(DeviceRow(boys::DeviceEntry::kDeviceSingleF32).name);
    const std::string single32Fast = Label(DeviceRow(boys::DeviceEntry::kDeviceSingleF32Fast, boys::RegionBExp::kFast).name);
    const std::string orders32 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32).name);
    const std::string allN32 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllNF32).name);
    const std::string each32 = Label(DeviceRow(boys::DeviceEntry::kDeviceEachOrderF32).name);
    const std::string single16 = Label(DeviceRow(boys::DeviceEntry::kDeviceSingleF16).name);
    const std::string orders16 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF16).name);
    const std::string allN16 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllNF16).name);
    const std::string each16 = Label(DeviceRow(boys::DeviceEntry::kDeviceEachOrderF16).name);

    slots.singleA = AddClaim(single.c_str(), "A", kBoundSingleA);
    slots.singleBand = AddClaim(single.c_str(), "band", kBoundSingleBand);
    slots.singleB = AddClaim(single.c_str(), "B", kBoundSingleB);
    slots.singleC = AddClaim(single.c_str(), "C", kBoundSingleC);
    slots.orders64 = AddClaim(orders64.c_str(), "A..C", kBoundDoubleBatch);
    slots.allN64 = AddClaim(allN64.c_str(), "A..C", kBoundDoubleBatch);
    slots.each64 = AddClaim(each64.c_str(), "A..C", kBoundDoubleBatch);
    slots.single32 = AddClaim(single32.c_str(), "A..C", kBoundFloat);
    // The fast option's own bound: the lane's 1.5e-7 plus the corrected seed's
    // contribution, which the option carries.
    slots.single32Fast =
        AddClaim(single32Fast.c_str(), "A..C", kBoundFloat + kFastExpContribution);
    slots.orders32 = AddClaim(orders32.c_str(), "A..C", kBoundFloat);
    slots.allN32 = AddClaim(allN32.c_str(), "A..C", kBoundFloat);
    slots.each32 = AddClaim(each32.c_str(), "A..C", kBoundFloat);
    slots.single16 = AddClaim(single16.c_str(), "A..C", kBoundHalfRow);
    slots.orders16 = AddClaim(orders16.c_str(), "A..C", kBoundHalfRow);
    slots.allN16 = AddClaim(allN16.c_str(), "A..C", kBoundHalfRow);
    slots.each16 = AddClaim(each16.c_str(), "A..C", kBoundHalfRow);

    // The partition and route axes reached in the caller's kernel. Each row's
    // name, precision and bound are its launched row's above, because the
    // arithmetic is the same arithmetic: the two rows differ in how a caller
    // reaches it.
    //
    // Every row of the float lane - the shipped and narrow partitions' Chebyshev
    // and monomial forms and the rational route on both partitions - is served
    // and claimed here, as the double lane's rows and the grid's are: a claim is
    // made wherever the row's own entry says the row answers, so the claim set
    // and the space the library reports cannot come apart here.
    slots.narrow64 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64Narrow).name).c_str(),
        "A..C",
        kBoundDoubleBatch);
    slots.narrowMono64 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64NarrowMono).name).c_str(),
        "A..C",
        kBoundDoubleBatch);
    slots.rat64 =
        AddClaim(Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64Rat).name).c_str(),
                 "A..C",
                 kBoundDoubleBatch);
    slots.ratHorner64 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64RatHorner).name).c_str(),
        "A..C",
        kBoundDoubleBatch);
    slots.narrowRat64 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64NarrowRat).name).c_str(),
        "A..C",
        kBoundDoubleBatch);
    slots.narrowRatHorner64 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64NarrowRatHorner).name).c_str(),
        "A..C",
        kBoundDoubleBatch);
    slots.uniform64 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64Uniform).name).c_str(),
        "A..C",
        kBoundDoubleBatch);
    slots.uniformHorner64 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64UniformHorner).name).c_str(),
        "A..C",
        kBoundDoubleBatch);
    slots.uniform32 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32Uniform).name).c_str(),
        "A..C",
        kBoundFloat);
    slots.uniformHorner32 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32UniformHorner).name).c_str(),
        "A..C",
        kBoundFloat);
    // The grid's rational member on both lanes, claimed for the reason the four
    // above are: the table has no cut to make.
    slots.uniformRat64 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64UniformRat).name).c_str(),
        "A..C",
        kBoundDoubleBatch);
    slots.uniformRatHorner64 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64UniformRatHorner).name)
            .c_str(),
        "A..C",
        kBoundDoubleBatch);
    slots.uniformRat32 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32UniformRat).name).c_str(),
        "A..C",
        kBoundFloat);
    slots.uniformRatHorner32 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32UniformRatHorner).name)
            .c_str(),
        "A..C",
        kBoundFloat);
    // The float lane's narrow rows: the handle carries the cut on both halves
    // of the body — the double lane's narrow pieces for region A and this lane's
    // own region-B seed, one table per basis — so the two rows are claimed like
    // the grid's.
    slots.narrow32 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32Narrow).name).c_str(),
        "A..C",
        kBoundFloat);
    slots.narrowMono32 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32NarrowMono).name).c_str(),
        "A..C",
        kBoundFloat);
    // The float lane's rational rows: the handle carries the cut on both halves
    // of the pair - the double lane's pairs for region A and this lane's own for
    // region B.
    slots.rat32 =
        AddClaim(Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32Rat).name).c_str(),
                 "A..C",
                 kBoundFloat);
    slots.ratHorner32 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32RatHorner).name).c_str(),
        "A..C",
        kBoundFloat);
    slots.narrowRat32 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32NarrowRat).name).c_str(),
        "A..C",
        kBoundFloat);
    slots.narrowRatHorner32 = AddClaim(
        Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32NarrowRatHorner).name)
            .c_str(),
        "A..C",
        kBoundFloat);

    return slots;
}

// A device entry and the batch entry of the same precision: one arithmetic
// reached two ways - the same degree tables, the same inlined body, a lane
// object that reads the caller's handle instead of a __constant__ symbol. A
// bound cannot say that, and the header claims it, so every value the device
// call wrote is compared bit for bit against the batch entry launched on the
// same orders and the same arguments; a divergence is a defect in one of the
// two routes, not a looser bound.
struct PairAgreement {
    std::string what;
    std::size_t identical = 0;
    std::size_t values = 0;
};

std::vector<PairAgreement>& PairAgreements() {
    static std::vector<PairAgreement> agreements;
    return agreements;
}

[[noreturn]] void Disagree(const std::string& what,
                           std::size_t cell,
                           int order,
                           double x,
                           double device,
                           double batch) {
    std::fprintf(stderr,
                 "cuda gate: %s disagree at element %zu, order %d, grid argument x=%.17g:"
                 " %.17g against %.17g\n",
                 what.c_str(),
                 cell,
                 order,
                 x,
                 device,
                 batch);
    std::exit(2);
}

// One value per element on both sides: the single entries, whose output is
// indexed by the cell the demo kernel formed its argument in.
template <typename T>
void AgreeCells(const std::string& what,
                    const std::vector<T>& device,
                    const std::vector<T>& batch,
                    const Reference& ref) {
    PairAgreement agreement;
    agreement.what = what;

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        for (std::size_t i = 0; i < ref.count; ++i)
        {
            const std::size_t e = ref.Index(n, i);
            ++agreement.values;

            if (device[e] == batch[e])
            {
                ++agreement.identical;
                continue;
            }

            Disagree(what, e, n, ref.x[i], static_cast<double>(device[e]),
                     static_cast<double>(batch[e]));
        }
    }

    PairAgreements().push_back(agreement);
}

// One top order's ladder against the batch entry launched at that same top
// order. Region A descends from the top order, so two top orders are two chains
// of rounding and the comparison has to be made within one; the counts land in
// the caller's agreement row.
template <typename T>
void AgreeOrder(PairAgreement& agreement,
                    const std::vector<T>& device,
                    const std::vector<T>& batch,
                    const std::vector<double>& args,
                    int top) {
    const std::size_t count = args.size();

    for (std::size_t i = 0; i < count; ++i)
    {
        const std::size_t e = static_cast<std::size_t>(top) * count + i;

        for (int l = 0; l <= top; ++l)
        {
            const std::size_t deviceSlot = FamilySlot(e, l);
            const std::size_t batchSlot = static_cast<std::size_t>(l) * count + i;
            ++agreement.values;

            if (device[deviceSlot] == batch[batchSlot])
            {
                ++agreement.identical;
                continue;
            }

            Disagree(agreement.what,
                     e,
                     l,
                     args[i],
                     static_cast<double>(device[deviceSlot]),
                     static_cast<double>(batch[batchSlot]));
        }
    }
}

// Every order of a family. The device call writes the element's own block and
// the batch entry the order-major plane, so the two indexings are compared
// through the element: `deviceTop` is the highest order this comparison covers,
// the element's own top order for the shapes that descend from it and
// kMaxBoysOrder for the shape that descends from kMaxBoysOrder at every
// argument. A slot above that is one neither call wrote.
template <typename T>
void AgreeFamily(const std::string& what,
                     const std::vector<T>& device,
                     const std::vector<T>& batch,
                     const std::vector<int>& orders,
                     const Reference& ref,
                     bool uniformTop) {
    PairAgreement agreement;
    agreement.what = what;
    const std::size_t cells = ref.count * (static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);

    for (std::size_t e = 0; e < cells; ++e)
    {
        const int top = uniformTop ? boys::kMaxBoysOrder : orders[e];
        const std::size_t i = e % ref.count;

        for (int l = 0; l <= top; ++l)
        {
            const std::size_t deviceSlot = FamilySlot(e, l);
            const std::size_t batchSlot = static_cast<std::size_t>(l) * ref.count + i;
            ++agreement.values;

            if (device[deviceSlot] == batch[batchSlot])
            {
                ++agreement.identical;
                continue;
            }

            Disagree(what, e, l, ref.x[i], static_cast<double>(device[deviceSlot]),
                     static_cast<double>(batch[batchSlot]));
        }
    }

    PairAgreements().push_back(agreement);
}

// One device-callable row of the ladder shape, measured: the entry called
// inside the consumer's kernel over the grid, its cells compared with the
// reference at the row's own bound, and its values compared bit for bit with
// the launched row of the same option.
//
// The two comparisons are two statements and neither implies the other. The
// bound is what a caller of the row is promised, and it is the claim the report
// carries. The bit-for-bit comparison is what the header claims for a
// device-callable entry - one arithmetic reached two ways, the same degree
// tables read through the handle instead of through a __constant__ symbol -
// which no bound can state.
void MeasureDeviceLadder64(
    const Reference& ref,
    const Grid& grid,
    const boys::BoysDeviceTables& tables,
    int slot,
    const char* name,
    int (*demo)(const boys::BoysDeviceTables*,
                const int*,
                const double*,
                const double*,
                double*,
                std::size_t,
                int,
                int*),
    boys::BoysStatus (*launch)(const int*, const double*, double*, std::size_t, void*,
                               boys::DivisionForm)) {
    const std::size_t count = ref.count;
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t cells = grid.cells;
    const std::size_t family = cells * (static_cast<std::size_t>(nmax) + 1);
    DevBuf<int> dN(cells);
    DevBuf<double> dRho(cells);
    DevBuf<double> dD2(cells);
    DevBuf<double> dOut(family);
    DevBuf<int> dStatus(cells);
    dN.Upload(grid.n);
    dRho.Upload(grid.rho);
    dD2.Upload(grid.d2);
    dOut.Upload(std::vector<double>(family, 0.0));
    dStatus.Upload(std::vector<int>(cells, -1));
    CheckDemo(demo(&tables,
                   dN.get(),
                   dRho.get(),
                   dD2.get(),
                   dOut.get(),
                   cells,
                   nmax + 1,
                   dStatus.get()),
              name);
    std::vector<double> out(family);
    std::vector<int> status(cells);
    dOut.Download(out);
    dStatus.Download(status);
    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    ExpectStatus(status, cells, BoysDeviceDemoStatusSuccess(), name);

    for (int n = 0; n <= nmax; ++n)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t e = ref.Index(n, i);
            const double got = out[FamilySlot(e, n)];
            Measure(slot,
                    n,
                    ref.x[i],
                    got,
                    ref.v[e],
                    ref.decade[e],
                    Claims()[static_cast<std::size_t>(slot)].baseBound,
                    Unrepresentable(got, -1022));
        }
    }

    PairAgreement agreement;
    agreement.what =
        ShapeLabel((std::string(name) + " and its launched row").c_str());
    DevBuf<int> dNb(count);
    DevBuf<double> dXb(count);
    DevBuf<double> dOutb(cells);
    dXb.Upload(grid.x);
    std::vector<double> batch(cells);

    for (int top = 0; top <= nmax; ++top)
    {
        dNb.Upload(std::vector<int>(count, top));
        CheckLaunch(launch(dNb.get(), dXb.get(), dOutb.get(), count, nullptr, kGateDivisionForm),
                    name);
        dOutb.Download(batch);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        AgreeOrder(agreement, out, batch, ref.x, top);
    }

    PairAgreements().push_back(agreement);
}

// The float lane's half of the same measurement: the reference is that lane's
// own column, because the entry evaluates the argument the thread formed and
// rounded, and the bound is the figure the row documents.
void MeasureDeviceLadder32(
    const Reference& ref,
    const Grid& grid,
    const boys::BoysDeviceTables& tables,
    int slot,
    const char* name,
    int (*demo)(const boys::BoysDeviceTables*,
                const int*,
                const double*,
                const double*,
                float*,
                std::size_t,
                int,
                int*),
    boys::BoysStatus (*launch)(const int*, const double*, float*, std::size_t, void*,
                               boys::DivisionForm)) {
    const std::size_t count = ref.count;
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t cells = grid.cells;
    const std::size_t family = cells * (static_cast<std::size_t>(nmax) + 1);
    DevBuf<int> dN(cells);
    DevBuf<double> dRho(cells);
    DevBuf<double> dD2(cells);
    DevBuf<float> dOut(family);
    DevBuf<int> dStatus(cells);
    dN.Upload(grid.n);
    dRho.Upload(grid.rho);
    dD2.Upload(grid.d2);
    dOut.Upload(std::vector<float>(family, 0.0f));
    dStatus.Upload(std::vector<int>(cells, -1));
    CheckDemo(demo(&tables,
                   dN.get(),
                   dRho.get(),
                   dD2.get(),
                   dOut.get(),
                   cells,
                   nmax + 1,
                   dStatus.get()),
              name);
    std::vector<float> out(family);
    std::vector<int> status(cells);
    dOut.Download(out);
    dStatus.Download(status);
    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    ExpectStatus(status, cells, BoysDeviceDemoStatusSuccess(), name);

    for (int n = 0; n <= nmax; ++n)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t e = ref.Index(n, i);
            const auto [got, unrepresentable] = Widen(out[FamilySlot(e, n)]);
            Measure(slot,
                    n,
                    ref.xf[i],
                    got,
                    ref.vf[e],
                    ref.decadeF[e],
                    Claims()[static_cast<std::size_t>(slot)].baseBound,
                    unrepresentable);
        }
    }

    PairAgreement agreement;
    agreement.what =
        ShapeLabel((std::string(name) + " and its launched row").c_str());
    DevBuf<int> dNb(count);
    DevBuf<double> dXb(count);
    DevBuf<float> dOutb(cells);
    dXb.Upload(grid.x);
    std::vector<float> batch(cells);

    for (int top = 0; top <= nmax; ++top)
    {
        dNb.Upload(std::vector<int>(count, top));
        CheckLaunch(launch(dNb.get(), dXb.get(), dOutb.get(), count, nullptr, kGateDivisionForm),
                    name);
        dOutb.Download(batch);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        AgreeOrder(agreement, out, batch, ref.x, top);
    }

    PairAgreements().push_back(agreement);
}

// Every device entry, measured against the grid through the consumer kernels of
// tests/boys_cuda_device_demo.cu, and compared bit for bit with the batch entry
// of the same precision.
void SweepDevice(const Reference& ref,
                 const Grid& grid,
                 const SortedArgs& sorted,
                 const boys::BoysDeviceTables& tables,
                 const DeviceSlots& slots) {
    const std::size_t count = ref.count;
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t cells = grid.cells;
    const std::size_t family = cells * (static_cast<std::size_t>(nmax) + 1);
    const int singleSlots[4] = {slots.singleA, slots.singleBand, slots.singleB, slots.singleC};
    const int success = BoysDeviceDemoStatusSuccess();

    // One order at one argument, in double precision: one element per cell,
    // dispatched by region exactly as the batch single entry is.
    {
        DevBuf<int> dN(cells);
        DevBuf<double> dRho(cells);
        DevBuf<double> dD2(cells);
        DevBuf<double> dOut(cells);
        DevBuf<int> dStatus(cells);
        dN.Upload(grid.n);
        dRho.Upload(grid.rho);
        dD2.Upload(grid.d2);
        std::vector<int> hostStatus(cells, -1);
        dStatus.Upload(hostStatus);
        CheckDemo(BoysDeviceDemoSingle64(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dOut.get(),
                                         cells,
                                         dStatus.get()),
                  "BoysDeviceDemoSingle64");
        std::vector<double> out(cells);
        std::vector<int> status(cells);
        dOut.Download(out);
        dStatus.Download(status);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ExpectStatus(status, cells, success, "BoysDeviceSingleF64");

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

        // The batch single entry on the same orders and the same arguments.
        {
            DevBuf<int> dNb(cells);
            DevBuf<double> dXb(cells);
            DevBuf<double> dOutb(cells);
            dNb.Upload(grid.n);
            dXb.Upload(grid.x);
            CheckLaunch(boys::BoysCuda::SingleF64(
                            dNb.get(), dXb.get(), dOutb.get(), cells, nullptr),
                        "SingleF64 (one arithmetic)");
            std::vector<double> batch(cells);
            dOutb.Download(batch);
            Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            AgreeCells(ShapeLabel("device single f64 and batch single f64"),
                           out,
                           batch,
                           ref);
        }
    }

    // The three double family shapes: one body reached by a runtime top order, a
    // compile-time top order and a sink that keeps no ladder, each measured at
    // every element's own top order and compared with each other beside the rows.
    {
        DevBuf<int> dN(cells);
        DevBuf<double> dRho(cells);
        DevBuf<double> dD2(cells);
        DevBuf<double> dLadder(family);
        DevBuf<double> dAllN(family);
        DevBuf<double> dEach(family);
        DevBuf<int> dStatusLadder(cells);
        DevBuf<int> dStatusAllN(cells);
        DevBuf<int> dStatusEach(cells);
        dN.Upload(grid.n);
        dRho.Upload(grid.rho);
        dD2.Upload(grid.d2);
        std::vector<double> hostZero(family, 0.0);
        std::vector<int> hostStatus(cells, -1);
        dLadder.Upload(hostZero);
        dAllN.Upload(hostZero);
        dEach.Upload(hostZero);
        dStatusLadder.Upload(hostStatus);
        dStatusAllN.Upload(hostStatus);
        dStatusEach.Upload(hostStatus);
        CheckDemo(BoysDeviceDemoLadder64(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dLadder.get(),
                                         cells,
                                         nmax + 1,
                                         dStatusLadder.get()),
                  "BoysDeviceDemoLadder64");
        CheckDemo(BoysDeviceDemoAllN64(&tables,
                                       dRho.get(),
                                       dD2.get(),
                                       dAllN.get(),
                                       cells,
                                       dStatusAllN.get()),
                  "BoysDeviceDemoAllN64");
        CheckDemo(BoysDeviceDemoEach64(&tables,
                                       dN.get(),
                                       dRho.get(),
                                       dD2.get(),
                                       dEach.get(),
                                       cells,
                                       dStatusEach.get()),
                  "BoysDeviceDemoEach64");
        std::vector<double> ladder(family);
        std::vector<double> allN(family);
        std::vector<double> each(family);
        std::vector<int> statusLadder(cells);
        std::vector<int> statusAllN(cells);
        std::vector<int> statusEach(cells);
        dLadder.Download(ladder);
        dAllN.Download(allN);
        dEach.Download(each);
        dStatusLadder.Download(statusLadder);
        dStatusAllN.Download(statusAllN);
        dStatusEach.Download(statusEach);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ExpectStatus(statusLadder, cells, success, "BoysDeviceAllOrdersF64");
        ExpectStatus(statusAllN, cells, success, "BoysDeviceAllNF64");
        ExpectStatus(statusEach, cells, success, "BoysDeviceEachOrderF64");

        const auto measureElement = [&](int slot, int n, std::size_t i, double got) {
            Measure(slot,
                    n,
                    ref.x[i],
                    got,
                    ref.v[ref.Index(n, i)],
                    ref.decade[ref.Index(n, i)],
                    Claims()[static_cast<std::size_t>(slot)].baseBound,
                    Unrepresentable(got, -1022));
        };

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t e = ref.Index(n, i);
                measureElement(slots.orders64, n, i, ladder[FamilySlot(e, n)]);
                measureElement(slots.allN64, n, i, allN[FamilySlot(e, n)]);
                measureElement(slots.each64, n, i, each[FamilySlot(e, n)]);
            }
        }

        // The batch entries of the same precision, on the same orders and the same
        // arguments, element for element. The single entry is launched over the
        // whole element list, so its element is the device's element; the ladders
        // are launched once per top order, because a ladder descends from the
        // order it is named; the all-n entry takes its arguments non-decreasing,
        // so its plane is permuted back to the reference order first.
        {
            PairAgreement ladderAgreement;
            PairAgreement eachAgreement;
            ladderAgreement.what =
                ShapeLabel("device all-orders f64 and batch all-orders f64");
            eachAgreement.what = ShapeLabel("device each-order f64 and batch all-orders f64");
            {
                DevBuf<int> dNb(count);
                DevBuf<double> dXb(count);
                DevBuf<double> dOutb(cells);
                dXb.Upload(grid.x);
                std::vector<double> batch(cells);

                for (int top = 0; top <= nmax; ++top)
                {
                    const std::vector<int> hostN(count, top);
                    dNb.Upload(hostN);
                    CheckLaunch(boys::BoysCuda::AllOrdersF64(
                                    dNb.get(), dXb.get(), dOutb.get(), count, nullptr),
                                "AllOrdersF64 (one arithmetic)");
                    dOutb.Download(batch);
                    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                    AgreeOrder(ladderAgreement, ladder, batch, ref.x, top);
                    AgreeOrder(eachAgreement, each, batch, ref.x, top);
                }
            }

            PairAgreements().push_back(ladderAgreement);
            PairAgreements().push_back(eachAgreement);

            DevBuf<double> dXs(count);
            DevBuf<double> dOuts(cells);
            dXs.Upload(sorted.x);
            CheckLaunch(boys::BoysCuda::AllNF64(
                            nmax, dXs.get(), dOuts.get(), count, nullptr),
                        "AllNF64 (one arithmetic)");
            std::vector<double> rawSorted(cells);
            dOuts.Download(rawSorted);
            Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

            std::vector<double> batchSorted(cells);
            for (int l = 0; l <= nmax; ++l)
            {
                for (std::size_t j = 0; j < count; ++j)
                {
                    const std::size_t plane = static_cast<std::size_t>(l) * count;
                    batchSorted[plane + sorted.order[j]] = rawSorted[plane + j];
                }
            }

            AgreeFamily(ShapeLabel("device all-n f64 and batch all-n f64"),
                            allN,
                            batchSorted,
                            grid.n,
                            ref,
                            true);
        }

        Agree(ShapeLabel("device all-orders f64 and device each-order f64"),
              ladder,
              each,
              ref,
              nmax);
        AgreeFixedTopOrder(ShapeLabel("device all-n f64 and device all-orders f64"),
                           allN,
                           ladder,
                           ref,
                           nmax);
    }

    // The float shapes. The lane evaluates at the reference's own xf column:
    // the thread forms the argument in double, as a fused kernel forms it from
    // a density factor and a squared distance, and narrows it at the call.
    {
        DevBuf<int> dN(cells);
        DevBuf<double> dRho(cells);
        DevBuf<double> dD2(cells);
        DevBuf<float> dSingle(cells);
        DevBuf<float> dSingleFast(cells);
        DevBuf<float> dLadder(family);
        DevBuf<float> dAllN(family);
        DevBuf<float> dEach(family);
        DevBuf<int> dStatus(cells);
        dN.Upload(grid.n);
        dRho.Upload(grid.rho);
        dD2.Upload(grid.d2);
        std::vector<float> hostZero(family, 0.0f);
        std::vector<int> hostStatus(cells, -1);
        dSingle.Upload(std::vector<float>(cells, 0.0f));
        dSingleFast.Upload(std::vector<float>(cells, 0.0f));
        dLadder.Upload(hostZero);
        dAllN.Upload(hostZero);
        dEach.Upload(hostZero);
        dStatus.Upload(hostStatus);
        CheckDemo(BoysDeviceDemoSingle32(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dSingle.get(),
                                         cells,
                                         dStatus.get()),
                  "BoysDeviceDemoSingle32");
        CheckDemo(BoysDeviceDemoSingle32Fast(&tables,
                                             dN.get(),
                                             dRho.get(),
                                             dD2.get(),
                                             dSingleFast.get(),
                                             cells,
                                             dStatus.get()),
                  "BoysDeviceDemoSingle32Fast");
        CheckDemo(BoysDeviceDemoLadder32(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dLadder.get(),
                                         cells,
                                         nmax + 1,
                                         dStatus.get()),
                  "BoysDeviceDemoLadder32");
        CheckDemo(BoysDeviceDemoAllN32(&tables,
                                       dRho.get(),
                                       dD2.get(),
                                       dAllN.get(),
                                       cells,
                                       dStatus.get()),
                  "BoysDeviceDemoAllN32");
        CheckDemo(BoysDeviceDemoEach32(&tables,
                                       dN.get(),
                                       dRho.get(),
                                       dD2.get(),
                                       dEach.get(),
                                       cells,
                                       dStatus.get()),
                  "BoysDeviceDemoEach32");
        std::vector<float> single(cells);
        std::vector<float> singleFast(cells);
        std::vector<float> ladder(family);
        std::vector<float> allN(family);
        std::vector<float> each(family);
        std::vector<int> status(cells);
        dSingle.Download(single);
        dSingleFast.Download(singleFast);
        dLadder.Download(ladder);
        dAllN.Download(allN);
        dEach.Download(each);
        dStatus.Download(status);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ExpectStatus(status, cells, success, "the float device entries");

        const auto measureElement = [&](int slot, int n, std::size_t i, float got) {
            const auto [widened, unrepresentable] = Widen(got);
            Measure(slot,
                    n,
                    ref.xf[i],
                    widened,
                    ref.vf[ref.Index(n, i)],
                    ref.decadeF[ref.Index(n, i)],
                    Claims()[static_cast<std::size_t>(slot)].baseBound,
                    unrepresentable);
        };

        // The entry's two options, measured side by side: the fast one's cells
        // go into their own row under the bound its own option carries, and the
        // difference between the two returns is the audit's contribution term.
        ExpAudit audit;
        audit.lane = ShapeLabel(DeviceRow(boys::DeviceEntry::kDeviceSingleF32Fast, boys::RegionBExp::kFast).name);

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t e = ref.Index(n, i);
                measureElement(slots.single32, n, i, single[e]);

                const auto [widenedFast, fastUnrepresentable] = Widen(singleFast[e]);
                Measure(slots.single32Fast,
                        n,
                        ref.xf[i],
                        widenedFast,
                        ref.vf[e],
                        ref.decadeF[e],
                        kBoundFloat + kFastExpContribution,
                        fastUnrepresentable);

                const double accurate = Widen(single[e]).first;
                const double contribution = std::abs(accurate - widenedFast);

                if (contribution > audit.contributed)
                {
                    audit.contributed = contribution;
                    audit.contributedN = n;
                    audit.contributedX = ref.xf[i];
                }

                const double want = ref.vf[e];

                if (want != 0.0 && std::fabs(want) >= kFloatMinNormal)
                {
                    ++audit.relativeCells;

                    if (widenedFast != 0.0 && std::signbit(widenedFast) != std::signbit(want))
                    {
                        ++audit.wrongSign;
                        const double relative = std::fabs(widenedFast - want) / std::fabs(want);

                        if (relative > audit.signRelative)
                        {
                            audit.signRelative = relative;
                            audit.signRelativeN = n;
                            audit.signRelativeX = ref.xf[i];
                            audit.signRelativeWant = want;
                            audit.signRelativeGot = widenedFast;
                        }
                    }
                }

                measureElement(slots.orders32, n, i, ladder[FamilySlot(e, n)]);
                measureElement(slots.allN32, n, i, allN[FamilySlot(e, n)]);
                measureElement(slots.each32, n, i, each[FamilySlot(e, n)]);
            }
        }

        ExpAudits().push_back(audit);

        // The batch entries of the same precision, on the same orders and the same
        // arguments, element for element, as in the f64 block above.
        {
            {
                DevBuf<int> dNb(cells);
                DevBuf<double> dXb(cells);
                DevBuf<float> dOutb(cells);
                dNb.Upload(grid.n);
                dXb.Upload(grid.x);
                CheckLaunch(boys::BoysCuda::SingleF32(
                                dNb.get(), dXb.get(), dOutb.get(), cells, nullptr),
                            "SingleF32 (one arithmetic)");
                std::vector<float> batchSingle(cells);
                dOutb.Download(batchSingle);
                CheckLaunch(boys::BoysCuda::SingleF32<boys::RegionBExp::kFast>(
                                dNb.get(), dXb.get(), dOutb.get(), cells, nullptr),
                            "SingleF32 (one arithmetic)");
                std::vector<float> batchSingleFast(cells);
                dOutb.Download(batchSingleFast);
                Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                AgreeCells(ShapeLabel("device single f32 and batch single f32"),
                               single,
                               batchSingle,
                               ref);
                AgreeCells(
                    ShapeLabel("device single f32 (fast exp) and batch single f32 (fast exp)"),
                    singleFast,
                    batchSingleFast,
                    ref);
            }

            PairAgreement ladderAgreement;
            PairAgreement eachAgreement;
            ladderAgreement.what =
                ShapeLabel("device all-orders f32 and batch all-orders f32");
            eachAgreement.what = ShapeLabel("device each-order f32 and batch all-orders f32");
            {
                DevBuf<int> dNb(count);
                DevBuf<double> dXb(count);
                DevBuf<float> dOutb(cells);
                dXb.Upload(grid.x);
                std::vector<float> batch(cells);

                for (int top = 0; top <= nmax; ++top)
                {
                    const std::vector<int> hostN(count, top);
                    dNb.Upload(hostN);
                    CheckLaunch(boys::BoysCuda::AllOrdersF32(
                                    dNb.get(), dXb.get(), dOutb.get(), count, nullptr),
                                "AllOrdersF32 (one arithmetic)");
                    dOutb.Download(batch);
                    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                    AgreeOrder(ladderAgreement, ladder, batch, ref.xf, top);
                    AgreeOrder(eachAgreement, each, batch, ref.xf, top);
                }
            }

            PairAgreements().push_back(ladderAgreement);
            PairAgreements().push_back(eachAgreement);

            DevBuf<double> dXs(count);
            DevBuf<float> dOuts(cells);
            dXs.Upload(sorted.x);
            CheckLaunch(boys::BoysCuda::AllNF32(
                            nmax, dXs.get(), dOuts.get(), count, nullptr),
                        "AllNF32 (one arithmetic)");
            std::vector<float> rawSorted(cells);
            dOuts.Download(rawSorted);
            Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

            std::vector<float> batchSorted(cells);
            for (int l = 0; l <= nmax; ++l)
            {
                for (std::size_t j = 0; j < count; ++j)
                {
                    const std::size_t plane = static_cast<std::size_t>(l) * count;
                    batchSorted[plane + sorted.order[j]] = rawSorted[plane + j];
                }
            }

            AgreeFamily(ShapeLabel("device all-n f32 and batch all-n f32"),
                            allN,
                            batchSorted,
                            grid.n,
                            ref,
                            true);
        }

        Agree(ShapeLabel("device all-orders f32 and device each-order f32"),
              ladder,
              each,
              ref,
              nmax);
        AgreeFixedTopOrder(ShapeLabel("device all-n f32 and device all-orders f32"),
                           allN,
                           ladder,
                           ref,
                           nmax);
    }

    // The fp16 shapes, at the half value of the argument the grid carries, with
    // the bound the fp16 lane documents and the floor the format sets. Closed
    // seam: the entries these four rows are measured through are behind the seam,
    // so the rows stay unmeasured rather than measured against nothing.
    {
#ifdef BOYS_CUDA_GATE_FP16
        DevBuf<int> dN(cells);
        DevBuf<double> dRho(cells);
        DevBuf<double> dD2(cells);
        DevBuf<boys::F16> dSingle(cells);
        DevBuf<boys::F16> dLadder(family);
        DevBuf<boys::F16> dAllN(family);
        DevBuf<boys::F16> dEach(family);
        DevBuf<int> dStatus(cells);
        dN.Upload(grid.n);
        dRho.Upload(grid.rho);
        dD2.Upload(grid.d2);
        std::vector<boys::F16> hostZero(family, boys::F16(0.0f));
        std::vector<int> hostStatus(cells, -1);
        dSingle.Upload(std::vector<boys::F16>(cells, boys::F16(0.0f)));
        dLadder.Upload(hostZero);
        dAllN.Upload(hostZero);
        dEach.Upload(hostZero);
        dStatus.Upload(hostStatus);
        CheckDemo(BoysDeviceDemoSingle16(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         static_cast<void*>(dSingle.get()),
                                         cells,
                                         dStatus.get()),
                  "BoysDeviceDemoSingle16");
        CheckDemo(BoysDeviceDemoLadder16(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         static_cast<void*>(dLadder.get()),
                                         cells,
                                         nmax + 1,
                                         dStatus.get()),
                  "BoysDeviceDemoLadder16");
        CheckDemo(BoysDeviceDemoAllN16(&tables,
                                       dRho.get(),
                                       dD2.get(),
                                       static_cast<void*>(dAllN.get()),
                                       cells,
                                       dStatus.get()),
                  "BoysDeviceDemoAllN16");
        CheckDemo(BoysDeviceDemoEach16(&tables,
                                       dN.get(),
                                       dRho.get(),
                                       dD2.get(),
                                       static_cast<void*>(dEach.get()),
                                       cells,
                                       dStatus.get()),
                  "BoysDeviceDemoEach16");
        std::vector<boys::F16> single(cells);
        std::vector<boys::F16> ladder(family);
        std::vector<boys::F16> allN(family);
        std::vector<boys::F16> each(family);
        std::vector<int> status(cells);
        dSingle.Download(single);
        dLadder.Download(ladder);
        dAllN.Download(allN);
        dEach.Download(each);
        dStatus.Download(status);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ExpectStatus(status, cells, success, "the fp16 device entries");

        const auto measureElement = [&](int slot, int n, std::size_t i, double got) {
            Measure(slot,
                    n,
                    ref.x16[i],
                    got,
                    ref.v16[ref.Index(n, i)],
                    ref.decade16[ref.Index(n, i)],
                    HalfBoundAt(got),
                    Unrepresentable(got, kF16MinNormalExp));
        };

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t e = ref.Index(n, i);

                // An argument past the fp16 range has no reference value at all
                // (the grid's x16 column is infinite there), so it is not a measured
                // cell - the rule the batch fp16 single row follows too.
                if (std::isfinite(ref.x16[i]))
                {
                    measureElement(slots.single16, n, i, static_cast<double>(single[e]));
                }

                measureElement(slots.orders16, n, i, static_cast<double>(ladder[FamilySlot(e, n)]));
                measureElement(slots.allN16, n, i, static_cast<double>(allN[FamilySlot(e, n)]));
                measureElement(slots.each16, n, i, static_cast<double>(each[FamilySlot(e, n)]));
            }
        }

        // The batch entries of the same precision, on the same orders and arguments,
        // element for element, as in the f64 block above. The fp16 batch entries
        // take their argument in fp16 already, the value the grid carries.
        {
            {
                DevBuf<int> dNb(cells);
                DevBuf<boys::F16> dXb(cells);
                DevBuf<boys::F16> dOutb(cells);
                dNb.Upload(grid.n);
                dXb.Upload(grid.x16);
                CheckLaunch(boys::BoysCuda::SingleF16(
                                dNb.get(), dXb.get(), dOutb.get(), cells, nullptr),
                            "SingleF16 (one arithmetic)");
                std::vector<boys::F16> batchSingle(cells);
                dOutb.Download(batchSingle);
                Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                AgreeCells(ShapeLabel("device single f16 and batch single f16"),
                               single,
                               batchSingle,
                               ref);
            }

            PairAgreement ladderAgreement;
            PairAgreement eachAgreement;
            ladderAgreement.what =
                ShapeLabel("device all-orders f16 and batch all-orders f16");
            eachAgreement.what = ShapeLabel("device each-order f16 and batch all-orders f16");
            {
                DevBuf<int> dNb(count);
                DevBuf<boys::F16> dXb(count);
                DevBuf<boys::F16> dOutb(cells);
                std::vector<boys::F16> hostX(count);

                for (std::size_t i = 0; i < count; ++i)
                {
                    hostX[i] = boys::F16(static_cast<float>(ref.x[i]));
                }

                dXb.Upload(hostX);
                std::vector<boys::F16> batch(cells);

                for (int top = 0; top <= nmax; ++top)
                {
                    const std::vector<int> hostN(count, top);
                    dNb.Upload(hostN);
                    CheckLaunch(boys::BoysCuda::AllOrdersF16(
                                    dNb.get(), dXb.get(), dOutb.get(), count, nullptr),
                                "AllOrdersF16 (one arithmetic)");
                    dOutb.Download(batch);
                    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                    AgreeOrder(ladderAgreement, ladder, batch, ref.x16, top);
                    AgreeOrder(eachAgreement, each, batch, ref.x16, top);
                }
            }

            PairAgreements().push_back(ladderAgreement);
            PairAgreements().push_back(eachAgreement);

            DevBuf<boys::F16> dXs(count);
            DevBuf<boys::F16> dOuts(cells);
            std::vector<boys::F16> hostSorted(count);

            for (std::size_t j = 0; j < count; ++j)
            {
                hostSorted[j] = boys::F16(static_cast<float>(ref.x[sorted.order[j]]));
            }

            dXs.Upload(hostSorted);
            CheckLaunch(boys::BoysCuda::AllNF16(
                            nmax, dXs.get(), dOuts.get(), count, nullptr),
                        "AllNF16 (one arithmetic)");
            std::vector<boys::F16> rawSorted(cells);
            dOuts.Download(rawSorted);
            Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

            std::vector<boys::F16> batchSorted(cells);
            for (int l = 0; l <= nmax; ++l)
            {
                for (std::size_t j = 0; j < count; ++j)
                {
                    const std::size_t plane = static_cast<std::size_t>(l) * count;
                    batchSorted[plane + sorted.order[j]] = rawSorted[plane + j];
                }
            }

            AgreeFamily(ShapeLabel("device all-n f16 and batch all-n f16"),
                            allN,
                            batchSorted,
                            grid.n,
                            ref,
                            true);
        }

        Agree(ShapeLabel("device all-orders f16 and device each-order f16"),
              ladder,
              each,
              ref,
              nmax);
        AgreeFixedTopOrder(ShapeLabel("device all-n f16 and device all-orders f16"),
                           allN,
                           ladder,
                           ref,
                           nmax);
#else
        NoteNotCarried(slots.single16);
        NoteNotCarried(slots.orders16);
        NoteNotCarried(slots.allN16);
        NoteNotCarried(slots.each16);
#endif // BOYS_CUDA_GATE_FP16
    }

    // The partition and route axes reached in the caller's kernel. Each is
    // measured the way the shapes above are: over the grid, against the
    // reference at the row's own bound, and bit for bit against the launched row
    // of the same option.
    //
    // The double lane's eight rows, and the float lane's grid, narrow and
    // rational rows, are all served: the float lane's rational pairs are cut as
    // its narrow seed is.
    MeasureDeviceLadder64(ref,
                                       grid,
                                       tables,
                                       slots.narrow64,
                                       DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64Narrow).name,
                                       &BoysDeviceDemoLadder64Narrow,
                                       &boys::BoysCuda::AllOrdersF64Narrow);
    MeasureDeviceLadder64(
        ref,
        grid,
        tables,
        slots.narrowMono64,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64NarrowMono).name,
        &BoysDeviceDemoLadder64NarrowMono,
        &boys::BoysCuda::AllOrdersF64NarrowMono);
    MeasureDeviceLadder64(ref,
                                       grid,
                                       tables,
                                       slots.rat64,
                                       DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64Rat).name,
                                       &BoysDeviceDemoLadder64Rat,
                                       &boys::BoysCuda::AllOrdersF64Rat);
    MeasureDeviceLadder64(
        ref,
        grid,
        tables,
        slots.ratHorner64,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64RatHorner).name,
        &BoysDeviceDemoLadder64RatHorner,
        &boys::BoysCuda::AllOrdersF64Rat);
    MeasureDeviceLadder64(
        ref,
        grid,
        tables,
        slots.narrowRat64,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64NarrowRat).name,
        &BoysDeviceDemoLadder64NarrowRat,
        &boys::BoysCuda::AllOrdersF64NarrowRat);
    MeasureDeviceLadder64(
        ref,
        grid,
        tables,
        slots.narrowRatHorner64,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64NarrowRatHorner).name,
        &BoysDeviceDemoLadder64NarrowRatHorner,
        &boys::BoysCuda::AllOrdersF64NarrowRat);
    MeasureDeviceLadder64(ref,
                                       grid,
                                       tables,
                                       slots.uniform64,
                                       DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64Uniform).name,
                                       &BoysDeviceDemoLadder64Uniform,
                                       &boys::BoysCuda::AllOrdersF64Uniform);
    MeasureDeviceLadder64(
        ref,
        grid,
        tables,
        slots.uniformHorner64,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64UniformHorner).name,
        &BoysDeviceDemoLadder64UniformHorner,
        &boys::BoysCuda::AllOrdersF64UniformHorner);
    MeasureDeviceLadder32(
        ref,
        grid,
        tables,
        slots.uniform32,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32Uniform).name,
        &BoysDeviceDemoLadder32Uniform,
        &boys::BoysCuda::AllOrdersF32Uniform);
    MeasureDeviceLadder32(
        ref,
        grid,
        tables,
        slots.uniformHorner32,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32UniformHorner).name,
        &BoysDeviceDemoLadder32UniformHorner,
        &boys::BoysCuda::AllOrdersF32UniformHorner);
    MeasureDeviceLadder64(
        ref,
        grid,
        tables,
        slots.uniformRat64,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64UniformRat).name,
        &BoysDeviceDemoLadder64UniformRat,
        &boys::BoysCuda::AllOrdersF64UniformRat);
    MeasureDeviceLadder64(
        ref,
        grid,
        tables,
        slots.uniformRatHorner64,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64UniformRatHorner).name,
        &BoysDeviceDemoLadder64UniformRatHorner,
        &boys::BoysCuda::AllOrdersF64UniformRatHorner);
    MeasureDeviceLadder32(
        ref,
        grid,
        tables,
        slots.uniformRat32,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32UniformRat).name,
        &BoysDeviceDemoLadder32UniformRat,
        &boys::BoysCuda::AllOrdersF32UniformRat);
    MeasureDeviceLadder32(
        ref,
        grid,
        tables,
        slots.uniformRatHorner32,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32UniformRatHorner).name,
        &BoysDeviceDemoLadder32UniformRatHorner,
        &boys::BoysCuda::AllOrdersF32UniformRatHorner);
    // The float lane's narrow partition, on both of the forms it is stored in:
    // each of these two rows is measured here like the grid's above, one
    // arithmetic reached two ways.
    MeasureDeviceLadder32(
        ref,
        grid,
        tables,
        slots.narrow32,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32Narrow).name,
        &BoysDeviceDemoLadder32Narrow,
        &boys::BoysCuda::AllOrdersF32Narrow);
    MeasureDeviceLadder32(
        ref,
        grid,
        tables,
        slots.narrowMono32,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32NarrowMono).name,
        &BoysDeviceDemoLadder32NarrowMono,
        &boys::BoysCuda::AllOrdersF32NarrowMono);

    // The float lane's rational rows, each measured like the rows above and bit
    // for bit against the launched row of the same option: the route's pair is
    // cut on both lanes.
    MeasureDeviceLadder32(
        ref,
        grid,
        tables,
        slots.rat32,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32Rat).name,
        &BoysDeviceDemoLadder32Rat,
        &boys::BoysCuda::AllOrdersF32Rat);
    MeasureDeviceLadder32(
        ref,
        grid,
        tables,
        slots.ratHorner32,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32RatHorner).name,
        &BoysDeviceDemoLadder32RatHorner,
        &boys::BoysCuda::AllOrdersF32RatHorner);
    MeasureDeviceLadder32(
        ref,
        grid,
        tables,
        slots.narrowRat32,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32NarrowRat).name,
        &BoysDeviceDemoLadder32NarrowRat,
        &boys::BoysCuda::AllOrdersF32NarrowRat);
    MeasureDeviceLadder32(
        ref,
        grid,
        tables,
        slots.narrowRatHorner32,
        DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32NarrowRatHorner).name,
        &BoysDeviceDemoLadder32NarrowRatHorner,
        &boys::BoysCuda::AllOrdersF32NarrowRatHorner);
}

// One fp64 entry's cells on the 45-digit grid. They are measured into the
// entry's own row as well as into the accumulator below, so a cell over bound
// here fails the gate like every other cell; the accumulator is what names the
// worst cell this grid produced.
struct DigitRow {
    Accum cell;
};

std::vector<DigitRow>& DigitRows() {
    static std::vector<DigitRow> rows;
    return rows;
}

void SweepDigit64(const DigitGrid& digits,
                  const boys::BoysDeviceTables& tables,
                  const DeviceSlots& slots) {
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = digits.count;
    const std::size_t cells = digits.cells;
    const std::size_t family = cells * (static_cast<std::size_t>(nmax) + 1);
    const int success = BoysDeviceDemoStatusSuccess();
    const int singleSlots[4] = {slots.singleA, slots.singleBand, slots.singleB, slots.singleC};
    const double singleBounds[4] = {kBoundSingleA,
                                    kBoundSingleBand,
                                    kBoundSingleB,
                                    kBoundSingleC};
    const double familyBound = kBoundDoubleBatch;

    // Every thread forms its own argument as rho * d2, exactly as the kernels of
    // the gate grid's sweep do: rho is a power of two, so the product is the
    // grid's own argument to the last bit.
    std::vector<double> hostRho(cells);
    std::vector<double> hostD2(cells);

    for (std::size_t e = 0; e < cells; ++e)
    {
        const double rho = std::ldexp(1.0, static_cast<int>((e % count) % 9) - 4);
        hostRho[e] = rho;
        hostD2[e] = digits.x[e] / rho;

        if (rho * hostD2[e] != digits.x[e])
        {
            std::fprintf(stderr,
                         "cuda gate: the factor pair is not exact at 45-digit row %zu: %.17g *"
                         " %.17g != %.17g\n",
                         e,
                         rho,
                         hostD2[e],
                         digits.x[e]);
            std::exit(2);
        }
    }

    DigitRow single;
    DigitRow orders;
    DigitRow each;
    DigitRow allN;
    single.cell.lane = "device single f64";
    orders.cell.lane = "device all-orders f64";
    each.cell.lane = "device each-order f64";
    allN.cell.lane = "device all-n f64";

    // The two accumulations of one reading: the shared row, judged with every
    // other cell, and this grid's own, which the table below prints.
    const auto measure = [&](Accum& local,
                             int slot,
                             int n,
                             double x,
                             double got,
                             double refValue,
                             int refDecade,
                             double bound,
                             bool unrepresentable) {
        Measure(slot, n, x, got, refValue, refDecade, bound, unrepresentable);
        MeasureAt(local, n, x, got, refValue, refDecade, bound, unrepresentable);
    };

    {
        DevBuf<int> dN(cells);
        DevBuf<double> dRho(cells);
        DevBuf<double> dD2(cells);
        DevBuf<double> dSingle(cells);
        DevBuf<double> dLadder(family);
        DevBuf<double> dEach(family);
        DevBuf<double> dAllN(family);
        DevBuf<int> dStatus(cells);
        dN.Upload(digits.n);
        dRho.Upload(hostRho);
        dD2.Upload(hostD2);
        std::vector<double> hostZero(family, 0.0);
        std::vector<int> hostStatus(cells, -1);
        dSingle.Upload(std::vector<double>(cells, 0.0));
        dLadder.Upload(hostZero);
        dEach.Upload(hostZero);
        dAllN.Upload(hostZero);
        dStatus.Upload(hostStatus);
        CheckDemo(BoysDeviceDemoSingle64(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dSingle.get(),
                                         cells,
                                         dStatus.get()),
                  "BoysDeviceDemoSingle64 (45-digit grid)");
        CheckDemo(BoysDeviceDemoLadder64(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dLadder.get(),
                                         cells,
                                         nmax + 1,
                                         dStatus.get()),
                  "BoysDeviceDemoLadder64 (45-digit grid)");
        CheckDemo(BoysDeviceDemoEach64(&tables,
                                       dN.get(),
                                       dRho.get(),
                                       dD2.get(),
                                       dEach.get(),
                                       cells,
                                       dStatus.get()),
                  "BoysDeviceDemoEach64 (45-digit grid)");
        CheckDemo(BoysDeviceDemoAllN64(&tables,
                                       dRho.get(),
                                       dD2.get(),
                                       dAllN.get(),
                                       cells,
                                       dStatus.get()),
                  "BoysDeviceDemoAllN64 (45-digit grid)");
        std::vector<double> gotSingle(cells);
        std::vector<double> gotLadder(family);
        std::vector<double> gotEach(family);
        std::vector<double> gotAllN(family);
        std::vector<int> status(cells);
        dSingle.Download(gotSingle);
        dLadder.Download(gotLadder);
        dEach.Download(gotEach);
        dAllN.Download(gotAllN);
        dStatus.Download(status);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ExpectStatus(status, cells, success, "the fp64 device entries on the 45-digit grid");

        for (std::size_t e = 0; e < cells; ++e)
        {
            const int n = digits.n[e];
            const double x = digits.x[e];
            const double refValue = digits.v[e];
            const int refDecade = digits.decade[e];
            const int region = SingleClaim(x);

            measure(single.cell,
                    singleSlots[region],
                    n,
                    x,
                    gotSingle[e],
                    refValue,
                    refDecade,
                    singleBounds[region],
                    Unrepresentable(gotSingle[e], -1022));
            measure(orders.cell,
                    slots.orders64,
                    n,
                    x,
                    gotLadder[FamilySlot(e, n)],
                    refValue,
                    refDecade,
                    familyBound,
                    Unrepresentable(gotLadder[FamilySlot(e, n)], -1022));
            measure(each.cell,
                    slots.each64,
                    n,
                    x,
                    gotEach[FamilySlot(e, n)],
                    refValue,
                    refDecade,
                    familyBound,
                    Unrepresentable(gotEach[FamilySlot(e, n)], -1022));
            measure(allN.cell,
                    slots.allN64,
                    n,
                    x,
                    gotAllN[FamilySlot(e, n)],
                    refValue,
                    refDecade,
                    familyBound,
                    Unrepresentable(gotAllN[FamilySlot(e, n)], -1022));
        }
    }

    DigitRows().push_back(std::move(single));
    DigitRows().push_back(std::move(orders));
    DigitRows().push_back(std::move(each));
    DigitRows().push_back(std::move(allN));
}

// The refusals, per shape where the shapes differ and per precision where they
// do not: an order outside the range, a capacity one value short, and - beside
// each - the same call repaired. The decision is on the order and the capacity
// and not on the argument, so four elements carry it.
void CheckRefusals(const boys::BoysDeviceTables& tables) {
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = 4;
    const std::size_t family = count * (static_cast<std::size_t>(nmax) + 1);
    const double sentinel = -12345.0;
    const double x = 3.5; // region A; the guard is upstream of the region
    const int success = BoysDeviceDemoStatusSuccess();
    const int outOfRange = BoysDeviceDemoStatusOrderOutOfRange();
    const int tooSmall = BoysDeviceDemoStatusCapacityTooSmall();
    const std::vector<int> hostN(count, nmax);
    const std::vector<double> hostRho(count, 1.0);
    const std::vector<double> hostD2(count, x);
    const std::vector<int> hostStatus(count, -1);

    // The ladder's two reasons, in double precision.
    const auto ladder64 = [&](int order, int capacity, int want) {
        const std::vector<int> hostOrder(count, order);
        std::vector<double> hostOut(family, sentinel);
        DevBuf<int> dN(count);
        DevBuf<double> dRho(count);
        DevBuf<double> dD2(count);
        DevBuf<double> dOut(family);
        DevBuf<int> dStatus(count);
        dN.Upload(hostOrder);
        dRho.Upload(hostRho);
        dD2.Upload(hostD2);
        dOut.Upload(hostOut);
        dStatus.Upload(hostStatus);
        CheckDemo(BoysDeviceDemoLadder64(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dOut.get(),
                                         count,
                                         capacity,
                                         dStatus.get()),
                  "BoysDeviceDemoLadder64 (refusal probe)");
        std::vector<double> out(family);
        std::vector<int> status(count);
        dOut.Download(out);
        dStatus.Download(status);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ExpectStatus(status, count, want, "BoysDeviceAllOrdersF64 (refusal probe)");
        ReportSentinel(out, sentinel, want == success, "BoysDeviceAllOrdersF64");
    };

    ladder64(nmax + 1, nmax + 1, outOfRange);
    ladder64(nmax, nmax, tooSmall);
    ladder64(nmax, nmax + 1, success);

    // The single entry has no capacity to be short of, so its one refusal is
    // the order.
    {
        const std::vector<int> hostOrder(count, nmax + 1);
        std::vector<double> hostOut(count, sentinel);
        DevBuf<int> dN(count);
        DevBuf<double> dRho(count);
        DevBuf<double> dD2(count);
        DevBuf<double> dOut(count);
        DevBuf<int> dStatus(count);
        dN.Upload(hostOrder);
        dRho.Upload(hostRho);
        dD2.Upload(hostD2);
        dOut.Upload(hostOut);
        dStatus.Upload(hostStatus);
        CheckDemo(BoysDeviceDemoSingle64(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dOut.get(),
                                         count,
                                         dStatus.get()),
                  "BoysDeviceDemoSingle64 (refusal probe)");
        std::vector<double> out(count);
        std::vector<int> status(count);
        dOut.Download(out);
        dStatus.Download(status);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ExpectStatus(status, count, outOfRange, "BoysDeviceSingleF64 (refusal probe)");
        ReportSentinel(out, sentinel, false, "BoysDeviceSingleF64");
    }

    // The sink shape must not call the caller's sink on a refused call: the
    // sentinel it would have written is the evidence.
    {
        const std::vector<int> hostOrder(count, nmax + 1);
        std::vector<double> hostOut(family, sentinel);
        DevBuf<int> dN(count);
        DevBuf<double> dRho(count);
        DevBuf<double> dD2(count);
        DevBuf<double> dOut(family);
        DevBuf<int> dStatus(count);
        dN.Upload(hostOrder);
        dRho.Upload(hostRho);
        dD2.Upload(hostD2);
        dOut.Upload(hostOut);
        dStatus.Upload(hostStatus);
        CheckDemo(BoysDeviceDemoEach64(&tables,
                                       dN.get(),
                                       dRho.get(),
                                       dD2.get(),
                                       dOut.get(),
                                       count,
                                       dStatus.get()),
                  "BoysDeviceDemoEach64 (refusal probe)");
        std::vector<double> out(family);
        std::vector<int> status(count);
        dOut.Download(out);
        dStatus.Download(status);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ExpectStatus(status, count, outOfRange, "BoysDeviceEachOrderF64 (refusal probe)");
        ReportSentinel(out, sentinel, false, "BoysDeviceEachOrderF64");
    }

    // The other two precisions, both reasons on one shape.
    const auto ladder32 = [&](int order, int capacity, int want) {
        const std::vector<int> hostOrder(count, order);
        std::vector<float> hostOut(family, static_cast<float>(sentinel));
        DevBuf<int> dN(count);
        DevBuf<double> dRho(count);
        DevBuf<double> dD2(count);
        DevBuf<float> dOut(family);
        DevBuf<int> dStatus(count);
        dN.Upload(hostOrder);
        dRho.Upload(hostRho);
        dD2.Upload(hostD2);
        dOut.Upload(hostOut);
        dStatus.Upload(hostStatus);
        CheckDemo(BoysDeviceDemoLadder32(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dOut.get(),
                                         count,
                                         capacity,
                                         dStatus.get()),
                  "BoysDeviceDemoLadder32 (refusal probe)");
        std::vector<float> out(family);
        std::vector<int> status(count);
        dOut.Download(out);
        dStatus.Download(status);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ExpectStatus(status, count, want, "BoysDeviceAllOrdersF32 (refusal probe)");
        ReportSentinel(out, static_cast<float>(sentinel), want == success,
                       "BoysDeviceAllOrdersF32");
    };

    ladder32(nmax + 1, nmax + 1, outOfRange);
    ladder32(nmax, nmax, tooSmall);

#if BoysFp16
    const auto ladder16 = [&](int order, int capacity, int want) {
        const std::vector<int> hostOrder(count, order);
        const boys::F16 halfSentinel(static_cast<float>(sentinel));
        std::vector<boys::F16> hostOut(family, halfSentinel);
        DevBuf<int> dN(count);
        DevBuf<double> dRho(count);
        DevBuf<double> dD2(count);
        DevBuf<boys::F16> dOut(family);
        DevBuf<int> dStatus(count);
        dN.Upload(hostOrder);
        dRho.Upload(hostRho);
        dD2.Upload(hostD2);
        dOut.Upload(hostOut);
        dStatus.Upload(hostStatus);
        CheckDemo(BoysDeviceDemoLadder16(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         static_cast<void*>(dOut.get()),
                                         count,
                                         capacity,
                                         dStatus.get()),
                  "BoysDeviceDemoLadder16 (refusal probe)");
        std::vector<boys::F16> out(family);
        std::vector<int> status(count);
        dOut.Download(out);
        dStatus.Download(status);
        Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ExpectStatus(status, count, want, "BoysDeviceAllOrdersF16 (refusal probe)");
        ReportSentinel(out, halfSentinel, want == success, "BoysDeviceAllOrdersF16");
    };

    ladder16(nmax, nmax, tooSmall);
#endif // BoysFp16
}

// ---------------------------------------------------------------------------
// The uniform grid's carriage on the rows that answer a policy naming it.
//
// Every row of the report that answers a policy naming the uniform grid is
// measured above for accuracy, and no accuracy row can tell that grid from the
// narrow member below it: the two partitions are different fits of the same
// function over the same intervals and hold the same bar, so a row that read
// the narrow member's tables under the grid's name would print the same numbers
// inside the same bound and every row above would stay green. The rows below
// are that question asked of the values themselves, in the shape the CPU gate
// asks it in its own single-precision block: each row is read twice in one pass
// - once through its own entry and once through the narrow member's of the same
// arm - and the cells where the two readings differ are counted per region
// against the per-argument host entry's own separation at that arm. A region
// where the per-argument entry's two readings differ and this row's do not is a
// region this row answered from the narrow member's fits.
//
// The reference is the host per-argument entry's own two readings and not the
// device's, because the question is about the fits: the device lane reads the
// same tables the host lane does, uploaded, so where the fits separate at this
// precision on the host they separate on the card. The rule is the CPU gate's:
// a region where the per-argument entry's two readings agree is a region where
// no cell can discriminate, so no row is held to it - which is why region C is
// carried as a token and not judged where the lane's grid ends inside region B
// (kFlatHiF32 on the float lane and the half lane that reads its tables, kFlatHi
// on the double lane) and every reading above kX1 is the asymptotic form's. The
// comparison is made in the format the row itself returns, which is why the
// float and double lanes' rows are held to a reference read in their own.
//
// The half lane's rows are not two fits read against each other but one lane's
// bodies with half I/O: the row named for a policy is the float lane's body of
// that name, its arguments and its return the half ones (measured, all four
// arms on both axes: the half row and the float body it names agree in 56694 of
// 56694 cells at the half lane's own arguments). Each is held to the body it
// names, cell for cell, in every region: a row that read the other member's
// fits would part from the body it names in every cell where the two bodies'
// half images differ - 39 cells of B today - and the float lane's own rows
// carry the rest. The reference the two bodies' own separation is recorded
// against says why the rest is not carried here: the two bodies differ over
// 4148 cells of A and 7789 of band at fp32, and the half format tells 0 and 0
// of them apart, so in A and band a row answering from the other member's fits
// and a row answering from its own are the same half values, cell for cell, and
// no rule reading half values can tell them apart. That separation is measured
// on the device pair of the same arm, not on the host's half entry: the host
// path's two readings part at the half boundaries while the device pair agrees,
// which is host rounding and not a difference any cell of the lane's can show.
//
// The rows are every row the report names over the grid, in all three
// precisions: the launched pair on both packing axes at each arm of each lane,
// and the device-callable pair reached through the consumer's kernels where the
// lane reports one. The float and double lanes carry twelve rows - the four
// arms of their cross, each on the argument axis, the orders axis and the
// device-callable entry, which is all-orders-fp64-orders-uniform and
// all-orders-f64-rat-horner-orders-uniform among the orders-axis rows - and the
// half lane eight, its four arms on the two launched axes, the half lane having
// no device-callable uniform/narrow pair over the grid to reach. An arm left
// out would be a row of the report that nothing below asks this of.
// ---------------------------------------------------------------------------

/// One carriage row: the entry's own reading against the narrow member's of the
/// same arm, cell for cell, region by region - or, where the row names a body
/// (`namesBody`), against the float body it names, the row being that body with
/// the half lane's argument and return.
struct PartitionCarriageRow {
    std::string entry;
    std::array<std::size_t, 4> refDiffer{};
    std::array<std::size_t, 4> cells{};
    std::array<std::size_t, 4> differ{};
    bool namesBody = false;
};

/// The per-argument host entry's own separation at one arm, which is the
/// reference the arm's rows are held to. `bodies` is the half lane's: the same
/// two bodies at fp32, where the arm's rows name float bodies and the lane's
/// own format is the coarser of the two - recorded only there, the float and
/// double lanes' references being read at their own resolution.
struct PartitionCarriageRef {
    std::string axes;
    std::array<std::size_t, 4> differ{};
    std::array<std::size_t, 4> bodies{};
    bool bodiesRecorded = false;
};

std::vector<PartitionCarriageRow>& PartitionCarriageRows() {
    static std::vector<PartitionCarriageRow> rows;
    return rows;
}

std::vector<PartitionCarriageRef>& PartitionCarriageRefs() {
    static std::vector<PartitionCarriageRef> refs;
    return refs;
}

/// The regions in the CPU gate's vocabulary, in its order.
const char* const kCarriageRegionTag[4] = {"A", "band", "B", "C"};

/// The launched entries' shape at one lane: a batch of arguments in, one value
/// per order per argument out. The value and the argument are the lane's - the
/// fp64 rows carry doubles on the entries' own argument and the fp16 rows the
/// half format on the half lane's - so one shape serves the three.
template <typename Value, typename Arg>
using LaunchAt = boys::BoysStatus (*)(const int*, const Arg*, Value*, std::size_t, void*,
                                      boys::DivisionForm);

/// The device-callable entries' shape at one lane: the handle, the factor pair
/// a consumer's kernel forms its own argument from, and a block per element.
template <typename Value>
using DemoAt = int (*)(const boys::BoysDeviceTables*, const int*, const double*, const double*,
                       Value*, std::size_t, int, int*);

/// The per-argument host entry's own separation at one arm: the cells where
/// naming the grid and naming the narrow member return different values. The
/// two readers are that entry at the arm's two granularities, read at the
/// argument the lane evaluates at, and the two values are compared in the
/// format the arm's own rows return.
template <typename Value, typename GridRead, typename NarrowRead>
std::array<std::size_t, 4> CarriageSeparation(const Reference& ref,
                                              const std::vector<double>& args,
                                              GridRead grid,
                                              NarrowRead narrow) {
    std::array<std::size_t, 4> differ{};

    for (std::size_t i = 0; i < ref.count; ++i) {
        const std::size_t region = static_cast<std::size_t>(SingleClaim(ref.x[i]));

        for (int n = 0; n <= boys::kMaxBoysOrder; ++n) {
            const Value got = grid(n, args[i]);
            const Value other = narrow(n, args[i]);

            if (std::memcmp(&got, &other, sizeof(Value)) != 0) {
                ++differ[region];
            }
        }
    }

    return differ;
}

/// The float lane's per-argument entry at one arm, in its own argument and its
/// own return: the two policies name the lane's grid and its narrow member.
template <boys::FitRoute kRoute, boys::EvalScheme kScheme>
std::array<std::size_t, 4> F32Separation(const Reference& ref) {
    using GridReading = boys::EvalPolicy<kRoute, kScheme, boys::BoysBudget::kFloat,
                                         boys::PackAxis::kArguments,
                                         boys::FitGranularity::kUniform>;
    using NarrowReading = boys::EvalPolicy<kRoute, kScheme, boys::BoysBudget::kFloat,
                                           boys::PackAxis::kArguments,
                                           boys::FitGranularity::kNarrow>;

    return CarriageSeparation<float>(
        ref,
        ref.xf,
        [](int n, double x) {
            return boys::BoysSingleF32<GridReading>(n, static_cast<float>(x));
        },
        [](int n, double x) {
            return boys::BoysSingleF32<NarrowReading>(n, static_cast<float>(x));
        });
}

/// The double lane's, at the same arm and the same two granularities. The
/// budget axis is inert on this lane, which is why the policy names kFloat as
/// the lane's own rows do.
template <boys::FitRoute kRoute, boys::EvalScheme kScheme>
std::array<std::size_t, 4> F64Separation(const Reference& ref) {
    using GridReading = boys::EvalPolicy<kRoute, kScheme, boys::BoysBudget::kFloat,
                                         boys::PackAxis::kArguments,
                                         boys::FitGranularity::kUniform>;
    using NarrowReading = boys::EvalPolicy<kRoute, kScheme, boys::BoysBudget::kFloat,
                                           boys::PackAxis::kArguments,
                                           boys::FitGranularity::kNarrow>;

    return CarriageSeparation<double>(
        ref,
        ref.x,
        [](int n, double x) { return boys::BoysSingle<GridReading>(n, x); },
        [](int n, double x) { return boys::BoysSingle<NarrowReading>(n, x); });
}

/// The half lane's, at the same arm: the entry is the half lane's own and it is
/// read at the argument that lane evaluates at, the half value the grid carries
/// in x16. The two readings are compared as half values, which is what the rows
/// this reference holds return.
template <boys::FitRoute kRoute, boys::EvalScheme kScheme>
std::array<std::size_t, 4> F16Separation(const Reference& ref) {
    using GridReading = boys::EvalPolicy<kRoute, kScheme, boys::BoysBudget::kFp16,
                                         boys::PackAxis::kArguments,
                                         boys::FitGranularity::kUniform>;
    using NarrowReading = boys::EvalPolicy<kRoute, kScheme, boys::BoysBudget::kFp16,
                                           boys::PackAxis::kArguments,
                                           boys::FitGranularity::kNarrow>;

    return CarriageSeparation<boys::F16>(
        ref,
        ref.x16,
        [](int n, double x) {
            return boys::BoysSingleF16<GridReading>(n, boys::F16(static_cast<float>(x)));
        },
        [](int n, double x) {
            return boys::BoysSingleF16<NarrowReading>(n, boys::F16(static_cast<float>(x)));
        });
}

/// One arm's reference line, recorded: the lane, the route and the scheme the
/// separation was measured at, which is what the line names it by, and - on the
/// half lane - the same two bodies' separation where the reference is measured
/// at the lane's own resolution rather than at the bodies'.
void RecordPartitionCarriageRef(const char* precision,
                                boys::FitRoute route,
                                boys::EvalScheme scheme,
                                const std::array<std::size_t, 4>& differ,
                                const std::array<std::size_t, 4>* bodies = nullptr) {
    PartitionCarriageRef reference;
    reference.axes = (std::string(precision) + ", " + RouteName(route) + ", "
                      + boys::EvalSchemeName(scheme));
    reference.differ = differ;

    if (bodies != nullptr) {
        reference.bodies = *bodies;
        reference.bodiesRecorded = true;
    }

    PartitionCarriageRefs().push_back(reference);
}

/// One row recorded. The slot function is the row's own layout: the launched
/// rows write the order-major cell indexing, the device-callable ones a block
/// per element. `namesBody` says the second reading is the float body the row
/// names rather than the narrow member's own row.
template <typename Value, typename Slot>
void RecordPartitionCarriage(const std::string& entry,
                             const std::array<std::size_t, 4>& refDiffer,
                             const Reference& ref,
                             const std::vector<Value>& uniform,
                             const std::vector<Value>& narrow,
                             Slot slot,
                             bool namesBody = false) {
    PartitionCarriageRow row;
    row.entry = entry;
    row.refDiffer = refDiffer;
    row.namesBody = namesBody;

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n) {
        for (std::size_t i = 0; i < ref.count; ++i) {
            const std::size_t e = ref.Index(n, i);
            const std::size_t region = static_cast<std::size_t>(SingleClaim(ref.x[i]));
            ++row.cells[region];

            if (std::memcmp(&uniform[slot(e, n)], &narrow[slot(e, n)], sizeof(Value)) != 0) {
                ++row.differ[region];
            }
        }
    }

    PartitionCarriageRows().push_back(std::move(row));
}

/// One launched row's cells over the grid, with no claim attached: a carriage
/// row is not a bound claim and is counted in the section of its own.
///
/// The entry is handed the lane's argument list - the first `count` elements of
/// the array the two entries below are launched over - and writes one value per
/// order per argument, which is the layout the cells are indexed in.
template <typename Value, typename Arg>
std::vector<Value> LaunchCarriageGrid(const std::vector<Arg>& args,
                                      std::size_t count,
                                      std::size_t cells,
                                      const char* what,
                                      LaunchAt<Value, Arg> launch) {
    DevBuf<int> dN(count);
    DevBuf<Arg> dX(count);
    DevBuf<Value> dOut(cells);
    const std::vector<int> tops(count, boys::kMaxBoysOrder);
    dN.Upload(tops);
    dX.Upload(args);
    CheckLaunch(launch(dN.get(), dX.get(), dOut.get(), count, nullptr, kGateDivisionForm), what);
    std::vector<Value> out(cells);
    dOut.Download(out);
    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return out;
}

/// One device-callable row's cells, through the consumer kernel that reaches
/// it: the family layout, one block per element.
template <typename Value>
std::vector<Value> LaunchCarriageDemo(const boys::BoysDeviceTables& tables,
                                      const Grid& grid,
                                      const char* what,
                                      DemoAt<Value> demo) {
    const std::size_t cells = grid.cells;
    const std::size_t family = cells * (static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);
    DevBuf<int> dN(cells);
    DevBuf<double> dRho(cells);
    DevBuf<double> dD2(cells);
    DevBuf<Value> dOut(family);
    DevBuf<int> dStatus(cells);
    dN.Upload(grid.n);
    dRho.Upload(grid.rho);
    dD2.Upload(grid.d2);
    dOut.Upload(std::vector<Value>(family, Value{}));
    dStatus.Upload(std::vector<int>(cells, -1));
    CheckDemo(demo(&tables,
                   dN.get(),
                   dRho.get(),
                   dD2.get(),
                   dOut.get(),
                   cells,
                   boys::kMaxBoysOrder + 1,
                   dStatus.get()),
              what);
    std::vector<Value> out(family);
    std::vector<int> status(cells);
    dOut.Download(out);
    dStatus.Download(status);
    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    ExpectStatus(status, cells, BoysDeviceDemoStatusSuccess(), what);
    return out;
}

/// One arm's launched rows: the pair on the arguments axis and the pair on the
/// orders axis. The rows are held to the arm's own reference, which the caller
/// computes once: the two axes are one arm's and not two.
///
/// Where the arm's row names a float body - the half lane's shape - the body's
/// values in the row's own format are handed in, and the row is read against
/// that body rather than against the narrow member's row. The narrow entry and
/// its launcher are the pair's other name and are not reached there.
template <typename Value, typename Arg>
void PartitionCarriageLaunchedRows(const Reference& ref,
                                   const Grid& grid,
                                   const std::array<std::size_t, 4>& refDiffer,
                                   const std::vector<Arg>& args,
                                   boys::DeviceEntry uniformArgs,
                                   LaunchAt<Value, Arg> uniformArgsLaunch,
                                   boys::DeviceEntry narrowArgs,
                                   LaunchAt<Value, Arg> narrowArgsLaunch,
                                   boys::DeviceEntry uniformOrders,
                                   LaunchAt<Value, Arg> uniformOrdersLaunch,
                                   boys::DeviceEntry narrowOrders,
                                   LaunchAt<Value, Arg> narrowOrdersLaunch,
                                   const std::vector<Value>* uniformArgsBody = nullptr,
                                   const std::vector<Value>* uniformOrdersBody = nullptr) {
    // The launched rows write the order-major cell indexing the report's rows
    // are measured in, so the cell index is the slot.
    const auto cellSlot = [](std::size_t e, int) { return e; };

    {
        const std::string uniformName = Label(DeviceRow(uniformArgs).name);
        const std::vector<Value> u = LaunchCarriageGrid<Value, Arg>(
            args, ref.count, grid.cells, uniformName.c_str(), uniformArgsLaunch);

        if (uniformArgsBody != nullptr) {
            RecordPartitionCarriage<Value>(
                uniformName, refDiffer, ref, u, *uniformArgsBody, cellSlot, true);
        } else {
            const std::string narrowName = Label(DeviceRow(narrowArgs).name);
            const std::vector<Value> v = LaunchCarriageGrid<Value, Arg>(
                args, ref.count, grid.cells, narrowName.c_str(), narrowArgsLaunch);

            RecordPartitionCarriage<Value>(uniformName, refDiffer, ref, u, v, cellSlot);
        }
    }

    // The orders axis: the same reference, the arm's other reading of region A.
    {
        const std::string uniformName = Label(DeviceRow(uniformOrders).name);
        const std::vector<Value> u = LaunchCarriageGrid<Value, Arg>(
            args, ref.count, grid.cells, uniformName.c_str(), uniformOrdersLaunch);

        if (uniformOrdersBody != nullptr) {
            RecordPartitionCarriage<Value>(
                uniformName, refDiffer, ref, u, *uniformOrdersBody, cellSlot, true);
        } else {
            const std::string narrowName = Label(DeviceRow(narrowOrders).name);
            const std::vector<Value> v = LaunchCarriageGrid<Value, Arg>(
                args, ref.count, grid.cells, narrowName.c_str(), narrowOrdersLaunch);

            RecordPartitionCarriage<Value>(uniformName, refDiffer, ref, u, v, cellSlot);
        }
    }
}

/// One arm's device-callable rows, where the lane reports a pair: the same
/// reference, reached through the consumer's kernel in the family layout.
template <typename Value>
void PartitionCarriageDemoRows(const Reference& ref,
                               const boys::BoysDeviceTables& tables,
                               const Grid& grid,
                               const std::array<std::size_t, 4>& refDiffer,
                               boys::DeviceEntry uniformDemo,
                               DemoAt<Value> uniformDemoLaunch,
                               boys::DeviceEntry narrowDemo,
                               DemoAt<Value> narrowDemoLaunch) {
    const std::string uniformName = Label(DeviceRow(uniformDemo).name);
    const std::string narrowName = Label(DeviceRow(narrowDemo).name);
    const std::vector<Value> u =
        LaunchCarriageDemo<Value>(tables, grid, uniformName.c_str(), uniformDemoLaunch);
    const std::vector<Value> v =
        LaunchCarriageDemo<Value>(tables, grid, narrowName.c_str(), narrowDemoLaunch);

    RecordPartitionCarriage<Value>(uniformName, refDiffer, ref, u, v, FamilySlot);
}

/// One arm's rows: the launched pair on the arguments axis, the launched pair
/// on the orders axis, and - where the lane reports one - the device-callable
/// pair. The three are one arm's and not three: the reference they are held to
/// is that arm's own, which the caller measures once and hands in.
///
/// On the half lane the two bodies the rows name are handed in with the arm:
/// each row is read against the body it names rather than against the narrow
/// member's row, and the bodies' own separation is recorded with the reference.
template <typename Value, typename Arg, boys::FitRoute kRoute, boys::EvalScheme kScheme>
void PartitionCarriageArm(const char* precision,
                          const Reference& ref,
                          const Grid& grid,
                          const std::vector<Arg>& args,
                          const std::array<std::size_t, 4>& refDiffer,
                          const boys::BoysDeviceTables& tables,
                          boys::DeviceEntry uniformArgs,
                          LaunchAt<Value, Arg> uniformArgsLaunch,
                          boys::DeviceEntry narrowArgs,
                          LaunchAt<Value, Arg> narrowArgsLaunch,
                          boys::DeviceEntry uniformOrders,
                          LaunchAt<Value, Arg> uniformOrdersLaunch,
                          boys::DeviceEntry narrowOrders,
                          LaunchAt<Value, Arg> narrowOrdersLaunch,
                          boys::DeviceEntry uniformDemo,
                          DemoAt<Value> uniformDemoLaunch,
                          boys::DeviceEntry narrowDemo,
                          DemoAt<Value> narrowDemoLaunch,
                          const std::vector<Value>* uniformArgsBody = nullptr,
                          const std::vector<Value>* uniformOrdersBody = nullptr,
                          const std::array<std::size_t, 4>* bodies = nullptr) {
    RecordPartitionCarriageRef(precision, kRoute, kScheme, refDiffer, bodies);

    PartitionCarriageLaunchedRows<Value, Arg>(ref,
                                              grid,
                                              refDiffer,
                                              args,
                                              uniformArgs,
                                              uniformArgsLaunch,
                                              narrowArgs,
                                              narrowArgsLaunch,
                                              uniformOrders,
                                              uniformOrdersLaunch,
                                              narrowOrders,
                                              narrowOrdersLaunch,
                                              uniformArgsBody,
                                              uniformOrdersBody);

    // The half lane's report carries no device-callable row over the grid, and
    // a launcher it does not have is the only way to say so: the pair is
    // reached where the lane reports one and nowhere else.
    if (uniformDemoLaunch != nullptr && narrowDemoLaunch != nullptr) {
        PartitionCarriageDemoRows<Value>(ref,
                                         tables,
                                         grid,
                                         refDiffer,
                                         uniformDemo,
                                         uniformDemoLaunch,
                                         narrowDemo,
                                         narrowDemoLaunch);
    }
}

/// Every fp32 row of the report that answers a policy naming the uniform grid,
/// read against the narrow member of its own arm. The four arms are the lane's:
/// each route at each scheme, which is the cross the report's float block is
/// written over, both packing axes and both reaches (launched and
/// device-callable).
void PartitionCarriageF32(const Reference& ref,
                          const Grid& grid,
                          const boys::BoysDeviceTables& tables) {
    PartitionCarriageArm<float,
                         double,
                         boys::FitRoute::kChebyshev,
                         boys::EvalScheme::kSplitClenshaw>(
        "fp32",
        ref,
        grid,
        ref.x,
        F32Separation<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw>(ref),
        tables,
        boys::DeviceEntry::kAllOrdersF32Uniform,
        &boys::BoysCuda::AllOrdersF32Uniform,
        boys::DeviceEntry::kAllOrdersF32Narrow,
        &boys::BoysCuda::AllOrdersF32Narrow,
        boys::DeviceEntry::kAllOrdersF32OrdersUniform,
        &boys::BoysCuda::AllOrdersF32OrdersUniform,
        boys::DeviceEntry::kAllOrdersF32NarrowOrders,
        &boys::BoysCuda::AllOrdersF32NarrowOrders,
        boys::DeviceEntry::kDeviceAllOrdersF32Uniform,
        &BoysDeviceDemoLadder32Uniform,
        boys::DeviceEntry::kDeviceAllOrdersF32Narrow,
        &BoysDeviceDemoLadder32Narrow);

    PartitionCarriageArm<float, double, boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>(
        "fp32",
        ref,
        grid,
        ref.x,
        F32Separation<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>(ref),
        tables,
        boys::DeviceEntry::kAllOrdersF32UniformHorner,
        &boys::BoysCuda::AllOrdersF32UniformHorner,
        boys::DeviceEntry::kAllOrdersF32NarrowMono,
        &boys::BoysCuda::AllOrdersF32NarrowMono,
        boys::DeviceEntry::kAllOrdersF32OrdersUniformHorner,
        &boys::BoysCuda::AllOrdersF32OrdersUniformHorner,
        boys::DeviceEntry::kAllOrdersF32NarrowOrdersMono,
        &boys::BoysCuda::AllOrdersF32NarrowOrdersMono,
        boys::DeviceEntry::kDeviceAllOrdersF32UniformHorner,
        &BoysDeviceDemoLadder32UniformHorner,
        boys::DeviceEntry::kDeviceAllOrdersF32NarrowMono,
        &BoysDeviceDemoLadder32NarrowMono);

    PartitionCarriageArm<float,
                         double,
                         boys::FitRoute::kRationalMinimax,
                         boys::EvalScheme::kSplitClenshaw>(
        "fp32",
        ref,
        grid,
        ref.x,
        F32Separation<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw>(ref),
        tables,
        boys::DeviceEntry::kAllOrdersF32UniformRat,
        &boys::BoysCuda::AllOrdersF32UniformRat,
        boys::DeviceEntry::kAllOrdersF32NarrowRat,
        &boys::BoysCuda::AllOrdersF32NarrowRat,
        boys::DeviceEntry::kAllOrdersF32OrdersUniformRat,
        &boys::BoysCuda::AllOrdersF32OrdersUniformRat,
        boys::DeviceEntry::kAllOrdersF32NarrowOrdersRat,
        &boys::BoysCuda::AllOrdersF32NarrowOrdersRat,
        boys::DeviceEntry::kDeviceAllOrdersF32UniformRat,
        &BoysDeviceDemoLadder32UniformRat,
        boys::DeviceEntry::kDeviceAllOrdersF32NarrowRat,
        &BoysDeviceDemoLadder32NarrowRat);

    PartitionCarriageArm<float, double, boys::FitRoute::kRationalMinimax, boys::EvalScheme::kHorner>(
        "fp32",
        ref,
        grid,
        ref.x,
        F32Separation<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kHorner>(ref),
        tables,
        boys::DeviceEntry::kAllOrdersF32UniformRatHorner,
        &boys::BoysCuda::AllOrdersF32UniformRatHorner,
        boys::DeviceEntry::kAllOrdersF32NarrowRatHorner,
        &boys::BoysCuda::AllOrdersF32NarrowRatHorner,
        boys::DeviceEntry::kAllOrdersF32OrdersUniformRatHorner,
        &boys::BoysCuda::AllOrdersF32OrdersUniformRatHorner,
        boys::DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner,
        &boys::BoysCuda::AllOrdersF32NarrowOrdersRatHorner,
        boys::DeviceEntry::kDeviceAllOrdersF32UniformRatHorner,
        &BoysDeviceDemoLadder32UniformRatHorner,
        boys::DeviceEntry::kDeviceAllOrdersF32NarrowRatHorner,
        &BoysDeviceDemoLadder32NarrowRatHorner);
}

/// The fp64 lane's, at the same four arms and the same two reaches. The
/// launched entries carry doubles on their own argument and the device-callable
/// ones the factor pair, so the argument vector is the reference's own.
void PartitionCarriageF64(const Reference& ref,
                          const Grid& grid,
                          const boys::BoysDeviceTables& tables) {
    PartitionCarriageArm<double,
                         double,
                         boys::FitRoute::kChebyshev,
                         boys::EvalScheme::kSplitClenshaw>(
        "fp64",
        ref,
        grid,
        ref.x,
        F64Separation<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw>(ref),
        tables,
        boys::DeviceEntry::kAllOrdersF64Uniform,
        &boys::BoysCuda::AllOrdersF64Uniform,
        boys::DeviceEntry::kAllOrdersF64Narrow,
        &boys::BoysCuda::AllOrdersF64Narrow,
        boys::DeviceEntry::kAllOrdersF64OrdersUniform,
        &boys::BoysCuda::AllOrdersF64OrdersUniform,
        boys::DeviceEntry::kAllOrdersF64NarrowOrders,
        &boys::BoysCuda::AllOrdersF64NarrowOrders,
        boys::DeviceEntry::kDeviceAllOrdersF64Uniform,
        &BoysDeviceDemoLadder64Uniform,
        boys::DeviceEntry::kDeviceAllOrdersF64Narrow,
        &BoysDeviceDemoLadder64Narrow);

    PartitionCarriageArm<double, double, boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>(
        "fp64",
        ref,
        grid,
        ref.x,
        F64Separation<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>(ref),
        tables,
        boys::DeviceEntry::kAllOrdersF64UniformHorner,
        &boys::BoysCuda::AllOrdersF64UniformHorner,
        boys::DeviceEntry::kAllOrdersF64NarrowMono,
        &boys::BoysCuda::AllOrdersF64NarrowMono,
        boys::DeviceEntry::kAllOrdersF64OrdersUniformHorner,
        &boys::BoysCuda::AllOrdersF64OrdersUniformHorner,
        boys::DeviceEntry::kAllOrdersF64NarrowOrdersMono,
        &boys::BoysCuda::AllOrdersF64NarrowOrdersMono,
        boys::DeviceEntry::kDeviceAllOrdersF64UniformHorner,
        &BoysDeviceDemoLadder64UniformHorner,
        boys::DeviceEntry::kDeviceAllOrdersF64NarrowMono,
        &BoysDeviceDemoLadder64NarrowMono);

    PartitionCarriageArm<double,
                         double,
                         boys::FitRoute::kRationalMinimax,
                         boys::EvalScheme::kSplitClenshaw>(
        "fp64",
        ref,
        grid,
        ref.x,
        F64Separation<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw>(ref),
        tables,
        boys::DeviceEntry::kAllOrdersF64UniformRat,
        &boys::BoysCuda::AllOrdersF64UniformRat,
        boys::DeviceEntry::kAllOrdersF64NarrowRat,
        &boys::BoysCuda::AllOrdersF64NarrowRat,
        boys::DeviceEntry::kAllOrdersF64OrdersUniformRat,
        &boys::BoysCuda::AllOrdersF64OrdersUniformRat,
        boys::DeviceEntry::kAllOrdersF64NarrowOrdersRat,
        &boys::BoysCuda::AllOrdersF64NarrowOrdersRat,
        boys::DeviceEntry::kDeviceAllOrdersF64UniformRat,
        &BoysDeviceDemoLadder64UniformRat,
        boys::DeviceEntry::kDeviceAllOrdersF64NarrowRat,
        &BoysDeviceDemoLadder64NarrowRat);

    PartitionCarriageArm<double,
                         double,
                         boys::FitRoute::kRationalMinimax,
                         boys::EvalScheme::kHorner>(
        "fp64",
        ref,
        grid,
        ref.x,
        F64Separation<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kHorner>(ref),
        tables,
        boys::DeviceEntry::kAllOrdersF64UniformRatHorner,
        &boys::BoysCuda::AllOrdersF64UniformRatHorner,
        boys::DeviceEntry::kAllOrdersF64NarrowRatHorner,
        &boys::BoysCuda::AllOrdersF64NarrowRat,
        boys::DeviceEntry::kAllOrdersF64OrdersUniformRatHorner,
        &boys::BoysCuda::AllOrdersF64OrdersUniformRatHorner,
        boys::DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner,
        &boys::BoysCuda::AllOrdersF64NarrowOrdersRat,
        boys::DeviceEntry::kDeviceAllOrdersF64UniformRatHorner,
        &BoysDeviceDemoLadder64UniformRatHorner,
        boys::DeviceEntry::kDeviceAllOrdersF64NarrowRatHorner,
        &BoysDeviceDemoLadder64NarrowRatHorner);
}

#if BoysFp16
/// The two float bodies one arm's half rows name, at the half lane's own
/// arguments: counted per region, the cells where the two differ at fp32 - the
/// separation the float lane's own rows carry - and the cells where their half
/// images differ, which is where a row answering from the other member's fits
/// would part from the body it names.
struct HalfBodies {
    std::array<std::size_t, 4> halves{};
    std::array<std::size_t, 4> bodies{};
};

/// The two bodies of one axis, read from the device pair of the same arm
/// launched at the half lane's own arguments.
HalfBodies HalfBodiesOf(const Reference& ref,
                        const std::vector<float>& uniform,
                        const std::vector<float>& narrow) {
    HalfBodies counted;

    for (std::size_t i = 0; i < ref.count; ++i) {
        const std::size_t region = static_cast<std::size_t>(SingleClaim(ref.x[i]));

        for (int n = 0; n <= boys::kMaxBoysOrder; ++n) {
            const std::size_t e = ref.Index(n, i);

            if (std::memcmp(&uniform[e], &narrow[e], sizeof(float)) != 0) {
                ++counted.bodies[region];
            }

            const boys::F16 got = boys::F16(uniform[e]);
            const boys::F16 other = boys::F16(narrow[e]);

            if (std::memcmp(&got, &other, sizeof(boys::F16)) != 0) {
                ++counted.halves[region];
            }
        }
    }

    return counted;
}

/// One half-lane arm. The lane's rows name the float lane's bodies with half
/// I/O, so the arm is handed those bodies' launchers: the pair is launched at
/// the half lane's own arguments, the body's half image is what each row is
/// read against, and the bodies' own separation is the reference recorded with
/// the arm. That reference is read at the lane's resolution, because a rule
/// reading this lane's rows cannot hold a difference the format has already
/// rounded away; the fp32 count recorded beside it is the same two bodies'
/// separation, which is what the float lane's own rows carry.
template <boys::FitRoute kRoute, boys::EvalScheme kScheme>
void PartitionCarriageHalfArm(const Reference& ref,
                              const Grid& grid,
                              const std::vector<boys::F16>& args,
                              const std::vector<double>& halfArgs,
                              const boys::BoysDeviceTables& tables,
                              boys::DeviceEntry uniformArgs,
                              LaunchAt<boys::F16, boys::F16> uniformArgsLaunch,
                              boys::DeviceEntry narrowArgs,
                              LaunchAt<boys::F16, boys::F16> narrowArgsLaunch,
                              boys::DeviceEntry uniformOrders,
                              LaunchAt<boys::F16, boys::F16> uniformOrdersLaunch,
                              boys::DeviceEntry narrowOrders,
                              LaunchAt<boys::F16, boys::F16> narrowOrdersLaunch,
                              boys::DeviceEntry bodyArgs,
                              LaunchAt<float, double> bodyArgsLaunch,
                              boys::DeviceEntry bodyNarrowArgs,
                              LaunchAt<float, double> bodyNarrowArgsLaunch,
                              boys::DeviceEntry bodyOrders,
                              LaunchAt<float, double> bodyOrdersLaunch,
                              boys::DeviceEntry bodyNarrowOrders,
                              LaunchAt<float, double> bodyNarrowOrdersLaunch) {
    const std::string bodyArgsName = Label(DeviceRow(bodyArgs).name);
    const std::string bodyNarrowArgsName = Label(DeviceRow(bodyNarrowArgs).name);
    const std::string bodyOrdersName = Label(DeviceRow(bodyOrders).name);
    const std::string bodyNarrowOrdersName = Label(DeviceRow(bodyNarrowOrders).name);

    const std::vector<float> ua = LaunchCarriageGrid<float, double>(
        halfArgs, ref.count, grid.cells, bodyArgsName.c_str(), bodyArgsLaunch);
    const std::vector<float> na = LaunchCarriageGrid<float, double>(
        halfArgs, ref.count, grid.cells, bodyNarrowArgsName.c_str(), bodyNarrowArgsLaunch);
    const std::vector<float> uo = LaunchCarriageGrid<float, double>(
        halfArgs, ref.count, grid.cells, bodyOrdersName.c_str(), bodyOrdersLaunch);
    const std::vector<float> no = LaunchCarriageGrid<float, double>(
        halfArgs, ref.count, grid.cells, bodyNarrowOrdersName.c_str(), bodyNarrowOrdersLaunch);

    const HalfBodies argsBodies = HalfBodiesOf(ref, ua, na);
    const HalfBodies ordersBodies = HalfBodiesOf(ref, uo, no);

    std::vector<boys::F16> argsImage(grid.cells);
    std::vector<boys::F16> ordersImage(grid.cells);

    for (std::size_t e = 0; e < grid.cells; ++e) {
        argsImage[e] = boys::F16(ua[e]);
        ordersImage[e] = boys::F16(uo[e]);
    }

    PartitionCarriageArm<boys::F16,
                         boys::F16,
                         kRoute,
                         kScheme>("fp16",
                                  ref,
                                  grid,
                                  args,
                                  argsBodies.halves,
                                  tables,
                                  uniformArgs,
                                  uniformArgsLaunch,
                                  narrowArgs,
                                  narrowArgsLaunch,
                                  uniformOrders,
                                  uniformOrdersLaunch,
                                  narrowOrders,
                                  narrowOrdersLaunch,
                                  boys::DeviceEntry::kAllOrdersF16Uniform,
                                  nullptr,
                                  boys::DeviceEntry::kAllOrdersF16Narrow,
                                  nullptr,
                                  &argsImage,
                                  &ordersImage,
                                  &argsBodies.bodies);
}

/// The fp16 lane's, at the same four arms: the launched row on each packing
/// axis, held to the float body it names, cell for cell, at the lane's own
/// arguments - the body's values in the row's format are the row's second
/// reading. The lane carries no device-callable row of the uniform pair, so
/// none is reached here. The argument vector is the half value the lane's own
/// rows are launched on, which is the value the reference's x16 column holds;
/// the float bodies are launched at those same values, widened, because that is
/// the argument list the lane the rows belong to is handed.
void PartitionCarriageF16(const Reference& ref,
                          const Grid& grid,
                          const boys::BoysDeviceTables& tables) {
    std::vector<boys::F16> args(ref.count);
    std::vector<double> halfArgs(ref.count);

    for (std::size_t i = 0; i < ref.count; ++i) {
        args[i] = boys::F16(static_cast<float>(ref.x[i]));
        halfArgs[i] = static_cast<double>(args[i]);
    }

    PartitionCarriageHalfArm<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw>(
        ref,
        grid,
        args,
        halfArgs,
        tables,
        boys::DeviceEntry::kAllOrdersF16Uniform,
        &boys::BoysCuda::AllOrdersF16Uniform,
        boys::DeviceEntry::kAllOrdersF16Narrow,
        &boys::BoysCuda::AllOrdersF16Narrow,
        boys::DeviceEntry::kAllOrdersF16OrdersUniform,
        &boys::BoysCuda::AllOrdersF16OrdersUniform,
        boys::DeviceEntry::kAllOrdersF16NarrowOrders,
        &boys::BoysCuda::AllOrdersF16NarrowOrders,
        boys::DeviceEntry::kAllOrdersF32Uniform,
        &boys::BoysCuda::AllOrdersF32Uniform,
        boys::DeviceEntry::kAllOrdersF32Narrow,
        &boys::BoysCuda::AllOrdersF32Narrow,
        boys::DeviceEntry::kAllOrdersF32OrdersUniform,
        &boys::BoysCuda::AllOrdersF32OrdersUniform,
        boys::DeviceEntry::kAllOrdersF32NarrowOrders,
        &boys::BoysCuda::AllOrdersF32NarrowOrders);

    PartitionCarriageHalfArm<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>(
        ref,
        grid,
        args,
        halfArgs,
        tables,
        boys::DeviceEntry::kAllOrdersF16UniformHorner,
        &boys::BoysCuda::AllOrdersF16UniformHorner,
        boys::DeviceEntry::kAllOrdersF16NarrowMono,
        &boys::BoysCuda::AllOrdersF16NarrowMono,
        boys::DeviceEntry::kAllOrdersF16OrdersUniformHorner,
        &boys::BoysCuda::AllOrdersF16OrdersUniformHorner,
        boys::DeviceEntry::kAllOrdersF16NarrowOrdersMono,
        &boys::BoysCuda::AllOrdersF16NarrowOrdersMono,
        boys::DeviceEntry::kAllOrdersF32UniformHorner,
        &boys::BoysCuda::AllOrdersF32UniformHorner,
        boys::DeviceEntry::kAllOrdersF32NarrowMono,
        &boys::BoysCuda::AllOrdersF32NarrowMono,
        boys::DeviceEntry::kAllOrdersF32OrdersUniformHorner,
        &boys::BoysCuda::AllOrdersF32OrdersUniformHorner,
        boys::DeviceEntry::kAllOrdersF32NarrowOrdersMono,
        &boys::BoysCuda::AllOrdersF32NarrowOrdersMono);

    PartitionCarriageHalfArm<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw>(
        ref,
        grid,
        args,
        halfArgs,
        tables,
        boys::DeviceEntry::kAllOrdersF16UniformRat,
        &boys::BoysCuda::AllOrdersF16UniformRat,
        boys::DeviceEntry::kAllOrdersF16NarrowRat,
        &boys::BoysCuda::AllOrdersF16NarrowRat,
        boys::DeviceEntry::kAllOrdersF16OrdersUniformRat,
        &boys::BoysCuda::AllOrdersF16OrdersUniformRat,
        boys::DeviceEntry::kAllOrdersF16NarrowOrdersRat,
        &boys::BoysCuda::AllOrdersF16NarrowOrdersRat,
        boys::DeviceEntry::kAllOrdersF32UniformRat,
        &boys::BoysCuda::AllOrdersF32UniformRat,
        boys::DeviceEntry::kAllOrdersF32NarrowRat,
        &boys::BoysCuda::AllOrdersF32NarrowRat,
        boys::DeviceEntry::kAllOrdersF32OrdersUniformRat,
        &boys::BoysCuda::AllOrdersF32OrdersUniformRat,
        boys::DeviceEntry::kAllOrdersF32NarrowOrdersRat,
        &boys::BoysCuda::AllOrdersF32NarrowOrdersRat);

    PartitionCarriageHalfArm<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kHorner>(
        ref,
        grid,
        args,
        halfArgs,
        tables,
        boys::DeviceEntry::kAllOrdersF16UniformRatHorner,
        &boys::BoysCuda::AllOrdersF16UniformRatHorner,
        boys::DeviceEntry::kAllOrdersF16NarrowRatHorner,
        &boys::BoysCuda::AllOrdersF16NarrowRatHorner,
        boys::DeviceEntry::kAllOrdersF16OrdersUniformRatHorner,
        &boys::BoysCuda::AllOrdersF16OrdersUniformRatHorner,
        boys::DeviceEntry::kAllOrdersF16NarrowOrdersRatHorner,
        &boys::BoysCuda::AllOrdersF16NarrowOrdersRatHorner,
        boys::DeviceEntry::kAllOrdersF32UniformRatHorner,
        &boys::BoysCuda::AllOrdersF32UniformRatHorner,
        boys::DeviceEntry::kAllOrdersF32NarrowRatHorner,
        &boys::BoysCuda::AllOrdersF32NarrowRatHorner,
        boys::DeviceEntry::kAllOrdersF32OrdersUniformRatHorner,
        &boys::BoysCuda::AllOrdersF32OrdersUniformRatHorner,
        boys::DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner,
        &boys::BoysCuda::AllOrdersF32NarrowOrdersRatHorner);
}
#endif // BoysFp16

/// The carriage section: the arms' references, the rows, and the verdict. A row
/// is held to every region where the per-argument entry's two readings differ,
/// and a row whose readings coincide over all of such a region is a row that
/// answered that region from the narrow member's fits.
///
/// Returns the number of rows that failed, which the caller's verdict carries.
std::size_t PrintPartitionCarriage() {
    const std::vector<PartitionCarriageRow>& rows = PartitionCarriageRows();

    if (rows.empty()) {
        return 0;
    }

    std::printf("\nthe uniform grid's carriage on the device rows: every row of the report that "
                "answers a\npolicy naming the grid is read twice in one pass, and the cells where "
                "the two readings\ndiffer are counted per region - which is the rule the CPU gate "
                "holds its own rows to. The\ntwo partitions hold the same bar over the same "
                "intervals, so no accuracy row can tell\nthem apart: a row that read the narrow "
                "member's tables under the grid's name would print\nthe same numbers inside the "
                "same bound.\n");

    std::printf("\n  the per-argument host entry's own separation, the reference the float and "
                "double\n  lanes' rows are held to: a region where the entry's own two readings "
                "differ and the\n  row's do not is that row answering from the narrow member's "
                "fits\n");

    for (const PartitionCarriageRef& armRef : PartitionCarriageRefs()) {
        if (armRef.bodiesRecorded) {
            continue;
        }

        std::printf("    %-40s A %zu, band %zu, B %zu, C %zu cell(s)\n",
                    armRef.axes.c_str(),
                    armRef.differ[0],
                    armRef.differ[1],
                    armRef.differ[2],
                    armRef.differ[3]);
    }

    std::printf("\n  the half lane's rows are not two fits read against each other: each names "
                "a float\n  body of the lane that stores half values with half I/O, and is held "
                "to the body it\n  names, cell for cell, at the half lane's own arguments. The "
                "two bodies an arm's rows\n  name, and the cells where a row answering from the "
                "other member's fits would part\n  from the body it names - the half format "
                "tells apart only what the half store still\n  carries, and the count at fp32 "
                "beside it is what the float lane's own rows hold:\n");

    for (const PartitionCarriageRef& armRef : PartitionCarriageRefs()) {
        if (!armRef.bodiesRecorded) {
            continue;
        }

        std::printf("    %-40s A %zu, band %zu, B %zu, C %zu cell(s) at fp32,\n",
                    armRef.axes.c_str(),
                    armRef.bodies[0],
                    armRef.bodies[1],
                    armRef.bodies[2],
                    armRef.bodies[3]);
        std::printf("    %-40s of which the half format tells A %zu, band %zu, B %zu, C %zu "
                    "apart\n",
                    "",
                    armRef.differ[0],
                    armRef.differ[1],
                    armRef.differ[2],
                    armRef.differ[3]);
    }

    std::printf("  %-58s %9s %9s  %-20s %s\n",
                "entry",
                "cells",
                "differ",
                "A/band/B/C",
                "verdict");
    std::printf("  %s\n", std::string(150, '-').c_str());

    std::size_t met = 0;
    std::vector<std::string> notMet;

    for (const PartitionCarriageRow& row : rows) {
        std::size_t cells = 0;
        std::size_t differ = 0;
        std::size_t missed = 0;
        char tokens[24];
        std::size_t at = 0;

        for (std::size_t r = 0; r < 4; ++r) {
            const char* tok = row.cells[r] == 0
                                  ? "-"
                                  : row.namesBody ? (row.differ[r] == 0 ? "same" : "DIFF")
                                                  : (row.differ[r] > 0 ? "yes" : "NO");
            at += static_cast<std::size_t>(std::snprintf(
                tokens + at, sizeof(tokens) - at, "%s%s", r == 0 ? "" : "/", tok));

            if (row.namesBody) {
                if (row.differ[r] > 0) {
                    ++missed;
                }
            } else if (row.refDiffer[r] > 0 && row.cells[r] > 0 && row.differ[r] == 0) {
                ++missed;
            }

            cells += row.cells[r];
            differ += row.differ[r];
        }

        if (missed == 0) {
            ++met;
            std::printf("  %-58s %9zu %9zu  %-20s %s\n",
                        row.entry.c_str(),
                        cells,
                        differ,
                        tokens,
                        row.namesBody ? "carries the body it names"
                                      : "answers with the grid's own values");
            continue;
        }

        notMet.push_back(row.entry);

        if (row.namesBody) {
            char which[64];
            std::size_t wat = 0;

            for (std::size_t r = 0; r < 4; ++r) {
                if (row.differ[r] > 0) {
                    wat += static_cast<std::size_t>(
                        std::snprintf(which + wat,
                                      sizeof(which) - wat,
                                      "%s%s: %zu",
                                      wat == 0 ? "" : ", ",
                                      kCarriageRegionTag[r],
                                      row.differ[r]));
                }
            }

            std::printf("  %-58s %9zu %9zu  %-20s NOT CARRIED - this row parts from the body "
                        "it\n      names, in %s cell(s), and the half lane's row for a policy "
                        "naming the grid is\n      that float body with half I/O\n",
                        row.entry.c_str(),
                        cells,
                        differ,
                        tokens,
                        which);
            continue;
        }

        char which[24];
        std::size_t wat = 0;

        for (std::size_t r = 0; r < 4; ++r) {
            if (row.refDiffer[r] > 0 && row.cells[r] > 0 && row.differ[r] == 0) {
                wat += static_cast<std::size_t>(std::snprintf(which + wat,
                                                              sizeof(which) - wat,
                                                              "%s%s",
                                                              wat == 0 ? "" : " and ",
                                                              kCarriageRegionTag[r]));
            }
        }

        std::printf("  %-58s %9zu %9zu  %-20s NOT CARRIED - region %s separates on the\n"
                    "      per-argument entry and nowhere on this row, so this row answered it "
                    "from the\n      narrow member's fits\n",
                    row.entry.c_str(),
                    cells,
                    differ,
                    tokens,
                    which);
    }

    std::printf("  %s\n", std::string(150, '-').c_str());
    std::printf("  PARTITION RESULT: %zu of %zu device row(s) over the uniform grid carry the "
                "body their\n                    policy names - the float and double lanes' rows "
                "a reading the narrow member\n                    does not answer with in every "
                "region where the two readings can differ,\n                    and the half "
                "lane's rows the float body each names, cell for cell\n",
                met,
                rows.size());

    if (!notMet.empty()) {
        std::printf("  NOT MET at this revision:");

        for (const std::string& id : notMet) {
            std::printf(" [%s]", id.c_str());
        }

        std::printf("\n  FAIL (exit status 1; a row that fails this answers a uniform policy "
                    "with fits it\n  does not name - the substitution the partition axis exists "
                    "to prevent)\n");
    }

    return notMet.size();
}
// The device-callable entries: the handle filled once, the rows its cells are
// measured into, and every entry through the consumer's kernels. Filling the
// handle is what makes the degree tables resident for the kernels, and the
// handle is the one every device entry below reads.
void SweepDeviceLane(const Reference& ref,
                     const Grid& grid,
                     const SortedArgs& sorted,
                     const DigitGrid& digits) {
    boys::BoysDeviceTables tables{};
    CheckLaunch(boys::BoysCuda::DeviceTables(&tables), "DeviceTables");
    const DeviceSlots slots = DeviceClaimSet();
    SweepDevice(ref, grid, sorted, tables, slots);
    SweepDigit64(digits, tables, slots);
    PartitionCarriageF32(ref, grid, tables);
    PartitionCarriageF64(ref, grid, tables);
#if BoysFp16
    PartitionCarriageF16(ref, grid, tables);
#endif
}

// One cell, every entry: what each device entry returns beside the reference, so
// a reported failure can be reproduced rather than believed. An argument that is
// not on the committed grid is printed as such, not measured against the nearest
// node.
void RunProbe(const Reference& ref,
              const boys::BoysDeviceTables& tables,
              int n,
              double x) {
    const int nmax = boys::kMaxBoysOrder;
    const int count = 1;
    const double xf = static_cast<double>(static_cast<float>(x));
    const double x16 = static_cast<double>(boys::F16(static_cast<float>(x)));
    const bool finiteX16 = std::isfinite(x16);
    const std::size_t notFound = ref.count * static_cast<std::size_t>(nmax + 1);

    // Each lane is measured at its own rounding of the argument, and the grid
    // carries a column per rounding: a column with no cell for that argument
    // leaves the lane's reference blank rather than measuring the nearest node.
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

    DevBuf<int> dN(1);
    DevBuf<double> dX(1);
    dN.Upload(hostN);
    dX.Upload(hostX);

    const std::size_t planes = static_cast<std::size_t>(nmax + 1);
    DevBuf<double> d64(planes);
    DevBuf<float> d32(planes);
#if BoysFp16
    std::vector<boys::F16> hostH(1, boys::F16(static_cast<float>(x)));
    DevBuf<boys::F16> dH(1);
    DevBuf<boys::F16> d16(planes);
    dH.Upload(hostH);
#endif

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
#if BoysFp16
    std::vector<boys::F16> out16(planes);
#endif
    const double ref64 = have64 ? ref.v[atX] : 0.0;
    const double ref32 = have32 ? ref.vf[atF] : 0.0;
    const double ref16 = have16 ? ref.v16[atH] : 0.0;


    CheckLaunch(boys::BoysCuda::SingleF64(dN.get(), dX.get(), d64.get(), count, nullptr),
                "SingleF64");
    d64.Download(out64);
    row("cuda single f64", out64[0], ref64, kBoundSingleC, have64);

    CheckLaunch(
        boys::BoysCuda::AllOrdersF64(dN.get(), dX.get(), d64.get(), count, nullptr),
        "AllOrdersF64");
    d64.Download(out64);
    row("cuda all-orders f64",
        out64[static_cast<std::size_t>(n)],
        ref64,
        kBoundDoubleBatch,
        have64);

    CheckLaunch(boys::BoysCuda::AllNF64(n, dX.get(), d64.get(), count, nullptr), "AllNF64");
    d64.Download(out64);
    row("cuda all-n f64", out64[static_cast<std::size_t>(n)], ref64, kBoundDoubleBatch, have64);

    CheckLaunch(boys::BoysCuda::SingleF32(dN.get(), dX.get(), d32.get(), count, nullptr),
                "SingleF32");
    d32.Download(out32);
    row("cuda single f32", static_cast<double>(out32[0]), ref32, kBoundFloat, have32);
    std::printf("  %-22s (evaluated at xf=%.17g)\n", "", xf);

    CheckLaunch(boys::BoysCuda::SingleF32<boys::RegionBExp::kFast>(
                    dN.get(), dX.get(), d32.get(), count, nullptr),
                "SingleF32 fast");
    d32.Download(out32);
    row("cuda single f32 (fast exp)",
        static_cast<double>(out32[0]),
        ref32,
        kBoundFloat + kFastExpContribution,
        have32);

    CheckLaunch(
        boys::BoysCuda::AllOrdersF32(dN.get(), dX.get(), d32.get(), count, nullptr),
        "AllOrdersF32");
    d32.Download(out32);
    row("cuda all-orders f32",
        static_cast<double>(out32[static_cast<std::size_t>(n)]),
        ref32,
        kBoundFloat,
        have32);

    CheckLaunch(boys::BoysCuda::AllNF32(n, dX.get(), d32.get(), count, nullptr), "AllNF32");
    d32.Download(out32);
    row("cuda all-n f32",
        static_cast<double>(out32[static_cast<std::size_t>(n)]),
        ref32,
        kBoundFloat,
        have32);

#if BoysFp16
    CheckLaunch(boys::BoysCuda::SingleF16(dN.get(), dH.get(), d16.get(), count, nullptr),
                "SingleF16");
    d16.Download(out16);
    row("cuda single f16",
        static_cast<double>(out16[0]),
        ref16,
        HalfBoundAt(static_cast<double>(out16[0])),
        have16);
    std::printf("  %-22s (evaluated at x16=%.17g)\n", "", x16);

    CheckLaunch(
        boys::BoysCuda::AllOrdersF16(dN.get(), dH.get(), d16.get(), count, nullptr),
        "AllOrdersF16");
    d16.Download(out16);
    row("cuda all-orders f16",
        static_cast<double>(out16[static_cast<std::size_t>(n)]),
        ref16,
        HalfBoundAt(static_cast<double>(out16[static_cast<std::size_t>(n)])),
        have16);

    CheckLaunch(boys::BoysCuda::AllNF16(n, dH.get(), d16.get(), count, nullptr), "AllNF16");
    d16.Download(out16);
    row("cuda all-n f16",
        static_cast<double>(out16[static_cast<std::size_t>(n)]),
        ref16,
        HalfBoundAt(static_cast<double>(out16[static_cast<std::size_t>(n)])),
        have16);
#else
    // The three fp16 rows of this table. Their entries are declared behind the
    // BoysFp16 seam, which this build has closed, so there is no entry here to
    // call; they are named in the table with the build fact that leaves them
    // unmeasured, because a row missing from a table reads as a row measured.
    (void)ref16;
    const char* const seam = "not carried: this build's BoysFp16 seam is closed";
    std::printf("  %-22s %s\n", "cuda single f16", seam);
    std::printf("  %-22s %s\n", "cuda all-orders f16", seam);
    std::printf("  %-22s %s\n", "cuda all-n f16", seam);
#endif // BoysFp16

    // The device-callable entries at the same cell, through the same consumer
    // kernels the rows above are measured through. Each thread forms its own
    // argument as rho * d2; rho = 1 here, which is exact.
    {
        const std::vector<double> hostRho(1, 1.0);
        const std::vector<double> hostD2(1, x);
        std::vector<int> hostStatus(1, -1);
        DevBuf<double> dRho(1);
        DevBuf<double> dD2(1);
        DevBuf<int> dStatus(1);
        dRho.Upload(hostRho);
        dD2.Upload(hostD2);
        dStatus.Upload(hostStatus);

        CheckDemo(BoysDeviceDemoSingle64(
                      &tables, dN.get(), dRho.get(), dD2.get(), d64.get(), count, dStatus.get()),
                  "BoysDeviceDemoSingle64");
        d64.Download(out64);
        row("device single f64", out64[0], ref64, kBoundSingleC, have64);

        CheckDemo(BoysDeviceDemoLadder64(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         d64.get(),
                                         count,
                                         planes,
                                         dStatus.get()),
                  "BoysDeviceDemoLadder64");
        d64.Download(out64);
        row("device all-orders f64",
            out64[static_cast<std::size_t>(n)],
            ref64,
            kBoundDoubleBatch,
            have64);

        CheckDemo(BoysDeviceDemoEach64(
                      &tables, dN.get(), dRho.get(), dD2.get(), d64.get(), count, dStatus.get()),
                  "BoysDeviceDemoEach64");
        d64.Download(out64);
        row("device each-order f64",
            out64[static_cast<std::size_t>(n)],
            ref64,
            kBoundDoubleBatch,
            have64);

        CheckDemo(
            BoysDeviceDemoAllN64(&tables, dRho.get(), dD2.get(), d64.get(), count, dStatus.get()),
            "BoysDeviceDemoAllN64");
        d64.Download(out64);
        row("device all-n f64", out64[static_cast<std::size_t>(n)], ref64, kBoundDoubleBatch,
            have64);

        CheckDemo(
            BoysDeviceDemoSingle32(&tables, dN.get(), dRho.get(), dD2.get(), d32.get(), count,
                                   dStatus.get()),
            "BoysDeviceDemoSingle32");
        d32.Download(out32);
        row("device single f32", static_cast<double>(out32[0]), ref32, kBoundFloat, have32);

        CheckDemo(
            BoysDeviceDemoSingle32Fast(&tables, dN.get(), dRho.get(), dD2.get(), d32.get(), count,
                                       dStatus.get()),
            "BoysDeviceDemoSingle32Fast");
        d32.Download(out32);
        row("device single f32 (fast exp)",
            static_cast<double>(out32[0]),
            ref32,
            kBoundFloat + kFastExpContribution,
            have32);

        CheckDemo(BoysDeviceDemoLadder32(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         d32.get(),
                                         count,
                                         planes,
                                         dStatus.get()),
                  "BoysDeviceDemoLadder32");
        d32.Download(out32);
        row("device all-orders f32",
            static_cast<double>(out32[static_cast<std::size_t>(n)]),
            ref32,
            kBoundFloat,
            have32);

        CheckDemo(BoysDeviceDemoEach32(
                      &tables, dN.get(), dRho.get(), dD2.get(), d32.get(), count, dStatus.get()),
                  "BoysDeviceDemoEach32");
        d32.Download(out32);
        row("device each-order f32",
            static_cast<double>(out32[static_cast<std::size_t>(n)]),
            ref32,
            kBoundFloat,
            have32);

        CheckDemo(
            BoysDeviceDemoAllN32(&tables, dRho.get(), dD2.get(), d32.get(), count, dStatus.get()),
            "BoysDeviceDemoAllN32");
        d32.Download(out32);
        row("device all-n f32", static_cast<double>(out32[static_cast<std::size_t>(n)]), ref32,
            kBoundFloat, have32);

#if BoysFp16
        CheckDemo(BoysDeviceDemoSingle16(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         static_cast<void*>(d16.get()),
                                         count,
                                         dStatus.get()),
                  "BoysDeviceDemoSingle16");
        d16.Download(out16);
        row("device single f16",
            static_cast<double>(out16[0]),
            ref16,
            HalfBoundAt(static_cast<double>(out16[0])),
            have16);

        CheckDemo(BoysDeviceDemoLadder16(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         static_cast<void*>(d16.get()),
                                         count,
                                         planes,
                                         dStatus.get()),
                  "BoysDeviceDemoLadder16");
        d16.Download(out16);
        row("device all-orders f16",
            static_cast<double>(out16[static_cast<std::size_t>(n)]),
            ref16,
            HalfBoundAt(static_cast<double>(out16[static_cast<std::size_t>(n)])),
            have16);

        CheckDemo(BoysDeviceDemoEach16(&tables,
                                       dN.get(),
                                       dRho.get(),
                                       dD2.get(),
                                       static_cast<void*>(d16.get()),
                                       count,
                                       dStatus.get()),
                  "BoysDeviceDemoEach16");
        d16.Download(out16);
        row("device each-order f16",
            static_cast<double>(out16[static_cast<std::size_t>(n)]),
            ref16,
            HalfBoundAt(static_cast<double>(out16[static_cast<std::size_t>(n)])),
            have16);

        CheckDemo(BoysDeviceDemoAllN16(&tables,
                                       dRho.get(),
                                       dD2.get(),
                                       static_cast<void*>(d16.get()),
                                       count,
                                       dStatus.get()),
                  "BoysDeviceDemoAllN16");
        d16.Download(out16);
        row("device all-n f16",
            static_cast<double>(out16[static_cast<std::size_t>(n)]),
            ref16,
            HalfBoundAt(static_cast<double>(out16[static_cast<std::size_t>(n)])),
            have16);
#else
        // The four device-callable fp16 rows, on the same cell as the f32 rows
        // above them. The demo entries that reach the fp16 arithmetic are declared
        // behind the BoysFp16 seam, which this build has closed, so the table says
        // four entries are missing from it and why.
        const char* const seamNote = "not carried: this build's BoysFp16 seam is closed";
        std::printf("  %-22s %s\n", "device single f16", seamNote);
        std::printf("  %-22s %s\n", "device all-orders f16", seamNote);
        std::printf("  %-22s %s\n", "device each-order f16", seamNote);
        std::printf("  %-22s %s\n", "device all-n f16", seamNote);
#endif // BoysFp16
    }
}

// ---------------------------------------------------------------------------
// The whole launched surface: one arm per entry the library's report carries,
// and the launch discipline every arm runs under.
// ---------------------------------------------------------------------------
//
// The sections above measure the arithmetic this gate was written for. What this
// section adds is the statement that the surface is reached at all. The library
// reports one row per entry (BoysDeviceOptions()) and every one of those rows has
// a kernel behind it, so an entry no arm launches is an entry whose documented
// bound is certified by nothing - and a run that never calls it prints exactly
// what a run in which every one of its cells passed prints. That is the failure
// this section exists for, and it is why an arm is owed for every row rather than
// for the rows somebody remembered to write down.
//
// The table below is keyed by the library's own enumerator and states only the
// one thing the report cannot: the function pointer. The shape and the lane are
// read off the row at run time, so a row whose shape or precision moves moves its
// arm with it and no second list can disagree with the first.
//
// A row an earlier section already claims is not measured twice: the entry is
// reached, the claim exists, and a second claim of the same name would double the
// table without adding a statement. The arm is counted either way, which is what
// makes the count a statement about the surface and not about this table.

// The launched signatures, by argument list and lane. The single and all-orders
// shapes share the range signature and differ only in the layout the kernel
// fills, so the row's shape selects the driver and never the pointer's type.
using Range64Fn = boys::BoysStatus (*)(const int*, const double*, double*, std::size_t, void*, boys::DivisionForm);
using Range32Fn = boys::BoysStatus (*)(const int*, const double*, float*, std::size_t, void*, boys::DivisionForm);
using Range16Fn = boys::BoysStatus (*)(const int*, const boys::F16*, boys::F16*, std::size_t, void*, boys::DivisionForm);
using AllN64Fn = boys::BoysStatus (*)(int, const double*, double*, std::size_t, void*, boys::DivisionForm);
using AllN32Fn = boys::BoysStatus (*)(int, const double*, float*, std::size_t, void*, boys::DivisionForm);
using AllN16Fn = boys::BoysStatus (*)(int, const boys::F16*, boys::F16*, std::size_t, void*, boys::DivisionForm);
using Each64Fn = boys::BoysStatus (*)(const int*, const double*, const int*, double*, std::size_t, void*, boys::DivisionForm);
using Each32Fn = boys::BoysStatus (*)(const int*, const double*, const int*, float*, std::size_t, void*, boys::DivisionForm);
using Each16Fn = boys::BoysStatus (*)(const int*, const boys::F16*, const int*, boys::F16*, std::size_t, void*, boys::DivisionForm);

/// One launched entry, and the kernel that serves it.
///
/// Exactly one of the nine pointers is set, and which one is decided by the
/// row's own shape and precision. The dispatch below reads the row and demands
/// the pointer that row's shape owns, so a row whose shape moved since the table
/// was written stops the gate rather than arming the wrong driver.
struct LaunchedArm {
    boys::DeviceEntry entry; ///< the library's own enumerator for the row
    Range64Fn range64 = nullptr;
    Range32Fn range32 = nullptr;
    Range16Fn range16 = nullptr;
    AllN64Fn allN64 = nullptr;
    AllN32Fn allN32 = nullptr;
    AllN16Fn allN16 = nullptr;
    Each64Fn each64 = nullptr;
    Each32Fn each32 = nullptr;
    Each16Fn each16 = nullptr;
    /// The row this arm is, where an entry carries two: the certified region-B
    /// exponential is an axis of its own and its two members are two rows of one
    /// entry, each with its own bound.
    boys::RegionBExp exp = boys::RegionBExp::kAccurate;
};

/// Every launched entry the report carries, one arm apiece.
///
/// Read off the library rather than written by hand: the enumerator is the
/// report's, and the row an entry names is resolved through DeviceRow at run
/// time, which stops the gate if the report no longer carries it. A launched row
/// the report carries and this table does not name is named by the survey below
/// rather than passed over.
///
/// A row whose entry the library has not yet given a function this table can
/// name is written where it belongs and commented out with that reason, rather
/// than left out: the report lists it as built, nothing here can launch it, and
/// the survey counts it as a row no arm reaches. Each such line is what an entry
/// half-landed looks like from this side, and the fix is to uncomment it when the
/// function exists.
const LaunchedArm kLaunchedArms[] = {
    // fp64 ------------------------
    {.entry = boys::DeviceEntry::kAllNF64, .allN64 = &boys::BoysCuda::AllNF64},
    {.entry = boys::DeviceEntry::kAllNF64Fast, .allN64 = &boys::BoysCuda::AllNF64Fast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64, .range64 = &boys::BoysCuda::AllOrdersF64},
    {.entry = boys::DeviceEntry::kAllOrdersF64Fast, .range64 = &boys::BoysCuda::AllOrdersF64Fast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64Mono, .range64 = &boys::BoysCuda::AllOrdersF64Mono},
    {.entry = boys::DeviceEntry::kAllOrdersF64MonoFast, .range64 = &boys::BoysCuda::AllOrdersF64MonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64Narrow, .range64 = &boys::BoysCuda::AllOrdersF64Narrow},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowFast, .range64 = &boys::BoysCuda::AllOrdersF64NarrowFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowMono, .range64 = &boys::BoysCuda::AllOrdersF64NarrowMono},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowMonoFast, .range64 = &boys::BoysCuda::AllOrdersF64NarrowMonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowOrders, .range64 = &boys::BoysCuda::AllOrdersF64NarrowOrders},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowOrdersFast, .range64 = &boys::BoysCuda::AllOrdersF64NarrowOrdersFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowOrdersMono, .range64 = &boys::BoysCuda::AllOrdersF64NarrowOrdersMono},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowOrdersMonoFast, .range64 = &boys::BoysCuda::AllOrdersF64NarrowOrdersMonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowOrdersRat, .range64 = &boys::BoysCuda::AllOrdersF64NarrowOrdersRat},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowOrdersRatFast, .range64 = &boys::BoysCuda::AllOrdersF64NarrowOrdersRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner, .range64 = &boys::BoysCuda::AllOrdersF64NarrowOrdersRat},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowOrdersRatHornerFast, .range64 = &boys::BoysCuda::AllOrdersF64NarrowOrdersRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowRat, .range64 = &boys::BoysCuda::AllOrdersF64NarrowRat},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowRatFast, .range64 = &boys::BoysCuda::AllOrdersF64NarrowRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowRatHorner, .range64 = &boys::BoysCuda::AllOrdersF64NarrowRat},
    {.entry = boys::DeviceEntry::kAllOrdersF64NarrowRatHornerFast, .range64 = &boys::BoysCuda::AllOrdersF64NarrowRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64Orders, .range64 = &boys::BoysCuda::AllOrdersF64Orders},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersFast, .range64 = &boys::BoysCuda::AllOrdersF64OrdersFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersMono, .range64 = &boys::BoysCuda::AllOrdersF64OrdersMono},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersMonoFast, .range64 = &boys::BoysCuda::AllOrdersF64OrdersMonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersRat, .range64 = &boys::BoysCuda::AllOrdersF64OrdersRat},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersRatFast, .range64 = &boys::BoysCuda::AllOrdersF64OrdersRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersRatHorner, .range64 = &boys::BoysCuda::AllOrdersF64OrdersRat},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersRatHornerFast, .range64 = &boys::BoysCuda::AllOrdersF64OrdersRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersUniform, .range64 = &boys::BoysCuda::AllOrdersF64OrdersUniform},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersUniformHorner, .range64 = &boys::BoysCuda::AllOrdersF64OrdersUniformHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersUniformRat, .range64 = &boys::BoysCuda::AllOrdersF64OrdersUniformRat},
    {.entry = boys::DeviceEntry::kAllOrdersF64OrdersUniformRatHorner, .range64 = &boys::BoysCuda::AllOrdersF64OrdersUniformRatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF64Rat, .range64 = &boys::BoysCuda::AllOrdersF64Rat},
    {.entry = boys::DeviceEntry::kAllOrdersF64RatFast, .range64 = &boys::BoysCuda::AllOrdersF64RatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64RatHorner, .range64 = &boys::BoysCuda::AllOrdersF64Rat},
    {.entry = boys::DeviceEntry::kAllOrdersF64RatHornerFast, .range64 = &boys::BoysCuda::AllOrdersF64RatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF64Uniform, .range64 = &boys::BoysCuda::AllOrdersF64Uniform},
    {.entry = boys::DeviceEntry::kAllOrdersF64UniformHorner, .range64 = &boys::BoysCuda::AllOrdersF64UniformHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF64UniformRat, .range64 = &boys::BoysCuda::AllOrdersF64UniformRat},
    {.entry = boys::DeviceEntry::kAllOrdersF64UniformRatHorner, .range64 = &boys::BoysCuda::AllOrdersF64UniformRatHorner},
    {.entry = boys::DeviceEntry::kEachOrderF64, .each64 = &boys::BoysCuda::EachOrderF64},
    {.entry = boys::DeviceEntry::kEachOrderF64Fast, .each64 = &boys::BoysCuda::EachOrderF64<boys::RegionBExp::kFast>, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kSingleF64, .range64 = &boys::BoysCuda::SingleF64},
    {.entry = boys::DeviceEntry::kSingleF64Fast, .range64 = &boys::BoysCuda::SingleF64Fast, .exp = boys::RegionBExp::kFast},
    // fp32 ------------------------
    {.entry = boys::DeviceEntry::kAllNF32, .allN32 = &boys::BoysCuda::AllNF32},
    {.entry = boys::DeviceEntry::kAllNF32Fast, .allN32 = &boys::BoysCuda::AllNF32Fast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32, .range32 = &boys::BoysCuda::AllOrdersF32},
    {.entry = boys::DeviceEntry::kAllOrdersF32Fast, .range32 = &boys::BoysCuda::AllOrdersF32Fast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32Mono, .range32 = &boys::BoysCuda::AllOrdersF32Mono},
    {.entry = boys::DeviceEntry::kAllOrdersF32MonoFast, .range32 = &boys::BoysCuda::AllOrdersF32MonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32Narrow, .range32 = &boys::BoysCuda::AllOrdersF32Narrow},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowFast, .range32 = &boys::BoysCuda::AllOrdersF32NarrowFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowMono, .range32 = &boys::BoysCuda::AllOrdersF32NarrowMono},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowMonoFast, .range32 = &boys::BoysCuda::AllOrdersF32NarrowMonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowOrders, .range32 = &boys::BoysCuda::AllOrdersF32NarrowOrders},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowOrdersFast, .range32 = &boys::BoysCuda::AllOrdersF32NarrowOrdersFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowOrdersMono, .range32 = &boys::BoysCuda::AllOrdersF32NarrowOrdersMono},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowOrdersMonoFast, .range32 = &boys::BoysCuda::AllOrdersF32NarrowOrdersMonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowOrdersRat, .range32 = &boys::BoysCuda::AllOrdersF32NarrowOrdersRat},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowOrdersRatFast, .range32 = &boys::BoysCuda::AllOrdersF32NarrowOrdersRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner, .range32 = &boys::BoysCuda::AllOrdersF32NarrowOrdersRatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowOrdersRatHornerFast, .range32 = &boys::BoysCuda::AllOrdersF32NarrowOrdersRatHornerFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowRat, .range32 = &boys::BoysCuda::AllOrdersF32NarrowRat},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowRatFast, .range32 = &boys::BoysCuda::AllOrdersF32NarrowRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowRatHorner, .range32 = &boys::BoysCuda::AllOrdersF32NarrowRatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF32NarrowRatHornerFast, .range32 = &boys::BoysCuda::AllOrdersF32NarrowRatHornerFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32Orders, .range32 = &boys::BoysCuda::AllOrdersF32Orders},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersFast, .range32 = &boys::BoysCuda::AllOrdersF32OrdersFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersMono, .range32 = &boys::BoysCuda::AllOrdersF32OrdersMono},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersMonoFast, .range32 = &boys::BoysCuda::AllOrdersF32OrdersMonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersRat, .range32 = &boys::BoysCuda::AllOrdersF32OrdersRat},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersRatFast, .range32 = &boys::BoysCuda::AllOrdersF32OrdersRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersRatHorner, .range32 = &boys::BoysCuda::AllOrdersF32OrdersRatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersRatHornerFast, .range32 = &boys::BoysCuda::AllOrdersF32OrdersRatHornerFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersUniform, .range32 = &boys::BoysCuda::AllOrdersF32OrdersUniform},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersUniformHorner, .range32 = &boys::BoysCuda::AllOrdersF32OrdersUniformHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersUniformRat, .range32 = &boys::BoysCuda::AllOrdersF32OrdersUniformRat},
    {.entry = boys::DeviceEntry::kAllOrdersF32OrdersUniformRatHorner, .range32 = &boys::BoysCuda::AllOrdersF32OrdersUniformRatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF32Rat, .range32 = &boys::BoysCuda::AllOrdersF32Rat},
    {.entry = boys::DeviceEntry::kAllOrdersF32RatFast, .range32 = &boys::BoysCuda::AllOrdersF32RatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32RatHorner, .range32 = &boys::BoysCuda::AllOrdersF32RatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF32RatHornerFast, .range32 = &boys::BoysCuda::AllOrdersF32RatHornerFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF32Uniform, .range32 = &boys::BoysCuda::AllOrdersF32Uniform},
    {.entry = boys::DeviceEntry::kAllOrdersF32UniformHorner, .range32 = &boys::BoysCuda::AllOrdersF32UniformHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF32UniformRat, .range32 = &boys::BoysCuda::AllOrdersF32UniformRat},
    {.entry = boys::DeviceEntry::kAllOrdersF32UniformRatHorner, .range32 = &boys::BoysCuda::AllOrdersF32UniformRatHorner},
    {.entry = boys::DeviceEntry::kEachOrderF32, .each32 = &boys::BoysCuda::EachOrderF32},
    {.entry = boys::DeviceEntry::kEachOrderF32Fast, .each32 = &boys::BoysCuda::EachOrderF32<boys::RegionBExp::kFast>, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kSingleF32, .range32 = &boys::BoysCuda::SingleF32},
    {.entry = boys::DeviceEntry::kSingleF32Fast, .range32 = &boys::BoysCuda::SingleF32<boys::RegionBExp::kFast>, .exp = boys::RegionBExp::kFast},
    // fp16 ------------------------
#if BOYS_CUDA_GATE_FP16
    // all-n-bfloat16 at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllNBf16, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllNBf16},
    // all-n-bfloat16-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllNBf16Fast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllNBf16Fast},
    {.entry = boys::DeviceEntry::kAllNF16, .allN16 = &boys::BoysCuda::AllNF16},
    {.entry = boys::DeviceEntry::kAllNF16Fast, .allN16 = &boys::BoysCuda::AllNF16Fast, .exp = boys::RegionBExp::kFast},
    // all-orders-bfloat16 at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16},
    // all-orders-bfloat16-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16Fast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16Fast},
    // all-orders-bfloat16-mono at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16Mono, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16Mono},
    // all-orders-bfloat16-mono-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16MonoFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16MonoFast},
    // all-orders-bfloat16-narrow at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16Narrow, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16Narrow},
    // all-orders-bfloat16-narrow-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowFast},
    // all-orders-bfloat16-narrow-mono at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowMono, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowMono},
    // all-orders-bfloat16-narrow-mono-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowMonoFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowMonoFast},
    // all-orders-bfloat16-narrow-orders at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowOrders, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowOrders},
    // all-orders-bfloat16-narrow-orders-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowOrdersFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowOrdersFast},
    // all-orders-bfloat16-narrow-orders-mono at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowOrdersMono, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowOrdersMono},
    // all-orders-bfloat16-narrow-orders-mono-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowOrdersMonoFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowOrdersMonoFast},
    // all-orders-bfloat16-narrow-orders-rat at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowOrdersRat, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowOrdersRat},
    // all-orders-bfloat16-narrow-orders-rat-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowOrdersRatFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowOrdersRatFast},
    // all-orders-bfloat16-narrow-orders-rat-horner at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowOrdersRatHorner, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowOrdersRatHorner},
    // all-orders-bfloat16-narrow-orders-rat-horner-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowOrdersRatHornerFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowOrdersRatHornerFast},
    // all-orders-bfloat16-narrow-rat at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowRat, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowRat},
    // all-orders-bfloat16-narrow-rat-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowRatFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowRatFast},
    // all-orders-bfloat16-narrow-rat-horner at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowRatHorner, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowRatHorner},
    // all-orders-bfloat16-narrow-rat-horner-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16NarrowRatHornerFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16NarrowRatHornerFast},
    // all-orders-bfloat16-orders at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16Orders, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16Orders},
    // all-orders-bfloat16-orders-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersFast},
    // all-orders-bfloat16-orders-mono at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersMono, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersMono},
    // all-orders-bfloat16-orders-mono-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersMonoFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersMonoFast},
    // all-orders-bfloat16-orders-rat at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersRat, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersRat},
    // all-orders-bfloat16-orders-rat-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersRatFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersRatFast},
    // all-orders-bfloat16-orders-rat-horner at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersRatHorner, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersRatHorner},
    // all-orders-bfloat16-orders-rat-horner-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersRatHornerFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersRatHornerFast},
    // all-orders-bfloat16-orders-uniform at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersUniform, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersUniform},
    // all-orders-bfloat16-orders-uniform-horner at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersUniformHorner, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersUniformHorner},
    // all-orders-bfloat16-orders-uniform-rat at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersUniformRat, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersUniformRat},
    // all-orders-bfloat16-orders-uniform-rat-horner at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16OrdersUniformRatHorner, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16OrdersUniformRatHorner},
    // all-orders-bfloat16-rat at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16Rat, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16Rat},
    // all-orders-bfloat16-rat-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16RatFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16RatFast},
    // all-orders-bfloat16-rat-horner at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16RatHorner, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16RatHorner},
    // all-orders-bfloat16-rat-horner-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16RatHornerFast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16RatHornerFast},
    // all-orders-bfloat16-uniform at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16Uniform, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16Uniform},
    // all-orders-bfloat16-uniform-horner at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16UniformHorner, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16UniformHorner},
    // all-orders-bfloat16-uniform-rat at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16UniformRat, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16UniformRat},
    // all-orders-bfloat16-uniform-rat-horner at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for AllOrdersBf16UniformRatHorner, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kAllOrdersBf16UniformRatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF16, .range16 = &boys::BoysCuda::AllOrdersF16},
    {.entry = boys::DeviceEntry::kAllOrdersF16Fast, .range16 = &boys::BoysCuda::AllOrdersF16Fast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16Mono, .range16 = &boys::BoysCuda::AllOrdersF16Mono},
    {.entry = boys::DeviceEntry::kAllOrdersF16MonoFast, .range16 = &boys::BoysCuda::AllOrdersF16MonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16Narrow, .range16 = &boys::BoysCuda::AllOrdersF16Narrow},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowFast, .range16 = &boys::BoysCuda::AllOrdersF16NarrowFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowMono, .range16 = &boys::BoysCuda::AllOrdersF16NarrowMono},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowMonoFast, .range16 = &boys::BoysCuda::AllOrdersF16NarrowMonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowOrders, .range16 = &boys::BoysCuda::AllOrdersF16NarrowOrders},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowOrdersFast, .range16 = &boys::BoysCuda::AllOrdersF16NarrowOrdersFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowOrdersMono, .range16 = &boys::BoysCuda::AllOrdersF16NarrowOrdersMono},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowOrdersMonoFast, .range16 = &boys::BoysCuda::AllOrdersF16NarrowOrdersMonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowOrdersRat, .range16 = &boys::BoysCuda::AllOrdersF16NarrowOrdersRat},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowOrdersRatFast, .range16 = &boys::BoysCuda::AllOrdersF16NarrowOrdersRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowOrdersRatHorner, .range16 = &boys::BoysCuda::AllOrdersF16NarrowOrdersRatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowOrdersRatHornerFast, .range16 = &boys::BoysCuda::AllOrdersF16NarrowOrdersRatHornerFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowRat, .range16 = &boys::BoysCuda::AllOrdersF16NarrowRat},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowRatFast, .range16 = &boys::BoysCuda::AllOrdersF16NarrowRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowRatHorner, .range16 = &boys::BoysCuda::AllOrdersF16NarrowRatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF16NarrowRatHornerFast, .range16 = &boys::BoysCuda::AllOrdersF16NarrowRatHornerFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16Orders, .range16 = &boys::BoysCuda::AllOrdersF16Orders},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersFast, .range16 = &boys::BoysCuda::AllOrdersF16OrdersFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersMono, .range16 = &boys::BoysCuda::AllOrdersF16OrdersMono},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersMonoFast, .range16 = &boys::BoysCuda::AllOrdersF16OrdersMonoFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersRat, .range16 = &boys::BoysCuda::AllOrdersF16OrdersRat},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersRatFast, .range16 = &boys::BoysCuda::AllOrdersF16OrdersRatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersRatHorner, .range16 = &boys::BoysCuda::AllOrdersF16OrdersRatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersRatHornerFast, .range16 = &boys::BoysCuda::AllOrdersF16OrdersRatHornerFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersUniform, .range16 = &boys::BoysCuda::AllOrdersF16OrdersUniform},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersUniformHorner, .range16 = &boys::BoysCuda::AllOrdersF16OrdersUniformHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersUniformRat, .range16 = &boys::BoysCuda::AllOrdersF16OrdersUniformRat},
    {.entry = boys::DeviceEntry::kAllOrdersF16OrdersUniformRatHorner, .range16 = &boys::BoysCuda::AllOrdersF16OrdersUniformRatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF16Rat, .range16 = &boys::BoysCuda::AllOrdersF16Rat},
    {.entry = boys::DeviceEntry::kAllOrdersF16RatFast, .range16 = &boys::BoysCuda::AllOrdersF16RatFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16RatHorner, .range16 = &boys::BoysCuda::AllOrdersF16RatHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF16RatHornerFast, .range16 = &boys::BoysCuda::AllOrdersF16RatHornerFast, .exp = boys::RegionBExp::kFast},
    {.entry = boys::DeviceEntry::kAllOrdersF16Uniform, .range16 = &boys::BoysCuda::AllOrdersF16Uniform},
    {.entry = boys::DeviceEntry::kAllOrdersF16UniformHorner, .range16 = &boys::BoysCuda::AllOrdersF16UniformHorner},
    {.entry = boys::DeviceEntry::kAllOrdersF16UniformRat, .range16 = &boys::BoysCuda::AllOrdersF16UniformRat},
    {.entry = boys::DeviceEntry::kAllOrdersF16UniformRatHorner, .range16 = &boys::BoysCuda::AllOrdersF16UniformRatHorner},
    // each-order-bfloat16 at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for EachOrderBf16, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kEachOrderBf16},
    // each-order-bfloat16-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for EachOrderBf16, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kEachOrderBf16Fast},
    {.entry = boys::DeviceEntry::kEachOrderF16, .each16 = &boys::BoysCuda::EachOrderF16},
    {.entry = boys::DeviceEntry::kEachOrderF16Fast, .each16 = &boys::BoysCuda::EachOrderF16<boys::RegionBExp::kFast>, .exp = boys::RegionBExp::kFast},
    // single-bfloat16 at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for SingleBf16, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kSingleBf16},
    // single-bfloat16-fast at this revision: the report carries the row and boys_cuda.hpp
    // declares no function for SingleBf16Fast, so there is nothing to arm.
    // {.entry = boys::DeviceEntry::kSingleBf16Fast},
    {.entry = boys::DeviceEntry::kSingleF16, .range16 = &boys::BoysCuda::SingleF16},
    {.entry = boys::DeviceEntry::kSingleF16Fast, .range16 = &boys::BoysCuda::SingleF16Fast, .exp = boys::RegionBExp::kFast},
#endif // BOYS_CUDA_GATE_FP16
};

// ---------------------------------------------------------------------------
// The launch discipline, and the instrument that proves it bites.
// ---------------------------------------------------------------------------

/// Clear whatever the last call left pending, so the read after a launch answers
/// for that launch and not for the one before it.
void BeginLaunchedCall() {
    (void)cudaGetLastError();
}

/// The read after a launch, and the one statement a reader of the numbers cannot
/// make for themselves: that a kernel ran.
///
/// A launcher that returned success and a launcher whose kernel the driver never
/// issued are the same value to anyone looking only at the output buffer, and the
/// second reads as a lane that agreed with the reference whenever the buffer
/// still holds what was put there.
///
/// Two channels carry that statement out of a launch and this reads both. The
/// library's launcher reads the runtime's error itself and maps it into the
/// status CheckLaunch stops the run on, so a refused launch is already a status
/// and not a silence - which is why the read below is not the statement and
/// cannot be presented as one. What it adds is what the launcher's own read
/// cannot see: an error raised after it, and, with the device synchronise beside
/// it, a kernel that started and then failed. Both leave the output buffer
/// holding whatever was put there, and both are the shape of failure this whole
/// discipline exists for.
void RequireLaunched(const char* entry, cudaError_t after) {
    if (after == cudaSuccess)
    {
        return;
    }

    std::fprintf(stderr,
                 "cuda gate: the launch of %s did not happen: %s\n",
                 entry,
                 cudaGetErrorString(after));
    std::exit(2);
}

/// How many launches this gate has read the runtime's error back after. Printed,
/// because a run that issued no launch is exactly the failure this section exists
/// for and a count of zero is what it looks like.
std::size_t& LaunchesChecked() {
    static std::size_t checked = 0;
    return checked;
}

/// The launch-error read, shown able to fail before anything is taken from it.
///
/// A check that cannot fail is worse than no check, and no input this gate
/// accepts can show this one failing: a launch that worked is a success. The
/// runtime is therefore asked for a failure deliberately - a launch naming no
/// function - and the read is required to report it. A read that answered success
/// there would certify every launch in the file, which is precisely what a reader
/// must not take on faith.
///
/// What this proves and what it does not: that the runtime reaches the read with
/// a refusal, so the read is live and a run in which it answers success is a run
/// in which this process was not told of a refused launch. It does not prove that
/// a library entry's refused launch reaches the read, because the library's
/// launcher consumes the error first - that one reaches the caller through the
/// status, and the read beside it is the second statement rather than the first.
void ProveLaunchErrorBites() {
    BeginLaunchedCall();
    const cudaError_t refused = cudaLaunchKernel(nullptr, dim3(1), dim3(1), nullptr, 0, nullptr);

    if (refused == cudaSuccess)
    {
        std::fprintf(stderr,
                     "cuda gate: the runtime accepted a launch naming no kernel, so the "
                     "launch-error read certifies nothing\n");
        std::exit(2);
    }

    const cudaError_t read = cudaGetLastError();

    if (read == cudaSuccess)
    {
        std::fprintf(stderr,
                     "cuda gate: a launch the runtime refused left no error to read, so a "
                     "launch that did not happen would pass the read\n");
        std::exit(2);
    }

    BeginLaunchedCall();
    std::printf("launch check : a launch naming no kernel reads back as \"%s\"; every launch "
                "below is read for its own\n",
                cudaGetErrorString(read));
}

// ---------------------------------------------------------------------------
// The arms: launch, read, compare.
// ---------------------------------------------------------------------------

/// The argument type one lane's launched entries read. The floating lanes are
/// handed the grid's own double; the half lane the same argument already rounded
/// to the format it stores, because an entry that takes an F16 reads the value
/// the reference's own half column states and not the double beside it.
template <typename Out> struct ArmArg {
    using type = double;
};

template <> struct ArmArg<boys::F16> {
    using type = boys::F16;
};

template <typename In> In ConvertArmArg(double x) {
    return static_cast<In>(x);
}

template <> inline boys::F16 ConvertArmArg<boys::F16>(double x) {
    return boys::F16(static_cast<float>(x));
}

/// One lane's column of the committed reference: the argument the entry actually
/// evaluated, the value at that argument, and that value's magnitude. The three
/// lanes share one cell layout and differ in which column their own rounding
/// produced, so a lane is a choice of column and never a second comparison.
template <typename Out> struct ArmLane;

template <> struct ArmLane<double> {
    static const std::vector<double>& arg(const Reference& ref) {
        return ref.x;
    }

    static const std::vector<double>& value(const Reference& ref) {
        return ref.v;
    }

    static const std::vector<int>& decade(const Reference& ref) {
        return ref.decade;
    }

    static double widen(double got, bool& unrepresentable) {
        unrepresentable = Unrepresentable(got, -1022);
        return got;
    }
};

template <> struct ArmLane<float> {
    static const std::vector<double>& arg(const Reference& ref) {
        return ref.xf;
    }

    static const std::vector<double>& value(const Reference& ref) {
        return ref.vf;
    }

    static const std::vector<int>& decade(const Reference& ref) {
        return ref.decadeF;
    }

    static double widen(float got, bool& unrepresentable) {
        const std::pair<double, bool> widened = Widen(got);
        unrepresentable = widened.second;
        return widened.first;
    }
};

template <> struct ArmLane<boys::F16> {
    static const std::vector<double>& arg(const Reference& ref) {
        return ref.x16;
    }

    static const std::vector<double>& value(const Reference& ref) {
        return ref.v16;
    }

    static const std::vector<int>& decade(const Reference& ref) {
        return ref.decade16;
    }

    static double widen(boys::F16 got, bool& unrepresentable) {
        const double widened = static_cast<double>(got);
        unrepresentable = Unrepresentable(widened, kF16MinNormalExp);
        return widened;
    }
};

/// The argument column one lane's kernels are handed, at one of two sizes: the
/// reference's own argument count for a batch entry, and the grid's cell count
/// for a single-shape entry, which is handed one element per cell. Both are the
/// same column read at a different stride, which is what the cell layout
/// (element e = order e / count at argument e % count) makes true.
template <typename In>
std::vector<In> ArmArguments(const Reference& ref, std::size_t count) {
    std::vector<In> out(count);

    for (std::size_t e = 0; e < count; ++e)
    {
        out[e] = ConvertArmArg<In>(ref.x[e % ref.count]);
    }

    return out;
}

/// The bound one cell of an arm is measured at: the row's own documented figure,
/// and for the half lane the figure its own cell states - the constant part plus
/// half of the last representable digit of the value the entry returned, which is
/// the form the header publishes. Both are read from the report and neither is
/// transcribed, so relaxing a figure in the library's table relaxes every cell
/// that measures it and no cell here keeps an older one.
double ArmBound(const boys::DeviceOptionInfo& row, double got) {
    if (row.precision == boys::DeviceOptionPrecision::kFp16)
    {
        return HalfBoundAt(got);
    }

    return row.bound;
}

/// How one arm's output array maps onto the reference's cells.
///
/// The single and ladder shapes both fill the grid's own layout - element e is
/// order e / count at argument e % count - so they share one mapping. The all-N
/// shape writes the sorted argument list, so an element's reference argument is
/// the one the sorted order sent there. The each-order shape writes a whole
/// ladder per element at the offsets this gate chose, so its element index splits
/// into argument and order rather than into order and argument.
enum class ArmLayout {
    kCells,  ///< element e is order e/count at argument e%count
    kSorted, ///< the same, over the argument list in non-decreasing order
    kLadderPerElement, ///< element e is argument e/planes at order e%planes
};

/// One arm's values, compared cell by cell with the committed reference at the
/// row's own bound, into the claim that carries the row's own name - which is
/// what lets the coverage statement at the end of this file read this arm as a
/// claim for that row rather than as a table beside it.
template <typename Out>
void MeasureArmCells(const Reference& ref,
                     int claim,
                     const boys::DeviceOptionInfo& row,
                     const std::vector<Out>& got,
                     ArmLayout layout,
                     const SortedArgs& sorted) {
    const std::size_t count = ref.count;
    const std::size_t planes = static_cast<std::size_t>(boys::kMaxBoysOrder) + 1;
    const std::vector<double>& arg = ArmLane<Out>::arg(ref);
    const std::vector<double>& value = ArmLane<Out>::value(ref);
    const std::vector<int>& decade = ArmLane<Out>::decade(ref);

    for (std::size_t e = 0; e < got.size(); ++e)
    {
        std::size_t i = 0;
        int n = 0;

        if (layout == ArmLayout::kCells)
        {
            n = static_cast<int>(e / count);
            i = e % count;
        }
        else if (layout == ArmLayout::kSorted)
        {
            n = static_cast<int>(e / count);
            i = sorted.order[e % count];
        }
        else
        {
            n = static_cast<int>(e % planes);
            i = e / planes;
        }

        bool unrepresentable = false;
        const double widened = ArmLane<Out>::widen(got[e], unrepresentable);

        Measure(claim,
                n,
                arg[i],
                widened,
                value[ref.Index(n, i)],
                decade[ref.Index(n, i)],
                ArmBound(row, widened),
                unrepresentable);
    }
}

/// One arm of the range shapes - the single shape and the ladder shape, which
/// share a signature and differ only in what they fill.
///
/// The single shape is handed one element per cell, so a single launch covers
/// every (order, argument) cell of the grid; the ladder shape is handed one
/// element per argument with a whole ladder written behind each. Both fill the
/// grid's own layout, which is why one measuring loop serves the two.
template <typename Out, typename Fn>
void SweepArmRange(const Reference& ref,
                   const Grid& grid,
                   const boys::DeviceOptionInfo& row,
                   Fn launch) {
    using In = typename ArmArg<Out>::type;
    const bool single = row.shape == boys::DeviceOptionShape::kSingle;
    const std::size_t count = single ? grid.cells : ref.count;

    const int claim = AddClaim(row.name, "A..C", row.bound);
    const std::vector<int> tops(count, boys::kMaxBoysOrder);
    std::vector<Out> out(grid.cells);

    {
        DevBuf<int> dN(count);
        DevBuf<In> dX(count);
        DevBuf<Out> dOut(grid.cells);
        dN.Upload(single ? grid.n : tops);
        dX.Upload(ArmArguments<In>(ref, count));

        BeginLaunchedCall();
        const boys::BoysStatus status =
            launch(dN.get(), dX.get(), dOut.get(), count, nullptr, kGateDivisionForm);
        RequireLaunched(row.name, cudaGetLastError());
        ++LaunchesChecked();
        CheckLaunch(status, row.name);
        dOut.Download(out);
    }

    RequireLaunched(row.name, cudaDeviceSynchronize());
    MeasureArmCells(ref, claim, row, out, ArmLayout::kCells, SortedArgs{});
}

/// One arm of the all-N shape: one common top order, the argument list its own
/// contract asks for (non-decreasing), and a whole ladder behind every argument.
template <typename Out, typename Fn>
void SweepArmAllN(const Reference& ref,
                  const Grid& grid,
                  const SortedArgs& sorted,
                  const boys::DeviceOptionInfo& row,
                  Fn launch) {
    using In = typename ArmArg<Out>::type;
    const std::size_t count = ref.count;
    const int claim = AddClaim(row.name, "A..C", row.bound);

    const std::vector<In> column = ArmArguments<In>(ref, count);
    std::vector<In> hostX(count);

    for (std::size_t j = 0; j < count; ++j)
    {
        hostX[j] = column[sorted.order[j]];
    }

    std::vector<Out> out(grid.cells);

    {
        DevBuf<In> dX(count);
        DevBuf<Out> dOut(grid.cells);
        dX.Upload(hostX);

        BeginLaunchedCall();
        const boys::BoysStatus status =
            launch(boys::kMaxBoysOrder, dX.get(), dOut.get(), count, nullptr, kGateDivisionForm);
        RequireLaunched(row.name, cudaGetLastError());
        ++LaunchesChecked();
        CheckLaunch(status, row.name);
        dOut.Download(out);
    }

    RequireLaunched(row.name, cudaDeviceSynchronize());
    MeasureArmCells(ref, claim, row, out, ArmLayout::kSorted, sorted);
}

/// One arm of the each-order shape, which is handed the offsets its ladders land
/// at. The gate places element i's ladder at i * (kMaxBoysOrder + 1), so the
/// output array is the grid's own layout and no two elements share a plane.
template <typename Out, typename Fn>
void SweepArmEach(const Reference& ref,
                  const Grid& grid,
                  const boys::DeviceOptionInfo& row,
                  Fn launch) {
    using In = typename ArmArg<Out>::type;
    const std::size_t count = ref.count;
    const int planes = boys::kMaxBoysOrder + 1;
    const int claim = AddClaim(row.name, "A..C", row.bound);

    const std::vector<int> tops(count, boys::kMaxBoysOrder);
    std::vector<int> offsets(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        offsets[i] = static_cast<int>(i) * planes;
    }

    std::vector<Out> out(grid.cells);

    {
        DevBuf<int> dN(count);
        DevBuf<In> dX(count);
        DevBuf<int> dOffset(count);
        DevBuf<Out> dOut(grid.cells);
        dN.Upload(tops);
        dX.Upload(ArmArguments<In>(ref, count));
        dOffset.Upload(offsets);

        BeginLaunchedCall();
        const boys::BoysStatus status = launch(
            dN.get(), dX.get(), dOffset.get(), dOut.get(), count, nullptr, kGateDivisionForm);
        RequireLaunched(row.name, cudaGetLastError());
        ++LaunchesChecked();
        CheckLaunch(status, row.name);
        dOut.Download(out);
    }

    RequireLaunched(row.name, cudaDeviceSynchronize());
    MeasureArmCells(ref, claim, row, out, ArmLayout::kLadderPerElement, SortedArgs{});
}

/// Whether a claim for a row is already in the book. The sections above register
/// a row under the report's own name for it, so the name is the row's identity
/// and a second arm for it would be a second reading of one arithmetic.
bool ClaimNamed(const char* lane) {
    for (const Accum& a : Claims())
    {
        if (a.lane == lane)
        {
            return true;
        }
    }

    return false;
}

/// One arm, launched by the driver its row's own shape and lane name.
///
/// The row decides and the arm answers, and a disagreement is a defect rather
/// than a fallback: a row whose shape moved since the table was written would
/// otherwise be measured by a driver that fills a different layout, which is a
/// bound compared against the wrong cells. \p ran is false when the row was
/// already claimed by a section above, in which case the entry is reached and
/// this arm adds no second reading of it.
void RunLaunchedArm(const Reference& ref,
                    const Grid& grid,
                    const SortedArgs& sorted,
                    const LaunchedArm& arm,
                    bool& ran) {
    const boys::DeviceOptionInfo& row = DeviceRow(arm.entry, arm.exp);
    ran = false;

    const bool lane64 = arm.range64 != nullptr || arm.allN64 != nullptr || arm.each64 != nullptr;
    const bool lane32 = arm.range32 != nullptr || arm.allN32 != nullptr || arm.each32 != nullptr;
    const bool lane16 = arm.range16 != nullptr || arm.allN16 != nullptr || arm.each16 != nullptr;
    const bool laneMatches = (row.precision == boys::DeviceOptionPrecision::kFp64 && lane64) ||
                             (row.precision == boys::DeviceOptionPrecision::kFp32 && lane32) ||
                             (row.precision == boys::DeviceOptionPrecision::kFp16 && lane16);

    if (!laneMatches)
    {
        std::fprintf(stderr,
                     "cuda gate: the arm for %s names a kernel of another lane than the row's\n",
                     row.name);
        std::exit(2);
    }

    if (ClaimNamed(row.name))
    {
        return;
    }

    switch (row.shape)
    {
    case boys::DeviceOptionShape::kSingle:
    case boys::DeviceOptionShape::kAllOrders:
        if (arm.range64 != nullptr)
        {
            SweepArmRange<double>(ref, grid, row, arm.range64);
        }
        else if (arm.range32 != nullptr)
        {
            SweepArmRange<float>(ref, grid, row, arm.range32);
        }
        else
        {
            SweepArmRange<boys::F16>(ref, grid, row, arm.range16);
        }

        break;

    case boys::DeviceOptionShape::kAllN:
        if (arm.allN64 != nullptr)
        {
            SweepArmAllN<double>(ref, grid, sorted, row, arm.allN64);
        }
        else if (arm.allN32 != nullptr)
        {
            SweepArmAllN<float>(ref, grid, sorted, row, arm.allN32);
        }
        else
        {
            SweepArmAllN<boys::F16>(ref, grid, sorted, row, arm.allN16);
        }

        break;

    case boys::DeviceOptionShape::kEachOrder:
        if (arm.each64 != nullptr)
        {
            SweepArmEach<double>(ref, grid, row, arm.each64);
        }
        else if (arm.each32 != nullptr)
        {
            SweepArmEach<float>(ref, grid, row, arm.each32);
        }
        else
        {
            SweepArmEach<boys::F16>(ref, grid, row, arm.each16);
        }

        break;

    default:
        // The report defines four shapes and every one of them has a driver
        // above; a row carrying anything else is a row this gate was not built
        // for, and a measurement started on it would be a bound compared against
        // another shape's cells.
        std::fprintf(stderr,
                     "cuda gate: the row for %s carries a shape this gate has no driver for\n",
                     row.name);
        std::exit(2);
    }

    ran = true;
}

/// Every launched entry the report carries, against the arms above.
///
/// The report is the account of the surface and the arms are the instrument, and
/// this is where the two are held to each other: an entry the report carries and
/// no arm names is an entry whose documented bound no cell of this run carries,
/// which is what it is printed as rather than passed over.
///
/// \returns the number of launched entries the report serves and no arm reaches.
std::size_t SurveyLaunchedSurface(const Reference& ref,
                                  const Grid& grid,
                                  const SortedArgs& sorted) {
    std::size_t launchedHere = 0;
    std::size_t claimedAbove = 0;

    for (const LaunchedArm& arm : kLaunchedArms)
    {
        bool ran = false;
        RunLaunchedArm(ref, grid, sorted, arm, ran);
        ran ? ++launchedHere : ++claimedAbove;
    }

    std::size_t armed = 0;
    std::size_t served = 0;
    std::size_t unarmed = 0;
    std::vector<std::string> missing;

    for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions())
    {
        if (!row.built || row.group != boys::DeviceOptionGroup::kLaunched)
        {
            continue;
        }

        ++served;

        bool named = false;

        for (const LaunchedArm& arm : kLaunchedArms)
        {
            named = named || arm.entry == row.entry;
        }

        if (named)
        {
            ++armed;
        }
        else
        {
            ++unarmed;
            missing.push_back(row.name);
        }
    }

    std::printf("\n  the launched surface, from BoysDeviceOptions() and the arms above:\n");
    std::printf("    %zu of %zu arm(s) for the %zu launched entr(ies) the report carries; "
                "%zu launched here,\n    %zu already measured by a section above; %zu launched "
                "entr(ies) the report carries and no\n    arm reaches\n",
                armed,
                std::size(kLaunchedArms),
                served,
                launchedHere,
                claimedAbove,
                unarmed);
    std::printf("    %zu launch(es) read the runtime's error back after them\n",
                LaunchesChecked());

    if (!missing.empty())
    {
        std::printf("    no arm for:");

        for (const std::string& name : missing)
        {
            std::printf(" %s", name.c_str());
        }

        std::printf("\n");
        std::printf("    Each of those names its reason where its arm belongs in the table "
                    "above: either\n    the row was added to the report and no arm was written "
                    "for it, or the library\n    declares the entry and this revision defines no "
                    "kernel for it, which the linker\n    reports as an unresolved external when "
                    "an arm names it.\n");
    }

    return unarmed;
}

// ---------------------------------------------------------------------------
// The coverage, and the two books' one number.
//
// The claims above are made for rows named by the library's report, so the
// gate's row list and the chooser's are one list. This says so out loud,
// because the failure it guards against is silent: an option added to the
// surface and certified by nothing, or a claim left behind for a row the
// library's report does not carry.
//
// It also checks the one figure the report and the shared reference book both
// state, so the fp16 cells' bound cannot come out of one book in the claims and
// the other in the comparisons.
//
// \returns the number of options the library reports that this build serves and
//          the gate has no claim for, plus the claims naming no reported option.
std::size_t ReportDeviceOptionCoverage() {
    char fp16Report[32];
    char fp16Book[32];
    std::snprintf(fp16Report, sizeof(fp16Report), "%.17g", kBoundHalfRow);
    std::snprintf(fp16Book, sizeof(fp16Book), "%.17g", kBoundHalfBase);

    std::printf("\n  device option coverage, from BoysDeviceOptions():\n");

    if (std::abs(kBoundHalfRow - kBoundHalfBase) > 0.0)
    {
        std::printf("    the report states the fp16 bound as %s and the reference book as %s, "
                    "which are not\n    the same number.\n",
                    fp16Report,
                    fp16Book);
    }

    std::size_t served = 0;
    std::size_t uncovered = 0;
    std::vector<std::string> missing;

    for (const boys::DeviceOptionInfo& option : boys::BoysDeviceOptions())
    {
        if (!option.built)
        {
            continue;
        }

        ++served;

        bool claimed = false;

        for (const Accum& a : Claims())
        {
            const std::string name = option.name;
            claimed = claimed || a.lane == name || a.lane.rfind(name + " ", 0) == 0;
        }

        if (!claimed)
        {
            ++uncovered;
            missing.push_back(option.name);
        }
    }

    // The other direction: a claim whose name is no reported option.
    std::vector<std::string> orphans;

    for (const Accum& a : Claims())
    {
        bool known = false;

        for (const boys::DeviceOptionInfo& option : boys::BoysDeviceOptions())
        {
            if (!option.built)
            {
                continue;
            }

            const std::string name = option.name;
            known = known || a.lane == name || a.lane.rfind(name + " ", 0) == 0;
        }

        if (!known && std::find(orphans.begin(), orphans.end(), a.lane) == orphans.end())
        {
            orphans.push_back(a.lane);
        }
    }

    std::printf("    %zu of %zu served option(s) claimed", served - uncovered, served);

    if (!missing.empty())
    {
        std::printf(", with no claim for:");

        for (const std::string& name : missing)
        {
            std::printf(" %s", name.c_str());
        }
    }

    std::printf("; %zu claim name(s) that are no reported option", orphans.size());

    for (const std::string& name : orphans)
    {
        std::printf(" %s", name.c_str());
    }

    std::printf("\n");

    return uncovered + orphans.size() + (std::abs(kBoundHalfRow - kBoundHalfBase) > 0.0 ? 1u : 0u);
}

} // namespace

int main(int argc, char** argv) {
    std::string reference = std::string(BoysDataDir) + "/boys_accuracy_gate_reference.csv";
    std::string digitReference = std::string(BoysDataDir) + "/boys_reference.csv";
    int probeN = -1;
    double probeX = 0.0;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--reference" && i + 1 < argc)
        {
            reference = argv[++i];
        }
        else if (arg == "--digit-reference" && i + 1 < argc)
        {
            digitReference = argv[++i];
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

    // The wall time the run cost, printed at the end: the host gate runs in about
    // sixteen seconds against a ten-minute budget, and a device gate that grew
    // past that budget without saying so would be a gate nobody runs.
    const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();

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
    const DigitGrid digits = LoadDigitGrid(digitReference);
    const Grid grid = MakeGrid(ref);
    const SortedArgs sorted = SortArgs(ref);
    CheckLaunch(boys::BoysCuda::InitializeTables(), "InitializeTables");

    // The handle the device-callable entries read: filled host-side once and
    // passed into the kernels by value.
    boys::BoysDeviceTables tables{};
    CheckLaunch(boys::BoysCuda::DeviceTables(&tables), "DeviceTables");

    if (probeN >= 0)
    {
        RunProbe(ref, tables, probeN, probeX);
        return 0;
    }

    const int slotSingleA = AddClaim(DeviceRow(boys::DeviceEntry::kSingleF64).name, "A", kBoundSingleA);
    const int slotSingleBand = AddClaim(DeviceRow(boys::DeviceEntry::kSingleF64).name, "band", kBoundSingleBand);
    const int slotSingleB = AddClaim(DeviceRow(boys::DeviceEntry::kSingleF64).name, "B", kBoundSingleB);
    const int slotSingleC = AddClaim(DeviceRow(boys::DeviceEntry::kSingleF64).name, "C", kBoundSingleC);
    const int slotDoubleOrders = AddClaim(DeviceRow(boys::DeviceEntry::kAllOrdersF64).name, "A..C", kBoundDoubleBatch);
    const int slotDoubleAllN = AddClaim(DeviceRow(boys::DeviceEntry::kAllNF64).name, "A..C", kBoundDoubleBatch);
    const int slotFloatSingle = AddClaim(DeviceRow(boys::DeviceEntry::kSingleF32).name, "A..C", kBoundFloat);
    // The single entry's fast region-B exponential: a second certified option
    // with a bound of its own, not a relaxation of the lane's.
    const int slotFloatSingleFast =
        AddClaim(DeviceRow(boys::DeviceEntry::kSingleF32Fast, boys::RegionBExp::kFast).name,
                     "A..C",
                     kBoundFloat + kFastExpContribution);
    const int slotFloatOrders = AddClaim(DeviceRow(boys::DeviceEntry::kAllOrdersF32).name, "A..C", kBoundFloat);
    const int slotFloatAllN = AddClaim(DeviceRow(boys::DeviceEntry::kAllNF32).name, "A..C", kBoundFloat);
    const int slotHalfSingle = AddClaim(DeviceRow(boys::DeviceEntry::kSingleF16).name, "A..C", kBoundHalfRow);
    const int slotHalfOrders = AddClaim(DeviceRow(boys::DeviceEntry::kAllOrdersF16).name, "A..C", kBoundHalfRow);
    const int slotHalfAllN = AddClaim(DeviceRow(boys::DeviceEntry::kAllNF16).name, "A..C", kBoundHalfRow);

    SweepDouble(ref,
                     grid,
                     sorted,
                     slotSingleA,
                     slotSingleBand,
                     slotSingleB,
                     slotSingleC,
                     slotDoubleOrders,
                     slotDoubleAllN);
    SweepFloat(ref,
                    grid,
                    sorted,
                    slotFloatSingle,
                    slotFloatSingleFast,
                    slotFloatOrders,
                    slotFloatAllN,
                    DeviceRow(boys::DeviceEntry::kSingleF32Fast, boys::RegionBExp::kFast).name);
    SweepHalf(ref, grid, sorted, slotHalfSingle, slotHalfOrders, slotHalfAllN);
    SweepDeviceChoices(ref, grid);
    SweepHalfChoices(ref, grid);

    // The device-callable entries: one handle, one set of rows, and the same
    // consumer kernels every time. The order and capacity refusals are exercised
    // first, beside the comparisons.
    CheckRefusals(tables);
    SweepDeviceLane(ref, grid, sorted, digits);

    // The whole launched surface, one arm per entry the report carries, and the
    // proof that the read which says a launch happened is able to say it did not.
    // This runs after the sections above so that a row one of them already
    // measured is counted rather than measured twice, and it is what turns the
    // coverage statement below from a list of names into an account of the
    // surface: an entry no arm launches is an entry no cell of this run measures.
    ProveLaunchErrorBites();
    const std::size_t launchedUnarmed = SurveyLaunchedSurface(ref, grid, sorted);

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
    std::printf("45-digit     : %s\n", digitReference.c_str());
    std::printf("               %zu orders x %zu arguments = %zu points per fp64 entry\n",
                static_cast<std::size_t>(boys::kMaxBoysOrder) + 1,
                digits.count,
                digits.cells);
    std::printf("boundaries   : x_ext=%.17g  x0=%.17g  x1=%.17g\n",
                boys::detail::kExtendedBX0,
                boys::detail::kX0,
                boys::detail::kX1);
    std::printf("bounds       : README \"CUDA fp64 | the same budgets as the CPU double lanes\",\n"
                "               README \"CUDA fp32, RegionBExp::kAccurate (the default) | the same"
                " budgets\n"
                "               as the CPU float lanes\" (1.5e-7),\n"
                "               README \"CUDA fp32, RegionBExp::kFast | <= 1.5e-7 + 8e-8 - the"
                " lane's\n"
                "               budget plus the corrected seed's own contribution\",\n"
                "               boys_cuda.hpp \"1e-7 + 1/2 ULP\" for the fp16 entries\n");

    std::printf("\n  lane                     region   points     real  delivered / bound        "
                "worst cell               vacuous   zeros\n");

    for (const Accum& a : Claims())
    {
        PrintClaim(a);
    }

    const std::size_t carriageMissed = PrintPartitionCarriage();

    // The device entries that are one body reached through different shapes: a
    // bound cannot say this and the header does, so the gate checks it. Every slot
    // a call writes is compared, in every one of the three precisions.
    if (!ShapeAgreements().empty())
    {
        std::printf("\n  the device entries that are one body reached through different shapes,\n"
                    "  compared bit for bit over every slot a call writes - kMaxBoysOrder + 1 per\n"
                    "  element, up to that element's own top order:\n");
        std::printf("  %-56s %s\n", "shapes", "identical slots");

        for (const ShapeAgreement& a : ShapeAgreements())
        {
            std::printf("  %-56s %zu / %zu\n", a.what.c_str(), a.identical, a.slots);
        }
    }

    // A device entry and the batch entry of the same precision are one arithmetic
    // reached two ways: the same degree tables, the same inlined body, a lane
    // object that reads the caller's handle where the batch kernel reads a
    // __constant__ symbol. Every value the device call wrote is compared bit for
    // bit with the batch entry of the same precision.
    if (!PairAgreements().empty())
    {
        std::printf("\n  the device entries against the batch entries of the same precision, one\n"
                    "  arithmetic reached two ways, compared bit for bit over every value a call\n"
                    "  writes - the single shapes per cell, the family shapes per order:\n");
        std::printf("  %-64s %s\n", "device entry against batch entry", "identical values");

        for (const PairAgreement& a : PairAgreements())
        {
            std::printf("  %-64s %zu / %zu\n", a.what.c_str(), a.identical, a.values);
        }
    }

    // The 45-digit grid beside the gate grid. Its cells are counted in the rows
    // above, so a cell over bound here fails the gate with them; this table names
    // the worst cell that grid produced.
    if (!DigitRows().empty())
    {
        std::printf("\n  the fp64 device entries on the 45-digit grid as well, at the bound of\n"
                    "  their own row and counted in it: the same entries and the same bounds, a\n"
                    "  second reference whose values carry no format rounding and whose arguments\n"
                    "  are the region boundaries and a logarithmic sweep:\n");
        std::printf("  %-24s %8s %8s  %-24s %s\n",
                    "entry",
                    "points",
                    "real",
                    "delivered / bound",
                    "worst cell");

        for (const DigitRow& r : DigitRows())
        {
            char delivered[64] = "-";
            char location[96] = "-";

            if (r.cell.worstN >= 0)
            {
                std::snprintf(delivered,
                              sizeof(delivered),
                              "%.3g / %.3g",
                              r.cell.worstErr,
                              r.cell.worstBound);
                std::snprintf(location,
                              sizeof(location),
                              "%.3g (n=%d, x=%.6g)",
                              r.cell.worstRatio,
                              r.cell.worstN,
                              r.cell.worstX);
            }

            std::printf("  %-24s %8zu %8zu  %-24s %s\n",
                        r.cell.lane.c_str(),
                        r.cell.points,
                        r.cell.points - r.cell.vacuous,
                        delivered,
                        location);
        }
    }

    // The fast option's row is a bound on the distance from F_n(x) and not on the
    // direction, and the seed it carries is a correction of the hardware
    // approximation rather than the approximation itself. The audit keeps both
    // facts checkable: the contribution is the term the fast bound carries on top
    // of the lane's, and the wrong-sign count is a tripwire for the return of the
    // defect this correction removes.
    if (!ExpAudits().empty())
    {
        std::printf(
            "\n  the single entry's fast region-B exponential, measured against the accurate one\n"
            "  on the same grid. Its bound's second term is the\n"
            "  contribution below, capped by the recurrence's amplification; the wrong-sign\n"
            "  count is the audit for the return of the defect the correction removes.\n");
        std::printf("  %-38s %-26s %-16s %s\n",
                    "lane",
                    "contribution",
                    "wrong sign",
                    "worst wrong-sign relative error");

        for (const ExpAudit& a : ExpAudits())
        {
            char contribution[32];
            char wrong[24];
            char relative[96];
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

            std::printf("  %-38s %-26s %-16s %s\n", a.lane.c_str(), contribution, wrong, relative);
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

    // Every device row is read against the host lane at the policy its own row
    // names, and there is no list of rows that could not be: a row this gate
    // cannot read across the lanes is a row whose own claim or the library's
    // carriage is wrong, and both are defects rather than a section to print.

    const std::size_t uncovered = ReportDeviceOptionCoverage();

    // What the run cost, so a gate that grew past the budget a gate is worth
    // says so in its own output rather than in somebody's recollection of it.
    {
        const std::chrono::duration<double> elapsed =
            std::chrono::steady_clock::now() - started;

        std::printf("\n  wall time: %.1f s\n", elapsed.count());
    }

    if (exceeded > 0)
    {
        std::printf("\n  RESULT: FAIL - %zu cells over their documented bound on this device "
                    "(exit status 1)\n",
                    exceeded);
        return 1;
    }

    // A launched entry the report carries and no arm reaches is a hole of its own
    // kind, and it is reported as one: no cell was over a bound because no cell
    // was compared. The status is its own so that a coverage gap and an accuracy
    // failure cannot be read as the same answer.
    if (launchedUnarmed > 0)
    {
        std::printf("\n  RESULT: FAIL - %zu launched entr(ies) the report carries and no arm "
                    "reaches (exit\n  status 3): their documented bounds are certified by no cell "
                    "of this run, which is a\n  coverage gap and not an accuracy figure - the "
                    "arms above name them.\n",
                    launchedUnarmed);
        return 3;
    }

    if (carriageMissed > 0)
    {
        std::printf("\n  RESULT: FAIL - %zu device row(s) answered a policy naming the "
                    "uniform grid\n  with fits it does not name (exit status 1): the two "
                    "partitions hold the same\n  bar over the same intervals, so no accuracy row "
                    "can tell them apart and the carriage\n  section above is the only statement "
                    "that can\n",
                    carriageMissed);
        return 1;
    }

    if (uncovered > 0)
    {
        std::printf("\n  RESULT: FAIL - %zu device option(s) of the library's report that this "
                    "gate does not account for (exit status 1): an option added to the surface "
                    "is certified by nothing until a claim names it, and a claim whose row the "
                    "library's report does not carry is a row this gate is still judging.\n",
                    uncovered);
        return 1;
    }

    // A build that carries the lane prints nothing here, so the two builds differ
    // only in what each one says it could not do.
    PrintNotCarried();

#ifdef BOYS_CUDA_GATE_FP16
    std::printf("\n  RESULT: every documented device bound met at this revision, %zu of %zu "
                "comparison cells able to discriminate (exit status 0)\n",
                cells - nonDiscriminating,
                cells);
#else
    // Closed seam: the rows named above were never measured, so the statement
    // below is scoped to the entries this build carries rather than left to read
    // as one about the whole library.
    std::printf("\n  RESULT: every documented device bound met at this revision for the entries "
                "this build\n  carries, %zu of %zu comparison cells able to discriminate; the fp16 "
                "rows named above are\n  not carried by this build (BoysFp16 = 0) and no bound of "
                "theirs is stated here (exit\n  status 0)\n",
                cells - nonDiscriminating,
                cells);
#endif // BOYS_CUDA_GATE_FP16
    return 0;
}
