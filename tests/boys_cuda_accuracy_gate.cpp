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
// and the delivered figures do not. Every rung is measured, because the header
// asserts a bound for every rung: the batch entries are swept once per
// multiplier, each sweep launching the instantiation its row names, and the
// device-callable entries once per rung with the rung named at the call.
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
// The device-callable entries (boys_cuda_device.hpp) are measured beside the
// batch ones, and through the kernels of tests/boys_cuda_device_demo.cu rather
// than through a host wrapper: that file includes the public device header and
// the CUDA runtime and nothing of this library's implementation, so a row for
// one of these entries is a measurement of what a consumer's own kernel
// reaches. Each of its threads forms its own argument from a factor pair the
// gate chose exact, so the value measured is the reference's own. The entries
// that are one body reached through different shapes are additionally compared
// with each other bit for bit, which is a stronger statement than a bound and
// is reported beside the rows.
//
// Each of those entries takes the rung as a run-time argument, so each is swept
// at every rung the lane instantiates - held to m * the m = 1 bound of its
// precision, which is the relaxation the contract states - and at every rung
// every device entry is compared bit for bit with the batch entry of the same
// precision and the same rung, because the two are one arithmetic reached two
// ways and no bound can say so. A rung is resident only while the last
// BoysCuda::DeviceTables call named it, and the refusal that follows from that
// is exercised beside the order and capacity refusals.
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
// Exits non-zero when any measured cell is over its bound.

#include "boys/boys.hpp"
#include "boys/boys_cuda.hpp"
#include "boys/boys_impl.hpp" // the region boundaries
#include "boys/f16.hpp"
#include "boys_cuda_device_demo.hpp"
#include "boys_gate_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
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
// The documented device bounds, and the option rows they are the bounds of.
//
// The fp32 and fp16 figures, and the rows the device cells below are claimed
// for, are read from boys::BoysDeviceOptions() — the library's own report of its
// device option space, the table the header's entries are documented in and the
// one a chooser reads. A gate that transcribed them would be a second source of
// truth for the same numbers, and the two could disagree: a row added to the
// surface would be certified by nothing, and a bound could say one thing to a
// chooser and another to the certifier. The fp64 single lane's per-region cells
// are the exception and are transcribed below, because the report states one
// figure per option and that lane's contract is four cells of one option.
// ---------------------------------------------------------------------------

/// The report's row for an option, by entry and axis member.
///
/// A row this gate asks for and the library does not report is the drift this
/// reads the report to prevent, so it stops the gate rather than substituting a
/// figure: a gate that can invent a bound is not a certifier.
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

// README's accuracy contract: "CUDA fp64 | same m*budgets as the CPU double
// lanes". The CPU double single lane is the one with per-region cells
// (1e-15 below the region-A edge, 3e-14 through the extended band and region
// B, 5.5e-14 in region C); the CPU double batch lane publishes one, 5.5e-14.
//
// Three of those cells are transcribed below, because the report states one
// figure per option and this lane's contract is four cells of one option. The
// region-C cell and the batch bound are read from the report, as every fp32 and
// fp16 figure below is.
constexpr double kBoundSingleA = 1e-15;
constexpr double kBoundSingleBand = 3e-14;
constexpr double kBoundSingleB = 3e-14;
const double kBoundSingleC = DeviceRow(boys::DeviceEntry::kSingleF64).bound;
const double kBoundDoubleBatch = DeviceRow(boys::DeviceEntry::kAllOrdersF64).bound;
// README: "CUDA fp32, RegionBExp::kAccurate (the default) | same m*budgets as
// the CPU float lanes" - 1.5e-7, the bound of the single entry's accurate
// region-B exponential, of the batch entries and of the all-n entry. Read from
// the report, which is the header's own figure for that option.
const double kBoundFloat = DeviceRow(boys::DeviceEntry::kDeviceSingleF32).bound;
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
//
// It is read as the difference between the two options' documented bounds - the
// fast option's figure is the lane's plus this contribution, and both figures are
// the report's. A contribution derived here from two reported numbers cannot
// disagree with either of them.
const double kFastExpContribution =
    DeviceRow(boys::DeviceEntry::kDeviceSingleF32Fast, boys::RegionBExp::kFast).bound -
                                    DeviceRow(boys::DeviceEntry::kDeviceSingleF32).bound;
// The fp16 entries' bound is the header's: m * 1e-7 + 1/2 ULP of the returned
// value, which is what HalfBound computes at m = 1. Its constant part is read
// from the report's fp16 row, and the coverage check at the end of this file
// compares it against kBoundHalfBase, the figure the shared reference book
// states: two books, one number, and a disagreement stops the gate.
const double kBoundHalfRow = DeviceRow(boys::DeviceEntry::kDeviceSingleF16).bound;

// The rungs the lane instantiates, in the order the rows print: the multiplier
// and the name a row's label carries for it. m = 1 is the unlabelled row, which
// a lane's plain name already means. The batch entries take one of these as a
// template argument and the device-callable entries as a run-time argument, and
// both read the degree tables cut for it.
struct Rung {
    double multiplier;
    const char* name;
};

constexpr Rung kRungs[] = {{1.0, nullptr},
                           {2.0, "2"},
                           {10.0, "10"},
                           {100.0, "100"},
                           {1e4, "1e4"},
                           {1e8, "1e8"}};

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

std::string Label(const char* lane, const char* rung) {
    return rung == nullptr ? std::string(lane) : std::string(lane) + " m=" + rung;
}

// The same rung, on a phrase rather than on a lane name.
std::string ShapeLabel(const char* what, const char* rung) {
    return rung == nullptr ? std::string(what) : std::string(what) + " (m=" + rung + ")";
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

// The fp16 bound at a multiplier: the header's m * 1e-7 + 1/2 ULP, in ULP of
// the value the entry returned rather than of the value it should have.
double HalfBoundAt(double got, double multiplier) {
    return multiplier * kBoundHalfRow + 0.5 * UlpOf(got, kF16MantissaBits, kF16MinNormalExp);
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
    const std::string doubleSingle = Label(DeviceRow(boys::DeviceEntry::kSingleF64).name, rung);
    const std::string doubleOrders = Label(DeviceRow(boys::DeviceEntry::kAllOrdersF64).name, rung);
    const std::string doubleAllN = Label(DeviceRow(boys::DeviceEntry::kAllNF64).name, rung);
    const std::string floatSingle = Label(DeviceRow(boys::DeviceEntry::kSingleF32).name, rung);
    const std::string floatSingleFast = Label(DeviceRow(boys::DeviceEntry::kSingleF32Fast, boys::RegionBExp::kFast).name, rung);
    const std::string floatOrders = Label(DeviceRow(boys::DeviceEntry::kAllOrdersF32).name, rung);
    const std::string floatAllN = Label(DeviceRow(boys::DeviceEntry::kAllNF32).name, rung);
    const std::string halfSingle = Label(DeviceRow(boys::DeviceEntry::kSingleF16).name, rung);
    const std::string halfOrders = Label(DeviceRow(boys::DeviceEntry::kAllOrdersF16).name, rung);
    const std::string halfAllN = Label(DeviceRow(boys::DeviceEntry::kAllNF16).name, rung);

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
    const int halfSingleSlot = AddClaim(halfSingle.c_str(), "A..C", kMultiplier * kBoundHalfRow);
    const int halfOrderSlot = AddClaim(halfOrders.c_str(), "A..C", kMultiplier * kBoundHalfRow);
    const int halfAllNSlot = AddClaim(halfAllN.c_str(), "A..C", kMultiplier * kBoundHalfRow);

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

// ---------------------------------------------------------------------------
// The device-callable entries, reached from a consumer's own kernel.
// ---------------------------------------------------------------------------

// The slot a family entry wrote for one element: a family call writes
// kMaxBoysOrder + 1 values per element, in the element's own block.
std::size_t FamilySlot(std::size_t element, int order) {
    return element * (static_cast<std::size_t>(boys::kMaxBoysOrder) + 1) +
           static_cast<std::size_t>(order);
}

// Two shapes of one body, compared bit for bit. A bound says how close a return
// is to F_n(x); this says two entries returned the same bits, which is what the
// header claims where two entries reach one body and what no bound can say.
// Only the slots a call writes are compared: an element writes its own top
// order and below, and the slots above it would only count agreements that say
// nothing.
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

// The compile-time-top-order entry against the runtime-top-order one at that
// same order. Both are one call to one body with one top order, so the claim is
// bit identity, and the runtime side of it is the element whose own top order is
// kMaxBoysOrder: a call at a lower order descends from that lower order and
// returns different last bits on purpose, so comparing the two at the element's
// own order would be comparing two arithmetic paths and not two entries.
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
// The entries report through their enum and never throw, so a caller that
// ignores a status is the caller's own risk and the gate is not that caller.
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
// except the double single lane, whose bound is published per region, and one
// set of rows per rung: the multiplier is an argument of the call, so the same
// entry's cells at two rungs are two claims with two bounds.
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
};

// The rows one rung's device cells are measured into, at that rung's bound:
// every documented device bound is m times the m = 1 figure, so a relaxed row
// that held the m = 1 bound would be asserting what it was relaxed out of.
DeviceSlots DeviceClaimSet(const char* rung, double multiplier) {
    DeviceSlots slots;
    const std::string single = Label(DeviceRow(boys::DeviceEntry::kDeviceSingleF64).name, rung);
    const std::string orders64 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF64).name, rung);
    const std::string allN64 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllNF64).name, rung);
    const std::string each64 = Label(DeviceRow(boys::DeviceEntry::kDeviceEachOrderF64).name, rung);
    const std::string single32 = Label(DeviceRow(boys::DeviceEntry::kDeviceSingleF32).name, rung);
    const std::string single32Fast = Label(DeviceRow(boys::DeviceEntry::kDeviceSingleF32Fast, boys::RegionBExp::kFast).name, rung);
    const std::string orders32 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF32).name, rung);
    const std::string allN32 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllNF32).name, rung);
    const std::string each32 = Label(DeviceRow(boys::DeviceEntry::kDeviceEachOrderF32).name, rung);
    const std::string single16 = Label(DeviceRow(boys::DeviceEntry::kDeviceSingleF16).name, rung);
    const std::string orders16 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllOrdersF16).name, rung);
    const std::string allN16 = Label(DeviceRow(boys::DeviceEntry::kDeviceAllNF16).name, rung);
    const std::string each16 = Label(DeviceRow(boys::DeviceEntry::kDeviceEachOrderF16).name, rung);

    slots.singleA = AddClaim(single.c_str(), "A", multiplier * kBoundSingleA);
    slots.singleBand = AddClaim(single.c_str(), "band", multiplier * kBoundSingleBand);
    slots.singleB = AddClaim(single.c_str(), "B", multiplier * kBoundSingleB);
    slots.singleC = AddClaim(single.c_str(), "C", multiplier * kBoundSingleC);
    slots.orders64 = AddClaim(orders64.c_str(), "A..C", multiplier * kBoundDoubleBatch);
    slots.allN64 = AddClaim(allN64.c_str(), "A..C", multiplier * kBoundDoubleBatch);
    slots.each64 = AddClaim(each64.c_str(), "A..C", multiplier * kBoundDoubleBatch);
    slots.single32 = AddClaim(single32.c_str(), "A..C", multiplier * kBoundFloat);
    // The fast option's own bound: the lane's m * 1.5e-7 plus the corrected
    // seed's contribution, which the option carries at every rung.
    slots.single32Fast =
        AddClaim(single32Fast.c_str(), "A..C", multiplier * kBoundFloat + kFastExpContribution);
    slots.orders32 = AddClaim(orders32.c_str(), "A..C", multiplier * kBoundFloat);
    slots.allN32 = AddClaim(allN32.c_str(), "A..C", multiplier * kBoundFloat);
    slots.each32 = AddClaim(each32.c_str(), "A..C", multiplier * kBoundFloat);
    slots.single16 = AddClaim(single16.c_str(), "A..C", multiplier * kBoundHalfRow);
    slots.orders16 = AddClaim(orders16.c_str(), "A..C", multiplier * kBoundHalfRow);
    slots.allN16 = AddClaim(allN16.c_str(), "A..C", multiplier * kBoundHalfRow);
    slots.each16 = AddClaim(each16.c_str(), "A..C", multiplier * kBoundHalfRow);
    return slots;
}

// A device entry and the batch entry of the same precision at the same rung:
// one arithmetic reached two ways - the same degree tables, the same inlined
// body, a lane object that reads the caller's handle instead of a __constant__
// symbol. A bound cannot say that, and the header claims it, so every value the
// device call wrote is compared bit for bit against the batch entry launched on
// the same orders and the same arguments. A divergence is a defect in one of
// the two routes, not a looser bound, so it stops the gate at the cell it was
// found at.
struct RungAgreement {
    std::string what;
    std::size_t identical = 0;
    std::size_t values = 0;
};

std::vector<RungAgreement>& RungAgreements() {
    static std::vector<RungAgreement> agreements;
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
void AgreeRungCells(const std::string& what,
                    const std::vector<T>& device,
                    const std::vector<T>& batch,
                    const Reference& ref) {
    RungAgreement agreement;
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

    RungAgreements().push_back(agreement);
}

// One top order's ladder against the batch entry launched at that same top
// order. Region A descends from the top order, so two top orders are two chains
// of rounding and the comparison has to be made within one top order; the
// counts land in the caller's agreement row, because the statement is about the
// entry rather than about the order.
template <typename T>
void AgreeRungOrder(RungAgreement& agreement,
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
// which is the element's own top order for the shapes that descend from it and
// kMaxBoysOrder for the shape that descends from kMaxBoysOrder at every
// argument. A slot above that is one neither call wrote.
template <typename T>
void AgreeRungFamily(const std::string& what,
                     const std::vector<T>& device,
                     const std::vector<T>& batch,
                     const std::vector<int>& orders,
                     const Reference& ref,
                     bool uniformTop) {
    RungAgreement agreement;
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

    RungAgreements().push_back(agreement);
}

// Every device entry at one rung, measured against the grid through the consumer
// kernels of tests/boys_cuda_device_demo.cu, and compared bit for bit with the
// batch entry of the same rung.
template <double kMultiplier>
void SweepDevice(const Reference& ref,
                 const Grid& grid,
                 const SortedArgs& sorted,
                 const boys::BoysDeviceTables& tables,
                 const DeviceSlots& slots,
                 const char* rung) {
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
                                         dStatus.get(),
                                         kMultiplier),
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
            CheckLaunch(boys::BoysCuda::SingleF64<kMultiplier>(
                            dNb.get(), dXb.get(), dOutb.get(), cells, nullptr),
                        "SingleF64 (one arithmetic)");
            std::vector<double> batch(cells);
            dOutb.Download(batch);
            Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            AgreeRungCells(ShapeLabel("device single f64 and batch single f64", rung),
                           out,
                           batch,
                           ref);
        }
    }

    // The three double family shapes: one body reached by a runtime top order,
    // a compile-time top order and a sink that keeps no ladder. Each is
    // measured at every element's own top order, and the three are compared
    // with each other beside the rows.
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
                                         dStatusLadder.get(),
                                         kMultiplier),
                  "BoysDeviceDemoLadder64");
        CheckDemo(BoysDeviceDemoAllN64(&tables,
                                       dRho.get(),
                                       dD2.get(),
                                       dAllN.get(),
                                       cells,
                                       dStatusAllN.get(),
                                       kMultiplier),
                  "BoysDeviceDemoAllN64");
        CheckDemo(BoysDeviceDemoEach64(&tables,
                                       dN.get(),
                                       dRho.get(),
                                       dD2.get(),
                                       dEach.get(),
                                       cells,
                                       dStatusEach.get(),
                                       kMultiplier),
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

        // The batch entries of the same rung on the same orders and the same
        // arguments, element for element. The single entry is launched over the
        // whole element list, so its element is the device's element. The
        // ladders are launched once per top order, because a ladder descends
        // from the order it is named; the all-n entry takes its arguments
        // non-decreasing, so its plane is permuted back to the reference order
        // before the comparison.
        {
            RungAgreement ladderAgreement;
            RungAgreement eachAgreement;
            ladderAgreement.what =
                ShapeLabel("device all-orders f64 and batch all-orders f64", rung);
            eachAgreement.what = ShapeLabel("device each-order f64 and batch all-orders f64", rung);
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
                    CheckLaunch(boys::BoysCuda::AllOrdersF64<kMultiplier>(
                                    dNb.get(), dXb.get(), dOutb.get(), count, nullptr),
                                "AllOrdersF64 (one arithmetic)");
                    dOutb.Download(batch);
                    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                    AgreeRungOrder(ladderAgreement, ladder, batch, ref.x, top);
                    AgreeRungOrder(eachAgreement, each, batch, ref.x, top);
                }
            }

            RungAgreements().push_back(ladderAgreement);
            RungAgreements().push_back(eachAgreement);

            DevBuf<double> dXs(count);
            DevBuf<double> dOuts(cells);
            dXs.Upload(sorted.x);
            CheckLaunch(boys::BoysCuda::AllNF64<kMultiplier>(
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

            AgreeRungFamily(ShapeLabel("device all-n f64 and batch all-n f64", rung),
                            allN,
                            batchSorted,
                            grid.n,
                            ref,
                            true);
        }

        Agree(ShapeLabel("device all-orders f64 and device each-order f64", rung),
              ladder,
              each,
              ref,
              nmax);
        AgreeFixedTopOrder(ShapeLabel("device all-n f64 and device all-orders f64", rung),
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
                                         dStatus.get(),
                                         kMultiplier),
                  "BoysDeviceDemoSingle32");
        CheckDemo(BoysDeviceDemoSingle32Fast(&tables,
                                             dN.get(),
                                             dRho.get(),
                                             dD2.get(),
                                             dSingleFast.get(),
                                             cells,
                                             dStatus.get(),
                                             kMultiplier),
                  "BoysDeviceDemoSingle32Fast");
        CheckDemo(BoysDeviceDemoLadder32(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dLadder.get(),
                                         cells,
                                         nmax + 1,
                                         dStatus.get(),
                                         kMultiplier),
                  "BoysDeviceDemoLadder32");
        CheckDemo(BoysDeviceDemoAllN32(&tables,
                                       dRho.get(),
                                       dD2.get(),
                                       dAllN.get(),
                                       cells,
                                       dStatus.get(),
                                       kMultiplier),
                  "BoysDeviceDemoAllN32");
        CheckDemo(BoysDeviceDemoEach32(&tables,
                                       dN.get(),
                                       dRho.get(),
                                       dD2.get(),
                                       dEach.get(),
                                       cells,
                                       dStatus.get(),
                                       kMultiplier),
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
        audit.lane = ShapeLabel(DeviceRow(boys::DeviceEntry::kDeviceSingleF32Fast, boys::RegionBExp::kFast).name, rung);

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
                        kMultiplier * kBoundFloat + kFastExpContribution,
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

        // The batch entries of the same rung, on the same orders and the same
        // arguments, element for element, as in the f64 block above.
        {
            {
                DevBuf<int> dNb(cells);
                DevBuf<double> dXb(cells);
                DevBuf<float> dOutb(cells);
                dNb.Upload(grid.n);
                dXb.Upload(grid.x);
                CheckLaunch(boys::BoysCuda::SingleF32<kMultiplier>(
                                dNb.get(), dXb.get(), dOutb.get(), cells, nullptr),
                            "SingleF32 (one arithmetic)");
                std::vector<float> batchSingle(cells);
                dOutb.Download(batchSingle);
                CheckLaunch(boys::BoysCuda::SingleF32<kMultiplier, boys::RegionBExp::kFast>(
                                dNb.get(), dXb.get(), dOutb.get(), cells, nullptr),
                            "SingleF32 (one arithmetic)");
                std::vector<float> batchSingleFast(cells);
                dOutb.Download(batchSingleFast);
                Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                AgreeRungCells(ShapeLabel("device single f32 and batch single f32", rung),
                               single,
                               batchSingle,
                               ref);
                AgreeRungCells(
                    ShapeLabel("device single f32 (fast exp) and batch single f32 (fast exp)",
                               rung),
                    singleFast,
                    batchSingleFast,
                    ref);
            }

            RungAgreement ladderAgreement;
            RungAgreement eachAgreement;
            ladderAgreement.what =
                ShapeLabel("device all-orders f32 and batch all-orders f32", rung);
            eachAgreement.what = ShapeLabel("device each-order f32 and batch all-orders f32", rung);
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
                    CheckLaunch(boys::BoysCuda::AllOrdersF32<kMultiplier>(
                                    dNb.get(), dXb.get(), dOutb.get(), count, nullptr),
                                "AllOrdersF32 (one arithmetic)");
                    dOutb.Download(batch);
                    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                    AgreeRungOrder(ladderAgreement, ladder, batch, ref.xf, top);
                    AgreeRungOrder(eachAgreement, each, batch, ref.xf, top);
                }
            }

            RungAgreements().push_back(ladderAgreement);
            RungAgreements().push_back(eachAgreement);

            DevBuf<double> dXs(count);
            DevBuf<float> dOuts(cells);
            dXs.Upload(sorted.x);
            CheckLaunch(boys::BoysCuda::AllNF32<kMultiplier>(
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

            AgreeRungFamily(ShapeLabel("device all-n f32 and batch all-n f32", rung),
                            allN,
                            batchSorted,
                            grid.n,
                            ref,
                            true);
        }

        Agree(ShapeLabel("device all-orders f32 and device each-order f32", rung),
              ladder,
              each,
              ref,
              nmax);
        AgreeFixedTopOrder(ShapeLabel("device all-n f32 and device all-orders f32", rung),
                           allN,
                           ladder,
                           ref,
                           nmax);
    }

    // The fp16 shapes, at the half value of the argument the grid carries, with
    // the bound the fp16 lane documents and the floor the format sets.
    {
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
                                         dStatus.get(),
                                         kMultiplier),
                  "BoysDeviceDemoSingle16");
        CheckDemo(BoysDeviceDemoLadder16(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         static_cast<void*>(dLadder.get()),
                                         cells,
                                         nmax + 1,
                                         dStatus.get(),
                                         kMultiplier),
                  "BoysDeviceDemoLadder16");
        CheckDemo(BoysDeviceDemoAllN16(&tables,
                                       dRho.get(),
                                       dD2.get(),
                                       static_cast<void*>(dAllN.get()),
                                       cells,
                                       dStatus.get(),
                                       kMultiplier),
                  "BoysDeviceDemoAllN16");
        CheckDemo(BoysDeviceDemoEach16(&tables,
                                       dN.get(),
                                       dRho.get(),
                                       dD2.get(),
                                       static_cast<void*>(dEach.get()),
                                       cells,
                                       dStatus.get(),
                                       kMultiplier),
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
                    HalfBoundAt(got, kMultiplier),
                    Unrepresentable(got, kF16MinNormalExp));
        };

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t e = ref.Index(n, i);

                // An argument past the fp16 range has no reference value at all
                // (the grid's x16 column is infinite there), so it is not a
                // measured cell of the single row - the same rule the batch fp16
                // single row follows.
                if (std::isfinite(ref.x16[i]))
                {
                    measureElement(slots.single16, n, i, static_cast<double>(single[e]));
                }

                measureElement(slots.orders16, n, i, static_cast<double>(ladder[FamilySlot(e, n)]));
                measureElement(slots.allN16, n, i, static_cast<double>(allN[FamilySlot(e, n)]));
                measureElement(slots.each16, n, i, static_cast<double>(each[FamilySlot(e, n)]));
            }
        }

        // The batch entries of the same rung, on the same orders and the same
        // arguments, element for element, as in the f64 block above. The fp16
        // batch entries take their argument in fp16 already, which is the value
        // the grid carries and the demo kernels form.
        {
            {
                DevBuf<int> dNb(cells);
                DevBuf<boys::F16> dXb(cells);
                DevBuf<boys::F16> dOutb(cells);
                dNb.Upload(grid.n);
                dXb.Upload(grid.x16);
                CheckLaunch(boys::BoysCuda::SingleF16<kMultiplier>(
                                dNb.get(), dXb.get(), dOutb.get(), cells, nullptr),
                            "SingleF16 (one arithmetic)");
                std::vector<boys::F16> batchSingle(cells);
                dOutb.Download(batchSingle);
                Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                AgreeRungCells(ShapeLabel("device single f16 and batch single f16", rung),
                               single,
                               batchSingle,
                               ref);
            }

            RungAgreement ladderAgreement;
            RungAgreement eachAgreement;
            ladderAgreement.what =
                ShapeLabel("device all-orders f16 and batch all-orders f16", rung);
            eachAgreement.what = ShapeLabel("device each-order f16 and batch all-orders f16", rung);
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
                    CheckLaunch(boys::BoysCuda::AllOrdersF16<kMultiplier>(
                                    dNb.get(), dXb.get(), dOutb.get(), count, nullptr),
                                "AllOrdersF16 (one arithmetic)");
                    dOutb.Download(batch);
                    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
                    AgreeRungOrder(ladderAgreement, ladder, batch, ref.x16, top);
                    AgreeRungOrder(eachAgreement, each, batch, ref.x16, top);
                }
            }

            RungAgreements().push_back(ladderAgreement);
            RungAgreements().push_back(eachAgreement);

            DevBuf<boys::F16> dXs(count);
            DevBuf<boys::F16> dOuts(cells);
            std::vector<boys::F16> hostSorted(count);

            for (std::size_t j = 0; j < count; ++j)
            {
                hostSorted[j] = boys::F16(static_cast<float>(ref.x[sorted.order[j]]));
            }

            dXs.Upload(hostSorted);
            CheckLaunch(boys::BoysCuda::AllNF16<kMultiplier>(
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

            AgreeRungFamily(ShapeLabel("device all-n f16 and batch all-n f16", rung),
                            allN,
                            batchSorted,
                            grid.n,
                            ref,
                            true);
        }

        Agree(ShapeLabel("device all-orders f16 and device each-order f16", rung),
              ladder,
              each,
              ref,
              nmax);
        AgreeFixedTopOrder(ShapeLabel("device all-n f16 and device all-orders f16", rung),
                           allN,
                           ladder,
                           ref,
                           nmax);
    }
}

// One rung's fp64 entries on the 45-digit grid. The cells are measured into the
// entry's own row as well as into the accumulator below, so a cell over bound
// here fails the gate like every other cell; the accumulator is what names the
// worst cell this grid produced, which the shared row cannot say once it holds
// both grids' cells.
struct DigitRow {
    std::string rung;
    Accum cell;
};

std::vector<DigitRow>& DigitRows() {
    static std::vector<DigitRow> rows;
    return rows;
}

template <double kMultiplier>
void SweepDigit64(const DigitGrid& digits,
                  const boys::BoysDeviceTables& tables,
                  const DeviceSlots& slots,
                  const char* rung) {
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = digits.count;
    const std::size_t cells = digits.cells;
    const std::size_t family = cells * (static_cast<std::size_t>(nmax) + 1);
    const int success = BoysDeviceDemoStatusSuccess();
    const int singleSlots[4] = {slots.singleA, slots.singleBand, slots.singleB, slots.singleC};
    const double singleBounds[4] = {kMultiplier * kBoundSingleA,
                                    kMultiplier * kBoundSingleBand,
                                    kMultiplier * kBoundSingleB,
                                    kMultiplier * kBoundSingleC};
    const double familyBound = kMultiplier * kBoundDoubleBatch;

    // Every thread forms its own argument as rho * d2, exactly as the kernels of
    // the gate grid's sweep do: rho is a power of two, so the product is the
    // grid's own argument to the last bit and the value measured is the
    // reference's own.
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
    single.rung = rung == nullptr ? "1" : rung;
    orders.rung = single.rung;
    each.rung = single.rung;
    allN.rung = single.rung;
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
                                         dStatus.get(),
                                         kMultiplier),
                  "BoysDeviceDemoSingle64 (45-digit grid)");
        CheckDemo(BoysDeviceDemoLadder64(&tables,
                                         dN.get(),
                                         dRho.get(),
                                         dD2.get(),
                                         dLadder.get(),
                                         cells,
                                         nmax + 1,
                                         dStatus.get(),
                                         kMultiplier),
                  "BoysDeviceDemoLadder64 (45-digit grid)");
        CheckDemo(BoysDeviceDemoEach64(&tables,
                                       dN.get(),
                                       dRho.get(),
                                       dD2.get(),
                                       dEach.get(),
                                       cells,
                                       dStatus.get(),
                                       kMultiplier),
                  "BoysDeviceDemoEach64 (45-digit grid)");
        CheckDemo(BoysDeviceDemoAllN64(&tables,
                                       dRho.get(),
                                       dD2.get(),
                                       dAllN.get(),
                                       cells,
                                       dStatus.get(),
                                       kMultiplier),
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
// each - the same call repaired, because a sentinel that survives every call is
// the other way a sentinel check can lie. The decision is on the order and the
// capacity and not on the argument, so four elements carry it.
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

// One shape run at one named rung, with the status the call must return and
// whether it must write. The sentinel is what separates a refusal from a silent
// fallback - an entry that reported a refusal and still wrote would leave its
// output changed, and one that wrote without reporting would leave the status
// clean - and it is also what separates an accepted call from a refusal this
// check mis-read, since a call that is served must leave the sentinel behind.
//
// The tally counts the probes that came back as required, so the report can
// state that the path was exercised rather than only that the gate exited zero.
std::size_t& RetiredRungProbes() {
    static std::size_t probes = 0;
    return probes;
}

std::size_t& ServedRungProbes() {
    static std::size_t probes = 0;
    return probes;
}

template <typename T, typename Launch>
void ProbeOnce(const char* label,
               const char* what,
               std::size_t elements,
               std::size_t slots,
               const T& sentinel,
               int want,
               bool wrote,
               std::size_t& tally,
               Launch&& launch) {
    const std::string name = std::string(what) + " at " + label;
    std::vector<T> hostOut(slots, sentinel);
    std::vector<int> hostStatus(elements, -1);
    DevBuf<T> dOut(slots);
    DevBuf<int> dStatus(elements);
    dOut.Upload(hostOut);
    dStatus.Upload(hostStatus);
    CheckDemo(launch(dOut.get(), dStatus.get()), name.c_str());
    std::vector<T> out(slots);
    std::vector<int> status(elements);
    dOut.Download(out);
    dStatus.Download(status);
    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    ExpectStatus(status, elements, want, name.c_str());
    ReportSentinel(out, sentinel, wrote, name.c_str());
    ++tally;
}

// Every shape run at one named rung, with the status each must return and
// whether it must write. One function carries both halves of the rung contract
// as a consumer meets it: a rung that is not resident, which must refuse at
// every shape rather than read whatever the tables now hold, and m = 1 while a
// relaxed rung is resident, which must be served at every shape because it
// reads the handle's own tables and never the resident one. Every lane is
// covered, and every shape, because both decisions are made per lane and the
// shapes reach them through different bodies.
void CheckRungCalls(const boys::BoysDeviceTables& tables,
                    double multiplier,
                    int want,
                    bool wrote,
                    const char* label,
                    std::size_t& tally) {
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = 2;
    const std::size_t family = count * (static_cast<std::size_t>(nmax) + 1);
    const double sentinel = -12345.0;
    const double x = 3.5;
    const std::vector<int> hostOrder(count, nmax);
    const std::vector<double> hostRho(count, 1.0);
    const std::vector<double> hostD2(count, x);
    std::vector<int> hostStatus(count, -1);
    DevBuf<int> dN(count);
    DevBuf<double> dRho(count);
    DevBuf<double> dD2(count);
    DevBuf<int> dStatus(count);
    dN.Upload(hostOrder);
    dRho.Upload(hostRho);
    dD2.Upload(hostD2);
    dStatus.Upload(hostStatus);

    ProbeOnce(label,
              "BoysDeviceSingleF64",
              count,
              count,
              sentinel,
              want,
              wrote,
              tally,
              [&](double* out, int* status) {
                  return BoysDeviceDemoSingle64(&tables,
                                                dN.get(),
                                                dRho.get(),
                                                dD2.get(),
                                                out,
                                                count,
                                                status,
                                                multiplier);
              });
    ProbeOnce(label,
              "BoysDeviceAllOrdersF64",
              count,
              family,
              sentinel,
              want,
              wrote,
              tally,
              [&](double* out, int* status) {
                  return BoysDeviceDemoLadder64(&tables,
                                                dN.get(),
                                                dRho.get(),
                                                dD2.get(),
                                                out,
                                                count,
                                                nmax + 1,
                                                status,
                                                multiplier);
              });
    ProbeOnce(label,
              "BoysDeviceAllNF64",
              count,
              family,
              sentinel,
              want,
              wrote,
              tally,
              [&](double* out, int* status) {
                  return BoysDeviceDemoAllN64(
                      &tables, dRho.get(), dD2.get(), out, count, status, multiplier);
              });
    ProbeOnce(label,
              "BoysDeviceEachOrderF64",
              count,
              family,
              sentinel,
              want,
              wrote,
              tally,
              [&](double* out, int* status) {
                  return BoysDeviceDemoEach64(&tables,
                                              dN.get(),
                                              dRho.get(),
                                              dD2.get(),
                                              out,
                                              count,
                                              status,
                                              multiplier);
              });

    ProbeOnce(label,
              "BoysDeviceSingleF32",
              count,
              count,
              0.0f,
              want,
              wrote,
              tally,
              [&](float* out, int* status) {
                  return BoysDeviceDemoSingle32(&tables,
                                                dN.get(),
                                                dRho.get(),
                                                dD2.get(),
                                                out,
                                                count,
                                                status,
                                                multiplier);
              });
    ProbeOnce(label,
              "BoysDeviceSingleF32 (fast exp)",
              count,
              count,
              0.0f,
              want,
              wrote,
              tally,
              [&](float* out, int* status) {
                  return BoysDeviceDemoSingle32Fast(&tables,
                                                    dN.get(),
                                                    dRho.get(),
                                                    dD2.get(),
                                                    out,
                                                    count,
                                                    status,
                                                    multiplier);
              });
    ProbeOnce(label,
              "BoysDeviceAllOrdersF32",
              count,
              family,
              0.0f,
              want,
              wrote,
              tally,
              [&](float* out, int* status) {
                  return BoysDeviceDemoLadder32(&tables,
                                                dN.get(),
                                                dRho.get(),
                                                dD2.get(),
                                                out,
                                                count,
                                                nmax + 1,
                                                status,
                                                multiplier);
              });
    ProbeOnce(label,
              "BoysDeviceAllNF32",
              count,
              family,
              0.0f,
              want,
              wrote,
              tally,
              [&](float* out, int* status) {
                  return BoysDeviceDemoAllN32(
                      &tables, dRho.get(), dD2.get(), out, count, status, multiplier);
              });
    ProbeOnce(label,
              "BoysDeviceEachOrderF32",
              count,
              family,
              0.0f,
              want,
              wrote,
              tally,
              [&](float* out, int* status) {
                  return BoysDeviceDemoEach32(&tables,
                                              dN.get(),
                                              dRho.get(),
                                              dD2.get(),
                                              out,
                                              count,
                                              status,
                                              multiplier);
              });

#if BoysFp16
    const boys::F16 halfSentinel(static_cast<float>(sentinel));

    ProbeOnce(label,
              "BoysDeviceSingleF16",
              count,
              count,
              halfSentinel,
              want,
              wrote,
              tally,
              [&](boys::F16* out, int* status) {
                  return BoysDeviceDemoSingle16(&tables,
                                                dN.get(),
                                                dRho.get(),
                                                dD2.get(),
                                                static_cast<void*>(out),
                                                count,
                                                status,
                                                multiplier);
              });
    ProbeOnce(label,
              "BoysDeviceAllOrdersF16",
              count,
              family,
              halfSentinel,
              want,
              wrote,
              tally,
              [&](boys::F16* out, int* status) {
                  return BoysDeviceDemoLadder16(&tables,
                                                dN.get(),
                                                dRho.get(),
                                                dD2.get(),
                                                static_cast<void*>(out),
                                                count,
                                                nmax + 1,
                                                status,
                                                multiplier);
              });
    ProbeOnce(label,
              "BoysDeviceAllNF16",
              count,
              family,
              halfSentinel,
              want,
              wrote,
              tally,
              [&](boys::F16* out, int* status) {
                  return BoysDeviceDemoAllN16(&tables,
                                              dRho.get(),
                                              dD2.get(),
                                              static_cast<void*>(out),
                                              count,
                                              status,
                                              multiplier);
              });
    ProbeOnce(label,
              "BoysDeviceEachOrderF16",
              count,
              family,
              halfSentinel,
              want,
              wrote,
              tally,
              [&](boys::F16* out, int* status) {
                  return BoysDeviceDemoEach16(&tables,
                                              dN.get(),
                                              dRho.get(),
                                              dD2.get(),
                                              static_cast<void*>(out),
                                              count,
                                              status,
                                              multiplier);
              });
#endif // BoysFp16
}

// One rung of the device-callable entries: the handle filled at that rung, the
// rows that rung's cells are measured into, and every entry through the
// consumer's kernels. Filling the handle at the rung is what makes it the
// resident one; the sweep at m = 1 is unaffected by a relaxed rung being
// resident, because that entry reads the m = 1 tables and never the resident
// rung's.
template <double kMultiplier>
void SweepRung(const Reference& ref,
               const Grid& grid,
               const SortedArgs& sorted,
               const DigitGrid& digits,
               const char* rung) {
    boys::BoysDeviceTables tables{};
    CheckLaunch(boys::BoysCuda::DeviceTables<kMultiplier>(&tables), "DeviceTables");
    const DeviceSlots slots = DeviceClaimSet(rung, kMultiplier);
    SweepDevice<kMultiplier>(ref, grid, sorted, tables, slots, rung);
    SweepDigit64<kMultiplier>(digits, tables, slots, rung);
}

// One cell, every entry: what each device entry returns beside the reference,
// so a reported failure can be reproduced and acted on rather than believed.
// The argument is looked up on the committed grid; an argument that is not on
// it is printed as such rather than measured against the nearest node.
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

    // The device-callable entries at the same cell, through the same consumer
    // kernels the rows above are measured through. Each thread forms its own
    // argument as rho * d2; rho = 1 here, which is exact, and the single rows
    // are printed against the same bound their sweep row holds.
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
            HalfBoundAt(static_cast<double>(out16[0]), 1.0),
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
            HalfBoundAt(static_cast<double>(out16[static_cast<std::size_t>(n)]), 1.0),
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
            HalfBoundAt(static_cast<double>(out16[static_cast<std::size_t>(n)]), 1.0),
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
            HalfBoundAt(static_cast<double>(out16[static_cast<std::size_t>(n)]), 1.0),
            have16);
#endif // BoysFp16
    }
}

// ---------------------------------------------------------------------------
// The coverage, and the two books' one number.
//
// The claims above are made for rows named by the library's report, so the
// gate's row list and the chooser's are one list. This says so out loud,
// because the failure it guards against is silent: an option added to the
// surface and certified by nothing, or a claim left behind for a row the
// library no longer reports.
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
            claimed = claimed || a.lane == name || a.lane.rfind(name + " m=", 0) == 0;
        }

        if (!claimed)
        {
            ++uncovered;
            missing.push_back(option.name);
        }
    }

    // The other direction: a claim whose name is no reported option. Names the
    // m = 1 book and the rung books share are counted once.
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
            known = known || a.lane == name || a.lane.rfind(name + " m=", 0) == 0;
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
    // with a bound of its own, not a relaxed rung of the lane's.
    const int slotFloatSingleFast =
        AddClaim(DeviceRow(boys::DeviceEntry::kSingleF32Fast, boys::RegionBExp::kFast).name,
                     "A..C",
                     kBoundFloat + kFastExpContribution);
    const int slotFloatOrders = AddClaim(DeviceRow(boys::DeviceEntry::kAllOrdersF32).name, "A..C", kBoundFloat);
    const int slotFloatAllN = AddClaim(DeviceRow(boys::DeviceEntry::kAllNF32).name, "A..C", kBoundFloat);
    const int slotHalfSingle = AddClaim(DeviceRow(boys::DeviceEntry::kSingleF16).name, "A..C", kBoundHalfRow);
    const int slotHalfOrders = AddClaim(DeviceRow(boys::DeviceEntry::kAllOrdersF16).name, "A..C", kBoundHalfRow);
    const int slotHalfAllN = AddClaim(DeviceRow(boys::DeviceEntry::kAllNF16).name, "A..C", kBoundHalfRow);

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
                    DeviceRow(boys::DeviceEntry::kSingleF32Fast, boys::RegionBExp::kFast).name);
    SweepHalf<1.0>(ref, grid, sorted, slotHalfSingle, slotHalfOrders, slotHalfAllN);

    SweepRelaxed<2.0>(ref, grid, sorted, kRungs[1].name);
    SweepRelaxed<10.0>(ref, grid, sorted, kRungs[2].name);
    SweepRelaxed<100.0>(ref, grid, sorted, kRungs[3].name);
    SweepRelaxed<1e4>(ref, grid, sorted, kRungs[4].name);
    SweepRelaxed<1e8>(ref, grid, sorted, kRungs[5].name);

    // The device-callable entries: one set of rows per rung, one handle per
    // rung, and the same consumer kernels every time. Every documented device
    // bound is a claim about the rung the caller names, so the rung is swept
    // like any other option - a relaxed rung held to the m = 1 bound would be
    // asserting exactly what it was relaxed out of. The order and the capacity
    // refusals are exercised first, while m = 1 is resident, and the retired
    // rung after the last upload, when every other rung has stopped being the
    // one the tables hold.
    CheckRefusals(tables);
    SweepRung<1.0>(ref, grid, sorted, digits, kRungs[0].name);
    SweepRung<2.0>(ref, grid, sorted, digits, kRungs[1].name);
    SweepRung<10.0>(ref, grid, sorted, digits, kRungs[2].name);
    SweepRung<100.0>(ref, grid, sorted, digits, kRungs[3].name);
    SweepRung<1e4>(ref, grid, sorted, digits, kRungs[4].name);
    SweepRung<1e8>(ref, grid, sorted, digits, kRungs[5].name);
    // A rung that was resident earlier in the run and is not any more: the last
    // upload left 1e8 resident, so m = 2 is a rung whose degree tables have been
    // replaced. It is refused rather than run against the tables that replaced
    // them. Every shape is then run once more at m = 1, which is still served -
    // the full-accuracy tables are the handle's own and no upload retires them -
    // so the refusal is one rung of the surface and not the surface closing.
    CheckRungCalls(tables,
                   2.0,
                   BoysDeviceDemoStatusMultiplierNotResident(),
                   false,
                   "m=2",
                   RetiredRungProbes());
    CheckRungCalls(tables,
                   boys::kBoysFullAccuracyMultiplier,
                   BoysDeviceDemoStatusSuccess(),
                   true,
                   "m=1 while m=1e8 is resident",
                   ServedRungProbes());

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
    std::printf("bounds       : README \"CUDA fp64 | same m*budgets as the CPU double lanes\",\n"
                "               README \"CUDA fp32, RegionBExp::kAccurate (the default) | same"
                " m*budgets\n"
                "               as the CPU float lanes\" (1.5e-7),\n"
                "               README \"CUDA fp32, RegionBExp::kFast | <= m*1.5e-7 + 8e-8 - the"
                " lane's\n"
                "               budget plus the corrected seed's own contribution\",\n"
                "               boys_cuda.hpp \"m * 1e-7 + 1/2 ULP\" for the fp16 entries\n"
                "               the device entries at every rung: m times the same figure, which\n"
                "               is the relaxation the multiplier is for\n");

    std::printf("\n  lane                     region   points     real  delivered / bound        "
                "worst cell               vacuous   zeros\n");

    for (const Accum& a : Claims())
    {
        PrintClaim(a);
    }

    // The device entries that are one body reached through different shapes: a
    // bound cannot say this and the header does, so the gate checks it. Every
    // slot a call writes is compared, at every element, in every one of the
    // three precisions.
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

    // A device entry and the batch entry of the same precision at the same rung
    // are one arithmetic reached two ways: the same degree tables, the same
    // inlined body, a lane object that reads the caller's handle where the
    // batch kernel reads a __constant__ symbol. No bound can say that, so every
    // value the device call wrote is compared bit for bit with the batch entry
    // of the same rung.
    if (!RungAgreements().empty())
    {
        std::printf("\n  the device entries against the batch entries of the same rung, one\n"
                    "  arithmetic reached two ways, compared bit for bit over every value a call\n"
                    "  writes - the single shapes per cell, the family shapes per order:\n");
        std::printf("  %-64s %s\n", "device entry against batch entry", "identical values");

        for (const RungAgreement& a : RungAgreements())
        {
            std::printf("  %-64s %zu / %zu\n", a.what.c_str(), a.identical, a.values);
        }
    }

    // What a consumer sees when it names a rung the tables do not hold: every
    // shape refuses, the status says which refusal it is, and no output slot is
    // written. A silent fallback would show up here as a written slot or as a
    // clean status on a call that could not be served. The same shapes are then
    // run at m = 1, which is the other half of the same contract: the rung the
    // caller named is the only thing that decides.
    if (RetiredRungProbes() != 0 && ServedRungProbes() != 0)
    {
        std::printf("\n  the rung, from the consumer kernels: %zu shapes run at m = %s while\n"
                    "  m = %s is resident, and the same %zu shapes run at m = 1 beside them.\n"
                    "  The first refused: every element of every shape reported\n"
                    "  kMultiplierNotResident and every output slot still held its sentinel, so\n"
                    "  the rung is a value the caller branches on. The second was served: every\n"
                    "  element reported kSuccess and every slot was written, because m = 1 reads\n"
                    "  the handle's own tables and no upload retires them. The order and the\n"
                    "  capacity refusals were exercised the same way, while m = %g was resident.\n",
                    RetiredRungProbes(),
                    kRungs[1].name,
                    kRungs[5].name,
                    ServedRungProbes(),
                    kRungs[0].multiplier);
    }

    // The 45-digit grid beside the gate grid. Its cells are counted in the rows
    // above, so a cell over bound here fails the gate with them; this table is
    // what names the worst cell that grid produced at each rung, which a row
    // holding both grids' cells cannot say.
    if (!DigitRows().empty())
    {
        std::printf("\n  the fp64 device entries on the 45-digit grid as well, at the bound of\n"
                    "  their own row and counted in it: the same entries and the same bounds, a\n"
                    "  second reference whose values carry no format rounding and whose arguments\n"
                    "  are the region boundaries and a logarithmic sweep:\n");
        std::printf("  %-6s %-24s %8s %8s  %-24s %s\n",
                    "rung",
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

            std::printf("  %-6s %-24s %8zu %8zu  %-24s %s\n",
                        r.rung.c_str(),
                        r.cell.lane.c_str(),
                        r.cell.points,
                        r.cell.points - r.cell.vacuous,
                        delivered,
                        location);
        }
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

    const std::size_t uncovered = ReportDeviceOptionCoverage();

    if (exceeded > 0)
    {
        std::printf("\n  RESULT: FAIL - %zu cells over their documented bound on this device "
                    "(exit status 1)\n",
                    exceeded);
        return 1;
    }

    if (uncovered > 0)
    {
        std::printf("\n  RESULT: FAIL - %zu device option(s) of the library's report that this "
                    "gate does not account for (exit status 1): an option added to the surface "
                    "is certified by nothing until a claim names it, and a claim whose row the "
                    "library no longer reports is a row this gate is still judging.\n",
                    uncovered);
        return 1;
    }

    std::printf("\n  RESULT: every documented device bound met at this revision, %zu of %zu "
                "comparison cells able to discriminate (exit status 0)\n",
                cells - nonDiscriminating,
                cells);
    return 0;
}
