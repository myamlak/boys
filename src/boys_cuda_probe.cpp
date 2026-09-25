// The option probe's host side: the workload, the protocol, the folding of the
// passes into figures, the refusal to order noise, and the text a consumer
// reads.
//
// The clock is not here. Every timed region is a CUDA event pair this file
// opens and closes through boys_cuda_probe_kernels.cu, because the host's own
// submission cost must not be in the figure and the events are the only thing
// that keeps it out. What this file decides is what to time, how many times, and
// what the results are allowed to say.
//
// The shape is the CPU option probe's (src/boys_probe.cpp in this library): one
// fixed question per class, an instrument that judges the run rather than the
// library, a pass the instrument cannot vouch for is discarded and never
// averaged in, an entry's figure is the minimum of its clean passes, the
// resolution is what the run measured rather than a bar chosen in advance, and a
// rival inside that resolution is named as unseparable instead of being ordered.
// The differences are all forced by the device: the instrument is a kernel and
// not a host spin, the card is named instead of the host, and the entries that
// exist to run inside the caller's kernel are measured by subtraction rather
// than launched.

#include "boys/boys_cuda_probe.hpp"

#include "boys/boys_cuda.hpp"
#include "boys/boys_device_tables.hpp"
#include "boys/f16.hpp"

#include "boys_cuda_probe_entries.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// The device-side boundary (boys_cuda_probe_kernels.cu). Return codes as the
// lane's own exported functions: 0 success, 1 an internal error, 2 a CUDA
// failure.
// ---------------------------------------------------------------------------
extern "C" {
int BoysCudaProbeDeviceCount(int* out);
int BoysCudaProbeSetDevice(int ordinal);
int BoysCudaProbeCurrentDevice(int* out);
int BoysCudaProbeDeviceQuery(int ordinal, boys::probe_detail::ProbeDeviceFacts* out);
int BoysCudaProbeBuildFacts(boys::probe_detail::ProbeBuildFacts* out);
int BoysCudaProbeAlloc(void** out, std::size_t bytes);
int BoysCudaProbeFree(void* p);
int BoysCudaProbeUpload(void* dst, const void* src, std::size_t bytes);
int BoysCudaProbeDownload(void* dst, const void* src, std::size_t bytes);
int BoysCudaProbeSynchronize();
int BoysCudaProbeTime(boys::probe_detail::ProbeTimeRequest* request);
}

namespace boys {
namespace {

using probe_detail::ProbeEntry;
using probe_detail::ProbeQuestion;
using probe_detail::ProbeTimeRequest;
using probe_detail::ProbeWhat;

// ---------------------------------------------------------------------------
// The documented bounds each precision's lane carries at full accuracy. These
// are read from the library's documentation, not measured here: what a lane
// delivers on a card is the accuracy gate's business, and this probe spends its
// time on cost. They are in the report so that a faster precision is not read as
// a faster option at the same accuracy.
// ---------------------------------------------------------------------------
constexpr double kFp64Bound = 5.5e-14;
constexpr double kFp32Bound = 1.5e-7;
constexpr double kFp32FastBound = 1.5e-7 + 8e-8;
constexpr double kFp16Bound = 1e-7 + 0x1p-11;

/// How the workload's orders are drawn: the sum of two shell angular momenta,
// which is the distribution a shell-pair batch presents. The shells are capped
/// so that the sum can reach the workload's own highest order.
constexpr int kShellMax = 7;

std::string Text(const char* literal) {
    return std::string(literal);
}

template <typename... Args>
std::string Text(const char* format, Args... args) {
    const int size = std::snprintf(nullptr, 0, format, args...);

    if (size <= 0)
    {
        return std::string();
    }

    std::string out(static_cast<std::size_t>(size), '\0');
    std::snprintf(out.data(), static_cast<std::size_t>(size) + 1, format, args...);
    return out;
}

/// Slowest over fastest, less one, in percent. Zero for a set that cannot
/// disagree with itself.
double SpreadPercent(const std::vector<double>& samples) {
    if (samples.size() < 2)
    {
        return 0.0;
    }

    const auto ends = std::minmax_element(samples.begin(), samples.end());

    if (!(*ends.first > 0.0))
    {
        return 0.0;
    }

    return 100.0 * (*ends.second / *ends.first - 1.0);
}

double MedianOf(std::vector<double> samples) {
    if (samples.empty())
    {
        return 0.0;
    }

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

/// A CUDA version as the runtime packs it, written the way a consumer reads it.
std::string FormatCudaVersion(int packed) {
    if (packed <= 0)
    {
        return "unknown";
    }

    return Text("%d.%d", packed / 1000, (packed % 1000) / 10);
}

/// Bytes as a consumer reads a memory size.
std::string FormatBytes(std::size_t bytes) {
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    return Text("%.1f GiB", gib);
}

/// The status in words, so a caller reading the text does not have to know the
/// enumerators to see why nothing was measured.
const char* StatusName(DeviceProbeStatus status) {
    switch (status)
    {
        case DeviceProbeStatus::kSuccess:
            return "measured";
        case DeviceProbeStatus::kNoDevice:
            return "this machine has no CUDA device";
        case DeviceProbeStatus::kDeviceNotFound:
            return "the device asked for is not one this machine has";
        case DeviceProbeStatus::kInvalidArgument:
            return "the request named no entry this library has";
        default:
            return "a CUDA operation failed";
    }
}

// ---------------------------------------------------------------------------
// The entries this build offers.
// ---------------------------------------------------------------------------

struct EntryInfo {
    ProbeEntry entry;
    const char* name;
    const char* precision;
    const char* shape;
    ProbeQuestion question;
    bool inKernel;
    double bound;
};

const char* QuestionName(ProbeQuestion question) {
    switch (question)
    {
        case ProbeQuestion::kSingle:
            return "single";
        case ProbeQuestion::kAllOrders:
            return "all-orders";
        default:
            return "all-n";
    }
}

const char* QuestionAsked(ProbeQuestion question) {
    switch (question)
    {
        case ProbeQuestion::kSingle:
            return "one value per argument: F_n(x) at that argument's own order n";
        case ProbeQuestion::kAllOrders:
            return "a ladder per argument: F_0(x)..F_n(x) at that argument's own order n";
        default:
            return "one common ladder for the whole batch: F_0(x)..F_nmax(x) for every argument";
    }
}

/// The option table, built from the library rather than from a literal list of
/// what this build carries: the fp16 half of it is present only when the fp16
/// seam is open, and the report says so instead of quietly offering fewer
/// options.
std::vector<EntryInfo> EnumerateEntries(bool fp16, std::vector<std::string>& unoffered) {
    const auto row = [](ProbeEntry entry,
                        const char* name,
                        const char* precision,
                        const char* shape,
                        ProbeQuestion question,
                        bool inKernel,
                        double bound) {
        return EntryInfo{entry, name, precision, shape, question, inKernel, bound};
    };

    std::vector<EntryInfo> entries{
        row(ProbeEntry::kSingleF64, "single-fp64", "fp64", "single", ProbeQuestion::kSingle, false,
            kFp64Bound),
        row(ProbeEntry::kSingleF32, "single-fp32", "fp32", "single", ProbeQuestion::kSingle, false,
            kFp32Bound),
        row(ProbeEntry::kSingleF32Fast, "single-fp32-fast", "fp32", "single",
            ProbeQuestion::kSingle, false, kFp32FastBound),
        row(ProbeEntry::kAllOrdersF64, "all-orders-fp64", "fp64", "all-orders",
            ProbeQuestion::kAllOrders, false, kFp64Bound),
        row(ProbeEntry::kAllOrdersF32, "all-orders-fp32", "fp32", "all-orders",
            ProbeQuestion::kAllOrders, false, kFp32Bound),
        row(ProbeEntry::kAllNF64, "all-n-fp64", "fp64", "all-n", ProbeQuestion::kAllN, false,
            kFp64Bound),
        row(ProbeEntry::kAllNF32, "all-n-fp32", "fp32", "all-n", ProbeQuestion::kAllN, false,
            kFp32Bound),
        row(ProbeEntry::kDeviceSingleF64, "device-single-fp64", "fp64", "single",
            ProbeQuestion::kSingle, true, kFp64Bound),
        row(ProbeEntry::kDeviceSingleF32, "device-single-fp32", "fp32", "single",
            ProbeQuestion::kSingle, true, kFp32Bound),
        row(ProbeEntry::kDeviceAllOrdersF64, "device-all-orders-fp64", "fp64", "all-orders",
            ProbeQuestion::kAllOrders, true, kFp64Bound),
        row(ProbeEntry::kDeviceAllOrdersF32, "device-all-orders-fp32", "fp32", "all-orders",
            ProbeQuestion::kAllOrders, true, kFp32Bound),
        row(ProbeEntry::kDeviceAllNF64, "device-all-n-fp64", "fp64", "all-n", ProbeQuestion::kAllN,
            true, kFp64Bound),
        row(ProbeEntry::kDeviceAllNF32, "device-all-n-fp32", "fp32", "all-n", ProbeQuestion::kAllN,
            true, kFp32Bound),
        row(ProbeEntry::kDeviceEachOrderF64, "device-each-order-fp64", "fp64", "each-order",
            ProbeQuestion::kAllOrders, true, kFp64Bound),
        row(ProbeEntry::kDeviceEachOrderF32, "device-each-order-fp32", "fp32", "each-order",
            ProbeQuestion::kAllOrders, true, kFp32Bound),
    };

    const EntryInfo fp16Rows[] = {
        row(ProbeEntry::kSingleF16, "single-fp16", "fp16", "single", ProbeQuestion::kSingle, false,
            kFp16Bound),
        row(ProbeEntry::kAllOrdersF16, "all-orders-fp16", "fp16", "all-orders",
            ProbeQuestion::kAllOrders, false, kFp16Bound),
        row(ProbeEntry::kAllNF16, "all-n-fp16", "fp16", "all-n", ProbeQuestion::kAllN, false,
            kFp16Bound),
        row(ProbeEntry::kDeviceSingleF16, "device-single-fp16", "fp16", "single",
            ProbeQuestion::kSingle, true, kFp16Bound),
        row(ProbeEntry::kDeviceAllOrdersF16, "device-all-orders-fp16", "fp16", "all-orders",
            ProbeQuestion::kAllOrders, true, kFp16Bound),
        row(ProbeEntry::kDeviceAllNF16, "device-all-n-fp16", "fp16", "all-n", ProbeQuestion::kAllN,
            true, kFp16Bound),
        row(ProbeEntry::kDeviceEachOrderF16, "device-each-order-fp16", "fp16", "each-order",
            ProbeQuestion::kAllOrders, true, kFp16Bound),
    };

    for (const EntryInfo& candidate : fp16Rows)
    {
        if (fp16)
        {
            entries.push_back(candidate);
        } else
        {
            unoffered.push_back(candidate.name);
        }
    }

    return entries;
}

// ---------------------------------------------------------------------------
// The workload.
// ---------------------------------------------------------------------------

/// The library's own question, as the CPU probe asks it: a set of arguments in
/// which each argument carries its own highest order, drawn as the sum of two
/// shell angular momenta, over a log-uniform range that populates every region
/// of the kernel.
///
/// The pairs are sorted by argument because the uniform-order entries document
/// that their arguments are non-decreasing: this probe does not sort for them,
/// so it has to present them a batch that already satisfies what they promise.
struct Workload {
    std::vector<int> n;
    std::vector<double> x;
    std::vector<float> xf;
    std::vector<std::uint16_t> xh;
};

Workload BuildWorkload(const DeviceProbeOptions& options) {
    Workload work;
    const std::size_t count = options.count;

    work.n.resize(count);
    work.x.resize(count);

    std::uint64_t state = options.seed;

    const auto next = [&state]() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>(state >> 33);
    };

    const double logSpan = std::log(options.xHi / options.xLo);

    for (std::size_t i = 0; i < count; ++i)
    {
        const int shellA = static_cast<int>(next() % static_cast<std::uint32_t>(kShellMax + 1));
        const int shellB = static_cast<int>(next() % static_cast<std::uint32_t>(kShellMax + 1));
        work.n[i] = std::min(shellA + shellB, options.nmax);

        const double u = static_cast<double>(next()) / 4294967296.0;
        work.x[i] = options.xLo * std::exp(u * logSpan);
    }

    // Sort the (order, argument) pairs by argument, so the uniform-order
    // entries' documented precondition holds.
    std::vector<std::size_t> order(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        order[i] = i;
    }

    std::sort(order.begin(), order.end(), [&work](std::size_t a, std::size_t b) {
        return work.x[a] < work.x[b];
    });

    std::vector<int> sortedN(count);
    std::vector<double> sortedX(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        sortedN[i] = work.n[order[i]];
        sortedX[i] = work.x[order[i]];
    }

    work.n.swap(sortedN);
    work.x.swap(sortedX);

    // The narrow lanes' argument arrays. The batch entries of the fp32 and fp16
    // lanes take the double argument and round it themselves; the device-callable
    // entries of those lanes take the narrowed argument, which is what a caller's
    // own kernel would hold in a register.
    work.xf.resize(count);
    work.xh.resize(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        work.xf[i] = static_cast<float>(work.x[i]);
        work.xh[i] = F16(static_cast<float>(work.x[i])).Bits();
    }

    return work;
}

// ---------------------------------------------------------------------------
// Device memory, owned for the length of the measurement. Everything is
// allocated and filled once, before the first clock, and freed after the last.
// ---------------------------------------------------------------------------

struct DeviceBuffer {
    void* pointer = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : pointer(other.pointer) {
        other.pointer = nullptr;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other)
        {
            if (pointer != nullptr)
            {
                BoysCudaProbeFree(pointer);
            }

            pointer = other.pointer;
            other.pointer = nullptr;
        }

        return *this;
    }
    ~DeviceBuffer() {
        if (pointer != nullptr)
        {
            BoysCudaProbeFree(pointer);
        }
    }
};

/// The calling thread's device, put back where it was however this returns.
struct DeviceGuard {
    int previous = 0;
    bool ok = false;

    DeviceGuard() {
        ok = BoysCudaProbeCurrentDevice(&previous) == 0;
    }

    ~DeviceGuard() {
        if (ok)
        {
            BoysCudaProbeSetDevice(previous);
        }
    }
};

// ---------------------------------------------------------------------------
// One timed region.
// ---------------------------------------------------------------------------

/// Device milliseconds of \p reps back-to-back launches of whatever the request
/// names. A non-positive result is a failure, never a fast entry.
bool TimeRegion(ProbeTimeRequest request, double& outMs) {
    request.outMs = &outMs;
    return BoysCudaProbeTime(&request) == 0 && outMs >= 0.0;
}

/// Nanoseconds per argument, which is the unit every figure in this report is
/// in: one argument's worth of the question the entry was asked.
double PerArgument(double milliseconds, int reps, std::size_t count) {
    return milliseconds * 1.0e6 / static_cast<double>(reps) / static_cast<double>(count);
}

/// What one entry's timed region or regions produced, in device milliseconds.
///
/// A launched row fills \c withMs and leaves \c withoutMs at zero. An in-kernel
/// row fills both: the caller's kernel with the call in it, and the same kernel
/// with the call removed and the traffic kept. Both halves are kept, not only
/// their difference, so the report can print the subtraction rather than assert
/// that one happened.
struct TimedEntry {
    double withMs = 0.0;
    double withoutMs = 0.0;
    bool subtracted = false;
    bool ok = false;
};

/// One entry's timed figure for one round, in milliseconds of device time.
///
/// A launched entry is one region. A device-callable entry is two, taken
/// adjacent in the same round: two adjacent regions and not two passes, so a
/// drift in the card's clocks moves both halves together and cancels in the
/// difference.
///
/// Which half goes first alternates by round, and each half's figure is its best
/// round. A region carries a small fixed cost of its own — the event pair, the
/// first launch's cold start, the host's submission of the launch — which is
/// divided by the repetition count rather than by the call, and whichever half
/// pays it lands in the difference as a term that shrinks as the repetition count
/// rises. That is a bias, not noise: it does not average away, and it moves the
/// figure between two repetition counts, which is what the subtraction control
/// found. Alternating the order puts each half in the favoured position on half
/// the rounds, so each half's best round is a round it went first in.
TimedEntry TimeEntry(const EntryInfo& info,
                     const ProbeTimeRequest& base,
                     int reps,
                     bool baselineFirst = false) {
    TimedEntry timed;

    if (!info.inKernel)
    {
        ProbeTimeRequest request = base;
        request.what = static_cast<int>(ProbeWhat::kLaunchedEntry);
        request.entry = static_cast<int>(info.entry);
        request.reps = reps;
        timed.ok = TimeRegion(request, timed.withMs);
        return timed;
    }

    ProbeTimeRequest withBoys = base;
    withBoys.what = static_cast<int>(ProbeWhat::kInKernelEntry);
    withBoys.entry = static_cast<int>(info.entry);
    withBoys.reps = reps;
    withBoys.withBoys = 1;

    ProbeTimeRequest withoutBoys = withBoys;
    withoutBoys.withBoys = 0;

    timed.subtracted = true;

    if (baselineFirst)
    {
        timed.ok = TimeRegion(withoutBoys, timed.withoutMs) && TimeRegion(withBoys, timed.withMs);
    } else
    {
        timed.ok = TimeRegion(withBoys, timed.withMs) && TimeRegion(withoutBoys, timed.withoutMs);
    }

    return timed;
}

/// Times one entry at two very different repetition counts and fills a
/// repetition-count control from what the two said.
///
/// The counts have to be far apart for the check to mean anything: a cost that
/// repeats with the launch rather than with the call is divided by a different
/// number of launches at each count, so it changes the per-argument figure by
/// that cost's share of it times the ratio of the counts. An entry whose figure
/// holds across the two has had such a cost divided out.
///
/// Both halves of a subtraction are reported, not only their difference, so that
/// a reader can see the difference was not two unstable readings cancelling.
/// Returns false when either count could not be timed.
///
/// Each count is read over several rounds with the order of the two halves
/// alternating, and each half's figure there is its best round, because that is
/// the estimator the reported figures use: checking a minimum over rounds against
/// a single reading would report the difference between the two estimators rather
/// than the difference between the two counts.
bool RunRepetitionControl(const EntryInfo& info,
                          const DeviceProbeMeasurement& row,
                          const ProbeTimeRequest& base,
                          const DeviceProbeOptions& clamped,
                          double judgedAgainst,
                          DeviceProbeRepetitionControl& out) {
    out.entry = row.name;
    out.route = row.route;
    out.repetitionsLow = clamped.controlRepetitionsLow;
    out.repetitionsHigh = clamped.controlRepetitionsHigh;
    out.judgedAgainst = judgedAgainst;

    constexpr int kRounds = 3;
    const double infinite = std::numeric_limits<double>::infinity();

    double withLow = infinite;
    double withoutLow = infinite;
    double withHigh = infinite;
    double withoutHigh = infinite;
    bool subtracted = false;
    bool ok = true;

    for (int round = 0; round < kRounds && ok; ++round)
    {
        // Alternated, so the per-region cost that whichever half goes second
        // pays is not read as a difference between the halves.
        const bool baselineFirst = (round % 2) == 1;
        const TimedEntry low = TimeEntry(info, base, clamped.controlRepetitionsLow, baselineFirst);
        const TimedEntry high =
            TimeEntry(info, base, clamped.controlRepetitionsHigh, baselineFirst);

        ok = low.ok && high.ok;
        subtracted = low.subtracted && high.subtracted;

        if (!ok)
        {
            break;
        }

        withLow = std::min(withLow, low.withMs);
        withHigh = std::min(withHigh, high.withMs);

        if (subtracted)
        {
            withoutLow = std::min(withoutLow, low.withoutMs);
            withoutHigh = std::min(withoutHigh, high.withoutMs);
        }
    }

    if (!ok)
    {
        out.entry.clear();
        out.note = Text("'%s' at %d and %d launches could not be timed on this run, so what the "
                        "%s figures exclude is not established by this run",
                        row.name.c_str(),
                        clamped.controlRepetitionsLow,
                        clamped.controlRepetitionsHigh,
                        row.route.c_str());
        return false;
    }

    // The same shape the figures are: a launched row is its region, an in-kernel
    // row the difference between the region that held the call and the one that
    // did not.
    const auto Figure = [&](double withMs, double withoutMs, int reps) {
        const double ms = subtracted ? std::max(0.0, withMs - withoutMs) : withMs;
        return PerArgument(ms, reps, clamped.count);
    };

    out.nsPerArgumentLow = Figure(withLow, withoutLow, clamped.controlRepetitionsLow);
    out.nsPerArgumentHigh = Figure(withHigh, withoutHigh, clamped.controlRepetitionsHigh);

    if (subtracted)
    {
        out.nsPerArgumentBaselineLow =
            PerArgument(withoutLow, clamped.controlRepetitionsLow, clamped.count);
        out.nsPerArgumentBaselineHigh =
            PerArgument(withoutHigh, clamped.controlRepetitionsHigh, clamped.count);
    }

    const double smaller = std::min(out.nsPerArgumentLow, out.nsPerArgumentHigh);

    out.difference =
        smaller > 0.0
            ? std::fabs(out.nsPerArgumentHigh - out.nsPerArgumentLow) / smaller
            : 0.0;
    // The resolution this run measured for that row is what the two counts are
    // judged against: a run that could not place that row has no yardstick for
    // its repetition counts either, and the control says so rather than inventing
    // a bar here.
    out.agrees = judgedAgainst > 0.0 && out.difference <= judgedAgainst;

    const std::string halves =
        subtracted
            ? Text("the subtraction's own two halves were %.3f ns/argument with the call against "
                   "%.3f without it at %d launches, and %.3f against %.3f at %d, so the difference "
                   "is between two readings that were themselves stable, not two unstable ones "
                   "cancelling",
                   out.nsPerArgumentBaselineLow + out.nsPerArgumentLow,
                   out.nsPerArgumentBaselineLow,
                   clamped.controlRepetitionsLow,
                   out.nsPerArgumentBaselineHigh + out.nsPerArgumentHigh,
                   out.nsPerArgumentBaselineHigh,
                   clamped.controlRepetitionsHigh)
            : Text("the region holds no subtraction, so there is one reading per count rather "
                   "than two");

    out.note = Text("'%s' on the %s route at %d and %d launches gives %.3f against %.3f "
                    "ns/argument, a disagreement of %.2f%%, %s the %.2f%% this row can be placed "
                    "at (the canary's widest admitted spread and this row's own passes together), "
                    "which is the check that a cost repeating with the launch rather than with the "
                    "call did not survive into the figures at the repetition count they were taken "
                    "at; %s",
                    row.name.c_str(),
                    row.route.c_str(),
                    clamped.controlRepetitionsLow,
                    clamped.controlRepetitionsHigh,
                    out.nsPerArgumentLow,
                    out.nsPerArgumentHigh,
                    100.0 * out.difference,
                    out.agrees ? "inside" : "OUTSIDE",
                    100.0 * judgedAgainst,
                    halves.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// The conclusion for one question class.
// ---------------------------------------------------------------------------

/// Folds one class's live measurements into a verdict, on the CPU probe's rule:
/// the resolution is what the run measured - the larger of the instrument's
/// widest admitted spread and the leader's own spread across its clean passes -
/// and a rival inside it is named rather than ordered.
void Conclude(DeviceProbeClass& clause,
              const std::vector<DeviceProbeMeasurement*>& live,
              double canarySpreadPercent,
              int cleanPasses,
              int totalPasses) {
    clause.verdict = DeviceProbeVerdict::kCannotDetermine;

    // Two kinds of row may not be ranked: an in-kernel row whose subtraction did
    // not resolve, and a row whose repetition control disagreed. Neither may be
    // silently dropped either, because a reader who expected it in the table
    // would take its absence for a build fact.
    std::vector<DeviceProbeMeasurement*> ordered;

    for (DeviceProbeMeasurement* measurement : live)
    {
        if (!measurement->subtractionResolved)
        {
            clause.notOrdered.push_back(
                Text("%s (in-kernel): the caller kernel with the call in it cost the same as the "
                     "same kernel with the call removed, within this run's clock; its arithmetic "
                     "is inside the traffic that kernel already does at this workload, so no cost "
                     "was resolved for it",
                     measurement->name.c_str()));
        } else if (measurement->repetitionChecked && !measurement->repetitionAgrees)
        {
            clause.notOrdered.push_back(
                Text("%s: %s", measurement->name.c_str(), measurement->repetitionNote.c_str()));
        } else
        {
            ordered.push_back(measurement);
        }
    }

    if (ordered.empty())
    {
        clause.reason = Text("no entry of this class produced a figure that resolved above the "
                             "%s it was measured against and held across its repetition control, so "
                             "there is nothing to order",
                             live.empty() ? "instrument" : "kernel without the call");
        clause.confidence =
            Text("CANNOT DETERMINE: %zu of %zu entries in this class were measured and none "
                 "resolved",
                 ordered.size(),
                 ordered.size());
        return;
    }

    const auto cheaper = [](const DeviceProbeMeasurement* a, const DeviceProbeMeasurement* b) {
        return a->nsPerArgument < b->nsPerArgument;
    };

    const DeviceProbeMeasurement* leader =
        *std::min_element(ordered.begin(), ordered.end(), cheaper);

    clause.fastestOverall = leader->name;

    if (cleanPasses < 2)
    {
        clause.reason = Text("'%s' is the fastest entry of this class (%.3f ns/argument), but "
                             "only %d of %d passes was admitted, and how far two entries can be "
                             "ordered apart rests on how the admitted passes disagreed; a second "
                             "pass the canary vouches for is what this needs",
                             leader->name.c_str(),
                             leader->nsPerArgument,
                             cleanPasses,
                             totalPasses);
        clause.confidence = Text("CANNOT DETERMINE: %d of %d passes admitted, and the resolution "
                                 "needs two",
                                 cleanPasses,
                                 totalPasses);
        return;
    }

    if (!(leader->nsPerArgument > 0.0))
    {
        // A per-argument figure of zero is not a fast entry: it is a timed region
        // whose device time rounded away, which happens when the workload is too
        // small for the clock that measured it. Ordering a class on that would
        // rank the workload rather than the arithmetic.
        clause.reason = Text("no entry of this class produced a figure above zero: at this "
                             "workload and this repetition count the timed regions' device time "
                             "rounded away, so the class is not ordered. Raising the argument "
                             "count or the repetition count is what this needs.");
        clause.confidence = "CANNOT DETERMINE: the fastest figure in this class is zero, which is "
                            "the timer's resolution rather than a cost";
        return;
    }

    clause.resolution = std::max(canarySpreadPercent / 100.0, leader->spread - 1.0);

    // Each rival is judged against its own spread as well as the leader's: a row
    // whose passes wandered 40% is not placed by a run that can order 2%, however
    // steady the row it is compared against was, and using the leader's figure
    // alone would either order that row or fail a class the run can place.
    struct InseparableRow {
        const DeviceProbeMeasurement* rival;
        double threshold;
    };

    std::vector<InseparableRow> inseparable;

    for (const DeviceProbeMeasurement* rival : ordered)
    {
        if (rival == leader)
        {
            continue;
        }

        const double threshold =
            std::max({canarySpreadPercent / 100.0, leader->spread - 1.0, rival->spread - 1.0});

        if (rival->nsPerArgument <= leader->nsPerArgument * (1.0 + threshold))
        {
            inseparable.push_back({rival, threshold});
        }
    }

    if (!inseparable.empty())
    {
        const InseparableRow& nearest = inseparable.front();

        clause.reason = Text("'%s' is the fastest entry of this class (%.3f ns/argument), but "
                             "'%s' (%.3f ns/argument) is %.2f%% behind it, inside the %.2f%% "
                             "this run can order; the probe does not order noise",
                             leader->name.c_str(),
                             leader->nsPerArgument,
                             nearest.rival->name.c_str(),
                             nearest.rival->nsPerArgument,
                             100.0 * (nearest.rival->nsPerArgument / leader->nsPerArgument - 1.0),
                             100.0 * nearest.threshold);
        clause.confidence = Text("CANNOT DETERMINE: %zu of %zu entries in this class are inside "
                                 "the %.2f%% this run can order, which is what the canary and the "
                                 "spread of the two rows measured",
                                 inseparable.size(),
                                 ordered.size(),
                                 100.0 * nearest.threshold);

        for (const InseparableRow& row : inseparable)
        {
            clause.inseparable.push_back(
                Text("%s %.3f ns/argument (%.2f%% behind the leader, inside the %.2f%% this run "
                     "can order: the canary's spread and the two rows' own spreads together)",
                     row.rival->name.c_str(),
                     row.rival->nsPerArgument,
                     100.0 * (row.rival->nsPerArgument / leader->nsPerArgument - 1.0),
                     100.0 * row.threshold));
        }

        return;
    }

    const DeviceProbeMeasurement* nearest = nullptr;
    double nearestThreshold = clause.resolution;

    for (const DeviceProbeMeasurement* rival : ordered)
    {
        if (rival == leader)
        {
            continue;
        }

        if (nearest == nullptr || rival->nsPerArgument < nearest->nsPerArgument)
        {
            nearest = rival;
            nearestThreshold =
                std::max({canarySpreadPercent / 100.0, leader->spread - 1.0, rival->spread - 1.0});
        }
    }

    clause.verdict = DeviceProbeVerdict::kRecommend;
    clause.recommended = leader->name;

    if (nearest == nullptr)
    {
        clause.reason = Text("'%s' is the fastest entry of this class: %.3f ns/argument over %d "
                             "admitted of %d passes, spread %.2fx",
                             leader->name.c_str(),
                             leader->nsPerArgument,
                             leader->cleanPasses,
                             totalPasses,
                             leader->spread);
        clause.confidence = Text("HIGH: it is the only entry of this class this run measured, so "
                                 "there is nothing in the class to order it against (canary spread "
                                 "%.2f%%, its own passes spread %.2f%%)",
                                 canarySpreadPercent,
                                 100.0 * (leader->spread - 1.0));
        return;
    }

    clause.reason = Text("'%s' is the fastest entry of this class: %.3f ns/argument over %d "
                         "admitted of %d passes, spread %.2fx",
                         leader->name.c_str(),
                         leader->nsPerArgument,
                         leader->cleanPasses,
                         totalPasses,
                         leader->spread);
    clause.confidence =
        Text("HIGH: the nearest rival in this class is at least %.2f%% behind, beyond the %.2f%% "
             "this pair can be ordered at (canary spread %.2f%%, leader's own passes spread %.2f%%, "
             "the rival's %.2f%%)",
             100.0 * (nearest->nsPerArgument / leader->nsPerArgument - 1.0),
             100.0 * nearestThreshold,
             canarySpreadPercent,
             100.0 * (leader->spread - 1.0),
             100.0 * (nearest->spread - 1.0));
}

/// The card-specific caveat, built from what the runtime reports about the card
/// that answered rather than from what was known about the card this was
/// written on.
std::string BuildCaveat(const DeviceProbeDevice& device) {
    std::string text = Text("This ranking is about this card: %s, compute capability %d.%d. ",
                            device.name.c_str(),
                            device.computeMajor,
                            device.computeMinor);

    if (device.singleToDoublePrecisionPerfRatio >= 2)
    {
        text += Text("It documents a ratio of single- to double-precision throughput of %d:1, so "
                     "how far the fp64 lane sits behind the fp32 lane here is this card's ratio "
                     "and not the library's -- a part with a ratio of 2 orders those two "
                     "differently, and a figure taken here would be the wrong one there. ",
                     device.singleToDoublePrecisionPerfRatio);
    } else if (device.singleToDoublePrecisionPerfRatio == 1)
    {
        text += Text("It reports equal single- and double-precision throughput, so the fp64 and "
                     "fp32 lanes here are separated by their arithmetic rather than by a "
                     "throughput ratio. ");
    }

    if (device.computeMajor < 8)
    {
        text += Text("Its compute capability predates the bf16 tensor instructions, so nothing "
                     "here runs on a tensor path and none of these figures is a tensor figure. ");
    }

    text += Text("A different card, a different toolkit or a different set of target "
                 "architectures can rank these entries differently, which is why the probe is run "
                 "where the numbers are used rather than read from documentation.");
    return text;
}

// ---------------------------------------------------------------------------
// The measurement.
// ---------------------------------------------------------------------------

} // namespace

DeviceProbeReport RunDeviceOptionProbe(const DeviceProbeOptions& options) {
    DeviceProbeReport report;
    report.options = options;

    int deviceCount = 0;

    if (BoysCudaProbeDeviceCount(&deviceCount) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = "the CUDA runtime could not report how many devices this machine has";
        return report;
    }

    if (deviceCount <= 0)
    {
        report.status = DeviceProbeStatus::kNoDevice;
        report.failure = "this machine has no CUDA device, so there is no device option to "
                         "measure";
        return report;
    }

    if (options.device < 0 || options.device >= deviceCount)
    {
        report.status = DeviceProbeStatus::kDeviceNotFound;
        report.failure = Text("device %d was asked for and this machine has %d device(s), numbered "
                              "0..%d",
                              options.device,
                              deviceCount,
                              deviceCount - 1);
        return report;
    }

    // The caller's own device is where they left it, whatever happens below.
    const DeviceGuard guard;

    if (BoysCudaProbeSetDevice(options.device) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the CUDA runtime could not make device %d current", options.device);
        return report;
    }

    probe_detail::ProbeDeviceFacts facts{};

    if (BoysCudaProbeDeviceQuery(options.device, &facts) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("device %d could not be queried", options.device);
        return report;
    }

    probe_detail::ProbeBuildFacts build{};

    if (BoysCudaProbeBuildFacts(&build) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = "the toolkit and architecture this library was built with could not be "
                         "read";
        return report;
    }

    report.device.ordinal = options.device;
    report.device.name = facts.name;
    report.device.computeMajor = facts.computeMajor;
    report.device.computeMinor = facts.computeMinor;
    report.device.totalMemoryBytes = static_cast<std::size_t>(facts.totalMemoryBytes);
    report.device.multiProcessorCount = facts.multiProcessorCount;
    report.device.clockKHz = facts.clockKHz;
    report.device.memoryClockKHz = facts.memoryClockKHz;
    report.device.singleToDoublePrecisionPerfRatio = facts.singleToDoublePrecisionPerfRatio;
    report.device.driverVersion = facts.driverVersion;
    report.device.runtimeVersion = facts.runtimeVersion;
    report.device.toolkitVersion = build.toolkit;
    report.device.architectures = build.architectures;
    report.caveat = BuildCaveat(report.device);

    // The lane's own table upload, for the device now current. Idempotent per
    // device, so a caller who had already set this device up pays nothing and
    // sees no change.
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the coefficient tables could not be uploaded to device %d",
                              options.device);
        return report;
    }

    BoysDeviceTables handle{};

    if (BoysCuda::DeviceTables(&handle) != BoysStatus::kSuccess)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the device-table handle for device %d could not be filled",
                              options.device);
        return report;
    }

    const bool fp16 = handle.pieceStart32 != nullptr;

    // --- The workload, and the buffers it lives in ---------------------------
    DeviceProbeOptions clamped = options;
    clamped.count = std::max<std::size_t>(options.count, 1u);
    clamped.nmax = std::min(std::max(options.nmax, 1), kMaxBoysOrder);
    clamped.passes = std::max(options.passes, 1);
    clamped.rounds = std::max(options.rounds, 1);
    clamped.repetitions = std::max(options.repetitions, 1);
    clamped.controlRepetitionsLow = std::max(options.controlRepetitionsLow, 1);
    clamped.controlRepetitionsHigh = std::max(options.controlRepetitionsHigh, 1);

    if (!(options.xHi > options.xLo) || !(options.xLo > 0.0))
    {
        clamped.xLo = 1e-3;
        clamped.xHi = 40.0;
    }

    report.options = clamped;

    const Workload work = BuildWorkload(clamped);
    report.workloadCount = clamped.count;

    std::vector<EntryInfo> entries = EnumerateEntries(fp16, report.unoffered);

    if (!clamped.only.empty())
    {
        std::vector<EntryInfo> narrowed;

        for (const EntryInfo& info : entries)
        {
            bool asked = false;

            for (const std::string& name : clamped.only)
            {
                if (name == info.name)
                {
                    asked = true;
                }
            }

            if (asked)
            {
                narrowed.push_back(info);
            }
        }

        for (const std::string& name : clamped.only)
        {
            bool known = false;

            for (const EntryInfo& info : entries)
            {
                if (name == info.name)
                {
                    known = true;
                }
            }

            for (const std::string& unbuilt : report.unoffered)
            {
                if (name == unbuilt)
                {
                    known = true;
                }
            }

            if (!known)
            {
                report.notAnEntry.push_back(name);
            }
        }

        if (narrowed.empty())
        {
            report.status = DeviceProbeStatus::kInvalidArgument;
            report.failure = "every name asked for is either no entry of this library or an entry "
                             "this build does not carry, so nothing was measured";
            return report;
        }

        entries.swap(narrowed);
    }

    const std::size_t ladderValues = clamped.count * (kMaxBoysOrder + 1);

    DeviceBuffer orders;
    DeviceBuffer args64;
    DeviceBuffer args32;
    DeviceBuffer args16;
    DeviceBuffer output;
    DeviceBuffer canarySink;

    if (BoysCudaProbeAlloc(&orders.pointer, clamped.count * sizeof(int)) != 0 ||
        BoysCudaProbeAlloc(&args64.pointer, clamped.count * sizeof(double)) != 0 ||
        BoysCudaProbeAlloc(&args32.pointer, clamped.count * sizeof(float)) != 0 ||
        BoysCudaProbeAlloc(&args16.pointer, clamped.count * sizeof(std::uint16_t)) != 0 ||
        BoysCudaProbeAlloc(&output.pointer, ladderValues * sizeof(double)) != 0 ||
        BoysCudaProbeAlloc(&canarySink.pointer, sizeof(unsigned long long)) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the workload's buffers could not be allocated on device %d",
                              options.device);
        return report;
    }

    if (BoysCudaProbeUpload(orders.pointer, work.n.data(), clamped.count * sizeof(int)) != 0 ||
        BoysCudaProbeUpload(args64.pointer, work.x.data(), clamped.count * sizeof(double)) != 0 ||
        BoysCudaProbeUpload(args32.pointer, work.xf.data(), clamped.count * sizeof(float)) != 0 ||
        BoysCudaProbeUpload(
            args16.pointer, work.xh.data(), clamped.count * sizeof(std::uint16_t)) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the workload could not be uploaded to device %d", options.device);
        return report;
    }

    // Every figure's request differs only in what it names; the buffers are the
    // same resident ones for the whole run, so nothing is transferred while any
    // clock is open.
    ProbeTimeRequest base{};
    base.handle = &handle;
    base.n = static_cast<const int*>(orders.pointer);
    base.x = static_cast<const double*>(args64.pointer);
    base.xf = static_cast<const float*>(args32.pointer);
    base.xh = args16.pointer;
    base.out = output.pointer;
    base.canarySink = static_cast<unsigned long long*>(canarySink.pointer);
    base.count = clamped.count;
    base.nmax = clamped.nmax;

    report.measurements.resize(entries.size());

    for (std::size_t e = 0; e < entries.size(); ++e)
    {
        const EntryInfo& info = entries[e];
        DeviceProbeMeasurement& measurement = report.measurements[e];
        measurement.name = info.name;
        measurement.precision = info.precision;
        measurement.shape = info.shape;
        measurement.question = QuestionName(info.question);
        measurement.route = info.inKernel ? "in-kernel" : "launched";
        measurement.launchedByLibrary = !info.inKernel;
        measurement.documentedBound = info.bound;
    }

    std::vector<std::vector<double>> passCost(entries.size(),
                                              std::vector<double>(static_cast<std::size_t>(
                                                                      clamped.passes),
                                                                  std::numeric_limits<double>::infinity()));
    std::vector<std::vector<double>> passBaseline(passCost);

    // --- Warm-up. The first launch of a kernel pays one-time costs that are
    // the card's and the context's rather than the entry's, so nothing here is
    // timed.
    bool warm = true;

    for (const EntryInfo& info : entries)
    {
        warm = TimeEntry(info, base, 1).ok && warm;
    }

    double canaryWarm = 0.0;
    ProbeTimeRequest canaryBase = base;
    canaryBase.what = static_cast<int>(ProbeWhat::kCanary);
    // One launch per region for the canary, which does its fixed work inside the
    // kernel rather than by being launched many times.
    canaryBase.reps = 1;

    if (TimeRegion(canaryBase, canaryWarm))
    {
        report.canaryFloorMs = canaryWarm;
    } else
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = "the canary kernel could not be timed, so this run has no instrument to "
                         "judge its own passes with and no figures are reported";
        return report;
    }

    if (!warm)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = "the warm-up launch of at least one entry failed, so the device is not "
                         "running this build's kernels and nothing was timed";
        return report;
    }

    // --- Passes --------------------------------------------------------------
    const int canarySamples = std::max(3, clamped.rounds + 1);

    for (int pass = 0; pass < clamped.passes; ++pass)
    {
        DeviceProbePass record;
        std::vector<double> canaryMs;
        // The two halves of a subtraction are minimised separately and over the
        // same rounds, not subtracted round by round. Each region in a round
        // opens its own stream and its own events, so whichever of the two is
        // timed first pays a ramp the other does not; taking each half's own best
        // round leaves that where it belongs, in the ramp, and not in the
        // difference.
        std::vector<double> withThisPass(entries.size(),
                                         std::numeric_limits<double>::infinity());
        std::vector<double> withoutThisPass(entries.size(),
                                            std::numeric_limits<double>::infinity());

        const auto takeCanary = [&]() {
            double ms = 0.0;

            if (TimeRegion(canaryBase, ms))
            {
                canaryMs.push_back(ms);
                report.canaryFloorMs = report.canaryFloorMs > 0.0
                                           ? std::min(report.canaryFloorMs, ms)
                                           : ms;
            }
        };

        takeCanary();

        const auto start = std::chrono::steady_clock::now();

        for (int round = 0; round < clamped.rounds; ++round)
        {
            for (std::size_t e = 0; e < entries.size(); ++e)
            {
                const TimedEntry timed =
                    TimeEntry(entries[e], base, clamped.repetitions, (round % 2) == 1);

                if (!timed.ok)
                {
                    continue;
                }

                withThisPass[e] = std::min(withThisPass[e], timed.withMs);

                if (timed.subtracted)
                {
                    withoutThisPass[e] = std::min(withoutThisPass[e], timed.withoutMs);
                }
            }

            takeCanary();
        }

        const auto finish = std::chrono::steady_clock::now();
        record.seconds = std::chrono::duration<double>(finish - start).count();

        while (static_cast<int>(canaryMs.size()) < canarySamples && !canaryMs.empty())
        {
            takeCanary();
        }

        record.canaryMedianMs = MedianOf(canaryMs);
        record.canarySpread = SpreadPercent(canaryMs);
        record.disturbed = record.canarySpread > clamped.canarySpreadThreshold;

        if (record.disturbed)
        {
            ++report.disturbedPasses;
        } else
        {
            ++report.cleanPasses;
            report.canarySpread = std::max(report.canarySpread, record.canarySpread);

            for (std::size_t e = 0; e < entries.size(); ++e)
            {
                const double with = PerArgument(withThisPass[e], clamped.repetitions, clamped.count);
                // A subtraction that comes out at or below zero is the entry's
                // arithmetic inside the noise of its own baseline: the row is a
                // cost and cannot be below zero, and the report says separately
                // that the subtraction did not resolve.
                const double without = std::isfinite(withoutThisPass[e])
                                           ? PerArgument(withoutThisPass[e],
                                                         clamped.repetitions,
                                                         clamped.count)
                                           : 0.0;

                passBaseline[e][static_cast<std::size_t>(pass)] = without;
                passCost[e][static_cast<std::size_t>(pass)] =
                    entries[e].inKernel ? std::max(0.0, with - without) : with;
            }
        }

        report.passes.push_back(record);
    }

    // --- Fold the passes into figures ---------------------------------------
    for (std::size_t e = 0; e < entries.size(); ++e)
    {
        DeviceProbeMeasurement& measurement = report.measurements[e];
        double fastest = std::numeric_limits<double>::infinity();
        double slowest = 0.0;
        double baseline = 0.0;
        int clean = 0;

        for (int pass = 0; pass < clamped.passes; ++pass)
        {
            const double cost = passCost[e][static_cast<std::size_t>(pass)];

            if (!std::isfinite(cost))
            {
                continue;
            }

            if (cost < fastest)
            {
                // The baseline is the one the fastest pass was taken with, so a
                // reader subtracting the two gets the subtraction this figure
                // came from and not two passes' worth.
                fastest = cost;
                baseline = passBaseline[e][static_cast<std::size_t>(pass)];
            }

            slowest = std::max(slowest, cost);
            ++clean;
        }

        measurement.cleanPasses = clean;
        measurement.disturbedPasses = report.disturbedPasses;
        measurement.measured = clean > 0;
        // A launched row is not a subtraction and always carries its figure. An
        // in-kernel row carries one only when its own best pass put the entry's
        // half above the half it was measured against.
        measurement.subtractionResolved = !entries[e].inKernel || fastest > 0.0;

        if (clean > 0)
        {
            measurement.nsPerArgument = fastest;
            measurement.nsPerArgumentMax = slowest;
            measurement.nsPerArgumentBaseline = baseline;
            measurement.spread = fastest > 0.0 ? slowest / fastest : 1.0;
        }
    }

    // --- What the timed work actually produced. One read-back per entry,
    // outside every clock, so the report can show that the launches ran and ran
    // on the intended arguments.
    {
        std::vector<double> host(ladderValues, 0.0);

        for (std::size_t e = 0; e < entries.size(); ++e)
        {
            const EntryInfo& info = entries[e];

            // One more launch per entry, and its result read back. It is not
            // timed: what it establishes is that the timed launches produced
            // values, on the arguments the workload holds.
            if (!TimeEntry(info, base, 1).ok)
            {
                continue;
            }

            const std::size_t bytes = info.precision == std::string("fp64")
                                          ? ladderValues * sizeof(double)
                                          : info.precision == std::string("fp32")
                                                ? ladderValues * sizeof(float)
                                                : ladderValues * sizeof(std::uint16_t);

            if (BoysCudaProbeDownload(host.data(), output.pointer, bytes) != 0)
            {
                continue;
            }

            double sum = 0.0;

            if (info.precision == std::string("fp64"))
            {
                for (std::size_t i = 0; i < ladderValues; ++i)
                {
                    sum += host[i];
                }
            } else if (info.precision == std::string("fp32"))
            {
                const auto* values = reinterpret_cast<const float*>(host.data());

                for (std::size_t i = 0; i < ladderValues; ++i)
                {
                    sum += static_cast<double>(values[i]);
                }
            } else
            {
                const auto* bits = reinterpret_cast<const std::uint16_t*>(host.data());

                for (std::size_t i = 0; i < ladderValues; ++i)
                {
                    sum += static_cast<double>(static_cast<float>(F16::FromBits(bits[i])));
                }
            }

            report.measurements[e].checkedSum = sum;
        }

        (void)BoysCudaProbeSynchronize();
    }

    // --- One conclusion per question class ----------------------------------
    //
    // After the controls, because a row the controls set aside is not ranked, and
    // run to a fixed point: every row a class would recommend is put through the
    // repetition control first, and if it does not agree the class falls to the
    // next row, which is checked in its turn.
    const auto ConcludeClasses = [&]() {
        report.classes.clear();

        for (int q = 0; q < static_cast<int>(ProbeQuestion::kCount); ++q)
        {
            const auto question = static_cast<ProbeQuestion>(q);

            DeviceProbeClass clause;
            clause.question = QuestionName(question);
            clause.asked = QuestionAsked(question);

            std::vector<DeviceProbeMeasurement*> live;

            for (DeviceProbeMeasurement& measurement : report.measurements)
            {
                if (measurement.measured && measurement.question == clause.question)
                {
                    live.push_back(&measurement);
                }
            }

            Conclude(clause, live, report.canarySpread, report.cleanPasses, clamped.passes);
            report.classes.push_back(clause);
        }
    };

    // A row is judged against what this run's own instrument and that row's own
    // passes measured, never against the widest threshold some other class
    // needed: a class whose rival wandered is a fact about that rival.
    const auto RowResolution = [&](const DeviceProbeMeasurement* row) {
        return std::max(report.canarySpread / 100.0, row->spread - 1.0);
    };

    const auto FindEntry = [&](const std::string& name) -> const EntryInfo* {
        for (const EntryInfo& info : entries)
        {
            if (info.name == name)
            {
                return &info;
            }
        }

        return nullptr;
    };

    /// Puts one row through the repetition control and records the outcome on the
    /// row, so that the class conclusions can see it.
    const auto CheckRow = [&](DeviceProbeMeasurement& row, DeviceProbeRepetitionControl* printed) {
        const EntryInfo* info = FindEntry(row.name);

        if (info == nullptr)
        {
            return false;
        }

        DeviceProbeRepetitionControl outcome;
        const bool timed = RunRepetitionControl(*info,
                                                row,
                                                base,
                                                clamped,
                                                RowResolution(&row),
                                                outcome);

        row.repetitionChecked = true;
        row.repetitionAgrees = timed && outcome.agrees;
        row.repetitionNote = outcome.note;

        if (printed != nullptr)
        {
            *printed = outcome;
        }

        return true;
    };

    // --- The controls --------------------------------------------------------
    //
    // One check per route, both under the protocol the figures were taken under.
    // The check is that one entry timed at two very different repetition counts
    // gives the same per-argument figure: a cost that repeats with the launch
    // rather than with the call is amortised over a different number of launches
    // at each count, so a timer that had absorbed one would not agree with
    // itself. The launched route is judged against the device-side floor as well
    // — a kernel launched the same way that does no Boys arithmetic at all —
    // because that route's figure has a launch inside it by construction.
    {
        DeviceProbeMeasurement* fastestLaunched = nullptr;
        DeviceProbeMeasurement* fastestSubtracted = nullptr;

        for (DeviceProbeMeasurement& measurement : report.measurements)
        {
            if (!measurement.measured)
            {
                continue;
            }

            if (measurement.launchedByLibrary)
            {
                if (fastestLaunched == nullptr ||
                    measurement.nsPerArgument < fastestLaunched->nsPerArgument)
                {
                    fastestLaunched = &measurement;
                }
            } else if (measurement.subtractionResolved)
            {
                // Only a row whose subtraction resolved a cost above its own
                // baseline has a figure for the control to check.
                if (fastestSubtracted == nullptr ||
                    measurement.nsPerArgument < fastestSubtracted->nsPerArgument)
                {
                    fastestSubtracted = &measurement;
                }
            }
        }

        if (fastestLaunched != nullptr &&
            CheckRow(*fastestLaunched, &report.control))
        {
            // The floor, taken once for the route that has a launch inside it.
            ProbeTimeRequest floorRequest = base;
            floorRequest.what = static_cast<int>(ProbeWhat::kFloor);
            floorRequest.reps = clamped.controlRepetitionsHigh;

            double floorMs = 0.0;

            if (report.control.entry.empty())
            {
                // RunRepetitionControl already said why in the note.
            } else if (!TimeRegion(floorRequest, floorMs))
            {
                report.control.note += "; the device-side floor could not be timed on this run, so "
                                       "how much of that figure is launch rather than arithmetic is "
                                       "not established here";
            } else
            {
                report.control.nsPerLaunchFloor =
                    floorMs * 1.0e6 / static_cast<double>(clamped.controlRepetitionsHigh);

                const double perCall = clamped.count * fastestLaunched->nsPerArgument;
                const double share =
                    perCall > 0.0 ? report.control.nsPerLaunchFloor / perCall : 0.0;

                // What the floor is, stated as what it measured rather than as
                // what it was meant to isolate. It is the per-launch cost of a
                // kernel that does no arithmetic, so it contains the device's own
                // launch and the host's submission of it together; on a platform
                // whose submission is expensive that is the larger part.
                //
                // When it is at or above a row's own per-call cost, the row is
                // the floor rather than the arithmetic, and a reader who took it
                // for an evaluation cost would be reading the launcher.
                const std::string floorClause =
                    share >= 1.0
                        ? Text("the floor - a kernel launched the same way that does no Boys "
                               "arithmetic - costs %.3f us per launch, which is %.0f%% of that "
                               "row's own cost per call. THAT ROW IS THE FLOOR: at this argument "
                               "count a launched entry is not being measured, because %d arguments "
                               "of its arithmetic cost less than one launch does. Raising the "
                               "argument count until the floor is a small fraction of the figure "
                               "is what this needs",
                               report.control.nsPerLaunchFloor / 1000.0,
                               100.0 * share,
                               static_cast<int>(clamped.count))
                        : Text("the floor - a kernel launched the same way that does no Boys "
                               "arithmetic - costs %.3f us per launch, %.2f%% of that row's own "
                               "cost per call, and is the part of every launched figure that is "
                               "launch and submission together rather than arithmetic",
                               report.control.nsPerLaunchFloor / 1000.0,
                               100.0 * share);

                report.control.note += "; ";
                report.control.note += floorClause;
                report.control.note += ".";
            }
        } else
        {
            report.control.note = "no launched entry produced a figure, so there was nothing to put "
                                  "the launched route's repetition control through";
        }

        if (fastestSubtracted != nullptr && CheckRow(*fastestSubtracted, &report.deviceCallControl))
        {
            if (!report.deviceCallControl.entry.empty())
            {
                // Neither bracket of a subtraction holds a launch of this
                // library, so there is no floor to take for this route: what
                // stands in for one is the baseline, reported above, and it is
                // already subtracted out of the figure being checked.
                report.deviceCallControl.note += "; nothing launched by this library is inside "
                                                 "either half of this row, so the floor that "
                                                 "applies to the launched route does not apply here "
                                                 "- the row's own baseline is what it was "
                                                 "subtracted against, and it is reported above.";
            }
        } else
        {
            report.deviceCallControl.note =
                "no in-kernel row of this run resolved a cost above its own baseline, so there was "
                "nothing to put the subtraction route's repetition control through";
        }
    }

    // --- The conclusions, to a fixed point -----------------------------------
    //
    // Every row the report names as a class's fastest or as its recommendation is
    // put through the repetition control before the report ships, and a row that
    // does not agree is set aside and the class falls to the next. Looped,
    // because setting a row aside can name a new leader, which then has to be
    // checked in its turn. Bounded by the number of rows: each round checks at
    // least one row not checked before, and the loop leaves when there is none.
    for (;;)
    {
        ConcludeClasses();

        DeviceProbeMeasurement* unchecked = nullptr;

        for (const DeviceProbeClass& clause : report.classes)
        {
            for (DeviceProbeMeasurement& measurement : report.measurements)
            {
                const bool named = !clause.recommended.empty()
                                       ? measurement.name == clause.recommended
                                       : measurement.name == clause.fastestOverall;

                if (named && !measurement.repetitionChecked)
                {
                    unchecked = &measurement;
                }
            }
        }

        if (unchecked == nullptr)
        {
            break;
        }

        if (!CheckRow(*unchecked, nullptr))
        {
            // No entry of this build is that row, which cannot happen for a row
            // that came out of the measurements; stop rather than loop.
            break;
        }
    }

    report.status = DeviceProbeStatus::kSuccess;
    return report;
}

// ---------------------------------------------------------------------------
// The text.
// ---------------------------------------------------------------------------

std::string FormatDeviceOptionProbe(const DeviceProbeReport& report) {
    std::string text;

    text += "boys device option probe - this result is about this device, this build, this process "
            "and the workload below.\n";

    if (report.status != DeviceProbeStatus::kSuccess)
    {
        if (!report.device.name.empty())
        {
            text += Text("  device: %s (ordinal %d, compute capability %d.%d)\n",
                         report.device.name.c_str(),
                         report.device.ordinal,
                         report.device.computeMajor,
                         report.device.computeMinor);

            if (report.device.singleToDoublePrecisionPerfRatio > 0)
            {
                text += Text("          single:double throughput ratio %d:1\n",
                             report.device.singleToDoublePrecisionPerfRatio);
            }
        }

        text += Text("\n  NOT MEASURED: %s\n", report.failure.c_str());
        text += Text("  status: %s\n", StatusName(report.status));

        for (const std::string& name : report.notAnEntry)
        {
            text += Text("  no such entry: %s\n", name.c_str());
        }

        for (const std::string& name : report.unoffered)
        {
            text += Text("  not carried by this build: %s\n", name.c_str());
        }

        return text;
    }

    text += Text("  device: %s\n", report.device.name.c_str());
    text += Text("    ordinal %d, compute capability %d.%d, %d streaming multiprocessors\n",
                 report.device.ordinal,
                 report.device.computeMajor,
                 report.device.computeMinor,
                 report.device.multiProcessorCount);
    text += Text("    memory %s, graphics clock %.2f GHz, memory clock %.2f GHz\n",
                 FormatBytes(report.device.totalMemoryBytes).c_str(),
                 static_cast<double>(report.device.clockKHz) / 1.0e6,
                 static_cast<double>(report.device.memoryClockKHz) / 1.0e6);
    text += Text("    single:double throughput ratio %d:1\n",
                 report.device.singleToDoublePrecisionPerfRatio);
    text += Text("    driver %s, runtime %s, built with toolkit %s, for architecture(s) %s\n",
                 FormatCudaVersion(report.device.driverVersion).c_str(),
                 FormatCudaVersion(report.device.runtimeVersion).c_str(),
                 report.device.toolkitVersion.c_str(),
                 report.device.architectures.c_str());

    text += "\n  workload:\n";
    text += Text("    %zu arguments, log-uniform over [%.3g, %.3g], each carrying its own highest\n",
                 report.workloadCount,
                 report.options.xLo,
                 report.options.xHi);
    text += Text("    order drawn as the sum of two shell angular momenta 0..%d, capped at %d,\n",
                 kShellMax,
                 report.options.nmax);
    text += "    sorted by argument so the uniform-order entries' stated precondition holds.\n";
    text += Text("    the fp32 and fp16 device-callable entries take the argument rounded to "
                 "their own\n    precision, which is what a caller's kernel holds in a register; "
                 "the batch entries\n    of those lanes take the double argument and round it "
                 "themselves.\n");

    text += "\n  protocol:\n";
    text += Text("    %d passes, %d rounds per pass, %d launches per timed region. Every entry "
                 "is timed by\n    a CUDA event pair around those launches, into buffers "
                 "uploaded once before the first\n    clock; the per-argument figure is the "
                 "device time divided by the launches and the\n    arguments. A pass whose "
                 "canary runs disagree by more than %.1f%% is discarded.\n",
                 report.options.passes,
                 report.options.rounds,
                 report.options.repetitions,
                 report.options.canarySpreadThreshold);
    text += Text("    in-kernel rows are a subtraction: the caller's kernel with the entry in it, "
                 "less the\n    same kernel with the call removed and the traffic kept, the two "
                 "timed adjacent in\n    the same round.\n");

    text += "\n  canary:\n";
    text += Text("    a fixed-work integer kernel, launched between the rounds, holding no "
                 "floating-point\n    state, so the arithmetic this probe ranks cannot change "
                 "the instrument's own cost.\n");
    text += Text("    its fastest run seen here is %.4f ms; the widest spread across an admitted "
                 "pass is\n    %.2f%%, which is this run's reading of the card's own timing "
                 "noise.\n",
                 report.canaryFloorMs,
                 report.canarySpread);

    text += "\npass  wall_s  canary_ms  canary%  verdict\n";

    if (report.passes.empty())
    {
        text += "     (no pass was run)\n";
    }

    for (std::size_t i = 0; i < report.passes.size(); ++i)
    {
        const DeviceProbePass& pass = report.passes[i];
        text += Text("%4zu  %6.3f  %9.4f  %7.2f  %s\n",
                     i + 1,
                     pass.seconds,
                     pass.canaryMedianMs,
                     pass.canarySpread,
                     pass.disturbed ? "DISTURBED (excluded)" : "admitted");
    }

    text += Text("\n  %d of %d passes admitted, %d discarded for the canary's own disagreement.\n",
                 report.cleanPasses,
                 report.options.passes,
                 report.disturbedPasses);

    text += "\nentries - ns/argument is device time per argument, transfer and host submission "
            "excluded\n";
    text += "  entry                     precision  shape        route      ns/arg   minus    "
            "spread  clean  bound      method\n";

    for (const DeviceProbeMeasurement& measurement : report.measurements)
    {
        const std::string baseline =
            measurement.nsPerArgumentBaseline > 0.0
                ? Text("%.3f", measurement.nsPerArgumentBaseline)
                : std::string("-");

        if (!measurement.measured)
        {
            text += Text("  %-24s %-10s %-11s %-10s %8s  %-7s  %6s  %2d/%d  %8.2g  %s\n",
                         measurement.name.c_str(),
                         measurement.precision.c_str(),
                         measurement.shape.c_str(),
                         measurement.route.c_str(),
                         "-",
                         "-",
                         "-",
                         measurement.cleanPasses,
                         report.options.passes,
                         measurement.documentedBound,
                         measurement.launchedByLibrary ? "library kernel launched"
                                                       : "caller kernel, subtracted");
            continue;
        }

        text += Text("  %-24s %-10s %-11s %-10s %8.3f  %-7s  %6.2fx  %2d/%d  %8.2g  %s\n",
                     measurement.name.c_str(),
                     measurement.precision.c_str(),
                     measurement.shape.c_str(),
                     measurement.route.c_str(),
                     measurement.nsPerArgument,
                     baseline.c_str(),
                     measurement.spread,
                     measurement.cleanPasses,
                     report.options.passes,
                     measurement.documentedBound,
                     measurement.launchedByLibrary ? "library kernel launched"
                                                   : "caller kernel, subtracted");
    }

    text += "\n  the minus column is the second half of a subtraction: the same caller-shaped "
            "kernel with\n  the call removed and the traffic kept, in the round the figure came "
            "from. A launched row\n  has no minus column because nothing was subtracted from it. "
            "The ns/arg column is the\n  difference between the region that held the call and "
            "that figure.\n";

    if (!report.unoffered.empty())
    {
        text += "\nentries this build does not carry, because its fp16 seam is closed:\n";

        for (const std::string& name : report.unoffered)
        {
            text += Text("    %s\n", name.c_str());
        }
    }

    if (!report.notAnEntry.empty())
    {
        text += "\nnames asked for that are no entry of this library, so nothing was measured "
                "for them:\n";

        for (const std::string& name : report.notAnEntry)
        {
            text += Text("    %s\n", name.c_str());
        }
    }

    text += "\n  the bound column is what each entry's lane documents at full accuracy, read "
            "from the\n  documentation and not measured here. What a lane delivers on this card "
            "is the accuracy\n  gate's business; a faster precision is not a faster option at "
            "the same accuracy, and\n  nothing below orders two precisions as though it were.\n";

    // --- The controls --------------------------------------------------------
    //
    // One per route, since the two routes are two methods and a check of one
    // says nothing about the other. The floor is the launched route's own: a
    // subtraction has no launch of this library's inside either half.
    const auto AppendControl = [&text](const char* heading,
                                       const DeviceProbeRepetitionControl& control,
                                       bool withFloor) {
        text += Text("\n%s\n", heading);

        if (control.note.empty())
        {
            text += "  not run\n";
            return;
        }

        if (control.entry.empty())
        {
            text += Text("  %s\n", control.note.c_str());
            return;
        }

        text += Text("  entry: %s (%s route)\n", control.entry.c_str(), control.route.c_str());
        text += Text("  %d launches: %.3f ns/argument\n",
                     control.repetitionsLow,
                     control.nsPerArgumentLow);
        text += Text("  %d launches: %.3f ns/argument\n",
                     control.repetitionsHigh,
                     control.nsPerArgumentHigh);

        if (control.nsPerArgumentBaselineLow > 0.0 || control.nsPerArgumentBaselineHigh > 0.0)
        {
            text += Text("  the half it was subtracted against: %.3f and %.3f ns/argument\n",
                         control.nsPerArgumentBaselineLow,
                         control.nsPerArgumentBaselineHigh);
        }

        text += Text("  disagreement %.2f%%: %s the %.2f%% this row can be placed at\n",
                     100.0 * control.difference,
                     control.agrees ? "AGREES within" : "DOES NOT AGREE within",
                     100.0 * control.judgedAgainst);

        if (withFloor)
        {
            text += Text("  floor, a kernel launched the same way with no Boys arithmetic in it: "
                         "%.3f us per launch\n",
                         control.nsPerLaunchFloor / 1000.0);
        }

        text += Text("  %s\n", control.note.c_str());
    };

    AppendControl("launch control", report.control, true);
    AppendControl("subtraction control", report.deviceCallControl, false);

    // --- The rankings --------------------------------------------------------
    for (const DeviceProbeClass& clause : report.classes)
    {
        text += Text("\nranking - %s\n", clause.question.c_str());
        text += Text("  every entry here was asked for %s\n", clause.asked.c_str());

        if (clause.fastestOverall.empty())
        {
            text += "  no entry of this class produced a figure\n";
            text += Text("  reason: %s\n", clause.reason.c_str());
            text += Text("  confidence: %s\n", clause.confidence.c_str());
            continue;
        }

        text += Text("  fastest measured: %s\n", clause.fastestOverall.c_str());

        if (clause.resolution > 0.0)
        {
            text += Text("  resolution: entries closer than %.2f%% cannot be ordered on this run. "
                         "The number\n              is measured, not assumed: it is the larger of "
                         "the canary's widest\n              admitted spread and the leader's own "
                         "spread across its clean passes.\n",
                         100.0 * clause.resolution);
        } else
        {
            text += "  resolution: not measurable on this run, so this class is not ordered\n";
        }

        bool looserThanAnotherLane = false;
        double recommendedBound = 0.0;

        for (const DeviceProbeMeasurement& measurement : report.measurements)
        {
            if (measurement.name != clause.recommended)
            {
                continue;
            }

            recommendedBound = measurement.documentedBound;

            for (const DeviceProbeMeasurement& other : report.measurements)
            {
                if (other.question == clause.question && other.documentedBound > 0.0 &&
                    other.documentedBound < recommendedBound)
                {
                    looserThanAnotherLane = true;
                }
            }
        }

        text += Text("  verdict: %s\n",
                     clause.verdict != DeviceProbeVerdict::kRecommend
                         ? "CANNOT DETERMINE"
                         : (looserThanAnotherLane
                                ? "RECOMMEND BY COST ONLY - a lane in this class documents a "
                                  "tighter bound"
                                : "RECOMMEND"));

        if (!clause.recommended.empty())
        {
            text += Text("  recommended entry: %s\n", clause.recommended.c_str());

            // A class spans the lanes that answer the same question, so its
            // ranking is by cost and not at one accuracy. The bound each lane
            // documents is what a reader has to weigh against the figure, and it
            // is repeated here because this line is the one that gets quoted.
            if (recommendedBound > 0.0)
            {
                text += Text("    this is a cost ranking, not a ranking at one accuracy: that "
                             "entry's lane\n    documents a bound of %.1e, and the other lanes of "
                             "this class document their own.\n    The bound column above is the "
                             "figure to weigh against this one.\n",
                             recommendedBound);
            }
        }

        text += Text("  reason: %s\n", clause.reason.c_str());

        for (const std::string& entry : clause.inseparable)
        {
            text += Text("  not separable: %s\n", entry.c_str());
        }

        for (const std::string& entry : clause.notOrdered)
        {
            text += Text("  set aside: %s\n", entry.c_str());
        }

        text += Text("  confidence: %s\n", clause.confidence.c_str());
    }

    text += Text("\n%s\n", report.caveat.c_str());

    text += "\nnot measured by this probe, by design: the relaxed accuracy multipliers (the "
            "device lane\n  fixes the multiplier at the call site, so ranking them would need an "
            "instantiation per\n  rung, and the host probe ranks the rungs where they are "
            "reachable at run time); the\n  ordered-batch sort (the AllN entries are handed a "
            "batch that already satisfies their\n  stated precondition, so nothing here is a "
            "figure for sorting one); the arithmetic's\n  accuracy (the device accuracy gate "
            "measures that, against the committed reference).\n";

    return text;
}

} // namespace boys
