// Does every row of the device option table read the axes it states?
//
// What this checks, and the defect it was written for. A row of the device option space names an
// entry, and the entry is really called; what that does not prove is that the entry *reads* the
// axis the row's coordinates state. Two rows that differ in one axis and execute identical
// arithmetic are one option under two names and a claim with nothing behind it - and a count of
// the combinations a surface serves, taken by comparing names, cannot see it. This check takes it
// by measurement: for every axis a row carries more than one member of, it runs the entries at
// each member, over the argument range the tables cover and at the orders each shape carries, and
// requires the values to differ somewhere. A cell whose members all deliver the same bits is
// reported by row, by axis and by member, and the run fails.
//
// The axes and where they come from. The five the option table varies a row along are its own
// DeviceOptionInfo::axis - route, scheme, partition, packing and the region-B exponential - and
// the members are the library's enumerations, read from each row and never listed here. The sixth
// is the division form, which every entry's own signature carries: this check runs each row at the
// build's default form and at the plain reciprocal, requires those to differ, and reads the
// launched rows at the third member beside them, whose relation to the first the library states in
// its own words and this run reports as measured. Nothing here states a member set of its own: the
// rows come from BoysDeviceOptions(), and a row this file has no call for is printed and counted as
// an option nothing here runs rather than quietly counting as covered - which is a coverage gap
// this run states, not an axis finding, so it does not decide the exit code.
//
// What "differ" means, and the honest reading of it. The comparison is over the raw bits of every
// value a run wrote - one block per argument, over 64 arguments spanning [0, 60] and orders 0..32
// - and it asks that a cell's members not all be one value. Two members that coincide while a
// third differs is not a failure: it is printed with the rows named, because that is what a name
// that denotes another name's arithmetic looks like from here, and it is the reader's to judge
// against what the library documents. A documented coincidence is expected and reported the same
// way.
//
// Three readings of one cell, and which of them is a failure. Two members delivering one value is
// a failure when the store they wrote through could have shown a difference. The fp64 and fp32
// lanes store exactly what their engine computed, so one value there is one arithmetic. The half
// lane computes in that same float engine and stores eleven or eight mantissa bits of the result,
// so two arithmetics that differ by less than the format's resolution - 2^-11 for fp16, 2^-8 for
// bf16 - are one value in it, and requiring a bit-difference in a format coarser than the
// difference is asking the wrong question of the row. The row's claim is about which arithmetic
// runs, so that is what is measured: the same cell is asked at the float lane, which is the same
// bodies over the same tables with the engine's own twenty-four mantissa bits kept, and the
// members are known to differ only where they differ there. The witness has to be one store: a
// fp64 row is another engine, and reading one beside a fp32 row would let the width of the store
// stand in for the axis being read - which is exactly the mistake the deliberate-defect control
// catches, because under it the float lane's own members are one value too and every cell of the
// collapsed axis has to fail. Such a cell is printed by row and counted as **not observable**,
// with the resolution and the widest difference the float lane showed beside it, and it does not
// fail: the run has no evidence that the row is a claim with nothing behind it, which is the
// treatment the rows this test cannot run already get. A cell whose float-lane siblings deliver
// one value too is one arithmetic under two names whatever the store, and stays a failure. The
// split is not a threshold chosen to pass: the wider store's own difference is
// printed for every cell moved, so a reader checks the classification against the resolution
// rather than taking it on trust.
//
// The in-kernel rows. 168 of the table's 352 rows are entries of boys_cuda_device.hpp, which are
// __device__ functions no host translation unit can call. They are run by
// tests/boys_cuda_axis_sensitivity_device.cu, the device half of this test, over the same
// arguments and the same handle (BoysCuda::DeviceTables); the host declares its one entry point
// below. The two halves are one target, registered beside the other boys-cuda-* targets.
//
// The deliberate-defect control. `--collapse=<axis>` runs every row of a cell at the entry of the
// cell's own first member, which is what an entry that does not read that axis delivers: the
// members then produce identical values and the check must fail, naming the rows, the axis and the
// members. A test that cannot fail is worse than no test, so the failure is shown rather than
// asserted:
//
//     boys-cuda-axis-sensitivity --collapse=route         # exits 1, every cell of that axis
//     boys-cuda-axis-sensitivity                          # exits 1 while a row is a finding
//
// The second line's exit code is the library's answer and not this file's: it is 1 exactly while
// some cell delivers one value where the store could have shown a difference, which is what the
// run prints and what a reader is meant to act on. It is 0 once no such cell remains.
//
// A note on the plumbing. Every value is also gated on being a value of the Boys function at all -
// finite and in [0, 1] wherever it is not the zero of a slot the entry did not write - so a handle
// that was not filled, an argument array that was not uploaded or a kernel that wrote nothing is
// reported as a broken run rather than quietly becoming an axis difference or an axis coincidence.

#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp" // the region boundaries the arguments are placed at
#include "boys/boys_cuda.hpp"
#include "boys/boys_cuda_options.hpp"
#include "boys/f16.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <map>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

// The device half of this test (tests/boys_cuda_axis_sensitivity_device.cu): one entry of
// boys_cuda_device.hpp, run inside the caller's own kernel over these arguments.
extern "C" int BoysAxisRunDevice(int entry,
                                 int form,
                                 int shape,
                                 const boys::BoysDeviceTables* tables,
                                 int nmax,
                                 const int* order,
                                 const double* x,
                                 const void* x16,
                                 const void* xbf,
                                 const int* offset,
                                 void* out,
                                 std::size_t count);

namespace {

using boys::BoysCuda;
using boys::BoysDeviceOptions;
using boys::BoysStatus;
using boys::DeviceEntry;
using boys::DeviceOptionAxis;
using boys::DeviceOptionGroup;
using boys::DeviceOptionInfo;
using boys::DeviceOptionPrecision;
using boys::DivisionForm;
using boys::kDefaultDeviceDivisionForm;

/// Arguments per run. Every argument costs a thread in the kernels and a block in the output, and
/// 64 is enough for the sweep below to place a point in every region and inside every kind of
/// piece while staying far below the smallest batch the lane's figures are read at.
constexpr int kArguments = 64;

/// Values one argument's block of the output holds: the tallest ladder any shape writes. One
/// block per argument for every shape, so that the buffer and the comparison are one thing.
constexpr int kLadder = boys::kMaxBoysOrder + 1;

/// One argument set, shared by every row: the region boundaries the tables are cut at, and a
/// log-uniform sweep from far below the smallest piece to above the flat grid's top, so that some
/// argument sits in every region and the ladder of some argument crosses every boundary.
std::vector<double> Arguments() {
    std::vector<double> x;

    x.push_back(0.0);

    for (const double bound : {boys::detail::kX0, boys::detail::kX1, boys::detail::kFlatHi})
    {
        x.push_back(std::nextafter(bound, 0.0));
        x.push_back(bound);
        x.push_back(std::nextafter(bound, 1.0e300));
    }

    const std::size_t sweep = static_cast<std::size_t>(kArguments) - x.size();

    for (std::size_t k = 0; k < sweep; ++k)
    {
        const double t = static_cast<double>(k) / static_cast<double>(sweep - 1);
        x.push_back(1.0e-6 * std::pow(6.0e1 / 1.0e-6, t));
    }

    return x;
}

/// The raw bits of one returned value: what the comparison is made of, so that two runs differing
/// in one bit are two arithmetics rather than one rounded answer.
template <typename T> std::uint64_t ValueBits(T value) {
    if constexpr (std::is_same_v<T, double>)
    {
        return std::bit_cast<std::uint64_t>(value);
    } else if constexpr (std::is_same_v<T, float>)
    {
        return std::bit_cast<std::uint32_t>(value);
    } else if constexpr (requires { value.Bits(); })
    {
        // The library's own half type, which carries its pattern.
        return value.Bits();
    } else
    {
        // The C++23 extended type, where the toolchain ships it.
        return std::bit_cast<std::uint16_t>(value);
    }
}

/// Everything one run needs: the arguments on the host and on the card, the output buffer, and
/// the handle the in-kernel entries read.
struct Context {
    std::vector<int> order;
    std::vector<double> x;
    std::vector<boys::F16> x16;
    std::vector<boys::Bf16> xbf;
    std::vector<int> offset;

    int nmax = boys::kMaxBoysOrder;
    std::size_t count = kArguments;

    int* dOrder = nullptr;
    double* dX = nullptr;
    boys::F16* dX16 = nullptr;
    boys::Bf16* dXBf = nullptr;
    int* dOffset = nullptr;
    void* dOut = nullptr;
    std::size_t outBytes = 0;

    boys::BoysDeviceTables tables{};

    bool Upload();
    void Release() const;
};

bool Context::Upload() {
    order.resize(count);
    x = Arguments();
    x16.resize(count);
    xbf.resize(count);
    offset.resize(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        // Every order is exercised, the top of the range included.
        order[i] = static_cast<int>(i % static_cast<std::size_t>(kLadder));
        x16[i] = boys::F16(static_cast<float>(x[i]));
        xbf[i] = boys::Bf16(static_cast<float>(x[i]));
        offset[i] = static_cast<int>(i) * kLadder;
    }

    static_assert(sizeof(boys::F16) == 2 && sizeof(boys::Bf16) == 2,
                  "the half lane's two formats are one 16-bit pattern, which is what lets a row "
                  "of either be read back without the host deciding which it was");

    outBytes = count * static_cast<std::size_t>(kLadder) * sizeof(double);

    const std::size_t orderBytes = count * sizeof(int);
    const std::size_t xBytes = count * sizeof(double);
    const std::size_t halfBytes = count * sizeof(boys::F16);

    if (cudaMalloc(reinterpret_cast<void**>(&dOrder), orderBytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dX), xBytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dX16), halfBytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dXBf), halfBytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dOffset), orderBytes) != cudaSuccess ||
        cudaMalloc(&dOut, outBytes) != cudaSuccess)
    {
        return false;
    }

    return cudaMemcpy(dOrder, order.data(), orderBytes, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(dX, x.data(), xBytes, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(dX16, x16.data(), halfBytes, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(dXBf, xbf.data(), halfBytes, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(dOffset, offset.data(), orderBytes, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemset(dOut, 0, outBytes) == cudaSuccess;
}

void Context::Release() const {
    cudaFree(dOrder);
    cudaFree(dX);
    cudaFree(dX16);
    cudaFree(dXBf);
    cudaFree(dOffset);
    cudaFree(dOut);
}

/// What one entry delivered for one row at one form.
struct Run {
    bool ok = false; ///< the entry ran and wrote its values
    std::string why; ///< why not, when it did not
    std::vector<std::uint64_t> bits; ///< every value's bits, in that shape's own layout
    bool sane = true; ///< every nonzero value is a value of the Boys function
    std::string insane; ///< the first slot that was not, with what it held
};

/// Reads the output buffer back, turns it into the bits the comparison is made of, and gates the
/// values on being values of the Boys function.
///
/// \param ctx     the run's arguments and output buffer
/// \param run     receives the bits
/// \param numeric whether the values may be read as numbers. The half lane's two formats are one
///                16-bit pattern and a row of either is read back here without the host deciding
///                which it was, so those values are compared as patterns and the range gate is
///                the double and the float lanes' own - a broken handle is caught by them.
template <typename T> bool Capture(const Context& ctx, Run& run, bool numeric) {
    std::vector<T> host(ctx.count * static_cast<std::size_t>(kLadder));

    if (cudaMemcpy(host.data(), ctx.dOut, host.size() * sizeof(T), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
    {
        run.ok = false;
        run.why = "the output buffer would not copy back";
        return false;
    }

    run.bits.clear();
    run.bits.reserve(host.size());

    bool wrote = false;

    for (std::size_t slot = 0; slot < host.size(); ++slot)
    {
        const T value = host[slot];
        const std::uint64_t bits = ValueBits(value);
        run.bits.push_back(bits);
        wrote = wrote || bits != 0;

        const double wide = numeric ? static_cast<double>(value) : 0.0;

        // F_n(x) is in (0, 1] for every argument these tables serve, and the tolerance is the
        // loosest stored lane's own rounding rather than the value's: what this gate is for is a
        // run that produced no value at all, not a lane's last digit.
        if (numeric && wide != 0.0 &&
            (!std::isfinite(wide) || wide < -1.0e-4 || wide > 1.0 + 1.0e-4))
        {
            if (run.sane)
            {
                char line[160];
                std::snprintf(line, sizeof(line), "slot %zu holds %.9g", slot, wide);
                run.insane = line;
            }

            run.sane = false;
        }
    }

    // F_n(x) > 0 for every argument these tables serve, so a run that wrote nothing at all is a
    // kernel with no arm for the row rather than an entry that answered with zeros.
    if (!wrote)
    {
        run.ok = false;
        run.why = "the entry wrote no value: nothing this test calls reaches it";
        return false;
    }

    return true;
}

/// The value type a launched entry writes and the argument type it reads, read off that entry's
/// own signature: the half lane takes its arguments as F16 or Bf16 and the other lanes as double.
/// The two half formats are two argument types and therefore two entries here, one per row of
/// each class the option table books - the formats share the bodies, never the spelling.
template <typename Fn> struct Signature;

template <typename T>
struct Signature<BoysStatus (*)(const int*, const double*, T*, std::size_t, void*, DivisionForm)> {
    using Value = T;
    using Arg = double;
};

template <typename T>
struct Signature<BoysStatus (*)(
    const int*, const boys::F16*, T*, std::size_t, void*, DivisionForm)> {
    using Value = T;
    using Arg = boys::F16;
};

template <typename T>
struct Signature<BoysStatus (*)(
    const int*, const boys::Bf16*, T*, std::size_t, void*, DivisionForm)> {
    using Value = T;
    using Arg = boys::Bf16;
};

template <typename T>
struct Signature<BoysStatus (*)(int, const double*, T*, std::size_t, void*, DivisionForm)> {
    using Value = T;
    using Arg = double;
};

template <typename T>
struct Signature<BoysStatus (*)(int, const boys::F16*, T*, std::size_t, void*, DivisionForm)> {
    using Value = T;
    using Arg = boys::F16;
};

template <typename T>
struct Signature<BoysStatus (*)(int, const boys::Bf16*, T*, std::size_t, void*, DivisionForm)> {
    using Value = T;
    using Arg = boys::Bf16;
};

template <typename T>
struct Signature<BoysStatus (*)(
    const int*, const double*, const int*, T*, std::size_t, void*, DivisionForm)> {
    using Value = T;
    using Arg = double;
};

template <typename T>
struct Signature<BoysStatus (*)(
    const int*, const boys::F16*, const int*, T*, std::size_t, void*, DivisionForm)> {
    using Value = T;
    using Arg = boys::F16;
};

template <typename T>
struct Signature<BoysStatus (*)(
    const int*, const boys::Bf16*, const int*, T*, std::size_t, void*, DivisionForm)> {
    using Value = T;
    using Arg = boys::Bf16;
};

template <typename Arg> const Arg* ArgumentArray(const Context& ctx) {
    if constexpr (std::is_same_v<Arg, double>)
    {
        return ctx.dX;
    } else if constexpr (std::is_same_v<Arg, boys::Bf16>)
    {
        return ctx.dXBf;
    } else
    {
        return ctx.dX16;
    }
}

/// The tail every launched run shares: the entry's status, the synchronisation, the read-back.
template <typename T>
bool Finish(BoysStatus status, const Context& ctx, Run& run, const char* what) {
    if (status != BoysStatus::kSuccess)
    {
        run.ok = false;
        run.why = std::string(what) + " refused the call (status ";
        run.why += std::to_string(static_cast<int>(status));
        run.why += ")";
        return false;
    }

    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess)
    {
        run.ok = false;
        run.why = std::string(what) + " failed on the device: ";
        run.why += cudaGetErrorString(error);
        return false;
    }

    if (!Capture<T>(ctx, run, true))
    {
        return false;
    }

    run.ok = true;
    return true;
}

/// The four launched shapes, each taking the entry's own signature. The arms below name the
/// entry of a row; nothing here names an axis.
template <typename Fn> bool RunSingle(Fn fn, Context& ctx, DivisionForm form, Run& run) {
    using Value = typename Signature<Fn>::Value;
    using Arg = typename Signature<Fn>::Arg;

    cudaMemset(ctx.dOut, 0, ctx.outBytes);
    const BoysStatus status = fn(ctx.dOrder,
                                 ArgumentArray<Arg>(ctx),
                                 static_cast<Value*>(ctx.dOut),
                                 ctx.count,
                                 nullptr,
                                 form);
    return Finish<Value>(status, ctx, run, "the single-order entry");
}

template <typename Fn> bool RunLadder(Fn fn, Context& ctx, DivisionForm form, Run& run) {
    using Value = typename Signature<Fn>::Value;
    using Arg = typename Signature<Fn>::Arg;

    cudaMemset(ctx.dOut, 0, ctx.outBytes);
    const BoysStatus status = fn(ctx.dOrder,
                                 ArgumentArray<Arg>(ctx),
                                 static_cast<Value*>(ctx.dOut),
                                 ctx.count,
                                 nullptr,
                                 form);
    return Finish<Value>(status, ctx, run, "the all-orders entry");
}

template <typename Fn> bool RunAllN(Fn fn, Context& ctx, DivisionForm form, Run& run) {
    using Value = typename Signature<Fn>::Value;
    using Arg = typename Signature<Fn>::Arg;

    cudaMemset(ctx.dOut, 0, ctx.outBytes);
    const BoysStatus status = fn(
        ctx.nmax, ArgumentArray<Arg>(ctx), static_cast<Value*>(ctx.dOut), ctx.count, nullptr, form);
    return Finish<Value>(status, ctx, run, "the all-n entry");
}

template <typename Fn> bool RunEachOrder(Fn fn, Context& ctx, DivisionForm form, Run& run) {
    using Value = typename Signature<Fn>::Value;
    using Arg = typename Signature<Fn>::Arg;

    cudaMemset(ctx.dOut, 0, ctx.outBytes);
    const BoysStatus status = fn(ctx.dOrder,
                                 ArgumentArray<Arg>(ctx),
                                 ctx.dOffset,
                                 static_cast<Value*>(ctx.dOut),
                                 ctx.count,
                                 nullptr,
                                 form);
    return Finish<Value>(status, ctx, run, "the each-order entry");
}

/// One row of the in-kernel half, run by the device translation unit inside the caller's own
/// kernel, over the same arguments and the same handle the launched rows read.
bool RunInKernel(const DeviceOptionInfo& row, Context& ctx, DivisionForm form, Run& run) {
    cudaMemset(ctx.dOut, 0, ctx.outBytes);

    const int code = BoysAxisRunDevice(static_cast<int>(row.entry),
                                       static_cast<int>(form),
                                       static_cast<int>(row.shape),
                                       &ctx.tables,
                                       ctx.nmax,
                                       ctx.dOrder,
                                       ctx.dX,
                                       ctx.dX16,
                                       ctx.dXBf,
                                       nullptr,
                                       ctx.dOut,
                                       ctx.count);

    if (code != 0)
    {
        run.ok = false;
        run.why = "the in-kernel entry would not run (code " + std::to_string(code) + ")";
        return false;
    }

    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess)
    {
        run.ok = false;
        run.why =
            std::string("the in-kernel entry failed on the device: ") + cudaGetErrorString(error);
        return false;
    }

    switch (row.precision)
    {
    case DeviceOptionPrecision::kFp64:
        if (!Capture<double>(ctx, run, true))
        {
            return false;
        }
        break;
    case DeviceOptionPrecision::kFp32:
        if (!Capture<float>(ctx, run, true))
        {
            return false;
        }
        break;
    default:
        // Either half format: the patterns are what is compared, and 16 bits is 16 bits.
        if (!Capture<boys::F16>(ctx, run, false))
        {
            return false;
        }
        break;
    }

    run.ok = true;
    return true;
}

/// One row's entry, called at one division form: the row's own group decides which half of the
/// surface reaches it, and its own enumerator decides which entry of that half.
bool RunRow(const DeviceOptionInfo& row, Context& ctx, DivisionForm form, Run& run) {
    if (!row.built)
    {
        run.ok = false;
        run.why = row.refusedBecause != nullptr ? row.refusedBecause : "this build refuses the row";
        return false;
    }

    if (row.group == DeviceOptionGroup::kLaunched)
    {
        switch (row.entry)
        {
        case boys::DeviceEntry::kSingleF64:
            return RunSingle(&boys::BoysCuda::SingleF64, ctx, form, run);
        case boys::DeviceEntry::kSingleF32:
            return RunSingle(
                &boys::BoysCuda::SingleF32<boys::RegionBExp::kAccurate>, ctx, form, run);
        case boys::DeviceEntry::kSingleF32Fast:
            return RunSingle(&boys::BoysCuda::SingleF32<boys::RegionBExp::kFast>, ctx, form, run);
        case boys::DeviceEntry::kSingleF16:
            return RunSingle(&boys::BoysCuda::SingleF16, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64:
            return RunLadder(&boys::BoysCuda::AllOrdersF64, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32:
            return RunLadder(&boys::BoysCuda::AllOrdersF32, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16:
            return RunLadder(&boys::BoysCuda::AllOrdersF16, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64Narrow:
            return RunLadder(&boys::BoysCuda::AllOrdersF64Narrow, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64Orders:
            return RunLadder(&boys::BoysCuda::AllOrdersF64Orders, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowOrders:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowOrders, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64Mono:
            return RunLadder(&boys::BoysCuda::AllOrdersF64Mono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64OrdersMono:
            return RunLadder(&boys::BoysCuda::AllOrdersF64OrdersMono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowMono:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowMono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowOrdersMono:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowOrdersMono, ctx, form, run);
        // The double lane's rational route has one member per partition, not two: the route's
        // pair is stored in one form and the two scheme names reach one kernel
        // (boys_cuda_options.hpp, DeviceEntryAxesOf, and the launcher comment in
        // src/boys_cuda.cu), so each -horner row of this lane runs the member its plain twin
        // names. The row is still run as the row it is: only the entry it reaches is shared.
        case boys::DeviceEntry::kAllOrdersF64Rat:
        case boys::DeviceEntry::kAllOrdersF64RatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF64Rat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64OrdersRat:
        case boys::DeviceEntry::kAllOrdersF64OrdersRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF64OrdersRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowRat:
        case boys::DeviceEntry::kAllOrdersF64NarrowRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowOrdersRat:
        case boys::DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowOrdersRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64Uniform:
            return RunLadder(&boys::BoysCuda::AllOrdersF64Uniform, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64UniformHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF64UniformHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64OrdersUniform:
            return RunLadder(&boys::BoysCuda::AllOrdersF64OrdersUniform, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64OrdersUniformHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF64OrdersUniformHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64UniformRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF64UniformRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64UniformRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF64UniformRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64OrdersUniformRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF64OrdersUniformRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64OrdersUniformRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF64OrdersUniformRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32Narrow:
            return RunLadder(&boys::BoysCuda::AllOrdersF32Narrow, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowMono:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowMono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32Uniform:
            return RunLadder(&boys::BoysCuda::AllOrdersF32Uniform, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32UniformHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF32UniformHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32Rat:
            return RunLadder(&boys::BoysCuda::AllOrdersF32Rat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32RatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF32RatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32UniformRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF32UniformRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32UniformRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF32UniformRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32Orders:
            return RunLadder(&boys::BoysCuda::AllOrdersF32Orders, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowOrders:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowOrders, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowOrdersMono:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowOrdersMono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowOrdersRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowOrdersRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowOrdersRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersUniform:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersUniform, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersUniformHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersUniformHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersUniformRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersUniformRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersUniformRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersUniformRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16Narrow:
            return RunLadder(&boys::BoysCuda::AllOrdersF16Narrow, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowMono:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowMono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16Uniform:
            return RunLadder(&boys::BoysCuda::AllOrdersF16Uniform, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16UniformHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF16UniformHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16Rat:
            return RunLadder(&boys::BoysCuda::AllOrdersF16Rat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16RatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF16RatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16UniformRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF16UniformRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16UniformRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF16UniformRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16Orders:
            return RunLadder(&boys::BoysCuda::AllOrdersF16Orders, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowOrders:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowOrders, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowOrdersMono:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowOrdersMono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowOrdersRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowOrdersRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowOrdersRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowOrdersRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersUniform:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersUniform, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersUniformHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersUniformHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersUniformRat:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersUniformRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersUniformRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersUniformRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllNF64:
            return RunAllN(&boys::BoysCuda::AllNF64, ctx, form, run);
        case boys::DeviceEntry::kAllNF32:
            return RunAllN(&boys::BoysCuda::AllNF32, ctx, form, run);
        case boys::DeviceEntry::kAllNF16:
            return RunAllN(&boys::BoysCuda::AllNF16, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32Mono:
            return RunLadder(&boys::BoysCuda::AllOrdersF32Mono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersMono:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersMono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16Mono:
            return RunLadder(&boys::BoysCuda::AllOrdersF16Mono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersMono:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersMono, ctx, form, run);
        case boys::DeviceEntry::kSingleF16Fast:
            return RunSingle(&boys::BoysCuda::SingleF16Fast, ctx, form, run);
        case boys::DeviceEntry::kEachOrderF64:
            return RunEachOrder(
                &boys::BoysCuda::EachOrderF64<boys::RegionBExp::kAccurate>, ctx, form, run);
        case boys::DeviceEntry::kEachOrderF64Fast:
            return RunEachOrder(
                &boys::BoysCuda::EachOrderF64<boys::RegionBExp::kFast>, ctx, form, run);
        case boys::DeviceEntry::kEachOrderF32:
            return RunEachOrder(
                &boys::BoysCuda::EachOrderF32<boys::RegionBExp::kAccurate>, ctx, form, run);
        case boys::DeviceEntry::kEachOrderF32Fast:
            return RunEachOrder(
                &boys::BoysCuda::EachOrderF32<boys::RegionBExp::kFast>, ctx, form, run);
        case boys::DeviceEntry::kEachOrderF16:
            return RunEachOrder(
                &boys::BoysCuda::EachOrderF16<boys::RegionBExp::kAccurate>, ctx, form, run);
        case boys::DeviceEntry::kEachOrderF16Fast:
            return RunEachOrder(
                &boys::BoysCuda::EachOrderF16<boys::RegionBExp::kFast>, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64Fast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64Fast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64OrdersFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64OrdersFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowOrdersFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowOrdersFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64MonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64MonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64OrdersMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64OrdersMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowOrdersMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowOrdersMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64RatFast:
        case boys::DeviceEntry::kAllOrdersF64RatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64RatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64OrdersRatFast:
        case boys::DeviceEntry::kAllOrdersF64OrdersRatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64OrdersRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowRatFast:
        case boys::DeviceEntry::kAllOrdersF64NarrowRatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF64NarrowOrdersRatFast:
        case boys::DeviceEntry::kAllOrdersF64NarrowOrdersRatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF64NarrowOrdersRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32Fast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32Fast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowOrdersFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowOrdersFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32MonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32MonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowOrdersMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowOrdersMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32RatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32RatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32RatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32RatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersRatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32OrdersRatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32OrdersRatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowRatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowRatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowRatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowOrdersRatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF32NarrowOrdersRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF32NarrowOrdersRatHornerFast:
            return RunLadder(
                &boys::BoysCuda::AllOrdersF32NarrowOrdersRatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16Fast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16Fast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowOrdersFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowOrdersFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16MonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16MonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowOrdersMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowOrdersMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16RatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16RatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16RatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16RatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersRatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16OrdersRatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16OrdersRatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowRatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowRatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowRatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowOrdersRatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersF16NarrowOrdersRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersF16NarrowOrdersRatHornerFast:
            return RunLadder(
                &boys::BoysCuda::AllOrdersF16NarrowOrdersRatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kSingleF64Fast:
            return RunSingle(&boys::BoysCuda::SingleF64Fast, ctx, form, run);
        case boys::DeviceEntry::kAllNF64Fast:
            return RunAllN(&boys::BoysCuda::AllNF64Fast, ctx, form, run);
        case boys::DeviceEntry::kAllNF32Fast:
            return RunAllN(&boys::BoysCuda::AllNF32Fast, ctx, form, run);
        case boys::DeviceEntry::kAllNF16Fast:
            return RunAllN(&boys::BoysCuda::AllNF16Fast, ctx, form, run);
        case boys::DeviceEntry::kSingleBf16:
            return RunSingle(&boys::BoysCuda::SingleBf16, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16Narrow:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16Narrow, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowMono:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowMono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16Uniform:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16Uniform, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16UniformHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16UniformHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16Rat:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16Rat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16RatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16RatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowRat:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16UniformRat:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16UniformRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16UniformRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16UniformRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16Orders:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16Orders, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowOrders:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowOrders, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowOrdersMono:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowOrdersMono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersRat:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowOrdersRat:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowOrdersRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowOrdersRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowOrdersRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersUniform:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersUniform, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersUniformHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersUniformHorner, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersUniformRat:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersUniformRat, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersUniformRatHorner:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersUniformRatHorner, ctx, form, run);
        case boys::DeviceEntry::kAllNBf16:
            return RunAllN(&boys::BoysCuda::AllNBf16, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16Mono:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16Mono, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersMono:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersMono, ctx, form, run);
        case boys::DeviceEntry::kSingleBf16Fast:
            return RunSingle(&boys::BoysCuda::SingleBf16Fast, ctx, form, run);
        case boys::DeviceEntry::kEachOrderBf16:
            return RunEachOrder(
                &boys::BoysCuda::EachOrderBf16<boys::RegionBExp::kAccurate>, ctx, form, run);
        case boys::DeviceEntry::kEachOrderBf16Fast:
            return RunEachOrder(
                &boys::BoysCuda::EachOrderBf16<boys::RegionBExp::kFast>, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16Fast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16Fast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowOrdersFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowOrdersFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16MonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16MonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowOrdersMonoFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowOrdersMonoFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16RatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16RatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16RatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16RatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersRatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16OrdersRatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16OrdersRatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowRatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowRatHornerFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowRatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowOrdersRatFast:
            return RunLadder(&boys::BoysCuda::AllOrdersBf16NarrowOrdersRatFast, ctx, form, run);
        case boys::DeviceEntry::kAllOrdersBf16NarrowOrdersRatHornerFast:
            return RunLadder(
                &boys::BoysCuda::AllOrdersBf16NarrowOrdersRatHornerFast, ctx, form, run);
        case boys::DeviceEntry::kAllNBf16Fast:
            return RunAllN(&boys::BoysCuda::AllNBf16Fast, ctx, form, run);
        default:
            run.ok = false;
            run.why = "no arm in this test's dispatch";
            return false;
        }
    }

    return RunInKernel(row, ctx, form, run);
}

/// The axis a row's own cell is compared along, and the name a report prints it by.
struct Axis {
    DeviceOptionAxis axis;
    const char* name;
};

const Axis kAxes[] = {
    {DeviceOptionAxis::kRoute, "route"},
    {DeviceOptionAxis::kScheme, "scheme"},
    {DeviceOptionAxis::kPartition, "partition"},
    {DeviceOptionAxis::kPacking, "packing"},
    {DeviceOptionAxis::kRegionBExp, "region-B exponential"},
};

/// The member a row states on one axis, as the library's own enumeration numbers it.
int MemberOf(const DeviceOptionInfo& row, DeviceOptionAxis axis) {
    switch (axis)
    {
    case DeviceOptionAxis::kRoute:
        return static_cast<int>(row.route);
    case DeviceOptionAxis::kScheme:
        return static_cast<int>(row.scheme);
    case DeviceOptionAxis::kPartition:
        return static_cast<int>(row.partition);
    case DeviceOptionAxis::kPacking:
        return static_cast<int>(row.packing);
    case DeviceOptionAxis::kRegionBExp:
        return static_cast<int>(row.regionBExp);
    default:
        return -1;
    }
}

/// The member's own name where the library gives one - the scheme, the partition and the region-B
/// exponential have one - and nothing where it does not: the route and the packing axis are then
/// named by the rows that state them, which is where a reader finds them anyway.
const char* MemberText(const DeviceOptionInfo& row, DeviceOptionAxis axis) {
    switch (axis)
    {
    case DeviceOptionAxis::kScheme:
        return boys::EvalSchemeName(row.scheme);
    case DeviceOptionAxis::kPartition:
        return boys::DevicePartitionName(row.partition);
    case DeviceOptionAxis::kRegionBExp:
        return boys::RegionBExpName(row.regionBExp);
    default:
        return nullptr;
    }
}

/// The cell a row belongs to along one axis: every coordinate but that axis's own member, so that
/// two rows of one cell differ in that axis and in nothing else. The shape is one of the
/// coordinates - two shapes write different arrays, whatever they share - and so are the group,
/// the precision and the lane whose tables the entry reads.
std::string CellKey(const DeviceOptionInfo& row, DeviceOptionAxis axis) {
    char key[192];
    std::snprintf(key,
                  sizeof(key),
                  "%d|%d|%d|%d|%d|%d|%d|%d|%d|%d",
                  static_cast<int>(row.group),
                  static_cast<int>(row.precision),
                  static_cast<int>(row.shape),
                  static_cast<int>(row.question),
                  static_cast<int>(row.lane),
                  axis == DeviceOptionAxis::kRoute ? -1 : static_cast<int>(row.route),
                  axis == DeviceOptionAxis::kScheme ? -1 : static_cast<int>(row.scheme),
                  axis == DeviceOptionAxis::kPartition ? -1 : static_cast<int>(row.partition),
                  axis == DeviceOptionAxis::kPacking ? -1 : static_cast<int>(row.packing),
                  axis == DeviceOptionAxis::kRegionBExp ? -1 : static_cast<int>(row.regionBExp));
    return key;
}

/// Whether the store a row writes through is narrower than the engine that computed the value.
///
/// The fp64 and fp32 lanes store exactly what they compute in, so identical bits there are one
/// arithmetic; the half lane computes in a float engine and stores eleven or eight mantissa bits
/// of the result, so identical bits there may be one arithmetic or two.
bool StoreHides(const DeviceOptionInfo& row) {
    return row.precision == DeviceOptionPrecision::kFp16 ||
           row.precision == DeviceOptionPrecision::kBf16;
}

/// Whether the store a row writes through is as wide as the engine the half lane computes in.
///
/// The half lane's arithmetic is the float lane's: the same bodies over the same tables, with the
/// result put into eleven or eight mantissa bits instead of the engine's twenty-four. So the
/// float lane is the witness that can show a difference the half store rounded away, and it is
/// the only one - a fp64 row is another engine, and reading one beside a fp32 row would let the
/// width of the store stand in for the axis being read.
bool WideWitness(const DeviceOptionInfo& row) {
    return row.precision == DeviceOptionPrecision::kFp32;
}

/// Half a unit in the last place of the value the row's store carries: the difference the format
/// cannot represent, and therefore the difference it cannot show. Stated in the terms the library
/// states those lanes in - eleven mantissa bits for fp16 and eight for bf16 - and not measured
/// here, because it is a property of the format rather than of this run.
double StoreResolution(const DeviceOptionInfo& row) {
    switch (row.precision)
    {
    case DeviceOptionPrecision::kFp16:
        return 0x1p-11;
    case DeviceOptionPrecision::kBf16:
        return 0x1p-8;
    default:
        return 0.0;
    }
}

/// The cell a row belongs to along one axis with the store dropped: every coordinate CellKey
/// fixes except the precision and the degree tables, which are what a wider store changes and
/// what the arithmetic does not depend on. Two rows that agree here run the same arithmetic over
/// the same class of tables and differ in the width they hand the result back in.
std::string WideKey(const DeviceOptionInfo& row, DeviceOptionAxis axis) {
    char key[192];
    std::snprintf(key,
                  sizeof(key),
                  "%d|%d|%d|%d|%d|%d|%d|%d",
                  static_cast<int>(row.group),
                  static_cast<int>(row.shape),
                  static_cast<int>(row.question),
                  axis == DeviceOptionAxis::kRoute ? -1 : static_cast<int>(row.route),
                  axis == DeviceOptionAxis::kScheme ? -1 : static_cast<int>(row.scheme),
                  axis == DeviceOptionAxis::kPartition ? -1 : static_cast<int>(row.partition),
                  axis == DeviceOptionAxis::kPacking ? -1 : static_cast<int>(row.packing),
                  axis == DeviceOptionAxis::kRegionBExp ? -1 : static_cast<int>(row.regionBExp));
    return key;
}

/// The value a wide row's stored bits denote, as a double: what the engine that computed it
/// produced, before any half store took its digits away. Read off the row's own precision, so
/// this is the number the entry delivered and not a widening of a rounded one.
double WideValue(const DeviceOptionInfo& row, std::uint64_t bits) {
    if (row.precision == DeviceOptionPrecision::kFp64)
    {
        return std::bit_cast<double>(bits);
    }

    return static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(bits)));
}

/// The widest per-slot spread the members of one wide cell show, in the engine's own value: the
/// number the classification is made on, printed so that a reader checks it against the store's
/// resolution rather than taking the split on trust. Zero when the members are one reading.
double WideSpread(const std::vector<std::size_t>& wide,
                  const std::vector<DeviceOptionInfo>& rows,
                  const std::vector<Run>& reference) {
    if (wide.size() < 2)
    {
        return 0.0;
    }

    const std::size_t slots = reference[wide.front()].bits.size();
    double widest = 0.0;

    for (std::size_t slot = 0; slot < slots; ++slot)
    {
        double lo = WideValue(rows[wide.front()], reference[wide.front()].bits[slot]);
        double hi = lo;

        for (std::size_t k = 1; k < wide.size(); ++k)
        {
            const double value = WideValue(rows[wide[k]], reference[wide[k]].bits[slot]);
            lo = std::min(lo, value);
            hi = std::max(hi, value);
        }

        widest = std::max(widest, hi - lo);
    }

    return widest;
}

/// Whether a cell whose members delivered one reading is a cell this store cannot resolve: the
/// same axis over the same arithmetic, run at the float lane, whose store is the engine's own,
/// delivers more than one reading there. That is the whole of the classification - the members
/// of an axis are known to differ only where a store can be told, and a cell whose wider siblings
/// are one reading too is one arithmetic under two names rather than one the format rounded
/// together, so it stays a failure.
bool UnresolvableAtThisStore(const DeviceOptionInfo& row,
                             DeviceOptionAxis axis,
                             const std::map<std::string, std::vector<std::size_t>>& wideCells,
                             const std::vector<DeviceOptionInfo>& rows,
                             const std::vector<Run>& reference,
                             double& spread,
                             std::string& why) {
    why.clear();

    if (!StoreHides(row))
    {
        return false;
    }

    const auto wide = wideCells.find(WideKey(row, axis));

    if (wide == wideCells.end())
    {
        why = "the float lane runs no counterpart of this cell, so the format's resolution is "
              "not what is left to explain it";
        return false;
    }

    // The wider members this run reached. A member the run could not reach is dropped rather
    // than fatal - the field holds rows of two widths and one of them may be a row this test
    // has no call for - but the two members of the axis have to survive it, or the cell would
    // be read as the half lane's resolution when what is missing is a measurement.
    std::vector<std::size_t> ran;
    std::map<int, std::size_t> wideMembers;
    std::map<std::vector<std::uint64_t>, std::size_t> readings;

    for (const std::size_t i : wide->second)
    {
        if (!reference[i].ok)
        {
            continue;
        }

        ran.push_back(i);
        ++wideMembers[MemberOf(rows[i], axis)];
        readings[reference[i].bits] = i;
    }

    if (wideMembers.size() < 2)
    {
        why = "the float lane ran no two members of this axis over this cell, so "
              "the format's resolution is not what is left to explain it";
        return false;
    }

    if (readings.size() < 2)
    {
        why = "the same axis at the float lane delivered one value too: this is "
              "one arithmetic under two names and not a difference the format rounded away";
        return false;
    }

    spread = WideSpread(ran, rows, reference);
    return true;
}

/// What one axis's walk found over the whole table.
struct AxisTally {
    std::size_t cells = 0;
    std::size_t compared = 0;
    std::size_t rowsCompared = 0;
    std::size_t skippedOneMember = 0;
    std::size_t skippedUnrun = 0;
    std::size_t noSibling = 0;
    std::size_t failed = 0;
    std::size_t notObservable = 0;
    std::size_t rowsNotObservable = 0;
};

} // namespace

int main(int argc, char** argv) {
    const char* collapse = nullptr;

    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];

        if (argument.rfind("--collapse=", 0) == 0)
        {
            collapse = argv[i] + std::strlen("--collapse=");
        } else
        {
            std::printf("usage: %s [--collapse=<axis>]  axis: route | scheme | partition | "
                        "packing | region-B exponential | division form\n",
                        argv[0]);
            return 2;
        }
    }

    int devices = 0;

    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    {
        std::printf("boys-cuda-axis-sensitivity: no CUDA device, nothing to measure\n");
        return 0;
    }

    cudaDeviceProp properties{};

    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess)
    {
        std::printf("boys-cuda-axis-sensitivity: the device would not answer for itself\n");
        return 1;
    }

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        std::printf("boys-cuda-axis-sensitivity: the tables would not upload\n");
        return 1;
    }

    Context ctx;

    if (BoysCuda::DeviceTables(&ctx.tables) != BoysStatus::kSuccess || !ctx.Upload())
    {
        std::printf("boys-cuda-axis-sensitivity: the arguments or the handle would not upload\n");
        return 1;
    }

    const std::span<const DeviceOptionInfo> table = BoysDeviceOptions();
    const std::vector<DeviceOptionInfo> rows(table.begin(), table.end());

    // The row each row is actually run as: identity, except under the deliberate-defect control,
    // where every row of a cell runs at the entry of the cell's own first member.
    std::vector<int> runAs(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        runAs[i] = static_cast<int>(i);
    }

    // The division axis has no cell of its own - every row states the build's form - so the
    // control for it is the other forms not being reached at all.
    const bool collapseDivision =
        collapse != nullptr && std::strcmp(collapse, "division form") == 0;

    if (collapse != nullptr)
    {
        bool matched = collapseDivision;

        for (const Axis& candidate : kAxes)
        {
            if (std::strcmp(collapse, candidate.name) != 0)
            {
                continue;
            }

            matched = true;

            std::map<std::string, int> first;

            for (std::size_t i = 0; i < rows.size(); ++i)
            {
                const std::string key = CellKey(rows[i], candidate.axis);

                if (first.find(key) == first.end())
                {
                    first[key] = static_cast<int>(i);
                } else
                {
                    runAs[i] = first[key];
                }
            }
        }

        if (!matched)
        {
            std::printf("boys-cuda-axis-sensitivity: no axis is named %s\n", collapse);
            return 2;
        }

        std::printf("boys-cuda-axis-sensitivity: COLLAPSED - %s\n",
                    collapseDivision ? "every row runs at the build's own division form"
                                     : "every row of a cell runs at the entry of the cell's first "
                                       "member on that axis");
    }

    std::printf("boys-cuda-axis-sensitivity\n");
    std::printf(
        "  device            %s (sm_%d%d)\n", properties.name, properties.major, properties.minor);
    std::printf("  rows              %zu (from BoysDeviceOptions())\n", rows.size());
    std::printf("  arguments         %zu over [0, %.3g], orders 0..%d, top order %d\n",
                ctx.count,
                ctx.x.back(),
                boys::kMaxBoysOrder,
                ctx.nmax);

    // ---- one run per row per form, and the reading of the division axis ---------------------
    const DivisionForm defaultForm = kDefaultDeviceDivisionForm;
    const DivisionForm otherForms[] = {DivisionForm::kPlainReciprocal,
                                       DivisionForm::kExactDivision};

    std::vector<Run> reference(rows.size());
    std::vector<std::vector<Run>> atOtherForm(rows.size());

    std::size_t unrun = 0;
    std::size_t insane = 0;

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        RunRow(rows[runAs[i]], ctx, defaultForm, reference[i]);

        if (!reference[i].ok)
        {
            ++unrun;
            std::printf("  not run: %-44s %s\n", rows[i].name, reference[i].why.c_str());
            continue;
        }

        if (!reference[i].sane)
        {
            ++insane;
            std::printf(
                "  not a Boys value: %-40s %s\n", rows[i].name, reference[i].insane.c_str());
        }

        // The third member is a second kernel instantiation per in-kernel entry, and the launched
        // rows - one instantiating site per device body - carry it for both halves.
        const std::size_t formCount = rows[i].group == DeviceOptionGroup::kLaunched ? 2 : 1;

        for (std::size_t f = 0; f < formCount; ++f)
        {
            atOtherForm[i].emplace_back();
            RunRow(rows[runAs[i]],
                   ctx,
                   collapseDivision ? defaultForm : otherForms[f],
                   atOtherForm[i].back());

            if (!atOtherForm[i].back().ok)
            {
                std::printf("  not run: %-44s at %s: %s\n",
                            rows[i].name,
                            boys::DivisionFormName(otherForms[f]),
                            atOtherForm[i].back().why.c_str());
            }
        }
    }

    std::printf("  runs              %zu rows, %zu not run, %zu with a value outside [0, 1]\n",
                rows.size(),
                unrun,
                insane);

    // ---- the five axes the option table varies a row along ----------------------------------
    std::size_t failed = 0;
    std::size_t coincidences = 0;
    std::size_t notObservable = 0;

    for (const Axis& axis : kAxes)
    {
        AxisTally tally;
        std::map<std::string, std::vector<std::size_t>> cells;

        // The rows that can witness what the half lane's store rounded away, keyed by the same
        // coordinates as the cells above with the store dropped: one store's own readings, so
        // that a difference between two widths cannot stand in for a difference between two
        // members of the axis.
        std::map<std::string, std::vector<std::size_t>> wideCells;

        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            cells[CellKey(rows[i], axis.axis)].push_back(i);

            if (WideWitness(rows[i]))
            {
                wideCells[WideKey(rows[i], axis.axis)].push_back(i);
            }
        }

        for (const auto& entry : cells)
        {
            const std::vector<std::size_t>& cell = entry.second;

            if (cell.size() < 2)
            {
                // No other row of this class states a member of this axis: the axis is not one
                // this entry carries a second member of, which for the packing axis and the
                // grid is what the library's own words say of that shape.
                tally.noSibling += cell.size();
                continue;
            }

            ++tally.cells;

            std::map<int, std::vector<std::size_t>> byMember;
            for (const std::size_t i : cell)
            {
                byMember[MemberOf(rows[i], axis.axis)].push_back(i);
            }

            if (byMember.size() < 2)
            {
                tally.skippedOneMember += cell.size();
                continue;
            }

            bool everyRun = true;
            for (const std::size_t i : cell)
            {
                everyRun = everyRun && reference[i].ok;
            }

            if (!everyRun)
            {
                tally.skippedUnrun += cell.size();
                continue;
            }

            // The distinct readings of the cell: one class per distinct block of bits.
            std::map<std::vector<std::uint64_t>, std::vector<std::size_t>> classes;
            for (const std::size_t i : cell)
            {
                classes[reference[i].bits].push_back(i);
            }

            ++tally.compared;
            tally.rowsCompared += cell.size();

            if (classes.size() < 2)
            {
                // One reading for every member. That is a failure only where the store the cell
                // wrote through could have shown a difference: the half lane's store is narrower
                // than the engine that computed the value, so a cell of it whose wider siblings
                // do differ is one this format cannot resolve rather than a claim with nothing
                // behind it, and is reported as its own class.
                double spread = 0.0;
                std::string why;
                const DeviceOptionInfo& first = rows[cell.front()];

                if (UnresolvableAtThisStore(
                        first, axis.axis, wideCells, rows, reference, spread, why))
                {
                    ++tally.notObservable;
                    tally.rowsNotObservable += cell.size();

                    for (const std::size_t i : cell)
                    {
                        std::printf(
                            "  not observable %-44s %s; this store cannot resolve it - %zu values "
                            "per row identical, the same axis at the float lane differs by up to "
                            "%.3g, half-ULP %.3g\n",
                            rows[i].name,
                            axis.name,
                            reference[i].bits.size(),
                            spread,
                            StoreResolution(first));
                    }

                    continue;
                }

                ++tally.failed;
                ++failed;

                std::printf("\nFAIL %s\n", first.name);
                std::printf(
                    "  the axis %s is not read: every member of this cell delivered the same "
                    "values\n",
                    axis.name);
                std::printf("  members:");

                for (const auto& member : byMember)
                {
                    const char* text = MemberText(rows[member.second.front()], axis.axis);

                    if (text != nullptr)
                    {
                        std::printf(" %s", text);
                    } else
                    {
                        // The route and the packing axis have no name of their own in the
                        // library; the rows below are the spelling of the member.
                        std::printf(" member %d", member.first);
                    }

                    for (const std::size_t i : member.second)
                    {
                        std::printf(" <%s>", rows[i].name);
                    }

                    std::printf(" |");
                }

                std::printf("\n  compared %zu values per row, every one identical\n",
                            reference[cell.front()].bits.size());

                if (!why.empty())
                {
                    // A cell of the half lane that could not be moved to the other class says
                    // which question it failed, so that a reader can tell a real finding from a
                    // cell this run had no wider store to check.
                    std::printf("  not moved to \"not observable\": %s\n", why.c_str());
                }
            } else if (classes.size() < byMember.size())
            {
                ++coincidences;
                std::printf("  coincidence on %s: %zu of %zu members share one reading:\n",
                            axis.name,
                            byMember.size() - classes.size(),
                            byMember.size());

                for (const auto& group : classes)
                {
                    if (group.second.size() < 2)
                    {
                        continue;
                    }

                    std::printf("    {");
                    for (const std::size_t i : group.second)
                    {
                        std::printf(" %s", rows[i].name);
                    }
                    std::printf(" }");
                    std::printf("  members %d\n", MemberOf(rows[group.second.front()], axis.axis));
                }
            }
        }

        std::printf("  axis %-20s cells %3zu  compared %3zu  rows %3zu  no other member %3zu  "
                    "skipped %zu (one member) + %zu (not run)  FAILED %3zu  not observable %3zu "
                    "(%zu rows)\n",
                    axis.name,
                    tally.cells,
                    tally.compared,
                    tally.rowsCompared,
                    tally.noSibling,
                    tally.skippedOneMember,
                    tally.skippedUnrun,
                    tally.failed,
                    tally.notObservable,
                    tally.rowsNotObservable);

        notObservable += tally.notObservable;
    }

    // ---- the division axis: the row's own entry at each form --------------------------------
    std::size_t divisionFailed = 0;
    std::size_t divisionCompared = 0;
    std::size_t divisionDiffered[2] = {0, 0};

    std::size_t divisionUnresolvable = 0;

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        if (!reference[i].ok || atOtherForm[i].empty())
        {
            continue;
        }

        // The half lane's two classes store eleven and eight mantissa bits of a float engine's
        // result, and what one division form adds to another is below both: the two forms' values
        // are one value once either format has it, so a row of either class cannot show the
        // difference either way and is not evidence about the axis. The launched half's rows are
        // the same rows.
        if (rows[i].precision == DeviceOptionPrecision::kFp16 ||
            rows[i].precision == DeviceOptionPrecision::kBf16)
        {
            ++divisionUnresolvable;
            continue;
        }

        bool everyForm = true;
        for (const Run& run : atOtherForm[i])
        {
            everyForm = everyForm && run.ok;
        }

        if (!everyForm)
        {
            continue;
        }

        ++divisionCompared;

        bool anyDiffers = false;

        for (std::size_t f = 0; f < atOtherForm[i].size(); ++f)
        {
            if (atOtherForm[i][f].bits != reference[i].bits)
            {
                anyDiffers = true;
                ++divisionDiffered[f];
            }
        }

        if (!anyDiffers)
        {
            ++divisionFailed;
            std::printf("\nFAIL %s\n", rows[i].name);
            std::printf("  the axis division form is not read: the entry delivered the same values "
                        "at %s and at %s\n",
                        boys::DivisionFormName(defaultForm),
                        boys::DivisionFormName(otherForms[0]));
            std::printf("  compared %zu values, every one identical\n", reference[i].bits.size());
        }
    }

    std::printf("  axis %-20s rows %3zu  compared %3zu  FAILED %zu  not judgeable at the half "
                "store %zu\n",
                "division form",
                rows.size(),
                divisionCompared,
                divisionFailed,
                divisionUnresolvable);
    std::printf("    %-22s differs from %s on %zu of %zu rows\n",
                boys::DivisionFormName(otherForms[0]),
                boys::DivisionFormName(defaultForm),
                divisionDiffered[0],
                divisionCompared);
    std::printf("    %-22s differs from %s on %zu of the launched rows\n",
                boys::DivisionFormName(otherForms[1]),
                boys::DivisionFormName(defaultForm),
                divisionDiffered[1]);

    failed += divisionFailed;

    std::printf(
        "\n  %zu row(s) failed an axis check, %zu coincidence(s) reported\n", failed, coincidences);
    // The cells are counted over the five axes, so one row can be in five of them and the row
    // figures are the per-axis ones above rather than a sum.
    std::printf("  %zu cell(s) the half lane's store cannot resolve: one value at the row's own "
                "format (half-ULP %.3g for fp16, %.3g for bf16) and more than one at the float "
                "lane. Named above and not failing.\n",
                notObservable,
                0x1p-11,
                0x1p-8);
    std::printf("  %zu row(s) cannot run: the entry the row names is not a member this test "
                "calls, so the row is a coverage gap and not a verdict.\n",
                unrun);

    ctx.Release();

    return failed == 0 ? 0 : 1;
}
