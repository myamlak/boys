#pragma once

/// ile
/// The device option space: one row per option of the CUDA lane, with what a
/// chooser needs to place it.
///
/// A header of its own, reached from boys_cuda.hpp, because these rows are one
/// table with two readers: the host surface that declares the entries
/// (BoysCuda, in boys_cuda.hpp) and the device functions that implement them
/// (boys_cuda_device.hpp). The second is compiled by nvcc, in a translation
/// unit that has to stay free of the library's C++23 headers, so the rows live
/// here rather than in the host header alone — which is what lets a probe's
/// rows and an accuracy gate's claims be projections of one report instead of
/// lists that drift from it.

#include "boys/accuracy.hpp"
#include "boys/boys_device_tables.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace boys {

// ---------------------------------------------------------------------------
// The device option space.
// ---------------------------------------------------------------------------

/// One option of the device lane: the row unit of BoysDeviceOptions().
///
/// The enumerator order is the report's row order — the launched entries of a
/// shape before the device-callable ones — and a report prints its rows in it.
///
/// A row is one *option* and not always one entry: an entry that carries an
/// axis is one row per member of that axis, because the members are different
/// arithmetic with different documented bounds. The f32 single entries are the
/// case this build has — two rows each, one per RegionBExp.
///
/// \ingroup boys
enum class DeviceEntry : int {
    kSingleF64 = 0, ///< BoysCuda::SingleF64, launched
    kSingleF32, ///< BoysCuda::SingleF32 at RegionBExp::kAccurate, launched
    kSingleF32Fast, ///< BoysCuda::SingleF32 at RegionBExp::kFast, launched
    kSingleF16, ///< BoysCuda::SingleF16, launched

    kAllOrdersF64, ///< BoysCuda::AllOrdersF64, launched
    kAllOrdersF32, ///< BoysCuda::AllOrdersF32, launched
    kAllOrdersF16, ///< BoysCuda::AllOrdersF16, launched

    kAllOrdersF64Narrow, ///< BoysCuda::AllOrdersF64Narrow, launched
    kAllOrdersF64Orders, ///< BoysCuda::AllOrdersF64Orders, launched
    kAllOrdersF64NarrowOrders, ///< BoysCuda::AllOrdersF64NarrowOrders, launched
    kAllOrdersF64Mono, ///< BoysCuda::AllOrdersF64Mono, launched
    kAllOrdersF64OrdersMono, ///< BoysCuda::AllOrdersF64OrdersMono, launched
    kAllOrdersF64NarrowMono, ///< BoysCuda::AllOrdersF64NarrowMono, launched
    kAllOrdersF64NarrowOrdersMono, ///< BoysCuda::AllOrdersF64NarrowOrdersMono, launched

    kAllOrdersF64Rat, ///< BoysCuda::AllOrdersF64Rat at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF64RatHorner, ///< BoysCuda::AllOrdersF64Rat at EvalScheme::kHorner, launched
    kAllOrdersF64OrdersRat, ///< BoysCuda::AllOrdersF64OrdersRat at kSplitClenshaw, launched
    kAllOrdersF64OrdersRatHorner, ///< BoysCuda::AllOrdersF64OrdersRat at kHorner, launched
    kAllOrdersF64NarrowRat, ///< BoysCuda::AllOrdersF64NarrowRat at kSplitClenshaw, launched
    kAllOrdersF64NarrowRatHorner, ///< BoysCuda::AllOrdersF64NarrowRat at kHorner, launched
    kAllOrdersF64NarrowOrdersRat, ///< BoysCuda::AllOrdersF64NarrowOrdersRat at kSplitClenshaw
    kAllOrdersF64NarrowOrdersRatHorner, ///< BoysCuda::AllOrdersF64NarrowOrdersRat at kHorner

    kAllOrdersF64Uniform, ///< BoysCuda::AllOrdersF64Uniform at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF64UniformHorner, ///< BoysCuda::AllOrdersF64UniformHorner at kHorner, launched
    kAllOrdersF64OrdersUniform, ///< BoysCuda::AllOrdersF64OrdersUniform at kSplitClenshaw, launched
    kAllOrdersF64OrdersUniformHorner, ///< BoysCuda::AllOrdersF64OrdersUniformHorner at kHorner, launched

    kAllOrdersF32Narrow, ///< BoysCuda::AllOrdersF32Narrow at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF32NarrowMono, ///< BoysCuda::AllOrdersF32NarrowMono at kHorner, launched
    kAllOrdersF32Uniform, ///< BoysCuda::AllOrdersF32Uniform at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF32UniformHorner, ///< BoysCuda::AllOrdersF32UniformHorner at kHorner, launched

    /// The float lane's rational route, which the two rows below carry on the
    /// shipped partition and the two after them on the narrow one — the same
    /// pair of partitions the double lane's route is carried on, and the same
    /// reason for two names per partition: the route's pair is stored in
    /// monomial form and read by Horner, so neither scheme name a caller may
    /// use selects a second arithmetic and the two rows of a partition run one
    /// kernel.
    kAllOrdersF32Rat, ///< BoysCuda::AllOrdersF32Rat at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF32RatHorner, ///< BoysCuda::AllOrdersF32RatHorner at kHorner, launched
    kAllOrdersF32NarrowRat, ///< BoysCuda::AllOrdersF32NarrowRat at kSplitClenshaw, launched
    kAllOrdersF32NarrowRatHorner, ///< BoysCuda::AllOrdersF32NarrowRatHorner at kHorner, launched

    /// The float lane's other packing axis: the counterpart of each row above
    /// that carries a packing axis's choice, over the same stored table and at
    /// the same rungs.
    ///
    /// The axis is a reading of region A and not a second fit: every order's own
    /// piece is located and its own fit summed, where the per-argument shape
    /// seeds the top order's fit and brings the lower orders back down a
    /// recurrence. The double lane's rows of the axis are the shape these
    /// mirror, one per partition and route, and the two agree row for row: the
    /// shipped partition's rung is that lane's own cut and its orders row carries
    /// the two forms a rung has, while the narrow partition's and the rational
    /// route's orders rows serve the reference multiplier alone — a rung of
    /// theirs would read a cut this lane does not hold, which is the statement
    /// their per-argument siblings already make.
    ///
    /// The shipped partition's row is one row for both scheme names, exactly as
    /// \c kAllOrdersF32 is: this lane stores that partition once, in the
    /// Chebyshev basis, and the scheme axis names which of the lane's two stored
    /// forms a body sums. The uniform grid's two are its per-argument rows'
    /// kernels, for the reason those rows state — the grid's cells carry their
    /// own degree and block start, so the route's packing axis has one member.
    kAllOrdersF32Orders, ///< BoysCuda::AllOrdersF32Orders, launched
    kAllOrdersF32NarrowOrders, ///< BoysCuda::AllOrdersF32NarrowOrders, launched
    kAllOrdersF32NarrowOrdersMono, ///< BoysCuda::AllOrdersF32NarrowOrdersMono, launched
    kAllOrdersF32OrdersRat, ///< BoysCuda::AllOrdersF32OrdersRat at kSplitClenshaw, launched
    kAllOrdersF32OrdersRatHorner, ///< BoysCuda::AllOrdersF32OrdersRatHorner at kHorner
    kAllOrdersF32NarrowOrdersRat, ///< BoysCuda::AllOrdersF32NarrowOrdersRat at kSplitClenshaw
    kAllOrdersF32NarrowOrdersRatHorner, ///< BoysCuda::AllOrdersF32NarrowOrdersRatHorner at kHorner
    kAllOrdersF32OrdersUniform, ///< BoysCuda::AllOrdersF32OrdersUniform, launched
    kAllOrdersF32OrdersUniformHorner, ///< BoysCuda::AllOrdersF32OrdersUniformHorner, launched

    kAllNF64, ///< BoysCuda::AllNF64, launched
    kAllNF32, ///< BoysCuda::AllNF32, launched
    kAllNF16, ///< BoysCuda::AllNF16, launched

    kDeviceSingleF64, ///< BoysDeviceSingleF64, inside the caller's kernel
    kDeviceSingleF32, ///< BoysDeviceSingleF32 at RegionBExp::kAccurate, in-kernel
    kDeviceSingleF32Fast, ///< BoysDeviceSingleF32 at RegionBExp::kFast, in-kernel
    kDeviceSingleF16, ///< BoysDeviceSingleF16, inside the caller's kernel

    kDeviceAllOrdersF64, ///< BoysDeviceAllOrdersF64, inside the caller's kernel
    kDeviceAllOrdersF32, ///< BoysDeviceAllOrdersF32, inside the caller's kernel
    kDeviceAllOrdersF16, ///< BoysDeviceAllOrdersF16, inside the caller's kernel

    kDeviceAllNF64, ///< BoysDeviceAllNF64, inside the caller's kernel
    kDeviceAllNF32, ///< BoysDeviceAllNF32, inside the caller's kernel
    kDeviceAllNF16, ///< BoysDeviceAllNF16, inside the caller's kernel

    kDeviceEachOrderF64, ///< BoysDeviceEachOrderF64, inside the caller's kernel
    kDeviceEachOrderF32, ///< BoysDeviceEachOrderF32, inside the caller's kernel
    kDeviceEachOrderF16, ///< BoysDeviceEachOrderF16, inside the caller's kernel

    /// The partition and route axes of the ladder shape, reachable from a
    /// caller's own kernel the way the entries above are reached: through the
    /// handle BoysCuda::DeviceTables filled rather than through a launch of this
    /// library's.
    ///
    /// Each of these is the same option as the launched row named beside it in
    /// the report — one arithmetic, two ways to reach it — so a chooser reading
    /// the two reads one partition, one route, one scheme and one bound.
    kDeviceAllOrdersF64Narrow, ///< BoysDeviceAllOrdersF64Narrow, inside the caller's kernel
    kDeviceAllOrdersF64NarrowMono, ///< BoysDeviceAllOrdersF64NarrowMono, in-kernel
    kDeviceAllOrdersF64Rat, ///< BoysDeviceAllOrdersF64Rat, inside the caller's kernel
    kDeviceAllOrdersF64RatHorner, ///< BoysDeviceAllOrdersF64RatHorner, in-kernel
    kDeviceAllOrdersF64NarrowRat, ///< BoysDeviceAllOrdersF64NarrowRat, in-kernel
    kDeviceAllOrdersF64NarrowRatHorner, ///< BoysDeviceAllOrdersF64NarrowRatHorner, in-kernel
    kDeviceAllOrdersF64Uniform, ///< BoysDeviceAllOrdersF64Uniform, inside the caller's kernel
    kDeviceAllOrdersF64UniformHorner, ///< BoysDeviceAllOrdersF64UniformHorner, in-kernel

    kDeviceAllOrdersF32Narrow, ///< BoysDeviceAllOrdersF32Narrow, inside the caller's kernel
    kDeviceAllOrdersF32NarrowMono, ///< BoysDeviceAllOrdersF32NarrowMono, in-kernel
    kDeviceAllOrdersF32Rat, ///< BoysDeviceAllOrdersF32Rat, inside the caller's kernel
    kDeviceAllOrdersF32RatHorner, ///< BoysDeviceAllOrdersF32RatHorner, in-kernel
    kDeviceAllOrdersF32NarrowRat, ///< BoysDeviceAllOrdersF32NarrowRat, in-kernel
    kDeviceAllOrdersF32NarrowRatHorner, ///< BoysDeviceAllOrdersF32NarrowRatHorner, in-kernel
    kDeviceAllOrdersF32Uniform, ///< BoysDeviceAllOrdersF32Uniform, inside the caller's kernel
    kDeviceAllOrdersF32UniformHorner, ///< BoysDeviceAllOrdersF32UniformHorner, in-kernel

    kCount, ///< rows this report defines; one past the last
};

/// How a caller reaches an option: through a call of this library's that
/// queues a kernel, or through a device function a caller's own kernel calls.
///
/// The two are not comparable as costs — a launched row's figure includes the
/// launch and a device-callable one's is measured by subtraction against the
/// same kernel with the call removed — so a cost report names which one
/// produced each row and never ranks across them.
///
/// \ingroup boys
enum class DeviceOptionGroup : int {
    kLaunched = 0, ///< an entry of BoysCuda, queued by this library
    kDeviceCallable, ///< a function of boys_cuda_device.hpp, inlined into the caller's kernel
};

/// The precision an option computes in: the choice a caller has already made
/// from the accuracy their calculation needs.
///
/// A bound never trades one of these for another — a row of a looser bound is a
/// faster way to compute the precision it names, not a different precision —
/// and the two half formats share one member, because both round a 32-bit
/// engine's result to their format at the boundary instead of computing in it.
/// That is the CPU lane's own reading of the same axis, whose \c Precision::kFp16
/// is "the fp16 and bfloat16 entries", so the member means one thing on both
/// lanes: the half lane, whose arithmetic is the float lane's and whose figure
/// carries a term of the format it stores.
///
/// **This build serves that lane's fp16 entries and not its bfloat16 ones.** A
/// bfloat16 entry of the CPU lane is the same float engine's value stored into
/// bf16, so a bf16 row here would be a row over this lane's existing tables
/// rather than new arithmetic — but no such row exists yet. Closing it is a
/// piece of its own: the launched entries at every rung, their device-callable
/// siblings, and the gate's half-precision block for the second format. Until
/// that lands this member is served for fp16 alone, which is unbuilt work and
/// not a property of the lane.
///
/// \ingroup boys
enum class DeviceOptionPrecision : int {
    kFp64 = 0, ///< the double lane
    kFp32, ///< the float lane
    kFp16, ///< the half lane: the fp16 entries this build serves, and the bfloat16 ones it owes
};

/// The output shape an option answers with: how much it produces for one
/// argument, which is what makes two options comparable questions or not.
///
/// \ingroup boys
enum class DeviceOptionShape : int {
    kSingle = 0, ///< F_n(x) at the argument's own order
    kAllOrders, ///< F_0(x)..F_n(x) at the argument's own order, written to an array
    kAllN, ///< F_0(x)..F_nmax(x) for every argument, one common top order
    kEachOrder, ///< F_0(x)..F_n(x) at the argument's own order, delivered to a sink
};

/// The question an option answers, which is the shape with the sink/array
/// difference folded out: the two are one question and not two.
///
/// \ingroup boys
enum class DeviceOptionQuestion : int {
    kSingle = 0, ///< one value per argument
    kAllOrders, ///< a ladder per argument, to that argument's own order
    kAllN, ///< one common ladder for the whole batch
};

/// The axis an option varies, where it varies one. Every other axis of this
/// surface is the shape (a different entry) or the accuracy multiplier (a rung
/// every row of every precision takes).
///
/// \ingroup boys
enum class DeviceOptionAxis : int {
    kNone = 0, ///< the option is the entry, with no second choice in it
    kRegionBExp, ///< which region-B exponential the entry's recurrence seeds with
    /// Which cut of the domain the entry reads its fits from: the shipped
    /// pieces, the narrow partition's per-order pieces, or the uniform grid,
    /// which covers [0, kFlatHi) in one table of equal intervals and leaves
    /// the range above it to the arm every route ends in. The route the CPU
    /// lane names \c FitGranularity::kUniform is this member of this axis.
    kPartition,
    kPacking, ///< how region A's fits are read: one ladder, or one fit per order
    kScheme, ///< which basis the entry sums its stored fits in; its member is \c scheme
    kRoute, ///< which family of fit the entry's pieces are; its member is \c route
};

// ---------------------------------------------------------------------------
// The accuracy axis: the rungs this lane serves.
// ---------------------------------------------------------------------------

/// The accuracy multipliers the CUDA lane serves, ascending: the values of m a
/// call of this lane may name.
///
/// The lane's relaxed degree tables are cut per multiplier, so a rung is a
/// value of the accuracy axis and the lane's set of them is what its entries
/// answer at. It is the union of two sets: the option space's rungs — the seven
/// multipliers the CPU tier lane's AccuracyTier names, which the accuracy
/// contract is published over and which the CPU lanes serve at run time — and
/// the lane's own six, the finer-at-the-low-end sample set the device
/// arithmetic was measured at. The two overlap at m = 1 alone, the default and
/// every lane's full-accuracy rung.
///
/// The table is the one the library reads. The accessors that answer for the
/// device lane consult it to decide whether a combination is carried, and the
/// entries of the lane are instantiated at each of its values, so the set the
/// API answers for and the set the kernels are compiled at are one set. A rung
/// added to either belongs here in the same change.
///
/// A caller reaches every one of them. The batch entries take the rung as a
/// template argument and are instantiated at each of these values; the
/// device-callable entries take it as a run-time argument and read it against
/// the resident rung, which is one of these at a time (BoysCuda::DeviceTables).
///
/// \ingroup boys
inline constexpr std::array<double, 12> kDeviceRungs = {
    kBoysFullAccuracyMultiplier, 2.0,    10.0,   64.0,   100.0, 256.0,
    1024.0,                      4096.0, 1e4,    16384.0, 65536.0, 1e8,
};

/// Where a multiplier sits in \c kDeviceRungs, or -1 where it is not one.
///
/// The position and not a pointer, because the position is what the lane's
/// run-time entries dispatch on: the set the accessors here answer for, the arms
/// an \c AtRung entry switches over and the multipliers its launchers are
/// instantiated at are then one table read three ways rather than three lists
/// that have to agree. The comparison is exact and the rungs are all exactly
/// representable, which is the same match every entry of this lane makes between
/// the multiplier a call names and the one its degree tables were cut for.
///
/// A value the table does not hold has no position and no rung. That is what a
/// caller is told at a multiplier the lane cannot be resident for — a rung of it
/// is one of the twelve or it does not exist, and a call that answered at
/// another rung than the one it named would be the one outcome the rung argument
/// is for ruling out.
///
/// \param multiplier the accuracy multiplier m a call names
///
/// \returns the position in \c kDeviceRungs, or -1 where \p multiplier is not a
///          rung of this lane
///
/// \ingroup boys
constexpr int DeviceRungIndex(double multiplier) noexcept {
    for (std::size_t i = 0; i < kDeviceRungs.size(); ++i)
    {
        if (kDeviceRungs[i] == multiplier)
        {
            return static_cast<int>(i);
        }
    }

    return -1;
}

/// Whether the CUDA lane serves a multiplier: whether it is one of
/// \c kDeviceRungs.
///
/// The entry names of the lane are instantiated at each of those values and its
/// run-time rung argument is matched against them, so a caller that reads this
/// and a call that dispatches read one table. A value that is not a rung is not
/// served by any entry: the lane answers no arithmetic at it, so a caller is
/// told so rather than handed another rung's tables.
///
/// \param multiplier the accuracy multiplier m a call names
///
/// \returns whether the lane serves it
///
/// \ingroup boys
constexpr bool DeviceRungServed(double multiplier) noexcept {
    return DeviceRungIndex(multiplier) >= 0;
}

// ---------------------------------------------------------------------------
// Which rungs a row of the space is served at, as one mask.
// ---------------------------------------------------------------------------

/// The number of rungs the lane serves, which is the width of a rung mask.
///
/// \ingroup boys
inline constexpr std::size_t kDeviceRungCount = kDeviceRungs.size();

/// A set of rungs of \c kDeviceRungs, one bit each: bit i is the rung
/// \c kDeviceRungs[i].
///
/// A mask and not a list per row, because the axis is one of twelve and a report
/// states it for every row: a row that carried the rungs as names would carry a
/// second spelling of the table beside it, and a row that carried none left a
/// reader to infer coverage from a field that does not have it. A report prints
/// the count a mask holds and names the rungs only of a row that holds less than
/// the whole of it.
///
/// \ingroup boys
using DeviceRungMask = std::uint32_t;

/// The rungs of the lane, all of them, as a mask.
///
/// \ingroup boys
inline constexpr DeviceRungMask kEveryDeviceRung = [] {
    DeviceRungMask mask = 0;

    for (std::size_t i = 0; i < kDeviceRungCount; ++i)
    {
        mask |= DeviceRungMask{1} << i;
    }

    return mask;
}();

static_assert(kDeviceRungCount <= 32,
              "a rung mask is a 32-bit word, so this lane cannot hold more rungs than it has "
              "bits: widen DeviceRungMask in the change that adds one");

/// The rung \p multiplier names, as the one bit of a mask that holds it, or no
/// bit at all where it names no rung of this lane.
///
/// \param multiplier the accuracy multiplier m
///
/// \returns the bit for that rung, or zero
///
/// \ingroup boys
constexpr DeviceRungMask DeviceRungBit(double multiplier) noexcept {
    const int index = DeviceRungIndex(multiplier);

    return index < 0 ? DeviceRungMask{0} : DeviceRungMask{1} << index;
}

/// Whether an entry of this lane answers at a multiplier.
///
/// Almost every entry does at every rung of \c kDeviceRungs, and this is
/// \c DeviceRungServed for them. The exceptions are the entries whose rung axis
/// has a cut to make that this lane does not hold, and there is one kind of
/// them in this lane: a table whose cut this lane does not hold.
///
/// The uniform route's six entries are the first kind and are *not* an
/// exception at all: the route's table is stored at one degree for every order
/// and every interval, that degree is admissible at every multiplier (the
/// criterion's Delta(deg) is zero, so the full degree always is — see
/// boys_effective_degrees.hpp), and every rung is therefore served by the
/// route's own coefficients.
/// The route's rows are one table and not six exceptions — the two members of
/// the scheme axis and the two of the packing axis are rows of the same table,
/// and the float lane's two are rows of that lane's own grid — so a rung the
/// route serves is one every row of it serves, and a call there is answered by
/// the route's own coefficients and by nothing else. What a rung does not buy
/// on this route is less work: there is no shorter fit to read, and the cost of
/// a call is the route's cost at every multiplier. That is a property of a
/// route whose table has no cut to make, and the row's bound is what states the
/// accuracy a rung of it delivers.
///
/// The float lane's narrow partition, and the rational route at that lane's
/// precision, are the kind above, and their reason is unbuilt work rather than
/// an empty axis: their tables *have* a cut to make per rung, and this lane
/// does not hold it. The derivation that cuts a stored fit to a rung's criterion
/// exists for the float lane's pieces (boys_effective_degrees.hpp derives the
/// float lane's narrow region A and region B cuts per role) and for its rational
/// pairs, but this lane derives and uploads the double role's cut alone and no
/// kernel of it reads either a float narrow degree or a float rational pair's
/// cut. A relaxed call would have to be answered by degrees no kernel here
/// holds, so it is refused — and the work that would lift the refusal is
/// deriving, uploading and reading that cut.
///
/// The rational route's four rows are refused for that same reason rather than
/// for the double lane's route's. A relaxed call of theirs would cut region A
/// from the double lane's rational cut, which *is* held here — the float lanes
/// seed region A from the double lane's pair — but region B is the float lane's
/// own pair, whose cut is not, and a body cut on one side and not the other is a
/// cut nobody derived. The two partitions' pairs of rows are refused together
/// because the missing table is the float lane's region-B pairs on either of
/// them.
///
/// Every other member of every axis this lane carries — the shipped and the
/// narrow partitions of the double lane, the level ladder and the per-order
/// reading, both schemes and both routes — keeps its twelve rungs, because each
/// is a stored table whose rung cut is derived per piece and uploaded.
///
/// The name is a property of the entry and not of the caller, which is why it
/// is here beside the rows: a probe that decided it for itself, and an entry
/// that decided it for itself, would be two answers to one question.
///
/// Every enumerator of \c DeviceEntry is named below, and the switch has no
/// default arm: an option added to the enumeration without a rung axis stated
/// here is a compile error rather than an option silently reported at every
/// rung. That is the one link this header can make on its own — the entries
/// themselves are one TU away and are tied to this list by the entry their own
/// refusal is written for, which they state by name.
///
/// \param entry      the entry a call would name
/// \param multiplier the accuracy multiplier m
///
/// \returns whether that entry answers at that rung
///
/// \ingroup boys
constexpr bool DeviceEntryServedAtRung(DeviceEntry entry, double multiplier) noexcept {
    switch (entry)
    {
        // The rungs of a lane whose cut this build does not hold: the entry's
        // own rung axis has a cut to make and the cut is not derived here, so
        // the one rung it serves is the reference multiplier. The reason is
        // above, per partition and per route, and is not a property of the row
        // that names it.
        case DeviceEntry::kAllOrdersF32Narrow:
        case DeviceEntry::kAllOrdersF32NarrowMono:
        case DeviceEntry::kAllOrdersF32Rat:
        case DeviceEntry::kAllOrdersF32RatHorner:
        case DeviceEntry::kAllOrdersF32NarrowRat:
        case DeviceEntry::kAllOrdersF32NarrowRatHorner:
        // The float lane's orders rows on those same two tables: the cut a rung
        // would read is a cut of the table, so the row that reads it on the other
        // axis is served at the same rungs and no others.
        case DeviceEntry::kAllOrdersF32NarrowOrders:
        case DeviceEntry::kAllOrdersF32NarrowOrdersMono:
        case DeviceEntry::kAllOrdersF32OrdersRat:
        case DeviceEntry::kAllOrdersF32OrdersRatHorner:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRat:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner:
            return multiplier == kBoysFullAccuracyMultiplier;

        // The uniform route's six rows, which are no exception: the route's
        // table has no cut to make, so every rung's own arithmetic *is* the
        // route's, and every rung of it is served. The reason is above, per
        // route, and it is why these fall through to the arm below. The float
        // lane's two orders rows of that route are the same table and the same
        // reason.
        case DeviceEntry::kAllOrdersF64Uniform:
        case DeviceEntry::kAllOrdersF64UniformHorner:
        case DeviceEntry::kAllOrdersF64OrdersUniform:
        case DeviceEntry::kAllOrdersF64OrdersUniformHorner:
        case DeviceEntry::kAllOrdersF32Uniform:
        case DeviceEntry::kAllOrdersF32UniformHorner:
        case DeviceEntry::kAllOrdersF32OrdersUniform:
        case DeviceEntry::kAllOrdersF32OrdersUniformHorner:

        // Every other option of the surface, which is the lane's rungs and no
        // fewer of them. The single-order entries, the three batch shapes.
        case DeviceEntry::kSingleF64:
        case DeviceEntry::kSingleF32:
        case DeviceEntry::kSingleF32Fast:
        case DeviceEntry::kSingleF16:
        case DeviceEntry::kAllOrdersF64:
        case DeviceEntry::kAllOrdersF32:
        // The float lane's orders row on the shipped partition, whose rung is
        // that lane's own cut of the float Chebyshev table: it reads one fit per
        // order where its per-argument twin reads the recurrence, so the two are
        // served at the same rungs and the cut is the table's.
        case DeviceEntry::kAllOrdersF32Orders:
        case DeviceEntry::kAllOrdersF16:
        case DeviceEntry::kAllNF64:
        case DeviceEntry::kAllNF32:
        case DeviceEntry::kAllNF16:
        // The double lane's level ladder and its per-order reading, on either
        // partition, in either scheme.
        case DeviceEntry::kAllOrdersF64Narrow:
        case DeviceEntry::kAllOrdersF64Orders:
        case DeviceEntry::kAllOrdersF64NarrowOrders:
        case DeviceEntry::kAllOrdersF64Mono:
        case DeviceEntry::kAllOrdersF64OrdersMono:
        case DeviceEntry::kAllOrdersF64NarrowMono:
        case DeviceEntry::kAllOrdersF64NarrowOrdersMono:
        // The double lane's rational route, whose cut this build derives,
        // uploads and reads.
        case DeviceEntry::kAllOrdersF64Rat:
        case DeviceEntry::kAllOrdersF64RatHorner:
        case DeviceEntry::kAllOrdersF64OrdersRat:
        case DeviceEntry::kAllOrdersF64OrdersRatHorner:
        case DeviceEntry::kAllOrdersF64NarrowRat:
        case DeviceEntry::kAllOrdersF64NarrowRatHorner:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRat:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner:
        // The device-callable entries, whose rung is the one the caller made
        // resident and which refuse a call naming another. Their being served at
        // every rung is what makes each of them a row of each class.
        case DeviceEntry::kDeviceSingleF64:
        case DeviceEntry::kDeviceSingleF32:
        case DeviceEntry::kDeviceSingleF32Fast:
        case DeviceEntry::kDeviceSingleF16:
        case DeviceEntry::kDeviceAllOrdersF64:
        case DeviceEntry::kDeviceAllOrdersF32:
        case DeviceEntry::kDeviceAllOrdersF16:
        case DeviceEntry::kDeviceAllNF64:
        case DeviceEntry::kDeviceAllNF32:
        case DeviceEntry::kDeviceAllNF16:
        case DeviceEntry::kDeviceEachOrderF64:
        case DeviceEntry::kDeviceEachOrderF32:
        case DeviceEntry::kDeviceEachOrderF16:
            return DeviceRungServed(multiplier);

        // The device-callable rows of the partition and route axes. Each reaches
        // the option its launched row names and reads that row's tables, so the
        // rungs are the launched row's and are read from the launched row rather
        // than written a second time here: a rung whose cut this build does not
        // hold is missing for both, and the cut that lands for the launched row
        // is one these rows answer at without this switch being edited. The two
        // rows of an option therefore cannot come apart about the rung axis, and
        // a row of either that named a rung the other does not is not a state
        // this file can hold.
        case DeviceEntry::kDeviceAllOrdersF64Narrow:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64Narrow, multiplier);
        case DeviceEntry::kDeviceAllOrdersF64NarrowMono:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64NarrowMono, multiplier);
        case DeviceEntry::kDeviceAllOrdersF64Rat:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64Rat, multiplier);
        case DeviceEntry::kDeviceAllOrdersF64RatHorner:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64RatHorner, multiplier);
        case DeviceEntry::kDeviceAllOrdersF64NarrowRat:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64NarrowRat, multiplier);
        case DeviceEntry::kDeviceAllOrdersF64NarrowRatHorner:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64NarrowRatHorner, multiplier);
        case DeviceEntry::kDeviceAllOrdersF64Uniform:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64Uniform, multiplier);
        case DeviceEntry::kDeviceAllOrdersF64UniformHorner:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64UniformHorner, multiplier);
        case DeviceEntry::kDeviceAllOrdersF32Narrow:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32Narrow, multiplier);
        case DeviceEntry::kDeviceAllOrdersF32NarrowMono:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32NarrowMono, multiplier);
        case DeviceEntry::kDeviceAllOrdersF32Rat:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32Rat, multiplier);
        case DeviceEntry::kDeviceAllOrdersF32RatHorner:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32RatHorner, multiplier);
        case DeviceEntry::kDeviceAllOrdersF32NarrowRat:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32NarrowRat, multiplier);
        case DeviceEntry::kDeviceAllOrdersF32NarrowRatHorner:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32NarrowRatHorner, multiplier);
        case DeviceEntry::kDeviceAllOrdersF32Uniform:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32Uniform, multiplier);
        case DeviceEntry::kDeviceAllOrdersF32UniformHorner:
            return DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32UniformHorner, multiplier);
    }

    // An enumerator no arm above names, which the check below turns into a
    // compile error. Nothing here serves an option whose rung axis is unstated.
    return false;
}

/// Whether the switch above states a rung axis for every enumerator of
/// \c DeviceEntry, read over the enumeration's own count.
///
/// The compilation of this header is what keeps the two in step: an enumerator
/// added to \c DeviceEntry and not named above falls through that switch's last
/// return, this predicate sees it, and the assertion below stops the build. A
/// new option therefore cannot arrive in the space without a statement of which
/// rungs it answers at — the omission is the error, rather than a default arm
/// quietly reporting it served everywhere.
constexpr bool DeviceEntriesAllStateTheirRungs() noexcept {
    for (int i = 0; i < static_cast<int>(DeviceEntry::kCount); ++i)
    {
        if (!DeviceEntryServedAtRung(static_cast<DeviceEntry>(i), kBoysFullAccuracyMultiplier))
        {
            return false;
        }
    }

    return true;
}

// The rung the check above reads every entry at, which every entry of this lane
// serves: the full-accuracy multiplier is the first rung of the table.
static_assert(DeviceRungServed(kBoysFullAccuracyMultiplier),
              "the full-accuracy multiplier is the rung every entry of this lane serves, and the "
              "table above does not hold it");

static_assert(DeviceEntriesAllStateTheirRungs(),
              "an enumerator of DeviceEntry states no rung axis: name it in "
              "DeviceEntryServedAtRung (boys_cuda_options.hpp) with the rungs its entry answers at");

/// The rungs a row of this space is served at, as a mask: the entry's own rungs
/// where the build serves the option, and none where it does not.
///
/// Derived and not listed. The rows of the space are initializers that name
/// their entry and whether the build serves it, and everything else about a
/// row's rungs follows from those two: a row whose entry answers at one rung is
/// served at that one, and a row no build serves is served at no rung, whatever
/// its entry's axis states. A row that carried its own copy of that answer would
/// be a second statement of a fact this header already holds, and the two could
/// come apart without anything failing.
///
/// \param entry the row's entry
/// \param built whether this build serves the option
///
/// \returns the mask of rungs that row is served at
///
/// \ingroup boys
constexpr DeviceRungMask DeviceServedRungMask(DeviceEntry entry, bool built) noexcept {
    if (!built)
    {
        return 0;
    }

    DeviceRungMask mask = 0;

    for (std::size_t i = 0; i < kDeviceRungCount; ++i)
    {
        if (DeviceEntryServedAtRung(entry, kDeviceRungs[i]))
        {
            mask |= DeviceRungMask{1} << i;
        }
    }

    return mask;
}

/// The partition member an entry reads, as the name a report prints, or the
/// stated name of an entry that reads no member at all.
///
/// The space carries the partition axis for the rows that take it but not yet a
/// field for its member, so the entries that read a partition are told apart
/// here — once, beside the rows they are rows of — rather than by each reader of
/// the space in turn. The member is a property of the *entry* and not of the
/// axis a row states: the narrow partition's pieces are read by the rows whose
/// axis is the partition, by the rows whose axis is the scheme inside it and by
/// the rows whose axis is the route inside it, and all three read that one
/// partition. The same holds of the grid.
///
/// The last arm is not a member and says so. An entry whose row carries no
/// partition axis — the single entries, the shipped partition's own ladders, the
/// all-N and each-order shapes, and the device-callable generics, which read
/// whatever partition their handle's tables name — reads no partition this
/// function could name, and answering one of the two members for it would state
/// a partition the entry does not read.
///
/// \param entry the entry of a row of this space
///
/// \returns the name of the partition it reads, or the statement that it reads
///          none
///
/// \ingroup boys
constexpr const char* DevicePartitionName(DeviceEntry entry) noexcept {
    switch (entry)
    {
        // The uniform grid: one table of equal intervals over [0, kFlatHi), the
        // double lane's and the float lane's. The scheme and the packing axis are
        // choices inside the route, so a row of either is still a row of that
        // partition.
        case DeviceEntry::kAllOrdersF64Uniform:
        case DeviceEntry::kAllOrdersF64UniformHorner:
        case DeviceEntry::kAllOrdersF64OrdersUniform:
        case DeviceEntry::kAllOrdersF64OrdersUniformHorner:
        case DeviceEntry::kAllOrdersF32Uniform:
        case DeviceEntry::kAllOrdersF32UniformHorner:
        case DeviceEntry::kAllOrdersF32OrdersUniform:
        case DeviceEntry::kAllOrdersF32OrdersUniformHorner:
        case DeviceEntry::kDeviceAllOrdersF64Uniform:
        case DeviceEntry::kDeviceAllOrdersF64UniformHorner:
        case DeviceEntry::kDeviceAllOrdersF32Uniform:
        case DeviceEntry::kDeviceAllOrdersF32UniformHorner:
            return "uniform";

        // The narrow partition, whose pieces are cut per order: the rows of the
        // partition axis, of the scheme axis inside it and of the route axis
        // inside it, on both packing axes and in both lanes, reachable by a
        // launch and from a caller's own kernel.
        case DeviceEntry::kAllOrdersF64Narrow:
        case DeviceEntry::kAllOrdersF64NarrowOrders:
        case DeviceEntry::kAllOrdersF64NarrowMono:
        case DeviceEntry::kAllOrdersF64NarrowOrdersMono:
        case DeviceEntry::kAllOrdersF64NarrowRat:
        case DeviceEntry::kAllOrdersF64NarrowRatHorner:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRat:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner:
        case DeviceEntry::kAllOrdersF32Narrow:
        case DeviceEntry::kAllOrdersF32NarrowMono:
        case DeviceEntry::kAllOrdersF32NarrowRat:
        case DeviceEntry::kAllOrdersF32NarrowRatHorner:
        case DeviceEntry::kAllOrdersF32NarrowOrders:
        case DeviceEntry::kAllOrdersF32NarrowOrdersMono:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRat:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF64Narrow:
        case DeviceEntry::kDeviceAllOrdersF64NarrowMono:
        case DeviceEntry::kDeviceAllOrdersF64NarrowRat:
        case DeviceEntry::kDeviceAllOrdersF64NarrowRatHorner:
        case DeviceEntry::kDeviceAllOrdersF32Narrow:
        case DeviceEntry::kDeviceAllOrdersF32NarrowMono:
        case DeviceEntry::kDeviceAllOrdersF32NarrowRat:
        case DeviceEntry::kDeviceAllOrdersF32NarrowRatHorner:
            return "narrow";

        default:
            return "no partition member";
    }
}

/// One row of the device option space: an option this surface offers, with
/// what a chooser needs to place it.
///
/// The rows are the space and not a list of the interesting ones: an entry that
/// exists but that a given build cannot serve is carried here with \c built
/// false and the reason, so its absence from a report is a stated refusal
/// rather than an omission. The build-time case this build has is the fp16
/// seam; whether a *device* is present is a run-time fact about a host and not
/// a property of the option, so a host report states that beside its figures
/// rather than here.
///
/// \ingroup boys
struct DeviceOptionInfo {
    DeviceEntry entry; ///< the option, and the report's row order
    const char* name; ///< the name a report prints and a selector names it by
    DeviceOptionGroup group; ///< how a caller reaches it
    DeviceOptionPrecision precision; ///< the precision it computes in
    DeviceOptionShape shape; ///< the output shape it answers with
    DeviceOptionQuestion question; ///< the question that shape answers
    DeviceOptionAxis axis; ///< the axis it varies, or kNone
    RegionBExp regionBExp; ///< the axis member it is; kAccurate for a row with no axis
    BoysDeviceLane lane; ///< the degree tables it reads, and whose seed it amplifies
    double bound; ///< the documented bound at m = 1, over the whole argument range
    const char* boundForm; ///< the documented form, as the multiplier m enters it
    bool built; ///< whether this build serves the option
    const char* refusedBecause; ///< why not, when \c built is false; nullptr otherwise

    /// The evaluation scheme the option's stored basis is summed in, and the
    /// route its pieces are cut from.
    ///
    /// Stated on every row, and not only on the rows that move them: the two
    /// name the family a row belongs to, so a chooser placing two rows side by
    /// side reads whether they are the same family off the rows themselves. The
    /// member of a row whose \c axis is kScheme is \c scheme; of one whose axis
    /// is kRoute, \c route.
    EvalScheme scheme = kDefaultEvalScheme;
    FitRoute route = kDefaultFitRoute; ///< the route its pieces are cut from

    /// The term of this row's documented bound that does not scale with the
    /// multiplier, as a figure.
    ///
    /// \c boundForm is the same statement in words and \c bound is the figure at
    /// m = 1, so the three agree: the bound a row documents at a rung m is
    /// \c m * (bound - boundFixed) + boundFixed, with the term a returned value
    /// decides dropped exactly as \c bound drops it. Zero for every form in this
    /// table but the fast region-B exponential's, whose form adds a seed
    /// contribution the truncation does not scale; a lane that documents another
    /// such term states it here.
    double boundFixed = 0.0;

    /// The rungs of \c kDeviceRungs this row is served at in this build, as a
    /// mask over them: bit i is the rung \c kDeviceRungs[i], \c kEveryDeviceRung
    /// is the whole axis and zero is no rung at all.
    ///
    /// The rung axis is the one a row carries into every class of the report,
    /// and a row that stated it nowhere was read as a row of every class: this
    /// row's own contract refuses eleven of the twelve rungs, and a report that
    /// printed the option as served without them said a caller could ask for
    /// arithmetic the entry does not answer with. A row states the set here, and
    /// a reader that needs the rungs themselves has \c kDeviceRungs in the same
    /// order the bits are in.
    ///
    /// Read from the row's own entry and from whether this build serves it,
    /// never listed beside them: \c DeviceEntryServedAtRung is the one statement
    /// of which rungs an entry answers at, and a row carrying a second copy of
    /// it would be a row that could disagree with the library it reports.
    ///
    /// Zero where this build does not serve the row, whatever its entry's axis
    /// states: a row no build serves is served at no rung, and a report that
    /// counted its cells as served would be counting arithmetic nothing here can
    /// run. The build's own refusal is stated beside this, in \c built and
    /// \c refusedBecause, so the two are read together and neither is inferred
    /// from the other.
    DeviceRungMask servedRungs = DeviceServedRungMask(entry, built);
};

/// The device option space this revision defines, one row per option, read from
/// the entries and the bounds in this header rather than listed beside them.
///
/// A report that enumerates *this* is a projection of the library and cannot
/// fall behind it: a precision, a shape or an axis member added to the surface
/// appears here as a row, and a row no build-time seam serves is carried with
/// the reason. What a chooser gets from a row is the entry, the precision it
/// computes in, the question it answers, the axis it varies where it has one,
/// the degree tables it reads, and the bound it is documented at — the same
/// facts the accuracy gate certifies the entry at, so the chooser and the
/// certifier read one table.
///
/// The bound is the figure at the full-accuracy multiplier, over the whole
/// argument range the entry serves; \c boundForm states the shape the
/// documentation gives it, in which the multiplier m enters. Where that form
/// carries a term a value decides — the fp16 rows' half ULPs — \c bound is the
/// figure with that term dropped, so a column of costs compares figures of one
/// kind, and the form beside it says what the dropped term is.
///
/// \returns the rows, in a fixed order: the enumerator order of DeviceEntry.
///
/// \ingroup boys
std::span<const DeviceOptionInfo> BoysDeviceOptions() noexcept;

} // namespace boys
