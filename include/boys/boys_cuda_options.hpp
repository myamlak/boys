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
/// and the two fp16 entries that return bf16 compute in fp16 as well, so the
/// precision named here is the arithmetic and not the return format.
///
/// \ingroup boys
enum class DeviceOptionPrecision : int {
    kFp64 = 0, ///< the double lane
    kFp32, ///< the float lane
    kFp16, ///< the half lane
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
    kPartition, ///< which cut of the regions the entry reads its fits from
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

/// Whether the CUDA lane serves a multiplier: whether it is one of
/// \c kDeviceRungs.
///
/// The comparison is exact and the rungs are all exactly representable, which
/// is the same match every entry of that lane makes between the multiplier a
/// call names and the one its degree tables were cut for. A value that is not a
/// rung is not served by any entry: the lane answers no arithmetic at it, so a
/// caller is told so rather than handed another rung's tables.
///
/// \param multiplier the accuracy multiplier m a call names
///
/// \returns whether the lane serves it
///
/// \ingroup boys
constexpr bool DeviceRungServed(double multiplier) noexcept {
    for (const double rung : kDeviceRungs)
    {
        if (rung == multiplier)
        {
            return true;
        }
    }

    return false;
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
