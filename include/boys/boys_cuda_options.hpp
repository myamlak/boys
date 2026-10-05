#pragma once

/// \file
/// The device option space: one row per option of the CUDA lane, with what a
/// chooser needs to place it.
///
/// A header of its own, reached from boys_cuda.hpp, because these rows are one table
/// with two readers: the host surface that declares the entries (BoysCuda, in
/// boys_cuda.hpp) and the device functions that implement them
/// (boys_cuda_device.hpp). The second is compiled by nvcc, in a translation unit that
/// has to stay free of the library's C++23 headers, so the rows live here and a
/// probe's rows and an accuracy gate's claims are projections of one report rather
/// than lists that drift from it.

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
    kAllOrdersF64UniformRat, ///< BoysCuda::AllOrdersF64UniformRat at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF64UniformRatHorner, ///< BoysCuda::AllOrdersF64UniformRatHorner at kHorner, launched
    kAllOrdersF64OrdersUniformRat, ///< BoysCuda::AllOrdersF64OrdersUniformRat at kSplitClenshaw
    kAllOrdersF64OrdersUniformRatHorner, ///< BoysCuda::AllOrdersF64OrdersUniformRatHorner at kHorner

    kAllOrdersF32Narrow, ///< BoysCuda::AllOrdersF32Narrow at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF32NarrowMono, ///< BoysCuda::AllOrdersF32NarrowMono at kHorner, launched
    kAllOrdersF32Uniform, ///< BoysCuda::AllOrdersF32Uniform at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF32UniformHorner, ///< BoysCuda::AllOrdersF32UniformHorner at kHorner, launched

    /// The float lane's rational route, on the coarsest partition here and the narrow
    /// one below — the same pair of partitions the double lane's route is carried on,
    /// and the same reason for two names per partition: the route's pair is stored in
    /// monomial form and read by Horner, so neither scheme name selects a second
    /// arithmetic and the two rows of a partition run one kernel.
    kAllOrdersF32Rat, ///< BoysCuda::AllOrdersF32Rat at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF32RatHorner, ///< BoysCuda::AllOrdersF32RatHorner at kHorner, launched
    kAllOrdersF32NarrowRat, ///< BoysCuda::AllOrdersF32NarrowRat at kSplitClenshaw, launched
    kAllOrdersF32NarrowRatHorner, ///< BoysCuda::AllOrdersF32NarrowRatHorner at kHorner, launched
    /// The rational route over the uniform grid: the same family and the same two
    /// names per pair as the rows above, over the grid's intervals instead of a
    /// derived partition's pieces, so neither scheme name reaches a second arithmetic.
    kAllOrdersF32UniformRat, ///< BoysCuda::AllOrdersF32UniformRat at kSplitClenshaw, launched
    kAllOrdersF32UniformRatHorner, ///< BoysCuda::AllOrdersF32UniformRatHorner at kHorner, launched

    /// The float lane's other packing axis: the counterpart of each row above that
    /// carries it, over the same stored table.
    ///
    /// The axis is a reading of region A and not a second fit: every order's own piece
    /// is located and its own fit summed, where the per-argument shape seeds the top
    /// order's fit and brings the lower orders back down a recurrence. The double
    /// lane's rows of the axis are the shape these mirror, one per partition and route,
    /// and the two agree row for row: each orders row sums the forms its own table
    /// carries, the narrow partition's and the rational route's included.
    ///
    /// The coarsest partition's row is one row for both scheme names, exactly as
    /// \c kAllOrdersF32 is. The uniform grid's are its per-argument rows' kernels: the
    /// grid's cells carry their own degree and block start, so the route's packing axis
    /// has one member, and the grid's rational member is the same shape — stored per
    /// interval at the interval's own pair and stored count.
    kAllOrdersF32Orders, ///< BoysCuda::AllOrdersF32Orders, launched
    kAllOrdersF32NarrowOrders, ///< BoysCuda::AllOrdersF32NarrowOrders, launched
    kAllOrdersF32NarrowOrdersMono, ///< BoysCuda::AllOrdersF32NarrowOrdersMono, launched
    kAllOrdersF32OrdersRat, ///< BoysCuda::AllOrdersF32OrdersRat at kSplitClenshaw, launched
    kAllOrdersF32OrdersRatHorner, ///< BoysCuda::AllOrdersF32OrdersRatHorner at kHorner
    kAllOrdersF32NarrowOrdersRat, ///< BoysCuda::AllOrdersF32NarrowOrdersRat at kSplitClenshaw
    kAllOrdersF32NarrowOrdersRatHorner, ///< BoysCuda::AllOrdersF32NarrowOrdersRatHorner at kHorner
    kAllOrdersF32OrdersUniform, ///< BoysCuda::AllOrdersF32OrdersUniform, launched
    kAllOrdersF32OrdersUniformHorner, ///< BoysCuda::AllOrdersF32OrdersUniformHorner, launched
    kAllOrdersF32OrdersUniformRat, ///< BoysCuda::AllOrdersF32OrdersUniformRat, launched
    kAllOrdersF32OrdersUniformRatHorner, ///< BoysCuda::AllOrdersF32OrdersUniformRatHorner

    /// The half lane's ladder family: the float lane's rows above, entry for entry, over the same
    /// kernels' bodies with the fp16 store around them.
    ///
    /// The half lane's own definition is that it runs the float engine's arithmetic and stores
    /// what that engine returns, so a row of it is not a second arithmetic and its figure is not
    /// the float lane's: every one of these is the float row of the same name at the same tables,
    /// the same partitions, the same seed lane and the same route, with the returned value
    /// rounded to the half format. What a chooser places them by is that store, which is why
    /// their bound carries the half format's own term.
    ///
    /// The rows are the float lane's, in that lane's order, so the two blocks read as one table
    /// with an fp16 name beside each f32 one: the narrow partition and the grid in both bases,
    /// the rational route's three pairs, and then the same families on the orders reading of
    /// region A.
    kAllOrdersF16Narrow, ///< BoysCuda::AllOrdersF16Narrow at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF16NarrowMono, ///< BoysCuda::AllOrdersF16NarrowMono at kHorner, launched
    kAllOrdersF16Uniform, ///< BoysCuda::AllOrdersF16Uniform at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF16UniformHorner, ///< BoysCuda::AllOrdersF16UniformHorner at kHorner, launched
    kAllOrdersF16Rat, ///< BoysCuda::AllOrdersF16Rat at EvalScheme::kSplitClenshaw, launched
    kAllOrdersF16RatHorner, ///< BoysCuda::AllOrdersF16RatHorner at kHorner, launched
    kAllOrdersF16NarrowRat, ///< BoysCuda::AllOrdersF16NarrowRat at kSplitClenshaw, launched
    kAllOrdersF16NarrowRatHorner, ///< BoysCuda::AllOrdersF16NarrowRatHorner at kHorner, launched
    kAllOrdersF16UniformRat, ///< BoysCuda::AllOrdersF16UniformRat at kSplitClenshaw, launched
    kAllOrdersF16UniformRatHorner, ///< BoysCuda::AllOrdersF16UniformRatHorner at kHorner, launched

    /// The half lane's other packing axis, the counterpart of each row above that carries it.
    ///
    /// The axis is the float lane's and these rows are that lane's rows in the fp16 store: the
    /// coarsest partition's row is one row for both scheme names as \c kAllOrdersF16 is, and the
    /// grid's two rows are its per-argument rows' kernels for the reason the float block states.
    kAllOrdersF16Orders, ///< BoysCuda::AllOrdersF16Orders, launched
    kAllOrdersF16NarrowOrders, ///< BoysCuda::AllOrdersF16NarrowOrders, launched
    kAllOrdersF16NarrowOrdersMono, ///< BoysCuda::AllOrdersF16NarrowOrdersMono, launched
    kAllOrdersF16OrdersRat, ///< BoysCuda::AllOrdersF16OrdersRat at kSplitClenshaw, launched
    kAllOrdersF16OrdersRatHorner, ///< BoysCuda::AllOrdersF16OrdersRatHorner at kHorner
    kAllOrdersF16NarrowOrdersRat, ///< BoysCuda::AllOrdersF16NarrowOrdersRat at kSplitClenshaw
    kAllOrdersF16NarrowOrdersRatHorner, ///< BoysCuda::AllOrdersF16NarrowOrdersRatHorner at kHorner
    kAllOrdersF16OrdersUniform, ///< BoysCuda::AllOrdersF16OrdersUniform, launched
    kAllOrdersF16OrdersUniformHorner, ///< BoysCuda::AllOrdersF16OrdersUniformHorner, launched
    kAllOrdersF16OrdersUniformRat, ///< BoysCuda::AllOrdersF16OrdersUniformRat, launched
    kAllOrdersF16OrdersUniformRatHorner, ///< BoysCuda::AllOrdersF16OrdersUniformRatHorner

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
    kDeviceAllOrdersF64UniformRat, ///< BoysDeviceAllOrdersF64UniformRat, inside the caller's kernel
    kDeviceAllOrdersF64UniformRatHorner, ///< BoysDeviceAllOrdersF64UniformRatHorner, in-kernel

    kDeviceAllOrdersF32Narrow, ///< BoysDeviceAllOrdersF32Narrow, inside the caller's kernel
    kDeviceAllOrdersF32NarrowMono, ///< BoysDeviceAllOrdersF32NarrowMono, in-kernel
    kDeviceAllOrdersF32Rat, ///< BoysDeviceAllOrdersF32Rat, inside the caller's kernel
    kDeviceAllOrdersF32RatHorner, ///< BoysDeviceAllOrdersF32RatHorner, in-kernel
    kDeviceAllOrdersF32NarrowRat, ///< BoysDeviceAllOrdersF32NarrowRat, in-kernel
    kDeviceAllOrdersF32NarrowRatHorner, ///< BoysDeviceAllOrdersF32NarrowRatHorner, in-kernel
    kDeviceAllOrdersF32Uniform, ///< BoysDeviceAllOrdersF32Uniform, inside the caller's kernel
    kDeviceAllOrdersF32UniformHorner, ///< BoysDeviceAllOrdersF32UniformHorner, in-kernel
    kDeviceAllOrdersF32UniformRat, ///< BoysDeviceAllOrdersF32UniformRat, inside the caller's kernel
    kDeviceAllOrdersF32UniformRatHorner, ///< BoysDeviceAllOrdersF32UniformRatHorner, in-kernel

    /// The rows appended after the block above: the combinations the classes already
    /// carry a kernel and a launcher for, whose option row was not written.
    ///
    /// Their enumerators are appended and never inserted, so every value above keeps
    /// the number it had: a row is reached by its name, and an ordinal that moved
    /// would move under a caller holding one. The consequence is that the row order
    /// this enumeration states is no longer the group order the header's preamble
    /// describes for these rows alone — a launched row appended after the
    /// device-callable block — which is the price of keeping the block above stable
    /// and is the reason each row below prints its group in its own right.
    ///
    /// Each is the counterpart of a row the same shape already has in another lane or
    /// at another member of an axis, over a body the lane already runs: the monomial
    /// Summation of the coarsest partition's ladder and of its per-order reading in
    /// the float and half lanes, whose double lane counterparts are
    /// \c kAllOrdersF64Mono and \c kAllOrdersF64OrdersMono, and the half lane's fast
    /// region-B reading of the single shape, whose counterpart is
    /// \c kSingleF32Fast.
    kAllOrdersF32Mono, ///< BoysCuda::AllOrdersF32Mono at EvalScheme::kHorner, launched
    kAllOrdersF32OrdersMono, ///< BoysCuda::AllOrdersF32OrdersMono at kHorner, launched
    kAllOrdersF16Mono, ///< BoysCuda::AllOrdersF16Mono at EvalScheme::kHorner, launched
    kAllOrdersF16OrdersMono, ///< BoysCuda::AllOrdersF16OrdersMono at kHorner, launched
    /// The orders reading of every partition, and the half lane's rows of the whole
    /// ladder family, reachable from a caller's own kernel.
    ///
    /// Each is the same option as the launched row named beside it in the report — one
    /// arithmetic, two ways to reach it — so a chooser reading the two reads one
    /// partition, one route, one scheme and one bound.
    ///
    /// The packing axis is the one axis of this block no device-callable row carried
    /// before it: the entries above read the ladder, and these read the per-order
    /// packing of the same pieces. The half lane's rows of the ladder family are the
    /// float lane's with this lane's store around them, which is this lane's own
    /// definition (src/boys_cuda.cu, the fp16 kernels), and its fast single is the
    /// float lane's fast reading in that store.
    kDeviceAllOrdersF64Orders, ///< BoysDeviceAllOrdersF64Orders, inside the caller's kernel
    kDeviceAllOrdersF64NarrowOrders, ///< BoysDeviceAllOrdersF64NarrowOrders, in-kernel
    kDeviceAllOrdersF64NarrowOrdersMono, ///< BoysDeviceAllOrdersF64NarrowOrdersMono, in-kernel
    kDeviceAllOrdersF64OrdersRat, ///< BoysDeviceAllOrdersF64OrdersRat, in-kernel
    kDeviceAllOrdersF64NarrowOrdersRat, ///< BoysDeviceAllOrdersF64NarrowOrdersRat, in-kernel
    kDeviceAllOrdersF64OrdersRatHorner, ///< BoysDeviceAllOrdersF64OrdersRatHorner, in-kernel
    kDeviceAllOrdersF64NarrowOrdersRatHorner, ///< BoysDeviceAllOrdersF64NarrowOrdersRatHorner,
                                              ///< in-kernel
    kDeviceAllOrdersF32Orders, ///< BoysDeviceAllOrdersF32Orders, inside the caller's kernel
    kDeviceAllOrdersF32NarrowOrders, ///< BoysDeviceAllOrdersF32NarrowOrders, in-kernel
    kDeviceAllOrdersF32NarrowOrdersMono, ///< BoysDeviceAllOrdersF32NarrowOrdersMono, in-kernel
    kDeviceAllOrdersF32OrdersRat, ///< BoysDeviceAllOrdersF32OrdersRat, in-kernel
    kDeviceAllOrdersF32NarrowOrdersRat, ///< BoysDeviceAllOrdersF32NarrowOrdersRat, in-kernel
    kDeviceAllOrdersF32OrdersRatHorner, ///< BoysDeviceAllOrdersF32OrdersRatHorner, in-kernel
    kDeviceAllOrdersF32NarrowOrdersRatHorner, ///< BoysDeviceAllOrdersF32NarrowOrdersRatHorner,
                                              ///< in-kernel
    kDeviceAllOrdersF16Orders, ///< BoysDeviceAllOrdersF16Orders, inside the caller's kernel
    kDeviceAllOrdersF16Narrow, ///< BoysDeviceAllOrdersF16Narrow, in-kernel
    kDeviceAllOrdersF16NarrowOrders, ///< BoysDeviceAllOrdersF16NarrowOrders, in-kernel
    kDeviceAllOrdersF16Uniform, ///< BoysDeviceAllOrdersF16Uniform, in-kernel
    kDeviceAllOrdersF16NarrowMono, ///< BoysDeviceAllOrdersF16NarrowMono, in-kernel
    kDeviceAllOrdersF16NarrowOrdersMono, ///< BoysDeviceAllOrdersF16NarrowOrdersMono, in-kernel
    kDeviceAllOrdersF16UniformHorner, ///< BoysDeviceAllOrdersF16UniformHorner, in-kernel
    kDeviceAllOrdersF16Rat, ///< BoysDeviceAllOrdersF16Rat, in-kernel
    kDeviceAllOrdersF16OrdersRat, ///< BoysDeviceAllOrdersF16OrdersRat, in-kernel
    kDeviceAllOrdersF16NarrowRat, ///< BoysDeviceAllOrdersF16NarrowRat, in-kernel
    kDeviceAllOrdersF16NarrowOrdersRat, ///< BoysDeviceAllOrdersF16NarrowOrdersRat, in-kernel
    kDeviceAllOrdersF16UniformRat, ///< BoysDeviceAllOrdersF16UniformRat, in-kernel
    kDeviceAllOrdersF16RatHorner, ///< BoysDeviceAllOrdersF16RatHorner, in-kernel
    kDeviceAllOrdersF16OrdersRatHorner, ///< BoysDeviceAllOrdersF16OrdersRatHorner, in-kernel
    kDeviceAllOrdersF16NarrowRatHorner, ///< BoysDeviceAllOrdersF16NarrowRatHorner, in-kernel
    kDeviceAllOrdersF16NarrowOrdersRatHorner, ///< BoysDeviceAllOrdersF16NarrowOrdersRatHorner,
                                              ///< in-kernel
    kDeviceAllOrdersF16UniformRatHorner, ///< BoysDeviceAllOrdersF16UniformRatHorner, in-kernel
    kDeviceSingleF16Fast, ///< BoysDeviceSingleF16Fast at RegionBExp::kFast, in-kernel
    kSingleF16Fast, ///< BoysCuda::SingleF16Fast at RegionBExp::kFast, launched


    /// The device-callable float-engine entries at \c RegionBExp::kFast, appended:
    /// each mirrors the entry its name extends, over the same body with the ladder's
    /// seed evaluated by the other member of the axis.
    kDeviceAllOrdersF32Fast, ///< BoysDeviceAllOrdersF32 at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32OrdersFast, ///< BoysDeviceAllOrdersF32Orders at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32RatFast, ///< BoysDeviceAllOrdersF32Rat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32OrdersRatFast, ///< BoysDeviceAllOrdersF32OrdersRat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32RatHornerFast, ///< BoysDeviceAllOrdersF32RatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32OrdersRatHornerFast, ///< BoysDeviceAllOrdersF32OrdersRatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32NarrowFast, ///< BoysDeviceAllOrdersF32Narrow at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32NarrowOrdersFast, ///< BoysDeviceAllOrdersF32NarrowOrders at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32NarrowMonoFast, ///< BoysDeviceAllOrdersF32NarrowMono at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32NarrowOrdersMonoFast, ///< BoysDeviceAllOrdersF32NarrowOrdersMono at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32NarrowRatFast, ///< BoysDeviceAllOrdersF32NarrowRat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32NarrowOrdersRatFast, ///< BoysDeviceAllOrdersF32NarrowOrdersRat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32NarrowRatHornerFast, ///< BoysDeviceAllOrdersF32NarrowRatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF32NarrowOrdersRatHornerFast, ///< BoysDeviceAllOrdersF32NarrowOrdersRatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllNF32Fast, ///< BoysDeviceAllNF32 at RegionBExp::kFast, in-kernel
    kDeviceEachOrderF32Fast, ///< BoysDeviceEachOrderF32 at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16Fast, ///< BoysDeviceAllOrdersF16 at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16OrdersFast, ///< BoysDeviceAllOrdersF16Orders at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16RatFast, ///< BoysDeviceAllOrdersF16Rat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16OrdersRatFast, ///< BoysDeviceAllOrdersF16OrdersRat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16RatHornerFast, ///< BoysDeviceAllOrdersF16RatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16OrdersRatHornerFast, ///< BoysDeviceAllOrdersF16OrdersRatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16NarrowFast, ///< BoysDeviceAllOrdersF16Narrow at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16NarrowOrdersFast, ///< BoysDeviceAllOrdersF16NarrowOrders at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16NarrowMonoFast, ///< BoysDeviceAllOrdersF16NarrowMono at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16NarrowOrdersMonoFast, ///< BoysDeviceAllOrdersF16NarrowOrdersMono at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16NarrowRatFast, ///< BoysDeviceAllOrdersF16NarrowRat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16NarrowOrdersRatFast, ///< BoysDeviceAllOrdersF16NarrowOrdersRat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16NarrowRatHornerFast, ///< BoysDeviceAllOrdersF16NarrowRatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF16NarrowOrdersRatHornerFast, ///< BoysDeviceAllOrdersF16NarrowOrdersRatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllNF16Fast, ///< BoysDeviceAllNF16 at RegionBExp::kFast, in-kernel
    kDeviceEachOrderF16Fast, ///< BoysDeviceEachOrderF16 at RegionBExp::kFast, in-kernel

    /// The each-order shape, launched: the same ladder the all-orders rows write into padded
    /// planes, written contiguously from the caller's own offset array. The shape's question is
    /// the all-orders one — \c DeviceOptionQuestion::kAllOrders — and what it adds is the
    /// layout: each argument's ladder stops at that argument's own top order, so a batch whose
    /// orders differ spends what it asks for instead of a plane per order.
    ///
    /// These are the rows the shape had none of: every other entry of it is reached from a
    /// caller's own kernel through a sink, and the launched route to the same ladder is
    /// \c BoysCuda::EachOrderF64 and its siblings, over the same body the all-orders kernels
    /// hand their lanes.
    kEachOrderF64, ///< BoysCuda::EachOrderF64 at RegionBExp::kAccurate, launched
    kEachOrderF64Fast, ///< BoysCuda::EachOrderF64 at RegionBExp::kFast, launched
    kEachOrderF32, ///< BoysCuda::EachOrderF32 at RegionBExp::kAccurate, launched
    kEachOrderF32Fast, ///< BoysCuda::EachOrderF32 at RegionBExp::kFast, launched
    kEachOrderF16, ///< BoysCuda::EachOrderF16 at RegionBExp::kAccurate, launched
    kEachOrderF16Fast, ///< BoysCuda::EachOrderF16 at RegionBExp::kFast, launched

    /// The double lane's ladders at the region-B exponential the rows above carry at
    /// \c RegionBExp::kAccurate.
    ///
    /// The axis is a coordinate of the option and not a second entry: each row below is the row of
    /// its own name at the other member of \c RegionBExp, over the same lane, the same pieces, the
    /// same seed lane and the same recurrence, with the region-B seed evaluated in that member's
    /// arithmetic. The two members carry a figure each and nothing substitutes one for the other.
    ///
    /// The rational route's pair is one member for both scheme names at this exponential, exactly
    /// as it is at the other (the rows above), so its two scheme rows below name one entry.
    kAllOrdersF64Fast, ///< BoysCuda::AllOrdersF64Fast at RegionBExp::kFast, launched
    kAllOrdersF64OrdersFast, ///< BoysCuda::AllOrdersF64OrdersFast at kFast, launched
    kAllOrdersF64NarrowFast, ///< BoysCuda::AllOrdersF64NarrowFast at kFast, launched
    kAllOrdersF64NarrowOrdersFast, ///< BoysCuda::AllOrdersF64NarrowOrdersFast at kFast, launched
    kAllOrdersF64MonoFast, ///< BoysCuda::AllOrdersF64MonoFast at kFast, launched
    kAllOrdersF64OrdersMonoFast, ///< BoysCuda::AllOrdersF64OrdersMonoFast at kFast, launched
    kAllOrdersF64NarrowMonoFast, ///< BoysCuda::AllOrdersF64NarrowMonoFast at kFast, launched
    kAllOrdersF64NarrowOrdersMonoFast, ///< BoysCuda::AllOrdersF64NarrowOrdersMonoFast at kFast
    kAllOrdersF64RatFast, ///< BoysCuda::AllOrdersF64RatFast at kFast, launched
    kAllOrdersF64RatHornerFast, ///< BoysCuda::AllOrdersF64RatFast at kFast, launched
    kAllOrdersF64OrdersRatFast, ///< BoysCuda::AllOrdersF64OrdersRatFast at kFast, launched
    kAllOrdersF64OrdersRatHornerFast, ///< BoysCuda::AllOrdersF64OrdersRatFast at kFast, launched
    kAllOrdersF64NarrowRatFast, ///< BoysCuda::AllOrdersF64NarrowRatFast at kFast, launched
    kAllOrdersF64NarrowRatHornerFast, ///< BoysCuda::AllOrdersF64NarrowRatFast at kFast, launched
    kAllOrdersF64NarrowOrdersRatFast, ///< BoysCuda::AllOrdersF64NarrowOrdersRatFast at kFast
    kAllOrdersF64NarrowOrdersRatHornerFast, ///< BoysCuda::AllOrdersF64NarrowOrdersRatFast at kFast


    /// The float and half lanes' ladders at the region-B exponential the rows above carry at
    /// \c RegionBExp::kAccurate, in the order of the f64 block above: the row of the same name
    /// at the other member of the axis, over the same lane, the same pieces, the same seed lane
    /// and the same recurrence, with the region-B seed evaluated in that member's arithmetic.
    /// The half lane's rows are the float lane's in this lane's store, as every half row is.
    kAllOrdersF32Fast, ///< BoysCuda::AllOrdersF32Fast at RegionBExp::kFast, launched
    kAllOrdersF32OrdersFast, ///< BoysCuda::AllOrdersF32OrdersFast at RegionBExp::kFast, launched
    kAllOrdersF32NarrowFast, ///< BoysCuda::AllOrdersF32NarrowFast at RegionBExp::kFast, launched
    kAllOrdersF32NarrowOrdersFast, ///< BoysCuda::AllOrdersF32NarrowOrdersFast at RegionBExp::kFast, launched
    kAllOrdersF32MonoFast, ///< BoysCuda::AllOrdersF32MonoFast at RegionBExp::kFast, launched
    kAllOrdersF32OrdersMonoFast, ///< BoysCuda::AllOrdersF32OrdersMonoFast at RegionBExp::kFast, launched
    kAllOrdersF32NarrowMonoFast, ///< BoysCuda::AllOrdersF32NarrowMonoFast at RegionBExp::kFast, launched
    kAllOrdersF32NarrowOrdersMonoFast, ///< BoysCuda::AllOrdersF32NarrowOrdersMonoFast at RegionBExp::kFast, launched
    kAllOrdersF32RatFast, ///< BoysCuda::AllOrdersF32RatFast at RegionBExp::kFast, launched
    kAllOrdersF32RatHornerFast, ///< BoysCuda::AllOrdersF32RatHornerFast at RegionBExp::kFast, launched
    kAllOrdersF32OrdersRatFast, ///< BoysCuda::AllOrdersF32OrdersRatFast at RegionBExp::kFast, launched
    kAllOrdersF32OrdersRatHornerFast, ///< BoysCuda::AllOrdersF32OrdersRatHornerFast at RegionBExp::kFast, launched
    kAllOrdersF32NarrowRatFast, ///< BoysCuda::AllOrdersF32NarrowRatFast at RegionBExp::kFast, launched
    kAllOrdersF32NarrowRatHornerFast, ///< BoysCuda::AllOrdersF32NarrowRatHornerFast at RegionBExp::kFast, launched
    kAllOrdersF32NarrowOrdersRatFast, ///< BoysCuda::AllOrdersF32NarrowOrdersRatFast at RegionBExp::kFast, launched
    kAllOrdersF32NarrowOrdersRatHornerFast, ///< BoysCuda::AllOrdersF32NarrowOrdersRatHornerFast at RegionBExp::kFast, launched
    kAllOrdersF16Fast, ///< BoysCuda::AllOrdersF16Fast at RegionBExp::kFast, launched
    kAllOrdersF16OrdersFast, ///< BoysCuda::AllOrdersF16OrdersFast at RegionBExp::kFast, launched
    kAllOrdersF16NarrowFast, ///< BoysCuda::AllOrdersF16NarrowFast at RegionBExp::kFast, launched
    kAllOrdersF16NarrowOrdersFast, ///< BoysCuda::AllOrdersF16NarrowOrdersFast at RegionBExp::kFast, launched
    kAllOrdersF16MonoFast, ///< BoysCuda::AllOrdersF16MonoFast at RegionBExp::kFast, launched
    kAllOrdersF16OrdersMonoFast, ///< BoysCuda::AllOrdersF16OrdersMonoFast at RegionBExp::kFast, launched
    kAllOrdersF16NarrowMonoFast, ///< BoysCuda::AllOrdersF16NarrowMonoFast at RegionBExp::kFast, launched
    kAllOrdersF16NarrowOrdersMonoFast, ///< BoysCuda::AllOrdersF16NarrowOrdersMonoFast at RegionBExp::kFast, launched
    kAllOrdersF16RatFast, ///< BoysCuda::AllOrdersF16RatFast at RegionBExp::kFast, launched
    kAllOrdersF16RatHornerFast, ///< BoysCuda::AllOrdersF16RatHornerFast at RegionBExp::kFast, launched
    kAllOrdersF16OrdersRatFast, ///< BoysCuda::AllOrdersF16OrdersRatFast at RegionBExp::kFast, launched
    kAllOrdersF16OrdersRatHornerFast, ///< BoysCuda::AllOrdersF16OrdersRatHornerFast at RegionBExp::kFast, launched
    kAllOrdersF16NarrowRatFast, ///< BoysCuda::AllOrdersF16NarrowRatFast at RegionBExp::kFast, launched
    kAllOrdersF16NarrowRatHornerFast, ///< BoysCuda::AllOrdersF16NarrowRatHornerFast at RegionBExp::kFast, launched
    kAllOrdersF16NarrowOrdersRatFast, ///< BoysCuda::AllOrdersF16NarrowOrdersRatFast at RegionBExp::kFast, launched
    kAllOrdersF16NarrowOrdersRatHornerFast, ///< BoysCuda::AllOrdersF16NarrowOrdersRatHornerFast at RegionBExp::kFast, launched

    /// The single-order and all-N shapes at the second member of the region-B axis, on the lanes
    /// whose classes carry it: the entry of the same name beside the row above at the other
    /// exponential, over the same tables and the same seed lane. The half lane's all-N row is the
    /// float lane's in this lane's store and carries that lane's figure.
    kSingleF64Fast, ///< BoysCuda::SingleF64Fast at RegionBExp::kFast, launched
    kAllNF64Fast, ///< BoysCuda::AllNF64Fast at RegionBExp::kFast, launched
    kAllNF32Fast, ///< BoysCuda::AllNF32Fast at RegionBExp::kFast, launched
    kAllNF16Fast, ///< BoysCuda::AllNF16Fast at RegionBExp::kFast, launched

    /// The double lane's other shapes at the second member of the region-B axis, reachable from a
    /// caller's own kernel: the entry of the same name beside the row above at the other
    /// exponential, over the same tables and the same lane. Each surface takes that axis as a
    /// template argument (boys_cuda_device.hpp), so the row books the surface it runs.
    kDeviceSingleF64Fast, ///< BoysDeviceSingleF64 at RegionBExp::kFast, in-kernel
    kDeviceAllNF64Fast, ///< BoysDeviceAllNF64 at RegionBExp::kFast, in-kernel
    kDeviceEachOrderF64Fast, ///< BoysDeviceEachOrderF64 at RegionBExp::kFast, in-kernel

    /// The double lane's all-orders ladders at the second member of the region-B axis,
    /// reachable from a caller's own kernel: each is the row of the same name beside the
    /// block above at the other exponential, over the same tables and the same device
    /// functions, which take that axis as a template argument (boys_cuda_device.hpp).
    kDeviceAllOrdersF64Fast, ///< BoysDeviceAllOrdersF64 at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64OrdersFast, ///< BoysDeviceAllOrdersF64Orders at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64NarrowFast, ///< BoysDeviceAllOrdersF64Narrow at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64NarrowOrdersFast, ///< BoysDeviceAllOrdersF64NarrowOrders at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64NarrowMonoFast, ///< BoysDeviceAllOrdersF64NarrowMono at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64NarrowOrdersMonoFast, ///< BoysDeviceAllOrdersF64NarrowOrdersMono at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64RatFast, ///< BoysDeviceAllOrdersF64Rat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64RatHornerFast, ///< BoysDeviceAllOrdersF64RatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64OrdersRatFast, ///< BoysDeviceAllOrdersF64OrdersRat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64OrdersRatHornerFast, ///< BoysDeviceAllOrdersF64OrdersRatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64NarrowRatFast, ///< BoysDeviceAllOrdersF64NarrowRat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64NarrowRatHornerFast, ///< BoysDeviceAllOrdersF64NarrowRatHorner at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64NarrowOrdersRatFast, ///< BoysDeviceAllOrdersF64NarrowOrdersRat at RegionBExp::kFast, in-kernel
    kDeviceAllOrdersF64NarrowOrdersRatHornerFast, ///< BoysDeviceAllOrdersF64NarrowOrdersRatHorner at RegionBExp::kFast, in-kernel

    kCount, ///< rows this report defines; one past the last
};

/// How a caller reaches an option: through a call of this library's that
/// queues a kernel, or through a device function a caller's own kernel calls.
///
/// The two are not comparable as costs — a launched row's figure includes the launch
/// and a device-callable one's is measured by subtraction against the same kernel with
/// the call removed — so a cost report never ranks across them.
///
/// \ingroup boys
enum class DeviceOptionGroup : int {
    kLaunched = 0, ///< an entry of BoysCuda, queued by this library
    kDeviceCallable, ///< a function of boys_cuda_device.hpp, inlined into the caller's kernel

    kCount, ///< groups this report defines; one past the last
};

/// The precision an option computes in: the choice a caller has already made
/// from the accuracy their calculation needs.
///
/// A bound never trades one of these for another — a row of a looser bound is a faster
/// way to compute the precision it names, not a different precision — and the two half
/// formats share one member, because both round a 32-bit engine's result to their
/// format at the boundary instead of computing in it. That is the CPU lane's own
/// reading of the same axis, whose \c Precision::kFp16 is "the fp16 and bfloat16
/// entries", so the member means one thing on both lanes: the half lane, whose
/// arithmetic is the float lane's and whose figure carries a term of the format it
/// stores.
///
/// **This build serves that lane's fp16 entries and not its bfloat16 ones.** A bf16
/// row here would be a row over this lane's existing tables rather than new arithmetic
/// — the same float engine's value stored into bf16 — but no such row exists yet.
/// Closing it is a piece of its own: the launched entries, their device-callable
/// siblings, and the gate's half-precision block for the second format. Until that
/// lands this member is served for fp16 alone, which is unbuilt work and not a
/// property of the lane.
///
/// \ingroup boys
enum class DeviceOptionPrecision : int {
    kFp64 = 0, ///< the double lane
    kFp32, ///< the float lane
    kFp16, ///< the half lane: the fp16 entries this build serves, and the bfloat16 ones it owes

    kCount, ///< precisions this report defines; one past the last
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

    kCount, ///< shapes this report defines; one past the last
};

/// The question an option answers, which is the shape with the sink/array
/// difference folded out: the two are one question and not two.
///
/// \ingroup boys
enum class DeviceOptionQuestion : int {
    kSingle = 0, ///< one value per argument
    kAllOrders, ///< a ladder per argument, to that argument's own order
    kAllN, ///< one common ladder for the whole batch

    kCount, ///< questions this report defines; one past the last
};

/// The axis an option varies, where it varies one. Every other difference
/// between two options of this surface is the shape (a different entry).
///
/// \ingroup boys
enum class DeviceOptionAxis : int {
    kNone = 0, ///< the option is the entry, with no second choice in it
    kRegionBExp, ///< which region-B exponential the entry's recurrence seeds with
    /// Which cut of the domain the entry reads its fits from: the coarsest
    /// pieces, the narrow partition's per-order pieces, or the uniform grid,
    /// which covers [0, kFlatHi) in one table of equal intervals and leaves
    /// the range above it to the arm every route ends in. The route the CPU
    /// lane names \c FitGranularity::kUniform is this member of this axis.
    kPartition,
    /// How region A's fits are read; its member is \c packing, and the readings
    /// it names are \c DevicePacking's.
    kPacking,
    kScheme, ///< which basis the entry sums its stored fits in; its member is \c scheme
    kRoute, ///< which family of fit the entry's pieces are; its member is \c route
    /// How the entry's ladder steps divide; its member is \c division, and the
    /// forms it names are \c DivisionForm's - the same three the host's calls are
    /// ranked on (boys/accuracy.hpp), because the recurrence is one arithmetic on
    /// both sides of the device boundary.
    kDivision,
};

/// The cut of the domain \c DevicePartitionOf answers for an enumerator its switch
/// has not been taught: not a cut and never a row's value.
///
/// FitGranularity is the host's enumeration and carries no sentinel of its own, so
/// the out-of-enum value is stated here, beside the one switch that can answer it,
/// and the assertion below that switch turns it into a compile error. The packing
/// axis, whose type this header owns, states its own sentinel as a member instead;
/// this constant is that member's counterpart for a type this header does not own.
inline constexpr FitGranularity kUnstatedPartition = static_cast<FitGranularity>(0xFF);

/// The cut of the domain an entry reads its fits from, read from that entry's own
/// implementation and stated once, as the row unit's own field.
///
/// **The member is the host's FitGranularity and not a device enumeration of its
/// own.** The CPU lane names the same three cuts, and this axis is one axis on both
/// sides of the device boundary: the axis's own entry states that the route the CPU
/// lane names \c FitGranularity::kUniform is this member, and the coarsest cut and
/// the narrow one are the same three tables the CPU's rungs read. Where the two
/// sides name one thing, they name it once, with one type — a second enumeration
/// would be a second statement that could come to disagree with the first.
///
/// The entries that read a partition are told apart here — once, beside the rows
/// they are rows of — and the row states the answer rather than a report deriving
/// it: \c DeviceOptionInfo::partition is this function's answer, so a report and a
/// chooser read one statement of which cut an entry reads. The cut is a
/// property of the *entry* and not of the axis a row states: the narrow
/// partition's pieces are read by the rows whose axis is the partition, by the
/// rows whose axis is the scheme inside it and by the rows whose axis is the route
/// inside it, and all three read that one partition. The same holds of the grid.
///
/// **Every entry reads one of the three cuts, so every arm names one.** The rows
/// whose partition axis is kNone are not entries that read no partition: the single
/// entries, the coarsest partition's own ladders, the all-N and each-order shapes
/// and the device-callable generics all read the shipped cut — the device-callable
/// generics read the coarsest slices of the handle their caller was given
/// (boys_cuda_device.hpp, TableLane64) — and the arm below says so rather than
/// answering a statement about the axis's membership in a field that states a cut.
///
/// Every enumerator of \c DeviceEntry is named below, and the switch has no default
/// arm: an option added to the enumeration without a statement of which cut it
/// reads is a compile error rather than an option silently reported as reading one.
/// That sentence is load-bearing here and not a formality — the answer this function
/// gives for an unstated entry, \c kUnstatedPartition, carries the same out-of-enum
/// value the packing axis's \c kUnstated does, so the assertion below can tell an
/// omission from an answer.
///
/// \param entry the entry of a row of this space
///
/// \returns the cut it reads, as the host's FitGranularity; \c kUnstatedPartition
///          only for an enumerator no arm above names, which the check below turns
///          into a compile error
///
/// \ingroup boys
constexpr FitGranularity DevicePartitionOf(DeviceEntry entry) noexcept {
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
        case DeviceEntry::kAllOrdersF64UniformRat:
        case DeviceEntry::kAllOrdersF64UniformRatHorner:
        case DeviceEntry::kAllOrdersF64OrdersUniformRat:
        case DeviceEntry::kAllOrdersF64OrdersUniformRatHorner:
        case DeviceEntry::kAllOrdersF32UniformRat:
        case DeviceEntry::kAllOrdersF32UniformRatHorner:
        case DeviceEntry::kAllOrdersF32OrdersUniformRat:
        case DeviceEntry::kAllOrdersF32OrdersUniformRatHorner:
        case DeviceEntry::kDeviceAllOrdersF64UniformRat:
        case DeviceEntry::kDeviceAllOrdersF64UniformRatHorner:
        case DeviceEntry::kDeviceAllOrdersF32UniformRat:
        case DeviceEntry::kDeviceAllOrdersF32UniformRatHorner:
        // The half lane's grid rows, which read the float lane's own intervals: this lane runs
        // that lane's bodies and its tables are the float lane's (src/boys_cuda.cu, the fp16
        // kernels behind BoysCuda::AllOrdersF16Uniform and its siblings).
        case DeviceEntry::kAllOrdersF16Uniform:
        case DeviceEntry::kAllOrdersF16UniformHorner:
        case DeviceEntry::kAllOrdersF16OrdersUniform:
        case DeviceEntry::kAllOrdersF16OrdersUniformHorner:
        case DeviceEntry::kAllOrdersF16UniformRat:
        case DeviceEntry::kAllOrdersF16UniformRatHorner:
        case DeviceEntry::kAllOrdersF16OrdersUniformRat:
        case DeviceEntry::kAllOrdersF16OrdersUniformRatHorner:
        // The half lane's grid rows reachable from a caller's own kernel, which read the float
        // lane's own intervals.
        case DeviceEntry::kDeviceAllOrdersF16Uniform:
        case DeviceEntry::kDeviceAllOrdersF16UniformHorner:
        case DeviceEntry::kDeviceAllOrdersF16UniformRat:
        case DeviceEntry::kDeviceAllOrdersF16UniformRatHorner:
            return FitGranularity::kUniform;

        // The narrow partition, whose pieces are cut per order: the rows of the
        // partition axis, of the scheme axis inside it and of the route axis
        // inside it, on both packing axes and in both lanes, reachable by a
        // launch and from a caller's own kernel.
        case DeviceEntry::kAllOrdersF64Narrow:
        case DeviceEntry::kAllOrdersF64NarrowFast:
        case DeviceEntry::kAllOrdersF64NarrowOrders:
        case DeviceEntry::kAllOrdersF64NarrowOrdersFast:
        case DeviceEntry::kAllOrdersF64NarrowMono:
        case DeviceEntry::kAllOrdersF64NarrowMonoFast:
        case DeviceEntry::kAllOrdersF64NarrowOrdersMono:
        case DeviceEntry::kAllOrdersF64NarrowOrdersMonoFast:
        case DeviceEntry::kAllOrdersF64NarrowRat:
        case DeviceEntry::kAllOrdersF64NarrowRatFast:
        case DeviceEntry::kAllOrdersF64NarrowRatHorner:
        case DeviceEntry::kAllOrdersF64NarrowRatHornerFast:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRat:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRatFast:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRatHornerFast:
        case DeviceEntry::kAllOrdersF32Narrow:
        case DeviceEntry::kAllOrdersF32NarrowFast:
        case DeviceEntry::kAllOrdersF32NarrowMono:
        case DeviceEntry::kAllOrdersF32NarrowMonoFast:
        case DeviceEntry::kAllOrdersF32NarrowRat:
        case DeviceEntry::kAllOrdersF32NarrowRatFast:
        case DeviceEntry::kAllOrdersF32NarrowRatHorner:
        case DeviceEntry::kAllOrdersF32NarrowRatHornerFast:
        case DeviceEntry::kAllOrdersF32NarrowOrders:
        case DeviceEntry::kAllOrdersF32NarrowOrdersFast:
        case DeviceEntry::kAllOrdersF32NarrowOrdersMono:
        case DeviceEntry::kAllOrdersF32NarrowOrdersMonoFast:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRat:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRatFast:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF64Narrow:
        case DeviceEntry::kDeviceAllOrdersF64NarrowFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowMono:
        case DeviceEntry::kDeviceAllOrdersF64NarrowMonoFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowRat:
        case DeviceEntry::kDeviceAllOrdersF64NarrowRatFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowRatHorner:
        case DeviceEntry::kDeviceAllOrdersF64NarrowRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF32Narrow:
        case DeviceEntry::kDeviceAllOrdersF32NarrowFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowMono:
        case DeviceEntry::kDeviceAllOrdersF32NarrowMonoFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowRat:
        case DeviceEntry::kDeviceAllOrdersF32NarrowRatFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowRatHorner:
        case DeviceEntry::kDeviceAllOrdersF32NarrowRatHornerFast:
        // The half lane's rows of the same cut: this lane's pieces are the float lane's, so each
        // of these reads the float lane's narrow partition.
        case DeviceEntry::kAllOrdersF16Narrow:
        case DeviceEntry::kAllOrdersF16NarrowFast:
        case DeviceEntry::kAllOrdersF16NarrowMono:
        case DeviceEntry::kAllOrdersF16NarrowMonoFast:
        case DeviceEntry::kAllOrdersF16NarrowRat:
        case DeviceEntry::kAllOrdersF16NarrowRatFast:
        case DeviceEntry::kAllOrdersF16NarrowRatHorner:
        case DeviceEntry::kAllOrdersF16NarrowRatHornerFast:
        case DeviceEntry::kAllOrdersF16NarrowOrders:
        case DeviceEntry::kAllOrdersF16NarrowOrdersFast:
        case DeviceEntry::kAllOrdersF16NarrowOrdersMono:
        case DeviceEntry::kAllOrdersF16NarrowOrdersMonoFast:
        case DeviceEntry::kAllOrdersF16NarrowOrdersRat:
        case DeviceEntry::kAllOrdersF16NarrowOrdersRatFast:
        case DeviceEntry::kAllOrdersF16NarrowOrdersRatHorner:
        case DeviceEntry::kAllOrdersF16NarrowOrdersRatHornerFast:
        // The narrow partition's pieces reachable from a caller's own kernel, on both packing
        // axes, in both bases and on both routes.
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrders:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersMono:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersMonoFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRat:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrders:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersMono:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersMonoFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRat:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF16Narrow:
        case DeviceEntry::kDeviceAllOrdersF16NarrowFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrders:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowMono:
        case DeviceEntry::kDeviceAllOrdersF16NarrowMonoFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersMono:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersMonoFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowRat:
        case DeviceEntry::kDeviceAllOrdersF16NarrowRatFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRat:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowRatHorner:
        case DeviceEntry::kDeviceAllOrdersF16NarrowRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRatHornerFast:
            return FitGranularity::kNarrow;

        // The entries whose row carries no partition axis, and which read the shipped
        // cut: the single entries and the batch generics are the coarsest fit read whole,
        // the coarsest partition's own ladders are that cut's pieces, and the all-N and
        // each-order shapes seed the coarsest ladder at one top order. Naming kUniform or
        // kNarrow for any of these would state a cut the entry does not read; the shipped
        // cut is the one it does, and the device-callable generics reach it through the
        // handle they were given (boys_cuda_device.hpp, TableLane64).
        case DeviceEntry::kSingleF64:
        case DeviceEntry::kSingleF64Fast:
        case DeviceEntry::kSingleF32:
        case DeviceEntry::kSingleF32Fast:
        case DeviceEntry::kSingleF16:
        case DeviceEntry::kAllOrdersF64:
        case DeviceEntry::kAllOrdersF64Fast:
        case DeviceEntry::kAllOrdersF32:
        case DeviceEntry::kAllOrdersF32Fast:
        case DeviceEntry::kAllOrdersF16:
        case DeviceEntry::kAllOrdersF16Fast:
        // The coarsest partition's own ladders: the level ladder, the per-order reading of
        // it and the rational route over the same pieces, on both packing axes. The
        // narrow and uniform rows of these same shapes are the cases above.
        case DeviceEntry::kAllOrdersF64Orders:
        case DeviceEntry::kAllOrdersF64OrdersFast:
        case DeviceEntry::kAllOrdersF64Mono:
        case DeviceEntry::kAllOrdersF64MonoFast:
        case DeviceEntry::kAllOrdersF64OrdersMono:
        case DeviceEntry::kAllOrdersF64OrdersMonoFast:
        case DeviceEntry::kAllOrdersF64Rat:
        case DeviceEntry::kAllOrdersF64RatFast:
        case DeviceEntry::kAllOrdersF64RatHorner:
        case DeviceEntry::kAllOrdersF64RatHornerFast:
        case DeviceEntry::kAllOrdersF64OrdersRat:
        case DeviceEntry::kAllOrdersF64OrdersRatFast:
        case DeviceEntry::kAllOrdersF64OrdersRatHorner:
        case DeviceEntry::kAllOrdersF64OrdersRatHornerFast:
        case DeviceEntry::kAllOrdersF32Rat:
        case DeviceEntry::kAllOrdersF32RatFast:
        case DeviceEntry::kAllOrdersF32RatHorner:
        case DeviceEntry::kAllOrdersF32RatHornerFast:
        case DeviceEntry::kAllOrdersF32Orders:
        case DeviceEntry::kAllOrdersF32OrdersFast:
        case DeviceEntry::kAllOrdersF32OrdersRat:
        case DeviceEntry::kAllOrdersF32OrdersRatFast:
        case DeviceEntry::kAllOrdersF32OrdersRatHorner:
        case DeviceEntry::kAllOrdersF32OrdersRatHornerFast:
        // The coarsest rows appended for the combinations the lane already had a kernel for: the
        // float and half lanes' monomial pair beside the double lane's above, and the half lane's
        // fast single, which reads the same cut the lane's other single rows do.
        case DeviceEntry::kAllOrdersF32Mono:
        case DeviceEntry::kAllOrdersF32MonoFast:
        case DeviceEntry::kAllOrdersF32OrdersMono:
        case DeviceEntry::kAllOrdersF32OrdersMonoFast:
        case DeviceEntry::kAllOrdersF16Mono:
        case DeviceEntry::kAllOrdersF16MonoFast:
        case DeviceEntry::kAllOrdersF16OrdersMono:
        case DeviceEntry::kAllOrdersF16OrdersMonoFast:
        case DeviceEntry::kSingleF16Fast:
        // The all-N and each-order shapes, whose one top order is the shape and not a
        // partition, and the device-callable entries that mirror the rows above.
        case DeviceEntry::kAllNF64:
        case DeviceEntry::kAllNF64Fast:
        case DeviceEntry::kAllNF32:
        case DeviceEntry::kAllNF32Fast:
        case DeviceEntry::kAllNF16:
        case DeviceEntry::kAllNF16Fast:
        case DeviceEntry::kDeviceSingleF64:
        case DeviceEntry::kDeviceSingleF64Fast:
        case DeviceEntry::kDeviceSingleF32:
        case DeviceEntry::kDeviceSingleF32Fast:
        case DeviceEntry::kDeviceSingleF16:
        case DeviceEntry::kDeviceAllOrdersF64:
        case DeviceEntry::kDeviceAllOrdersF64Fast:
        case DeviceEntry::kDeviceAllOrdersF32:
        case DeviceEntry::kDeviceAllOrdersF32Fast:
        case DeviceEntry::kDeviceAllOrdersF16:
        case DeviceEntry::kDeviceAllOrdersF16Fast:
        case DeviceEntry::kDeviceAllNF64:
        case DeviceEntry::kDeviceAllNF64Fast:
        case DeviceEntry::kDeviceAllNF32:
        case DeviceEntry::kDeviceAllNF32Fast:
        case DeviceEntry::kDeviceAllNF16:
        case DeviceEntry::kDeviceAllNF16Fast:
        case DeviceEntry::kDeviceEachOrderF64:
        case DeviceEntry::kDeviceEachOrderF64Fast:
        case DeviceEntry::kDeviceEachOrderF32:
        case DeviceEntry::kDeviceEachOrderF32Fast:
        case DeviceEntry::kDeviceEachOrderF16:
        case DeviceEntry::kDeviceEachOrderF16Fast:
        case DeviceEntry::kDeviceAllOrdersF64Rat:
        case DeviceEntry::kDeviceAllOrdersF64RatFast:
        case DeviceEntry::kDeviceAllOrdersF64RatHorner:
        case DeviceEntry::kDeviceAllOrdersF64RatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF32Rat:
        case DeviceEntry::kDeviceAllOrdersF32RatFast:
        case DeviceEntry::kDeviceAllOrdersF32RatHorner:
        case DeviceEntry::kDeviceAllOrdersF32RatHornerFast:
        // The half lane's rows of the shipped cut: its coarsest ladder is the float lane's, read
        // whole or per order, in its Chebyshev and its rational family.
        case DeviceEntry::kAllOrdersF16Rat:
        case DeviceEntry::kAllOrdersF16RatFast:
        case DeviceEntry::kAllOrdersF16RatHorner:
        case DeviceEntry::kAllOrdersF16RatHornerFast:
        case DeviceEntry::kAllOrdersF16Orders:
        case DeviceEntry::kAllOrdersF16OrdersFast:
        case DeviceEntry::kAllOrdersF16OrdersRat:
        case DeviceEntry::kAllOrdersF16OrdersRatFast:
        case DeviceEntry::kAllOrdersF16OrdersRatHorner:
        case DeviceEntry::kAllOrdersF16OrdersRatHornerFast:
        // The shipped cut reachable from a caller's own kernel: the coarsest partition's own
        // ladders on its other packing axis, in every lane and on both routes.
        case DeviceEntry::kDeviceAllOrdersF64Orders:
        case DeviceEntry::kDeviceAllOrdersF64OrdersFast:
        case DeviceEntry::kDeviceAllOrdersF64OrdersRat:
        case DeviceEntry::kDeviceAllOrdersF64OrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF64OrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF64OrdersRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF32Orders:
        case DeviceEntry::kDeviceAllOrdersF32OrdersFast:
        case DeviceEntry::kDeviceAllOrdersF32OrdersRat:
        case DeviceEntry::kDeviceAllOrdersF32OrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF32OrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF32OrdersRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF16Orders:
        case DeviceEntry::kDeviceAllOrdersF16OrdersFast:
        case DeviceEntry::kDeviceAllOrdersF16Rat:
        case DeviceEntry::kDeviceAllOrdersF16RatFast:
        case DeviceEntry::kDeviceAllOrdersF16OrdersRat:
        case DeviceEntry::kDeviceAllOrdersF16OrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF16RatHorner:
        case DeviceEntry::kDeviceAllOrdersF16RatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF16OrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF16OrdersRatHornerFast:
        case DeviceEntry::kDeviceSingleF16Fast:
        // The launched each-order rows, whose ladder is the coarsest partition's ladder the
        // all-orders rows of the arm above read.
        case DeviceEntry::kEachOrderF64:
        case DeviceEntry::kEachOrderF64Fast:
        case DeviceEntry::kEachOrderF32:
        case DeviceEntry::kEachOrderF32Fast:
        case DeviceEntry::kEachOrderF16:
        case DeviceEntry::kEachOrderF16Fast:
            return FitGranularity::kCoarsest;

        // The sentinel one past the last row this report defines, and not a row a call
        // can name, so no partition member is owed for it. It is named rather than left
        // to a default arm: a default would swallow the next enumerator as quietly as it
        // swallows this one. Nothing else here catches that row: gcc and clang
        // warn for an unhandled enumerator through -Wswitch, but only while the
        // switch has no `default` arm. This tree builds on MSVC at /W4, which emits
        // nothing for one either way. The assertion below the switch is what fails
        // the build for a row this switch has not been taught, and on MSVC it is the
        // only thing that does.
        case DeviceEntry::kCount:
            break;
    }

    // An enumerator no arm above names, which the check below turns into a compile
    // error. Nothing here names a cut for an option that has not stated one.
    return kUnstatedPartition;
}

/// The name a report prints a cut of the domain under.
///
/// The strings are the host's: \c GranularityName (backend.hpp) prints the same
/// three cuts under these three names, and a device report that printed others
/// would be a second vocabulary for one axis. This function is the device
/// header's copy of that one because the header it stands in has to stay
/// compilable by nvcc, which backend.hpp is not.
///
/// \param partition the cut
///
/// \returns a string literal naming it: "shipped", "narrow" or "uniform"; \c
///          nullptr for a value outside the enumerators, which is not a cut and
///          has no name
///
/// \ingroup boys
constexpr const char* DevicePartitionName(FitGranularity partition) noexcept {
    switch (partition)
    {
        case FitGranularity::kCoarsest:
            return "shipped";
        case FitGranularity::kNarrow:
            return "narrow";
        case FitGranularity::kUniform:
            return "uniform";
    }

    return nullptr;
}

/// The name a report prints the cut an entry reads its fits from under, read
/// through the two functions above.
///
/// \param entry the entry of a row of this space
///
/// \returns the name of the cut it reads, or \c nullptr for an enumerator
///          \c DevicePartitionOf has not been taught
///
/// \ingroup boys
constexpr const char* DevicePartitionName(DeviceEntry entry) noexcept {
    return DevicePartitionName(DevicePartitionOf(entry));
}

/// Whether the switch above states a cut for every enumerator of
/// \c DeviceEntry, read over the enumeration's own count.
///
/// The compilation of this header is what keeps the two in step: an enumerator
/// added to \c DeviceEntry and not named above falls through that switch's last
/// return, this predicate sees it, and the assertion below stops the build. A
/// new option therefore cannot arrive in the space without a statement of which
/// cut it reads — the omission is the error, rather than a default arm quietly
/// reporting it as reading one.
///
/// \returns true when every enumerator of \c DeviceEntry is named by the switch
///          above
constexpr bool DeviceEntriesAllNameTheirPartition() noexcept {
    for (int i = 0; i < static_cast<int>(DeviceEntry::kCount); ++i)
    {
        if (DevicePartitionOf(static_cast<DeviceEntry>(i)) == kUnstatedPartition)
        {
            return false;
        }
    }

    return true;
}

static_assert(DeviceEntriesAllNameTheirPartition(),
              "an enumerator of DeviceEntry names no partition member: name it in "
              "DevicePartitionOf (boys_cuda_options.hpp) as the cut its entry reads");

/// How an entry reads the fits of region A: as one ladder, seeded at the top
/// order and stepped down, or as one fit per order — or neither, for the shape
/// that asks for one order and has no ladder in it.
///
/// The axis is the one \c DeviceOptionAxis::kPacking names, and it covers region A
/// and nothing else: past the region's edge every shape runs the certified body its
/// own declaration names, which is the interval a row of this axis states its claim
/// over (boys_cuda_arithmetic.hpp, DeviceOrdersF64).
///
/// **It is this header's own enumeration, and the one axis whose member set the two
/// sides of the device boundary do not share.** The other four axes name the host's
/// types — \c FitRoute, \c EvalScheme, \c FitGranularity and \c DivisionForm — because
/// the members the host offers for them are the members the device rows run. This one
/// is not that case, and the member sets are what say so: the host's \c PackAxis names
/// the axis a packed lane packs and carries two members, where a single-order shape
/// has no second order to pack and fixes no reading at all, so this enumeration adds
/// \c kNotApplicable beside the two readings both sides name, and \c kUnstated as its
/// out-of-enum sentinel — two members \c PackAxis has none of. The device probe's seam
/// translates the readings the two do share into the host's cells, where that third
/// member has no cell of its own (src/boys_cuda_probe.cpp, SeamPackCell).
///
/// The split is therefore a decision and not an oversight: where the member sets
/// genuinely differ the device lane states its own, and where they do not it names the
/// host's type rather than a mirror of it.
///
/// \ingroup boys
enum class DevicePacking : int {
    /// The top order's own fit is seeded and every lower order is brought back
    /// down the recurrence, so one fit answer serves the whole ladder.
    kLadder = 0,
    /// Every order's own fit is located and summed where it lies, so no value
    /// comes down a recurrence from a higher order's piece.
    kPerOrder,
    /// The single-order shape: one order is asked for and one fit answers it, so
    /// the entry fixes no reading of this axis. Stated rather than left out — a
    /// row that said nothing here would be read as the ladder.
    kNotApplicable,

    /// Not an axis member and never a row's value: what \c DeviceEntryAxesOf
    /// answers for an enumerator its switch has not been taught, which the
    /// assertion below that function turns into a compile error. An option
    /// therefore cannot arrive in the space without a statement of the three axes
    /// it runs.
    kUnstated,
};

/// The form \c DeviceEntryAxesOf answers for the division axis of an enumerator its
/// switch has not been taught: not a form and never a row's value.
///
/// \c DivisionForm is the host's enumeration and carries no sentinel of its own, so the
/// out-of-enum value is stated here, beside the one switch that can answer it, as
/// \c kUnstatedPartition is stated beside the partition's. The assertion below that
/// switch turns the value into a compile error rather than a form a report prints.
inline constexpr DivisionForm kUnstatedDivisionForm = static_cast<DivisionForm>(0xFF);

/// The four axes an entry's arithmetic fixes: the family its fits are cut from,
/// the summation its stored coefficients are read by, its reading of region A, and
/// the form its ladder's divisions are performed in.
///
/// The constructor is the caller's rather than the compiler's so that an arm stating
/// fewer than all four is a compile error. A left-out member of an aggregate would be
/// value-initialized, and \c DivisionForm's first enumerator is \c kExactDivision: an
/// arm that forgot the form would state exact division for an entry that may run
/// another, which is a wrong answer a report prints rather than an error.
///
/// \ingroup boys
struct DeviceEntryAxes {
    FitRoute route; ///< the family its fits are of
    EvalScheme scheme; ///< the summation its stored coefficients are read by
    DevicePacking packing; ///< its reading of region A, or kNotApplicable
    DivisionForm division; ///< the form its ladder's divisions are performed in

    /// States an entry's four axes.
    ///
    /// \param r the family its fits are of
    /// \param s the summation its stored coefficients are read by
    /// \param p its reading of region A
    /// \param d the form its ladder's divisions are performed in
    constexpr DeviceEntryAxes(FitRoute r, EvalScheme s, DevicePacking p, DivisionForm d) noexcept
        : route(r), scheme(s), packing(p), division(d)
    {
    }
};

/// The axes one entry of the device option space runs, read from that entry's own
/// implementation: the lane object each kernel and device function of the entry names,
/// and the body it hands that lane to.
///
/// The route and the scheme are the lane's own: a lane states \c kRational when its
/// pieces are rational minimax pairs and \c kMonomial when its coefficients are read
/// in the monomial basis, and a lane that states neither sums the Chebyshev form of
/// the fit by the split Clenshaw recurrence (boys_cuda_arithmetic.hpp, the two traits
/// at the head of its bodies). The packing is the body's: \c DeviceAllOrdersF64 seeds
/// the top order's piece and steps down the recurrence, where \c DeviceOrdersF64
/// locates and sums every order's own piece, and the single-order bodies ask the fit
/// for the one order they were given. Where a row is one kernel under two names — the
/// rational route's pairs — the two names are two rows and each states the name it was
/// reached by, which is the one thing about such a row its kernel does not decide.
///
/// **The division is the device lane's default form and not a per-entry property.** No
/// entry of this space names a form: the bodies every kernel of them hands its lane
/// take the form as a compile-time parameter, and the kernels instantiate it from this
/// switch, so what an entry runs is \c kDefaultDeviceDivisionForm —
/// \c BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM, the device lane's own name in the
/// build's seam (boys/boys_build_defaults.hpp), which is what keeps the build's one
/// statement of the device's default and the arithmetic its entries run from coming
/// apart. An entry that ran another form would be a row of the option probe's cross,
/// where the form it ran is the form the run named rather than this field.
///
/// The report's rows read this and do not list it beside it: two statements of one
/// arithmetic can come apart, and the row a chooser places is the one that must not.
///
/// \param entry the option
///
/// \returns its four axes, or \c DevicePacking::kUnstated in \c packing and
///          \c kUnstatedDivisionForm in \c division when this switch has not been
///          taught the entry — which the assertion below turns into a compile error
///          rather than a default
///
/// \ingroup boys
constexpr DeviceEntryAxes DeviceEntryAxesOf(DeviceEntry entry) noexcept {
    switch (entry)
    {
        // The single-order shapes: one order asked for, one fit answering it, so no
        // ladder is read and the packing axis has no member here. The route and the
        // scheme are the coarsest lane's on both lanes and in both formats: every one
        // of these kernels hands its body Lane64Full or Lane32Full, which sums the
        // Chebyshev pieces by the split Clenshaw recurrence (src/boys_cuda.cu,
        // BoysSingleF64Kernel and BoysSingleF32Kernel; the fp16 entry runs the float
        // lane's body, BoysSingleF16Kernel), and the device-callable four hand it the
        // handle's own lanes, TableLane64 and TableLane32
        // (include/boys/boys_cuda_device.hpp, BoysDeviceSingleF64 and its siblings).
        case DeviceEntry::kSingleF64:
        case DeviceEntry::kSingleF64Fast:
        case DeviceEntry::kSingleF32:
        case DeviceEntry::kSingleF32Fast:
        case DeviceEntry::kSingleF16:
        case DeviceEntry::kDeviceSingleF64:
        case DeviceEntry::kDeviceSingleF64Fast:
        case DeviceEntry::kDeviceSingleF32:
        case DeviceEntry::kDeviceSingleF32Fast:
        case DeviceEntry::kDeviceSingleF16:
        // The half lane's fast region-B reading of the shape, appended: the same body the four
        // above hand their lanes to, with the half lane's store around it and the flag the f32
        // single pair states set.
        case DeviceEntry::kSingleF16Fast:
        // The half lane's fast single in the caller's own kernel: the same shape, the same
        // route and the same scheme as the row above, over the other region-B exponential.
        case DeviceEntry::kDeviceSingleF16Fast:
            return {FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, DevicePacking::kNotApplicable,
                    kDefaultDeviceDivisionForm};

        // The ladder read on the Chebyshev pieces: the coarsest partition's ladders on
        // both lanes and in both formats, the all-N and each-order shapes, the
        // device-callable generics, and the narrow partition's two. Every kernel of
        // them hands DeviceAllOrdersF64 or DeviceAllOrdersF32 a lane of the Chebyshev
        // family, and what differs between them is the partition the lane reads, which
        // DevicePartitionName states and this switch does not.
        case DeviceEntry::kAllOrdersF64:
        case DeviceEntry::kAllOrdersF64Fast:
        case DeviceEntry::kAllOrdersF32:
        case DeviceEntry::kAllOrdersF32Fast:
        case DeviceEntry::kAllOrdersF16:
        case DeviceEntry::kAllOrdersF16Fast:
        case DeviceEntry::kAllNF64:
        case DeviceEntry::kAllNF64Fast:
        case DeviceEntry::kAllNF32:
        case DeviceEntry::kAllNF32Fast:
        case DeviceEntry::kAllNF16:
        case DeviceEntry::kAllNF16Fast:
        case DeviceEntry::kAllOrdersF64Narrow:
        case DeviceEntry::kAllOrdersF64NarrowFast:
        case DeviceEntry::kAllOrdersF32Narrow:
        case DeviceEntry::kAllOrdersF32NarrowFast:
        case DeviceEntry::kDeviceAllOrdersF64:
        case DeviceEntry::kDeviceAllOrdersF64Fast:
        case DeviceEntry::kDeviceAllOrdersF32:
        case DeviceEntry::kDeviceAllOrdersF32Fast:
        case DeviceEntry::kDeviceAllOrdersF16:
        case DeviceEntry::kDeviceAllOrdersF16Fast:
        case DeviceEntry::kDeviceAllNF64:
        case DeviceEntry::kDeviceAllNF64Fast:
        case DeviceEntry::kDeviceAllNF32:
        case DeviceEntry::kDeviceAllNF32Fast:
        case DeviceEntry::kDeviceAllNF16:
        case DeviceEntry::kDeviceAllNF16Fast:
        case DeviceEntry::kDeviceEachOrderF64:
        case DeviceEntry::kDeviceEachOrderF64Fast:
        case DeviceEntry::kDeviceEachOrderF32:
        case DeviceEntry::kDeviceEachOrderF32Fast:
        case DeviceEntry::kDeviceEachOrderF16:
        case DeviceEntry::kDeviceEachOrderF16Fast:
        case DeviceEntry::kDeviceAllOrdersF64Narrow:
        case DeviceEntry::kDeviceAllOrdersF64NarrowFast:
        case DeviceEntry::kDeviceAllOrdersF32Narrow:
        case DeviceEntry::kDeviceAllOrdersF32NarrowFast:
        // The half lane's row of the same pieces: this lane's own definition is that it runs the
        // float lane's bodies, and the float lane's narrow ladder is the body
        // BoysCuda::AllOrdersF16Narrow hands its lane.
        case DeviceEntry::kAllOrdersF16Narrow:
        case DeviceEntry::kAllOrdersF16NarrowFast:
        // The half lane's narrow ladder from a caller's own kernel, which is the float lane's
        // body over this lane's store.
        case DeviceEntry::kDeviceAllOrdersF16Narrow:
        case DeviceEntry::kDeviceAllOrdersF16NarrowFast:
            return {FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, DevicePacking::kLadder,
                    kDefaultDeviceDivisionForm};

        // The same pieces and the same summation read the other way: each order's own
        // fit, through DeviceOrdersBody and DeviceOrdersBody32 over a Chebyshev lane.
        case DeviceEntry::kAllOrdersF64Orders:
        case DeviceEntry::kAllOrdersF64OrdersFast:
        case DeviceEntry::kAllOrdersF64NarrowOrders:
        case DeviceEntry::kAllOrdersF64NarrowOrdersFast:
        case DeviceEntry::kAllOrdersF32Orders:
        case DeviceEntry::kAllOrdersF32OrdersFast:
        case DeviceEntry::kAllOrdersF32NarrowOrders:
        case DeviceEntry::kAllOrdersF32NarrowOrdersFast:
        // The half lane's two rows of the reading: the float lane's per-order bodies in the fp16
        // store.
        case DeviceEntry::kAllOrdersF16Orders:
        case DeviceEntry::kAllOrdersF16OrdersFast:
        case DeviceEntry::kAllOrdersF16NarrowOrders:
        case DeviceEntry::kAllOrdersF16NarrowOrdersFast:
        // The per-order reading of the same pieces from a caller's own kernel, on the coarsest
        // and the narrow partitions of both precision pairs: the launched rows above them read
        // the same bodies through DeviceOrdersBody and DeviceOrdersBody32, and the device
        // entries call those bodies directly over the handle's lanes.
        case DeviceEntry::kDeviceAllOrdersF64Orders:
        case DeviceEntry::kDeviceAllOrdersF64OrdersFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrders:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersFast:
        case DeviceEntry::kDeviceAllOrdersF32Orders:
        case DeviceEntry::kDeviceAllOrdersF32OrdersFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrders:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersFast:
        case DeviceEntry::kDeviceAllOrdersF16Orders:
        case DeviceEntry::kDeviceAllOrdersF16OrdersFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrders:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersFast:
            return {FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, DevicePacking::kPerOrder,
                    kDefaultDeviceDivisionForm};

        // The monomial form of the same fits: Lane64MonoFull, Lane64MonoEff, the narrow
        // partition's monomial lanes and their float lane counterparts state kMonomial,
        // and the bodies above sum what they read by Horner
        // (boys_cuda_arithmetic.hpp, DevicePieceSum).
        case DeviceEntry::kAllOrdersF64Mono:
        case DeviceEntry::kAllOrdersF64MonoFast:
        case DeviceEntry::kAllOrdersF64NarrowMono:
        case DeviceEntry::kAllOrdersF64NarrowMonoFast:
        case DeviceEntry::kAllOrdersF32NarrowMono:
        case DeviceEntry::kAllOrdersF32NarrowMonoFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowMono:
        case DeviceEntry::kDeviceAllOrdersF64NarrowMonoFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowMono:
        case DeviceEntry::kDeviceAllOrdersF32NarrowMonoFast:
        // The half lane's monomial row of the same pieces, read by the same bodies.
        case DeviceEntry::kAllOrdersF16NarrowMono:
        case DeviceEntry::kAllOrdersF16NarrowMonoFast:
        // The coarsest partition's monomial rows, appended: the float and half lanes' rows of
        // the pieces the double lane's two rows above carry, over the same kernels' bodies with
        // those lanes' tables and stores around them.
        case DeviceEntry::kAllOrdersF32Mono:
        case DeviceEntry::kAllOrdersF32MonoFast:
        case DeviceEntry::kAllOrdersF16Mono:
        case DeviceEntry::kAllOrdersF16MonoFast:
        // The half lane's monomial narrow ladder from a caller's own kernel.
        case DeviceEntry::kDeviceAllOrdersF16NarrowMono:
        case DeviceEntry::kDeviceAllOrdersF16NarrowMonoFast:
            return {FitRoute::kChebyshev, EvalScheme::kHorner, DevicePacking::kLadder,
                    kDefaultDeviceDivisionForm};

        case DeviceEntry::kAllOrdersF64OrdersMono:
        case DeviceEntry::kAllOrdersF64OrdersMonoFast:
        case DeviceEntry::kAllOrdersF64NarrowOrdersMono:
        case DeviceEntry::kAllOrdersF64NarrowOrdersMonoFast:
        case DeviceEntry::kAllOrdersF32NarrowOrdersMono:
        case DeviceEntry::kAllOrdersF32NarrowOrdersMonoFast:
        // The half lane's row of it.
        case DeviceEntry::kAllOrdersF16NarrowOrdersMono:
        case DeviceEntry::kAllOrdersF16NarrowOrdersMonoFast:
        // The coarsest partition's rows of that reading, appended for the same reason as the
        // ladder rows above.
        case DeviceEntry::kAllOrdersF32OrdersMono:
        case DeviceEntry::kAllOrdersF32OrdersMonoFast:
        case DeviceEntry::kAllOrdersF16OrdersMono:
        case DeviceEntry::kAllOrdersF16OrdersMonoFast:
        // The monomial form of that reading, from a caller's own kernel, on the narrow
        // partition of both lanes.
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersMono:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersMonoFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersMono:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersMonoFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersMono:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersMonoFast:
            return {FitRoute::kChebyshev, EvalScheme::kHorner, DevicePacking::kPerOrder,
                    kDefaultDeviceDivisionForm};

        // The uniform grid, whose every block is one order's own fit at the interval's
        // own degree: every order is read from its own block and none is built from
        // another's, which is the per-order member of the packing axis, and the grid
        // has no ladder to read (boys_cuda_arithmetic.hpp, DeviceAllOrdersF64Flat).
        // The route's rows of that axis run this same body: a grid stored per order and
        // per interval has no seeded ladder for a second reading to be.
        case DeviceEntry::kAllOrdersF64Uniform:
        case DeviceEntry::kAllOrdersF64OrdersUniform:
        case DeviceEntry::kAllOrdersF32Uniform:
        case DeviceEntry::kAllOrdersF32OrdersUniform:
        case DeviceEntry::kDeviceAllOrdersF64Uniform:
        case DeviceEntry::kDeviceAllOrdersF32Uniform:
        // The half lane's grid rows, which read the float lane's grid.
        case DeviceEntry::kAllOrdersF16Uniform:
        case DeviceEntry::kAllOrdersF16OrdersUniform:
        // The half lane's rows of the float lane's grid, from a caller's own kernel.
        case DeviceEntry::kDeviceAllOrdersF16Uniform:
            return {FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, DevicePacking::kPerOrder,
                    kDefaultDeviceDivisionForm};

        case DeviceEntry::kAllOrdersF64UniformHorner:
        case DeviceEntry::kAllOrdersF64OrdersUniformHorner:
        case DeviceEntry::kAllOrdersF32UniformHorner:
        case DeviceEntry::kAllOrdersF32OrdersUniformHorner:
        case DeviceEntry::kDeviceAllOrdersF64UniformHorner:
        case DeviceEntry::kDeviceAllOrdersF32UniformHorner:
        // The half lane's rows of the grid's monomial blocks.
        case DeviceEntry::kAllOrdersF16UniformHorner:
        case DeviceEntry::kAllOrdersF16OrdersUniformHorner:
        // The same grid at the scheme's other name, from a caller's own kernel.
        case DeviceEntry::kDeviceAllOrdersF16UniformHorner:
            return {FitRoute::kChebyshev, EvalScheme::kHorner, DevicePacking::kPerOrder,
                    kDefaultDeviceDivisionForm};

        // The grid reads no exponential at any argument: below the join every order is
        // summed from its own stored block, and above it the call falls to the
        // asymptote, whose seed is the reciprocal square root. The region-B walk, and
        // the exponential that seeds it, are two of the walks this partition replaces
        // with a table, so RegionBExp has no member at FitGranularity::kUniform.

        // The fit route: the same partitions, region structure and shapes with a piece
        // stored as a numerator and a denominator, which is a family of its own and the
        // lane's kRational member (boys_cuda_arithmetic.hpp, DeviceRatSum). Its pair is
        // stored once - in the monomial form the family is stored in - so the two scheme
        // names a caller may use reach one kernel and each row states the name it was
        // reached by: the plain name is kSplitClenshaw and its -Horner twin is kHorner,
        // which is the whole of what separates those two rows (src/boys_cuda.cu,
        // BoysAllOrdersF64RatKernel and the launcher comment over its twins).
        case DeviceEntry::kAllOrdersF64Rat:
        case DeviceEntry::kAllOrdersF64RatFast:
        case DeviceEntry::kAllOrdersF64NarrowRat:
        case DeviceEntry::kAllOrdersF64NarrowRatFast:
        case DeviceEntry::kAllOrdersF32Rat:
        case DeviceEntry::kAllOrdersF32RatFast:
        case DeviceEntry::kAllOrdersF32NarrowRat:
        case DeviceEntry::kAllOrdersF32NarrowRatFast:
        case DeviceEntry::kDeviceAllOrdersF64Rat:
        case DeviceEntry::kDeviceAllOrdersF64RatFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowRat:
        case DeviceEntry::kDeviceAllOrdersF64NarrowRatFast:
        case DeviceEntry::kDeviceAllOrdersF32Rat:
        case DeviceEntry::kDeviceAllOrdersF32RatFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowRat:
        case DeviceEntry::kDeviceAllOrdersF32NarrowRatFast:
        // The half lane's pair of that family, over the float lane's coarsest and narrow pieces.
        case DeviceEntry::kAllOrdersF16Rat:
        case DeviceEntry::kAllOrdersF16RatFast:
        case DeviceEntry::kAllOrdersF16NarrowRat:
        case DeviceEntry::kAllOrdersF16NarrowRatFast:
        // The half lane's pair of the route's ladders, from a caller's own kernel.
        case DeviceEntry::kDeviceAllOrdersF16Rat:
        case DeviceEntry::kDeviceAllOrdersF16RatFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowRat:
        case DeviceEntry::kDeviceAllOrdersF16NarrowRatFast:
            return {FitRoute::kRationalMinimax, EvalScheme::kSplitClenshaw, DevicePacking::kLadder,
                    kDefaultDeviceDivisionForm};

        case DeviceEntry::kAllOrdersF64RatHorner:
        case DeviceEntry::kAllOrdersF64RatHornerFast:
        case DeviceEntry::kAllOrdersF64NarrowRatHorner:
        case DeviceEntry::kAllOrdersF64NarrowRatHornerFast:
        case DeviceEntry::kAllOrdersF32RatHorner:
        case DeviceEntry::kAllOrdersF32RatHornerFast:
        case DeviceEntry::kAllOrdersF32NarrowRatHorner:
        case DeviceEntry::kAllOrdersF32NarrowRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF64RatHorner:
        case DeviceEntry::kDeviceAllOrdersF64RatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowRatHorner:
        case DeviceEntry::kDeviceAllOrdersF64NarrowRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF32RatHorner:
        case DeviceEntry::kDeviceAllOrdersF32RatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowRatHorner:
        case DeviceEntry::kDeviceAllOrdersF32NarrowRatHornerFast:
        // The half lane's pair of the same names, reaching the same two kernels.
        case DeviceEntry::kAllOrdersF16RatHorner:
        case DeviceEntry::kAllOrdersF16RatHornerFast:
        case DeviceEntry::kAllOrdersF16NarrowRatHorner:
        case DeviceEntry::kAllOrdersF16NarrowRatHornerFast:
        // The same two names at the route's other scheme name, from a caller's own kernel.
        case DeviceEntry::kDeviceAllOrdersF16RatHorner:
        case DeviceEntry::kDeviceAllOrdersF16RatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowRatHorner:
        case DeviceEntry::kDeviceAllOrdersF16NarrowRatHornerFast:
            return {FitRoute::kRationalMinimax, EvalScheme::kHorner, DevicePacking::kLadder,
                    kDefaultDeviceDivisionForm};

        // The route's pieces on its orders reading: each order's own piece at A = 1,
        // through the same bodies the Chebyshev rows of that axis use
        // (src/boys_cuda.cu, BoysAllOrdersF64OrdersRatKernel and its siblings).
        case DeviceEntry::kAllOrdersF64OrdersRat:
        case DeviceEntry::kAllOrdersF64OrdersRatFast:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRat:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRatFast:
        case DeviceEntry::kAllOrdersF32OrdersRat:
        case DeviceEntry::kAllOrdersF32OrdersRatFast:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRat:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRatFast:
        // The half lane's rows of the route on its orders reading.
        case DeviceEntry::kAllOrdersF16OrdersRat:
        case DeviceEntry::kAllOrdersF16OrdersRatFast:
        case DeviceEntry::kAllOrdersF16NarrowOrdersRat:
        case DeviceEntry::kAllOrdersF16NarrowOrdersRatFast:
        // The route's per-order reading, from a caller's own kernel, on both partitions of
        // every precision.
        case DeviceEntry::kDeviceAllOrdersF64OrdersRat:
        case DeviceEntry::kDeviceAllOrdersF64OrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRat:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF32OrdersRat:
        case DeviceEntry::kDeviceAllOrdersF32OrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRat:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF16OrdersRat:
        case DeviceEntry::kDeviceAllOrdersF16OrdersRatFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRat:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRatFast:
            return {FitRoute::kRationalMinimax, EvalScheme::kSplitClenshaw, DevicePacking::kPerOrder,
                    kDefaultDeviceDivisionForm};

        case DeviceEntry::kAllOrdersF64OrdersRatHorner:
        case DeviceEntry::kAllOrdersF64OrdersRatHornerFast:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner:
        case DeviceEntry::kAllOrdersF64NarrowOrdersRatHornerFast:
        case DeviceEntry::kAllOrdersF32OrdersRatHorner:
        case DeviceEntry::kAllOrdersF32OrdersRatHornerFast:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner:
        case DeviceEntry::kAllOrdersF32NarrowOrdersRatHornerFast:
        // The half lane's rows of the same, at the route's other scheme name.
        case DeviceEntry::kAllOrdersF16OrdersRatHorner:
        case DeviceEntry::kAllOrdersF16OrdersRatHornerFast:
        case DeviceEntry::kAllOrdersF16NarrowOrdersRatHorner:
        case DeviceEntry::kAllOrdersF16NarrowOrdersRatHornerFast:
        // The same rows at the route's other scheme name, from a caller's own kernel.
        case DeviceEntry::kDeviceAllOrdersF64OrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF64OrdersRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF32OrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF32OrdersRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF16OrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF16OrdersRatHornerFast:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRatHorner:
        case DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRatHornerFast:
            return {FitRoute::kRationalMinimax, EvalScheme::kHorner, DevicePacking::kPerOrder,
                    kDefaultDeviceDivisionForm};

        // The grid on the rational route: one pair per interval at the interval's own
        // stored count, read by the two Horner sums every pair of this family is read by
        // (boys_cuda_arithmetic.hpp, DeviceAllOrdersF64FlatRat), and per order as the
        // grid's Chebyshev member is. The scheme name splits the pairs as it does above.
        case DeviceEntry::kAllOrdersF64UniformRat:
        case DeviceEntry::kAllOrdersF64OrdersUniformRat:
        case DeviceEntry::kAllOrdersF32UniformRat:
        case DeviceEntry::kAllOrdersF32OrdersUniformRat:
        case DeviceEntry::kDeviceAllOrdersF64UniformRat:
        case DeviceEntry::kDeviceAllOrdersF32UniformRat:
        // The half lane's rows of the grid's rational route.
        case DeviceEntry::kAllOrdersF16UniformRat:
        case DeviceEntry::kAllOrdersF16OrdersUniformRat:
        // The half lane's rows of the grid's rational member, from a caller's own kernel.
        case DeviceEntry::kDeviceAllOrdersF16UniformRat:
            return {FitRoute::kRationalMinimax, EvalScheme::kSplitClenshaw, DevicePacking::kPerOrder,
                    kDefaultDeviceDivisionForm};

        case DeviceEntry::kAllOrdersF64UniformRatHorner:
        case DeviceEntry::kAllOrdersF64OrdersUniformRatHorner:
        case DeviceEntry::kAllOrdersF32UniformRatHorner:
        case DeviceEntry::kAllOrdersF32OrdersUniformRatHorner:
        case DeviceEntry::kDeviceAllOrdersF64UniformRatHorner:
        case DeviceEntry::kDeviceAllOrdersF32UniformRatHorner:
        // The half lane's rows of the same, at the route's other scheme name.
        case DeviceEntry::kAllOrdersF16UniformRatHorner:
        case DeviceEntry::kAllOrdersF16OrdersUniformRatHorner:
        // The same grid at the route's other scheme name, from a caller's own kernel.
        case DeviceEntry::kDeviceAllOrdersF16UniformRatHorner:
            return {FitRoute::kRationalMinimax, EvalScheme::kHorner, DevicePacking::kPerOrder,
                    kDefaultDeviceDivisionForm};

        // The each-order shape, launched: the ladder of the all-orders rows above, written from the
        // caller's own offsets rather than into planes. The lane and the body are the ones the row
        // of that name in the arm above states - the coarsest ladder of this lane's Chebyshev
        // pieces, read by the split Clenshaw recurrence - and the offset a sink indexes by is the
        // kernel's and not the arithmetic's, so the two rows of a lane differ in nothing but the
        // region-B exponential the kernel is instantiated at.
        case DeviceEntry::kEachOrderF64:
        case DeviceEntry::kEachOrderF64Fast:
        case DeviceEntry::kEachOrderF32:
        case DeviceEntry::kEachOrderF32Fast:
        case DeviceEntry::kEachOrderF16:
        case DeviceEntry::kEachOrderF16Fast:
            return {FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, DevicePacking::kLadder,
                    kDefaultDeviceDivisionForm};

        // The sentinel one past the last row this report defines, and not a row a call
        // can name. It is named rather than left to a default arm for the reason
        // DevicePartitionOf's kCount arm gives: a default would swallow the next
        // enumerator as quietly as it swallows this one, and the assertion below is what
        // fails the build for a row this switch has not been taught.
        case DeviceEntry::kCount:
            break;
    }

    // An enumerator no arm above names, which the check below turns into a compile
    // error. Nothing here states an arithmetic for an option that has not named one.
    return {FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, DevicePacking::kUnstated,
            kUnstatedDivisionForm};
}

/// Whether the switch above states the four axes for every enumerator of
/// \c DeviceEntry, read over the enumeration's own count.
///
/// The compilation of this header is what keeps the two in step: an enumerator added to
/// \c DeviceEntry and not named above falls through that switch, this predicate sees the
/// unstated value it answered, and the assertion below stops the build. A new option
/// therefore cannot arrive in the space without a statement of the route, the scheme,
/// the packing and the division its entry runs — the omission is the error, rather than
/// a row quietly reporting the build's defaults for arithmetic it does not run.
///
/// Both sentinels are read and not only the packing's: a predicate that tested one axis
/// would pass an arm that stated every axis but the other, which is the omission it
/// exists to catch. The division's sentinel carries a value outside \c DivisionForm's
/// enumerators — the axis has no member of its own for it, because its type is the
/// host's — and \c kUnstatedDivisionForm is that value (boys_cuda_options.hpp).
///
/// \returns true when every enumerator of \c DeviceEntry states its axes above
constexpr bool DeviceEntriesAllStateTheirAxes() noexcept {
    for (int i = 0; i < static_cast<int>(DeviceEntry::kCount); ++i)
    {
        const DeviceEntryAxes axes = DeviceEntryAxesOf(static_cast<DeviceEntry>(i));

        if (axes.packing == DevicePacking::kUnstated || axes.division == kUnstatedDivisionForm)
        {
            return false;
        }
    }

    return true;
}

static_assert(DeviceEntriesAllStateTheirAxes(),
              "an enumerator of DeviceEntry states no axes: name it in DeviceEntryAxesOf "
              "(boys_cuda_options.hpp) as the route, the scheme, the packing and the division "
              "its entry runs");

/// One row of the device option space: an option this surface offers, with
/// what a chooser needs to place it.
///
/// The rows are the space and not a list of the interesting ones: an entry that exists
/// but that a given build cannot serve is carried here with \c built false and the
/// reason, so its absence from a report is a stated refusal rather than an omission.
/// The build-time case this build has is the fp16 seam; whether a *device* is present
/// is a run-time fact about a host and not a property of the option, so a host report
/// states that beside its figures rather than here.
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
    double bound; ///< the documented bound, over the whole argument range the entry serves
    const char* boundForm; ///< the documented form, in words, as the entry states it
    bool built; ///< whether this build serves the option
    const char* refusedBecause; ///< why not, when \c built is false; nullptr otherwise

    /// The fit route the entry's pieces are cut from, the summation its stored
    /// coefficients are read by, its reading of region A, and the form its ladder's
    /// divisions are performed in.
    ///
    /// Stated on every row and not only on the rows that move them: the four name the
    /// arithmetic a row is, so a chooser placing two rows side by side reads whether
    /// they are the same arithmetic off the rows themselves, and a winner's identity is
    /// its own tuple rather than something a reader recovers from its name. The member
    /// of a row whose \c axis is kRoute is \c route, of one whose axis is kScheme
    /// \c scheme, of one whose axis is kPacking \c packing, and of one whose axis is
    /// kDivision \c division.
    ///
    /// Read from the row's own entry and never listed beside it:
    /// \c DeviceEntryAxesOf is the one statement of what an entry runs, read off the
    /// lane and the body its kernels and device functions name, so a copy written here
    /// could disagree with the arithmetic the row reports. A \c kNotApplicable packing
    /// is the entry's own statement that it fixes nothing on that axis — the
    /// single-order shape — and not a row nobody filled in: an entry that statement has
    /// not been taught fails to compile where it is written.
    FitRoute route = DeviceEntryAxesOf(entry).route;
    EvalScheme scheme = DeviceEntryAxesOf(entry).scheme; ///< the row's scheme-axis value
    DevicePacking packing = DeviceEntryAxesOf(entry).packing; ///< the row's packing-axis value
    DivisionForm division = DeviceEntryAxesOf(entry).division; ///< the row's division-axis value

    /// The cut of the domain the entry reads its fits from.
    ///
    /// Stated on every row, as the four axes above are: the cut a row reads is a
    /// property of the arithmetic and not of the axis it varies, so the rows whose
    /// \c axis is the partition, those whose axis is the scheme inside it and those
    /// whose axis is the route inside it all state the cut they read here, and a
    /// chooser placing two rows side by side sees whether they read the same tables.
    /// The member of a row whose \c axis is kPartition is this field.
    ///
    /// Read from the row's own entry and never listed beside it:
    /// \c DevicePartitionOf is the one statement of which cut an entry reads, read off
    /// the structure of the kernel bodies and the tables they are handed
    /// (boys_cuda_device.hpp, BoysDeviceTables), so a copy written here could disagree
    /// with the arithmetic the row reports.
    FitGranularity partition = DevicePartitionOf(entry);

};

/// The device option space this revision defines, one row per option, read from
/// the entries and the bounds in this header rather than listed beside them.
///
/// A report that enumerates *this* is a projection of the library and cannot fall
/// behind it: a precision, a shape or an axis member added to the surface appears here
/// as a row, and a row no build-time seam serves is carried with the reason. What a
/// chooser gets from a row is the entry, the precision it computes in, the question it
/// answers, the axis it varies where it has one, the route, the summation, the packing
/// and the division its entry runs, the cut of the domain it reads its fits from, the
/// degree tables it reads, and the bound it is documented at — the same facts the
/// accuracy gate certifies the entry at.
///
/// The bound is the documented figure, over the whole argument range the entry serves;
/// \c boundForm states the shape the documentation gives it. Where that form carries a
/// term a value decides — the fp16 rows' half ULPs — \c bound is the figure with that
/// term dropped, so a column of costs compares figures of one kind.
///
/// \returns the rows, in a fixed order: the enumerator order of DeviceEntry.
///
/// \ingroup boys
std::span<const DeviceOptionInfo> BoysDeviceOptions() noexcept;

} // namespace boys
