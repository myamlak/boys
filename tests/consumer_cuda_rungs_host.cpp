// The consumer check on the CUDA lane's run-time rung.
//
// A caller of the device lane that decides its accuracy at run time writes what
// this file writes:
//
//   boys::BoysCuda::AllOrdersF64AtRung(m, n, x, out, count, stream);
//
// and names one of the twelve rungs of `boys::kDeviceRungs` in `m`, where a
// caller that had fixed the rung where the call is written would write
// `boys::BoysCuda::AllOrdersF64<4096.0>(...)`. There is no switch over the rungs
// at the call site, no table of entry names and no run-time search: the
// combination is the entry's name and the rung is the call's own argument.
//
// What this file checks, and against what:
//
//  * the book. `BoysDeviceOptions()` is the lane's own report of the options it
//    serves, and it is read here rather than retyped: its launched rows are the
//    served combinations, and every launched row this build serves is required
//    to have a rung-argument name. The census below is counted from that book,
//    per precision, as combinations served and combinations named, and as cells
//    served and cells named and evaluated.
//
//  * the rung it was handed. Every cell is evaluated twice — once through the
//    rung-argument sibling and once through the same entry's compile-time
//    spelling at that rung — and the two values are compared bit for bit. A
//    sibling that accepted a rung and answered at another would differ here; the
//    count of cells where the rung moved the value is printed beside the count
//    of comparisons, so the comparison is visible as one that could have failed.
//
//  * the bound that cell carries. Every value is judged against the committed
//    45-digit reference grid at the figure the book states for its own row at
//    the rung it ran at — the row's documented bound, scaled by the rung, plus
//    the term the row's own form adds.
//
//  * the refusal, on the card. The device-callable entry is the one that reads
//    the resident rung, so it is the one that can refuse it; the other half of
//    this check (tests/consumer_cuda_rungs.cu) runs it inside a kernel at the
//    resident rung and at one that is not resident, and this half prints what
//    came back for each. An AtRung call that makes the rung it names resident
//    leaves that rule standing; one that answered at whatever was resident
//    instead would not, and that is what the check is for.
//
// Run:  cmake --build <build> --config Release --target boys-consumer-cuda-rungs
//       <build>/Release/boys-consumer-cuda-rungs
//       ctest --test-dir <build> -C Release -R boys-consumer-cuda-rungs

#include <algorithm>
#include <boys/boys_cuda.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <vector>

// Defined in tests/consumer_cuda_rungs.cu; C linkage, so the two sides are one
// signature and a drift between them is a link error rather than a silent
// second reading.
extern "C" int BoysConsumerCudaRungSingle(const boys::BoysDeviceTables* tables,
                                          const int* n,
                                          const double* x,
                                          double* out,
                                          std::size_t count,
                                          double multiplier,
                                          int* statuses);

/// The two values the device half's entry returns, named from that side because
/// boys_cuda_device.hpp is a CUDA header and this half cannot include it: one
/// library, one definition of what each status is worth.
extern "C" int BoysConsumerCudaRungSuccess();
extern "C" int BoysConsumerCudaRungNotResident();

namespace {

std::size_t gFailures = 0;
std::size_t gPrinted = 0;
constexpr std::size_t kPrinted = 5;

void Require(bool held, const char* what) {
    if (!held)
    {
        if (gPrinted < kPrinted)
        {
            std::printf("  FAILED %s\n", what);
            ++gPrinted;
        }

        ++gFailures;
    }
}

void Check(cudaError_t error, const char* what) {
    if (error != cudaSuccess)
    {
        std::printf("boys consumer cuda rungs: %s failed: %s\n", what, cudaGetErrorString(error));
        std::exit(2);
    }
}

// --- the committed reference grid -------------------------------------------

/// One cell of the committed 45-digit grid: F_n(x) to more digits than any lane
/// here returns.
struct Cell {
    int n = 0;
    double x = 0.0;
    double value = 0.0;
};

struct Grid {
    std::vector<Cell> cells;  ///< order-major: cell (n, i) is n * argCount + i
    std::vector<double> args; ///< the swept arguments, ascending
    std::size_t argCount = 0;
};

/// The column pair the grid is read through: the argument the lane is handed,
/// and the value at that argument.
struct Columns {
    std::size_t x = 0;
    std::size_t value = 0;
};

/// Reads the committed grid and sweeps every k-th distinct argument of it,
/// keeping the whole ladder at every swept argument: the fits, the region-B seed
/// and the asymptotic form are three pieces of arithmetic behind one entry, and
/// a combination named at one argument would not say which of them it reached.
bool LoadGrid(const char* path, Columns columns, Grid& grid) {
    std::ifstream in(path);

    if (!in)
    {
        return false;
    }

    std::string line;
    std::getline(in, line); // the header row

    std::vector<Cell> rows;

    while (std::getline(in, line))
    {
        std::vector<std::string> fields;
        std::size_t start = 0;

        for (;;)
        {
            const std::size_t comma = line.find(',', start);
            fields.push_back(line.substr(start, comma - start));

            if (comma == std::string::npos)
            {
                break;
            }

            start = comma + 1;
        }

        if (fields.size() <= std::max(columns.x, columns.value))
        {
            continue;
        }

        const int n = std::atoi(fields[0].c_str());
        const double x = std::atof(fields[columns.x].c_str());
        const double value = std::atof(fields[columns.value].c_str());

        if (n < 0 || n > boys::kMaxBoysOrder || !std::isfinite(x) || !std::isfinite(value))
        {
            continue;
        }

        rows.push_back(Cell{n, x, value});
    }

    if (rows.empty())
    {
        return false;
    }

    constexpr std::size_t kStride = 8;

    std::vector<double> distinct;

    for (const Cell& cell : rows)
    {
        distinct.push_back(cell.x);
    }

    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());

    for (std::size_t i = 0; i < distinct.size(); i += kStride)
    {
        grid.args.push_back(distinct[i]);
    }

    if (!distinct.empty() && (grid.args.empty() || grid.args.back() != distinct.back()))
    {
        grid.args.push_back(distinct.back());
    }

    grid.argCount = grid.args.size();
    grid.cells.assign((boys::kMaxBoysOrder + 1) * grid.argCount, Cell{});

    for (const Cell& cell : rows)
    {
        const auto at = std::lower_bound(grid.args.begin(), grid.args.end(), cell.x);

        if (at == grid.args.end() || *at != cell.x)
        {
            continue; // an argument the sweep does not visit
        }

        const std::size_t i = static_cast<std::size_t>(at - grid.args.begin());
        grid.cells[static_cast<std::size_t>(cell.n) * grid.argCount + i] = cell;
    }

    // The indexing below is the rectangular shape's, so the shape is checked
    // rather than assumed: every swept argument carries every order.
    for (std::size_t i = 0; i < grid.cells.size(); ++i)
    {
        const auto n = static_cast<int>(i / grid.argCount);

        if (grid.cells[i].n != n || grid.cells[i].x != grid.args[i % grid.argCount])
        {
            return false;
        }
    }

    return true;
}

// --- the device buffers -----------------------------------------------------

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
        Check(cudaMemcpy(mPtr, src.data(), mCount * sizeof(T), cudaMemcpyHostToDevice),
              "cudaMemcpy host to device");
    }

    void Download(std::vector<T>& dst) const {
        Check(cudaMemcpy(dst.data(), mPtr, mCount * sizeof(T), cudaMemcpyDeviceToHost),
              "cudaMemcpy device to host");
    }

    /// A pattern no entry writes, so a slot the entry did not reach reads back
    /// as one rather than as a value.
    void Fill(unsigned char pattern) {
        Check(cudaMemset(mPtr, pattern, mCount * sizeof(T)), "cudaMemset");
    }

private:
    T* mPtr = nullptr;
    std::size_t mCount = 0;
};

// --- what a launch takes ----------------------------------------------------

/// The arguments every launched entry of this check takes: the per-element order
/// array (or, for the uniform-order shape, the scalar it reads instead), the
/// arguments, the count and the stream.
struct Batch {
    const int* n = nullptr;
    const void* x = nullptr;
    int nmax = 0;
    std::size_t count = 0;
    void* stream = nullptr;
};

/// The value type an entry returns, and the width one value occupies in the
/// caller's array.
enum class Kind { kDouble, kFloat, kHalf };

std::size_t Width(Kind kind) {
    return kind == Kind::kDouble ? sizeof(double) : kind == Kind::kFloat ? sizeof(float) : 2u;
}

/// The value one slot holds, widened to double for the bound judgement. The half
/// lane's widening is the library's, so the two sides agree by construction
/// rather than by a second rounding rule written here.
double Widen(Kind kind, const unsigned char* buffer, std::size_t index) {
    switch (kind)
    {
    case Kind::kDouble:
        return reinterpret_cast<const double*>(buffer)[index];
    case Kind::kFloat:
        return reinterpret_cast<const float*>(buffer)[index];
    case Kind::kHalf:
        break;
    }

    return static_cast<float>(
        boys::detail::F16FromBits(reinterpret_cast<const std::uint16_t*>(buffer)[index]));
}

bool SameBits(Kind kind, const unsigned char* left, const unsigned char* right, std::size_t index) {
    const std::size_t width = Width(kind);
    return std::memcmp(left + index * width, right + index * width, width) == 0;
}

// --- the tags: one per entry, named where a caller writes it ----------------

struct SingleF64Tag {
    static constexpr const char* kEntry = "SingleF64";
    static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {
        return boys::BoysCuda::SingleF64AtRung(m, b.n, static_cast<const double*>(b.x),
                                               static_cast<double*>(out), b.count, b.stream);
    }
    template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {
        return boys::BoysCuda::SingleF64<M>(b.n, static_cast<const double*>(b.x),
                                            static_cast<double*>(out), b.count, b.stream);
    }
};

struct AllOrdersF64Tag {
    static constexpr const char* kEntry = "AllOrdersF64";
    static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {
        return boys::BoysCuda::AllOrdersF64AtRung(m, b.n, static_cast<const double*>(b.x),
                                                  static_cast<double*>(out), b.count, b.stream);
    }
    template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {
        return boys::BoysCuda::AllOrdersF64<M>(b.n, static_cast<const double*>(b.x),
                                               static_cast<double*>(out), b.count, b.stream);
    }
};

struct AllNF64Tag {
    static constexpr const char* kEntry = "AllNF64";
    static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {
        return boys::BoysCuda::AllNF64AtRung(m, b.nmax, static_cast<const double*>(b.x),
                                             static_cast<double*>(out), b.count, b.stream);
    }
    template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {
        return boys::BoysCuda::AllNF64<M>(b.nmax, static_cast<const double*>(b.x),
                                          static_cast<double*>(out), b.count, b.stream);
    }
};

struct AllOrdersF32Tag {
    static constexpr const char* kEntry = "AllOrdersF32";
    static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {
        return boys::BoysCuda::AllOrdersF32AtRung(m, b.n, static_cast<const double*>(b.x),
                                                  static_cast<float*>(out), b.count, b.stream);
    }
    template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {
        return boys::BoysCuda::AllOrdersF32<M>(b.n, static_cast<const double*>(b.x),
                                               static_cast<float*>(out), b.count, b.stream);
    }
};

struct AllNF32Tag {
    static constexpr const char* kEntry = "AllNF32";
    static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {
        return boys::BoysCuda::AllNF32AtRung(m, b.nmax, static_cast<const double*>(b.x),
                                             static_cast<float*>(out), b.count, b.stream);
    }
    template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {
        return boys::BoysCuda::AllNF32<M>(b.nmax, static_cast<const double*>(b.x),
                                          static_cast<float*>(out), b.count, b.stream);
    }
};

struct SingleF32AccurateTag {
    static constexpr const char* kEntry = "SingleF32<kAccurate>";
    static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {
        return boys::BoysCuda::SingleF32AtRung<boys::RegionBExp::kAccurate>(
            m, b.n, static_cast<const double*>(b.x), static_cast<float*>(out), b.count, b.stream);
    }
    template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {
        return boys::BoysCuda::SingleF32<M, boys::RegionBExp::kAccurate>(
            b.n, static_cast<const double*>(b.x), static_cast<float*>(out), b.count, b.stream);
    }
};

struct SingleF32FastTag {
    static constexpr const char* kEntry = "SingleF32<kFast>";
    static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {
        return boys::BoysCuda::SingleF32AtRung<boys::RegionBExp::kFast>(
            m, b.n, static_cast<const double*>(b.x), static_cast<float*>(out), b.count, b.stream);
    }
    template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {
        return boys::BoysCuda::SingleF32<M, boys::RegionBExp::kFast>(
            b.n, static_cast<const double*>(b.x), static_cast<float*>(out), b.count, b.stream);
    }
};

/// The eleven siblings of AllOrdersF64 that carry the four structural axes: one
/// tag each, and the same two spellings of one rung.
#define BOYS_CONSUMER_F64_AXIS_TAG(Tag, Entry)                                                  \
    struct Tag {                                                                                \
        static constexpr const char* kEntry = #Entry;                                           \
        static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {                   \
            return boys::BoysCuda::Entry##AtRung(m, b.n, static_cast<const double*>(b.x),       \
                                                 static_cast<double*>(out), b.count, b.stream); \
        }                                                                                       \
        template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {    \
            return boys::BoysCuda::Entry<M>(b.n, static_cast<const double*>(b.x),               \
                                            static_cast<double*>(out), b.count, b.stream);       \
        }                                                                                       \
    };

BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64NarrowTag, AllOrdersF64Narrow)
BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64OrdersTag, AllOrdersF64Orders)
BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64NarrowOrdersTag, AllOrdersF64NarrowOrders)
BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64MonoTag, AllOrdersF64Mono)
BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64OrdersMonoTag, AllOrdersF64OrdersMono)
BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64NarrowMonoTag, AllOrdersF64NarrowMono)
BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64NarrowOrdersMonoTag, AllOrdersF64NarrowOrdersMono)
BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64RatTag, AllOrdersF64Rat)
BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64OrdersRatTag, AllOrdersF64OrdersRat)
BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64NarrowRatTag, AllOrdersF64NarrowRat)
BOYS_CONSUMER_F64_AXIS_TAG(AllOrdersF64NarrowOrdersRatTag, AllOrdersF64NarrowOrdersRat)

#undef BOYS_CONSUMER_F64_AXIS_TAG

#if BoysFp16
struct SingleF16Tag {
    static constexpr const char* kEntry = "SingleF16";
    static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {
        return boys::BoysCuda::SingleF16AtRung(m, b.n, static_cast<const boys::F16*>(b.x),
                                               static_cast<boys::F16*>(out), b.count, b.stream);
    }
    template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {
        return boys::BoysCuda::SingleF16<M>(b.n, static_cast<const boys::F16*>(b.x),
                                            static_cast<boys::F16*>(out), b.count, b.stream);
    }
};

struct AllOrdersF16Tag {
    static constexpr const char* kEntry = "AllOrdersF16";
    static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {
        return boys::BoysCuda::AllOrdersF16AtRung(m, b.n, static_cast<const boys::F16*>(b.x),
                                                  static_cast<boys::F16*>(out), b.count, b.stream);
    }
    template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {
        return boys::BoysCuda::AllOrdersF16<M>(b.n, static_cast<const boys::F16*>(b.x),
                                               static_cast<boys::F16*>(out), b.count, b.stream);
    }
};

struct AllNF16Tag {
    static constexpr const char* kEntry = "AllNF16";
    static boys::BoysStatus AtRung(double m, const Batch& b, void* out) {
        return boys::BoysCuda::AllNF16AtRung(m, b.nmax, static_cast<const boys::F16*>(b.x),
                                             static_cast<boys::F16*>(out), b.count, b.stream);
    }
    template <double M> static boys::BoysStatus CompileTime(const Batch& b, void* out) {
        return boys::BoysCuda::AllNF16<M>(b.nmax, static_cast<const boys::F16*>(b.x),
                                          static_cast<boys::F16*>(out), b.count, b.stream);
    }
};
#endif // BoysFp16

// --- one arm per rung -------------------------------------------------------

/// The twelve arms a tag is reached through, in the order of the library's own
/// rung table. A call names the entry and the rung, and this is where the rung
/// becomes the compile-time spelling the sibling is one call of: the switch an
/// `AtRung` body carries, written on the other side of the call so that the two
/// can be compared cell for cell.
template <typename Tag> boys::BoysStatus CompileTimeAt(int rung, const Batch& b, void* out) {
    switch (rung)
    {
    case 0:
        return Tag::template CompileTime<boys::kBoysFullAccuracyMultiplier>(b, out);
    case 1:
        return Tag::template CompileTime<2.0>(b, out);
    case 2:
        return Tag::template CompileTime<10.0>(b, out);
    case 3:
        return Tag::template CompileTime<64.0>(b, out);
    case 4:
        return Tag::template CompileTime<100.0>(b, out);
    case 5:
        return Tag::template CompileTime<256.0>(b, out);
    case 6:
        return Tag::template CompileTime<1024.0>(b, out);
    case 7:
        return Tag::template CompileTime<4096.0>(b, out);
    case 8:
        return Tag::template CompileTime<1e4>(b, out);
    case 9:
        return Tag::template CompileTime<16384.0>(b, out);
    case 10:
        return Tag::template CompileTime<65536.0>(b, out);
    case 11:
        return Tag::template CompileTime<1e8>(b, out);
    default:
        return boys::BoysStatus::kInvalidArgument;
    }
}

template <typename Tag> boys::BoysStatus RowAtRung(double m, const Batch& b, void* out) {
    return Tag::AtRung(m, b, out);
}

template <typename Tag> boys::BoysStatus RowCompileTime(int rung, const Batch& b, void* out) {
    return CompileTimeAt<Tag>(rung, b, out);
}

// --- the rows ---------------------------------------------------------------

using AtRungFn = boys::BoysStatus (*)(double, const Batch&, void*);
using CompileTimeFn = boys::BoysStatus (*)(int, const Batch&, void*);

/// One launched option of the device lane's book, with the two spellings of one
/// cell: the sibling that takes the rung in the call, and the entry that takes
/// it where the call is written.
struct Row {
    boys::DeviceEntry entry = boys::DeviceEntry::kSingleF64;
    const char* entryName = "";
    boys::DeviceOptionPrecision precision = boys::DeviceOptionPrecision::kFp64;
    boys::DeviceOptionShape shape = boys::DeviceOptionShape::kSingle;
    Kind kind = Kind::kDouble;
    AtRungFn atRung = nullptr;
    CompileTimeFn compileTime = nullptr;
};

template <typename Tag>
Row MakeRow(boys::DeviceEntry entry,
            boys::DeviceOptionPrecision precision,
            boys::DeviceOptionShape shape,
            Kind kind) {
    Row row;
    row.entry = entry;
    row.entryName = Tag::kEntry;
    row.precision = precision;
    row.shape = shape;
    row.kind = kind;
    row.atRung = &RowAtRung<Tag>;
    row.compileTime = &RowCompileTime<Tag>;
    return row;
}

/// The rung-argument surface, one row per option of the device book that carries
/// one. The Rat rows come in pairs — the scheme axis is inert on that route, so
/// both scheme names select one arithmetic and one entry — and each row is one
/// option of the book and is swept as one.
const std::vector<Row>& Rows() {
    static const std::vector<Row> rows = [] {
        using boys::DeviceEntry;
        using boys::DeviceOptionPrecision;
        using boys::DeviceOptionShape;
        constexpr auto kFp64 = DeviceOptionPrecision::kFp64;
        constexpr auto kFp32 = DeviceOptionPrecision::kFp32;
        constexpr auto kFp16 = DeviceOptionPrecision::kFp16;

        std::vector<Row> built;

        built.push_back(MakeRow<SingleF64Tag>(DeviceEntry::kSingleF64, kFp64,
                                              DeviceOptionShape::kSingle, Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64Tag>(DeviceEntry::kAllOrdersF64, kFp64,
                                                 DeviceOptionShape::kAllOrders, Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64NarrowTag>(DeviceEntry::kAllOrdersF64Narrow, kFp64,
                                                       DeviceOptionShape::kAllOrders,
                                                       Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64OrdersTag>(DeviceEntry::kAllOrdersF64Orders, kFp64,
                                                       DeviceOptionShape::kAllOrders,
                                                       Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64NarrowOrdersTag>(
            DeviceEntry::kAllOrdersF64NarrowOrders, kFp64, DeviceOptionShape::kAllOrders,
            Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64MonoTag>(DeviceEntry::kAllOrdersF64Mono, kFp64,
                                                     DeviceOptionShape::kAllOrders, Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64OrdersMonoTag>(
            DeviceEntry::kAllOrdersF64OrdersMono, kFp64, DeviceOptionShape::kAllOrders,
            Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64NarrowMonoTag>(
            DeviceEntry::kAllOrdersF64NarrowMono, kFp64, DeviceOptionShape::kAllOrders,
            Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64NarrowOrdersMonoTag>(
            DeviceEntry::kAllOrdersF64NarrowOrdersMono, kFp64, DeviceOptionShape::kAllOrders,
            Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64RatTag>(DeviceEntry::kAllOrdersF64Rat, kFp64,
                                                    DeviceOptionShape::kAllOrders, Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64RatTag>(DeviceEntry::kAllOrdersF64RatHorner, kFp64,
                                                    DeviceOptionShape::kAllOrders, Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64OrdersRatTag>(DeviceEntry::kAllOrdersF64OrdersRat,
                                                          kFp64, DeviceOptionShape::kAllOrders,
                                                          Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64OrdersRatTag>(
            DeviceEntry::kAllOrdersF64OrdersRatHorner, kFp64, DeviceOptionShape::kAllOrders,
            Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64NarrowRatTag>(DeviceEntry::kAllOrdersF64NarrowRat,
                                                          kFp64, DeviceOptionShape::kAllOrders,
                                                          Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64NarrowRatTag>(
            DeviceEntry::kAllOrdersF64NarrowRatHorner, kFp64, DeviceOptionShape::kAllOrders,
            Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64NarrowOrdersRatTag>(
            DeviceEntry::kAllOrdersF64NarrowOrdersRat, kFp64, DeviceOptionShape::kAllOrders,
            Kind::kDouble));
        built.push_back(MakeRow<AllOrdersF64NarrowOrdersRatTag>(
            DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner, kFp64, DeviceOptionShape::kAllOrders,
            Kind::kDouble));
        built.push_back(MakeRow<AllNF64Tag>(DeviceEntry::kAllNF64, kFp64,
                                            DeviceOptionShape::kAllN, Kind::kDouble));

        built.push_back(MakeRow<SingleF32AccurateTag>(DeviceEntry::kSingleF32, kFp32,
                                                      DeviceOptionShape::kSingle, Kind::kFloat));
        built.push_back(MakeRow<SingleF32FastTag>(DeviceEntry::kSingleF32Fast, kFp32,
                                                  DeviceOptionShape::kSingle, Kind::kFloat));
        built.push_back(MakeRow<AllOrdersF32Tag>(DeviceEntry::kAllOrdersF32, kFp32,
                                                 DeviceOptionShape::kAllOrders, Kind::kFloat));
        built.push_back(MakeRow<AllNF32Tag>(DeviceEntry::kAllNF32, kFp32,
                                            DeviceOptionShape::kAllN, Kind::kFloat));

#if BoysFp16
        built.push_back(MakeRow<SingleF16Tag>(DeviceEntry::kSingleF16, kFp16,
                                              DeviceOptionShape::kSingle, Kind::kHalf));
        built.push_back(MakeRow<AllOrdersF16Tag>(DeviceEntry::kAllOrdersF16, kFp16,
                                                 DeviceOptionShape::kAllOrders, Kind::kHalf));
        built.push_back(MakeRow<AllNF16Tag>(DeviceEntry::kAllNF16, kFp16,
                                            DeviceOptionShape::kAllN, Kind::kHalf));
#endif

        return built;
    }();

    return rows;
}

const char* PrecisionName(boys::DeviceOptionPrecision precision) {
    switch (precision)
    {
    case boys::DeviceOptionPrecision::kFp64:
        return "fp64";
    case boys::DeviceOptionPrecision::kFp32:
        return "fp32";
    case boys::DeviceOptionPrecision::kFp16:
        return "fp16";
    }

    return "?";
}

/// The book's row for one entry, or nullptr where this build's book has none.
const boys::DeviceOptionInfo* BookRow(boys::DeviceEntry entry) {
    for (const boys::DeviceOptionInfo& info : boys::BoysDeviceOptions())
    {
        if (info.entry == entry)
        {
            return &info;
        }
    }

    return nullptr;
}

// --- one row's sweep --------------------------------------------------------

/// What a row's sweep compares, built once per row: the output slot each
/// compared value lives in, the reference value there, and the (order, argument)
/// pair it belongs to so that a failing cell can name itself, and the launch the
/// row's own shape takes.
///
/// The ladder and the uniform-order shape both write `out[k * count + i]`, so
/// both are swept over the arguments at one top order: the whole ladder at every
/// swept argument. The single-order shape writes one value per element, so it is
/// swept over the grid's own cells, each element carrying its own order.
struct Sweep {
    std::vector<std::size_t> outIndex;
    std::vector<double> reference;
    std::vector<int> order;
    std::vector<double> argument;
    std::size_t elements = 0;
    int nmax = 0;
};

Sweep MakeSweep(const Grid& source, boys::DeviceOptionShape shape) {
    Sweep sweep;

    if (shape == boys::DeviceOptionShape::kSingle)
    {
        for (std::size_t e = 0; e < source.cells.size(); ++e)
        {
            sweep.outIndex.push_back(e);
            sweep.reference.push_back(source.cells[e].value);
            sweep.order.push_back(source.cells[e].n);
            sweep.argument.push_back(source.cells[e].x);
        }

        sweep.elements = source.cells.size();
        return sweep;
    }

    for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
    {
        for (std::size_t i = 0; i < source.argCount; ++i)
        {
            const Cell& cell = source.cells[static_cast<std::size_t>(k) * source.argCount + i];
            sweep.outIndex.push_back(static_cast<std::size_t>(k) * source.argCount + i);
            sweep.reference.push_back(cell.value);
            sweep.order.push_back(k);
            sweep.argument.push_back(cell.x);
        }
    }

    sweep.elements = source.argCount;
    sweep.nmax = boys::kMaxBoysOrder;
    return sweep;
}

/// The figure the book states for one row at one rung: the row's documented
/// bound, scaled by the rung, plus the term the row's own form adds and does not
/// scale. The half rows' form adds half an ULP of the returned value, which is a
/// property of the value rather than of the call, so it is added per value.
double BoundAt(const boys::DeviceOptionInfo& row, double multiplier) {
    return multiplier * (row.bound - row.boundFixed) + row.boundFixed;
}

/// Half an ULP of a binary16 value: its own quantum, except at zero, where it is
/// the format's smallest subnormal — the narrowest the format can be, and what a
/// zero reference is judged with.
double HalfUlp(double value) {
    const double magnitude = std::fabs(value);

    if (magnitude == 0.0)
    {
        return std::ldexp(1.0, -24);
    }

    return std::ldexp(1.0, std::ilogb(magnitude) - 10);
}

struct Census {
    std::size_t servedRows = 0;
    std::size_t namedRows = 0;
    std::size_t servedCells = 0;
    std::size_t namedCells = 0;
    std::size_t compared = 0;
    std::size_t held = 0;
    std::size_t moved = 0;
    std::size_t judged = 0;
    std::size_t exceeded = 0;
};

/// Sweeps one row over every rung of the lane's table and folds the result into
/// the census: whether the rung the call was handed is the rung the same cell is
/// spelled at, and whether the value that came back is inside the figure the
/// book states for that cell.
void SweepRow(const Row& row, const Grid& source, Census& census) {
    const Sweep sweep = MakeSweep(source, row.shape);

    std::vector<int> orders(sweep.elements);
    std::vector<double> xs(sweep.elements);
    std::vector<boys::F16> xHalf(sweep.elements);

    for (std::size_t e = 0; e < sweep.elements; ++e)
    {
        const bool perElement = row.shape == boys::DeviceOptionShape::kSingle;
        orders[e] = perElement ? source.cells[e].n : boys::kMaxBoysOrder;
        xs[e] = perElement ? source.cells[e].x : source.args[e];
        xHalf[e] = boys::F16(static_cast<float>(xs[e]));
    }

    DevBuf<int> dN(sweep.elements);
    DevBuf<double> dX(sweep.elements);
    DevBuf<boys::F16> dXHalf(sweep.elements);
    dN.Upload(orders);
    dX.Upload(xs);
    dXHalf.Upload(xHalf);

    Batch batch;
    batch.n = dN.get();
    batch.x = row.kind == Kind::kHalf ? static_cast<const void*>(dXHalf.get())
                                      : static_cast<const void*>(dX.get());
    batch.nmax = sweep.nmax;
    batch.count = sweep.elements;

    const std::size_t slots = row.shape == boys::DeviceOptionShape::kSingle
                                  ? sweep.elements
                                  : sweep.elements * (boys::kMaxBoysOrder + 1);
    const std::size_t bytes = slots * Width(row.kind);

    DevBuf<unsigned char> produced(bytes);
    DevBuf<unsigned char> spelled(bytes);
    std::vector<unsigned char> hostProduced(bytes);
    std::vector<unsigned char> hostSpelled(bytes);
    std::vector<unsigned char> referenceRung;

    std::size_t moved = 0;
    double worst = 0.0;
    std::size_t worstN = 0;
    double worstX = 0.0;

    for (std::size_t rung = 0; rung < boys::kDeviceRungs.size(); ++rung)
    {
        const double multiplier = boys::kDeviceRungs[rung];
        const double bound = BoundAt(*BookRow(row.entry), multiplier);

        produced.Fill(0xCD);
        spelled.Fill(0xCD);

        const boys::BoysStatus named = row.atRung(multiplier, batch, produced.get());
        const boys::BoysStatus written =
            row.compileTime(static_cast<int>(rung), batch, spelled.get());

        if (named != boys::BoysStatus::kSuccess || written != boys::BoysStatus::kSuccess)
        {
            std::printf("  %s at m = %g: the rung-argument call returned %d, the compile-time "
                        "spelling %d\n",
                        row.entryName, multiplier, static_cast<int>(named),
                        static_cast<int>(written));
            ++gFailures;
            continue;
        }

        Check(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
        produced.Download(hostProduced);
        spelled.Download(hostSpelled);
        ++census.namedCells;

        for (std::size_t i = 0; i < sweep.outIndex.size(); ++i)
        {
            const std::size_t slot = sweep.outIndex[i];
            const bool same = SameBits(row.kind, hostProduced.data(), hostSpelled.data(), slot);

            ++census.compared;

            if (!same)
            {
                Require(false, "the rung-argument spelling returns the compile-time cell's value");
                continue;
            }

            ++census.held;

            if (!referenceRung.empty() &&
                !SameBits(row.kind, hostProduced.data(), referenceRung.data(), slot))
            {
                ++moved;
            }

            const double got = Widen(row.kind, hostProduced.data(), slot);

            // The bound this cell carries: the row's own figure at this rung,
            // plus the half lane's own quantum where the row is a half one — the
            // term that row's form adds beside the bound because it is a
            // property of the returned value rather than of the call.
            const double figure = bound + (row.kind == Kind::kHalf ? HalfUlp(got) : 0.0);
            const double error = std::fabs(got - sweep.reference[i]);

            ++census.judged;

            if (error > figure)
            {
                ++census.exceeded;

                if (census.exceeded <= kPrinted)
                {
                    std::printf("  EXCEEDED %s at m = %g n = %d x = %.17g measured %.17g "
                                "reference %.17g bound + %.17g\n",
                                row.entryName, multiplier, sweep.order[i], sweep.argument[i], got,
                                sweep.reference[i], figure);
                }
            }

            if (figure > 0.0 && error / figure > worst)
            {
                worst = error / figure;
                worstN = static_cast<std::size_t>(sweep.order[i]);
                worstX = sweep.argument[i];
            }
        }

        if (rung == 0)
        {
            referenceRung = hostProduced;
        }
    }

    census.moved += moved;

    const std::size_t swept = sweep.outIndex.size() * boys::kDeviceRungs.size();

    std::printf("  %-30s %-5s %8zu values judged, worst %.4g of the bound (n=%zu, x=%.6g)%s\n",
                row.entryName, PrecisionName(row.precision), swept, worst, worstN, worstX,
                moved == 0 ? "  [no cell moved between the rungs]" : "");
}

// --- the device half's own check --------------------------------------------

/// The resident-rung rule, on the card. The handle is filled at one relaxed rung,
/// the device-callable entry is run at that rung and at another, and the two
/// answers are printed: the first is a value, the second is the refusal and the
/// output the refused call did not write.
void CheckResidencyRefusal(const Grid& grid, void* stream) {
    constexpr double kResident = 2.0;
    constexpr double kOther = 100.0;
    constexpr std::size_t kCount = 16;
    constexpr double kSentinel = -12345.0;

    boys::BoysDeviceTables tables{};
    const boys::BoysStatus filled = boys::BoysCuda::DeviceTables<kResident>(&tables);
    Require(filled == boys::BoysStatus::kSuccess, "the handle is filled at a relaxed rung");

    if (filled != boys::BoysStatus::kSuccess)
    {
        return;
    }

    std::vector<int> n(kCount);
    std::vector<double> x(kCount);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        n[i] = static_cast<int>(i) % (boys::kMaxBoysOrder + 1);
        x[i] = grid.args.empty() ? 0.0 : grid.args[i * grid.args.size() / kCount];
    }

    DevBuf<int> dN(kCount);
    DevBuf<double> dX(kCount);
    DevBuf<double> dOut(kCount);
    DevBuf<int> dStatus(kCount);
    dN.Upload(n);
    dX.Upload(x);

    std::vector<double> out(kCount);
    std::vector<int> status(kCount);

    for (const double rung : {kResident, kOther})
    {
        std::fill(out.begin(), out.end(), kSentinel);
        std::fill(status.begin(), status.end(), 0);
        dOut.Upload(out);
        dStatus.Upload(status);

        const int launch = BoysConsumerCudaRungSingle(&tables, dN.get(), dX.get(), dOut.get(),
                                                      kCount, rung, dStatus.get());

        if (launch != 0)
        {
            std::printf("boys consumer cuda rungs: the device half's launch failed: %s\n",
                        cudaGetErrorString(static_cast<cudaError_t>(launch)));
            std::exit(2);
        }

        Check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
        dOut.Download(out);
        dStatus.Download(status);

        const int servedStatus = BoysConsumerCudaRungSuccess();
        const int refusedStatus = BoysConsumerCudaRungNotResident();
        const bool served = std::all_of(status.begin(), status.end(), [servedStatus](int value) {
            return value == servedStatus;
        });
        const bool refused = std::all_of(status.begin(), status.end(), [refusedStatus](int value) {
            return value == refusedStatus;
        });
        const bool wrote = std::none_of(out.begin(), out.end(), [](double value) {
            return value == kSentinel;
        });

        if (rung == kResident)
        {
            Require(served, "the device entry serves the rung that is resident");
            Require(wrote, "the device entry writes at the rung that is resident");
            std::printf("  m = %-8g resident: every element kSuccess, %zu values, first %.17g\n",
                        rung, kCount, out[0]);
        } else
        {
            Require(refused, "the device entry refuses the rung that is not resident");
            Require(!wrote, "the device entry writes nothing for a refused rung");
            std::printf("  m = %-8g not resident: every element kMultiplierNotResident, every "
                        "slot still the sentinel\n",
                        rung);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const char* gridPath = nullptr;

#ifdef BoysConsumerReference
    gridPath = BoysConsumerReference;
#endif

    if (argc > 1)
    {
        gridPath = argv[1];
    }

    if (gridPath == nullptr)
    {
        std::printf("boys consumer cuda rungs: no reference grid path compiled in\n");
        return 1;
    }

    int device = 0;
    Check(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp properties{};
    Check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

    const boys::BoysStatus tables = boys::BoysCuda::InitializeTables();

    if (tables != boys::BoysStatus::kSuccess)
    {
        std::printf("boys consumer cuda rungs: no usable device: the table upload returned %d\n",
                    static_cast<int>(tables));
        return 1;
    }

    Grid grid;
    Grid halfGrid;

    if (!LoadGrid(gridPath, Columns{1, 2}, grid))
    {
        std::printf("boys consumer cuda rungs: the reference grid is missing or malformed (%s)\n",
                    gridPath);
        return 1;
    }

    const bool haveHalf = LoadGrid(gridPath, Columns{7, 8}, halfGrid);

    std::printf("boys consumer cuda rungs: device %d '%s', %zu rungs, %zu swept arguments, "
                "%zu grid cells per row\n",
                device, properties.name, boys::kDeviceRungs.size(), grid.args.size(),
                grid.cells.size());

    // The rung vocabulary the entries answer at is the library's own table, read
    // rather than retyped, so a rung added to the lane appears here as an arm.
    Require(boys::kDeviceRungs.size() == 12, "the lane serves twelve rungs");
#if BoysFp16
    Require(haveHalf, "the half lane's own argument column is in the grid");
#endif

    // The census, counted from the book: what this build serves, and what has a
    // rung-argument name here.
    std::vector<Census> census(3);

    for (const boys::DeviceOptionInfo& info : boys::BoysDeviceOptions())
    {
        if (info.group != boys::DeviceOptionGroup::kLaunched || !info.built)
        {
            continue;
        }

        Census& classCensus = census[static_cast<std::size_t>(info.precision)];
        ++classCensus.servedRows;
        classCensus.servedCells += boys::kDeviceRungs.size();

        const bool named = std::any_of(Rows().begin(), Rows().end(), [&info](const Row& candidate) {
            return candidate.entry == info.entry;
        });

        if (named)
        {
            ++classCensus.namedRows;
        } else
        {
            std::printf("  %-24s served and has no rung-argument name\n", info.name);
        }
    }

    for (const Row& row : Rows())
    {
        const boys::DeviceOptionInfo* info = BookRow(row.entry);

        if (info == nullptr || !info->built)
        {
            std::printf("  %s is named here and not served by this build's book\n", row.entryName);
            ++gFailures;
            continue;
        }

        const Grid& source = row.kind == Kind::kHalf ? halfGrid : grid;

        if (source.cells.empty())
        {
            std::printf("  %s has no reference cells at this precision\n", row.entryName);
            ++gFailures;
            continue;
        }

        SweepRow(row, source, census[static_cast<std::size_t>(row.precision)]);
    }

    std::printf("\nper precision, counted from the device book:\n");
    std::printf("  %-6s %8s %8s %9s %9s\n", "class", "combos", "named", "cells", "named");

    std::size_t servedRows = 0;
    std::size_t namedRows = 0;
    std::size_t servedCells = 0;
    std::size_t namedCells = 0;
    std::size_t compared = 0;
    std::size_t held = 0;
    std::size_t moved = 0;
    std::size_t judged = 0;
    std::size_t exceeded = 0;

    for (std::size_t i = 0; i < census.size(); ++i)
    {
        const auto precision = static_cast<boys::DeviceOptionPrecision>(i);
        std::printf("  %-6s %8zu %8zu %9zu %9zu\n", PrecisionName(precision), census[i].servedRows,
                    census[i].namedRows, census[i].servedCells, census[i].namedCells);
        servedRows += census[i].servedRows;
        namedRows += census[i].namedRows;
        servedCells += census[i].servedCells;
        namedCells += census[i].namedCells;
        compared += census[i].compared;
        held += census[i].held;
        moved += census[i].moved;
        judged += census[i].judged;
        exceeded += census[i].exceeded;
    }

    std::printf("combinations served %zu, named %zu; cells served %zu, named %zu\n", servedRows,
                namedRows, servedCells, namedCells);
    std::printf("values judged %zu, exceeded %zu\n", judged, exceeded);
    std::printf("bit-for-bit: %zu comparisons, %zu held, over cells where the rung moved the "
                "values: %zu\n",
                compared, held, moved);

    Require(namedRows == servedRows,
            "every launched combination the device book serves has a rung-argument name");
    Require(namedCells == servedCells,
            "every launched cell the device book serves is named and evaluated");
    Require(exceeded == 0, "every value is inside the bound the book states for its own cell");
    Require(judged > 0, "values were judged");
    Require(moved > 0, "the rung moved values, so the bit-for-bit check could have failed");

    // The refusal, on the card: this is the rule the launched entries must not
    // soften, so it is run rather than argued about.
    std::printf("\nthe resident rung, on the card:\n");

    CheckResidencyRefusal(grid, nullptr);

    if (gFailures == 0)
    {
        std::printf("PASS: every launched combination the device book serves is named with the "
                    "rung as the call's argument, every cell is evaluated at the rung it was "
                    "handed and inside its own bound, and the device entry still refuses a rung "
                    "that is not resident\n");
        return 0;
    }

    std::printf("FAIL: %zu failed check(s)\n", gFailures);
    return 1;
}
