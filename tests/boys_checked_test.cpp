// The checked entries beside the five double-precision shapes (include/boys/boys.hpp; ctest case
// boys-checked-tests). An in-contract call hands back the unchecked entry's own bits, a call
// outside the contract is refused, a refused call writes nothing, and a refused batch names the
// element it refused. Nothing here is timed, and no value here is compared with a tolerance.

#include "boys/boys.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>

namespace {

using namespace boys;

// A bit pattern no evaluation returns, so a cell still holding it was not written. It is a quiet
// NaN as well, so a check that compared values rather than bits would fail on it too.
constexpr std::uint64_t kSentinelBits = 0x7FF8DEADBEEFCAFEull;
constexpr double kSentinel = std::bit_cast<double>(kSentinelBits);

// No index of these calls is this one, so a badIndex still holding it was not written.
constexpr std::size_t kIndexSentinel = 0x5A5A5A5Aull;

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

constexpr int kOrders[] = {0, 1, 5, kMaxBoysOrder};
constexpr double kArguments[] = {0.0, 1e-8, 0.5, 3.0, 30.0};
constexpr std::size_t kCount = std::size(kArguments);

// An order below the contract's range and one above it; an argument below zero and a NaN.
constexpr int kRefusedOrders[] = {-1, kMaxBoysOrder + 1};
constexpr double kRefusedArguments[] = {-1.0, kNaN};

constexpr std::size_t kPlanes = kCount * (kMaxBoysOrder + 1);

int checks = 0;
int failures = 0;

void Fail(const char* what) noexcept {
    std::printf("  FAILED: %s\n", what);
    ++failures;
}

void Check(bool ok, const char* what) noexcept {
    ++checks;

    if (!ok)
    {
        Fail(what);
    }
}

void CheckAt(bool ok, const char* what, int n, double x) noexcept {
    ++checks;

    if (!ok)
    {
        std::printf("  FAILED: %s (n = %d, x = %.17g)\n", what, n, x);
        ++failures;
    }
}

void CheckElement(bool ok, const char* what, std::size_t index) noexcept {
    ++checks;

    if (!ok)
    {
        std::printf("  FAILED: %s (element %zu)\n", what, index);
        ++failures;
    }
}

void CheckStrided(bool ok, const char* what, int n, std::size_t stride) noexcept {
    ++checks;

    if (!ok)
    {
        std::printf("  FAILED: %s (n = %d, stride = %zu)\n", what, n, stride);
        ++failures;
    }
}

void CheckOrder(bool ok, const char* what, int nmax) noexcept {
    ++checks;

    if (!ok)
    {
        std::printf("  FAILED: %s (nmax = %d, over the %zu arguments of kArguments)\n", what, nmax,
                    kCount);
        ++failures;
    }
}

bool SameBitsOf(double a, double b) noexcept {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

bool SameBits(const double* a, const double* b, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!SameBitsOf(a[i], b[i]))
        {
            return false;
        }
    }

    return true;
}

bool SameBitsStrided(
    const double* a, const double* b, std::size_t count, std::size_t stride) noexcept {
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!SameBitsOf(a[i * stride], b[i * stride]))
        {
            return false;
        }
    }

    return true;
}

bool Untouched(const double* buffer, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!SameBitsOf(buffer[i], kSentinel))
        {
            return false;
        }
    }

    return true;
}

/// Whether every cell the entry was not asked to write still holds the sentinel.
bool GapsUntouched(const double* buffer, std::size_t count, std::size_t stride) noexcept {
    for (std::size_t i = 0; i < count * stride; ++i)
    {
        if (i % stride != 0 && !SameBitsOf(buffer[i], kSentinel))
        {
            return false;
        }
    }

    return true;
}

/// The in-contract arguments with one element replaced by a refused value.
std::array<double, kCount> ArgumentsWith(std::size_t bad, double value) noexcept {
    std::array<double, kCount> args{};

    for (std::size_t i = 0; i < kCount; ++i)
    {
        args[i] = kArguments[i];
    }

    args[bad] = value;
    return args;
}

// [1] Each checked entry accepts what its unchecked twin accepts and writes its twin's bits.
void Agreement() {
    std::printf("[1] in contract: the checked call returns the unchecked entry's bits\n");

    for (const int n : kOrders)
    {
        for (const double x : kArguments)
        {
            double got = kSentinel;
            const BoysStatus status = BoysSingleChecked(n, x, &got);

            CheckAt(status == BoysStatus::kSuccess,
                    "BoysSingleChecked refused an in-contract call", n, x);
            CheckAt(SameBitsOf(got, BoysSingle(n, x)),
                    "BoysSingleChecked wrote other bits than BoysSingle", n, x);
        }
    }

    std::array<double, kMaxBoysOrder + 1> gotAll{};
    std::array<double, kMaxBoysOrder + 1> twinAll{};

    for (const int nmax : kOrders)
    {
        for (const double x : kArguments)
        {
            gotAll.fill(kSentinel);
            twinAll.fill(kSentinel);

            const BoysStatus status = BoysAllOrdersChecked(nmax, x, gotAll.data());
            BoysAllOrders(nmax, x, twinAll.data());
            const std::size_t written = static_cast<std::size_t>(nmax) + 1;

            CheckAt(status == BoysStatus::kSuccess,
                    "BoysAllOrdersChecked refused an in-contract call", nmax, x);
            CheckAt(SameBits(gotAll.data(), twinAll.data(), written),
                    "BoysAllOrdersChecked wrote other bits than BoysAllOrders", nmax, x);
        }
    }

    constexpr std::size_t kStrides[] = {1, 2};
    std::array<double, kCount * 2> gotFixed{};
    std::array<double, kCount * 2> twinFixed{};

    for (const int n : kOrders)
    {
        for (const std::size_t stride : kStrides)
        {
            gotFixed.fill(kSentinel);
            twinFixed.fill(kSentinel);

            const BoysStatus status =
                BoysFixedNChecked(n, kArguments, gotFixed.data(), kCount, stride);
            BoysFixedN(n, kArguments, twinFixed.data(), kCount, stride);

            CheckStrided(status == BoysStatus::kSuccess,
                         "BoysFixedNChecked refused an in-contract call", n, stride);
            CheckStrided(SameBitsStrided(gotFixed.data(), twinFixed.data(), kCount, stride),
                         "BoysFixedNChecked wrote other bits than BoysFixedN", n, stride);
            CheckStrided(GapsUntouched(gotFixed.data(), kCount, stride),
                         "BoysFixedNChecked wrote between the strided outputs", n, stride);
        }
    }

    std::array<double, kPlanes> gotPlanes{};
    std::array<double, kPlanes> twinPlanes{};

    for (const int nmax : kOrders)
    {
        gotPlanes.fill(kSentinel);
        twinPlanes.fill(kSentinel);

        const BoysStatus status = BoysAllNChecked(nmax, kArguments, gotPlanes.data(), kCount);
        BoysAllN(nmax, kArguments, twinPlanes.data(), kCount);
        const std::size_t written = kCount * (static_cast<std::size_t>(nmax) + 1);

        CheckOrder(status == BoysStatus::kSuccess,
                   "BoysAllNChecked refused an in-contract call", nmax);
        CheckOrder(SameBits(gotPlanes.data(), twinPlanes.data(), written),
                   "BoysAllNChecked wrote other bits than BoysAllN", nmax);
    }

    // Where the grouping scratch lives is the caller's choice and not a value's.
    constexpr int kScratchOrder = 5;
    std::array<std::size_t, BoysAllNWorkspaceSize(kCount)> gotScratch{};
    std::array<std::size_t, BoysAllNWorkspaceSize(kCount)> twinScratch{};

    gotPlanes.fill(kSentinel);
    twinPlanes.fill(kSentinel);

    const BoysStatus scratchStatus =
        BoysAllNChecked(kScratchOrder, kArguments, gotPlanes.data(), kCount, gotScratch.data());
    BoysAllN(kScratchOrder, kArguments, twinPlanes.data(), kCount, twinScratch.data());

    Check(scratchStatus == BoysStatus::kSuccess,
          "BoysAllNChecked refused an in-contract call with a workspace");
    Check(SameBits(gotPlanes.data(), twinPlanes.data(),
                   kCount * (static_cast<std::size_t>(kScratchOrder) + 1)),
          "BoysAllNChecked's caller-supplied workspace changed the values");

    constexpr int kAtOrders[] = {kMaxBoysOrder, 1, 5, 7, 2};

    gotPlanes.fill(kSentinel);
    twinPlanes.fill(kSentinel);

    const BoysStatus atStatus =
        BoysAllNAtOrdersChecked(kAtOrders, kArguments, gotPlanes.data(), kCount);
    BoysAllNAtOrders(kAtOrders, kArguments, twinPlanes.data(), kCount);
    const std::size_t atWritten = kCount * (static_cast<std::size_t>(kMaxBoysOrder) + 1);

    Check(atStatus == BoysStatus::kSuccess,
          "BoysAllNAtOrdersChecked refused an in-contract call");
    Check(SameBits(gotPlanes.data(), twinPlanes.data(), atWritten),
          "BoysAllNAtOrdersChecked wrote other bits than BoysAllNAtOrders");
}

// [2] An order outside 0..kMaxBoysOrder, an argument below zero and a NaN are refused; [3] a
// refused call writes nothing at all, whichever of the two was refused.
void Refusals() {
    std::printf("[2] a refused call returns kInvalidArgument\n");
    std::printf("[3] a refused call writes nothing\n");

    for (const int n : kRefusedOrders)
    {
        double got = kSentinel;
        const BoysStatus status = BoysSingleChecked(n, 1.0, &got);

        Check(status == BoysStatus::kInvalidArgument,
              "BoysSingleChecked accepted an order outside 0..kMaxBoysOrder");
        Check(SameBitsOf(got, kSentinel), "a refused BoysSingleChecked wrote its output");
    }

    for (const double x : kRefusedArguments)
    {
        double got = kSentinel;
        const BoysStatus status = BoysSingleChecked(3, x, &got);

        Check(status == BoysStatus::kInvalidArgument,
              "BoysSingleChecked accepted a negative or NaN argument");
        Check(SameBitsOf(got, kSentinel), "a refused BoysSingleChecked wrote its output");
    }

    std::array<double, kMaxBoysOrder + 1> all{};

    for (const int nmax : kRefusedOrders)
    {
        all.fill(kSentinel);
        const BoysStatus status = BoysAllOrdersChecked(nmax, 1.0, all.data());

        Check(status == BoysStatus::kInvalidArgument,
              "BoysAllOrdersChecked accepted an order outside 0..kMaxBoysOrder");
        Check(Untouched(all.data(), all.size()),
              "a refused BoysAllOrdersChecked wrote its output");
    }

    for (const double x : kRefusedArguments)
    {
        all.fill(kSentinel);
        const BoysStatus status = BoysAllOrdersChecked(3, x, all.data());

        Check(status == BoysStatus::kInvalidArgument,
              "BoysAllOrdersChecked accepted a negative or NaN argument");
        Check(Untouched(all.data(), all.size()),
              "a refused BoysAllOrdersChecked wrote its output");
    }

    std::array<double, kCount * 2> fixed{};

    for (const int n : kRefusedOrders)
    {
        fixed.fill(kSentinel);
        const BoysStatus status = BoysFixedNChecked(n, kArguments, fixed.data(), kCount, 1);

        Check(status == BoysStatus::kInvalidArgument,
              "BoysFixedNChecked accepted an order outside 0..kMaxBoysOrder");
        Check(Untouched(fixed.data(), fixed.size()), "a refused BoysFixedNChecked wrote its output");
    }

    for (const double refused : kRefusedArguments)
    {
        const std::array<double, kCount> args = ArgumentsWith(2, refused);

        fixed.fill(kSentinel);
        const BoysStatus status = BoysFixedNChecked(3, args.data(), fixed.data(), kCount, 1);

        CheckElement(status == BoysStatus::kInvalidArgument,
                     "BoysFixedNChecked accepted a negative or NaN argument", 2);
        Check(Untouched(fixed.data(), fixed.size()), "a refused BoysFixedNChecked wrote its output");
    }

    for (const int nmax : kRefusedOrders)
    {
        std::array<double, kPlanes> planes{};
        planes.fill(kSentinel);
        const BoysStatus status = BoysAllNChecked(nmax, kArguments, planes.data(), kCount);

        Check(status == BoysStatus::kInvalidArgument,
              "BoysAllNChecked accepted an order outside 0..kMaxBoysOrder");
        Check(Untouched(planes.data(), planes.size()), "a refused BoysAllNChecked wrote its output");
    }

    for (const double refused : kRefusedArguments)
    {
        const std::array<double, kCount> args = ArgumentsWith(2, refused);
        std::array<double, kPlanes> planes{};

        planes.fill(kSentinel);
        const BoysStatus status = BoysAllNChecked(3, args.data(), planes.data(), kCount);

        CheckElement(status == BoysStatus::kInvalidArgument,
                     "BoysAllNChecked accepted a negative or NaN argument", 2);
        Check(Untouched(planes.data(), planes.size()), "a refused BoysAllNChecked wrote its output");
    }

    std::array<int, kCount> orders{};
    orders.fill(3);

    for (const int refused : kRefusedOrders)
    {
        std::array<int, kCount> bad = orders;
        std::array<double, kPlanes> planes{};

        bad[2] = refused;
        planes.fill(kSentinel);
        const BoysStatus status =
            BoysAllNAtOrdersChecked(bad.data(), kArguments, planes.data(), kCount);

        Check(status == BoysStatus::kInvalidArgument,
              "BoysAllNAtOrdersChecked accepted an order outside 0..kMaxBoysOrder");
        Check(Untouched(planes.data(), planes.size()),
              "a refused BoysAllNAtOrdersChecked wrote its output");
    }

    for (const double refused : kRefusedArguments)
    {
        const std::array<double, kCount> args = ArgumentsWith(2, refused);
        std::array<double, kPlanes> planes{};

        planes.fill(kSentinel);
        const BoysStatus status =
            BoysAllNAtOrdersChecked(orders.data(), args.data(), planes.data(), kCount);

        CheckElement(status == BoysStatus::kInvalidArgument,
                     "BoysAllNAtOrdersChecked accepted a negative or NaN argument", 2);
        Check(Untouched(planes.data(), planes.size()),
              "a refused BoysAllNAtOrdersChecked wrote its output");
    }
}

// [4] A refused batch names the first element it refused, and a served batch leaves the index be.
void BadIndex() {
    std::printf("[4] a refused batch names the first refused element, and a served one does not\n");

    std::array<double, kCount * 2> fixed{};
    std::array<double, kPlanes> planes{};
    std::array<int, kCount> orders{};
    orders.fill(3);

    // Two refused elements, the later one first in the array: the index reported is the first.
    std::array<double, kCount> twoBad = ArgumentsWith(3, kNaN);
    twoBad[1] = -1.0;

    std::size_t badIndex = kIndexSentinel;
    fixed.fill(kSentinel);
    BoysStatus status = BoysFixedNChecked(3, twoBad.data(), fixed.data(), kCount, 1, &badIndex);
    Check(status == BoysStatus::kInvalidArgument, "BoysFixedNChecked served a refused argument");
    Check(badIndex == 1, "BoysFixedNChecked named an element other than the first it refused");

    const std::array<double, kCount> lastBad = ArgumentsWith(4, kNaN);
    badIndex = kIndexSentinel;
    fixed.fill(kSentinel);
    status = BoysFixedNChecked(3, lastBad.data(), fixed.data(), kCount, 1, &badIndex);
    Check(status == BoysStatus::kInvalidArgument, "BoysFixedNChecked served a refused argument");
    Check(badIndex == 4, "BoysFixedNChecked named an element other than the one it refused");

    badIndex = kIndexSentinel;
    fixed.fill(kSentinel);
    status = BoysFixedNChecked(3, kArguments, fixed.data(), kCount, 1, &badIndex);
    Check(status == BoysStatus::kSuccess, "BoysFixedNChecked refused an in-contract call");
    Check(badIndex == kIndexSentinel, "a served BoysFixedNChecked wrote badIndex");

    badIndex = kIndexSentinel;
    planes.fill(kSentinel);
    status = BoysAllNChecked(3, twoBad.data(), planes.data(), kCount, nullptr, &badIndex);
    Check(status == BoysStatus::kInvalidArgument, "BoysAllNChecked served a refused argument");
    Check(badIndex == 1, "BoysAllNChecked named an element other than the first it refused");

    badIndex = kIndexSentinel;
    planes.fill(kSentinel);
    status = BoysAllNChecked(3, lastBad.data(), planes.data(), kCount, nullptr, &badIndex);
    Check(status == BoysStatus::kInvalidArgument, "BoysAllNChecked served a refused argument");
    Check(badIndex == 4, "BoysAllNChecked named an element other than the one it refused");

    badIndex = kIndexSentinel;
    planes.fill(kSentinel);
    status = BoysAllNChecked(3, kArguments, planes.data(), kCount, nullptr, &badIndex);
    Check(status == BoysStatus::kSuccess, "BoysAllNChecked refused an in-contract call");
    Check(badIndex == kIndexSentinel, "a served BoysAllNChecked wrote badIndex");

    // The order of an element is that element's, so an out-of-range order is named the same way.
    std::array<int, kCount> badOrder = orders;
    badOrder[2] = kMaxBoysOrder + 1;

    badIndex = kIndexSentinel;
    planes.fill(kSentinel);
    status = BoysAllNAtOrdersChecked(badOrder.data(), kArguments, planes.data(), kCount, &badIndex);
    Check(status == BoysStatus::kInvalidArgument, "BoysAllNAtOrdersChecked served a refused order");
    Check(badIndex == 2,
          "BoysAllNAtOrdersChecked named an element other than the first it refused");

    badIndex = kIndexSentinel;
    planes.fill(kSentinel);
    status = BoysAllNAtOrdersChecked(orders.data(), twoBad.data(), planes.data(), kCount, &badIndex);
    Check(status == BoysStatus::kInvalidArgument,
          "BoysAllNAtOrdersChecked served a refused argument");
    Check(badIndex == 1,
          "BoysAllNAtOrdersChecked named an element other than the first it refused");

    badIndex = kIndexSentinel;
    planes.fill(kSentinel);
    status = BoysAllNAtOrdersChecked(orders.data(), kArguments, planes.data(), kCount, &badIndex);
    Check(status == BoysStatus::kSuccess, "BoysAllNAtOrdersChecked refused an in-contract call");
    Check(badIndex == kIndexSentinel, "a served BoysAllNAtOrdersChecked wrote badIndex");
}

// [5] The top order at a zero argument, and an infinite argument, are inside the contract.
void Boundary() {
    std::printf("[5] the boundary is in contract: the top order at zero, and an infinite "
                "argument\n");

    std::array<double, kCount> zeros{};
    std::array<double, kCount> infinities{};
    std::array<int, kCount> top{};
    infinities.fill(kInfinity);
    top.fill(kMaxBoysOrder);

    double single = kSentinel;
    Check(BoysSingleChecked(kMaxBoysOrder, 0.0, &single) == BoysStatus::kSuccess,
          "BoysSingleChecked refused the top order at x = 0");
    Check(BoysSingleChecked(0, kInfinity, &single) == BoysStatus::kSuccess,
          "BoysSingleChecked refused x = +inf");

    std::array<double, kMaxBoysOrder + 1> all{};
    Check(BoysAllOrdersChecked(kMaxBoysOrder, 0.0, all.data()) == BoysStatus::kSuccess,
          "BoysAllOrdersChecked refused the top order at x = 0");
    Check(BoysAllOrdersChecked(0, kInfinity, all.data()) == BoysStatus::kSuccess,
          "BoysAllOrdersChecked refused x = +inf");

    std::array<double, kCount * 2> fixed{};
    Check(BoysFixedNChecked(kMaxBoysOrder, zeros.data(), fixed.data(), kCount, 1) ==
              BoysStatus::kSuccess,
          "BoysFixedNChecked refused the top order at x = 0");
    Check(BoysFixedNChecked(0, infinities.data(), fixed.data(), kCount, 1) == BoysStatus::kSuccess,
          "BoysFixedNChecked refused x = +inf");

    std::array<double, kPlanes> planes{};
    Check(BoysAllNChecked(kMaxBoysOrder, zeros.data(), planes.data(), kCount) ==
              BoysStatus::kSuccess,
          "BoysAllNChecked refused the top order at x = 0");
    Check(BoysAllNChecked(0, infinities.data(), planes.data(), kCount) == BoysStatus::kSuccess,
          "BoysAllNChecked refused x = +inf");

    Check(BoysAllNAtOrdersChecked(top.data(), zeros.data(), planes.data(), kCount) ==
              BoysStatus::kSuccess,
          "BoysAllNAtOrdersChecked refused the top order at x = 0");
    Check(BoysAllNAtOrdersChecked(top.data(), infinities.data(), planes.data(), kCount) ==
              BoysStatus::kSuccess,
          "BoysAllNAtOrdersChecked refused x = +inf");
}

// [6] An output pointer of null is refused, whichever shape asked for it.
void NullOutput() {
    std::printf("[6] a null output pointer is refused\n");

    std::array<int, kCount> orders{};
    orders.fill(3);

    Check(BoysSingleChecked(1, 1.0, nullptr) == BoysStatus::kInvalidArgument,
          "BoysSingleChecked accepted a null output");
    Check(BoysAllOrdersChecked(1, 1.0, nullptr) == BoysStatus::kInvalidArgument,
          "BoysAllOrdersChecked accepted a null output");
    Check(BoysFixedNChecked(1, kArguments, nullptr, kCount, 1) == BoysStatus::kInvalidArgument,
          "BoysFixedNChecked accepted a null output");
    Check(BoysAllNChecked(1, kArguments, nullptr, kCount) == BoysStatus::kInvalidArgument,
          "BoysAllNChecked accepted a null output");
    Check(BoysAllNAtOrdersChecked(orders.data(), kArguments, nullptr, kCount) ==
              BoysStatus::kInvalidArgument,
          "BoysAllNAtOrdersChecked accepted a null output");
}

// [7] What the unchecked entries' contract accepts, the checked ones accept. A batch of no elements
//     reads and writes nothing, so it is in contract with no buffers at all; a stride is not, since
//     the layout requires one of at least 1 and stride = 0 would write every element to out[0] and
//     report success.
void BatchContract() {
    std::printf("[7] a batch of no elements, and a stride below the layout's floor\n");

    Check(BoysFixedNChecked(1, nullptr, nullptr, 0, 1) == BoysStatus::kSuccess,
          "BoysFixedNChecked refused an empty batch with no buffers");
    Check(BoysAllNChecked(1, nullptr, nullptr, 0) == BoysStatus::kSuccess,
          "BoysAllNChecked refused an empty batch with no buffers");
    Check(BoysAllNAtOrdersChecked(nullptr, nullptr, nullptr, 0) == BoysStatus::kSuccess,
          "BoysAllNAtOrdersChecked refused an empty batch with no buffers");

    std::array<double, kCount> out{};
    out.fill(kSentinel);

    Check(BoysFixedNChecked(1, kArguments, out.data(), kCount, 0) == BoysStatus::kInvalidArgument,
          "BoysFixedNChecked accepted a stride of 0");

    for (std::size_t i = 0; i < kCount; ++i)
    {
        CheckElement(SameBitsOf(out[i], kSentinel), "wrote through a refused stride", i);
    }
}

} // namespace

int main() {
    Agreement();
    Refusals();
    BadIndex();
    Boundary();
    NullOutput();
    BatchContract();

    if (failures != 0)
    {
        std::printf("%d of %d checks failed\n", failures, checks);
        return 1;
    }

    std::printf("all %d checks passed: the checked entries agree with their unchecked twins bit for "
                "bit, refuse what is out of contract, write nothing when they refuse, name the "
                "element they refused, and hold the boundary, the null output and the batch "
                "contract\n",
                checks);

    return 0;
}
