#pragma once

/// \file
/// The status one call returns, shared by the host lanes, the C interface and the CUDA lane, so
/// that one enum answers "did this call do what it promised" wherever the question is asked.

namespace boys {

/// What a call reports when it is asked to check its arguments.
///
/// The host entries do not check: an argument outside the contract is undefined behaviour, and it
/// is not benign - a release build faults on `x < 0` and on a `NaN` anywhere in a batch, and
/// answers a wrong number for `n > kMaxBoysOrder`. The `Boys*Checked` overloads beside each entry
/// validate instead, and return one of these. The C interface returns the same values as an `int`.
enum class BoysStatus {
    kSuccess = 0,     ///< the call ran and wrote its output
    kInvalidArgument, ///< an argument was outside the contract the entry documents
    kDeviceError,     ///< a CUDA operation failed
};

} // namespace boys
