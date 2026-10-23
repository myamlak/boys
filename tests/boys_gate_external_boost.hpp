// The external-row application of the accuracy gate: Boost.Math's documented Airy accuracy
// sentence, measured with the same instrument. Header-only; active only when
// BOYS_GATE_EXTERNAL_BOOST is defined.
#ifdef BOYS_GATE_EXTERNAL_BOOST

#ifndef BOYS_GATE_EXTERNAL_BOOST_HPP
#define BOYS_GATE_EXTERNAL_BOOST_HPP

#include <boost/math/special_functions/airy.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/version.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace boost_gate_external
{

// 2^-52, the epsilon Boost.Math's own accuracy tables are stated in.
inline constexpr double kEps = std::numeric_limits<double>::epsilon();

// The documented figure: "less than 100 epsilon".
inline constexpr double kClaimEps = 100.0;

// The audit's reported worst cells; 78.12521875 is its linear:65 grid cell that prints as
// 78.1252, and that literal is a witness too.
inline constexpr double kAuditAiX = 35.9;
inline constexpr double kAuditBiX = 78.12521875;

// The dense scan's resolution per function, roughly eight times the audit's 65-point grid.
inline constexpr std::size_t kScanPoints = 512;

// Significant digits kept from the cpp_bin_float_100 reference.
inline constexpr unsigned kReferenceDigits = 30;

// One measured argument.
struct AiryCell
{
    double x = 0.0;          ///< the argument
    double eps = 0.0;        ///< relative difference from the reference, in 2^-52 units
    bool breached = false;   ///< eps > kClaimEps
    std::string reference;   ///< cpp_bin_float_100 value at x, kReferenceDigits significant digits
};

// One function's witness set, after it was evaluated.
struct AiryFunctionWitness
{
    AiryCell worst;          ///< the worst cell of the whole witness set
    AiryCell audit;          ///< the audit's own reported argument, measured in this run
    std::size_t points = 0;  ///< witness arguments evaluated
};

// Both functions of the documented sentence.
struct AiryWitness
{
    AiryFunctionWitness ai;
    AiryFunctionWitness bi;
    std::size_t points = 0;  ///< witness arguments evaluated across both functions
};

namespace detail
{

using Ref = boost::multiprecision::cpp_bin_float_100;

// Boost's own relative_difference(got, reference) / epsilon, at 100 digits.
inline double RelativeErrorEps(double got, const Ref& reference)
{
    return static_cast<double>(boost::math::relative_difference(got, reference) / Ref(kEps));
}

inline void AppendLinear(std::vector<double>& xs, double lo, double hi, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        xs.push_back(lo
                     + (hi - lo) * static_cast<double>(i) / static_cast<double>(count - 1));
    }
}

inline void Dedupe(std::vector<double>& xs)
{
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
}

// The two evaluations are one transcription of the call: double for the delivered value, Ref
// for the reference.
template <class Function>
AiryFunctionWitness MeasureOne(Function function,
                               const std::vector<double>& arguments,
                               double auditX)
{
    AiryFunctionWitness out;
    out.points = arguments.size();

    for (const double x : arguments)
    {
        const Ref reference = function(Ref(x));
        const double eps = RelativeErrorEps(function(x), reference);

        if (eps > out.worst.eps)
        {
            out.worst.x = x;
            out.worst.eps = eps;
            out.worst.breached = eps > kClaimEps;
            out.worst.reference = reference.str(kReferenceDigits);
        }
    }

    const Ref auditReference = function(Ref(auditX));
    out.audit.x = auditX;
    out.audit.eps = RelativeErrorEps(function(auditX), auditReference);
    out.audit.breached = out.audit.eps > kClaimEps;
    out.audit.reference = auditReference.str(kReferenceDigits);

    return out;
}

} // namespace detail

// The Boost version the figure is stated at, so a row prints the number with its revision.
inline std::string VersionString()
{
    std::string version = BOOST_LIB_VERSION;
    std::replace(version.begin(), version.end(), '_', '.');
    return version;
}

// The witness set: the audit's grids and published arguments plus a dense scan on (0, 35.9]
// for ai and (0, 100] for bi, the intervals the audit's rows sample.
inline AiryWitness MeasureBoostAiry()
{
    const auto airyAi = [](auto x) { return boost::math::airy_ai(x); };
    const auto airyBi = [](auto x) { return boost::math::airy_bi(x); };

    std::vector<double> aiArguments;
    detail::AppendLinear(aiArguments, 0.001, 35.9, kScanPoints);
    detail::AppendLinear(aiArguments, 0.001, 30.0, 65);
    detail::AppendLinear(aiArguments, 30.0, 35.9, 33);
    aiArguments.push_back(kAuditAiX);
    detail::Dedupe(aiArguments);

    std::vector<double> biArguments;
    detail::AppendLinear(biArguments, 0.001, 100.0, kScanPoints);
    detail::AppendLinear(biArguments, 0.001, 100.0, 65);
    biArguments.push_back(78.1252);
    biArguments.push_back(kAuditBiX);
    detail::Dedupe(biArguments);

    AiryWitness witness;
    witness.ai = detail::MeasureOne(airyAi, aiArguments, kAuditAiX);
    witness.bi = detail::MeasureOne(airyBi, biArguments, kAuditBiX);
    witness.points = witness.ai.points + witness.bi.points;

    return witness;
}

} // namespace boost_gate_external

#endif // BOYS_GATE_EXTERNAL_BOOST_HPP
#endif // BOYS_GATE_EXTERNAL_BOOST
