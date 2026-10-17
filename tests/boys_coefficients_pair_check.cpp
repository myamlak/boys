// The uniform grid's transposed twin, held to the interval-major table it permutes. The check
// indexes both tables at compile time, so it lives in the one translation unit written to
// include both: the lane header asserted it before, which required both tables wherever the
// lane was parsed. The generator's flat_transposed_findings checks the same relation.

#include "boys/boys_coefficients.hpp"
#include "boys/boys_coefficients_d_uniform_cheb.hpp"
#include "boys/boys_coefficients_d_uniform_cheb_transposed.hpp"

namespace {

namespace detail = boys::detail;

// The twin's k-th coefficient of order l is the interval-major table's k-th coefficient of that
// order: the coefficient index steps by kFlatTwinRow and the order by one.
static_assert(detail::kFlatTransposedCoeffs[0] == detail::kFlatCoeffs[0] &&
                  detail::kFlatTransposedCoeffs[detail::kFlatTwinRow] == detail::kFlatCoeffs[1] &&
                  detail::kFlatTransposedCoeffs[1] ==
                      detail::kFlatCoeffs[detail::kFlatDegs[0] + 1],
              "the twin's k-th coefficient of order l is the interval-major table's k-th "
              "coefficient of that order: the coefficient index steps by kFlatTwinRow and the "
              "order by one");

} // namespace
