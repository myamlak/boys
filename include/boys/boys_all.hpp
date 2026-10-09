#pragma once

/// \file
/// Everything this library carries, in one include.
///
/// The front door (`boys/boys.hpp`) is the calls: the entries a call site names no policy
/// for, the tables a report reads, and the accuracy queries. This header adds what a caller
/// reaches for beyond them - the policy-taking entries (`boys/kernels.hpp`), the `std::span`
/// overloads (`boys/boys_span.hpp`), the region-A transform lane (`boys/boys_transform.hpp`)
/// and the option probe (`boys/boys_probe.hpp`).
///
/// **Include this one when a call site names a policy.** The front door declares no
/// policy-taking entry at all, so a call naming one from a translation unit that included
/// only the front door does not compile - which is the property, and not an omission: the
/// compiler is what keeps a caller from believing it asked for one arithmetic while the
/// library answered with another.
///
/// \ingroup boys

#include "boys/boys.hpp"
#include "boys/boys_probe.hpp"
#include "boys/boys_span.hpp"
#include "boys/boys_transform.hpp"
#include "boys/kernels.hpp"
