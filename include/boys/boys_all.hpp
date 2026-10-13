#pragma once

/// \file
/// Everything this library carries, in one include.
///
/// The front door (`boys/boys.hpp`) is the calls: the entries a call site names no policy
/// for, the tables a report reads, and the accuracy queries; it carries the policy-taking
/// entries (`boys/kernels.hpp`) at its end. This header adds what sits beside them: the
/// `std::span` overloads (`boys/boys_span.hpp`), the region-A transform lane
/// (`boys/boys_transform.hpp`) and the option probe (`boys/boys_probe.hpp`).
///
/// Include this one when a translation unit wants one of those; the front door alone
/// carries every call, the policy-taking spelling included.
///
/// \ingroup boys

#include "boys/boys.hpp"
#include "boys/boys_probe.hpp"
#include "boys/boys_span.hpp"
#include "boys/boys_transform.hpp"
#include "boys/kernels.hpp"
