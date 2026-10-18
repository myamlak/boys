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
/// carries every call, the policy-taking spelling included. The front door opens the
/// default coefficient groups and names the header to add for a policy outside them; this
/// header opens all of them, so a translation unit that includes it can name any policy.
///
/// \ingroup boys

#include "boys/boys_coefficients_d_coarse_cheb.hpp"
#include "boys/boys_coefficients_d_coarse_mono.hpp"
#include "boys/boys_coefficients_d_coarse_rat.hpp"
#include "boys/boys_coefficients_d_far.hpp"
#include "boys/boys_coefficients_d_narrow_cheb.hpp"
#include "boys/boys_coefficients_d_narrow_mono.hpp"
#include "boys/boys_coefficients_d_narrow_rat.hpp"
#include "boys/boys_coefficients_d_uniform_cheb.hpp"
#include "boys/boys_coefficients_d_uniform_cheb_transposed.hpp"
#include "boys/boys_coefficients_d_uniform_mono.hpp"
#include "boys/boys_coefficients_d_uniform_mono_transposed.hpp"
#include "boys/boys_coefficients_d_uniform_rat.hpp"
#include "boys/boys_coefficients_f32_coarse_cheb.hpp"
#include "boys/boys_coefficients_f32_coarse_mono.hpp"
#include "boys/boys_coefficients_f32_coarse_rat.hpp"
#include "boys/boys_coefficients_f32_narrow_cheb.hpp"
#include "boys/boys_coefficients_f32_narrow_mono.hpp"
#include "boys/boys_coefficients_f32_narrow_rat.hpp"
#include "boys/boys_coefficients_f32_uniform_cheb.hpp"
#include "boys/boys_coefficients_f32_uniform_mono.hpp"
#include "boys/boys_coefficients_f32_uniform_rat.hpp"
#include "boys/boys.hpp"
#include "boys/boys_probe.hpp"
#include "boys/boys_span.hpp"
#include "boys/boys_transform.hpp"
#include "boys/kernels.hpp"
