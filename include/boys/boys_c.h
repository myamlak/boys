/**
 * \file boys_c.h
 * \brief C linkage surface of the Boys kernel.
 *
 * C-clean (no exceptions, no templates, no C++ headers): callable from C
 * directly, from Fortran through ISO_C_BINDING, and from other FFI
 * consumers. Every entry returns an int status; results are written
 * through the pointer arguments.
 *
 * The multiplier entries dispatch on m by exact double equality over the
 * sampled set {1.0, 2.0, 10.0, 100.0, 1e4, 1e8}; m = 1.0 is the certified
 * full-accuracy lane, larger m relax the asserted error bound to m * B via
 * compile-time Chebyshev degree truncation. Any other m is rejected with
 * BOYS_ERROR_UNSUPPORTED_MULTIPLIER.
 *
 * \ingroup boys
 */

#ifndef BOYSYMETRIAD_BOYS_C_H
#define BOYSYMETRIAD_BOYS_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/** Success return code of every C entry. */
#define BOYS_SUCCESS 0
/** Invalid-argument return code (out-of-range order, negative/NaN x, NULL pointer). */
#define BOYS_ERROR_INVALID_ARGUMENT 1
/** Return code when m is not one of the sampled multiplier set. */
#define BOYS_ERROR_UNSUPPORTED_MULTIPLIER 2

/** Highest Boys order supported by the kernel (F_0..F_BOYS_MAX_ORDER). */
#define BOYS_MAX_ORDER 32

/** F_n(x) in double precision; writes *out = F_n(x).
 *
 * \param n   order, 0..BOYS_MAX_ORDER
 * \param x   argument, >= 0
 * \param out receives the value
 * \returns BOYS_SUCCESS, or BOYS_ERROR_INVALID_ARGUMENT when n is outside
 * [0, BOYS_MAX_ORDER], x is negative or NaN, or out is NULL.
 */
int BoysDouble(int n, double x, double* out);

/** F_n(x) in single precision; contract as BoysDouble.
 *
 * \param n   order, 0..BOYS_MAX_ORDER
 * \param x   argument, >= 0
 * \param out receives the value
 * \returns BOYS_SUCCESS, or BOYS_ERROR_INVALID_ARGUMENT when n is outside
 * [0, BOYS_MAX_ORDER], x is negative or NaN, or out is NULL.
 */
int BoysFloat(int n, float x, float* out);

/** F_n(x) in double precision at the relaxed accuracy multiplier m.
 *
 * \param m   accuracy multiplier; exact equality over the sampled set
 *            {1.0, 2.0, 10.0, 100.0, 1e4, 1e8}
 * \param n   order, 0..BOYS_MAX_ORDER
 * \param x   argument, >= 0
 * \param out receives the value
 * \returns BOYS_SUCCESS, or BOYS_ERROR_INVALID_ARGUMENT when n is outside
 * [0, BOYS_MAX_ORDER], x is negative or NaN, or out is NULL;
 * BOYS_ERROR_UNSUPPORTED_MULTIPLIER when m is not in the sampled set.
 */
int BoysDoubleWithMultiplier(double m, int n, double x, double* out);

/** F_n(x) in single precision at the relaxed accuracy multiplier m.
 *
 * \param m   accuracy multiplier; exact equality over the sampled set
 *            {1.0, 2.0, 10.0, 100.0, 1e4, 1e8}
 * \param n   order, 0..BOYS_MAX_ORDER
 * \param x   argument, >= 0
 * \param out receives the value
 * \returns BOYS_SUCCESS, or BOYS_ERROR_INVALID_ARGUMENT when n is outside
 * [0, BOYS_MAX_ORDER], x is negative or NaN, or out is NULL;
 * BOYS_ERROR_UNSUPPORTED_MULTIPLIER when m is not in the sampled set.
 */
int BoysFloatWithMultiplier(double m, int n, float x, float* out);

/** F_0(x[i])..F_nmax(x[i]) in double precision for count arguments.
 *
 * Output layout: out[k * count + i] = F_k(x[i]) — order-major, matching the
 * batch lanes of the C++ surface. x and out must hold count elements each;
 * a zero count is a no-op.
 *
 * \param nmax  highest order, 0..BOYS_MAX_ORDER
 * \param count number of arguments
 * \param x     arguments, each >= 0
 * \param out   receives count * (nmax + 1) values
 * \returns BOYS_SUCCESS, or BOYS_ERROR_INVALID_ARGUMENT when nmax is outside
 * [0, BOYS_MAX_ORDER], count < 0, (count > 0 and (x or out is NULL)), or any
 * x[i] is negative or NaN.
 */
int BoysDoubleBatch(int nmax, int count, const double* x, double* out);

/** F_0(x[i])..F_nmax(x[i]) in single precision; contract as BoysDoubleBatch.
 *
 * Routes to the library's single-precision all-N batch entry, so the C caller
 * and the C++ caller receive the same values and the same lane contract.
 *
 * \param nmax  highest order, 0..BOYS_MAX_ORDER
 * \param count number of arguments
 * \param x     arguments, each >= 0
 * \param out   receives count * (nmax + 1) values
 * \returns BOYS_SUCCESS, or BOYS_ERROR_INVALID_ARGUMENT when nmax is outside
 * [0, BOYS_MAX_ORDER], count < 0, (count > 0 and (x or out is NULL)), or any
 * x[i] is negative or NaN.
 */
int BoysFloatBatch(int nmax, int count, const float* x, float* out);

/** F_0(x[i])..F_n[i](x[i]) in double precision for count arguments, the top
 * order of each argument supplied as an array.
 *
 * Output layout: out[k * count + i] = F_k(x[i]) for k = 0..n[i] - the
 * order-major planes of BoysDoubleBatch, with each argument's column stopping
 * at that argument's own top order. Cells above an argument's top order are
 * left as the caller left them. x must hold count elements; out must hold
 * count * (1 + the largest n[i]) elements; a zero count is a no-op.
 *
 * This is the C surface's entry of the shape a shell-quartet caller has, where
 * each quartet of a batch carries its own highest order: it needs no padding of
 * the arguments to a common top order. The C++ surface's spelling of the same
 * call is boys::BoysAllNAtOrders.
 *
 * \param n     top order per argument, each 0..BOYS_MAX_ORDER
 * \param count number of arguments
 * \param x     arguments, each >= 0
 * \param out   receives the planes
 * \returns BOYS_SUCCESS, or BOYS_ERROR_INVALID_ARGUMENT when count < 0,
 * (count > 0 and (n, x or out is NULL)), or any n[i] is outside
 * [0, BOYS_MAX_ORDER], or any x[i] is negative or NaN.
 */
int BoysDoubleBatchAtOrders(const int* n, int count, const double* x, double* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOYSYMETRIAD_BOYS_C_H */
