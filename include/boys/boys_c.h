/**
 * \file boys_c.h
 * \brief C linkage surface of the Boys kernel.
 *
 * C-clean (no exceptions, no templates, no C++ headers): callable from C
 * directly, from Fortran through ISO_C_BINDING, and from other FFI consumers.
 * Every entry returns an int status; results are written through the pointer
 * arguments.
 *
 * B is the per-region bound of the lane an entry reaches, at the one accuracy
 * this library carries. The rows this surface reaches are: the single-order
 * double entries, 1e-15 in region A and 3e-14 over the extended band and
 * region B, and 5.5e-14 in region C; the double batch entries, 5.5e-14 in every
 * region; and the single-precision entries, 1.5e-7 in every region.
 *
 * \ingroup boys
 */

#ifndef BOYS_BOYS_C_H
#define BOYS_BOYS_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/** Success return code of every C entry. */
#define BOYS_SUCCESS 0
/** Invalid-argument return code (out-of-range order, negative/NaN x, NULL pointer). */
#define BOYS_ERROR_INVALID_ARGUMENT 1
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

/** F_0(x[i])..F_nmax(x[i]) in double precision for count arguments.
 *
 * Output layout: out[k * count + i] = F_k(x[i]) — order-major. x must hold
 * count elements and out must hold count * (nmax + 1); a zero count is a no-op,
 * and only then may x and out be NULL.
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

#endif /* BOYS_BOYS_C_H */
