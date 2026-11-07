// The carried tiers' entry list: one row per front-door entry the dispatcher serves.
//
// The file is included more than once in one translation unit - the dispatcher includes it three
// times, once per namespace it expands into - so it carries NO include guard: a guard would leave
// the second and third expansions empty. The BOYS_TIER_ENTRY check below refuses an include that
// forgot to define the macro, and the file undefines it at the end so the next include starts from
// a stated state.
//
// A dispatch build compiles the same entry once per tier and forwards to the selected one. The
// rows are written here once and expanded three ways - the dispatcher's forwarding definitions and
// each tier unit's noinline definitions - so a new entry cannot leave one of the three out.
//
// The header is internal: it is not included by boys/boys.hpp and is not part of the public API.
// A unit includes it inside a namespace it has already opened, with BOYS_TIER_ENTRY defined:
//
//     BOYS_TIER_ENTRY(RET, NAME, PARAMS, ARGS, PRECISION, SHAPE)
//
// PARAMS is the parameter list as a definition spells it (types and names, no defaults; a
// declaration that carried a default is already visible), ARGS the argument names in the same
// order. PRECISION and SHAPE are the two axes the entry's default policy names, so a tier unit's
// definition is one call to the public template at the documented default row.

#if !defined(BOYS_TIER_ENTRY)
#error "boys/detail/isa_tier_entries.hpp is an X-macro list: define BOYS_TIER_ENTRY(RET, NAME, \
PARAMS, ARGS, PRECISION, SHAPE) before including it"
#endif

// --- double, one argument at a time -----------------------------------------

BOYS_TIER_ENTRY(double,
                BoysSingle,
                (int n, double x),
                (n, x),
                Precision::kFp64,
                Shape::kSingle)

BOYS_TIER_ENTRY(BoysStatus,
                BoysSingleChecked,
                (int n, double x, double* out),
                (n, x, out),
                Precision::kFp64,
                Shape::kSingle)

BOYS_TIER_ENTRY(void,
                BoysAllOrders,
                (int nmax, double x, double* out),
                (nmax, x, out),
                Precision::kFp64,
                Shape::kAllOrders)

// --- double, many arguments -------------------------------------------------

BOYS_TIER_ENTRY(void,
                BoysFixedN,
                (int n, const double* x, double* out, std::size_t count, std::size_t stride),
                (n, x, out, count, stride),
                Precision::kFp64,
                Shape::kFixedN)

BOYS_TIER_ENTRY(void,
                BoysAllN,
                (int nmax,
                 const double* x,
                 double* out,
                 std::size_t count,
                 std::size_t* workspace),
                (nmax, x, out, count, workspace),
                Precision::kFp64,
                Shape::kAllN)

// The already-sorted overload of the same template: the tag argument is the whole of the
// difference at the call, and the definition below passes it as a value.
BOYS_TIER_ENTRY(void,
                BoysAllN,
                (int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs),
                (nmax, x, out, count, BoysSortedArgs{}),
                Precision::kFp64,
                Shape::kAllN)

BOYS_TIER_ENTRY(void,
                BoysAllNAtOrders,
                (const int* n, const double* x, double* out, std::size_t count),
                (n, x, out, count),
                Precision::kFp64,
                Shape::kAllNAtOrders)

// --- the checked entries ----------------------------------------------------

BOYS_TIER_ENTRY(BoysStatus,
                BoysAllOrdersChecked,
                (int nmax, double x, double* out),
                (nmax, x, out),
                Precision::kFp64,
                Shape::kAllOrders)

BOYS_TIER_ENTRY(BoysStatus,
                BoysFixedNChecked,
                (int n,
                 const double* x,
                 double* out,
                 std::size_t count,
                 std::size_t stride,
                 std::size_t* badIndex),
                (n, x, out, count, stride, badIndex),
                Precision::kFp64,
                Shape::kFixedN)

BOYS_TIER_ENTRY(BoysStatus,
                BoysAllNChecked,
                (int nmax,
                 const double* x,
                 double* out,
                 std::size_t count,
                 std::size_t* workspace,
                 std::size_t* badIndex),
                (nmax, x, out, count, workspace, badIndex),
                Precision::kFp64,
                Shape::kAllN)

BOYS_TIER_ENTRY(BoysStatus,
                BoysAllNAtOrdersChecked,
                (const int* n,
                 const double* x,
                 double* out,
                 std::size_t count,
                 std::size_t* badIndex),
                (n, x, out, count, badIndex),
                Precision::kFp64,
                Shape::kAllNAtOrders)

// --- single precision -------------------------------------------------------

BOYS_TIER_ENTRY(float,
                BoysSingleF32,
                (int n, float x),
                (n, x),
                Precision::kFp32,
                Shape::kSingle)

BOYS_TIER_ENTRY(void,
                BoysAllOrdersF32,
                (int nmax, float x, float* out),
                (nmax, x, out),
                Precision::kFp32,
                Shape::kAllOrders)

BOYS_TIER_ENTRY(void,
                BoysAllNF32,
                (int nmax, const float* x, float* out, std::size_t count),
                (nmax, x, out, count),
                Precision::kFp32,
                Shape::kAllN)

BOYS_TIER_ENTRY(void,
                BoysFixedNF32,
                (int n, const float* x, float* out, std::size_t count, std::size_t stride),
                (n, x, out, count, stride),
                Precision::kFp32,
                Shape::kFixedN)

BOYS_TIER_ENTRY(void,
                BoysAllNAtOrdersF32,
                (const int* n, const float* x, float* out, std::size_t count),
                (n, x, out, count),
                Precision::kFp32,
                Shape::kAllNAtOrders)

// --- the half lanes: present exactly where the fp16 seam is open ------------

#if BoysFp16

BOYS_TIER_ENTRY(F16,
                BoysSingleF16,
                (int n, F16 x),
                (n, x),
                Precision::kFp16,
                Shape::kSingle)

BOYS_TIER_ENTRY(void,
                BoysAllOrdersF16,
                (int nmax, F16 x, F16* out),
                (nmax, x, out),
                Precision::kFp16,
                Shape::kAllOrders)

BOYS_TIER_ENTRY(void,
                BoysFixedNF16,
                (int n, const F16* x, F16* out, std::size_t count, std::size_t stride),
                (n, x, out, count, stride),
                Precision::kFp16,
                Shape::kFixedN)

BOYS_TIER_ENTRY(void,
                BoysAllNF16,
                (int nmax, const F16* x, F16* out, std::size_t count),
                (nmax, x, out, count),
                Precision::kFp16,
                Shape::kAllN)

BOYS_TIER_ENTRY(void,
                BoysAllNAtOrdersF16,
                (const int* n, const F16* x, F16* out, std::size_t count),
                (n, x, out, count),
                Precision::kFp16,
                Shape::kAllNAtOrders)

BOYS_TIER_ENTRY(Bf16,
                BoysSingleBf16,
                (int n, Bf16 x),
                (n, x),
                Precision::kBf16,
                Shape::kSingle)

BOYS_TIER_ENTRY(void,
                BoysAllOrdersBf16,
                (int nmax, Bf16 x, Bf16* out),
                (nmax, x, out),
                Precision::kBf16,
                Shape::kAllOrders)

BOYS_TIER_ENTRY(void,
                BoysFixedNBf16,
                (int n, const Bf16* x, Bf16* out, std::size_t count, std::size_t stride),
                (n, x, out, count, stride),
                Precision::kBf16,
                Shape::kFixedN)

BOYS_TIER_ENTRY(void,
                BoysAllNBf16,
                (int nmax, const Bf16* x, Bf16* out, std::size_t count),
                (nmax, x, out, count),
                Precision::kBf16,
                Shape::kAllN)

BOYS_TIER_ENTRY(void,
                BoysAllNAtOrdersBf16,
                (const int* n, const Bf16* x, Bf16* out, std::size_t count),
                (n, x, out, count),
                Precision::kBf16,
                Shape::kAllNAtOrders)

#endif // BoysFp16

#undef BOYS_TIER_ENTRY
