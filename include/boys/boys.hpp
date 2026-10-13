#pragma once

/// \file
/// The Boys function calls: the declarations, and the arithmetic that defines them.
///
/// Every entry a call site names no policy for is defined inline in `boys/kernels.hpp` at the row
/// this build's default-policy table carries for its class, so a caller compiles the body it calls
/// and pays no call boundary. A translation unit that declares a wrapper, names a type or holds an
/// entry as a value and calls none of them needs `boys/boys_decl.hpp` alone, and pays for none of
/// this arithmetic.

#include "boys/boys_decl.hpp"

// The default entries declared above are defined inline in boys/kernels.hpp: a caller that
// names no policy compiles the body it calls, and this include is where that body comes from.
#include "boys/kernels.hpp"
