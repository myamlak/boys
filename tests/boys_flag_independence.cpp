// One digest of every default entry's bits, for the build to print from several sets of flags: the
// entries are the caller's to compile, and the readings agree because the arithmetic is one source
// and the ISA tiers are documented bit-equal to the scalar twins. A digest that differs is a
// finding. The sweep and the digest live in entry_digest.hpp, shared with the dispatch build's
// forced-tier test, so the two hold the tiers to one value rather than to two hand-copied ones.

#include "entry_digest.hpp"

#include <cstdint>
#include <cstdio>

int main() {
    std::printf("%016llx\n",
                static_cast<unsigned long long>(entry_digest::Bits()));
    return 0;
}
