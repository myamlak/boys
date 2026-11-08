// One digest of every default entry's bits, for the build to print from several sets of flags: the
// readings agree because the arithmetic is one source and the ISA tiers are bit-equal to the
// scalar twins, and a digest that differs is a finding. The sweep and digest live in
// entry_digest.hpp, shared with the dispatch build's forced-tier test.

#include "entry_digest.hpp"

#include <cstdint>
#include <cstdio>

int main() {
    std::printf("%016llx\n",
                static_cast<unsigned long long>(entry_digest::Bits()));
    return 0;
}
