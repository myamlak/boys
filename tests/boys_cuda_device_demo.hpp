#pragma once

// The launchers of tests/boys_cuda_device_demo.cu, declared once so the
// translation unit that defines them and the gate that drives them cannot drift
// apart.
//
// Each returns the CUDA error code of its launch, 0 when the launch was
// accepted. A family launcher writes kMaxBoysOrder + 1 values per element, in
// the element's own block, and a single launcher writes one value per element;
// \c status receives the entry's own status per element.

#include "boys/boys_device_tables.hpp"

#include <cstddef>

#if BoysFp16
extern "C" int BoysDeviceDemoLadder16(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      void* out,
                                      std::size_t count,
                                      int capacity,
                                      int* status);

extern "C" int BoysDeviceDemoAllN16(const boys::BoysDeviceTables* tables,
                                    const double* rho,
                                    const double* d2,
                                    void* out,
                                    std::size_t count,
                                    int* status);

extern "C" int BoysDeviceDemoEach16(const boys::BoysDeviceTables* tables,
                                    const int* n,
                                    const double* rho,
                                    const double* d2,
                                    void* out,
                                    std::size_t count,
                                    int* status);

extern "C" int BoysDeviceDemoSingle16(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      void* out,
                                      std::size_t count,
                                      int* status);
#endif // BoysFp16

extern "C" int BoysDeviceDemoLadder32(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      float* out,
                                      std::size_t count,
                                      int capacity,
                                      int* status);

extern "C" int BoysDeviceDemoAllN32(const boys::BoysDeviceTables* tables,
                                    const double* rho,
                                    const double* d2,
                                    float* out,
                                    std::size_t count,
                                    int* status);

extern "C" int BoysDeviceDemoEach32(const boys::BoysDeviceTables* tables,
                                    const int* n,
                                    const double* rho,
                                    const double* d2,
                                    float* out,
                                    std::size_t count,
                                    int* status);

extern "C" int BoysDeviceDemoSingle32(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      float* out,
                                      std::size_t count,
                                      int* status);

// The float single entry with the lane's other region-B exponential, so the
// gate measures both options of the entry and the difference between them.
extern "C" int BoysDeviceDemoSingle32Fast(const boys::BoysDeviceTables* tables,
                                          const int* n,
                                          const double* rho,
                                          const double* d2,
                                          float* out,
                                          std::size_t count,
                                          int* status);

extern "C" int BoysDeviceDemoLadder64(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      double* out,
                                      std::size_t count,
                                      int capacity,
                                      int* status);

extern "C" int BoysDeviceDemoAllN64(const boys::BoysDeviceTables* tables,
                                    const double* rho,
                                    const double* d2,
                                    double* out,
                                    std::size_t count,
                                    int* status);

extern "C" int BoysDeviceDemoEach64(const boys::BoysDeviceTables* tables,
                                    const int* n,
                                    const double* rho,
                                    const double* d2,
                                    double* out,
                                    std::size_t count,
                                    int* status);

extern "C" int BoysDeviceDemoSingle64(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      double* out,
                                      std::size_t count,
                                      int* status);

// The partition and route axes of the ladder shape, one launcher per entry.
//
// One spelling of the parameter list, instantiated per entry: the launcher's
// shape and the definition's are then the same declaration rather than two that
// have to agree, which is what this header is for. The entries are the ladder's
// own — the same parameters and the same capacity — over another partition,
// another route or the other stored form.
#define BOYS_DEVICE_DEMO_LADDER_DECL(SUFFIX, VALUE)                                                \
    extern "C" int BoysDeviceDemoLadder##SUFFIX(const boys::BoysDeviceTables* tables,              \
                                                const int* n,                                      \
                                                const double* rho,                                 \
                                                const double* d2,                                  \
                                                VALUE* out,                                        \
                                                std::size_t count,                                 \
                                                int capacity,                                      \
                                                int* status);

BOYS_DEVICE_DEMO_LADDER_DECL(64Narrow, double)
BOYS_DEVICE_DEMO_LADDER_DECL(64NarrowMono, double)
BOYS_DEVICE_DEMO_LADDER_DECL(64Rat, double)
BOYS_DEVICE_DEMO_LADDER_DECL(64RatHorner, double)
BOYS_DEVICE_DEMO_LADDER_DECL(64NarrowRat, double)
BOYS_DEVICE_DEMO_LADDER_DECL(64NarrowRatHorner, double)
BOYS_DEVICE_DEMO_LADDER_DECL(64Uniform, double)
BOYS_DEVICE_DEMO_LADDER_DECL(64UniformHorner, double)
BOYS_DEVICE_DEMO_LADDER_DECL(64UniformRat, double)
BOYS_DEVICE_DEMO_LADDER_DECL(64UniformRatHorner, double)

BOYS_DEVICE_DEMO_LADDER_DECL(32Narrow, float)
BOYS_DEVICE_DEMO_LADDER_DECL(32NarrowMono, float)
BOYS_DEVICE_DEMO_LADDER_DECL(32Rat, float)
BOYS_DEVICE_DEMO_LADDER_DECL(32RatHorner, float)
BOYS_DEVICE_DEMO_LADDER_DECL(32NarrowRat, float)
BOYS_DEVICE_DEMO_LADDER_DECL(32NarrowRatHorner, float)
BOYS_DEVICE_DEMO_LADDER_DECL(32Uniform, float)
BOYS_DEVICE_DEMO_LADDER_DECL(32UniformHorner, float)
BOYS_DEVICE_DEMO_LADDER_DECL(32UniformRat, float)
BOYS_DEVICE_DEMO_LADDER_DECL(32UniformRatHorner, float)

#undef BOYS_DEVICE_DEMO_LADDER_DECL

// The status values the entries can return, so the gate compares against the
// entries' own enum rather than a transcription of it. Device code cannot
// throw, so a refusal is one of these and nothing else.
extern "C" int BoysDeviceDemoStatusSuccess();
extern "C" int BoysDeviceDemoStatusOrderOutOfRange();
extern "C" int BoysDeviceDemoStatusCapacityTooSmall();
