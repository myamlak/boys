#pragma once

// The launchers of tests/boys_cuda_device_demo.cu, declared once for both sides
// of the check: the consumer translation unit that defines them includes this
// header and the gate that drives them includes it too, so a signature cannot
// drift between the two.
//
// Each launcher returns the CUDA error code of its launch (0 when the launch
// was accepted), the way this lane's own launchers report through BoysStatus,
// and takes the handle the gate filled host-side. A family launcher writes
// kMaxBoysOrder + 1 values per element, in the element's own block, and a
// single launcher writes one value per element. \c status receives the entry's
// own status per element, so the caller can check what the entry returned and
// not only what it wrote.
//
// The rung is the last argument, defaulted to m = 1: the entry refuses a rung
// whose degree tables are not the resident ones, and a default that named a
// relaxed rung would turn that refusal into a silent choice of whichever rung
// happened to be resident.

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
                                      int* status,
                                      double multiplier = boys::kBoysFullAccuracyMultiplier);

extern "C" int BoysDeviceDemoAllN16(const boys::BoysDeviceTables* tables,
                                    const double* rho,
                                    const double* d2,
                                    void* out,
                                    std::size_t count,
                                    int* status,
                                    double multiplier = boys::kBoysFullAccuracyMultiplier);

extern "C" int BoysDeviceDemoEach16(const boys::BoysDeviceTables* tables,
                                    const int* n,
                                    const double* rho,
                                    const double* d2,
                                    void* out,
                                    std::size_t count,
                                    int* status,
                                    double multiplier = boys::kBoysFullAccuracyMultiplier);

extern "C" int BoysDeviceDemoSingle16(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      void* out,
                                      std::size_t count,
                                      int* status,
                                      double multiplier = boys::kBoysFullAccuracyMultiplier);
#endif // BoysFp16

extern "C" int BoysDeviceDemoLadder32(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      float* out,
                                      std::size_t count,
                                      int capacity,
                                      int* status,
                                      double multiplier = boys::kBoysFullAccuracyMultiplier);

extern "C" int BoysDeviceDemoAllN32(const boys::BoysDeviceTables* tables,
                                    const double* rho,
                                    const double* d2,
                                    float* out,
                                    std::size_t count,
                                    int* status,
                                    double multiplier = boys::kBoysFullAccuracyMultiplier);

extern "C" int BoysDeviceDemoEach32(const boys::BoysDeviceTables* tables,
                                    const int* n,
                                    const double* rho,
                                    const double* d2,
                                    float* out,
                                    std::size_t count,
                                    int* status,
                                    double multiplier = boys::kBoysFullAccuracyMultiplier);

extern "C" int BoysDeviceDemoSingle32(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      float* out,
                                      std::size_t count,
                                      int* status,
                                      double multiplier = boys::kBoysFullAccuracyMultiplier);

// The float single entry with the lane's other region-B exponential
// (boys::RegionBExp::kFast), so the gate measures both options of the entry and
// the difference between them.
extern "C" int BoysDeviceDemoSingle32Fast(const boys::BoysDeviceTables* tables,
                                          const int* n,
                                          const double* rho,
                                          const double* d2,
                                          float* out,
                                          std::size_t count,
                                          int* status,
                                          double multiplier = boys::kBoysFullAccuracyMultiplier);

extern "C" int BoysDeviceDemoLadder64(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      double* out,
                                      std::size_t count,
                                      int capacity,
                                      int* status,
                                      double multiplier = boys::kBoysFullAccuracyMultiplier);

extern "C" int BoysDeviceDemoAllN64(const boys::BoysDeviceTables* tables,
                                    const double* rho,
                                    const double* d2,
                                    double* out,
                                    std::size_t count,
                                    int* status,
                                    double multiplier = boys::kBoysFullAccuracyMultiplier);

extern "C" int BoysDeviceDemoEach64(const boys::BoysDeviceTables* tables,
                                    const int* n,
                                    const double* rho,
                                    const double* d2,
                                    double* out,
                                    std::size_t count,
                                    int* status,
                                    double multiplier = boys::kBoysFullAccuracyMultiplier);

extern "C" int BoysDeviceDemoSingle64(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      double* out,
                                      std::size_t count,
                                      int* status,
                                      double multiplier = boys::kBoysFullAccuracyMultiplier);

// The status values the entries can return, so the gate compares against the
// entries' own enum rather than against a transcription of it. Device code
// cannot throw, so a refusal is one of these and nothing else.
extern "C" int BoysDeviceDemoStatusSuccess();
extern "C" int BoysDeviceDemoStatusOrderOutOfRange();
extern "C" int BoysDeviceDemoStatusCapacityTooSmall();
extern "C" int BoysDeviceDemoStatusMultiplierNotResident();
