# Third-party notices

Repo license: BSD-3-Clause (`LICENSE`). This file lists licenses and
attribution obligations for every third-party component the repository
ships or consumes, so the notices must travel with any redistribution.

| Component | Version | Purpose | License | Full text |
|---|---|---|---|---|
| GoogleTest | 1.17.0 (pinned submodule) | test harness | BSD-3-Clause | `third_party/googletest/LICENSE` |
| Google Benchmark | 1.9.5 (pinned submodule, optional build flag) | scaling benchmarks | Apache-2.0 | `third_party/benchmark/LICENSE` |
| mpmath (Python) | pinned in `requirements-boys.txt` | build-time-only regeneration of the coefficient tables and reference grid (`tools/gen_boys_coefficients.py`) | BSD-3-Clause | pip package metadata (not redistributed) |
| CUDA toolkit | optional build flag | CUDA lane (kernel sources and host wrapper) | NVIDIA toolkit license | NVIDIA package (not redistributed; never CI-built) |

The pinned submodules carry their own LICENSE files inside the submodule;
their full texts are not duplicated here (the submodule rule). GoogleTest
and Google Benchmark are the only vendored code. mpmath and the CUDA
toolkit are build-time-only tools and are not redistributed with any
release; their licenses apply only if the tooling itself is shipped.

## Data provenance

The committed reference grid (`tests/data/boys_reference.csv`) and the
generated Chebyshev coefficient tables (the `boys_coefficients.hpp`
header) are this project's own output — numerical values generated
in-repo by `tools/gen_boys_coefficients.py` from the published evaluation
formulas ([Boys1950], [Shavitt1963]) — and are licensed under the
repository's BSD-3-Clause license. No third-party data ships in the
repository. The kernel source contains no third-party code.
