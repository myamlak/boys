# What this directory carries

`results.json` and `README.md` are copied unchanged from the half-lane sweep's deposit, taken on
27 October 2026 against the library at tag `v5.1.1`. `README.md` describes the deposit as a whole;
`results.json` is the record the accuracy gate reads.

The harness (`half_sweep.cpp`), the 45-digit oracle (`half_sweep_oracle.py`) and the command log
(`build.txt`) stay with the deposit: they are the paper's artifacts rather than sources of this
library. The gate runs neither program and does not re-derive the oracle. It checks the record's
counts, margins and worst cells, and names them as the deposit's figures, never as its own.
