# Changelog

Notable changes to the boys library. Versioning rules live in
`CONTRIBUTING.md` (the "Versioning" section); the public numerical
contract (the `m*B_region` bound) is documented in the README.

## [Unreleased — the v1.1.0 cut]

### Numerical behavior change (the per-range seed design)

- **Affected inputs:** `x` in `[1.0855252345349333, 11.899848152108484)`
  (the extended band below the region-B floor), the double lane, `m = 1`,
  every order `n = 0..32`.
- **Old evaluation:** the region-A per-order Chebyshev fits.
- **New evaluation:** the extended-band seed (a second `F_0` fit on the
  band, degree 24, the split-Clenshaw form) plus the upward recursion,
  dispatched per order through the certified per-order threshold table
  `kTierThresholds` (the pure per-`(n, x)` rule: an order `n` takes the
  extended seed exactly when `x >= kTierThresholds[n]`, so every entry
  point returns the same value for the same order and argument). The
  `m > 1` branch and the float lanes keep their previous evaluation in
  the band.
- **Contract impact:** no documented domain or budget changes; validated
  against the committed reference grid; max observed error over the
  band's grid rows `2.1e-15`
  (measured on the corrected dispatch over the regenerated grid, which
  pins the corrected boundaries and the vacated slices' edges), not
  exceeding the `3e-14` budget the band asserts. (Honesty note: the
  band's certificate is the range-uniform a-priori seed bound
  `delta_0' = 1.0527e-15` --- the truncation tail `4.112e-17` plus the
  split-Clenshaw forward rounding bound `1.0116e-15`, no x-sampling ---
  not the measured width, which enters only as a consistency check; see
  the Boundary-table entry.)
- **Bitwise impact:** not bitwise-identical to v1.0.0 inside the band
  (up to the asserted budget; well below it in practice);
  hash-pipeline consumers must regenerate baselines.
- **Boundary table:** the certified per-`k_max` recursion boundaries
  changed from the honest-extension table
  `11.8372 / 11.8372 / 11.8372 / 11.8464` (the shipped seed alone, the
  `k_max = 4/8/16/32` rows) to the per-range design's certified rows
  `1.0855 / 2.0136 / 4.8960 / 10.7818` (1-ulp-exp form), wired into the
  kernel as the dispatch thresholds. (Honesty note: the initial v1.1
  increment carried the empirical-bound crossings
  `1.8575 / 4.7031 / 10.6551` --- the envelope crossings under the
  per-argument measured interval width of the extended seed; an
  independent verification pass derived the band's range-uniform
  a-priori seed bound
  `delta_0' = 1.0527e-15` --- the truncation tail `4.112e-17` plus the
  split-Clenshaw forward rounding bound `1.0116e-15`, no x-sampling ---
  and the crossings under it, `2.0136 / 4.8960 / 10.7818`, replaced the
  empirical rows before the v1.1.0 tag cut.)
- **Unaffected:** the `m > 1` branch, the float lanes, and every input
  outside the band.
- **Old behavior:** pin tag `v1.0.0`.

The public history is intentionally compressed to two commits: the Initial
commit (tagged v1.0.0) and this release's cumulative commit; the items
above are the content of that cumulative commit.
