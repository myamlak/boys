# Calibration corpus for tools/check_doc_arithmetic.py

A test fixture for tools/check_doc_arithmetic.py, not a document of the
library. It exists so that each check `--selftest` pins still has a site to
fire on.

Every section below writes a sentence of the shape the check is built to catch.
The sentences are left wrong on purpose, and are deliberately not corrected
here: a calibration corpus that is right is a corpus that calibrates nothing.
The prose the check runs on in this tree carries none of these shapes.

Two of the sections near the end are right on purpose. They pin a *parse*
rather than a defect - a spelled number that is a quantity, and a margin whose
ratio to the bound is printed - and they are here so that a later tightening of
the rules cannot quietly stop reading them.

## A fraction, two values, one system size

For the 586-basis-function case the screened set is 4.59% of the shell
quartets, measured on the same grid the rest of this report uses.

For the 586-basis-function case the screened set is 8.9% of the shell
quartets, and the screening therefore costs less than the integrals it
replaces.

## A penalty stated as a factor

The batch entry needs 12.3 instructions per argument against the scalar
entry's 7.0, which is a penalty of a factor of three in instructions per
argument.

## A ratio with no printed operands

The stored coefficients are rounded to double precision, and they hold only the
16 or so significant digits that a double can hold. The mathematical fit behind
them is about sixty times more accurate than the numbers used to store it, so no
accuracy below roughly 1e-16 is reachable, however the evaluation is arranged.

## A claim whose unit is a step

The measured worst cell is 4.243 ULP against a representation allowance of half
a step, so the lane loses about 7.7 times as much as the half lane's bound allows
for representing its result.

## Two readings of one sentence

Just below x = 16, at order 32, the relative error reaches 5.6e4, which is five
to eight orders of magnitude past the 5.5e-14 bound.

## A margin whose multiplier is not printed

The lane is bounded by m·1e-7 plus half of the last representable digit of the
result. The worst case sits at 0.9999 of the bound — a margin of thousandths of
a per cent, not a per cent. At that distance, half of one representable digit of
the result is 99% of the whole bound.

## A spelled number that is prose and not a quantity

The three rows came out within 0.9% of one another, against the 5.6 to 7.4
points one of them moves by from one run to the next, and two of these axes
have been measured while two have not.

## A spelled number that is a quantity

The measured worst cell is 4.243 of those digits against the half-ULP the other
lane budgets, which is 8.5 times its representation allowance.

## A ratio stated between two named lanes

It retires 3.06 times fewer instructions than the loop it replaces, and 3.13
times fewer retired slots than the float lane's default entry, which serves all
33 orders from one seed fit.

## Figures separated by thousands

The packed variant retires 6,061,196,228 instructions against the loop's
2,568,303,487, which is 2.36 times fewer.

## A margin whose ratio to the bound is printed

For the 16-bit format the worst case is 0.9993 of the bound, a margin of 0.07
per cent.

## A share of a bound the sentence names

The two lanes parted by at most 4.04e-09 at the worst, which is 2.7% of the
lane's bar.

## A run of digits inside an identifier

The table is regenerated from the digest 9f18e45f, which is 2 times the earlier
key. The lane is documented for 32bit and 64bit builds, which is 2 times the
count the earlier one covered.
