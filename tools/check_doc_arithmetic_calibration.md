# Calibration corpus for tools/check_doc_arithmetic.py

Not a document of this library and not part of the published set. It exists so
that each check `--selftest` pins still has a site to fire on, and so the
calibration survives the repair of the defects it was built from.

Every section below reproduces the shape of one defect that was found in the
published documents and then fixed. The sentences are left wrong on purpose.
They are deliberately not corrected here, because a calibration corpus that is
right is a corpus that calibrates nothing.

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
