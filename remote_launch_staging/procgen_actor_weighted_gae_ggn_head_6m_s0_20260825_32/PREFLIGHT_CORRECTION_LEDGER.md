# Task32 preflight correction ledger

## Attempt 1: jobs 19280366-19280369

All four environment preflights constructed the exact 938,979-parameter production network and failed at the same PopArt affine-input assertion before any scientific launch. The assertion used an absolute tolerance of `2e-16`, below the roundoff accumulated by the deliberately reparameterized FP64 affine expression. This was a validation-harness tolerance defect, not an algorithm, formula, environment, numerical-solver, or scientific failure.

The attempt-1 launcher also captured the status as `PRECHECK_FAIL` but incorrectly recorded `rc=0` and exited zero because it read `$?` after the completed `if` compound command rather than immediately after Python. Slurm therefore reported `COMPLETED/0:0`; the authoritative root status and traceback remain `PRECHECK_FAIL`.

The immutable attempt-1 evidence remains under remote `preflight/<environment>` roots. No science root or scientific marker was created.

## Versioned pure-code correction

- Preserve the mathematical PopArt affine regression and compare the corresponding FP64 inputs at `16 * eps`; report the exact input, direction, and prediction discrepancies.
- Preserve exact solve and acceptance semantics; compare direction and prediction changes at explicit FP64 tolerances.
- Capture the Python return code immediately and propagate it to both the evidence root and Slurm.
- Route the corrected bounded run to fresh `preflight_attempt2/<environment>` roots so attempt 1 cannot be overwritten.

Trainer, scientific config, scientific launcher, stage monitor, method definition, RNG/data order, and all scientific roots remain byte-identical.
