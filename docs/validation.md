# Validation Checks

This project uses explicit phase-boundary validation to catch bad model states
before they become plausible-looking MTM numbers.

## Simulation

`simulate()` requires an explicit `xi_0` anchor by default. For valuation runs,
pass:

```python
xi_0 = np.full(n_paths, np.log(SCHWARTZ_SMITH["forward_anchor_gbp_mwh"]))
```

Use `allow_unanchored=True` only for process-only tests where a GBP 1/MWh
initial price level is intentional.

`validate_path_bundle()` checks:

- expected path array shapes
- finite simulated factors
- non-negative ancillary prices
- plausible clipped spot-price summary metrics
- optional initial spot anchor against `forward_anchor_gbp_mwh`

## LSMC

`run_lsmc()` runs validation by default. Set `lsmc_cfg["run_validation"] = False`
only for diagnostics or intentionally malformed experiments.

The backward pass stores regression health metrics on `policy.diagnostics`:

- `regression_count`
- `beta_abs_max`
- `nonfinite_beta_count`
- `sample_condition_max`
- `sample_rank_deficient_count`
- `fallback_lstsq_count`
- `fallback_zero_count`
- `continuation_clip_fraction_max`

Hard failures:

- invalid asset config
- malformed or non-finite path bundle
- non-finite policy coefficients
- explosive beta coefficients above `1e8`
- forward valuation paths outside physical SoC bounds

Warnings:

- sampled rank-deficient regressions
- very ill-conditioned sampled regressions
- zero PV variance
- SoH increases without an explicit augmentation model

## Notebook Use

After a phase object is built, run the matching validator and print its summary:

```python
check = validate_path_bundle(
    bundle,
    forward_anchor_gbp_mwh=SCHWARTZ_SMITH["forward_anchor_gbp_mwh"],
)
check.raise_if_failed()
print(check.summary())
```

For published runs, include `policy.diagnostics` in the JSON summary alongside
the valuation numbers.
