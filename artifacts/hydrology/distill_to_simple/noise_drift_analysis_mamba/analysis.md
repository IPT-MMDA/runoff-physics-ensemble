# Inference-noise drift analysis

This analysis pairs every noisy evaluation with the clean evaluation of the same basin, period, training condition, seed, and inference-noise draw. The highest analyzed noise is `0.25`.

Positive NSE/KGE drift means degradation; positive RMSE and physics-error percentages mean increased error.

## Results at maximum noise

| Model and training | Basins | Clean NSE | NSE degradation | Basins degraded | RMSE increase | Physics-error increase |
|---|---:|---:|---:|---:|---:|---:|
| learned · pinnmamba · joint baseline | 15 | 0.308 | +0.034 [+0.022, +0.046] | 87% | +4.0% | +6.7% |
| morse · pinnmamba · distill to simple | 15 | 0.529 | +0.012 [-0.005, +0.030] | 73% | +1.6% | +7.7% |

## Key observations

- **morse · pinnmamba · distill to simple has the smallest mean NSE drift** at noise `0.25` (+0.012).
- **learned · pinnmamba · joint baseline degrades more consistently:** its mean NSE loss is +0.034, about 2.7× larger, and 87% of basins degrade.
- **learned · pinnmamba · joint baseline:** mean KGE degrades by 0.011.
- **morse · pinnmamba · distill to simple:** mean KGE degrades by 0.002.

## Most noise-sensitive basins

- **learned · pinnmamba · joint baseline:** 4023 (+0.064 NSE), 4022 (+0.064 NSE), 3014 (+0.061 NSE)
- **morse · pinnmamba · distill to simple:** 2488 (+0.080 NSE), 4009 (+0.076 NSE), 3014 (+0.031 NSE)

## Interpretation limits

- The result sets use different model/training combinations, so their contrast is descriptive rather than a controlled causal estimate of distillation alone.
- There is one training seed and one inference-noise draw per basin. Bootstrap intervals quantify variation across basins, not seed or noise-realization uncertainty.
- Input noise is expressed in standardized feature units.

## Inputs

- `/home/ubuntu/work/numba/ensemble_review_git/legendary-broccoli/artifacts/hydrology/distill_to_simple/15_basins_mamba_learned_only/results.json`
- `/home/ubuntu/work/numba/ensemble_review_git/legendary-broccoli/artifacts/hydrology/distill_to_simple/15_morse_mamba_morse_only/results.json`
