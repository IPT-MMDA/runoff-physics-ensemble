# Inference-noise drift analysis

This analysis pairs every noisy evaluation with the clean evaluation of the same basin, period, training condition, seed, and inference-noise draw. The highest analyzed noise is `0.25`.

Positive NSE/KGE drift means degradation; positive RMSE and physics-error percentages mean increased error.

## Results at maximum noise

| Model and training | Basins | Clean NSE | NSE degradation | Basins degraded | RMSE increase | Physics-error increase |
|---|---:|---:|---:|---:|---:|---:|
| learned · mlp · joint baseline | 15 | 0.572 | +0.022 [+0.015, +0.031] | 100% | +3.4% | +7.7% |
| morse · mlp · distill to simple | 15 | 0.578 | +0.003 [-0.000, +0.007] | 60% | +0.6% | +4.3% |

## Key observations

- **morse · mlp · distill to simple has the smallest mean NSE drift** at noise `0.25` (+0.003).
- **learned · mlp · joint baseline degrades more consistently:** its mean NSE loss is +0.022, about 6.9× larger, and 100% of basins degrade.
- **learned · mlp · joint baseline:** mean KGE degrades by 0.000.
- **morse · mlp · distill to simple:** mean KGE improves by 0.005.

## Most noise-sensitive basins

- **learned · mlp · joint baseline:** 3014 (+0.072 NSE), 4022 (+0.039 NSE), 4009 (+0.035 NSE)
- **morse · mlp · distill to simple:** 4022 (+0.018 NSE), 4008 (+0.013 NSE), 5016 (+0.010 NSE)

## Interpretation limits

- The result sets use different model/training combinations, so their contrast is descriptive rather than a controlled causal estimate of distillation alone.
- There is one training seed and one inference-noise draw per basin. Bootstrap intervals quantify variation across basins, not seed or noise-realization uncertainty.
- Input noise is expressed in standardized feature units.

## Inputs

- `/home/ubuntu/work/numba/ensemble_review_git/legendary-broccoli/artifacts/hydrology/distill_to_simple/15_basins_learned_only/results.json`
- `/home/ubuntu/work/numba/ensemble_review_git/legendary-broccoli/artifacts/hydrology/distill_to_simple/15_basins_mlp_morse_only/results.json`
