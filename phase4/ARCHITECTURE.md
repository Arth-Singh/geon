# Phase 4 — full-metric ansatz, architecture sketch

## Inputs and outputs

| name | shape | meaning |
|---|---|---|
| `x` | `[N]` | axial coordinate |
| `ρ` | `[N]` | cylindrical radius |
| `g(x, ρ)` | `[N, 4, 4]` | symmetric metric tensor |
| `T(x, ρ)` | `[N, 4, 4]` | stress-energy tensor |
| `NEC(x, ρ; k)` | `[N]` | `T_μν k^μ k^ν` |
| `loss` | `[]` | scalar: integrated max(0, -NEC) + penalties |

## Metric ansatz

In coords (t, x, ρ, φ), assuming cylindrical symmetry (no t or φ dependence):

```
g_tt(x, ρ) = -1 + h_tt(x, ρ) · m(x, ρ)
g_tx(x, ρ) = h_tx(x, ρ) · m(x, ρ)
g_xx(x, ρ) =  1 + h_xx(x, ρ) · m(x, ρ)
g_ρρ(x, ρ) =  1 + h_ρρ(x, ρ) · m(x, ρ)
g_φφ(x, ρ) = ρ² (1 + h_φφ(x, ρ) · m(x, ρ))
```

with `m(x, ρ) = exp(-((x² + ρ²) / L²)^p)` a soft mask that decays at infinity
(L set to ~3R, p = 2 gives a smooth top-hat-like cutoff). This *bakes in*
asymptotic flatness: outside L, the metric is Minkowski by construction.

Each `h_•` is a small MLP `[x, ρ] → scalar`. Initial weights small so the
network starts close to Minkowski.

## Christoffel via finite difference

Same as `src/curvature.rs` — for each component `g_μν`:

```
∂_α g_μν ≈ (g_μν(x + h ê_α) - g_μν(x - h ê_α)) / 2h     α ∈ {x, ρ}
```

(t and φ derivatives are zero by symmetry.) Three forward passes through
the network per training step: one at the collocation points and two at
the shifted points.

Then `Γ^μ_αβ = ½ g^μσ (∂_α g_σβ + ∂_β g_σα - ∂_σ g_αβ)`.

## Riemann via finite difference on Γ

```
R^ρ_σμν = ∂_μ Γ^ρ_νσ − ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_μσ
```

Same FD pattern, two more forward passes (`x + 2h`, `x - 2h` etc.) to get
∂Γ. Total: 5 forward passes per training step (at points x, x±h, x±2h).

This is exactly the curvature computation already implemented in pure-Rust
fp64 in `src/curvature.rs`. The Phase 4 version is the same algorithm
re-expressed against burn tensors so the outer autograd works end-to-end.

## Loss decomposition

| term | weight | meaning |
|---|---|---|
| `∫_V max(0, -T_kk) dV` | 1 | NEC violation, integrated over bubble volume |
| `∫_∞ ‖g - η‖² dV_outer` | λ_flat ≈ 10² | asymptotic flatness penalty (outside L) |
| `(g_tx(0, 0) + v)²` | λ_ship ≈ 10² | ship moves with velocity v in coord-x |
| `∫ ‖∇² h‖² dV` | λ_smooth ≈ 10⁻³ | smoothness regulariser |
| `(min eigval of g̃) violation` | λ_sig ≈ 10⁴ | Lorentzian signature preservation |

`k` ranges over a discrete set of null directions sampled at each
collocation point (e.g. 6 directions: ±x, ±y, ±z in the local Minkowski
frame, normalised to be null under g at that point).

## Falsifiability protocol — every claimed improvement runs this

Same as Phase 3c, scaled to 3D:

1. **Three quadrature schemes** — fixed grid n=N, fixed grid n=4N, and
   Monte-Carlo with 8N random samples. All three must agree to within 0.5%.
2. **Random collocation** every training step.
3. **TV check**: total variation of each `h_•` must be ≤ the canonical
   Alcubierre values at matched bubble parameters.
4. **Constraint convergence**: all soft penalties below 1e-4 at end.
5. **Multi-seed**: median over ≥ 16 seeds, variance reported.
6. **Held-out collocation**: train on N points, evaluate on disjoint 10N.

## Computational cost per step

- 5 forward passes × 5 metric components × N collocation points × MLP cost
- For N = 5000 and 4-layer 128-hidden MLP: ~25M flops per pass
- 5 × 5 × 25M = 625M flops per step (forward)
- Backward ~ 3× forward = ~2 Gflops/step
- On a single B200 (~100 Tflops FP32 sustained): ~20 μs/step in theory
- Realistic with launch overhead: ~10-30 ms/step
- 6000 steps = 1-3 minutes per training run
- 50 seeds × 80 (v, R, λ, depth) cells = 4000 runs = 4000 minutes = 67
  hours on one GPU; on 4 GPUs in parallel = ~17 hours

This fits comfortably in the 2-day budget.

## File layout under `phase4/`

```
phase4/
  RESEARCH_PLAN.md
  ARCHITECTURE.md           (this file)
  warp_search_v3.png        (Phase 3c reference plot)
src/bin/
  phase4_shape.rs           (Rust port of Phase 3c — Alcubierre-form shape function)
  phase4_full_metric.rs     (the actual Phase 4 — full-metric ansatz)
  phase4_sweep.rs           (multi-cell driver for the 2-day run)
src/
  metric_nn.rs              (FullMetric type with burn-tensor Christoffel/Riemann)
  losses.rs                 (NEC, ANEC, flatness, signature, etc.)
```
