# Phase 4 — research plan

## The actual scientific question

Phase 3 demonstrated, with falsifiability protocol, that the canonical
Alcubierre tanh shape function is **near-optimal within the 1-D Alcubierre
ansatz family** (0.37% reduction at most, under random-grid + smoothness +
adaptive-quad evaluation). That family fixes the metric form:

```
ds² = -dt² + (dx - v f(x,ρ) dt)² + dρ² + ρ² dφ²
```

and varies only `f`.  The remaining free degrees of freedom are *outside*
this ansatz — i.e. in the off-diagonal `g_tx`, in `g_xx`, in `g_ρρ`, and in
how those components relate to each other.  The Lentz 2021 and
Bobrick-Martire 2021 results live in exactly this larger space.

The Phase 4 hypothesis:

> **There exists a 4-parameter family of cylindrically-symmetric metrics
> with bubble-like causal structure for which the integrated NEC violation
> over the entire 3-D bubble interior is strictly less than 50% of the
> canonical Alcubierre value at matched bubble radius and ship velocity.**

If this hypothesis holds with rigorous falsifiability protocols, the
result is publishable in PRD or CQG.  If it holds with > 80% reduction
*and* avoids the well-known Pfenning-Ford / Olum quantum-inequality
bounds, it's PRL-level.  If it holds without violating ANEC anywhere
(strong null energy condition), it's potentially Nature-level — but that
last case is widely believed impossible and we should not predict it.

## Ansatz

Cylindrically symmetric, time-independent (in co-moving frame), 4 indep
metric components plus the Alcubierre-style off-diagonal:

```
g_tt(x, ρ) = -1 + h_tt(x, ρ)
g_tx(x, ρ) = h_tx(x, ρ)             ← the "shift" — Alcubierre keeps only this nonzero in its form
g_xx(x, ρ) =  1 + h_xx(x, ρ)
g_ρρ(x, ρ) =  1 + h_ρρ(x, ρ)
g_φφ(x, ρ) = ρ² (1 + h_φφ(x, ρ))
```

with each `h_•` a small MLP `R² → R` that goes to 0 at large `r = √(x² + ρ²)`.

The asymptotic-flatness condition is *baked in* by writing the metric as
`η + h(x, ρ)` and adding an asymptotic mask `exp(-r²/L²)` multiplied into
each `h_•` output, so the network can't disturb Minkowski outside the
bubble — only modify it inside.

The bubble interior condition (the ship actually moves) is enforced by a
soft penalty on `g_tx` at the bubble centre matching `-v` (so an observer
at rest in coord-x feels effective velocity `v`).

## Loss

```
L = λ_NEC  · ∫_V max(0, -T_kk(x, ρ)) dV
  + λ_ANEC · ∫_γ T_kk dλ (over null geodesics through the bubble)
  + λ_flat · ∫_∞ ‖g - η‖² dV
  + λ_smooth · ∫ ‖∇²h‖² dV
  + λ_ship · (g_tx(0, 0) + v)²
  + λ_signature · count(det(g) > 0)        # Lorentzian-signature penalty
```

`T_kk` is computed numerically from the network outputs via finite-difference
Christoffel (same approach as `src/curvature.rs`), then standard Riemann →
Ricci → Einstein → T.  All of this is in burn's compute graph so the
outer ∂L/∂params backprops cleanly.

Null vectors for the NEC contraction: sampled isotropically in the local
frame at each point, not just the (1,1,0,0) axis we used in Phase 3.

## Falsifiability protocols (mandatory before any claim)

Every claimed improvement must pass:

1. **Three quadrature schemes** for the NEC integral (trapezoid n=N,
   trapezoid n=10N, adaptive Romberg). All three must agree to within 0.5%
   of the reported ratio.
2. **Random collocation points** every training step (no fixed grid the
   network can over-fit to).
3. **TV(g) lower or equal to canonical** at matched bubble parameters.
4. **Constraint satisfaction**: all soft penalties below 1e-4 at end of
   training.
5. **Multi-seed**: at least 16 independent seeds, claim is the median
   across seeds (not the best one). Variance reported.
6. **Held-out sample points**: train on N collocation points, evaluate on a
   disjoint set of 10N points uniformly drawn from V. No improvement on
   held-out = overfit.

If we make a claim that fails *any* of these, the result is artifactual.
This protocol comes directly from the v3 falsifiability work — reviewer
was right that we needed it, and we should not relax it for Phase 4.

## Experiment matrix (the 2-day sweep)

For each of the four B200 GPUs, run a slice of:

| dimension | values |
|---|---|
| bubble velocity `v` | 0.1, 0.3, 0.5, 0.8, 1.0 |
| bubble radius `R` | 1.0, 2.0, 4.0, 8.0 |
| smoothness λ | 1e-4, 1e-3, 1e-2, 1e-1 |
| ansatz depth | 4 layers ×128, 6 ×256, 8 ×384 |
| seeds | 16 per cell |

= 5 × 4 × 4 × 3 × 16 = **3840 runs**. At ~5 min/run on one B200 that's
~320 GPU-hours, comfortably in the 2-day × 4-GPU = 192 GPU-hour budget if
we batch runs across GPUs.

Practical schedule (4 GPUs in parallel, ~50 hours each):
- GPU 0: `v ∈ {0.1, 0.3}` full sweep
- GPU 1: `v ∈ {0.5}` full sweep
- GPU 2: `v ∈ {0.8}` full sweep
- GPU 3: `v ∈ {1.0}` full sweep, plus rerun the best cells from GPU 0-2

## What "good" looks like

- *Floor result* (workshop / arXiv): null-result with the protocols all
  passing, demonstrating no improvement > 1% under our constraints. That's
  still publishable because the search was competent.
- *Solid result* (PRD / CQG): a metric family with 20-50% reduction in
  integrated NEC violation that survives all 6 protocol checks.
- *Strong result* (PRL): > 80% reduction or a qualitatively new bubble
  geometry (e.g. toroidal, double-shell).
- *Nature class*: a metric with bubble-like causal structure that
  satisfies *averaged* NEC over all complete null geodesics.  Almost
  certainly impossible but: we run the search and find out.

## Repo layout under `phase4/`

```
phase4/
  RESEARCH_PLAN.md             # this file
  ansatz.rs                    # the parameterised metric model
  loss.rs                      # NEC, ANEC, asymptotic, smoothness, ship
  train.rs                     # training loop, single-cell
  sweep.rs                     # multi-cell driver
  results/
    <cell-hash>/log.jsonl
    <cell-hash>/best_metric.bin
    <cell-hash>/protocol.json  # 6-protocol check results
```

## What this rules out and what it does not

This pipeline will *not* answer:
- Whether matter sources for the required negative energy exist in nature
- Whether the metric can be dynamically formed
- Whether quantum effects (Pfenning-Ford) wash out any classical optimum

It *will* answer:
- Whether classical GR admits warp-bubble metrics with substantially less
  ANEC violation than Alcubierre, within a well-defined neural ansatz
- The variational optimum (or absence thereof) in this ansatz family
- Whether canonical Lentz 2021 / Bobrick-Martire 2021 metrics are
  locally optimal in our wider space
