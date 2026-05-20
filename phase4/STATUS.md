# Phase 4 — current status

Date: 2026-05-20 evening session.

## What's working end-to-end on the B200

- `geon` crate compiles with `cuda` feature
- `burn 0.21` + CUDA backend autograd verified (`burn_smoke`):
  `max|ad - fd| ≈ 1.5e-3` on a tiny MLP, agreeing with central finite
  differences to within fp32 noise.
- `burn_train_sanity`: a small MLP fitting `f(x) = 1` drives MSE from
  `0.27 → 4e-8` in 100 Adam steps. Training plumbing is solid.
- `phase4_shape`: Rust port of Phase 3c, full pipeline (random colloc
  grid, FD-based f'/f'', region penalties, pin, smoothness, Adam) runs
  end-to-end on B200 at **~5 ms/step**. Constraints converge to
  `interior_pen, exterior_pen ~ 1e-6`. The exotic-mass ratio varies with
  hyperparameters between 0.82 (lax) and 1.34 (over-regularised) — not
  yet matched to the Python v3 result of 0.997. Pipeline is fine, tuning
  is open.

## The burn 0.21 footgun

`#[derive(Module)]` requires the struct's generic to be **literally named
`B`**. Anything else (`BB`, `Backend`, etc.) silently emits no-op impls
— `num_params() = 0`, empty `GradientsParams`, `optim.step` is a no-op.
There's no compile error and no warning. Cost: 4 hours of training that
went nowhere before the smoking gun `n_grad_params=0` showed up in my
debug print. Documented in `[[feedback-burn-module-derive-generic-name]]`.

Convention going forward in the repo: struct generic is `B: Backend`,
backend type alias is `Bk`, no explicit `Clone` in the derive (the macro
auto-impls it).

## Next steps (resume here next session)

1. **hparam tune** `phase4_shape` (lr, h_fd, lambda_smooth) to converge
   to a ratio ≈ 0.997 ± 0.01 with TV(f') ≤ canonical's. That's the
   apples-to-apples reproduction of Phase 3c in Rust.
2. **Write `phase4_full_metric.rs`** properly — implement Christoffel /
   Riemann / Einstein / NEC contraction over burn tensors, using the
   same 5-point FD stencil that `src/curvature.rs` already uses for
   pure-Rust f64 GR. The proxy_loss stub gets replaced with the real
   NEC integral.
3. **Multi-seed sweep** driver — spawn 4× concurrent training runs
   (one per GPU), each with a different (seed, R, v, lambda) cell,
   collect ratios, dump to JSONL.
4. **Run for ≥ 24 hours** across the 5 × 4 × 4 × 3 × 16 cell space.
5. Analysis + arXiv writeup.

## Honest expectation

Workshop / arXiv paper achievable in 2-3 weeks of focused work.
PRD/CQG submission target: 3-6 months.
Nature/PRL: not on the realistic 6-month horizon; depends entirely on
what the full-metric search actually finds.
