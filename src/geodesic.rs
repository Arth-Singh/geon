//! Geodesic integration for any `Metric`.
//!
//! State y = [x⁰, x¹, x², x³, p⁰, p¹, p², p³] ∈ ℝ⁸.
//! Equations of motion:
//!     dx^μ/dλ = p^μ
//!     dp^μ/dλ = -Γ^μ_αβ p^α p^β
//!
//! For null geodesics the affine parameter λ is normalised by the choice of
//! initial p — we don't enforce |p|=1 explicitly. We monitor g_μν p^μ p^ν as
//! a diagnostic; for a correct integrator it should stay close to its initial
//! value (zero for null rays) throughout.

use crate::integrator::{State, rk4_step};
use crate::metric::{Metric, Termination};
use crate::types::{Scalar, Vec4, dot4};

#[derive(Debug, Clone)]
pub struct TraceResult {
    pub termination: Termination,
    pub final_state: State,
    pub steps_taken: usize,
    /// |g(p, p)| at the end of integration. Should be ~0 for a null ray; a
    /// large value means the integrator drifted off the null cone.
    pub null_residual: Scalar,
}

#[inline]
pub fn pos(y: &State) -> Vec4 {
    [y[0], y[1], y[2], y[3]]
}

#[inline]
pub fn mom(y: &State) -> Vec4 {
    [y[4], y[5], y[6], y[7]]
}

/// Evaluate the geodesic ODE at state y.
pub fn rhs<M: Metric>(metric: &M, y: &State) -> State {
    let x = pos(y);
    let p = mom(y);
    let g = metric.christoffel(&x);

    let mut out = [0.0; 8];
    // dx^μ/dλ = p^μ
    for mu in 0..4 {
        out[mu] = p[mu];
    }
    // dp^μ/dλ = -Γ^μ_αβ p^α p^β
    for mu in 0..4 {
        let mut s = 0.0;
        for alpha in 0..4 {
            for beta in 0..4 {
                s += g[mu][alpha][beta] * p[alpha] * p[beta];
            }
        }
        out[4 + mu] = -s;
    }
    out
}

/// Integrate a single ray until termination.
pub fn trace<M: Metric>(
    metric: &M,
    y0: State,
    step: Scalar,
    max_steps: usize,
) -> TraceResult {
    let mut y = y0;
    let f = |y: &State| rhs(metric, y);

    for n in 0..max_steps {
        // Check termination on current position.
        let x = pos(&y);
        if let Some(reason) = metric.terminate(&x) {
            let g = metric.g(&x);
            let p = mom(&y);
            let resid = dot4(&g, &p, &p).abs();
            return TraceResult {
                termination: reason,
                final_state: y,
                steps_taken: n,
                null_residual: resid,
            };
        }

        // Step.
        let y_next = rk4_step(&y, step, &f);

        // NaN guard.
        if y_next.iter().any(|v| !v.is_finite()) {
            let g = metric.g(&x);
            let p = mom(&y);
            let resid = dot4(&g, &p, &p).abs();
            return TraceResult {
                termination: Termination::Diverged,
                final_state: y,
                steps_taken: n,
                null_residual: resid,
            };
        }
        y = y_next;
    }

    // Ran out of steps without terminating — treat as diverged so it's
    // visible. If this happens for many rays, raise `max_steps` or shrink
    // `step` rather than papering over it.
    let x = pos(&y);
    let g = metric.g(&x);
    let p = mom(&y);
    let resid = dot4(&g, &p, &p).abs();
    TraceResult {
        termination: Termination::Diverged,
        final_state: y,
        steps_taken: max_steps,
        null_residual: resid,
    }
}
