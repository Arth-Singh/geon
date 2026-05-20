//! Generic ODE integrator. We only need RK4 right now — the geodesic equation
//! is well-conditioned for null rays at fixed step away from horizons, and
//! the camera math is more likely to be wrong than the integration step is.
//!
//! Adaptive stepping (DOP853 with error control) is a TODO when we start
//! caring about the photon-sphere fringe of the shadow. Cash-Karp would be
//! the natural upgrade — it shares stages with RK4 so the cost is low.

use crate::types::Scalar;

/// State vector type. We use a fixed 8-element array (position + momentum).
pub type State = [Scalar; 8];

/// Standard 4th-order Runge–Kutta step.
#[inline]
pub fn rk4_step<F: Fn(&State) -> State>(y: &State, h: Scalar, f: &F) -> State {
    let k1 = f(y);
    let mut y2 = *y;
    for i in 0..8 {
        y2[i] = y[i] + 0.5 * h * k1[i];
    }
    let k2 = f(&y2);
    let mut y3 = *y;
    for i in 0..8 {
        y3[i] = y[i] + 0.5 * h * k2[i];
    }
    let k3 = f(&y3);
    let mut y4 = *y;
    for i in 0..8 {
        y4[i] = y[i] + h * k3[i];
    }
    let k4 = f(&y4);

    let mut out = *y;
    let sixth = h / 6.0;
    for i in 0..8 {
        out[i] += sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}
