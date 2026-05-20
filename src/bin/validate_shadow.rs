//! Validate the Schwarzschild ray tracer against the analytic shadow.
//!
//! Method: shoot rays in the equatorial plane (θ = π/2) from a static
//! observer at r_obs with various local-frame angles α from the radial
//! direction. Bisect on α to find the critical angle α_c separating
//! captured from escaping rays. Convert to impact parameter
//!   b = r_obs · sin(α) / √(1 - 2M/r_obs)
//! Compare b_c (measured) to b_c (analytic) = 3√3 M ≈ 5.19615.
//!
//! If |b_c_meas - b_c_analytic| / b_c_analytic < 0.01, Phase 1 passes.

use geon::geodesic::trace;
use geon::integrator::State;
use geon::metric::schwarzschild::Schwarzschild;
use geon::types::Scalar;

/// Build a null ray in the equatorial plane at the observer, with local
/// angle `alpha` from the inward radial direction. This bypasses the
/// general camera so we test the *physics*, not the camera scaffolding.
fn equatorial_ray(metric: &Schwarzschild, r_obs: Scalar, alpha: Scalar) -> State {
    let m = metric.mass;
    let f = 1.0 - 2.0 * m / r_obs;
    let sin_a = alpha.sin();
    let cos_a = alpha.cos();

    // Local-frame components: p^(t̂) = 1, photon launched inward at angle α.
    // Spatial: n̂_r = -cos α (inward), n̂_φ = +sin α.
    // Lift to coordinate basis via the analytic Schwarzschild tetrad.
    let pt = 1.0 / f.sqrt();
    let pr = -cos_a * f.sqrt();
    let pphi = sin_a / r_obs; // r sin θ = r at θ=π/2
    let ptheta = 0.0;

    [0.0, r_obs, std::f64::consts::FRAC_PI_2, 0.0, pt, pr, ptheta, pphi]
}

fn is_captured(metric: &Schwarzschild, r_obs: Scalar, alpha: Scalar) -> bool {
    let y0 = equatorial_ray(metric, r_obs, alpha);
    let res = trace(metric, y0, 0.01, 200_000);
    matches!(res.termination, geon::Termination::Captured)
}

fn main() {
    let metric = Schwarzschild::new(1.0);
    let r_obs = 20.0;

    let b_analytic = 3.0_f64.sqrt() * 3.0;
    let f = 1.0 - 2.0 * metric.mass / r_obs;
    let alpha_analytic = (b_analytic * f.sqrt() / r_obs).asin();

    println!("== Schwarzschild shadow validation ==");
    println!("M = {}, r_obs = {}", metric.mass, r_obs);
    println!(
        "analytic: b_c = 3√3 M = {:.6}, α_c = {:.6} rad = {:.4}°",
        b_analytic,
        alpha_analytic,
        alpha_analytic.to_degrees()
    );

    // Bisect on α. Capture must be monotone in α (smaller α → more radial →
    // more likely to be captured), so bisection is well-defined.
    let mut lo = 0.0_f64;
    let mut hi = std::f64::consts::FRAC_PI_2 - 1e-3;
    assert!(is_captured(&metric, r_obs, lo), "radial ray should be captured");
    assert!(
        !is_captured(&metric, r_obs, hi),
        "grazing ray should escape"
    );

    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if is_captured(&metric, r_obs, mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let alpha_c = 0.5 * (lo + hi);
    let b_c = r_obs * alpha_c.sin() / f.sqrt();

    println!(
        "measured: b_c = {:.6}, α_c = {:.6} rad = {:.4}°",
        b_c,
        alpha_c,
        alpha_c.to_degrees()
    );
    let rel_err = (b_c - b_analytic).abs() / b_analytic;
    println!("relative error: {:.4e}", rel_err);
    if rel_err < 0.01 {
        println!("PASS (< 1% error)");
    } else {
        println!("FAIL — Phase 1 not validated. Integrator or metric is wrong.");
        std::process::exit(1);
    }
}
