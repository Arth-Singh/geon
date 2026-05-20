//! Quantify the energy-condition violation of the Phase 2 metrics.
//!
//! For each metric we:
//!   1. Sweep along a relevant 1D line (axis of bubble motion for Alcubierre,
//!      radial axis for Morris–Thorne).
//!   2. At each point, compute T^μν numerically and contract with a null
//!      vector k aligned with the natural direction (e.g. parallel to the
//!      ship's velocity for Alcubierre).
//!   3. Report ρ_null = T_μν k^μ k^ν.  Any negative value violates the NEC
//!      and confirms the metric requires exotic matter.
//!
//! For sanity, we also run Schwarzschild and Minkowski — Schwarzschild is
//! vacuum so T = 0; Minkowski is trivially T = 0.

use geon::curvature::{nec_contraction, ricci_scalar, stress_energy};
use geon::metric::alcubierre::Alcubierre;
use geon::metric::minkowski::Minkowski;
use geon::metric::morris_thorne::MorrisThorne;
use geon::metric::schwarzschild::Schwarzschild;
use geon::types::Vec4;

fn header(name: &str) {
    println!("\n== {name} ==");
}

fn print_row(label: &str, val: f64, threshold: f64, note: &str) {
    let flag = if val < -threshold {
        "NEC VIOLATED"
    } else if val.abs() < threshold {
        "≈ 0"
    } else {
        "NEC ok"
    };
    println!("  {label:<28}  {val:+13.4e}   [{flag}]  {note}");
}

fn main() {
    // Sanity: Minkowski (T = 0).
    header("Minkowski (sanity, T = 0 expected)");
    {
        let m = Minkowski;
        let x = [0.0, 1.0, 1.0, 1.0];
        let k = [1.0, 1.0, 0.0, 0.0]; // null in (-,+,+,+)
        let nec = nec_contraction(&m, &x, &k);
        let r = ricci_scalar(&m, &x);
        print_row("T_μν k^μ k^ν", nec, 1e-10, "should be 0");
        print_row("Ricci scalar", r, 1e-10, "should be 0");
    }

    // Sanity: Schwarzschild (vacuum, T = 0, R = 0; Riemann ≠ 0).
    header("Schwarzschild (sanity, vacuum: T = 0, R = 0)");
    {
        let m = Schwarzschild::new(1.0);
        // Try several radii outside the horizon.
        for r in [5.0, 10.0, 30.0] {
            let x: Vec4 = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
            // A null vector in Schwarzschild at this point: p^t = 1/√f, p^r = -√f.
            let f = 1.0 - 2.0 / r;
            let k: Vec4 = [1.0 / f.sqrt(), -f.sqrt(), 0.0, 0.0];
            let nec = nec_contraction(&m, &x, &k);
            let s = ricci_scalar(&m, &x);
            let label = format!("r = {r:>4}  NEC contraction");
            print_row(&label, nec, 1e-4, "vacuum → ~0");
            let label = format!("r = {r:>4}  Ricci scalar");
            print_row(&label, s, 1e-4, "vacuum → ~0");
        }
    }

    // Alcubierre: scan along the bubble's direction of motion.
    header("Alcubierre warp bubble (expecting NEC violation across bubble wall)");
    {
        let m = Alcubierre {
            bubble_radius: 2.0,
            bubble_thickness: 0.5,
            ship_velocity: 2.0,
            ship_x: 0.0,
        };
        // Null vector along +x in coordinate basis. In flat asymptotic
        // regions this is null; inside the bubble it isn't quite — but
        // we're probing the *background* null cone, which is what physical
        // observers (radiation passing through) experience.
        let k: Vec4 = [1.0, 1.0, 0.0, 0.0];
        let mut min_val = f64::INFINITY;
        let mut min_x = 0.0;
        println!("  scanning along x at y=z=0:");
        for n in 0..41 {
            let x_pos = -4.0 + 0.2 * n as f64;
            let x: Vec4 = [0.0, x_pos, 0.0, 0.0];
            let nec = nec_contraction(&m, &x, &k);
            if nec < min_val {
                min_val = nec;
                min_x = x_pos;
            }
            // Print every 4th step so the table stays readable.
            if n % 4 == 0 {
                let label = format!("x = {x_pos:>6.2}");
                print_row(&label, nec, 1e-6, "");
            }
        }
        println!(
            "  minimum T_μν k^μ k^ν = {min_val:+.4e} at x = {min_x:.2}  (negative = exotic matter required)"
        );

        // Now compare to expected: integrated negative energy along the
        // axis, in units of 1/(8π). This is a crude estimate of the
        // exotic-matter budget.
        let mut integral = 0.0;
        let dx = 0.05;
        let mut x_pos = -5.0;
        while x_pos <= 5.0 {
            let x: Vec4 = [0.0, x_pos, 0.0, 0.0];
            let nec = nec_contraction(&m, &x, &k);
            if nec < 0.0 {
                integral += nec * dx;
            }
            x_pos += dx;
        }
        println!(
            "  ∫ T_kk dx (negative part only, axis only) ≈ {integral:+.4e}  (rough exotic-mass proxy)"
        );
    }

    // Morris–Thorne: scan across the throat.
    header("Morris–Thorne wormhole (expecting NEC violation near throat)");
    {
        let m = MorrisThorne { throat_radius: 1.0 };
        let k: Vec4 = [1.0, 1.0, 0.0, 0.0]; // (t, l, θ, φ) — null in flat regions.
        println!("  scanning along l (proper radial) at θ=π/2, φ=0:");
        let mut min_val = f64::INFINITY;
        let mut min_l = 0.0;
        for n in 0..41 {
            let l = -4.0 + 0.2 * n as f64;
            let x: Vec4 = [0.0, l, std::f64::consts::FRAC_PI_2, 0.0];
            let nec = nec_contraction(&m, &x, &k);
            if nec < min_val {
                min_val = nec;
                min_l = l;
            }
            if n % 4 == 0 {
                let label = format!("l = {l:>6.2}");
                print_row(&label, nec, 1e-6, "");
            }
        }
        println!(
            "  minimum T_μν k^μ k^ν = {min_val:+.4e} at l = {min_l:.2}  (negative ⇒ exotic matter at throat)"
        );

        // Stress-energy diagonal at the throat itself.
        let x_throat: Vec4 = [0.0, 0.0, std::f64::consts::FRAC_PI_2, 0.0];
        let t = stress_energy(&m, &x_throat);
        println!("\n  T^μ_ν diagonal at throat (l=0):");
        for i in 0..4 {
            println!("    T[{i}][{i}] = {:+.4e}", t[i][i]);
        }
    }
}
