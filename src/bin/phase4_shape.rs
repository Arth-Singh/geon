//! Phase 4 step 3 — Rust port of phase3/warp_search_v3.py.
//!
//! Reproduces the falsifiability-tested 0.9963 ratio against canonical
//! Alcubierre, in pure Rust + burn + CUDA on the B200.
//!
//! Design choice: instead of autograd-through-the-MLP for `df/dr`, we
//! evaluate `f` on a slightly-shifted grid and form a central
//! finite-difference for `f'` and `f''`. This:
//!   - avoids any double-backward question in burn
//!   - matches the discretization scheme we use for Christoffel/Riemann
//!     in `src/curvature.rs`, so Phase 4 full-metric work will reuse the
//!     same pattern
//!   - is what most PINN implementations do for spatial derivatives anyway
//!
//! The OUTER gradient (∂loss/∂params) is still via burn autograd — that's
//! the part that needs to actually backprop through the network.
//!
//! Build:
//!   cargo build --release --features cuda --bin phase4_shape
//! Run:
//!   ./target/release/phase4_shape

use std::f32::consts::PI;
use std::time::Instant;

use burn::backend::{Autodiff, Cuda};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{gelu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};
use rand::Rng;

// CRITICAL: burn 0.21 #[derive(Module)] requires the struct generic to be
// literally named `B`. Anything else (`BB`, `Backend`, etc.) causes the
// macro to silently emit no-op impls (num_params=0, no visit) and training
// becomes a no-op. So we alias the backend to `Bk` and keep `B` for the
// struct generic.
type Bk = Autodiff<Cuda>;

#[derive(Module, Debug)]
struct ShapeNet<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    l4: Linear<B>,
}

impl<B: Backend> ShapeNet<B> {
    fn new(hidden: usize, device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(1, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            l3: LinearConfig::new(hidden, hidden).init(device),
            l4: LinearConfig::new(hidden, 1).init(device),
        }
    }

    fn forward(&self, r: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = gelu(self.l1.forward(r));
        let h = gelu(self.l2.forward(h));
        let h = gelu(self.l3.forward(h));
        sigmoid(self.l4.forward(h))
    }
}

// ---------- canonical Alcubierre baseline (CPU-side fp32) ----------

fn canonical_f_cpu(r: &[f32], big_r: f32, sigma: f32) -> Vec<f32> {
    let denom = 2.0 * (sigma * big_r).tanh();
    r.iter()
        .map(|&x| ((sigma * (x + big_r)).tanh() - (sigma * (x - big_r)).tanh()) / denom)
        .collect()
}

fn canonical_fp_cpu(r: &[f32], big_r: f32, sigma: f32) -> Vec<f32> {
    let denom = 2.0 * (sigma * big_r).tanh();
    r.iter()
        .map(|&x| {
            let s_p = 1.0 - (sigma * (x + big_r)).tanh().powi(2);
            let s_m = 1.0 - (sigma * (x - big_r)).tanh().powi(2);
            sigma * (s_p - s_m) / denom
        })
        .collect()
}

// ---------- knobs ----------

struct Cfg {
    steps: usize,
    n_samples: usize,
    big_r: f32,
    sigma: f32,
    delta: f32,
    r_max: f32,
    v: f32,
    h_fd: f32, // finite-difference step for f' and f''
    lambda_interior: f32,
    lambda_exterior: f32,
    lambda_pin: f32,
    lambda_smooth: f32,
    lr: f64,
    hidden: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        // Matches `phase3/warp_search_v3.py --steps 6000 --lambda_smooth 1e-3`.
        // The Python run converges to ratio = 0.9963 under these settings;
        // we expect the Rust+CUDA version to match within fp32 + FD noise.
        Self {
            steps: 6000,
            n_samples: 600,
            big_r: 2.0,
            sigma: 4.0,
            delta: 0.4,
            r_max: 8.0,
            v: 1.0,
            h_fd: 1e-2,
            lambda_interior: 1e5,
            lambda_exterior: 1e5,
            lambda_pin: 1e4,
            lambda_smooth: 1e-3,
            lr: 1e-3,
            hidden: 128,
        }
    }
}

// ---------- randomised collocation each step ----------

fn make_grid_cpu(cfg: &Cfg) -> (Vec<f32>, usize, usize) {
    let mut rng = rand::rng();
    let n_in = cfg.n_samples / 3;
    let n_wall = cfg.n_samples / 3;
    let n_out = cfg.n_samples - n_in - n_wall;

    let mut r_in: Vec<f32> = (0..n_in - 2)
        .map(|_| rng.random::<f32>() * (cfg.big_r - cfg.delta))
        .collect();
    r_in.push(0.0);
    r_in.push(cfg.big_r - cfg.delta);
    r_in.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut r_wall: Vec<f32> = (0..n_wall)
        .map(|_| rng.random::<f32>() * (2.0 * cfg.delta) + (cfg.big_r - cfg.delta))
        .collect();
    r_wall.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut r_out: Vec<f32> = (0..n_out - 2)
        .map(|_| {
            rng.random::<f32>() * (cfg.r_max - (cfg.big_r + cfg.delta)) + (cfg.big_r + cfg.delta)
        })
        .collect();
    r_out.push(cfg.big_r + cfg.delta);
    r_out.push(cfg.r_max);
    r_out.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut all = Vec::with_capacity(cfg.n_samples);
    all.extend_from_slice(&r_in);
    all.extend_from_slice(&r_wall);
    all.extend_from_slice(&r_out);
    (all, n_in, n_out)
}

// ---------- losses, expressed as burn-tensor ops over [n] vectors ----------

/// Trapezoidal integration on a sorted-ascending r grid, using burn tensors.
fn trap<BB: Backend>(integrand: Tensor<BB, 1>, r: Tensor<BB, 1>) -> Tensor<BB, 1> {
    let n = integrand.dims()[0];
    let i_a = integrand.clone().slice([0..n - 1]);
    let i_b = integrand.slice([1..n]);
    let r_a = r.clone().slice([0..n - 1]);
    let r_b = r.slice([1..n]);
    let dr = r_b - r_a;
    let avg = (i_a + i_b) * 0.5;
    (avg * dr).sum()
}

// ---------- training ----------

fn train(cfg: Cfg) {
    let device: Device<Bk> = Default::default();
    println!("device: {:?}", device);

    let mut net: ShapeNet<Bk> = ShapeNet::new(cfg.hidden, &device);
    let mut optim = AdamConfig::new().init::<Bk, ShapeNet<Bk>>();

    let r_pin: Tensor<Bk, 2> =
        Tensor::<Bk, 1>::from_floats([cfg.big_r], &device).reshape([1, 1]);

    let t0 = Instant::now();

    for step in 0..cfg.steps {
        // ---- new randomised collocation grid ----
        let (r_vec, n_in, n_out) = make_grid_cpu(&cfg);
        let n = cfg.n_samples;
        let h = cfg.h_fd;
        let r_plus: Vec<f32> = r_vec.iter().map(|x| x + h).collect();
        let r_minus: Vec<f32> = r_vec.iter().map(|x| x - h).collect();

        let r_t: Tensor<Bk, 2> =
            Tensor::<Bk, 1>::from_floats(r_vec.as_slice(), &device).reshape([n, 1]);
        let r_p: Tensor<Bk, 2> =
            Tensor::<Bk, 1>::from_floats(r_plus.as_slice(), &device).reshape([n, 1]);
        let r_m: Tensor<Bk, 2> =
            Tensor::<Bk, 1>::from_floats(r_minus.as_slice(), &device).reshape([n, 1]);

        // ---- f, f', f'' via finite differences on three forward passes ----
        let f_t: Tensor<Bk, 1> = net.forward(r_t.clone()).squeeze_dim::<1>(1);
        let f_p: Tensor<Bk, 1> = net.forward(r_p).squeeze_dim::<1>(1);
        let f_m: Tensor<Bk, 1> = net.forward(r_m).squeeze_dim::<1>(1);
        let fp_tensor = (f_p.clone() - f_m.clone()) / (2.0 * h);
        let fpp_tensor = (f_p - f_t.clone() * 2.0 + f_m) / (h * h);

        // ---- exotic-mass integrand: (v²/12) (f')² r² · 4π ----
        let r1: Tensor<Bk, 1> = r_t.clone().squeeze_dim::<1>(1);
        let coef = (cfg.v * cfg.v / 12.0) * 4.0 * PI;
        let integrand = fp_tensor.clone().powi_scalar(2) * r1.clone().powi_scalar(2) * coef;
        let exotic = trap(integrand, r1.clone());

        // ---- smoothness penalty: integrate (f'')² over r ----
        let smooth = trap(fpp_tensor.powi_scalar(2), r1);

        // ---- region penalties ----
        let f_in = f_t.clone().slice([0..n_in]);
        let f_out = f_t.slice([n - n_out..n]);
        let interior_pen = (f_in - 1.0).powi_scalar(2).mean();
        let exterior_pen = f_out.powi_scalar(2).mean();

        // ---- pin f(R) = 0.5 ----
        let f_at_r = net.forward(r_pin.clone()).squeeze_dim::<1>(1);
        let pin = (f_at_r - 0.5).powi_scalar(2).mean();

        // ---- total ----
        let loss = exotic.clone()
            + smooth.clone() * cfg.lambda_smooth
            + interior_pen.clone() * cfg.lambda_interior
            + exterior_pen.clone() * cfg.lambda_exterior
            + pin.clone() * cfg.lambda_pin;

        // ---- step ----
        let grads = loss.clone().backward();
        let grads_params = GradientsParams::from_grads::<Bk, ShapeNet<Bk>>(grads, &net);
        net = optim.step(cfg.lr, net, grads_params);

        if step % (cfg.steps / 25).max(1) == 0 || step == cfg.steps - 1 {
            let l = loss.into_scalar();
            let ex = exotic.into_scalar();
            let sm = smooth.into_scalar();
            let ip = interior_pen.into_scalar();
            let op = exterior_pen.into_scalar();
            let pp = pin.into_scalar();
            println!(
                "step {step:5}  loss={l:.4e}  ex={ex:.4e}  sm={sm:.3e}  in={ip:.2e}  out={op:.2e}  pin={pp:.2e}",
            );
        }
    }
    let dt = t0.elapsed().as_secs_f32();
    println!(
        "\ntrained {} steps in {:.1}s ({:.1} ms/step)",
        cfg.steps,
        dt,
        1000.0 * dt / cfg.steps as f32
    );

    // ---- evaluate on a dense grid, compare to canonical ----
    let neval = 4096;
    let r_dense_v: Vec<f32> = (0..neval)
        .map(|i| cfg.r_max * (i as f32 + 0.5) / neval as f32)
        .collect();
    let r_dense: Tensor<Bk, 2> =
        Tensor::<Bk, 1>::from_floats(r_dense_v.as_slice(), &device).reshape([neval, 1]);
    let f_learned: Vec<f32> = net.forward(r_dense).into_data().to_vec().unwrap();

    let f_canon = canonical_f_cpu(&r_dense_v, cfg.big_r, cfg.sigma);
    let fp_canon = canonical_fp_cpu(&r_dense_v, cfg.big_r, cfg.sigma);

    let mut fp_learned = vec![0.0f32; neval];
    fp_learned[0] = (f_learned[1] - f_learned[0]) / (r_dense_v[1] - r_dense_v[0]);
    fp_learned[neval - 1] = (f_learned[neval - 1] - f_learned[neval - 2])
        / (r_dense_v[neval - 1] - r_dense_v[neval - 2]);
    for i in 1..neval - 1 {
        fp_learned[i] =
            (f_learned[i + 1] - f_learned[i - 1]) / (r_dense_v[i + 1] - r_dense_v[i - 1]);
    }

    let coef = (cfg.v * cfg.v / 12.0) * 4.0 * PI;
    let trap_dense = |fp: &[f32]| -> f32 {
        let mut s = 0.0f64;
        for i in 0..neval - 1 {
            let a = coef * fp[i].powi(2) * r_dense_v[i].powi(2);
            let b = coef * fp[i + 1].powi(2) * r_dense_v[i + 1].powi(2);
            s += 0.5 * (a + b) as f64 * (r_dense_v[i + 1] - r_dense_v[i]) as f64;
        }
        s as f32
    };
    let mass_learned = trap_dense(&fp_learned);
    let mass_canon = trap_dense(&fp_canon);

    println!("\n=== eval on 4096-pt dense grid ===");
    println!("  canonical exotic mass = {:.6e}", mass_canon);
    println!("  learned   exotic mass = {:.6e}", mass_learned);
    println!("  ratio learned/canon  = {:.4}", mass_learned / mass_canon);

    let tv: fn(&[f32]) -> f32 = |fp: &[f32]| fp.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    println!("  TV(f') learned  = {:.4e}", tv(&fp_learned));
    println!("  TV(f') canonical = {:.4e}", tv(&fp_canon));
}

fn main() {
    train(Cfg::default());
}
