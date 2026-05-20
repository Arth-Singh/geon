//! Phase 4 step 3 — Rust port of phase3/warp_search_v3.py.
//!
//! Reproduces the falsifiability-tested 0.9963 ratio against canonical
//! Alcubierre, in pure Rust + burn + CUDA. Once this matches the Python
//! result to ~0.5% on the same hyperparameters, Phase 4 full-metric ansatz
//! can be built on the same training loop.
//!
//! Build:
//!   cargo build --release --features cuda --bin phase4_shape
//! Run:
//!   ./target/release/phase4_shape

use std::f32::consts::PI;
use std::time::Instant;

use burn::backend::{Autodiff, Cuda};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{gelu, sigmoid};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution, Tensor, TensorData};

type B = Autodiff<Cuda>;
type Inner = <B as AutodiffBackend>::InnerBackend;

#[derive(Module, Debug, Clone)]
struct ShapeNet<BB: Backend> {
    l1: Linear<BB>,
    l2: Linear<BB>,
    l3: Linear<BB>,
    l4: Linear<BB>,
}

impl<BB: Backend> ShapeNet<BB> {
    fn new(hidden: usize, device: &BB::Device) -> Self {
        Self {
            l1: LinearConfig::new(1, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            l3: LinearConfig::new(hidden, hidden).init(device),
            l4: LinearConfig::new(hidden, 1).init(device),
        }
    }

    fn forward(&self, r: Tensor<BB, 2>) -> Tensor<BB, 2> {
        let h = gelu(self.l1.forward(r));
        let h = gelu(self.l2.forward(h));
        let h = gelu(self.l3.forward(h));
        sigmoid(self.l4.forward(h))
    }
}

// ---------- canonical Alcubierre baseline ----------

fn canonical_f(r: &[f32], big_r: f32, sigma: f32) -> Vec<f32> {
    let denom = 2.0 * (sigma * big_r).tanh();
    r.iter()
        .map(|&x| ((sigma * (x + big_r)).tanh() - (sigma * (x - big_r)).tanh()) / denom)
        .collect()
}

fn canonical_fp(r: &[f32], big_r: f32, sigma: f32) -> Vec<f32> {
    let denom = 2.0 * (sigma * big_r).tanh();
    r.iter()
        .map(|&x| {
            let sech_p = 1.0 - (sigma * (x + big_r)).tanh().powi(2);
            let sech_m = 1.0 - (sigma * (x - big_r)).tanh().powi(2);
            sigma * (sech_p - sech_m) / denom
        })
        .collect()
}

// ---------- physics ----------

/// Exotic-mass integrand kernel: (v²/12) · (f')² · r² · 4π.  Tensor version.
fn integrand_kernel<BB: Backend>(
    fp: Tensor<BB, 1>,
    r: Tensor<BB, 1>,
    v: f32,
) -> Tensor<BB, 1> {
    let coef = (v * v / 12.0) * 4.0 * PI;
    fp.powi_scalar(2) * r.powi_scalar(2) * coef
}

/// Trapezoidal integration on a *sorted-ascending* r grid.
fn trap<BB: Backend>(integrand: Tensor<BB, 1>, r: Tensor<BB, 1>) -> Tensor<BB, 1> {
    let n = integrand.dims()[0];
    let int_a = integrand.clone().slice([0..n - 1]);
    let int_b = integrand.slice([1..n]);
    let r_a = r.clone().slice([0..n - 1]);
    let r_b = r.slice([1..n]);
    let dr = r_b - r_a;
    let avg = (int_a + int_b) * 0.5;
    (avg * dr).sum()
}

// ---------- training ----------

struct Cfg {
    steps: usize,
    n_samples: usize,
    big_r: f32,
    sigma: f32,
    delta: f32,
    r_max: f32,
    v: f32,
    lambda_interior: f32,
    lambda_exterior: f32,
    lambda_pin: f32,
    lambda_smooth: f32,
    lr: f64,
    hidden: usize,
    seed: u64,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            steps: 6000,
            n_samples: 600,
            big_r: 2.0,
            sigma: 4.0,
            delta: 0.4,
            r_max: 8.0,
            v: 1.0,
            lambda_interior: 1e5,
            lambda_exterior: 1e5,
            lambda_pin: 1e4,
            lambda_smooth: 1e-3,
            lr: 1e-3,
            hidden: 128,
            seed: 42,
        }
    }
}

/// Build the randomised collocation grid for one training step.
/// Returns r sorted ascending, plus the (n_in, n_wall, n_out) slice lengths.
fn make_grid(cfg: &Cfg, device: &<B as Backend>::Device) -> (Tensor<B, 1>, usize, usize, usize) {
    use rand::Rng;
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
        .map(|_| rng.random::<f32>() * (cfg.r_max - (cfg.big_r + cfg.delta)) + (cfg.big_r + cfg.delta))
        .collect();
    r_out.push(cfg.big_r + cfg.delta);
    r_out.push(cfg.r_max);
    r_out.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut all: Vec<f32> = Vec::with_capacity(cfg.n_samples);
    all.extend_from_slice(&r_in);
    all.extend_from_slice(&r_wall);
    all.extend_from_slice(&r_out);

    let r_tensor: Tensor<B, 1> = Tensor::<B, 1>::from_floats(all.as_slice(), device);
    (r_tensor, n_in, n_wall, n_out)
}

fn train(cfg: Cfg) {
    let device: <B as Backend>::Device = Default::default();
    println!("device: {:?}", device);

    let mut net: ShapeNet<B> = ShapeNet::new(cfg.hidden, &device);
    let mut optim = AdamConfig::new().init();

    let r_pin: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats([cfg.big_r], &device).reshape([1, 1]);

    let t0 = Instant::now();

    for step in 0..cfg.steps {
        let (r, n_in, _n_wall, n_out) = make_grid(&cfg, &device);
        let r_in = r.clone().slice([0..n_in]).require_grad();
        let _r_wall = r.clone().slice([n_in..n_in + _n_wall]).require_grad();
        let _r_out = r.clone().slice([n_in + _n_wall..cfg.n_samples]).require_grad();

        // Recompose full r with grad tracked. We need r to be a leaf for the
        // ∂f/∂r autograd to work — easiest path: build one require_grad'd r
        // tensor for the forward pass.
        let r_full: Tensor<B, 1> = Tensor::<B, 1>::from_floats(
            &r.clone().into_data().to_vec::<f32>().unwrap()[..],
            &device,
        )
        .require_grad();
        let r2 = r_full.clone().reshape([cfg.n_samples, 1]);
        let f = net.forward(r2).reshape([cfg.n_samples]);

        // ∂f/∂r per sample (since the MLP is pointwise, ∂(sum f)/∂r_i = ∂f_i/∂r_i).
        let grads_r = f.clone().sum().backward();
        let fp = r_full
            .grad(&grads_r)
            .expect("r should have grad")
            .clone();
        let fp_t: Tensor<B, 1> = Tensor::<B, 1>::from_inner(fp);

        // Loss = exotic + smoothness (finite-diff f'') + region penalties + pin.
        let exotic = trap(integrand_kernel(fp_t.clone(), r_full.clone(), cfg.v), r_full.clone());

        // Smoothness: finite-diff f' on the sorted grid.
        let n = cfg.n_samples;
        let fp_a = fp_t.clone().slice([0..n - 1]);
        let fp_b = fp_t.clone().slice([1..n]);
        let r_a = r_full.clone().slice([0..n - 1]);
        let r_b = r_full.clone().slice([1..n]);
        let fpp = (fp_b - fp_a) / (r_b - r_a);
        let smooth = (fpp.powi_scalar(2)).sum();

        // Region penalties.
        let f_in = f.clone().slice([0..n_in]);
        let f_out = f.clone().slice([n - n_out..n]);
        let interior_pen = (f_in - 1.0).powi_scalar(2).mean();
        let exterior_pen = f_out.powi_scalar(2).mean();

        // Pin f(big_r) = 0.5.
        let pin = (net.forward(r_pin.clone()).squeeze::<1>(0).squeeze::<0>(0) - 0.5).powi_scalar(2);

        let loss = exotic.clone()
            + smooth.clone() * cfg.lambda_smooth
            + interior_pen.clone() * cfg.lambda_interior
            + exterior_pen.clone() * cfg.lambda_exterior
            + pin.clone() * cfg.lambda_pin;

        let grads = loss.clone().backward();
        let grads = GradientsParams::from_grads(grads, &net);
        net = optim.step(cfg.lr, net, grads);

        if step % (cfg.steps / 20).max(1) == 0 || step == cfg.steps - 1 {
            let loss_v = loss.into_scalar();
            let ex_v = exotic.into_scalar();
            let sm_v = smooth.into_scalar();
            let ip = interior_pen.into_scalar();
            let op = exterior_pen.into_scalar();
            let pp = pin.into_scalar();
            println!(
                "step {step:5}  loss={loss_v:.4e}  ex={ex_v:.4e}  sm={sm_v:.3e}  in={ip:.2e}  out={op:.2e}  pin={pp:.2e}",
            );
        }
    }
    let dt = t0.elapsed().as_secs_f32();
    println!("\ntrained {} steps in {:.1}s ({:.1} ms/step)", cfg.steps, dt, 1000.0 * dt / cfg.steps as f32);

    // ----- evaluation on dense grid -----
    let net_inner: ShapeNet<Inner> = net.valid();
    let device_inner: <Inner as Backend>::Device = Default::default();

    let neval = 4096usize;
    let r_dense_v: Vec<f32> = (0..neval)
        .map(|i| cfg.r_max * (i as f32 + 0.5) / neval as f32)
        .collect();
    let r_dense: Tensor<Inner, 2> =
        Tensor::<Inner, 1>::from_floats(r_dense_v.as_slice(), &device_inner).reshape([neval, 1]);
    let f_learned: Vec<f32> = net_inner.forward(r_dense).into_data().to_vec().unwrap();

    let f_canon = canonical_f(&r_dense_v, cfg.big_r, cfg.sigma);
    let fp_canon = canonical_fp(&r_dense_v, cfg.big_r, cfg.sigma);

    // Finite-difference fp on f_learned.
    let mut fp_learned = vec![0.0f32; neval];
    fp_learned[0] = (f_learned[1] - f_learned[0]) / (r_dense_v[1] - r_dense_v[0]);
    fp_learned[neval - 1] = (f_learned[neval - 1] - f_learned[neval - 2])
        / (r_dense_v[neval - 1] - r_dense_v[neval - 2]);
    for i in 1..neval - 1 {
        fp_learned[i] = (f_learned[i + 1] - f_learned[i - 1]) / (r_dense_v[i + 1] - r_dense_v[i - 1]);
    }

    let coef = (cfg.v * cfg.v / 12.0) * 4.0 * PI;
    let trap_dense = |fp: &[f32]| -> f32 {
        let mut s = 0.0_f64;
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

    // TV(f') comparison.
    let tv: fn(&[f32]) -> f32 = |fp: &[f32]| {
        fp.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
    };
    println!("  TV(f') learned  = {:.4e}", tv(&fp_learned));
    println!("  TV(f') canonical = {:.4e}", tv(&fp_canon));
}

fn main() {
    train(Cfg::default());
}
