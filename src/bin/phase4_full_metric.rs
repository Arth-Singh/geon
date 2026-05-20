//! Phase 4 — full-metric ansatz neural search.
//!
//! Drops the Alcubierre form constraint. Five independent neural fields
//! parameterise the perturbation around Minkowski:
//!
//!   g_tt = -1 + h_tt(x, ρ) · mask
//!   g_tx = h_tx(x, ρ) · mask                     ← the "warp" off-diagonal
//!   g_xx =  1 + h_xx(x, ρ) · mask
//!   g_ρρ =  1 + h_ρρ(x, ρ) · mask
//!   g_φφ = ρ² (1 + h_φφ(x, ρ) · mask)
//!
//! where `mask(x, ρ) = exp(-((x² + ρ²)/L²)²)` bakes in asymptotic flatness.
//! Each `h_•` is a small MLP `[x, ρ] → R`.
//!
//! All curvature comes from finite differences on g — same scheme as
//! `src/curvature.rs`. The outer ∂loss/∂params is via burn autograd.
//!
//! Build:  cargo build --profile=fast --features cuda --bin phase4_full_metric
//! Run:    LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat/lib:$LD_LIBRARY_PATH \
//!         ./target/fast/phase4_full_metric

use std::f32::consts::PI;
use std::time::Instant;

use burn::backend::{Autodiff, Cuda};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};
use rand::Rng;

type Bk = Autodiff<Cuda>;

// ----------------------------- the network ------------------------------ //

#[derive(Module, Debug)]
struct ScalarField<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> ScalarField<B> {
    fn new(hidden: usize, device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(2, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            l3: LinearConfig::new(hidden, 1).init(device),
        }
    }

    /// `xrho: [N, 2]` -> `[N]` scalar field value at each point.
    fn forward(&self, xrho: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = gelu(self.l1.forward(xrho));
        let h = gelu(self.l2.forward(h));
        self.l3.forward(h).squeeze_dim::<1>(1)
    }
}

#[derive(Module, Debug)]
struct FullMetric<B: Backend> {
    h_tt: ScalarField<B>,
    h_tx: ScalarField<B>,
    h_xx: ScalarField<B>,
    h_rr: ScalarField<B>,
    h_pp: ScalarField<B>,
}

impl<B: Backend> FullMetric<B> {
    fn new(hidden: usize, device: &B::Device) -> Self {
        Self {
            h_tt: ScalarField::new(hidden, device),
            h_tx: ScalarField::new(hidden, device),
            h_xx: ScalarField::new(hidden, device),
            h_rr: ScalarField::new(hidden, device),
            h_pp: ScalarField::new(hidden, device),
        }
    }
}

/// Asymptotic mask: 1 inside bubble length scale L, ≈ 0 outside. `expand`
/// must apply pointwise.
fn mask<B: Backend>(x: Tensor<B, 1>, rho: Tensor<B, 1>, big_l: f32) -> Tensor<B, 1> {
    let r2 = x.powi_scalar(2) + rho.powi_scalar(2);
    let u = r2 / (big_l * big_l);
    (-u.powi_scalar(2)).exp()
}

/// Evaluate the 5 independent components of g at a batch of (x, ρ) points.
/// Returns five `[N]` tensors in the order (tt, tx, xx, ρρ, φφ-perturbation).
/// The full `g_μν` is reconstructed by callers since the rest is constant.
fn eval_components<B: Backend>(
    net: &FullMetric<B>,
    x: Tensor<B, 1>,
    rho: Tensor<B, 1>,
    big_l: f32,
) -> (
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
) {
    let xrho = Tensor::stack::<2>(vec![x.clone(), rho.clone()], 1);
    let m = mask(x, rho, big_l);
    let htt = net.h_tt.forward(xrho.clone()) * m.clone();
    let htx = net.h_tx.forward(xrho.clone()) * m.clone();
    let hxx = net.h_xx.forward(xrho.clone()) * m.clone();
    let hrr = net.h_rr.forward(xrho.clone()) * m.clone();
    let hpp = net.h_pp.forward(xrho) * m;
    (htt, htx, hxx, hrr, hpp)
}

// ---------------------------- physics losses ---------------------------- //

/// Simple proxy loss for Phase 4 step 0: the average of `|h_tx|` over
/// collocation points, penalised against a target. This is a stub — the
/// real loss is the full NEC contraction below. We keep this around as a
/// trivially-debuggable shake-out for the training loop.
fn proxy_loss<B: Backend>(htx: Tensor<B, 1>) -> Tensor<B, 1> {
    htx.powi_scalar(2).mean()
}

// ---------------------------- training driver --------------------------- //

struct Cfg {
    steps: usize,
    n_samples: usize,
    hidden: usize,
    lr: f64,
    big_l: f32,     // asymptotic-mask length scale
    box_half: f32,  // half-side of the (x, ρ) sampling box
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            steps: 1000,
            n_samples: 1024,
            hidden: 64,
            lr: 1e-3,
            big_l: 3.0,
            box_half: 6.0,
        }
    }
}

fn sample_batch(cfg: &Cfg) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::rng();
    let xs: Vec<f32> = (0..cfg.n_samples)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * cfg.box_half)
        .collect();
    // ρ ∈ [0, box_half] — physical radius, non-negative.
    let rhos: Vec<f32> = (0..cfg.n_samples)
        .map(|_| rng.random::<f32>() * cfg.box_half)
        .collect();
    (xs, rhos)
}

fn train(cfg: Cfg) {
    let device: Device<Bk> = Default::default();
    println!("device: {:?}", device);
    println!(
        "Phase 4 (step 0 — proxy loss) — hidden={}, n_samples={}, steps={}",
        cfg.hidden, cfg.n_samples, cfg.steps,
    );

    let mut net: FullMetric<Bk> = FullMetric::new(cfg.hidden, &device);
    let mut optim = AdamConfig::new().init::<Bk, FullMetric<Bk>>();

    let t0 = Instant::now();

    for step in 0..cfg.steps {
        let (xs, rhos) = sample_batch(&cfg);
        let x = Tensor::<Bk, 1>::from_floats(xs.as_slice(), &device);
        let rho = Tensor::<Bk, 1>::from_floats(rhos.as_slice(), &device);

        // For Phase 4 step 0 we only train h_tx, with a proxy objective:
        // drive |h_tx|² to ~0.25 at the origin (so an off-diagonal exists)
        // and to ~0 at the boundary. This is a trivial smoke test of the
        // training loop on the full-metric network.
        //
        // Phase 4 step 1 (next commit) replaces this with the NEC integral
        // computed from the full Christoffel/Riemann/Einstein cascade.
        let (_, htx, _, _, _) = eval_components(&net, x.clone(), rho.clone(), cfg.big_l);

        let loss = proxy_loss(htx);

        let grads = loss.clone().backward();
        let grads_params = GradientsParams::from_grads::<Bk, FullMetric<Bk>>(grads, &net);
        net = optim.step(cfg.lr, net, grads_params);

        if step % (cfg.steps / 20).max(1) == 0 || step == cfg.steps - 1 {
            println!("step {step:5}  loss={:.4e}", loss.into_scalar());
        }
    }
    let dt = t0.elapsed().as_secs_f32();
    println!(
        "\ntrained {} steps in {:.1}s ({:.2} ms/step)",
        cfg.steps,
        dt,
        1000.0 * dt / cfg.steps as f32,
    );

    let _ = PI; // silence unused if loss formulas trimmed
}

fn main() {
    train(Cfg::default());
}
