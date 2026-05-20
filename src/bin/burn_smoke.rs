//! Burn autodiff smoke test on B200 CUDA. Phase 4 prerequisite.
//!
//! Builds a tiny MLP f: ℝ → ℝ in burn, gets ∂f/∂x for each input via
//! reverse-mode autodiff, and verifies against a central finite-difference
//! computation. If max|ad - fd| < 1e-3, burn's autograd is wired correctly
//! against the CUDA backend and Phase 4's training loop can be built on
//! top of it.
//!
//! Build (on the B200):
//!   cargo build --release --features cuda --bin burn_smoke
//! Run:
//!   ./target/release/burn_smoke

use burn::backend::{Autodiff, Cuda};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{gelu, sigmoid};

type B = Autodiff<Cuda>;
type Inner = Cuda;

#[derive(Module, Debug)]
struct Mlp<BB: burn::tensor::backend::Backend> {
    l1: Linear<BB>,
    l2: Linear<BB>,
    l3: Linear<BB>,
}

impl<BB: burn::tensor::backend::Backend> Mlp<BB> {
    fn new(device: &BB::Device) -> Self {
        Self {
            l1: LinearConfig::new(1, 32).init(device),
            l2: LinearConfig::new(32, 32).init(device),
            l3: LinearConfig::new(32, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<BB, 2>) -> Tensor<BB, 2> {
        let h = gelu(self.l1.forward(x));
        let h = gelu(self.l2.forward(h));
        sigmoid(self.l3.forward(h))
    }
}

fn main() {
    let device: <Inner as burn::tensor::backend::Backend>::Device = Default::default();
    println!("burn backend: {:?}  device: {:?}", std::any::type_name::<B>(), &device);

    let net: Mlp<B> = Mlp::new(&device);

    let n = 8;
    let xs: Vec<f32> = (0..n)
        .map(|i| 0.5 + 3.0 * (i as f32 + 0.5) / n as f32)
        .collect();

    // ---- autodiff ----
    let x: Tensor<B, 2> = Tensor::<B, 1>::from_floats(xs.as_slice(), &device)
        .reshape([n, 1])
        .require_grad();
    let y = net.forward(x.clone());
    let grads = y.clone().sum().backward();
    let dy_dx_ad = x.grad(&grads).expect("grad should exist on x").into_data();

    // ---- finite difference (central, h = 1e-3) on the inner (non-autodiff) backend ----
    let net_fd: Mlp<Inner> = net.valid();
    let h = 1e-3_f32;
    let xs_p: Vec<f32> = xs.iter().map(|v| v + h).collect();
    let xs_m: Vec<f32> = xs.iter().map(|v| v - h).collect();
    let xp: Tensor<Inner, 2> =
        Tensor::<Inner, 1>::from_floats(xs_p.as_slice(), &device).reshape([n, 1]);
    let xm: Tensor<Inner, 2> =
        Tensor::<Inner, 1>::from_floats(xs_m.as_slice(), &device).reshape([n, 1]);
    let yp_vec: Vec<f32> = net_fd.forward(xp).into_data().to_vec().unwrap();
    let ym_vec: Vec<f32> = net_fd.forward(xm).into_data().to_vec().unwrap();
    let dy_dx_fd: Vec<f32> = (0..n)
        .map(|i| (yp_vec[i] - ym_vec[i]) / (2.0 * h))
        .collect();

    // ---- compare ----
    let ad_vec: Vec<f32> = dy_dx_ad.to_vec().unwrap();
    let y_vec: Vec<f32> = y.into_data().to_vec().unwrap();
    println!(
        "{:>4}  {:>10}  {:>14}  {:>14}  {:>14}  {:>10}",
        "i", "x", "y", "dy/dx ad", "dy/dx fd", "|err|"
    );
    let mut max_abs_err = 0.0f32;
    for i in 0..n {
        let err = (ad_vec[i] - dy_dx_fd[i]).abs();
        max_abs_err = max_abs_err.max(err);
        println!(
            "{:>4}  {:>10.5}  {:>14.6e}  {:>14.6e}  {:>14.6e}  {:>10.3e}",
            i, xs[i], y_vec[i], ad_vec[i], dy_dx_fd[i], err
        );
    }
    println!("\nmax |ad - fd| = {:.3e}", max_abs_err);
    if max_abs_err < 1e-3 {
        println!("PASS — burn autograd on CUDA is wired correctly.");
        std::process::exit(0);
    } else {
        println!("FAIL — autograd disagrees with finite difference.");
        std::process::exit(1);
    }
}
