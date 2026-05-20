//! Burn autodiff smoke test on B200 CUDA. Phase 4 prerequisite.
//!
//! Tiny MLP f: R -> R, then verify that burn's reverse-mode autograd
//! against the CUDA backend agrees with a central finite-difference
//! computation to better than 1e-3.

use burn::backend::{Autodiff, Cuda};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{gelu, sigmoid};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

type B = Autodiff<Cuda>;
type Inner = <B as AutodiffBackend>::InnerBackend;

#[derive(Module, Debug)]
struct Mlp<BB: Backend> {
    l1: Linear<BB>,
    l2: Linear<BB>,
    l3: Linear<BB>,
}

impl<BB: Backend> Mlp<BB> {
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
    let device: <B as Backend>::Device = Default::default();
    println!("backend: {}  device: {:?}", std::any::type_name::<B>(), &device);

    let net: Mlp<B> = Mlp::new(&device);

    let n = 8usize;
    let xs: Vec<f32> = (0..n)
        .map(|i| 0.5 + 3.0 * (i as f32 + 0.5) / n as f32)
        .collect();

    // ---- autodiff ----
    let x: Tensor<B, 2> = Tensor::<B, 1>::from_floats(xs.as_slice(), &device)
        .reshape([n, 1])
        .require_grad();
    let y = net.forward(x.clone());
    let grads = y.clone().sum().backward();
    let dy_dx_ad: Vec<f32> = x.grad(&grads).expect("grad on x").into_data().to_vec().unwrap();
    let y_vec: Vec<f32> = y.into_data().to_vec().unwrap();

    // ---- finite difference on the inner (non-autodiff) backend ----
    let net_fd: Mlp<Inner> = net.valid();
    let device_inner: <Inner as Backend>::Device = Default::default();
    let h = 1e-3_f32;
    let xs_p: Vec<f32> = xs.iter().map(|v| v + h).collect();
    let xs_m: Vec<f32> = xs.iter().map(|v| v - h).collect();
    let xp: Tensor<Inner, 2> = Tensor::<Inner, 1>::from_floats(xs_p.as_slice(), &device_inner).reshape([n, 1]);
    let xm: Tensor<Inner, 2> = Tensor::<Inner, 1>::from_floats(xs_m.as_slice(), &device_inner).reshape([n, 1]);
    let yp: Vec<f32> = net_fd.forward(xp).into_data().to_vec().unwrap();
    let ym: Vec<f32> = net_fd.forward(xm).into_data().to_vec().unwrap();
    let dy_dx_fd: Vec<f32> = (0..n).map(|i| (yp[i] - ym[i]) / (2.0 * h)).collect();

    // ---- compare ----
    println!(
        "{:>4}  {:>10}  {:>14}  {:>14}  {:>14}  {:>10}",
        "i", "x", "y", "dy/dx ad", "dy/dx fd", "|err|"
    );
    let mut max_err = 0.0f32;
    for i in 0..n {
        let err = (dy_dx_ad[i] - dy_dx_fd[i]).abs();
        max_err = max_err.max(err);
        println!(
            "{:>4}  {:>10.5}  {:>14.6e}  {:>14.6e}  {:>14.6e}  {:>10.3e}",
            i, xs[i], y_vec[i], dy_dx_ad[i], dy_dx_fd[i], err
        );
    }
    println!("\nmax |ad - fd| = {:.3e}", max_err);
    if max_err < 1e-3 {
        println!("PASS — burn autograd on CUDA matches finite differences.");
    } else {
        println!("FAIL — autograd disagrees with finite difference.");
        std::process::exit(1);
    }
}
