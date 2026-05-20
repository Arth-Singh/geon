//! Minimal training sanity check: can we drive a small MLP `f(x) -> [0,1]`
//! to match the constant target f(x) = 1 over x ∈ [0, 1]? If after 1000
//! Adam steps the MSE is < 1e-2, the training plumbing works.
//!
//! IMPORTANT DEBUGGING NOTE (burn 0.21): the `Module` derive macro keys on
//! the generic *being named `B`*. If you name the struct generic anything
//! else (`BB`, `Backend`, etc.), the derive silently emits
//! `num_params = 0` and `visit` is a no-op, which means `from_grads`
//! returns an empty `GradientsParams` and `optim.step` is a no-op.
//! Discovered the hard way — keep the struct generic literally `B`.

use burn::backend::{Autodiff, Cuda};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{gelu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};
use rand::Rng;

// We alias to `Bk` (not `B`) so the struct can keep `B` as its generic
// name — required by burn's Module derive macro.
type Bk = Autodiff<Cuda>;

#[derive(Module, Debug)]
struct Net<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> Net<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(1, 32).init(device),
            l2: LinearConfig::new(32, 32).init(device),
            l3: LinearConfig::new(32, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = gelu(self.l1.forward(x));
        let h = gelu(self.l2.forward(h));
        sigmoid(self.l3.forward(h))
    }
}

fn main() {
    let device: Device<Bk> = Default::default();
    println!("backend: Autodiff<Cuda>  device: {:?}", device);

    let mut net: Net<Bk> = Net::new(&device);
    println!(
        "net.num_params() = {}",
        <Net<Bk> as Module<Bk>>::num_params(&net)
    );

    let mut optim = AdamConfig::new().init::<Bk, Net<Bk>>();
    let lr = 1e-2_f64;

    for step in 0..1000 {
        let mut rng = rand::rng();
        let xs: Vec<f32> = (0..64).map(|_| rng.random::<f32>()).collect();
        let x: Tensor<Bk, 2> =
            Tensor::<Bk, 1>::from_floats(xs.as_slice(), &device).reshape([64, 1]);

        let y = net.forward(x);
        let loss = (y - 1.0).powi_scalar(2).mean();
        let loss_val = loss.clone().into_scalar();
        let grads = loss.backward();
        let gp = GradientsParams::from_grads::<Bk, Net<Bk>>(grads, &net);
        let n_params = gp.len();
        net = optim.step(lr, net, gp);

        if step % 100 == 0 || step == 999 {
            println!("step {step:4}  loss={loss_val:.6e}  n_grad_params={n_params}");
        }
    }
    println!("\nIf loss is <1e-3 after 1000 steps, training plumbing works.");
}
