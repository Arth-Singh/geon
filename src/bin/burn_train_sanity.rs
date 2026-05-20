//! Minimal training sanity check: can we drive a small MLP `f(x) -> [0,1]`
//! to match the constant target f(x) = 1 over x ∈ [0, 1]? If after 1000
//! Adam steps the MSE is < 1e-2, the training plumbing works. If not, we
//! have a real burn/autograd issue independent of physics.

use burn::backend::{Autodiff, Cuda};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{gelu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};
use rand::Rng;

type B = Autodiff<Cuda>;

#[derive(Module, Debug, Clone)]
struct Net<BB: Backend> {
    l1: Linear<BB>,
    l2: Linear<BB>,
    l3: Linear<BB>,
}

impl<BB: Backend> Net<BB> {
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
    let device: Device<B> = Default::default();
    println!("backend: Autodiff<Cuda>  device: {:?}", device);

    let mut net: Net<B> = Net::new(&device);
    let mut optim = AdamConfig::new().init::<B, Net<B>>();
    let lr = 1.0_f64;  // intentionally huge — if loss doesn't move, step is a no-op

    for step in 0..1000 {
        let mut rng = rand::rng();
        let xs: Vec<f32> = (0..64).map(|_| rng.random::<f32>()).collect();
        let x: Tensor<B, 2> = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).reshape([64, 1]);

        let y = net.forward(x);
        let loss = (y - 1.0).powi_scalar(2).mean();
        let loss_val = loss.clone().into_scalar();
        let grads = loss.backward();
        let gp = GradientsParams::from_grads::<B, Net<B>>(grads, &net);
        let n_params = gp.len();
        net = optim.step(lr, net, gp);

        if step % 100 == 0 || step == 999 {
            println!("step {step:4}  loss={:.6e}  n_grad_params={n_params}", loss_val);
        }
    }
    println!("\nIf loss is <1e-3 after 1000 steps, training plumbing works.");
}
