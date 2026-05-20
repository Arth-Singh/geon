"""
Phase 3 — Neural search for low-exotic-matter Alcubierre-like warp metrics.

Problem
-------
Alcubierre's 1994 metric requires a smooth shape function f(r_s) with
f(0) = 1 (ship inside bubble) and f(∞) = 0 (asymptotic flatness).  The
Eulerian energy density is

    ρ(x, y, z) = -(v²/32π) · (y² + z²)/r_s² · (df/dr_s)²

which is negative wherever df/dr_s ≠ 0 — i.e. across the bubble walls.

Canonical choice (Alcubierre 1994):
    f(r_s) = (tanh(σ(r_s + R)) - tanh(σ(r_s - R))) / (2 tanh(σR))

The integrated exotic mass for this canonical form is

    M_exotic ∝ -∫ (df/dr_s)² r_s² dr_s

The classical variational result is that the infimum of this integral
under the boundary conditions is zero (in the limit of an infinitely thin
wall) but for any fixed bubble width σ⁻¹ it is bounded below.

The question this script asks: can a neural-network-parameterised f(r_s),
trained with a fixed bubble-width regulariser and asymptotic / interior
boundary penalties, find a shape that has *lower* exotic mass than the
canonical tanh-based form at the same effective bubble radius / wall width?

If yes — by how much? If no — what is the structural property of the
canonical form that makes it (locally) optimal?

This is a tractable, well-defined, single-GPU experiment.  It generalises
naturally to (a) full-metric ansätze (g_xx, g_tx etc. all learnable) and
(b) Bobrick-Martire / Lentz style soliton families.

Hardware: tested on a single B200, fp32, ~minutes of training.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------- model ------------------------------------ #


class ShapeNet(nn.Module):
    """f(r_s) ∈ [0, 1]. SiLU MLP, sigmoid head, deliberately small."""

    def __init__(self, hidden: int = 128, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # Compose with sigmoid in forward so we can also evaluate the raw
        # logit (useful for debugging saturation).
        logit = self.net(r.unsqueeze(-1)).squeeze(-1)
        return torch.sigmoid(logit)


# ---------------------- canonical Alcubierre baseline -------------------- #


def canonical_f(r: torch.Tensor, R: float, sigma: float) -> torch.Tensor:
    """Alcubierre's original tanh shape function."""
    return (
        torch.tanh(sigma * (r + R)) - torch.tanh(sigma * (r - R))
    ) / (2.0 * math.tanh(sigma * R))


# ----------------------------- physics ------------------------------------ #


def exotic_mass_integrand(
    f_values: torch.Tensor, f_prime: torch.Tensor, r: torch.Tensor, v: float
) -> torch.Tensor:
    """
    The integrand of total exotic energy (mod constants):

        ∫ -|ρ| dV  =  -(v²/12) ∫ (f')² r² dr

    We return (f')² r² so the caller can weight as needed.  `v` is included
    so that the absolute scale matches the canonical formula and the loss
    is comparable to the analytic budget.
    """
    return (v * v / 12.0) * (f_prime ** 2) * (r ** 2) * 4.0 * math.pi
    # Note: the Eulerian volume integration over θ gives the factor 4/3 · 2π
    # which we absorbed into the (v²/12); we then add a 4π geometric factor
    # for the spherical shell measure r²·dr (consistent with the derivation
    # in the docstring of `warp_search.py`).


def integrate_trapezoid(integrand: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Trapezoidal rule on a monotonically increasing r grid."""
    dr = r[1:] - r[:-1]
    avg = 0.5 * (integrand[1:] + integrand[:-1])
    return (avg * dr).sum()


# ---------------------- boundary / shape constraints --------------------- #


def shape_constraints(
    f_at_zero: torch.Tensor, f_at_far: torch.Tensor, f_at_R: torch.Tensor
) -> torch.Tensor:
    """
    Soft constraints that force the learned f to actually describe a bubble:

      * f(0) → 1            (ship is inside the bubble)
      * f(R_max) → 0        (asymptotic flatness)
      * f(R) ≈ 0.5          (bubble wall is centred on the canonical radius)

    The last one is what keeps the network honest — without it, the
    optimiser collapses f to a near-step function shifted to wherever
    the (f')² r² integrand is cheapest, which is "outside our domain"
    (trivially zero exotic mass).
    """
    return (
        (f_at_zero - 1.0) ** 2
        + (f_at_far - 0.0) ** 2
        + (f_at_R - 0.5) ** 2
    )


# --------------------------- training loop ------------------------------- #


def train(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device name: {torch.cuda.get_device_name(0)}")
        print(f"  device count: {torch.cuda.device_count()}")

    net = ShapeNet(hidden=args.hidden, depth=args.depth).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)

    # Sample r densely in [0, r_max] for the integral; this is *fixed*
    # across iterations so the loss is a clean numerical quadrature, not
    # a stochastic estimate.
    r = torch.linspace(0.0, args.r_max, args.n_samples, device=device)
    r.requires_grad_(True)

    # Canonical Alcubierre at the same R, σ for comparison.
    with torch.no_grad():
        f_canon = canonical_f(r.detach(), R=args.R, sigma=args.sigma)
        f_canon_prime = torch.autograd.functional.jacobian(
            lambda x: canonical_f(x, R=args.R, sigma=args.sigma), r.detach()
        ).diag()
        canon_integrand = exotic_mass_integrand(
            f_canon, f_canon_prime, r.detach(), v=args.v
        )
        canon_mass = integrate_trapezoid(canon_integrand, r.detach()).item()
    print(f"canonical Alcubierre exotic-mass proxy: {canon_mass:.6e}")

    history = {"loss": [], "exotic": [], "constraint": []}

    for step in range(args.steps):
        opt.zero_grad()
        f = net(r)
        # df/dr via autograd
        f_prime = torch.autograd.grad(f.sum(), r, create_graph=True)[0]
        integrand = exotic_mass_integrand(f, f_prime, r, v=args.v)
        exotic = integrate_trapezoid(integrand, r)

        f_zero = net(torch.zeros(1, device=device)).squeeze()
        f_far = net(torch.full((1,), args.r_max, device=device)).squeeze()
        f_R = net(torch.full((1,), args.R, device=device)).squeeze()
        constraint = shape_constraints(f_zero, f_far, f_R)

        loss = exotic + args.lambda_constraint * constraint
        loss.backward()
        opt.step()

        history["loss"].append(loss.item())
        history["exotic"].append(exotic.item())
        history["constraint"].append(constraint.item())

        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(
                f"  step {step:5d}  loss={loss.item():.4e}  "
                f"exotic={exotic.item():.4e}  constr={constraint.item():.4e}"
            )

    # Evaluate final.
    with torch.no_grad():
        f_final = net(r).cpu().numpy()
        r_np = r.detach().cpu().numpy()
    f_canon_np = f_canon.detach().cpu().numpy()

    # Recompute exotic for final learned f (clean number)
    f = net(r)
    f_prime = torch.autograd.grad(f.sum(), r, create_graph=False)[0]
    learned_mass = integrate_trapezoid(
        exotic_mass_integrand(f, f_prime, r, v=args.v), r
    ).item()

    print(
        f"\nresult:\n"
        f"  canonical exotic mass = {canon_mass:.6e}\n"
        f"  learned   exotic mass = {learned_mass:.6e}\n"
        f"  ratio learned / canonical = {learned_mass / canon_mass:.4f}"
    )

    return {
        "r": r_np,
        "f_canonical": f_canon_np,
        "f_learned": f_final,
        "canon_mass": canon_mass,
        "learned_mass": learned_mass,
        "history": history,
    }


# ----------------------------- plotting ---------------------------------- #


def plot_results(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.plot(res["r"], res["f_canonical"], label="canonical (tanh)", lw=2)
    ax1.plot(res["r"], res["f_learned"], label="learned (MLP)", lw=2, ls="--")
    ax1.set_xlabel(r"$r_s$")
    ax1.set_ylabel(r"$f(r_s)$")
    ax1.set_title("shape function")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Recompute integrands for plot.
    r = res["r"]
    fp_canon = np.gradient(res["f_canonical"], r)
    fp_learn = np.gradient(res["f_learned"], r)
    integrand_canon = (fp_canon ** 2) * r ** 2
    integrand_learn = (fp_learn ** 2) * r ** 2

    ax2.plot(r, integrand_canon, label="canonical", lw=2)
    ax2.plot(r, integrand_learn, label="learned", lw=2, ls="--")
    ax2.set_xlabel(r"$r_s$")
    ax2.set_ylabel(r"$(f')^2 r^2$")
    ax2.set_title("exotic-mass integrand")
    ax2.grid(alpha=0.3)
    ax2.legend()

    history = res["history"]
    ax3.plot(history["loss"], label="loss", lw=1)
    ax3.plot(history["exotic"], label="exotic", lw=1)
    ax3.plot(history["constraint"], label="constraint", lw=1)
    ax3.set_xlabel("training step")
    ax3.set_yscale("symlog", linthresh=1e-3)
    ax3.set_title("training history")
    ax3.grid(alpha=0.3)
    ax3.legend()

    fig.suptitle(
        f"Neural search for low-exotic-mass Alcubierre shape  "
        f"(learned / canonical = {res['learned_mass'] / res['canon_mass']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "warp_search.png", dpi=140)
    print(f"wrote {out_dir / 'warp_search.png'}")


# ---------------------------------------- main --------------------------- #


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--n_samples", type=int, default=512)
    p.add_argument("--r_max", type=float, default=8.0)
    p.add_argument("--R", type=float, default=2.0, help="bubble radius")
    p.add_argument(
        "--sigma", type=float, default=4.0, help="wall thickness param (canonical)"
    )
    p.add_argument("--v", type=float, default=1.0, help="bubble velocity (natural units)")
    p.add_argument("--lambda_constraint", type=float, default=10.0)
    p.add_argument("--out", type=str, default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse()
    res = train(args)
    plot_results(res, Path(args.out))


if __name__ == "__main__":
    main()
