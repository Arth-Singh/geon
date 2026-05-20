"""
Phase 3b — Honest apples-to-apples comparison: neural f(r_s) under HARD
interior / exterior constraints.

Difference from v1:
  - Phase 3a let the network exploit the r² weighting in the integrand by
    putting an infinitesimally sharp step at r=0 (where r²≈0) and pulling
    f down to ~0.4 across the rest of the domain.  That collapses the bubble.
  - Here we explicitly sample the bubble *interior* and *exterior* regions
    densely and enforce f≈1 inside and f≈0 outside with high-weight
    pointwise penalties.  The wall region [R-δ, R+δ] is where the network
    has freedom to choose the transition shape.
  - We also enforce f(R) = 0.5 via a high-weight pin so the wall location
    matches the canonical bubble.

If the network *still* beats canonical Alcubierre under these constraints,
the result is meaningful.
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


class ShapeNet(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(r.unsqueeze(-1)).squeeze(-1))


def canonical_f(r: torch.Tensor, R: float, sigma: float) -> torch.Tensor:
    return (
        torch.tanh(sigma * (r + R)) - torch.tanh(sigma * (r - R))
    ) / (2.0 * math.tanh(sigma * R))


def exotic_integrand(f_prime: torch.Tensor, r: torch.Tensor, v: float) -> torch.Tensor:
    return (v * v / 12.0) * (f_prime ** 2) * (r ** 2) * 4.0 * math.pi


def trap(integrand: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    dr = r[1:] - r[:-1]
    return (0.5 * (integrand[1:] + integrand[:-1]) * dr).sum()


def train(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(0)}")

    net = ShapeNet(hidden=args.hidden, depth=args.depth).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)

    # Three disjoint sampling regions on r.
    # interior:  r ∈ [0, R - δ]      f must ≈ 1
    # wall:      r ∈ [R - δ, R + δ]  f is free
    # exterior:  r ∈ [R + δ, r_max]  f must ≈ 0
    n = args.n_samples
    r_in = torch.linspace(0.0, args.R - args.delta, n // 3, device=device)
    r_wall = torch.linspace(args.R - args.delta, args.R + args.delta, n // 3, device=device)
    r_out = torch.linspace(args.R + args.delta, args.r_max, n - 2 * (n // 3), device=device)
    r_full = torch.cat([r_in, r_wall, r_out]).requires_grad_(True)

    # Canonical baseline at the same R, σ — recomputed cleanly with autograd.
    r_eval = torch.linspace(0.0, args.r_max, args.n_eval, device=device, requires_grad=True)
    f_canon = canonical_f(r_eval, R=args.R, sigma=args.sigma)
    fp_canon = torch.autograd.grad(f_canon.sum(), r_eval, create_graph=False)[0]
    canon_mass = trap(exotic_integrand(fp_canon, r_eval, v=args.v), r_eval).item()
    print(f"canonical exotic mass (R={args.R}, σ={args.sigma}): {canon_mass:.6e}")

    history = {"loss": [], "exotic": [], "interior": [], "exterior": [], "pin": []}

    for step in range(args.steps):
        opt.zero_grad()

        f = net(r_full)
        f_prime = torch.autograd.grad(f.sum(), r_full, create_graph=True)[0]
        exotic = trap(exotic_integrand(f_prime, r_full, v=args.v), r_full)

        # Region-wise hard constraints. Use mean so they don't scale with N.
        n_in = r_in.shape[0]
        n_out = r_out.shape[0]
        f_interior = f[:n_in]
        f_exterior = f[-n_out:]
        interior_pen = ((f_interior - 1.0) ** 2).mean()
        exterior_pen = (f_exterior ** 2).mean()

        # Pin f(R) = 0.5 hard.
        r_R = torch.tensor([args.R], device=device)
        pin = ((net(r_R).squeeze() - 0.5) ** 2)

        loss = (
            exotic
            + args.lambda_interior * interior_pen
            + args.lambda_exterior * exterior_pen
            + args.lambda_pin * pin
        )
        loss.backward()
        opt.step()

        history["loss"].append(loss.item())
        history["exotic"].append(exotic.item())
        history["interior"].append(interior_pen.item())
        history["exterior"].append(exterior_pen.item())
        history["pin"].append(pin.item())

        if step % max(1, args.steps // 25) == 0 or step == args.steps - 1:
            print(
                f"  step {step:5d}  loss={loss.item():.4e}  "
                f"exotic={exotic.item():.4e}  "
                f"in={interior_pen.item():.2e}  out={exterior_pen.item():.2e}  "
                f"pin={pin.item():.2e}"
            )

    # Final clean evaluation on a dense grid.
    r_dense = torch.linspace(0.0, args.r_max, 1024, device=device, requires_grad=True)
    f_learn = net(r_dense)
    fp_learn = torch.autograd.grad(f_learn.sum(), r_dense, create_graph=False)[0]
    learned_mass = trap(exotic_integrand(fp_learn, r_dense, v=args.v), r_dense).item()

    # Constraint violation report
    with torch.no_grad():
        r_test_in = torch.linspace(0.0, args.R - args.delta, 200, device=device)
        r_test_out = torch.linspace(args.R + args.delta, args.r_max, 200, device=device)
        f_test_in = net(r_test_in).cpu().numpy()
        f_test_out = net(r_test_out).cpu().numpy()
        max_in_dev = float(np.abs(f_test_in - 1.0).max())
        max_out_dev = float(np.abs(f_test_out).max())
        f_R_actual = float(net(torch.tensor([args.R], device=device)).item())

    print(
        "\n=== result ===\n"
        f"  canonical exotic mass : {canon_mass:.6e}\n"
        f"  learned   exotic mass : {learned_mass:.6e}\n"
        f"  ratio (learned/canon) : {learned_mass / canon_mass:.4f}\n"
        f"  max |f - 1| in interior region: {max_in_dev:.3e} "
        f"({'OK' if max_in_dev < 0.05 else 'CONSTRAINT VIOLATED'})\n"
        f"  max |f|   in exterior region : {max_out_dev:.3e} "
        f"({'OK' if max_out_dev < 0.05 else 'CONSTRAINT VIOLATED'})\n"
        f"  f(R={args.R}) actual: {f_R_actual:.4f} (target 0.5)"
    )

    return {
        "r": r_dense.detach().cpu().numpy(),
        "f_canonical": canonical_f(r_dense.detach(), R=args.R, sigma=args.sigma).cpu().numpy(),
        "f_learned": f_learn.detach().cpu().numpy(),
        "canon_mass": canon_mass,
        "learned_mass": learned_mass,
        "history": history,
        "args": args,
        "max_in_dev": max_in_dev,
        "max_out_dev": max_out_dev,
        "f_R_actual": f_R_actual,
    }


def plot_results(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = res["args"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.plot(res["r"], res["f_canonical"], label="canonical (tanh)", lw=2)
    ax1.plot(res["r"], res["f_learned"], label="learned (MLP)", lw=2, ls="--")
    ax1.axvspan(0, args.R - args.delta, alpha=0.1, color="green", label="interior (f≈1)")
    ax1.axvspan(args.R + args.delta, args.r_max, alpha=0.1, color="red", label="exterior (f≈0)")
    ax1.axvline(args.R, color="k", ls=":", lw=0.8, alpha=0.5)
    ax1.set_xlabel(r"$r_s$")
    ax1.set_ylabel(r"$f(r_s)$")
    ax1.set_title("shape function (with constraint regions)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    r = res["r"]
    fp_c = np.gradient(res["f_canonical"], r)
    fp_l = np.gradient(res["f_learned"], r)
    ax2.plot(r, fp_c ** 2 * r ** 2, label="canonical", lw=2)
    ax2.plot(r, fp_l ** 2 * r ** 2, label="learned", lw=2, ls="--")
    ax2.set_xlabel(r"$r_s$")
    ax2.set_ylabel(r"$(f')^2 r^2$")
    ax2.set_title("exotic-mass integrand")
    ax2.grid(alpha=0.3)
    ax2.legend()

    h = res["history"]
    ax3.plot(h["exotic"], label="exotic", lw=1)
    ax3.plot(h["interior"], label="interior pen", lw=1)
    ax3.plot(h["exterior"], label="exterior pen", lw=1)
    ax3.plot(h["pin"], label="pin pen", lw=1)
    ax3.set_yscale("symlog", linthresh=1e-4)
    ax3.set_xlabel("training step")
    ax3.set_title("training history")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    ratio = res["learned_mass"] / res["canon_mass"]
    in_ok = res["max_in_dev"] < 0.05
    out_ok = res["max_out_dev"] < 0.05
    title = (
        f"Phase 3b: hard-constrained neural Alcubierre search   "
        f"learned/canon = {ratio:.3f}   "
        f"interior: {'OK' if in_ok else 'X'}  exterior: {'OK' if out_ok else 'X'}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "warp_search_v2.png", dpi=140)
    print(f"wrote {out_dir / 'warp_search_v2.png'}")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--n_samples", type=int, default=600)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--r_max", type=float, default=8.0)
    p.add_argument("--R", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--delta", type=float, default=0.5, help="wall half-width — match 1/σ-ish for fairness")
    p.add_argument("--v", type=float, default=1.0)
    p.add_argument("--lambda_interior", type=float, default=1e3)
    p.add_argument("--lambda_exterior", type=float, default=1e3)
    p.add_argument("--lambda_pin", type=float, default=1e2)
    p.add_argument("--out", type=str, default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    res = train(args)
    plot_results(res, Path(args.out))
