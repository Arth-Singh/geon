"""
Phase 3c — testing the reviewer's hypothesis: is the 2.3% improvement real,
or is it the optimizer gaming our fixed-grid trapezoid quadrature?

The reviewer's claim:
  * The two-peak integrand structure and the wigginess of f near the wall
    are signs that the MLP is placing (f')² peaks in regions the trapezoid
    grid under-samples.
  * On a finer evaluation grid (or with adaptive `scipy.integrate.quad`),
    the learned mass should *increase*, possibly above canonical.

Three controls added on top of v2:
  1. **Randomised quadrature grid** every step (sorted uniform random points
     on [0, r_max]). The network can no longer over-fit to a fixed set of
     collocation points.
  2. **Smoothness penalty**: integrate (f'')² dr and add to loss. Canonical
     tanh has near-minimal TV; force the MLP to compete on that axis too.
  3. **Adaptive `scipy.integrate.quad`** for final evaluation, in addition
     to dense trapezoid. Report all three numbers so the user (and the
     reviewer) can see whether the win survives.

Train on the same hyperparameters as v2-tight. If learned/canon stays
meaningfully below 1.0 under randomised grid + smoothness penalty + adaptive
quad evaluation, the result is robust. If it rises to ≥ 1.0 on adaptive
quad, the reviewer was right and 2.3% was quadrature gaming.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------- model ------------------------------------- #


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


def canonical_f_t(r: torch.Tensor, R: float, sigma: float) -> torch.Tensor:
    return (
        torch.tanh(sigma * (r + R)) - torch.tanh(sigma * (r - R))
    ) / (2.0 * math.tanh(sigma * R))


def canonical_f_np(r: np.ndarray, R: float, sigma: float) -> np.ndarray:
    return (
        np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))
    ) / (2.0 * math.tanh(sigma * R))


def canonical_fp_np(r: np.ndarray, R: float, sigma: float) -> np.ndarray:
    return (
        sigma * (1.0 - np.tanh(sigma * (r + R)) ** 2)
        - sigma * (1.0 - np.tanh(sigma * (r - R)) ** 2)
    ) / (2.0 * math.tanh(sigma * R))


# ----------------------------- losses ------------------------------------ #


def integrand_kernel(f_prime: torch.Tensor, r: torch.Tensor, v: float) -> torch.Tensor:
    return (v * v / 12.0) * (f_prime ** 2) * (r ** 2) * 4.0 * math.pi


def trap_sorted(integrand: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    # r is assumed sorted ascending.
    dr = r[1:] - r[:-1]
    return (0.5 * (integrand[1:] + integrand[:-1]) * dr).sum()


# ----------------------------- training ---------------------------------- #


def train(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(0)}")

    net = ShapeNet(hidden=args.hidden, depth=args.depth).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)

    history = {
        "loss": [], "exotic": [], "smooth": [],
        "interior": [], "exterior": [], "pin": [],
    }

    R, delta = args.R, args.delta
    r_R = torch.tensor([R], device=device)

    import time
    t0 = time.time()

    for step in range(args.steps):
        opt.zero_grad()

        # ---- RANDOMISED quadrature grid this step ----
        # interior, wall, exterior buckets get fresh uniform samples each
        # step.  Sorted ascending so we can trapezoid-integrate.
        n_in = args.n_samples // 3
        n_wall = args.n_samples // 3
        n_out = args.n_samples - n_in - n_wall

        # Always include the constraint boundary points exactly so the
        # interior/exterior anchors are evaluated, then add jittered uniform
        # samples inside each region.
        r_in_seed = torch.rand(n_in - 2, device=device) * (R - delta)
        r_in = torch.cat(
            [torch.zeros(1, device=device), r_in_seed,
             torch.full((1,), R - delta, device=device)]
        )
        r_in, _ = torch.sort(r_in)

        r_wall_seed = torch.rand(n_wall, device=device) * (2 * delta) + (R - delta)
        r_wall, _ = torch.sort(r_wall_seed)

        r_out_seed = torch.rand(n_out - 2, device=device) * (args.r_max - (R + delta)) + (R + delta)
        r_out = torch.cat(
            [torch.full((1,), R + delta, device=device), r_out_seed,
             torch.full((1,), args.r_max, device=device)]
        )
        r_out, _ = torch.sort(r_out)

        r_full = torch.cat([r_in, r_wall, r_out]).requires_grad_(True)

        f = net(r_full)
        f_prime = torch.autograd.grad(f.sum(), r_full, create_graph=True)[0]

        # Smoothness: integrate (f'')² over r. Use grad-of-grad.
        f_prime_prime = torch.autograd.grad(f_prime.sum(), r_full, create_graph=True)[0]

        exotic = trap_sorted(integrand_kernel(f_prime, r_full, v=args.v), r_full)
        smooth = trap_sorted(f_prime_prime ** 2, r_full)

        interior_pen = ((f[: r_in.shape[0]] - 1.0) ** 2).mean()
        exterior_pen = (f[-r_out.shape[0]:] ** 2).mean()
        pin = (net(r_R).squeeze() - 0.5) ** 2

        loss = (
            exotic
            + args.lambda_smooth * smooth
            + args.lambda_interior * interior_pen
            + args.lambda_exterior * exterior_pen
            + args.lambda_pin * pin
        )
        loss.backward()
        opt.step()

        history["loss"].append(loss.item())
        history["exotic"].append(exotic.item())
        history["smooth"].append(smooth.item())
        history["interior"].append(interior_pen.item())
        history["exterior"].append(exterior_pen.item())
        history["pin"].append(pin.item())

        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(
                f"  step {step:5d}  loss={loss.item():.4e}  "
                f"exotic={exotic.item():.4e}  smooth={smooth.item():.4e}  "
                f"in={interior_pen.item():.2e}  out={exterior_pen.item():.2e}  "
                f"pin={pin.item():.2e}"
            )

    train_time = time.time() - t0
    print(f"\ntraining took {train_time:.1f}s ({args.steps} steps)")

    # ===== EVALUATION =====
    # The key test: report exotic mass under three different quadrature
    # schemes on the LEARNED shape and the CANONICAL shape.

    R, sigma, v = args.R, args.sigma, args.v

    # (A) The same trapezoid grid the network trained against — same as v2.
    def eval_trapezoid(n: int) -> tuple[float, float]:
        r = torch.linspace(0.0, args.r_max, n, device=device, requires_grad=True)
        f_l = net(r)
        fp_l = torch.autograd.grad(f_l.sum(), r, create_graph=False)[0]
        learned = trap_sorted(integrand_kernel(fp_l, r, v=v), r).item()
        f_c = canonical_f_t(r.detach(), R=R, sigma=sigma)
        r_d = r.detach().clone().requires_grad_(True)
        f_c2 = canonical_f_t(r_d, R=R, sigma=sigma)
        fp_c = torch.autograd.grad(f_c2.sum(), r_d, create_graph=False)[0]
        canon = trap_sorted(integrand_kernel(fp_c, r_d, v=v), r_d).item()
        return learned, canon

    tr1024 = eval_trapezoid(1024)
    tr10240 = eval_trapezoid(10240)

    # (B) scipy.integrate.quad — adaptive, ~14 digits if integrand is smooth.
    # We wrap the network as a numpy callable: cheap because each call is
    # one MLP forward + autograd on a single point.
    def f_prime_at(r_val: float, model: nn.Module) -> float:
        r = torch.tensor([r_val], device=device, requires_grad=True)
        f = model(r)
        fp = torch.autograd.grad(f.sum(), r)[0]
        return float(fp.item())

    def integrand_at(r_val: float, model: nn.Module) -> float:
        fp = f_prime_at(r_val, model)
        return (v * v / 12.0) * (fp ** 2) * (r_val ** 2) * 4.0 * math.pi

    print("\nrunning scipy.integrate.quad on learned shape (adaptive)...")
    learned_quad, learned_err = scipy.integrate.quad(
        integrand_at, 0.0, args.r_max, args=(net,),
        limit=200, epsabs=1e-8, epsrel=1e-8,
    )
    print(f"  learned: quad = {learned_quad:.6e}  (est err {learned_err:.2e})")

    print("running scipy.integrate.quad on canonical shape (adaptive)...")
    canon_quad, canon_err = scipy.integrate.quad(
        lambda r: (v * v / 12.0) * (canonical_fp_np(np.array(r), R, sigma) ** 2) * (r ** 2) * 4.0 * math.pi,
        0.0, args.r_max, limit=200, epsabs=1e-8, epsrel=1e-8,
    )
    print(f"  canon:   quad = {canon_quad:.6e}  (est err {canon_err:.2e})")

    # ===== TV of f' as a proxy for smoothness =====
    with torch.no_grad():
        r = torch.linspace(0.0, args.r_max, 4096, device=device)
        f_l = net(r).cpu().numpy()
        fp_l = np.gradient(f_l, r.cpu().numpy())
        f_c = canonical_f_np(r.cpu().numpy(), R, sigma)
        fp_c = canonical_fp_np(r.cpu().numpy(), R, sigma)
        tv_l = np.abs(np.diff(fp_l)).sum()
        tv_c = np.abs(np.diff(fp_c)).sum()
    print(f"\ntotal variation of f' (4096 pts):")
    print(f"  learned   TV(f') = {tv_l:.6e}")
    print(f"  canonical TV(f') = {tv_c:.6e}")

    # ===== final report =====
    print("\n" + "=" * 60)
    print("REPORT — verdict on the reviewer's hypothesis")
    print("=" * 60)
    print(f"  exotic mass, trapezoid n=1024 :  learned={tr1024[0]:.6e}  canon={tr1024[1]:.6e}  ratio={tr1024[0]/tr1024[1]:.4f}")
    print(f"  exotic mass, trapezoid n=10240:  learned={tr10240[0]:.6e}  canon={tr10240[1]:.6e}  ratio={tr10240[0]/tr10240[1]:.4f}")
    print(f"  exotic mass, scipy.quad adaptive: learned={learned_quad:.6e}  canon={canon_quad:.6e}  ratio={learned_quad/canon_quad:.4f}")
    print()
    if learned_quad / canon_quad > 1.0:
        verdict = "REVIEWER WAS RIGHT: trapezoid showed a win that adaptive quad refutes — it was quadrature gaming."
    elif learned_quad / canon_quad < 0.99:
        verdict = f"WIN IS REAL: under adaptive quadrature the learned shape beats canon by {(1.0 - learned_quad/canon_quad)*100:.2f}%."
    else:
        verdict = "WIN COLLAPSED TO NOISE: under adaptive quadrature the two are within 1% — canonical is essentially optimal in this ansatz."
    print(f"  VERDICT: {verdict}")

    return {
        "r": r.cpu().numpy() if isinstance(r, torch.Tensor) else r,
        "f_canonical": f_c, "f_learned": f_l,
        "fp_canonical": fp_c, "fp_learned": fp_l,
        "history": history,
        "tr1024": tr1024, "tr10240": tr10240,
        "quad": (learned_quad, canon_quad, learned_err, canon_err),
        "tv": (tv_l, tv_c),
        "args": args,
        "train_time": train_time,
        "verdict": verdict,
    }


# ----------------------------- plotting ---------------------------------- #


def plot_results(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = res["args"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(res["r"], res["f_canonical"], label="canonical tanh", lw=2)
    ax.plot(res["r"], res["f_learned"], label="learned MLP", lw=2, ls="--")
    ax.axvspan(0, args.R - args.delta, alpha=0.1, color="green", label="interior")
    ax.axvspan(args.R + args.delta, args.r_max, alpha=0.1, color="red", label="exterior")
    ax.set_xlabel(r"$r_s$"); ax.set_ylabel(r"$f(r_s)$")
    ax.set_title("shape function")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(res["r"], (res["fp_canonical"]) ** 2 * res["r"] ** 2, label="canonical", lw=2)
    ax.plot(res["r"], (res["fp_learned"]) ** 2 * res["r"] ** 2, label="learned", lw=2, ls="--")
    ax.set_xlabel(r"$r_s$"); ax.set_ylabel(r"$(f')^2 r^2$")
    ax.set_title("exotic-mass integrand (4096-pt fine grid)")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1, 0]
    h = res["history"]
    ax.plot(h["exotic"], label="exotic", lw=1)
    ax.plot(h["smooth"], label="smooth (×λ)", lw=1)
    ax.plot(h["interior"], label="interior", lw=1)
    ax.plot(h["exterior"], label="exterior", lw=1)
    ax.plot(h["pin"], label="pin", lw=1)
    ax.set_yscale("symlog", linthresh=1e-5); ax.set_xlabel("step")
    ax.set_title("training history (random grid each step)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    ll, lc = res["tr1024"]
    lll, lcc = res["tr10240"]
    lq, cq, le, ce = res["quad"]
    tvl, tvc = res["tv"]
    text = (
        f"VERDICT\n{'=' * 56}\n\n"
        f"  trapezoid n=1024:   learned/canon = {ll / lc:.4f}\n"
        f"                       ({ll:.4e} / {lc:.4e})\n\n"
        f"  trapezoid n=10240:  learned/canon = {lll / lcc:.4f}\n"
        f"                       ({lll:.4e} / {lcc:.4e})\n\n"
        f"  scipy.quad adaptive: learned/canon = {lq / cq:.4f}\n"
        f"                       ({lq:.4e} / {cq:.4e})\n"
        f"  abs error estimates: l={le:.1e}  c={ce:.1e}\n\n"
        f"  TV(f') learned  = {tvl:.4e}\n"
        f"  TV(f') canonical = {tvc:.4e}\n"
        f"  TV ratio l/c    = {tvl / tvc:.3f}\n\n"
        f"{res['verdict']}"
    )
    ax.text(0.0, 1.0, text, family="monospace", fontsize=9, va="top")

    fig.suptitle(
        f"Phase 3c: random quadrature + smoothness penalty + scipy.quad eval  "
        f"(train time {res['train_time']:.0f}s)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "warp_search_v3.png", dpi=140)
    print(f"wrote {out_dir / 'warp_search_v3.png'}")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--n_samples", type=int, default=600)
    p.add_argument("--r_max", type=float, default=8.0)
    p.add_argument("--R", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--delta", type=float, default=0.4)
    p.add_argument("--v", type=float, default=1.0)
    p.add_argument("--lambda_interior", type=float, default=1e5)
    p.add_argument("--lambda_exterior", type=float, default=1e5)
    p.add_argument("--lambda_pin", type=float, default=1e4)
    p.add_argument("--lambda_smooth", type=float, default=1e-3,
                   help="penalty on integrated (f'')^2. Set 0 to disable.")
    p.add_argument("--out", type=str, default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    res = train(parse())
    plot_results(res, Path(parse().out))
