"""Run optimizer comparisons on a convex quadratic and plot results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from optimizers import AdaGrad, Adam, Momentum, Nesterov, RMSProp, SGD
from utils import grad_fn, loss_fn


def ensure_results_dir() -> Path:
    """Create results directory relative to project root if missing."""
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    return results


def build_optimizers() -> Dict[str, object]:
    # Learning rates tuned lightly for the ill-conditioned quadratic (a=1, b=10)
    return {
        "SGD": SGD(lr=0.05),
        "Momentum": Momentum(lr=0.05),
        "Nesterov": Nesterov(lr=0.05),
        "AdaGrad": AdaGrad(lr=0.4),
        "RMSProp": RMSProp(lr=0.05),
        "Adam": Adam(lr=0.05),
    }


def run_convergence(
    optimizers: Dict[str, object],
    iterations: int = 100,
    init_params: np.ndarray | None = None,
    tol: float = 1e-4,
) -> Tuple[Dict[str, List[float]], Dict[str, dict]]:
    init = np.array([5.0, 5.0]) if init_params is None else init_params
    losses: Dict[str, List[float]] = {}
    stats: Dict[str, dict] = {}

    for name, optimizer in optimizers.items():
        params = init.copy()
        history: List[float] = []
        increases = 0
        first_below_tol = None

        for step in range(iterations):
            grads = grad_fn(params)
            params = optimizer.step(params, grads)
            current_loss = float(loss_fn(params))
            history.append(current_loss)

            if step > 0 and current_loss > history[-2]:
                increases += 1  # count small oscillations/instability

            if first_below_tol is None and current_loss < tol:
                first_below_tol = step + 1  # 1-based for readability

        losses[name] = history
        stats[name] = {
            "final_loss": history[-1],
            "steps_to_tol": first_below_tol,
            "loss_increases": increases,
        }

    return losses, stats


def plot_convergence(losses: Dict[str, List[float]], save_dir: Path) -> Path:
    plt.figure(figsize=(10, 6))
    for name, history in losses.items():
        plt.plot(history, label=name)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Optimizer Convergence on Quadratic")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = save_dir / "convergence_plot.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_paths(optimizers: Dict[str, object], save_dir: Path) -> Path:
    x = np.linspace(-6, 6, 400)
    y = np.linspace(-6, 6, 400)
    X, Y = np.meshgrid(x, y)
    Z = 1 * X**2 + 10 * Y**2

    colors = {
        "SGD": "red",
        "Momentum": "blue",
        "Nesterov": "green",
        "AdaGrad": "orange",
        "RMSProp": "purple",
        "Adam": "black",
    }

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=40, cmap="viridis")

    iterations = 50
    init_params = np.array([5.0, 5.0])

    for name, opt in optimizers.items():
        params = init_params.copy()
        path = [params.copy()]
        for _ in range(iterations):
            grads = grad_fn(params)
            params = opt.step(params, grads)
            path.append(params.copy())

        path_arr = np.array(path)
        plt.plot(
            path_arr[:, 0],
            path_arr[:, 1],
            marker="o",
            markersize=3,
            color=colors[name],
            label=name,
            linewidth=2,
        )

    plt.scatter(0, 0, color="gold", s=120, label="Global Minimum")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Optimizer Paths on Convex Loss Surface")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = save_dir / "contour_paths.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def format_stats(stats: Dict[str, dict]) -> str:
    lines = ["Optimizer | Final Loss | Steps<1e-4 | Loss Increases"]
    lines.append("-" * len(lines[0]))
    for name, s in stats.items():
        lines.append(
            f"{name:9s} | {s['final_loss']:.4e} | "
            f"{s['steps_to_tol'] if s['steps_to_tol'] else 'n/a':>9} | "
            f"{s['loss_increases']:>13}"
        )
    return "\n".join(lines)


def main() -> None:
    results_dir = ensure_results_dir()
    optimizers = build_optimizers()

    losses, stats = run_convergence(optimizers)
    conv_path = plot_convergence(losses, results_dir)
    contour_path = plot_paths(optimizers, results_dir)

    summary = format_stats(stats)
    print(summary)

    # Persist the textual summary for quick reference next to plots
    (results_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"Saved convergence plot to {conv_path}")
    print(f"Saved contour plot to {contour_path}")


if __name__ == "__main__":
    main()
