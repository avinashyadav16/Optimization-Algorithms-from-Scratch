# Optimization Algorithms from Scratch

## Problem Overview

Implement and compare six first-order optimizers on a convex 2D quadratic using Python, NumPy, and Matplotlib. The goal is to visualize convergence speed and stability side by side.

## Test Function

Quadratic objective::

$$f(x, y) = a x^2 + b y^2, \quad \nabla f = [2 a x, 2 b y]$$

Global minimum at $(0, 0)$.

## Optimizer Update Rules

- **SGD:** $\theta_{t+1} = \theta_t - \eta g_t$
- **Momentum:** $v_t = \beta v_{t-1} + (1-\beta) g_t$, $\theta_{t+1} = \theta_t - \eta v_t$
- **Nesterov:** lookahead with previous velocity $v_{t-1}$ for the update
- **AdaGrad:** per-parameter scaling $G_t = \sum_{i\le t} g_i^2$, $\theta_{t+1} = \theta_t - \eta g_t/(\sqrt{G_t}+\varepsilon)$
- **RMSProp:** $G_t = \beta G_{t-1} + (1-\beta) g_t^2$
- **Adam:** momentum + RMSProp with bias correction $\hat m_t, \hat v_t$

## How to Run

```bash
pip install -r requirements.txt
cd src
python runner.py
```

Outputs (saved to results/)

## Findings (current run)

- Adam and Nesterov reach $<1e-4$ loss in the fewest steps and show zero loss increases.
- Momentum tracks closely but may oscillate slightly on steeper axes.
- RMSProp is stable and steady on the ill-conditioned quadratic.
- AdaGrad slows once its accumulated square gradients shrink the step size.
- Vanilla SGD is most sensitive to the learning rate and shows the most oscillations.

## Technologies

- Python 3.x, NumPy, Matplotlib

## Learning Outcomes

- See how adaptive methods trade off speed vs. stability on an ill-conditioned bowl.
- Understand the impact of momentum and per-parameter scaling on convergence.
