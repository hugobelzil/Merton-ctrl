# Minimal local repo for the Merton problem with TD / dTD

A **small, local, CPU-friendly** repo for fixed-policy value evaluation on the
continuous-time Merton consumption-investment problem, following the
**differential-TD (dTD)** method of Settai, Takeishi, Yairi.

It is intentionally designed to be:
- easy to run on a laptop,
- easy to understand,
- close to the **policy-evaluation** spirit of the dTD paper,
- a clean stepping stone before adding a learned actor (or moving to
  multi-asset / finite-horizon variants).

## What this repo does

It implements the **infinite-horizon CRRA Merton problem** with:
- one risky asset,
- one risk-free asset,
- consumption,
- a **constant policy** `(pi, kappa)` where
  - `pi` = risky portfolio weight,
  - `kappa` = consumption rate with `c_t = kappa * W_t`.

It provides:
1. **closed-form optimal policy** and **closed-form exact value** for any
   constant policy,
2. **exact one-step simulation** of wealth under a fixed policy,
3. critic training with
   - standard **TD**,
   - **dTD**,
   - **beta-dTD**,
4. evaluation of the learned critic against the closed-form value.

The critic is an **unstructured MLP** in `log W`. No CRRA homogeneity is baked
into the critic and no exact-A initialization is used, so the training actually
has to learn the value function — making the comparison between TD and dTD
meaningful.

---

## Mathematical setup

Wealth dynamics:

$$
\frac{dW_t}{W_t} = \big(r + \pi_t(\mu-r) - \kappa_t\big) dt + \pi_t \sigma dB_t,
\qquad c_t = \kappa_t W_t.
$$

Objective:

$$
\mathbb E\left[ \int_0^\infty e^{-\rho t} U(c_t)\,dt \right],
\qquad U(c)=\frac{c^{1-\gamma}}{1-\gamma}, \quad \gamma \neq 1.
$$

In this repo we restrict to **constant policies** $\pi_t\equiv\pi$,
$\kappa_t\equiv\kappa$. For any such policy, the exact value has the form

$$
V^{\pi,\kappa}(w)=A(\pi,\kappa)\frac{w^{1-\gamma}}{1-\gamma},
\qquad
A(\pi,\kappa)=
\frac{\kappa^{1-\gamma}}
{\rho-(1-\gamma)\left[r+\pi(\mu-r)-\kappa-\tfrac{1}{2}\gamma\pi^2\sigma^2\right]}.
$$

The closed-form optimum is

$$
\pi^* = \frac{\mu-r}{\gamma\sigma^2},
\qquad
\kappa^* = \frac{\rho-(1-\gamma)\left(r + \frac{1}{2}\frac{(\mu-r)^2}{\gamma\sigma^2}\right)}{\gamma}.
$$

This is why the repo is so light: every learned critic can be compared to the
exact answer.

---

## What each loss is

Let $\Delta W_t = W_{t+\Delta t}-W_t$, $r_t = U(c_t)\,\Delta t$,
$\gamma_{\text{disc}} = e^{-\rho\Delta t}$.

### TD

$$
\delta_{TD} = r_t + \gamma_{\text{disc}}\, V(W_{t+\Delta t}) - V(W_t).
$$

`V_next` is detached (semi-gradient).

### dTD (paper's preferred split)

$$
\delta_{dTD}
=\underbrace{\Delta W_t\,V_w(W_t) + \tfrac{1}{2}(\Delta W_t)^2\,V_{ww}(W_t)}_{\text{prediction}}
\;-\;
\underbrace{\bigl(-r_t + \rho\,\Delta t\, V(W_{t+\Delta t})\bigr)}_{\text{target}}.
$$

Derivatives at $W_t$ carry the gradient; $V(W_{t+\Delta t})$ is detached.
This matches the discrete-compatible form in §5.1 of the paper, with the
identifications $\rho_{\text{paper}}\Delta t = r_t$ and
$-\log\gamma_{\text{disc}} = \rho\,\Delta t$ (the paper's $\gamma$ is the
discount rate, called `rho` in this repo to avoid clashing with CRRA risk
aversion).

### beta-dTD

$$
\mathcal{L}_{\beta\text{-dTD}}
= (1-\beta)\,\mathbb{E}[\delta_{TD}^2]
+ \beta\,\mathbb{E}[\delta_{dTD}^2].
$$

Usually the most stable choice.

---

## Repo structure

```text
Merton-ctrl/
│
├─ README.md
├─ requirements.txt
├─ pyproject.toml
│
├─ paper/
│  └─ Paper.tex / Paper.pdf    # the dTD paper
│
├─ scripts/
│  └─ train_critic.py
│
├─ src/merton_dtd/
│  ├─ __init__.py
│  ├─ config.py                # MertonParams, PolicyParams, TrainConfig
│  ├─ merton.py                # closed forms, exact step, reward rate
│  ├─ critic.py                # VanillaMLPCritic
│  ├─ sampling.py              # log-uniform wealth sampling
│  ├─ losses.py                # TD, dTD, beta-dTD
│  ├─ eval.py                  # error vs closed-form on a grid
│  ├─ training.py              # training loop
│  └─ plotting.py              # training curves + value fit
│
└─ results/
```

---

## Local setup

### Linux / macOS / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If activation is blocked, run once in an Administrator PowerShell:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Running

Train a critic on the closed-form optimal policy with beta-dTD:

```bash
python scripts/train_critic.py --loss beta_dtd --pi 0.75 --kappa 0.06125
```

This writes:
- `results/train_critic/checkpoint.pt`
- `results/train_critic/training_curves.png`
- `results/train_critic/value_fit.png`
- `results/train_critic/summary.json`

The default parameters give $\pi^\star\approx 0.75$ and
$\kappa^\star\approx 0.06125$, so the command above is targeting the optimum.

### Compare TD vs dTD vs beta-dTD

```bash
python scripts/train_critic.py --loss td       --pi 0.75 --kappa 0.06125 --out-dir results/td
python scripts/train_critic.py --loss dtd      --pi 0.75 --kappa 0.06125 --out-dir results/dtd
python scripts/train_critic.py --loss beta_dtd --pi 0.75 --kappa 0.06125 --out-dir results/beta_dtd
```

### Robustness in time step

```bash
python scripts/train_critic.py --loss beta_dtd --dt 0.0833333333 --out-dir results/dt_monthly
python scripts/train_critic.py --loss beta_dtd --dt 0.0192307692 --out-dir results/dt_weekly
python scripts/train_critic.py --loss beta_dtd --dt 0.0039682540 --out-dir results/dt_daily
```

(Approximately monthly $1/12$, weekly $1/52$, daily $1/252$.)

### Different fixed policies

```bash
python scripts/train_critic.py --loss beta_dtd --pi 0.50 --kappa 0.04 --out-dir results/policy_a
python scripts/train_critic.py --loss beta_dtd --pi 1.00 --kappa 0.08 --out-dir results/policy_b
```

Because the exact value is known for every constant policy, you can always
compute exact errors.

---

## Good defaults

The default parameters are:
- `r = 0.02`
- `mu = 0.08`
- `sigma = 0.20`
- `gamma = 2.0` (CRRA risk aversion)
- `rho = 0.08` (discount rate)

These give a sensible closed-form optimum and are stable for local experiments.

---

## Suggested order of work

1. Train with `--loss beta_dtd` on $(\pi^\star, \kappa^\star)$ — sanity check.
2. Compare `td`, `dtd`, `beta_dtd` on the same policy.
3. Vary `--dt` to see how each loss handles coarser/finer time steps.
4. Move to a learned policy (see below).

---

## What to modify next

This repo only does **fixed-policy evaluation**. To actually solve Merton you
need a policy-improvement step on top of the critic. Natural extensions:

### Extension 1: parametric actor on $V_\theta$
Parameterize $(\pi_\phi, \kappa_\phi)$ as two scalars (or wealth-dependent
heads) and update $\phi$ by gradient ascent on $V_\theta(w)$ using the
beta-dTD critic. This is the cheapest way to "close the loop".

### Extension 2: A2C-style actor-critic
Gaussian policy + policy gradient + entropy bonus. Far simpler than PPO and
sufficient for this problem class.

### Extension 3: multi-asset Merton
$\pi\in\mathbb{R}^d$ instead of scalar. The state stays 1D in wealth (CRRA
homogeneity), so the **critic side is unchanged** — what scales is the actor.

### Extension 4: richer state
Finite horizon $(t, W)$, stochastic volatility $(W, v_t)$, or return
predictability $(W, X_t)$. These are the cases where the dTD method's
multi-dimensional Hessian term actually does work.

---

## Important limitation

This is a **starter repo** for fixed-policy evaluation, not a full
control benchmark. It is deliberately small so you can:
- run everything locally,
- understand every file,
- validate the math against a closed form,
- produce first figures quickly.
