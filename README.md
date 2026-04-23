# Minimal local repo for the Merton problem with TD / dTD

This is a **small, local, CPU-friendly starter repo** for the continuous-time Merton consumption-investment problem.

It is intentionally designed to be:
- easy to run on a laptop,
- easy to understand,
- close to the **policy-evaluation** spirit of the differential-TD paper,
- a good first stepping stone before you move to a larger actor-critic or PPO/SAC/DDPG benchmark.

## What this repo does

It implements the **infinite-horizon CRRA Merton problem** with:
- one risky asset,
- one risk-free asset,
- consumption,
- a **constant policy** `(pi, kappa)` where
  - `pi` = risky portfolio weight,
  - `kappa` = consumption rate with `c_t = kappa * W_t`.

It includes:
1. **closed-form optimal policy**,
2. **closed-form exact value** for any constant policy,
3. **exact one-step simulation** of wealth under a fixed policy,
4. critic training with
   - standard **TD**,
   - **dTD**,
   - **beta-dTD**,
5. a **policy grid sweep** to compare the closed-form optimum to a cheap numerical search.

---

## Why this is a good first repo

The differential-TD paper is really a **fixed-policy value evaluation** method first.
So the cleanest Merton starting point is:
- pick a policy,
- evaluate its value function,
- compare TD vs dTD vs beta-dTD,
- then do a very cheap policy search.

This repo does exactly that.

It is **not** yet the final ambitious project setup with PPO/SAC/DDPG baselines.
It is the local starter version you can actually understand and run fast.

---

## Mathematical setup

We use the infinite-horizon Merton problem

$$
\frac{dW_t}{W_t} = \big(r + \pi_t(\mu-r) - \kappa_t\big) dt + \pi_t \sigma dB_t,
\qquad c_t = \kappa_t W_t.
$$

Objective:

$$
\mathbb E\left[ \int_0^\infty e^{-\rho t} U(c_t)\,dt \right],
\qquad U(c)=\frac{c^{1-\gamma}}{1-\gamma}, \quad \gamma \neq 1.
$$

In this starter repo we restrict to **constant policies**:

$$
\pi_t \equiv \pi, \qquad \kappa_t \equiv \kappa.
$$

For any such constant policy, the exact value has the form

$$
V^{\pi,\kappa}(w)=A(\pi,\kappa)\frac{w^{1-\gamma}}{1-\gamma},
$$

with

$$
A(\pi,\kappa)=
\frac{\kappa^{1-\gamma}}
{\rho-(1-\gamma)\left[r+\pi(\mu-r)-\kappa-\frac{1}{2}\gamma\pi^2\sigma^2\right]}.
$$

The closed-form optimal policy is

$$
\pi^* = \frac{\mu-r}{\gamma\sigma^2},
\qquad
\kappa^* = \frac{\rho-(1-\gamma)\left(r + \frac{1}{2}\frac{(\mu-r)^2}{\gamma\sigma^2}\right)}{\gamma}.
$$

This is why the repo is so light: you can always compare your learned critic to the exact answer.

---

## Repo structure

```text
merton_dtd_local_repo/
│
├─ README.md
├─ requirements.txt
├─ pyproject.toml
│
├─ scripts/
│  ├─ check_closed_form.py
│  ├─ train_critic.py
│  └─ sweep_policies.py
│
├─ src/merton_dtd/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ merton.py
│  ├─ critic.py
│  ├─ sampling.py
│  ├─ losses.py
│  ├─ eval.py
│  ├─ training.py
│  ├─ plotting.py
│  └─ sweep.py
│
└─ results/
```

---

## Local setup on Windows

### Option A: plain Windows + VS Code + PowerShell

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PowerShell blocks activation, run this once in a PowerShell opened as Administrator:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Option B: WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## First commands to run

### 1) Check the closed-form optimum

```bash
python scripts/check_closed_form.py
```

You should see something like:
- `pi* ≈ 0.75`
- `kappa* ≈ 0.06125`
with the default parameters.

### 2) Sweep a grid of constant policies

```bash
python scripts/sweep_policies.py
```

This creates:
- `results/policy_sweep/value_heatmap.png`
- `results/policy_sweep/summary.json`

This is the cheapest possible control benchmark.

### 3) Train a critic with beta-dTD on the optimal policy

```bash
python scripts/train_critic.py --loss beta_dtd --critic scalar --pi 0.75 --kappa 0.06125
```

This creates:
- `results/train_critic/checkpoint.pt`
- `results/train_critic/training_curves.png`
- `results/train_critic/value_fit.png`
- `results/train_critic/summary.json`

---

## Recommended experiments

### Easiest first experiment

Run the **structured scalar critic** on the closed-form optimal policy:

```bash
python scripts/train_critic.py --loss td       --critic scalar --pi 0.75 --kappa 0.06125 --out-dir results/td_scalar
python scripts/train_critic.py --loss dtd      --critic scalar --pi 0.75 --kappa 0.06125 --out-dir results/dtd_scalar
python scripts/train_critic.py --loss beta_dtd --critic scalar --pi 0.75 --kappa 0.06125 --out-dir results/betadtd_scalar
```

### Slightly harder experiment

Switch to the MLP critic:

```bash
python scripts/train_critic.py --loss td       --critic mlp --pi 0.75 --kappa 0.06125 --out-dir results/td_mlp
python scripts/train_critic.py --loss dtd      --critic mlp --pi 0.75 --kappa 0.06125 --out-dir results/dtd_mlp
python scripts/train_critic.py --loss beta_dtd --critic mlp --pi 0.75 --kappa 0.06125 --out-dir results/betadtd_mlp
```

### Robustness in time step

```bash
python scripts/train_critic.py --loss beta_dtd --critic mlp --dt 0.0833333333 --out-dir results/dt_monthly
python scripts/train_critic.py --loss beta_dtd --critic mlp --dt 0.0192307692 --out-dir results/dt_weekly
python scripts/train_critic.py --loss beta_dtd --critic mlp --dt 0.0039682540 --out-dir results/dt_daily
```

Those correspond approximately to:
- monthly: `1/12`
- weekly: `1/52`
- daily: `1/252`

### Different fixed policies

For example:

```bash
python scripts/train_critic.py --loss beta_dtd --critic mlp --pi 0.50 --kappa 0.04 --out-dir results/policy_a
python scripts/train_critic.py --loss beta_dtd --critic mlp --pi 1.00 --kappa 0.08 --out-dir results/policy_b
```

Because the exact value is known for every constant policy, you can always compute exact errors.

---

## What each loss means

### TD

The one-step TD residual is

$$
\delta_{TD} = U(c_t)\Delta t + e^{-\rho \Delta t}V(W_{t+\Delta t}) - V(W_t).
$$

### dTD

The differential residual is

$$
\delta_{dTD} = U(c_t)
+ \frac{\Delta W_t}{\Delta t}V_w(W_t)
+ \frac{1}{2}\frac{(\Delta W_t)^2}{\Delta t}V_{ww}(W_t)
- \rho V(W_t).
$$

### beta-dTD

A weighted hybrid:

$$
(1-\beta)\,\delta_{TD}^2 + \beta\,(\Delta t\,\delta_{dTD})^2.
$$

This is usually the most stable place to start.

---

## What to modify next

Once this repo works on your machine, the natural extensions are:

### Extension 1: finite horizon
Add time as an input and switch from infinite-horizon to a terminal-utility version. Then your state becomes `(t, W_t)`.

### Extension 2: policy class beyond constants
Let the policy depend on wealth and time, for example:
- `pi_theta(t, w)`
- `kappa_theta(t, w)`

### Extension 3: real actor-critic
Instead of sweeping over a grid of constant policies, learn the policy with an actor update.

### Extension 4: compare against standard RL baselines
Only after the above steps work should you try to compare against PPO/SAC/DDPG.

---

## Why the scalar critic exists

The scalar critic is not a trick. It uses the exact homogeneity of the infinite-horizon CRRA Merton problem. It is the best way to:
- sanity-check your pipeline,
- debug TD vs dTD,
- verify local reproducibility quickly.

The MLP critic is there so you can move toward a more ML-style setup.

---

## Good defaults

The default parameters in the code are:
- `r = 0.02`
- `mu = 0.08`
- `sigma = 0.20`
- `gamma = 2.0`
- `rho = 0.08`

These give a sensible closed-form optimum and are stable for local experiments.

---

## Suggested order of work

1. Run `check_closed_form.py`
2. Run `sweep_policies.py`
3. Run `train_critic.py` with `scalar + beta_dtd`
4. Compare `td`, `dtd`, `beta_dtd`
5. Switch to `mlp`
6. Vary `dt`
7. Only then think about a learned actor or finite horizon

---

## Important limitation

This repo is a **starter repo**, not the final semester-project codebase.

It is deliberately small so you can:
- run everything locally,
- understand every file,
- validate the math,
- produce your first figures quickly.

That is the point.
