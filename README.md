# Modeling and Control of a Planar Quadrotor

**Author:** Konstantinos Spinos

Modeling, simulation and control of a planar quadrotor with aerodynamic drag, using the **Linear Quadratic Regulator (LQR)** and **Convex Model Predictive Control (MPC)** formulated as a **Quadratic Program (QP)** with input constraints.

The full write-up is in [`Modeling_and_Control_of_A_Planar_Quadrotor.pdf`](./Modeling_and_Control_of_A_Planar_Quadrotor.pdf).

## Highlights

- Nonlinear 6-state planar quadrotor model with quadratic aerodynamic drag, expressed in the body-velocity frame.
- RK4 discretization, with automatic linearization (`A`, `B`, affine term `c`) via [JAX](https://github.com/google/jax) (`jacfwd`).
- Infinite-horizon LQR through Riccati recursion, which provides the terminal cost `P_inf` for the MPC.
- MPC as a QP solved with [ProxSuite](https://github.com/Simple-Robotics/proxsuite), with thrust/torque limits.
- Robust to large initial offsets: MPC converges from **200+ m** away, while LQR with saturated inputs stops converging at about 70 m.
- Tests with Gaussian process noise and tracking of a 2D circular trajectory.

## System model

State `x = [x, y, θ, ẋ, ẏ, θ̇]ᵀ ∈ ℝ⁶`, input `u = [T, τ]ᵀ ∈ ℝ²` (thrust and torque).

```
         [ ẋ                                         ]
         [ ẏ                                         ]
f(x,u) = [ $\dot{θ}$                                   ]
         [ (1/m)(−sinθ (T + f_y) + cosθ f_x)         ]
         [ (1/m)( cosθ (T + f_y) + sinθ f_x) − g     ]
         [ (τ − f_θ) / J                             ]
```

Drag terms (in the local frame):

| Term | Expression | Coefficient |
|------|-----------|-------------|
| `f_x` | `−k_x v_x │v_x│` | `k_x = 0.3` |
| `f_y` | `−k_y v_y │v_y│` | `k_y = 0.3` |
| `f_θ` | `−k_θ θ̇` | `k_θ = 0.02` |

Parameters: `m = 1 kg`, `g = 9.81 m/s²`, `J = 0.05`, `dt = 0.05 s`.
Local velocities: `v = Rᵀ [ẋ, ẏ]ᵀ`, where `R` is the 2D rotation matrix by `θ`.

Input limits used in the constrained experiments: `T ∈ [0.5mg, 1.5mg]`, `τ ∈ [−0.5, 0.5]`.

## Repository structure

| Notebook | Contents |
|----------|----------|
| [`Question1.ipynb`](./Question1.ipynb) | Dynamics, system visualization, RK4 integration, open-loop hover simulation (position/velocity plots). |
| [`Question2.ipynb`](./Question2.ipynb) | Linearization with JAX and finite-horizon Riccati recursion for LQR, plus closed-loop simulation with input clipping from several initial states. |
| [`Question3.ipynb`](./Question3.ipynb) | MPC as a QP (ProxSuite) with `P_inf` terminal cost and input bounds. Covers far-away initial states, arbitrary reference points and noise. |
| [`Trajectory.ipynb`](./Trajectory.ipynb) | MPC tracking of a circular reference trajectory (radius 20 m, period 60 s). |

## Method summary

### 1. Discretization (RK4)
```
f1 = f(x, u)            f2 = f(x + f1·dt/2, u)
f3 = f(x + f2·dt/2, u)  f4 = f(x + f3·dt, u)
x_next = x + dt/6 · (f1 + 2 f2 + 2 f3 + f4)
```

### 2. Linearization
A first-order Taylor expansion of the discretized dynamics around `(x̄, ū)`:

```
x_{k+1} ≈ f_d(x̄, ū) + A (x_k − x̄) + B (u_k − ū),    A = ∂f_d/∂x,  B = ∂f_d/∂u
```

### 3. LQR
The Riccati recursion is run backward for `K ≈ 200` steps, and the first `P`, `K` are taken as the infinite-horizon solution:

```
K_k = (R + Bᵀ P_{k+1} B)⁻¹ Bᵀ P_{k+1} A
P_k = Q + Aᵀ P_{k+1} (A − B K_k)
Δu  = −K_inf Δx
```

### 4. MPC as a QP
Over a horizon of `H` steps, the stacked decision vector `z = [u₀, x₁, u₁, x₂, …]` minimizes

```
J = zᵀ H z + 2 hᵀ z      s.t.   G z = d,   lb ≤ C z ≤ ub
```

with the LQR `P_inf` as terminal cost, which gives stability and convergence from long distances. The default horizon is `H = 10` steps (0.5 s look-ahead at `dt = 0.05`).

## Results

### LQR (Q = diag[1, 1, 80, 5, 1, 80], R = diag[0.1, 0.1])

| Initial state `x₀` | Outcome |
|---|---|
| `[10, 10, 0, 0, 0, 0]` | Converges to hover |
| `[30, 30, 1.2, 0, 0, 0]` | Converges |
| `[50, 50, 2.0, 0, 0, 0]` | Converges, slower |
| `[70, 70, 1.6, 0, 0, 0]` | Converges, near the limit (about 70 m) |

The farther the start, the slower the convergence (up to about 30 s of simulation time).

### MPC (Q = diag[1, 1, 2000, 10, 100, 2000], R = diag[0.1, 0.1])

| Initial state `x₀` | Outcome |
|---|---|
| `[50, 50, 1.2, 0, 0, 0]` | Converges |
| `[100, 100, 0.94, 0, 0, 0]` | Converges |
| `[200, 200, 0.94, 0, 0, 0]` | Converges (about 60 s) |
| `[80, 80, 2.0, 0, 0, 0]` → ref `[30, 20, 0, 0, 0, 0]` | Tracks a non-hover reference |

- **Noise:** zero-mean Gaussian noise with `Σ = diag(0.01, 0.01, 0.0025, 0.0025, 0.0025, 0.0025)`. θ and the control inputs are the most affected, and larger covariances can fully mask the true signal.
- **Trajectory tracking:** with `Q = diag[100, 100, 10, 100, 100, 10]`, the quadrotor follows a 20 m radius circle closely. Heavily penalizing θ gives poor tracking, so the weights must be balanced between position and angle.

### Key takeaways
- LQR is excellent near the operating point, but it degrades once input saturation is introduced and the start is far away.
- MPC handles constraints natively and converges from much farther away.
- The very large θ weights needed for long distances slow the approach and may not be practical on real hardware.
- A longer horizon is slower to solve but can give better results; a shorter horizon converges faster but only over shorter distances.

## Getting started

### Requirements
- Python 3.9+
- `numpy`, `matplotlib`
- `jax` (with `jaxlib`)
- `proxsuite`
- `jupyter` or JupyterLab

```bash
pip install numpy matplotlib jax jaxlib proxsuite jupyter
```

### Run
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
jupyter notebook
```

Run the notebooks in order (`Question1` → `Question2` → `Question3` → `Trajectory`). Each notebook is self-contained: it redefines the dynamics, RK4 integrator, linearization and LQR before using them.

## Tuning tips

- **Weights `Q`, `R`:** a large θ weight keeps the vehicle stable from far starts, while larger position weights improve trajectory tracking.
- **Horizon (`tc`, `K = round(tc / dt)`):** a longer horizon is slower but more accurate.
- **Linearization point:** it does not have to equal the reference. Any fixed point works as long as the matching control input `ū` is used (for hover, `ū = [mg, 0]`).

## Citation

If you use this work, please cite:

```
K. Spinos, "Modeling and Control of A Planar Quadrotor."
```

## License

Add your preferred license here (e.g. MIT).
