"""Classical and data-driven control baselines for the canonical random-LDS problem.

Each controller tests one assumption (see ``docs/neurostim-baselines.md``):

- ``PIController``: minimal feedback. Output error, integral action, static
  decoupling through the *nominal* steady-state gain.
- ``HInfController``: robust. An H∞ state-feedback gain for the nominal model,
  treating the patient mismatch ``(B - B̄) u`` as a disturbance to be
  attenuated. One controller for the whole population, no identification.
- ``ILQRController`` (privileged): nonlinear MPC by iterative LQR on the
  **true** nonlinear model. It measures the difficulty of *control*, with no
  learning involved.
- ``MPPIController`` (privileged): sampling-based MPC (model-predictive path
  integral) on the true model. Unlike iLQR it does not need gradients and can
  jump between modes.
- ``DeePCController``: data-enabled predictive control. It predicts directly
  from a Hankel matrix of the patient's own recent input/output data, with no
  model fitted.
- ``KoopmanController``: extended DMD with control. The same RLS identifier
  as the adaptive controller, run on lifted observables ``ψ(o)``, so that
  nonlinear dynamics become approximately linear.

Oracle LQG, the nominal (population) LQG and adaptive RLS + CE are in
``canonical.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from .canonical import (
    _LQG,
    AdaptiveController,
    Controller,
    KalmanFilter,
    lq_tracker,
    recruit_grad,
    step_jacobians,
    step_mean,
)


def _nominal_dc_gain(pop):
    """Steady-state percept gain ``D (I - Ā)^{-1} B̄`` of the mean patient (linear drive)."""
    return pop.D @ np.linalg.inv(np.eye(pop.cfg.n) - pop.A_bar) @ pop.B_bar


def _percept_obs(pop, o):
    return pop.D @ o if pop.cfg.observe == "full" else o


# ----------------------------------------------------------------- PI
class PIController(Controller):
    """``u = G⁺ (K_p e + K_i Σ e)`` with ``e = z* - ẑ`` and ``G = D(I-Ā)^{-1}B̄``."""

    name = "PI (nominal decoupling)"

    def __init__(self, kp: float = 0.05, ki: float = 0.02):  # grid-tuned on the mean patient
        self.kp, self.ki = kp, ki

    def reset(self, pop, z_star, info):
        super().reset(pop, z_star, info)
        self.Ginv = np.linalg.pinv(_nominal_dc_gain(pop))
        self.integral = np.zeros(pop.cfg.dz)

    def act(self, o, info):
        c = self.cfg
        e = self.z_star - _percept_obs(self.pop, o)
        u = self.Ginv @ (self.kp * e + self.ki * self.integral)
        u_sat = np.clip(u, -c.u_max, c.u_max)
        if np.allclose(u, u_sat):  # anti-windup: freeze the integrator when saturated
            self.integral = self.integral + e
        return u_sat


# ----------------------------------------------------------------- H-infinity
def hinf_gain(A, B, W, Q, R, gamma, iters: int = 2000, tol: float = 1e-10):
    """Discrete-time H∞ state feedback by iterating the game Riccati equation.

    Minimax of ``Σ xᵀQx + uᵀRu - γ² wᵀw`` for ``x' = Ax + Bu + Ww``. Returns
    ``K`` (``u = -Kx``) or ``None`` if ``γ`` is below the achievable level.
    """
    n, m = B.shape
    Bw = np.hstack([B, W])
    Rw = np.block(
        [
            [R, np.zeros((m, W.shape[1]))],
            [np.zeros((W.shape[1], m)), -(gamma**2) * np.eye(W.shape[1])],
        ]
    )
    P = Q.copy()
    for _ in range(iters):
        S = Rw + Bw.T @ P @ Bw
        P_new = Q + A.T @ P @ A - A.T @ P @ Bw @ np.linalg.solve(S, Bw.T @ P @ A)
        P_new = 0.5 * (P_new + P_new.T)
        if not np.all(np.isfinite(P_new)) or np.max(np.abs(P_new)) > 1e8:
            return None
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    else:
        return None
    if np.min(np.linalg.eigvalsh(gamma**2 * np.eye(W.shape[1]) - W.T @ P @ W)) <= 0:
        return None  # saddle point does not exist at this gamma
    K = np.linalg.solve(Rw + Bw.T @ P @ Bw, Bw.T @ P @ A)[:m]
    if np.max(np.abs(np.linalg.eigvals(A - B @ K))) >= 1:
        return None  # near γ_min the iteration can land on a non-stabilising branch
    return K


class HInfController(_LQG):
    """Robust: nominal Kalman filter + H∞ gain (``γ`` = 1.5 × smallest feasible)."""

    name = "H∞ robust (nominal)"

    def __init__(self, margin: float = 1.5):
        self.margin = margin

    def reset(self, pop, z_star, info):
        Controller.reset(self, pop, z_star, info)
        c = pop.cfg
        A, B = pop.A_bar, pop.B_bar
        self.kf = KalmanFilter(A, B, pop.C, c.process_noise, c.obs_noise)
        self.x_ss, self.u_ss, _ = lq_tracker(A, B, pop.D, np.zeros(c.n), z_star, c.lam_u, c.u_max)
        Q, R, W = pop.D.T @ pop.D + 1e-6 * np.eye(c.n), c.lam_u * np.eye(c.m), np.eye(c.n)
        lo, hi = 0.1, 100.0
        for _ in range(30):  # bisection on the attenuation level
            mid = np.sqrt(lo * hi)
            lo, hi = (
                (lo, mid) if hinf_gain(A, B, W, Q, R, mid, iters=400) is not None else (mid, hi)
            )
        self.gamma = self.margin * hi
        K = hinf_gain(A, B, W, Q, R, self.gamma)
        self.K = K if K is not None else np.zeros((c.m, c.n))


# ----------------------------------------------------------------- iLQR / NMPC
class ILQRController(Controller):
    """Receding-horizon iLQR on the true nonlinear model (privileged).

    Stage cost ``‖D x_k - z*‖² + λ‖u_k‖²``, terminal weight ``w_f``. Gauss-Newton
    (no second derivatives of the dynamics), Levenberg regularisation, a
    backtracking line search and box constraints by clipping. The previous
    plan is shifted as a warm start. A small random initial plan breaks the
    ``u = 0`` saddle of polarity-insensitive recruitment.
    """

    name = "iLQR-NMPC *"
    privileged = True

    def __init__(self, horizon: int = 30, iters: int = 3, wf: float = 5.0, seed: int = 0):
        self.H, self.iters, self.wf = horizon, iters, wf
        self.rng = np.random.default_rng(seed)

    def reset(self, pop, z_star, info):
        super().reset(pop, z_star, info)
        self.U = 0.3 * self.rng.normal(size=(self.H, pop.cfg.m))

    def _rollout(self, A, B, x0, U):
        X = [x0]
        for u in U:
            X.append(step_mean(self.cfg, A, B, X[-1], u))
        return np.array(X)

    def _cost(self, X, U):
        e = X[1:] @ self.pop.D.T - self.z_star
        stage = np.sum(e[:-1] ** 2) + self.cfg.lam_u * np.sum(U**2)
        return stage + self.wf * np.sum(e[-1] ** 2)

    def act(self, o, info):
        c, D, z = self.cfg, self.pop.D, self.z_star
        A, B, x0 = info["A"], info["B_t"], info["x"]
        U = self.U
        X = self._rollout(A, B, x0, U)
        J = self._cost(X, U)
        mu = 1e-3
        for _ in range(self.iters):
            Vx = 2 * self.wf * D.T @ (D @ X[-1] - z)
            Vxx = 2 * self.wf * D.T @ D
            ks, Ks = [], []
            for k in reversed(range(self.H)):
                fx, fu = step_jacobians(c, A, B, X[k], U[k])
                lx = 2 * D.T @ (D @ X[k] - z) if k > 0 else np.zeros(c.n)
                lxx = 2 * D.T @ D if k > 0 else np.zeros((c.n, c.n))
                Qx = lx + fx.T @ Vx
                Qu = 2 * c.lam_u * U[k] + fu.T @ Vx
                Qxx = lxx + fx.T @ Vxx @ fx
                Quu = 2 * c.lam_u * np.eye(c.m) + fu.T @ Vxx @ fu + mu * np.eye(c.m)
                Qux = fu.T @ Vxx @ fx
                kff = -np.linalg.solve(Quu, Qu)
                Kfb = -np.linalg.solve(Quu, Qux)
                Vx = Qx + Kfb.T @ Quu @ kff + Kfb.T @ Qu + Qux.T @ kff
                Vxx = Qxx + Kfb.T @ Quu @ Kfb + Kfb.T @ Qux + Qux.T @ Kfb
                Vxx = 0.5 * (Vxx + Vxx.T)
                ks.append(kff)
                Ks.append(Kfb)
            ks, Ks = ks[::-1], Ks[::-1]
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1):
                xn, Xn, Un = x0, [x0], []
                for k in range(self.H):
                    u = np.clip(U[k] + alpha * ks[k] + Ks[k] @ (xn - X[k]), -c.u_max, c.u_max)
                    xn = step_mean(c, A, B, xn, u)
                    Xn.append(xn)
                    Un.append(u)
                Xn, Un = np.array(Xn), np.array(Un)
                Jn = self._cost(Xn, Un)
                if Jn < J:
                    X, U, J, improved = Xn, Un, Jn, True
                    mu = max(mu / 2, 1e-6)
                    break
            if not improved:
                mu *= 10
        self.U = np.vstack([U[1:], U[-1:]])
        return U[0]


# ----------------------------------------------------------------- MPPI
class MPPIController(Controller):
    """Model-predictive path integral control on the true model (privileged).

    Sample ``K`` perturbed plans, weight them by ``exp(-J / λ)`` and average.
    The temperature adapts to the spread of costs, and the previous plan is
    shifted as a warm start. The sampling noise shrinks over the iterations
    of each step (coarse search, then refinement). Perturbations are
    piecewise constant over ``knots`` blocks. Sampling all ``H x m`` numbers
    independently makes the weighted average random-walk along directions
    the cost barely sees. Gradient-free, so the ``±u`` symmetry of
    polarity-insensitive recruitment is not a trap.

    Terminal cost: the infinite-horizon LQR cost-to-go ``(x - x_ss)ᵀP(x - x_ss)``
    of the model linearised at a typical operating point. Without it, a
    finite horizon lets energy build up in modes that the percept does not
    see yet, and the plan drifts.
    """

    name = "MPPI *"
    privileged = True

    def __init__(
        self, horizon=15, samples=256, sigma=0.6, temp=0.1, iters=4, wf=5.0, knots=3, seed=0
    ):
        self.H, self.K, self.sigma, self.temp, self.knots = horizon, samples, sigma, temp, knots
        self.iters, self.wf = iters, wf
        self.rng = np.random.default_rng(seed)

    def reset(self, pop, z_star, info):
        super().reset(pop, z_star, info)
        self.U = np.zeros((self.H, pop.cfg.m))
        self.t = 0

    def _terminal(self, A, B):
        c = self.cfg
        u_op = np.full(c.m, 0.5 if c.recruit == "even" else 0.0)
        Bl = B * recruit_grad(c, u_op)[None, :]
        Q = self.pop.D.T @ self.pop.D + 1e-6 * np.eye(c.n)
        self.P = solve_discrete_are(A, Bl, Q, c.lam_u * np.eye(c.m))
        self.x_ss, _, _ = lq_tracker(
            A, Bl, self.pop.D, np.zeros(c.n), self.z_star, c.lam_u, c.u_max
        )

    def plan_cost(self, A, B, x0, Us):
        """Costs of ``K`` plans ``Us (K, H, m)`` from ``x0``."""
        c, D = self.cfg, self.pop.D
        X = np.broadcast_to(x0, (Us.shape[0], c.n)).copy()
        J = c.lam_u * np.sum(Us**2, axis=(1, 2))
        for k in range(self.H):
            X = step_mean(c, A, B, X, Us[:, k])
            e = X @ D.T - self.z_star
            J += np.sum(e**2, 1)
        dx = X - self.x_ss
        return J + np.einsum("ki,ij,kj->k", dx, self.P, dx)

    def act(self, o, info):
        c = self.cfg
        A, B, x0 = info["A"], info["B_t"], info["x"]
        if self.t % 10 == 0:  # B_t may drift
            self._terminal(A, B)
        self.t += 1
        for i in range(self.iters):
            eps = self.sigma * 0.5**i * self.rng.normal(size=(self.K, self.knots, c.m))
            eps = np.repeat(eps, -(-self.H // self.knots), axis=1)[:, : self.H]
            eps[0] = 0.0  # keep the current plan in the pool
            Us = np.clip(self.U[None] + eps, -c.u_max, c.u_max)
            J = self.plan_cost(A, B, x0, Us)
            w = np.exp(-(J - J.min()) / (self.temp * (J.std() + 1e-9)))
            self.U = np.tensordot(w / w.sum(), Us, axes=1)
        u = self.U[0].copy()
        self.U = np.vstack([self.U[1:], self.U[-1:]])
        return u


# ----------------------------------------------------------------- DeePC
class DeePCController(Controller):
    """Data-enabled predictive control (Coulson, Lygeros & Dörfler 2019), regularised.

    With Hankel matrices of recent data ``[U_p; Y_p; U_f; Y_f]``, solve
    ``min_g ‖Y_f g - z*‖² + λ‖U_f g‖² + λ_g‖g‖²`` subject (softly) to the
    last ``T_ini`` samples, ``U_p g = u_ini`` and ``Y_p g = y_ini``, and apply
    the first input of ``U_f g``. The output ``y`` is the observed percept.

    By default the Hankel matrices hold only the persistently exciting probe
    data (classic DeePC). A sliding ``window`` would follow drift, but closed-loop
    data barely excite the system: the Hankel matrix loses rank and tracking
    degrades (0.001 -> 0.04 even without noise).
    """

    name = "DeePC"

    def __init__(
        self,
        t_ini=4,
        n_pred=10,
        window=None,
        n_explore=80,
        probe=0.5,
        lam_g=1.0,
        w_ini=1e3,
        dither=0.02,
        seed=0,
    ):
        self.t_ini, self.N, self.window = t_ini, n_pred, window
        self.n_explore, self.probe, self.lam_g, self.w_ini = n_explore, probe, lam_g, w_ini
        self.dither = dither
        self.rng = np.random.default_rng(seed)

    def reset(self, pop, z_star, info):
        super().reset(pop, z_star, info)
        self.us, self.ys = [], []

    @staticmethod
    def _hankel(w, L):
        T = len(w)
        return np.stack([w[i : T - L + 1 + i].reshape(T - L + 1, -1) for i in range(L)], 0)

    def act(self, o, info):
        c, m = self.cfg, self.cfg.m
        y = _percept_obs(self.pop, o)
        self.ys.append(y)
        t = len(self.us)
        if t < self.n_explore:
            u = self.probe * self.rng.normal(size=m)
        else:
            L, Ti, N = self.t_ini + self.N, self.t_ini, self.N
            # Pair each input u_k with the output it produces, y_{k+1}.
            if self.window:  # sliding window: follows drift, but closed-loop data excite poorly
                Uw, Yw = np.array(self.us[-self.window :]), np.array(self.ys[1:][-self.window :])
            else:  # classic DeePC: the persistently exciting probe data only
                Uw, Yw = (
                    np.array(self.us[: self.n_explore]),
                    np.array(self.ys[1 : self.n_explore + 1]),
                )
            HU, HY = self._hankel(Uw, L), self._hankel(Yw, L)  # (L, cols, dim)
            cols = HU.shape[1]
            Up, Uf = (
                HU[:Ti].transpose(0, 2, 1).reshape(-1, cols),
                HU[Ti:].transpose(0, 2, 1).reshape(-1, cols),
            )
            Yp, Yf = (
                HY[:Ti].transpose(0, 2, 1).reshape(-1, cols),
                HY[Ti:].transpose(0, 2, 1).reshape(-1, cols),
            )
            u_ini = np.concatenate(self.us[-Ti:])
            y_ini = np.concatenate(self.ys[-Ti:])
            s = np.sqrt(self.w_ini)
            M = np.vstack(
                [s * Up, s * Yp, Yf, np.sqrt(c.lam_u) * Uf, np.sqrt(self.lam_g) * np.eye(cols)]
            )
            rhs = np.concatenate(
                [s * u_ini, s * y_ini, np.tile(self.z_star, N), np.zeros(N * m + cols)]
            )
            g = np.linalg.lstsq(M, rhs, rcond=None)[0]
            u = (Uf @ g)[:m] + self.dither * self.rng.normal(size=m)
        u = np.clip(u, -c.u_max, c.u_max)
        self.us.append(u)
        return u


# ----------------------------------------------------------------- Koopman
class KoopmanController(AdaptiveController):
    """EDMD with control: RLS on lifted observables, then CE LQ tracking.

    ``ψ(o) = [o, o³, tanh(W o + b)]`` (fixed random features). The linear
    identifier and tracker of ``AdaptiveController`` then run on ``ψ(o)``, so a
    polynomial or saturating nonlinearity in the dynamics can be absorbed into
    a linear model of the lifted state. The input still enters linearly, so a
    nonlinear *recruitment* ``g(u)`` stays out of reach.
    """

    name = "Koopman-MPC (EDMD)"

    def __init__(
        self, n_rff: int = 16, forget: float = 0.995, prior: float = 10.0, n_explore: int = 60, **kw
    ):
        # 28 lifted features from a short probe overfit: a ridge prior (P0 = 10 I) and
        # a longer probe cut failures from 45 % to 10 % on the known patient.
        super().__init__(forget=forget, n_explore=n_explore, **kw)
        self.n_rff, self.prior = n_rff, prior

    def reset(self, pop, z_star, info):
        p = pop.C.shape[0]
        rng = np.random.default_rng(7)
        self.W = rng.normal(size=(self.n_rff, p))
        self.b = rng.uniform(-np.pi, np.pi, self.n_rff)
        self._p_raw = p
        super().reset(pop, z_star, info)
        # Redefine the identified output as the lifted vector; the target reads its first block.
        self.p = 2 * p + self.n_rff
        self.L = 1
        self.nxi = self.p
        k = self.nxi + self.cfg.m + 1
        self.theta = np.zeros((k, self.p))
        self.P = self.prior * np.eye(k)
        self.obs_hist = [np.zeros(self.p)]
        self.u_hist = [np.zeros(self.cfg.m)]
        Dz = np.zeros((self.Dz_obs.shape[0], self.p))
        Dz[:, :p] = self.Dz_obs
        self.Dz_obs = Dz

    def lift(self, o):
        return np.concatenate([o, o**3, np.tanh(self.W @ o + self.b)])

    def act(self, o, info):
        return super().act(self.lift(o), info)
