"""Unabhängige Rechenwege für die Orakel-Tests und die Messreihen (nur in tests/ und tools/, nie in der App): Sinkhorn in der Log-Domäne, Brüche, KL-Projektion über die
Lagrange-Gleichungen, LP-Zulässigkeit mit Maximierung der kleinsten Zelle."""

from fractions import Fraction as F

import numpy as np
from scipy.optimize import linprog, root
from scipy.special import logsumexp


def sinkhorn_log(K, a, b, durchlaeufe):
    """Skalierung in der Log-Domäne (Potentiale f, g mit u = exp(f), v = exp(g)); gibt die Matrix nach `durchlaeufe` Durchläufen zurück. Der Start (v = 1) ist derselbe wie im Abgleich."""
    logK = np.log(np.asarray(K, float))
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    g = np.zeros(len(b))
    f = np.zeros(len(a))
    for _ in range(durchlaeufe):
        f = np.log(a) - logsumexp(logK + g[None, :], axis=1)
        g = np.log(b) - logsumexp(logK + f[:, None], axis=0)
    return np.exp(f[:, None] + logK + g[None, :])


def kl_projektion_root(K, a, b):
    """KL-Projektion von K auf die Matrizen mit Randsummen a, b über die Lagrange-Gleichungen: P_ij = K_ij exp(x_i + y_j), Unbekannte (x, y) mit y_0 = 0 (n ≤ 6)."""
    K = np.asarray(K, float)
    n, m = K.shape

    def plan(z):
        x = z[:n]
        y = np.concatenate([[0.0], z[n:]])
        return K * np.exp(x[:, None] + y[None, :])

    def gl(z):
        P = plan(z)
        return np.concatenate([P.sum(1) - a, P.sum(0)[1:] - b[1:]])

    z0 = np.zeros(n + m - 1)
    z0[:n] = np.log(a / K.sum(1))
    sol = root(gl, z0, method="hybr", tol=1e-12)
    assert np.abs(gl(sol.x)).max() < 1e-12, sol.message
    return plan(sol.x)


def klasse_lp(S, a, b):
    """0, 1 oder 2 wie in den Handbeispielen: ein LP maximiert die kleinste Zelle auf dem Träger (Massen mit 1e3 skaliert, HiGHS meldet bei kleinen Massen „infeasible“)."""
    S = np.asarray(S, bool)
    n, m = S.shape
    zellen = [(i, j) for i in range(n) for j in range(m) if S[i, j]]
    nv = len(zellen) + 1
    A_eq = np.zeros((n + m, nv))
    for k, (i, j) in enumerate(zellen):
        A_eq[i, k] = 1
        A_eq[n + j, k] = 1
    A_ub = np.zeros((len(zellen), nv))
    for k in range(len(zellen)):
        A_ub[k, k] = -1
        A_ub[k, -1] = 1
    c = np.zeros(nv)
    c[-1] = -1
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(zellen)), A_eq=A_eq, b_eq=np.concatenate([a, b]) * 1e3, bounds=[(0, None)] * nv, method="highs")
    if res.status == 2:
        return 0
    assert res.status == 0, res.message
    return 2 if -res.fun / 1e3 > 1e-9 else 1


def schritt_zeilen(M, a):
    z = [sum(r) for r in M]
    return [[x * a[i] / z[i] for x in M[i]] for i in range(len(M))]


def schritt_spalten(M, b):
    s = [sum(c) for c in zip(*M)]
    return [[M[i][j] * b[j] / s[j] for j in range(len(M[0]))] for i in range(len(M))]


def bruch_matrix(M):
    return [[F(x) for x in r] for r in M]


def kreuzverhaeltnis(M):
    return M[0][0] * M[1][1] / (M[0][1] * M[1][0])
