"""Orakel (ORACLE-PLAYBOOK §5): jede Kernroutine gegen einen anderen Rechenweg, alles unter 10 s.

(a) der Abgleich gegen Sinkhorn in der Log-Domäne nach jedem Durchlauf, gegen die KL-Projektion über die Lagrange-Gleichungen (`root`) und gegen SLSQP;
(b) OPT: eigener Transportsimplex gegen HiGHS und `networkx.network_simplex`; (c) die Hyman-Schleife gegen `brentq` und gegen die geschlossene Formel von Handbeispiel 4;
(d) das Strafverfahren gegen die direkte Minimierung der Zielfunktion; (e) Zonen und Hilfsgrößen. Die Zulässigkeit steht in `test_feasibility.py`."""

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import brentq, linprog, minimize

import mab_constants as C
import mab_evaluation as E
import mab_scaling as S
from mab_hyman import ST_OK, hyman_stapel
from mab_scenario import basis_matrix, instanz_rng, randsummen_wachstum, zonen
from mab_simplex import loese_transport
from orakel_hilfe import kl_projektion_root, sinkhorn_log


def _instanz(i, n=6):
    g = instanz_rng(77, 99, i)
    z = zonen(n, g)
    T0 = basis_matrix(z.kosten, g)
    a, b = randsummen_wachstum(T0, g, C.RS_BREITE[C.RS_V3])
    return z, T0, a, b


def test_furness_gleich_sinkhorn_in_der_log_domaene_nach_jedem_durchlauf():
    groesste = 0.0
    for i in range(10):
        z, T0, a, b = _instanz(i, n=5 + i % 4)
        for k in (1, 2, 3, 5, 8, 20):
            P1 = S.plan_nach(T0, a, b, k)
            P2 = sinkhorn_log(T0, a, b, k)
            groesste = max(groesste, float(np.abs(P1 - P2).max() / P2.max()))
    assert groesste < 1e-11


def test_abgleich_gleich_kl_projektion_ueber_die_lagrange_gleichungen():
    for i in range(6):
        z, T0, a, b = _instanz(i, n=3 + i % 3)
        P = S.abgleichen(T0, a, b, tol=1e-13).plan
        Q = kl_projektion_root(T0, a, b)
        assert np.abs(P - Q).max() / Q.max() < 1e-9


def test_abgleich_gleich_slsqp_auf_der_kl_zielfunktion():
    """Dritter Rechenweg: SLSQP auf min Σ P ln(P/K) unter den Randsummen (n = 3, Variablen O(0,1)); die Löser-Toleranz erlaubt 1e-5."""
    for i in range(3):
        z, T0, a, b = _instanz(i, n=3)
        K = T0 / T0.sum()
        P0 = np.outer(a, b)
        f = lambda x: float((x * np.log(x / K.ravel())).sum())
        grad = lambda x: np.log(x / K.ravel()) + 1.0
        cons = [{"type": "eq", "fun": lambda x, i=i: x.reshape(3, 3).sum(1)[i] - a[i]} for i in range(3)] + \
               [{"type": "eq", "fun": lambda x, j=j: x.reshape(3, 3).sum(0)[j] - b[j]} for j in range(2)]
        res = minimize(f, P0.ravel(), jac=grad, constraints=cons, bounds=[(1e-9, 1)] * 9, method="SLSQP", options=dict(ftol=1e-14, maxiter=500))
        assert res.success
        P = S.abgleichen(T0, a, b, tol=1e-13).plan
        assert np.abs(res.x.reshape(3, 3) - P).max() < 1e-5


def test_gibbs_form_und_randsummen_des_ergebnisses():
    z, T0, a, b = _instanz(0, n=7)
    ab = S.abgleichen(T0, a, b)
    assert ab.plan.sum(1) == pytest.approx(a, abs=1e-8) and ab.plan.sum(0) == pytest.approx(b, abs=1e-8) and (ab.plan > 0).all()
    u = ab.plan[:, 0] / T0[:, 0] / (ab.plan[0, 0] / T0[0, 0]) * 1.0                 # P_ij / T_ij hat Rang 1: P = diag(u) T diag(v)
    r = ab.plan / T0
    assert np.linalg.matrix_rank(np.log(r), tol=1e-7) <= 2 and u.shape == (7,)       # log r_ij = ln u_i + ln v_j hat Rang höchstens 2


# ----------------------------------------------------------------------------------------------------------------------------------
# OPT
# ----------------------------------------------------------------------------------------------------------------------------------

def test_opt_eigener_simplex_gleich_highs_gleich_networkx():
    for i in range(12):
        z = zonen(5 + i % 5, instanz_rng(5, 98, i))
        n = z.n
        eigen = E.opt_wert(z.kosten, z.a, z.b)
        A = np.zeros((2 * n, n * n))
        for r in range(n):
            A[r, r * n:(r + 1) * n] = 1
            A[n + r, r::n] = 1
        res = linprog(z.kosten.ravel(), A_eq=A, b_eq=np.concatenate([z.a, z.b]) * 1e3, bounds=(0, None), method="highs")
        assert eigen == pytest.approx(res.fun / 1e3, rel=1e-6, abs=1e-9)           # HiGHS rechnet mit Toleranz 1e-7
        # networkx: ganzzahlig skaliert (Mengen = Fahrten, Kosten auf 1e6 gerundet)
        S_ = z.gesamt
        G = nx.DiGraph()
        for r in range(n):
            G.add_node(("h", r), demand=-int(round(z.a[r] * S_)))
            G.add_node(("z", r), demand=int(round(z.b[r] * S_)))
        for r in range(n):
            for c in range(n):
                G.add_edge(("h", r), ("z", c), weight=int(round(z.kosten[r, c] * 1e6)))
        wert = nx.network_simplex(G)[0] / 1e6 / S_
        assert eigen == pytest.approx(wert, abs=1e-6)


# ----------------------------------------------------------------------------------------------------------------------------------
# Hyman-Schleife
# ----------------------------------------------------------------------------------------------------------------------------------

def test_hyman_schleife_gleich_brentq_und_findet_das_wahre_epsilon_wieder():
    for i in range(5):
        z = zonen(6, instanz_rng(6, 97, i))
        opt = E.opt_wert(z.kosten, z.a, z.b)
        wahr = 1.0 * opt
        Pw = S.abgleichen(np.exp(-z.kosten / wahr), z.a, z.b, tol=1e-13, max_durchlaeufe=100000).plan
        c_obs = float((z.kosten * Pw).sum())
        h = hyman_stapel(z.kosten[None], z.a[None], z.b[None], np.array([c_obs]))
        assert h.status[0] == ST_OK
        f = lambda beta: float((z.kosten * S.abgleichen(np.exp(-beta * z.kosten), z.a, z.b, tol=1e-13, max_durchlaeufe=100000).plan).sum()) - c_obs
        beta_bq = brentq(f, 0.2 / opt, 5.0 / opt, xtol=1e-12, rtol=1e-12)
        assert h.beta[0] == pytest.approx(beta_bq, rel=1e-4)
        assert 1.0 / (h.beta[0] * wahr) == pytest.approx(1.0, abs=1e-4)           # Wiedergewinnung von ε in der exp-Welt (1e-4)


def test_kalibrierung_bei_zwei_zonen_gleich_der_geschlossenen_formel():
    """Handbeispiel 4: aus P gilt q² = P12 P21 / (P11 P22) mit q = exp(−1/ε)."""
    C2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    a, b = np.array([0.4, 0.6]), np.array([0.6, 0.4])
    for kosten, q2 in ((0.24, 11 / 361), (0.40, 1 / 3)):
        h = hyman_stapel(C2[None], a[None], b[None], np.array([kosten]))
        assert h.status[0] == ST_OK
        eps = 1.0 / h.beta[0]
        P = S.abgleichen(np.exp(-C2 / eps), a, b, tol=1e-13).plan
        assert P[0, 1] * P[1, 0] / (P[0, 0] * P[1, 1]) == pytest.approx(q2, rel=1e-4)
        assert eps == pytest.approx(-1.0 / np.log(np.sqrt(q2)), rel=1e-4)


# ----------------------------------------------------------------------------------------------------------------------------------
# Strafverfahren
# ----------------------------------------------------------------------------------------------------------------------------------

def test_strafverfahren_gleich_direkter_minimierung_der_zielfunktion():
    """min ε KL(P|K) + τ KL(P1|a) + τ KL(Pᵀ1|b) (verallgemeinerte KL) über log P mit L-BFGS und analytischem Gradienten; n = 3."""
    eps, tau = C.STRAFE_EPS, 10.0
    for i in range(3):
        g = instanz_rng(4, 96, i)
        K = basis_matrix(zonen(3, g).kosten, g)
        a = np.array([0.5, 0.3, 0.2])
        b = np.array([0.2, 0.3, 0.55])                                              # Summe 1,05: die Strafe lässt verschiedene Summen zu

        def f(y):
            P = np.exp(y).reshape(3, 3)
            r, c = P.sum(1), P.sum(0)
            kl = lambda x, w: float((x * np.log(x / w) - x + w).sum())
            return eps * kl(P.ravel(), K.ravel()) + tau * kl(r, a) + tau * kl(c, b)

        def grad(y):
            P = np.exp(y).reshape(3, 3)
            r, c = P.sum(1), P.sum(0)
            d = eps * np.log(P / K) + tau * np.log(r / a)[:, None] + tau * np.log(c / b)[None, :]
            return (d * P).ravel()

        res = minimize(f, np.log(np.full(9, 0.1)), jac=grad, method="L-BFGS-B", options=dict(ftol=1e-15, gtol=1e-12, maxiter=2000))
        P, it = S.strafe_stapel(K[None], a[None], b[None], tau, eps)
        assert it[0] > 0
        assert np.abs(P[0] - np.exp(res.x).reshape(3, 3)).max() < 1e-6


# ----------------------------------------------------------------------------------------------------------------------------------
# Zonen und Reihen
# ----------------------------------------------------------------------------------------------------------------------------------

def test_mittlere_kosten_fallen_mit_fallendem_epsilon_in_allen_instanzen():
    for i in range(20):
        z = zonen(8, instanz_rng(3, 95, i))
        opt = E.opt_wert(z.kosten, z.a, z.b)
        kosten = []
        for r in (0.3, 1.0, 3.0, 10.0):
            P = S.abgleichen(S.kern_gravitation(z.kosten, r * opt), z.a, z.b).plan
            kosten.append(float((z.kosten * P).sum()))
        assert all(x >= opt - 1e-12 for x in kosten) and kosten == sorted(kosten)         # Wilsons Bedingung in der Lesart „Aufwand fest“: monoton (der Grenzwert ist das LP-Minimum)


def test_handbeispiel_4_anteil_theta_und_epsilon_ueber_opt_in_bruechen_und_im_code():
    from fractions import Fraction as F

    a, b = [F(2, 5), F(3, 5)], [F(3, 5), F(2, 5)]
    opt, produkt = F(1, 5), a[0] * b[1] + a[1] * b[0]
    assert produkt == F(13, 25)
    for t, theta, q2 in ((F(1, 50), F(1, 8), F(11, 361)), (F(1, 10), F(5, 8), F(1, 3))):
        P11, P12 = a[0] - t, t
        P21 = b[0] - P11
        P22 = a[1] - P21
        assert P12 * P21 / (P11 * P22) == q2
        assert (opt + 2 * t - opt) / (produkt - opt) == theta                    # Kosten 1/5 + 2t
    assert 2 / np.log(361 / 11) == pytest.approx(0.5729, abs=5e-4) and 2 / np.log(361 / 11) / 0.2 == pytest.approx(2.86, abs=5e-3)
    assert 2 / np.log(3) == pytest.approx(1.8205, abs=5e-4) and 2 / np.log(3) / 0.2 == pytest.approx(9.10, abs=5e-3)
    C2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = hyman_stapel(C2[None], np.array([[0.4, 0.6]]), np.array([[0.6, 0.4]]), np.array([0.19]))        # Kosten unter dem LP-Minimum 1/5: kein ε
    assert h.status[0] != ST_OK
