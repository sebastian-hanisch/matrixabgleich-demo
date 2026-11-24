"""Eigenschaftstests (je 20 bis 50 Zufallsinstanzen, fester Zufalls-Seed): Invarianzen des Abgleichs und der Kalibrierung."""

from fractions import Fraction as F

import numpy as np
import pytest

import mab_constants as C
import mab_scaling as S
from mab_hyman import ST_GRENZE, ST_INNEN, ST_OK, hyman_stapel
from mab_scenario import basis_matrix, instanz_rng, randsummen_wachstum, traeger_matrix, zonen
import mab_evaluation as E
from orakel_hilfe import bruch_matrix, schritt_spalten, schritt_zeilen


def _inst(i, n=6, nullen=0.0):
    g = instanz_rng(12, 94, i)
    z = zonen(n, g)
    T0 = traeger_matrix(n, nullen, g) if nullen else basis_matrix(z.kosten, g)
    a, b = randsummen_wachstum(T0, g, C.RS_BREITE[C.RS_V3])
    return z, T0, a, b


def test_alte_matrix_mal_konstante_gibt_dieselbe_loesung():
    for i in range(20):
        z, T0, a, b = _inst(i)
        P1 = S.abgleichen(T0, a, b).plan
        P2 = S.abgleichen(7.3 * T0, a, b).plan
        assert P2 == pytest.approx(P1, rel=1e-8)


def test_randsummen_mal_konstante_geben_die_loesung_mal_konstante():
    for i in range(20):
        z, T0, a, b = _inst(i)
        P1 = S.abgleichen(T0, a, b).plan
        P2 = S.abgleichen(T0, 3 * a, 3 * b).plan
        assert P2 == pytest.approx(3 * P1, rel=1e-7)


def test_umnummerieren_und_transponieren():
    rng = np.random.default_rng(1)
    for i in range(20):
        z, T0, a, b = _inst(i, n=7)
        P = S.abgleichen(T0, a, b).plan
        pz, ps = rng.permutation(7), rng.permutation(7)
        Pp = S.abgleichen(T0[pz][:, ps], a[pz], b[ps]).plan
        assert Pp == pytest.approx(P[pz][:, ps], rel=1e-7)
        Pt = S.abgleichen(T0.T, b, a).plan
        assert Pt == pytest.approx(P.T, rel=1e-7)


def test_zeilen_oder_spaltenschritt_zuerst_gibt_denselben_grenzwert():
    """Der Spaltenschritt zuerst ist dieselbe Rechnung an der transponierten Aufgabe."""
    for i in range(20):
        z, T0, a, b = _inst(i)
        P1 = S.abgleichen(T0, a, b, tol=1e-12).plan
        P2 = S.abgleichen(T0.T, b, a, tol=1e-12).plan.T
        assert P2 == pytest.approx(P1, rel=1e-8)


def test_nullen_bleiben_nullen_und_die_randsummen_stimmen():
    for i in range(30):
        z, T0, a, b = _inst(i, n=6, nullen=0.3)
        ab = S.abgleichen(T0, a, b)
        if ab.status == S.STATUS_OK:
            assert (ab.plan[T0 == 0] == 0).all()
            assert ab.plan.sum(1) == pytest.approx(a, abs=1e-8) and ab.plan.sum(0) == pytest.approx(b, abs=1e-8)


def test_kreuzverhaeltnisse_aller_zweier_minoren_bleiben_in_bruechen_erhalten():
    rng = np.random.default_rng(2)
    for _ in range(20):
        M = bruch_matrix(rng.integers(1, 9, (3, 3)).tolist())
        ziel_z = [F(int(x)) for x in rng.integers(3, 9, 3)]
        s = sum(ziel_z)
        ziel_s = [F(int(x)) for x in rng.integers(1, 9, 3)]
        ziel_s = [x * s / sum(ziel_s) for x in ziel_s]
        minoren = lambda X: [X[i][j] * X[k][l] / (X[i][l] * X[k][j]) for i in range(3) for k in range(i + 1, 3) for j in range(3) for l in range(j + 1, 3)]
        vorher = minoren(M)
        for _ in range(4):
            M = schritt_spalten(schritt_zeilen(M, ziel_z), ziel_s)
        assert minoren(M) == vorher


def test_abgleich_auf_die_eigenen_randsummen_aendert_nichts():
    for i in range(20):
        z, T0, a, b = _inst(i)
        P = S.abgleichen(T0, T0.sum(1) / T0.sum(), T0.sum(0) / T0.sum())
        assert P.plan == pytest.approx(T0 / T0.sum(), rel=1e-9)


def test_strafe_mit_tau_gegen_unendlich_naehert_sich_dem_abgleich_monoton():
    z, T0, a, b = _inst(0, n=5)
    K = (T0 / a.sum())[None]
    exakt = S.abgleich_stapel(K, a[None], b[None], tol=1e-13).plan
    abstaende = []
    for tau in (1.0, 10.0, 100.0):
        P, it = S.strafe_stapel(K, a[None], b[None], tau)
        assert it[0] > 0
        abstaende.append(float(np.abs(P - exakt).max()))
    assert abstaende == sorted(abstaende, reverse=True)


def test_kalibrierung_gleiche_schritte_kalt_und_warm_und_warm_spart_durchlaeufe():
    zs = [zonen(8, instanz_rng(13, 93, i)) for i in range(10)]
    kosten = np.array([z.kosten for z in zs])
    a, b = np.array([z.a for z in zs]), np.array([z.b for z in zs])
    opt = np.array([E.opt_wert(z.kosten, z.a, z.b) for z in zs])
    prod = np.array([z.a @ z.kosten @ z.b for z in zs])
    c_obs = opt + 0.2 * (prod - opt)
    kalt = hyman_stapel(kosten, a, b, c_obs, warm=False)
    warm = hyman_stapel(kosten, a, b, c_obs, warm=True)
    assert (kalt.status == ST_OK).all() and (warm.status == ST_OK).all()
    assert (kalt.schritte == warm.schritte).all()
    assert warm.durchlaeufe.sum() < kalt.durchlaeufe.sum()
    assert kalt.beta == pytest.approx(warm.beta, rel=1e-4)


def test_beobachtung_unter_dem_lp_minimum_hat_kein_epsilon_und_wird_gemeldet():
    zs = [zonen(8, instanz_rng(14, 92, i)) for i in range(6)]
    kosten = np.array([z.kosten for z in zs])
    a, b = np.array([z.a for z in zs]), np.array([z.b for z in zs])
    opt = np.array([E.opt_wert(z.kosten, z.a, z.b) for z in zs])
    h = hyman_stapel(kosten, a, b, 0.9 * opt, max_schritte=C.MAX_SCHRITTE_RANDFALL)
    assert not (h.status == ST_OK).any()
    assert set(h.status) <= {ST_INNEN, ST_GRENZE}
    assert (h.beta > 10 / opt).all()                                     # β läuft weit über den Startwert hinaus


def test_beobachtung_gleich_produktmass_gibt_beta_gegen_null():
    zs = [zonen(8, instanz_rng(15, 91, i)) for i in range(4)]
    kosten = np.array([z.kosten for z in zs])
    a, b = np.array([z.a for z in zs]), np.array([z.b for z in zs])
    prod = np.array([z.a @ z.kosten @ z.b for z in zs])
    h = hyman_stapel(kosten, a, b, prod * 0.999)
    assert (h.status == ST_OK).all() and (h.beta * prod < 0.5).all()
