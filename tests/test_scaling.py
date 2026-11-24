"""Der Abgleich gegen die Handbeispiele (Brüche), die Zählweise, die Einheiten und die Randfälle.

Die Erwartungswerte stehen als Rechnung im Test (Brüche mit `fractions`), nicht als Abschrift der Ausgabe des Codes."""

from fractions import Fraction as F

import numpy as np
import pytest

import mab_constants as C
import mab_scaling as S
from orakel_hilfe import bruch_matrix, kreuzverhaeltnis, schritt_spalten, schritt_zeilen


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 1: Furness von Hand
# ----------------------------------------------------------------------------------------------------------------------------------

def test_handbeispiel_1_erster_durchlauf_in_bruechen_und_im_code(hand):
    T, a, b = hand
    M = bruch_matrix(T)
    M = schritt_zeilen(M, [F(5), F(6)])
    assert M == [[F(5, 3), F(10, 3)], [F(18, 7), F(24, 7)]]                       # Zeile 1 mal 5/3, Zeile 2 mal 6/7
    assert [sum(c) for c in zip(*M)] == [F(89, 21), F(142, 21)]
    M = schritt_spalten(M, [F(5), F(6)])
    assert M == [[F(175, 89), F(210, 71)], [F(270, 89), F(216, 71)]]
    zeilen = [sum(r) for r in M]
    fehler = sum(abs(z - t) for z, t in zip(zeilen, [5, 6]))
    assert fehler == F(960, 6319)
    P = S.plan_nach(T, a / 11, b / 11, 1) * 11                                      # derselbe Durchlauf im Code (Gesamtmasse 1, zurück in Fahrten)
    assert P == pytest.approx(np.array([[float(x) for x in r] for r in M]), rel=1e-13)
    ab = S.abgleichen(T, a, b)
    assert ab.fehler_verlauf[0] * 11 == pytest.approx(float(fehler), rel=1e-12)    # der Fehler nach Durchlauf 1 ist 960/6319 (Fahrten)


def test_handbeispiel_1_grenzwert_und_faktoren(hand):
    T, a, b = hand
    ab = S.abgleichen(T, a, b)
    assert ab.status == S.STATUS_OK and ab.durchlaeufe == 5 and ab.produkte == 10
    assert ab.plan == pytest.approx(np.array([[2.0, 3.0], [3.0, 3.0]]), abs=1e-8)
    u, v = np.array([2.0, 1.0]), np.array([1.0, 0.75])
    assert u[:, None] * T * v[None, :] == pytest.approx(np.array([[2.0, 3.0], [3.0, 3.0]]), rel=1e-14)    # Q = diag(u) T diag(v)


def test_handbeispiel_1_fehler_faellt_um_den_faktor_hundert_je_durchlauf(hand):
    T, a, b = hand
    fehler = S.abgleichen(T, a, b).fehler_verlauf
    for k in range(3):
        assert fehler[k + 1] / fehler[k] == pytest.approx(0.01, rel=1e-2)


def test_kreuzverhaeltnis_bleibt_in_jedem_schritt_exakt_erhalten():
    M = bruch_matrix([[1, 2], [3, 4]])
    for _ in range(6):
        M = schritt_zeilen(M, [F(5), F(6)])
        assert kreuzverhaeltnis(M) == F(2, 3)
        M = schritt_spalten(M, [F(5), F(6)])
        assert kreuzverhaeltnis(M) == F(2, 3)


def test_zaehlweise_zehn_durchlaeufe_kosten_zwanzig_produkte_plus_eine_pruefung(hand, monkeypatch):
    """Ein Durchlauf = zwei Matrix-Vektor-Produkte; die Prüfung nach dem letzten Durchlauf braucht ein weiteres K v (nicht mitgezählt)."""
    T, a, b = hand
    zaehler = {"n": 0}
    echt = np.einsum

    def zaehlend(*args, **kwargs):
        zaehler["n"] += 1
        return echt(*args, **kwargs)

    monkeypatch.setattr(S.np, "einsum", zaehlend)
    res = S.abgleich_stapel(T[None], (a / 11)[None], (b / 11)[None], tol=0.0, max_durchlaeufe=10)       # tol = 0: läuft die vollen 10 Durchläufe
    assert res.status[0] == 1
    assert zaehler["n"] == 2 * 10 + 1


def test_produkte_sind_zwei_je_durchlauf(hand):
    T, a, b = hand
    ab = S.abgleichen(T, a, b)
    assert ab.produkte == 2 * ab.durchlaeufe


def test_die_gezeigte_schleife_ist_die_ausgefuehrte_und_gleich_der_buchfuehrung(hand):
    T, a, b = hand
    an, bn = a / 11, b / 11
    for k in (1, 2, 5):
        stapel = S.abgleich_stapel(T[None], an[None], bn[None], tol=0.0, max_durchlaeufe=k)
        assert S.vier_zeilen(T, an, bn, k) == pytest.approx(stapel.plan[0], rel=1e-14)
    assert S.plan_nach(T, an, bn, 0) == pytest.approx(T)
    import inspect

    quelle = inspect.getsource(S.vier_zeilen)
    assert "u = a / (K @ v)" in quelle and "v = b / (K.T @ u)" in quelle


def test_u_v_form_und_matrix_form_geben_bei_zulaessigen_aufgaben_dasselbe(hand):
    T, a, b = hand
    p1 = S.abgleichen(T, a, b, form=C.FORM_UV)
    p2 = S.abgleichen(T, a, b, form=C.FORM_MATRIX)
    assert p1.durchlaeufe == p2.durchlaeufe and p1.plan == pytest.approx(p2.plan, rel=1e-12)


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 2: Summen passen nicht
# ----------------------------------------------------------------------------------------------------------------------------------

def test_handbeispiel_2_zweier_zyklus_und_zeilenfehler_gleich_differenz_der_summen():
    Q = [[F(2), F(3)], [F(3), F(3)]]
    Q2 = [[2 * x for x in r] for r in Q]
    assert schritt_zeilen(Q2, [F(5), F(6)]) == Q                                   # 2Q wird zu Q
    assert schritt_spalten(Q, [F(10), F(12)]) == Q2                                # Q wird zu 2Q
    assert sum(abs(z - t) for z, t in zip([sum(r) for r in Q2], [5, 6])) == 11     # |B − A| = 22 − 11
    T = np.array([[1.0, 2.0], [3.0, 4.0]])
    ab = S.abgleichen(T, [5, 6], [10, 12], form=C.FORM_MATRIX, max_durchlaeufe=200)
    assert ab.status == S.STATUS_GRENZE
    assert ab.fehler_verlauf[-1] * 11 == pytest.approx(11.0, abs=1e-9)
    assert ab.plan == pytest.approx(2 * np.array([[2.0, 3.0], [3.0, 3.0]]), abs=1e-9)


def test_verschiedene_summen_konvergieren_nie_und_der_fehler_bleibt_bei_der_differenz():
    rng = np.random.default_rng(3)
    for delta in (0.001, 0.01, 0.1):
        K = rng.uniform(0.5, 2.0, (6, 6))
        a = rng.dirichlet(np.ones(6))
        b = rng.dirichlet(np.ones(6)) * (1.0 + delta)
        ab = S.abgleichen(K, a, b, max_durchlaeufe=300)
        assert ab.status == S.STATUS_GRENZE and ab.fehler_verlauf[-1] == pytest.approx(delta, rel=1e-6)


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 3: Nullen
# ----------------------------------------------------------------------------------------------------------------------------------

K0 = np.array([[1.0, 1.0], [0.0, 1.0]])


def test_handbeispiel_3a_konvergiert_gegen_die_loesung_mit_nullen():
    ab = S.abgleichen(K0, [4, 6], [3, 7])
    assert ab.status == S.STATUS_OK
    assert ab.plan == pytest.approx(np.array([[3.0, 1.0], [0.0, 6.0]]), abs=1e-7)
    assert ab.plan[1, 0] == 0.0                                                    # Nullen bleiben Nullen


def test_handbeispiel_3b_gleiche_summe_aber_keine_loesung_beide_formen_konvergieren_nicht():
    uv = S.abgleichen(K0, [4, 6], [5, 5], form=C.FORM_UV, max_durchlaeufe=20000)
    assert uv.status == S.STATUS_NAN and 1000 < uv.durchlaeufe < 20000             # Plan: 3 176 in dieser Rechnung (Band)
    mat = S.abgleichen(K0, [4, 6], [5, 5], form=C.FORM_MATRIX, max_durchlaeufe=20000)
    assert mat.status == S.STATUS_GRENZE and np.isfinite(mat.plan).all()
    assert mat.fehler_verlauf[-1] * 10 == pytest.approx(2.0, abs=1e-3)             # Zeilenfehler nach 20 000 Durchläufen etwa 2 (Fahrten)


def test_handbeispiel_3c_randfehler_ist_genau_zwei_durch_zwei_k_plus_eins():
    M = bruch_matrix([[1, 1], [0, 1]])
    ab = S.abgleichen(K0, [1, 1], [1, 1], max_durchlaeufe=40)
    assert ab.status == S.STATUS_GRENZE
    for k in range(1, 31):
        M = schritt_zeilen(M, [F(1), F(1)])
        M = schritt_spalten(M, [F(1), F(1)])
        assert M[0][1] == F(1, 2 * k + 1)
        fehler = sum(abs(sum(r) - 1) for r in M)
        assert fehler == F(2, 2 * k + 1)
        assert ab.fehler_verlauf[k - 1] * 2 == pytest.approx(float(fehler), rel=1e-9)     # Gesamtmasse 2 (Fahrten)
    assert F(2, 2 * 10 ** 9 + 1) < F(1, 10 ** 9) < F(2, 2 * 10 ** 9 - 1)         # in Fahrten (Gesamtmasse 2): unter 1e-9 erst nach etwa 10^9 Durchläufen
    assert F(1, 2 * (5 * 10 ** 8) + 1) < F(1, 10 ** 9) <= F(1, 2 * (5 * 10 ** 8 - 1) + 1)   # bezogen auf die Gesamtmasse 1 wie beim Abbruch: 1/(2k + 1) < 1e-9 ab k = 5·10^8


# ----------------------------------------------------------------------------------------------------------------------------------
# Randfälle
# ----------------------------------------------------------------------------------------------------------------------------------

def test_n_gleich_1_und_n_gleich_2():
    ab = S.abgleichen(np.array([[2.0]]), [3.0], [3.0])
    assert ab.status == S.STATUS_OK and ab.plan[0, 0] == pytest.approx(3.0)
    ab = S.abgleichen(np.array([[1.0, 1.0], [1.0, 1.0]]), [1.0, 3.0], [2.0, 2.0])
    assert ab.status == S.STATUS_OK and ab.plan == pytest.approx(np.array([[0.5, 0.5], [1.5, 1.5]]))


def test_abgleich_auf_die_eigenen_randsummen_aendert_nichts():
    T = np.array([[1.0, 2.0], [3.0, 4.0]])
    ab = S.abgleichen(T, T.sum(1), T.sum(0))
    assert ab.status == S.STATUS_OK and ab.durchlaeufe <= 1 and ab.plan == pytest.approx(T, rel=1e-12)


def test_zeile_komplett_null_wird_gemeldet_nicht_still_beendet():
    K = np.array([[0.0, 0.0], [1.0, 1.0]])
    ab = S.abgleichen(K, [1.0, 1.0], [1.0, 1.0])
    assert ab.status != S.STATUS_OK                                                # Zeilensumme 0 bei Ziel 1: keine Lösung, Meldung statt Ende ohne Worte


def test_randsumme_null_ist_erlaubt_die_zeile_bleibt_leer():
    ab = S.abgleichen(np.ones((2, 2)), [0.0, 1.0], [0.5, 0.5])
    assert ab.status == S.STATUS_OK and ab.plan[0].sum() == 0.0 and ab.plan[1] == pytest.approx([0.5, 0.5])


def test_stapel_und_einzelinstanz_stimmen_ueberein():
    rng = np.random.default_rng(5)
    K = rng.uniform(0.2, 2.0, (4, 5, 5))
    a = rng.dirichlet(np.ones(5), 4)
    b = rng.dirichlet(np.ones(5), 4)
    st = S.abgleich_stapel(K, a, b)
    for i in range(4):
        e = S.abgleichen(K[i], a[i], b[i])
        assert e.durchlaeufe == st.durchlaeufe[i] and e.plan == pytest.approx(st.plan[i], rel=1e-7)   # der Stapel rechnet nach der Konvergenz einer Instanz weiter (unter der Toleranz)


def test_worst_case_obergrenze_wird_gemeldet_und_dauert_nicht_ewig():
    rng = np.random.default_rng(8)
    K = np.exp(-rng.uniform(0, 1, (200, 12, 12)) / 0.01)                           # sehr kleines ε: viele Durchläufe
    a = rng.dirichlet(np.ones(12), 200)
    b = rng.dirichlet(np.ones(12), 200)
    res = S.abgleich_stapel(K, a, b, max_durchlaeufe=300)
    assert set(np.unique(res.status)) <= {0, 1, 2} and (res.status == 1).any()
    assert (res.durchlaeufe[res.status != 0] == -1).all()


def test_strafe_mit_grossem_tau_ist_nahe_am_abgleich_und_bei_kleinem_tau_weicht_sie_ab():
    rng = np.random.default_rng(11)
    K = rng.uniform(0.5, 2.0, (1, 5, 5))
    a = rng.dirichlet(np.ones(5))[None]
    b = rng.dirichlet(np.ones(5))[None]
    exakt = S.abgleich_stapel(K, a, b, tol=1e-13).plan
    gross, it = S.strafe_stapel(K, a, b, 1e3)
    klein, _ = S.strafe_stapel(K, a, b, 1.0)
    assert np.abs(gross - exakt).max() / exakt.max() < 2e-3 and it[0] > 0
    assert np.abs(klein - exakt).max() / exakt.max() > 5 * np.abs(gross - exakt).max() / exakt.max()


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 5: Randsummen stimmen, die Matrix nicht
# ----------------------------------------------------------------------------------------------------------------------------------

def test_handbeispiel_5_ein_sechstel_in_der_falschen_zelle_und_rang_2n_minus_1():
    T = [[F(4), F(2)], [F(2), F(4)]]
    eins = [[F(1), F(1)], [F(1), F(1)]]
    P = schritt_spalten(schritt_zeilen(eins, [F(6), F(6)]), [F(6), F(6)])
    assert P == [[F(3), F(3)], [F(3), F(3)]]                                                    # ein Durchlauf der Einsmatrix
    assert [sum(r) for r in P] == [sum(r) for r in T] == [6, 6] and [sum(c) for c in zip(*P)] == [sum(c) for c in zip(*T)] == [6, 6]
    assert kreuzverhaeltnis(T) == 4 and kreuzverhaeltnis(P) == 1
    assert F(1, 2) * sum(abs(P[i][j] - T[i][j]) for i in range(2) for j in range(2)) / 12 == F(1, 6)
    ab = S.abgleichen(np.ones((2, 2)), [6, 6], [6, 6])
    assert ab.plan == pytest.approx(np.full((2, 2), 3.0)) and ab.durchlaeufe <= 2
    for n in range(2, 8):                                                                       # die 2n Randbedingungen haben Rang 2n − 1, es bleiben (n − 1)² freie Zahlen
        A = np.zeros((2 * n, n * n))
        for i in range(n):
            A[i, i * n:(i + 1) * n] = 1
            A[n + i, i::n] = 1
        assert np.linalg.matrix_rank(A) == 2 * n - 1 and n * n - (2 * n - 1) == (n - 1) ** 2


def test_unterlauf_des_kerns_bei_sehr_kleinem_epsilon_wird_gemeldet_oder_stimmt():
    """exp(−C/ε) mit ε = 5e-4 unterläuft zu Nullen (C/ε bis 2 000): der Abgleich meldet dann „nicht konvergiert“ und behauptet nie ein Ergebnis, dessen Randsummen nicht stimmen."""
    rng = np.random.default_rng(21)
    unterlaufen = 0
    for _ in range(12):
        pts = rng.uniform(size=(8, 2))
        Cm = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        Cm = Cm / Cm.max()
        K = S.kern_gravitation(Cm, 5e-4)
        assert (K == 0).any()
        a, b = rng.dirichlet(np.ones(8)), rng.dirichlet(np.ones(8))
        ab = S.abgleichen(K, a, b, max_durchlaeufe=2000)
        if ab.status == S.STATUS_OK:
            assert ab.plan.sum(1) == pytest.approx(a, abs=1e-8) and ab.plan.sum(0) == pytest.approx(b, abs=1e-8)
        else:
            unterlaufen += 1
    assert unterlaufen >= 6                                                                  # die meisten Läufe scheitern; keiner endet still mit falschen Randsummen
