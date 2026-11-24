"""Zulässigkeit: die Hall-Bedingung gegen ein lineares Programm, gegen die Aufzählung aller Träger (n = 3) und gegen die Aufzählung der Permutationen (positive Diagonale, totaler Träger)."""

import itertools

import numpy as np
import pytest

import mab_feasibility as F
from mab_scenario import dirichlet, instanz_rng, traeger
from orakel_hilfe import klasse_lp


def test_handbeispiel_3_klassen_gegen_das_lp_und_die_hall_bedingung():
    S0 = np.array([[1, 1], [0, 1]], bool)
    assert F.klasse(S0, [0.4, 0.6], [0.3, 0.7]) == klasse_lp(S0, [0.4, 0.6], [0.3, 0.7]) == 2     # (a) Lösung [[3,1],[0,6]]
    assert F.klasse(S0, [0.4, 0.6], [0.5, 0.5]) == klasse_lp(S0, [0.4, 0.6], [0.5, 0.5]) == 0     # (b) gleiche Summe, keine Lösung
    assert F.klasse(S0, [0.5, 0.5], [0.5, 0.5]) == klasse_lp(S0, [0.5, 0.5], [0.5, 0.5]) == 1     # (c) nur die Einheitsmatrix
    assert F.klasse(np.ones((2, 2), bool), [0.5, 0.5], [0.5, 0.5]) == 2                           # dichte Matrix
    assert F.klasse(S0, [0.4, 0.6], [0.5, 0.6]) == 0                                              # verschiedene Summen: Klasse 0


def _zufaellige_aufgabe(g, n, art):
    pat = traeger(n, 0.2 + 0.5 * g.random(), g)
    if art == 0:
        a, b = dirichlet(g, n, 1), dirichlet(g, n, 1)
    elif art == 1:
        a = b = np.full(n, 1.0 / n)
    else:                                                    # ganzzahlige Randsummen aus einer kleinen Zerlegung: erzeugt oft Gleichheiten der Hall-Bedingung
        z = np.array([1 + g.below(3) for _ in range(n)], float)
        w = np.array([1 + g.below(3) for _ in range(n)], float)
        w = w * z.sum() / w.sum()
        a, b = z / z.sum(), w / w.sum()
    return pat, a, b


def test_hall_klassen_gleich_lp_klassen_auf_vierhundert_zufallsaufgaben_alle_drei_klassen_kommen_vor():
    g = instanz_rng(1, 90, 0)
    zaehl = {0: 0, 1: 0, 2: 0}
    for t in range(400):
        n = 2 + t % 4
        pat, a, b = _zufaellige_aufgabe(g, n, t % 3)
        k1 = F.klasse(pat, a, b)
        k2 = klasse_lp(pat, a, b)
        assert k1 == k2, (t, n, pat.astype(int).tolist(), a, b)
        zaehl[k1] += 1
    assert min(zaehl.values()) >= 10, zaehl


def test_alle_512_traeger_fuer_n_gleich_3_gegen_das_lp():
    a = np.array([0.2, 0.3, 0.5])
    b = np.array([0.4, 0.4, 0.2])
    for paar in (None, "gleich"):
        for bits in range(512):
            pat = np.array([(bits >> k) & 1 for k in range(9)], bool).reshape(3, 3)
            aa, bb = (np.full(3, 1 / 3), np.full(3, 1 / 3)) if paar else (a, b)
            assert F.klasse(pat, aa, bb) == klasse_lp(pat, aa, bb), (bits, paar)


def _permutationen(S):
    n = S.shape[0]
    return [p for p in itertools.permutations(range(n)) if all(S[i, p[i]] for i in range(n))]


def test_traeger_klasse_gleich_aufzaehlung_der_permutationen_und_gleich_hall_bei_gleichen_randsummen():
    """Positive Diagonale = es gibt eine Permutation im Träger; totaler Träger = jede Zelle des Trägers liegt auf einer; für Randsummen 1/n dieselben Klassen wie die Hall-Bedingung."""
    g = instanz_rng(2, 91, 0)
    vorkommen = set()
    muster = [np.array([(b >> k) & 1 for k in range(9)], bool).reshape(3, 3) for b in range(512)]
    muster += [traeger(4, 0.2 + 0.5 * g.random(), g) for _ in range(300)]
    for pat in muster:
        n = pat.shape[0]
        perms = _permutationen(pat)
        erwartet = 0 if not perms else (2 if all(any(p[i] == j for p in perms) for i in range(n) for j in range(n) if pat[i, j]) else 1)
        # Zeilen oder Spalten ohne Eintrag: keine Permutation, Klasse 0
        assert F.traeger_klasse(pat) == erwartet
        a = np.full(n, 1.0 / n)
        assert F.klasse(pat, a, a) == erwartet
        vorkommen.add(erwartet)
    assert vorkommen == {0, 1, 2}


def test_klassen_stapel_gibt_je_instanz_eine_klasse():
    S0 = np.array([[1, 1], [0, 1]], bool)
    k = F.klassen_stapel(np.array([S0, S0, S0]), np.array([[0.4, 0.6], [0.4, 0.6], [0.5, 0.5]]), np.array([[0.3, 0.7], [0.5, 0.5], [0.5, 0.5]]))
    assert k.tolist() == [2, 0, 1]


def test_randsumme_null_auf_einer_zeile_mit_eintraegen_ist_hoechstens_klasse_1():
    assert F.klasse(np.ones((2, 2), bool), [0.0, 1.0], [0.5, 0.5]) == 1 == klasse_lp(np.ones((2, 2), bool), [0.0, 1.0], [0.5, 0.5])
