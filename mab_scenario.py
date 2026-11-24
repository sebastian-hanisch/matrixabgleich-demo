"""Szenario: Zonen, alte Matrizen und neue Randsummen aus dem Hausgenerator SplitMix64 (`mab_rng.py`).

Alles ist deterministisch aus (Zufalls-Seed, Zweck, Index): gleiche Eingabe, gleiche Instanz, auf jeder Plattform (bis auf die letzte Binärstelle von `math.exp`).

Zonen: n Mittelpunkte gleichverteilt im Einheitsquadrat; Kosten C = Abstand / größter Abstand (max C = 1); Herkunftssummen a_i ganzzahlig 10 bis 99, Zielsummen
b_j = 1 + Multinomial(S − n; Dirichlet(3)) mit S = Σ a_i; beide durch S geteilt (Gesamtmasse 1).
Alte Matrizen: (i) dichte Basismatrix T⁰_ij = exp(−C_ij/3) · U(0,5; 1,5); (ii) Poisson-Zählung mit λ_ij = 90 · n · exp(−C_ij/3) · a_i · a_j (natürliche Nullen);
(iii) zufälliger Träger mit gegebenem Nullanteil (jede Zeile und jede Spalte hat mindestens einen Eintrag), Einträge U(0,5; 2).
Randsummen: beliebig (je Seite Dirichlet(1)), aus Wachstumsfaktoren der alten Matrix (je Zeile und Spalte ein unabhängiger Faktor, Spaltenziele auf die Summe der
Zeilenziele skaliert) oder gleich (1/n)."""

import math
from dataclasses import dataclass

import numpy as np

import mab_constants as C
from mab_rng import MASK, SplitMix64


def instanz_rng(seed: int, zweck: int, index: int) -> SplitMix64:
    """Eigener Zufallsstrom je (Seed, Zweck, Index); die ersten Zahlen werden verworfen, damit benachbarte Indizes unabhängig wirken."""
    g = SplitMix64((int(seed) * 1_000_003 + int(zweck) * 7_919 + int(index) * 104_729 + 12_345) & MASK)
    g.next_u64()
    g.next_u64()
    return g


def dirichlet(g: SplitMix64, n: int, form: int) -> np.ndarray:
    """Dirichlet(form, ..., form) für ganzzahliges form: normierte Summen von `form` Exponentialzahlen je Eintrag."""
    e = np.array([sum(g.expovariate(1.0) for _ in range(form)) for _ in range(n)])
    return e / e.sum()


def gleichverteilt(g: SplitMix64, lo: float, hi: float, shape) -> np.ndarray:
    z = np.array([g.random() for _ in range(int(np.prod(shape)))]).reshape(shape)
    return lo + (hi - lo) * z


@dataclass(frozen=True)
class Zonen:
    kosten: np.ndarray       # n × n, größter Wert 1
    a: np.ndarray            # Herkunftssummen, Summe 1
    b: np.ndarray            # Zielsummen, Summe 1
    punkte: np.ndarray       # n × 2
    gesamt: int              # S: Summe der ganzzahligen Herkunftssummen

    @property
    def n(self) -> int:
        return len(self.a)


def zonen(n: int, g: SplitMix64) -> Zonen:
    """Zonen wie oben: Reihenfolge der Ziehungen Punkte, Herkunftssummen, Dirichlet(3), Zielsummen."""
    punkte = np.array([[g.random(), g.random()] for _ in range(n)])
    d = np.sqrt(((punkte[:, None, :] - punkte[None, :, :]) ** 2).sum(-1))
    kosten = d / d.max()
    a_int = np.array([10 + g.below(90) for _ in range(n)], dtype=float)
    S = int(a_int.sum())
    w = dirichlet(g, n, 3)
    cum = np.cumsum(w)
    cum[-1] = 1.0
    ziehungen = np.array([g.random() for _ in range(S - n)])
    zaehl = np.bincount(np.searchsorted(cum, ziehungen, side="right").clip(0, n - 1), minlength=n)
    b_int = 1.0 + zaehl
    return Zonen(kosten, a_int / S, b_int / S, punkte, S)


def basis_matrix(kosten: np.ndarray, g: SplitMix64) -> np.ndarray:
    """(i) Dichte Basismatrix: exp(−C/3) mal Rauschfaktor U(0,5; 1,5) je Zelle."""
    return np.exp(-kosten / C.GRAVITATION_VOR) * gleichverteilt(g, C.RAUSCH_LO, C.RAUSCH_HI, kosten.shape)


def _traeger_reparieren(pat: np.ndarray, g: SplitMix64) -> np.ndarray:
    n, m = pat.shape
    for i in range(n):
        if not pat[i].any():
            pat[i, g.below(m)] = True
    for j in range(m):
        if not pat[:, j].any():
            pat[g.below(n), j] = True
    return pat


def traeger(n: int, nullanteil: float, g: SplitMix64) -> np.ndarray:
    """Zufälliger Träger (bool): Zelle ist Nullzelle, wenn eine gleichverteilte Zahl unter dem Nullanteil liegt; leere Zeilen und Spalten bekommen einen Eintrag."""
    pat = gleichverteilt(g, 0.0, 1.0, (n, n)) >= nullanteil
    return _traeger_reparieren(pat, g)


def traeger_matrix(n: int, nullanteil: float, g: SplitMix64) -> np.ndarray:
    """(iii) Matrix mit Nullen: Träger wie oben, Einträge U(0,5; 2)."""
    pat = traeger(n, nullanteil, g)
    werte = gleichverteilt(g, C.TRAEGER_LO, C.TRAEGER_HI, (n, n))
    return np.where(pat, werte, 0.0)


def poisson(g: SplitMix64, lam: float) -> int:
    """Poisson-Zufallszahl durch Inversion (kleine λ)."""
    u = g.random()
    k = 0
    p = math.exp(-lam)
    while u > p and k < 10_000:
        u -= p
        k += 1
        p *= lam / k
    return k


def poisson_matrix(z: Zonen, g: SplitMix64) -> np.ndarray:
    """(ii) Zählung: Poisson mit λ_ij = 90 · n · exp(−C/3) · a_i · a_j; leere Zeilen und Spalten bekommen eine 1."""
    n = z.n
    lam = C.POISSON_FAKTOR * n * np.exp(-z.kosten / C.GRAVITATION_VOR) * np.outer(z.a, z.a)
    T = np.array([[poisson(g, float(lam[i, j])) for j in range(n)] for i in range(n)], dtype=float)
    pat = _traeger_reparieren(T > 0, g)
    return np.where(pat & (T == 0), 1.0, T)


def randsummen_beliebig(g: SplitMix64, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Zeilen- und Spaltenziele je Dirichlet(1), Summe 1."""
    return dirichlet(g, n, 1), dirichlet(g, n, 1)


def randsummen_wachstum(T0: np.ndarray, g: SplitMix64, breite: tuple[float, float], mit_summe: bool = False):
    """Zeilen- und Spaltensummen der alten Matrix mal unabhängige Faktoren U(breite), Spaltenziele auf die Summe der Zeilenziele skaliert; Gesamtmasse 1.
    Mit `mit_summe` kommt als drittes Ergebnis die Summe der Zeilenziele vor der Normierung (der Maßstab der alten Matrix)."""
    n, m = T0.shape
    f = gleichverteilt(g, breite[0], breite[1], (n,))
    h = gleichverteilt(g, breite[0], breite[1], (m,))
    a = T0.sum(1) * f
    b = T0.sum(0) * h
    b = b * a.sum() / b.sum()
    if mit_summe:
        return a / a.sum(), b / b.sum(), float(a.sum())
    return a / a.sum(), b / b.sum()


def randsummen(art: str, T0: np.ndarray, g: SplitMix64) -> tuple[np.ndarray, np.ndarray]:
    n = T0.shape[0]
    if art == C.RS_BELIEBIG:
        return randsummen_beliebig(g, n)
    if art == C.RS_GLEICH:
        return np.full(n, 1.0 / n), np.full(n, 1.0 / n)
    return randsummen_wachstum(T0, g, C.RS_BREITE[art])


# ------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel und Primäransicht
# ------------------------------------------------------------------------------------------------------------------------------------

HAND_T0 = np.array([[1.0, 2.0], [3.0, 4.0]])
HAND_A = np.array([5.0, 6.0])
HAND_B = np.array([5.0, 6.0])


@dataclass(frozen=True)
class Ansicht:
    """Die Aufgabe der Primäransicht in Fahrten (Anzeige); gerechnet wird mit Gesamtmasse 1."""
    quelle: str
    T0: np.ndarray            # alte Matrix in Fahrten
    a: np.ndarray             # Zeilenziele in Fahrten
    b: np.ndarray             # Spaltenziele in Fahrten (kann eine andere Summe haben als a)
    kosten: np.ndarray | None  # nur Zonen
    punkte: np.ndarray | None

    @property
    def n(self) -> int:
        return len(self.a)


def ansicht(quelle: str, n: int, seed: int, nullanteil: float, delta: float, rs: str = C.RS_MILD, gravitation: bool = False) -> Ansicht:
    """Alte Matrix und neue Randsummen der Primäransicht; delta macht die Spaltenziele um den Anteil delta größer als die Zeilenziele."""
    if quelle == C.QUELLE_HAND:
        return Ansicht(quelle, HAND_T0.copy(), HAND_A.copy(), HAND_B * (1.0 + delta), None, None)
    g = instanz_rng(seed, C.ZWECK_ANSICHT, 0)
    z = zonen(n, g)
    T0 = traeger_matrix(n, nullanteil, g) if nullanteil > 0 else basis_matrix(z.kosten, g)
    T0 = T0 / T0.sum() * C.GESAMT_FAHRTEN
    if gravitation:     # doppelt beschränktes Gravitationsmodell: die Randsummen der Zonen selbst (ganzzahlige Fahrten)
        S = float(z.gesamt)
        return Ansicht(quelle, T0, z.a * S, z.b * S * (1.0 + delta), z.kosten, z.punkte)
    a, b = randsummen(rs, T0, g)
    A = float(T0.sum())
    return Ansicht(quelle, T0, a * A, b * A * (1.0 + delta), z.kosten, z.punkte)
