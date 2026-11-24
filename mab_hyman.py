"""ε aus einer Beobachtung zurückrechnen: die Schleife der Verkehrsplanung (Sekantenverfahren auf β = 1/ε), als Stapel über Instanzen (nur numpy, ohne Streamlit).

Aufgabe: beobachtet wird nur die mittlere Reiseweite c_obs = ⟨C, T⟩ (Gesamtmasse 1, max C = 1). Gesucht ist β = 1/ε, sodass der abgeglichene Plan mit Kern exp(−β C) und
den bekannten Randsummen diese mittlere Reiseweite hat. Jeder **Schritt** wertet den Abgleich einmal vollständig aus (ein innerer Lauf bis 1e-10); **Durchläufe** sind die
Summe aller inneren Durchläufe, Matrix-Vektor-Produkte das Doppelte.

Startwert β₀ = 1/c_obs und erster Schritt (β·c/c_obs oder 1,5·β) sind eigene Wahl, nicht aus der Quelle (Hyman 1969) gelesen; danach Sekante, bei unbrauchbarem
Schritt (nicht endlich oder ≤ 0) wird β halbiert. Abbruch bei relativer Abweichung der Reiseweite unter 1e-6. Die Schleife lässt sich mit dem Startwert des inneren Abgleichs
„warm“ (v vom Schritt davor) oder „kalt“ (v = 1) rechnen."""

from dataclasses import dataclass

import numpy as np

import mab_constants as C
from mab_scaling import abgleich_stapel

ST_OK, ST_INNEN, ST_GRENZE = 0, 1, 2
STATUS_TEXT = {ST_OK: "konvergiert", ST_INNEN: "innerer Abgleich gescheitert (Unterlauf mit NaN oder Obergrenze der inneren Durchläufe)", ST_GRENZE: "Schrittgrenze erreicht"}


@dataclass(frozen=True)
class HymanErgebnis:
    schritte: np.ndarray       # N
    durchlaeufe: np.ndarray    # N, Summe aller inneren Durchläufe
    beta: np.ndarray           # N, β am Ende
    reiseweite: np.ndarray     # N, mittlere Reiseweite am Ende
    status: np.ndarray         # N, ST_OK / ST_INNEN / ST_GRENZE
    beta_verlauf: np.ndarray   # Schritte × N (NaN nach dem Ende der Instanz)
    weite_verlauf: np.ndarray  # Schritte × N
    innen_nan: np.ndarray = None  # N, wahr, wenn der innere Abgleich durch NaN (Summe null) scheiterte; falsch bei Obergrenze der inneren Durchläufe

    @property
    def produkte(self) -> np.ndarray:
        return 2 * self.durchlaeufe


def hyman_stapel(kosten, a, b, c_obs, start=C.START_PROP, warm=False, max_schritte=C.MAX_SCHRITTE_HYMAN, max_innen=C.MAX_INNEN_HYMAN) -> HymanErgebnis:
    """Stapel: kosten N × n × n, a und b N × n (Summe 1), c_obs N."""
    kosten = np.asarray(kosten, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c_obs = np.asarray(c_obs, float)
    N, n, m = kosten.shape
    beta = 1.0 / c_obs
    beta_alt = np.full(N, np.nan)
    f_alt = np.full(N, np.nan)
    lebt = np.ones(N, bool)
    schritte = np.zeros(N, int)
    summe = np.zeros(N, int)
    status = np.full(N, ST_GRENZE)
    weite = np.full(N, np.nan)
    innen_nan = np.zeros(N, bool)
    v = np.ones((N, m))
    bv = np.full((max_schritte, N), np.nan)
    wv = np.full((max_schritte, N), np.nan)
    for schritt in range(1, max_schritte + 1):
        idx = np.flatnonzero(lebt)
        if idx.size == 0:
            break
        with np.errstate(all="ignore"):
            K = np.exp(-beta[idx, None, None] * kosten[idx])
        res = abgleich_stapel(K, a[idx], b[idx], tol=C.TOL_HYMAN_INNEN, max_durchlaeufe=max_innen, v0=v[idx] if warm else None)
        gerechnet = np.where(res.status == 0, res.durchlaeufe, np.where(res.status == 2, res.nan_ab, max_innen))
        summe[idx] += gerechnet
        schritte[idx] = schritt
        bv[schritt - 1, idx] = beta[idx]
        gescheitert = res.status != 0
        status[idx[gescheitert]] = ST_INNEN
        innen_nan[idx[res.status == 2]] = True
        lebt[idx[gescheitert]] = False
        gut = ~gescheitert
        gi = idx[gut]
        if gi.size == 0:
            continue
        v[gi] = res.v[gut]
        c = (kosten[gi] * res.plan[gut]).sum(axis=(1, 2))
        weite[gi] = c
        wv[schritt - 1, gi] = c
        f = c - c_obs[gi]
        fertig = np.abs(f) / c_obs[gi] < C.TOL_HYMAN_REL
        status[gi[fertig]] = ST_OK
        lebt[gi[fertig]] = False
        weiter = gi[~fertig]
        if weiter.size == 0:
            continue
        cw, fw = c[~fertig], f[~fertig]
        with np.errstate(all="ignore"):
            if schritt == 1:
                neu = beta[weiter] * (cw / c_obs[weiter] if start == C.START_PROP else 1.5)
            else:
                b0, f0 = beta_alt[weiter], f_alt[weiter]
                neu = np.where(fw != f0, beta[weiter] - fw * (beta[weiter] - b0) / (fw - f0), -1.0)
        schlecht = ~np.isfinite(neu) | (neu <= 0)
        neu = np.where(schlecht, beta[weiter] / 2, neu)
        beta_alt[weiter] = beta[weiter]
        f_alt[weiter] = fw
        beta[weiter] = neu
    return HymanErgebnis(schritte, summe, beta, weite, status, bv, wv, innen_nan)
