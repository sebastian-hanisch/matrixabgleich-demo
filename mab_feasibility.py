"""Zulässigkeit: hat die Aufgabe „alte Matrix mit Nullen, neue Randsummen“ überhaupt eine Lösung? (nur numpy, ohne Streamlit)

Gegeben ein Träger S (die Zellen, die ungleich null sein dürfen) und Randsummen a (Zeilen) und b (Spalten) mit gleicher Summe. Drei Klassen:
  Klasse 0: keine Matrix auf dem Träger hat diese Randsummen (der Abgleich konvergiert nie),
  Klasse 1: es gibt eine, aber keine, die auf dem ganzen Träger streng positiv ist (der Randfehler fällt nur wie 1/k),
  Klasse 2: eine auf dem ganzen Träger streng positive Lösung existiert (der Abgleich konvergiert linear).
Der Test ist die Hall-Bedingung über alle Zeilenmengen I: a(I) ≤ b(N(I)) mit N(I) = Spalten, die I berührt (für n ≤ 12 sind das höchstens 4095 Mengen; exakt).
Klasse 2 braucht zusätzlich: gilt für eine Menge I Gleichheit, darf der Träger keine Zelle in (Zeilen außerhalb von I) × N(I) haben (Menon 1968; hier nach Idel 2016
gelesen). Randsummen mit Nullen und Zeilen mit Träger und Summe null liegen höchstens in Klasse 1.

Dazu der Sonderfall gleicher Randsummen 1/n: Klasse 0 = keine positive Diagonale (keine perfekte Zuordnung im Träger), Klasse 2 = totaler Träger (jede Zelle des Trägers liegt
auf einer perfekten Zuordnung), dazwischen Klasse 1. Beide Zählungen sind für gleiche Randsummen dieselben Klassen (Test)."""

import numpy as np

TOL_HALL = 1e-9


def _teilmengen_summen(werte: np.ndarray) -> np.ndarray:
    """s[mask] = Summe der Werte, deren Bit in mask gesetzt ist, für alle 2^len Masken."""
    s = np.zeros(1 << len(werte))
    for i, w in enumerate(werte):
        s[1 << i: 1 << (i + 1)] = s[: 1 << i] + w
    return s


def klasse(S: np.ndarray, a, b, tol: float = TOL_HALL) -> int:
    """Klasse 0, 1 oder 2 (siehe Kopf). Die Summen von a und b müssen gleich sein (sonst Klasse 0)."""
    S = np.asarray(S, bool)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = S.shape
    if abs(a.sum() - b.sum()) > tol * max(1.0, a.sum()):
        return 0
    spaltenmaske = np.array([sum(1 << j for j in range(m) if S[i, j]) for i in range(n)], dtype=np.int64)
    masken = np.arange(1 << n, dtype=np.int64)
    nachbarn = np.zeros(1 << n, dtype=np.int64)
    for i in range(n):
        nachbarn[1 << i: 1 << (i + 1)] = nachbarn[: 1 << i] | spaltenmaske[i]
    a_summe = _teilmengen_summen(a)
    b_summe = _teilmengen_summen(b)
    luecke = b_summe[nachbarn] - a_summe        # ≥ 0 verlangt die Hall-Bedingung
    if (luecke < -tol).any():
        return 0
    # Klasse 1: Gleichheit bei einer Menge I (nicht leer, nicht alle Zeilen), und der Träger hat eine Zelle in (Zeilen außerhalb I) × N(I)
    beruehrt = np.zeros(1 << n, dtype=bool)
    for i in range(n):
        ausserhalb = ((masken >> i) & 1) == 0
        beruehrt |= ausserhalb & ((spaltenmaske[i] & nachbarn) != 0)
    gleich = (np.abs(luecke) <= tol) & beruehrt
    gleich[0] = False
    gleich[-1] = False
    if gleich.any():
        return 1
    # Zeilen oder Spalten mit Träger, aber Summe null, erzwingen eine Nullzelle
    if ((a <= tol) & S.any(axis=1)).any() or ((b <= tol) & S.any(axis=0)).any():
        return 1
    return 2


def _zuordnung(adj: list[list[int]], n_rechts: int, ausgeschlossen_zeile: int = -1, ausgeschlossen_spalte: int = -1) -> int:
    """Größe einer größten Zuordnung (Kuhn), wobei eine Zeile und eine Spalte ausgelassen werden können."""
    partner = [-1] * n_rechts

    def versuche(i, besucht):
        for j in adj[i]:
            if j == ausgeschlossen_spalte or besucht[j]:
                continue
            besucht[j] = True
            if partner[j] < 0 or versuche(partner[j], besucht):
                partner[j] = i
                return True
        return False

    gross = 0
    for i in range(len(adj)):
        if i == ausgeschlossen_zeile:
            continue
        if versuche(i, [False] * n_rechts):
            gross += 1
    return gross


def traeger_klasse(S: np.ndarray) -> int:
    """Für gleiche Randsummen 1/n: 0 = keine positive Diagonale, 1 = positive Diagonale ohne totalen Träger, 2 = totaler Träger."""
    S = np.asarray(S, bool)
    n = S.shape[0]
    adj = [[int(j) for j in np.flatnonzero(S[i])] for i in range(n)]
    if _zuordnung(adj, n) < n:
        return 0
    for i in range(n):
        for j in adj[i]:
            if _zuordnung(adj, n, i, j) < n - 1:
                return 1
    return 2


def klassen_stapel(S: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Klassen für N Instanzen (N × n × n Träger, N × n Randsummen)."""
    return np.array([klasse(S[k], a[k], b[k]) for k in range(len(S))])
