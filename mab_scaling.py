"""Der Abgleich einer Matrix auf neue Randsummen: die Rechnung, die in vier Fächern vier Namen hat (nur numpy, ohne Streamlit).

Furness-Verfahren (Verkehrsplanung), RAS bzw. biproportionales Verfahren (Volkswirtschaft), Raking bzw. iterative proportionale Anpassung (IPF, Umfragestatistik),
Sinkhorn-Knopp (Mathematik): abwechselnd Zeilen und Spalten anpassen. Die Gleichheit der Rechenvorschrift ist trivial; die Demo zählt, was darum herum geschieht.

Zählweise: ein **Durchlauf** (sweep) = Zeilenschritt + Spaltenschritt = zwei Matrix-Vektor-Produkte (K v und Kᵀ u). Die Prüfung des Randfehlers nach einem Durchlauf
verwendet das K v des nächsten Durchlaufs und kostet kein drittes Produkt; die Prüfung nach dem letzten Durchlauf braucht noch ein K v, das nicht mitgezählt wird.
Abbruch: Zeilenfehler (L1, Summe der Zeilenziele = 1) unter TOL in float64, Obergrenze MAX_DURCHLAEUFE.

Zwei gleichwertige Rechenformen: die u-v-Form (u = a / (K v), v = b / (Kᵀ u); sie zählt die Produkte) und die Matrix-Form (die Matrix selbst wird zeilen- und
spaltenweise skaliert). Bei unzulässigen Aufgaben verhalten sie sich verschieden: die u-v-Form endet oft in NaN (Teilung durch null), die Matrix-Form behält einen
endlichen Fehler; beide konvergieren nicht, und die App meldet es.

Alle Reihen laufen als Stapel (Form N × n × m): ein Aufruf von einsum rechnet einen Durchlauf für alle Instanzen zugleich."""

from dataclasses import dataclass

import numpy as np

import mab_constants as C

STATUS_OK = "konvergiert"
STATUS_GRENZE = "nicht konvergiert: Obergrenze der Durchläufe erreicht"
STATUS_NAN = "nicht konvergiert: Summe wurde null (NaN)"


def vier_zeilen(K, a, b, durchlaeufe):
    """Furness, RAS, Raking und Sinkhorn-Knopp: die Schleife, die die App zeigt und ausführt (mindestens ein Durchlauf)."""
    v = np.ones(len(b))
    for _ in range(durchlaeufe):
        u = a / (K @ v)
        v = b / (K.T @ u)
    return u[:, None] * K * v[None, :]


def plan_nach(K, a, b, durchlaeufe):
    """Die Matrix nach `durchlaeufe` Durchläufen (bei 0 die Ausgangsmatrix K), gerechnet mit der gezeigten Schleife."""
    K = np.asarray(K, float)
    if durchlaeufe <= 0:
        return K.copy()
    with np.errstate(all="ignore"):
        return vier_zeilen(K, np.asarray(a, float), np.asarray(b, float), int(durchlaeufe))


@dataclass(frozen=True)
class AbgleichStapel:
    durchlaeufe: np.ndarray   # N, Zahl der Durchläufe bis zum Abbruch; bei Nichtkonvergenz -1
    status: np.ndarray        # N, 0 = konvergiert, 1 = Obergrenze, 2 = NaN
    fehler: np.ndarray        # N, letzter berechneter Zeilenfehler (NaN, wenn die Rechnung ungültig wurde)
    nan_ab: np.ndarray        # N, Durchlauf, nach dem der Fehler zum ersten Mal nicht endlich war (-1: nie)
    plan: np.ndarray          # N x n x m, Matrix nach dem letzten Durchlauf
    v: np.ndarray             # N x m (nur u-v-Form; sonst die Spaltenfaktoren 1)
    verlauf: list             # Zeilenfehler je Durchlauf (nur wenn verlangt): Liste von Feldern der Länge N
    punkte: dict              # Fehler bei vorgegebenen Durchläufen (checkpoints): {k: Feld der Länge N}

    @property
    def produkte(self) -> np.ndarray:
        """Matrix-Vektor-Produkte (2 je Durchlauf), -1 bei Nichtkonvergenz."""
        return np.where(self.durchlaeufe >= 0, 2 * self.durchlaeufe, -1)


def abgleich_stapel(K, a, b, tol=C.TOL, max_durchlaeufe=C.MAX_DURCHLAEUFE, form=C.FORM_UV, v0=None, verlauf=False, checkpoints=()) -> AbgleichStapel:
    """Gleicht N Matrizen K (N × n × m) auf die Randsummen a (N × n) und b (N × m) ab. Die Summe der Zeilenziele soll 1 sein (der Fehler ist dann relativ)."""
    K = np.asarray(K, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    N, n, m = K.shape
    erledigt = np.zeros(N, bool)
    kaputt = np.zeros(N, bool)
    durchlaeufe = np.full(N, -1)
    nan_ab = np.full(N, -1)
    fehler = np.full(N, np.inf)
    hist = []
    punkte = {}
    cps = set(int(c) for c in checkpoints)
    v = np.ones((N, m)) if v0 is None else np.array(v0, float)
    u = np.ones((N, n))
    T = K.copy()
    with np.errstate(all="ignore"):
        for k in range(0, max_durchlaeufe + 1):
            if form == C.FORM_UV:
                Kv = np.einsum("Nij,Nj->Ni", K, v)           # Produkt 1 des nächsten Durchlaufs; liefert zugleich die Zeilensummen des vorigen
                zeilen = u * Kv
            else:
                zeilen = T.sum(axis=2)
            if k > 0:
                err = np.abs(zeilen - a).sum(axis=1)
                if verlauf:
                    hist.append(err.copy())
                if k in cps:
                    punkte[k] = err.copy()
                neu_kaputt = ~np.isfinite(err) & ~kaputt & ~erledigt
                nan_ab[neu_kaputt] = k
                kaputt |= neu_kaputt
                fehler = np.where(erledigt, fehler, err)
                neu_fertig = (err < tol) & ~erledigt & ~kaputt
                durchlaeufe[neu_fertig] = k
                erledigt |= neu_fertig
                if (erledigt | kaputt).all() and not cps - set(punkte):
                    break
            if k == max_durchlaeufe:
                break
            if form == C.FORM_UV:
                u = a / Kv
                v = b / np.einsum("Nij,Ni->Nj", K, u)
            else:
                T = T * (a / T.sum(axis=2))[:, :, None]
                T = T * (b / T.sum(axis=1))[:, None, :]
    status = np.where(erledigt, 0, np.where(kaputt, 2, 1))
    if form == C.FORM_UV:
        with np.errstate(all="ignore"):
            plan = u[:, :, None] * K * v[:, None, :]
    else:
        plan = T
    return AbgleichStapel(durchlaeufe, status, fehler, nan_ab, plan, v, hist, punkte)


@dataclass(frozen=True)
class Abgleich:
    plan: np.ndarray
    durchlaeufe: int          # bei Nichtkonvergenz die Zahl der gerechneten Durchläufe (Obergrenze) bzw. der Durchlauf des NaN
    produkte: int             # 2 × Durchläufe
    status: str
    fehler_verlauf: np.ndarray   # Zeilenfehler nach Durchlauf 1, 2, ...
    v: np.ndarray


def abgleichen(K, a, b, tol=C.TOL, max_durchlaeufe=C.MAX_DURCHLAEUFE, form=C.FORM_UV, v0=None) -> Abgleich:
    """Eine Matrix abgleichen (Stapel der Größe 1). Die Summe der Zeilenziele wird auf 1 normiert (b mit demselben Faktor, ungleiche Summen bleiben ungleich)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    s = a.sum()
    res = abgleich_stapel(np.asarray(K, float)[None], (a / s)[None], (b / s)[None], tol, max_durchlaeufe, form, None if v0 is None else np.asarray(v0)[None], verlauf=True)
    verlauf = np.array([h[0] for h in res.verlauf])
    st = int(res.status[0])
    if st == 0:
        d = int(res.durchlaeufe[0])
        status = STATUS_OK
    elif st == 2:
        d = int(res.nan_ab[0])
        status = STATUS_NAN
    else:
        d = max_durchlaeufe
        status = STATUS_GRENZE
    return Abgleich(res.plan[0] * s, d, 2 * d, status, verlauf, res.v[0])


def kern_gravitation(C_mat, eps):
    """K = exp(−C/ε); C hat den größten Wert 1."""
    return np.exp(-np.asarray(C_mat, float) / eps)


def alte_matrix_als_kern(C_mat, eps):
    """Brücke: jede positive Matrix T ist ein Gibbs-Kern, T = exp(−C/ε) mit C = −ε ln T, für jedes ε."""
    return -eps * np.log(np.asarray(C_mat, float))


def strafe_stapel(K, a, b, tau, eps=C.STRAFE_EPS, tol=C.STRAFE_TOL, max_iter=C.STRAFE_MAX):
    """Skalierungsverfahren für die KL-Strafe statt harter Randsummen (Stufe G: aus dem Gedächtnis nach Chizat u. a. nachgebaut, nicht gegen die Quelle geprüft):
    minimiere ε KL(P|K) + τ KL(P1|a) + τ KL(Pᵀ1|b) mit u = (a / (K v))^ρ, v = (b / (Kᵀ u))^ρ, ρ = τ / (τ + ε).
    Stapel N × n × m; Abbruch, wenn die größte Änderung von ln u und ln v unter tol liegt. Gibt (Plan, Iterationen (−1: nicht konvergiert)) zurück."""
    K = np.asarray(K, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    N, n, m = K.shape
    rho = tau / (tau + eps)
    u = np.ones((N, n))
    v = np.ones((N, m))
    iter_ = np.full(N, -1)
    fertig = np.zeros(N, bool)
    with np.errstate(all="ignore"):
        for k in range(1, max_iter + 1):
            u_neu = (a / np.einsum("Nij,Nj->Ni", K, v)) ** rho
            v_neu = (b / np.einsum("Nij,Ni->Nj", K, u_neu)) ** rho
            dl = np.maximum(np.abs(np.log(u_neu / u)).max(axis=1), np.abs(np.log(v_neu / v)).max(axis=1))
            u, v = u_neu, v_neu
            neu = (dl < tol) & ~fertig
            iter_[neu] = k
            fertig |= neu
            if fertig.all():
                break
    return u[:, :, None] * K * v[:, None, :], iter_
