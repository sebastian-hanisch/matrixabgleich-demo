"""Kennzahlen, Reihen und Verdikte der Demo (reine Funktionen, ohne Streamlit).

Aufbau: die Primäransicht (`ansicht_ergebnis`), die Reihen je Experiment (`reihe_summen`, `reihe_strafe`, `reihe_nullen`, `reihe_kalibrierung`, `reihe_randsummen`) und die
Verdikte `verdikt_*` als Funktionen (Art, Text), die die App mit `getattr(st, Art)(Text)` anzeigt. Jede Zahl eines Verdikts stammt aus der Reihe, die es bekommt.

Zählweise: ein Durchlauf = zwei Matrix-Vektor-Produkte (siehe `mab_scaling`). Skala: ε/OPT mit OPT = LP-Minimum derselben Instanz bei max C = 1 und Gesamtmasse 1.
Alle Reihen laufen mit dem Zufalls-Seed `SERIEN_SEED`, nie mit dem Regler „Zufalls-Seed“; die Zufallsströme der Instanzen sind getrennt nach Zweck und Index."""

import math
from dataclasses import dataclass

import numpy as np

import mab_constants as C
from mab_feasibility import klasse, traeger_klasse
from mab_hyman import ST_GRENZE, ST_INNEN, ST_OK, hyman_stapel
from mab_scaling import STATUS_OK, abgleich_stapel, abgleichen, kern_gravitation, plan_nach, strafe_stapel
from mab_scenario import (Ansicht, ansicht, basis_matrix, instanz_rng, poisson_matrix, randsummen, randsummen_wachstum, traeger, traeger_matrix, zonen)
from mab_simplex import loese_transport


# ----------------------------------------------------------------------------------------------------------------------------------------
# Formatierung
# ----------------------------------------------------------------------------------------------------------------------------------------

def de(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{stellen}f}".replace(".", ",")


def de_g(x: float, signifikant: int = 3) -> str:
    """Zahl mit Dezimalkomma und höchstens `signifikant` gültigen Stellen (Exponent nur, wo es lesbar bleibt)."""
    if x == 0:
        return "0"
    text = f"{x:.{signifikant}g}"
    if "e" in text:
        mant, exp = text.split("e")
        return f"{mant.replace('.', ',')}e{int(exp)}"
    return text.replace(".", ",")


def ganz(x: float) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(round(x)):,}".replace(",", " ")


def band_ganz(b) -> str:
    """[Median, Minimum, Maximum] einer Zählung als „Median [Minimum; Maximum]“."""
    return f"{ganz(b[0])} [{ganz(b[1])}; {ganz(b[2])}]"


def band_g(b, sig: int = 3) -> str:
    return f"{de_g(b[0], sig)} [{de_g(b[1], sig)}; {de_g(b[2], sig)}]"


def prozent(x: float, stellen: int = 1) -> str:
    return de(100.0 * x, stellen) + " %"


def band(werte):
    """[Median, Minimum, Maximum] einer Liste (als Liste, damit die Werte unverändert durch JSON gehen)."""
    v = np.asarray(list(werte), float)
    if v.size == 0:
        return [float("nan")] * 3
    return [float(np.median(v)), float(v.min()), float(v.max())]


def jsonbar(x):
    """Wandelt Ergebnisse in reine Python-Typen um: dieselbe Gestalt wie nach dem Laden aus der JSON-Datei."""
    if isinstance(x, dict):
        return {str(k): jsonbar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonbar(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonbar(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ----------------------------------------------------------------------------------------------------------------------------------------
# Einstellungen
# ----------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    quelle: str = C.DEFAULTS["quelle"]
    n: int = C.DEFAULTS["n_slider"]
    seed: int = C.DEFAULTS["seed_input"]
    nullanteil: float = C.DEFAULTS["nullanteil"]
    rs_ansicht: str = C.DEFAULTS["rs_ansicht"]
    delta: float = C.DEFAULTS["delta"]
    kern: str = C.DEFAULTS["kern"]
    eps_rel: float = C.DEFAULTS["eps_rel"]
    k_durch: int = C.DEFAULTS["k_durch"]
    tau: str = C.DEFAULTS["tau"]
    nullanteil2: float = C.DEFAULTS["nullanteil2"]
    randsummen: str = C.DEFAULTS["randsummen"]
    inst2: int = C.DEFAULTS["inst2"]
    form: str = C.DEFAULTS["form"]
    n_mess: int = C.DEFAULTS["n_mess"]
    beobachtung: str = C.DEFAULTS["beobachtung"]
    theta: float = C.DEFAULTS["theta"]
    startregel: str = C.DEFAULTS["startregel"]
    warmstart: str = C.DEFAULTS["warmstart"]
    sigma: float = C.DEFAULTS["sigma"]
    familie: str = C.DEFAULTS["familie"]
    inst4: int = C.DEFAULTS["inst4"]


_ZUSTAND_ZU_FELD = {"quelle": "quelle", "n_slider": "n", "seed_input": "seed", "nullanteil": "nullanteil", "rs_ansicht": "rs_ansicht", "delta": "delta", "kern": "kern", "eps_rel": "eps_rel",
                    "k_durch": "k_durch", "tau": "tau", "nullanteil2": "nullanteil2", "randsummen": "randsummen", "inst2": "inst2", "form": "form", "n_mess": "n_mess",
                    "beobachtung": "beobachtung", "theta": "theta", "startregel": "startregel", "warmstart": "warmstart", "sigma": "sigma", "familie": "familie",
                    "inst4": "inst4"}


def settings_aus_zustand(zustand) -> Settings:
    """Settings aus dem Session State (Schlüssel der Regler); fehlende Schlüssel nehmen den Standardwert."""
    felder = {feld: zustand[key] for key, feld in _ZUSTAND_ZU_FELD.items() if key in zustand}
    for feld in ("n", "seed", "k_durch", "inst2", "n_mess", "inst4"):
        if feld in felder:
            felder[feld] = int(felder[feld])
    for feld in ("nullanteil", "delta", "eps_rel", "nullanteil2", "theta", "sigma"):
        if feld in felder:
            felder[feld] = float(felder[feld])
    return Settings(**felder)


# ----------------------------------------------------------------------------------------------------------------------------------------
# OPT und Instanzen der Reihen
# ----------------------------------------------------------------------------------------------------------------------------------------

def opt_wert(kosten, a, b) -> float:
    """LP-Minimum ⟨C, P⟩ bei den Randsummen a und b (eigener Transportsimplex; die Summen müssen übereinstimmen)."""
    return loese_transport(a, b, kosten, "steilste").kosten


def zonen_stapel(n: int, anzahl: int, zweck: int, seed: int = C.SERIEN_SEED):
    """Zonen für Reihen: (Zonen-Liste, Kosten N × n × n, a N × n, b N × n, OPT N, Produktmaß-Kosten N)."""
    zs = [zonen(n, instanz_rng(seed, zweck, i)) for i in range(anzahl)]
    kosten = np.array([z.kosten for z in zs])
    a = np.array([z.a for z in zs])
    b = np.array([z.b for z in zs])
    opt = np.array([opt_wert(z.kosten, z.a, z.b) for z in zs])
    prod = np.array([z.a @ z.kosten @ z.b for z in zs])
    return zs, kosten, a, b, opt, prod


# ----------------------------------------------------------------------------------------------------------------------------------------
# Primäransicht
# ----------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AnsichtErgebnis:
    an: Ansicht
    K: np.ndarray                 # Kern (beliebiger Maßstab)
    a: np.ndarray                 # Zeilenziele, Summe 1
    b: np.ndarray                 # Spaltenziele, Summe 1 + delta
    gesamt: float                 # Fahrten, die zur Summe 1 gehören (Summe der Zeilenziele)
    ab: object                    # Abgleich (mab_scaling.Abgleich)
    klasse: int | None            # Zulässigkeitsklasse (None bei verschiedenen Summen: keine Lösung)
    opt: float | None
    eps: float | None             # ε des Gravitationskerns (absolut), None sonst
    eps_zurueck: float | None     # ε*/OPT der Rückrechnung aus der mittleren Reiseweite des Ergebnisses (nur Zonen, nur bei Konvergenz)
    schritte_zurueck: int | None
    reiseweite: float | None
    nullanteil_real: float


def ansicht_ergebnis(s: Settings) -> AnsichtErgebnis:
    an = ansicht(s.quelle, s.n, s.seed, s.nullanteil, s.delta, s.rs_ansicht, gravitation=s.quelle == C.QUELLE_ZONEN and s.kern == C.KERN_GRAV)
    A = float(an.a.sum())
    a = an.a / A
    b = an.b / A
    opt = eps = None
    if an.kosten is not None:
        opt = opt_wert(an.kosten, a, b / b.sum())
    if s.quelle == C.QUELLE_ZONEN and s.kern == C.KERN_GRAV:
        eps = s.eps_rel * opt
        K = kern_gravitation(an.kosten, eps)
    else:
        K = an.T0
    ab = abgleichen(K, a, b)
    kl = None if s.delta > 0 else klasse(K > 0, a, b)
    eps_zurueck = schritte = reise = None
    if an.kosten is not None and ab.status == STATUS_OK:
        reise = float((an.kosten * ab.plan).sum())
        if reise > opt * (1 + 1e-9):
            h = hyman_stapel(an.kosten[None], a[None], (b / b.sum())[None], np.array([reise]))
            if h.status[0] == ST_OK:
                eps_zurueck = float(1.0 / (h.beta[0] * opt))
                schritte = int(h.schritte[0])
    return AnsichtErgebnis(an, K, a, b, A, ab, kl, opt, eps, eps_zurueck, schritte, reise, float((an.T0 == 0).mean()))


def plan_fahrten(erg: AnsichtErgebnis, k: int) -> np.ndarray:
    """Matrix nach k Durchläufen in Fahrten (k = 0: die Ausgangsmatrix im Maßstab der Zeilenziele)."""
    P = plan_nach(erg.K, erg.a, erg.b, k)
    if k <= 0:
        P = P / P.sum() * erg.b.sum()
    return P * erg.gesamt


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: die Summen passen nicht
# ----------------------------------------------------------------------------------------------------------------------------------------

DELTAS_EXP1 = (0.0, 0.001, 0.01, 0.1)


def _verlauf_punkte(max_durchlaeufe: int):
    ks = sorted({int(round(x)) for x in np.geomspace(1, max_durchlaeufe, 45)})
    return ks


def reihe_summen(deltas=DELTAS_EXP1, n: int = C.N_ZEILE_EXP1, anzahl: int = C.ANZAHL_EXP1, max_durchlaeufe: int = C.MAX_DURCHLAEUFE):
    """Abgleich auf Zeilenziele a und Spaltenziele (1 + δ) · b (dieselben Instanzen für jedes δ), u-v-Form, n = 8."""
    insts = []
    for i in range(anzahl):
        g = instanz_rng(C.SERIEN_SEED, C.ZWECK_EXP1, i)
        z = zonen(n, g)
        T0 = basis_matrix(z.kosten, g)
        a, b = randsummen_wachstum(T0, g, C.RS_BREITE[C.RS_MILD])
        insts.append((T0, a, b))
    K = np.array([t for t, _, _ in insts])
    a = np.array([x for _, x, _ in insts])
    b = np.array([x for _, _, x in insts])
    ks = _verlauf_punkte(max_durchlaeufe)
    zeilen = []
    for d in deltas:
        r = abgleich_stapel(K, a, b * (1.0 + d), max_durchlaeufe=max_durchlaeufe, verlauf=True)
        verl = np.array(r.verlauf)           # Durchläufe × N
        konv = int((r.status == 0).sum())
        zeilen.append(dict(delta=d, konvergiert=konv, anzahl=anzahl, fehler=band(r.fehler), durchlaeufe=band(r.durchlaeufe[r.status == 0]) if konv else None,
                           verlauf_ks=ks, verlauf=[verl[k - 1].tolist() if k <= len(verl) else [float("nan")] * anzahl for k in ks],
                           nan=int((r.status == 2).sum())))
    return dict(n=n, anzahl=anzahl, max_durchlaeufe=max_durchlaeufe, zeilen=zeilen)


def reihe_strafe(taus=C.STRAFE_TAUS, n: int = C.N_STRAFE, anzahl: int = C.ANZAHL_EXP1):
    """Strafverfahren statt harter Randsummen: Kern = alte Matrix (Zählung mit natürlichen Nullen), ε = 1; Abweichung vom exakten Furness und Durchläufe;
    dazu der Fall verschiedener Gesamtsummen (Zeilenziele +5 %)."""
    Ks, As, Bs, nullen = [], [], [], []
    for i in range(anzahl):
        g = instanz_rng(C.SERIEN_SEED, C.ZWECK_STRAFE, i)
        z = zonen(n, g)
        T0 = poisson_matrix(z, g)
        a, b, massstab = randsummen_wachstum(T0, g, C.RS_BREITE[C.RS_V3], mit_summe=True)
        Ks.append(T0 / massstab)           # Kern im Maßstab der Randsummen (Gesamtmasse 1): die Strafe KL(P|K) ist nicht maßstabsfrei
        As.append(a)
        Bs.append(b)
        nullen.append(float((T0 == 0).mean()))
    K, a, b = np.array(Ks), np.array(As), np.array(Bs)
    exakt = abgleich_stapel(K, a, b, tol=1e-12, max_durchlaeufe=100000)
    ok = exakt.status == 0
    zeilen = []
    for tau in taus:
        P, it = strafe_stapel(K, a, b, tau)
        abw = np.abs(P - exakt.plan).reshape(len(K), -1).max(axis=1) / np.abs(exakt.plan).reshape(len(K), -1).max(axis=1)
        rand = np.abs(P.sum(2) - a).sum(axis=1)
        zeilen.append(dict(tau=tau, abweichung=band(abw[ok]), durchlaeufe=band(it[ok & (it > 0)]), konvergiert=int(((it > 0) & ok).sum()), rand=band(rand[ok])))
    # verschiedene Gesamtsummen: Zeilenziele +5 %
    a5 = a * (1.0 + C.STRAFE_SUMMEN_PLUS)
    fur = abgleich_stapel(K, a5, b, max_durchlaeufe=C.MAX_DURCHLAEUFE)       # fur.fehler: absolut (Spaltenziele Gesamtmasse 1); unten relativ zur Summe der Zeilenziele wie die Strafe
    P, it = strafe_stapel(K, a5, b, 100.0)
    rel = np.abs(P.sum(2) - a5).sum(axis=1) / a5.sum(axis=1)
    return dict(n=n, anzahl=anzahl, instanzen_furness_ok=int(ok.sum()), nullanteil=band(nullen), zeilen=zeilen,
                summen=dict(plus=C.STRAFE_SUMMEN_PLUS, furness_konvergiert=int((fur.status == 0).sum()), furness_fehler=band(fur.fehler / a5.sum(axis=1)),
                            strafe_konvergiert=int((it > 0).sum()), strafe_fehler=band(rel), tau=100.0))


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Nullen in der alten Matrix
# ----------------------------------------------------------------------------------------------------------------------------------------

def instanzen_nullen(p: float, art: str, anzahl: int, n: int = C.N_EXP2, p_index: int = 0, seed: int = C.SERIEN_SEED):
    """Träger (N × n × n, bool), Matrizen, a, b für einen Nullanteil. Der Zufallsstrom hängt am Index der Stufe, nicht am Wert (gleiche Instanzen für alle Reglerwerte)."""
    Ks, As, Bs = [], [], []
    for i in range(anzahl):
        g = instanz_rng(seed, C.ZWECK_EXP2 + 100 * p_index, i)
        T0 = traeger_matrix(n, p, g)
        a, b = randsummen(art, T0, g)
        Ks.append(T0)
        As.append(a)
        Bs.append(b)
    return np.array(Ks), np.array(As), np.array(Bs)


def reihe_nullen(art: str, anzahl: int, form: str, stufen=C.NULLANTEIL_STUFEN, n: int = C.N_EXP2, max_durchlaeufe: int = C.MAX_DURCHLAEUFE, seed: int = C.SERIEN_SEED):
    """Je Nullanteil `anzahl` Instanzen: Zulässigkeitsklasse (Hall-Bedingung), Konvergenz je Klasse, Median der Durchläufe in Klasse 2, NaN-Abbrüche in Klasse 0."""
    zeilen = []
    for pi, p in enumerate(stufen):
        K, a, b = instanzen_nullen(p, art, anzahl, n, pi, seed)
        kl = np.array([klasse(K[i] > 0, a[i], b[i]) for i in range(anzahl)])
        cps = tuple(k for k in (100, 1000, max_durchlaeufe) if k <= max_durchlaeufe)
        r = abgleich_stapel(K, a, b, max_durchlaeufe=max_durchlaeufe, form=form, checkpoints=cps)
        konv = r.status == 0
        z = dict(nullanteil=p, nullanteil_real=float((K == 0).mean()), anzahl=anzahl)
        for c in (0, 1, 2):
            sel = kl == c
            z[f"klasse{c}"] = int(sel.sum())
            z[f"konv{c}"] = int((konv & sel).sum())
        z["nan0"] = int(((r.status == 2) & (kl == 0)).sum())
        z["durchlaeufe2"] = band(r.durchlaeufe[konv & (kl == 2)]) if (konv & (kl == 2)).any() else None
        z["fehler1"] = {str(k): band(r.punkte[k][kl == 1]) for k in cps} if (kl == 1).any() else None
        if art == C.RS_GLEICH:
            tk = np.array([traeger_klasse(K[i] > 0) for i in range(anzahl)])
            z["traeger_gleich_klasse"] = bool((tk == kl).all())
        zeilen.append(z)
    gesamt = dict(anzahl=sum(z["anzahl"] for z in zeilen), klasse0=sum(z["klasse0"] for z in zeilen), klasse1=sum(z["klasse1"] for z in zeilen),
                  klasse2=sum(z["klasse2"] for z in zeilen), konv0=sum(z["konv0"] for z in zeilen), konv1=sum(z["konv1"] for z in zeilen),
                  konv2=sum(z["konv2"] for z in zeilen))
    return dict(art=art, form=form, n=n, anzahl=anzahl, max_durchlaeufe=max_durchlaeufe, zeilen=zeilen, gesamt=gesamt)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: ε aus der Beobachtung zurückrechnen
# ----------------------------------------------------------------------------------------------------------------------------------------

def reihe_kalibrierung(theta: float, beobachtung: str, startregel: str, warm: bool, n: int, anzahl: int = C.ANZAHL_EXP3, seed: int = C.SERIEN_SEED):
    """Beobachtet wird die mittlere Reiseweite c_obs = OPT + θ · (Produktmaß − OPT) (oder 0,9 · OPT im Randfall); die Hyman-Schleife sucht ε."""
    zs, kosten, a, b, opt, prod = zonen_stapel(n, anzahl, C.ZWECK_EXP3, seed)
    randfall = beobachtung == C.BEOB_UNTER
    c_obs = 0.9 * opt if randfall else opt + theta * (prod - opt)
    h = hyman_stapel(kosten, a, b, c_obs, startregel, warm, C.MAX_SCHRITTE_RANDFALL if randfall else C.MAX_SCHRITTE_HYMAN)
    ok = h.status == ST_OK
    eps_opt = 1.0 / (h.beta * opt)
    s0 = int(h.schritte[0])
    return dict(
        theta=theta, randfall=randfall, n=n, anzahl=anzahl, warm=warm, startregel=startregel,
        konvergiert=int(ok.sum()), innen=int((h.status == ST_INNEN).sum()), innen_nan=int(h.innen_nan.sum()), grenze=int((h.status == ST_GRENZE).sum()),
        schritte=band(h.schritte[ok]) if ok.any() else None, durchlaeufe=band(h.durchlaeufe[ok]) if ok.any() else None,
        produkte=band(h.produkte[ok]) if ok.any() else None,
        eps_opt=band(eps_opt[ok]) if ok.any() else None,
        eps_verhaeltnis=float(eps_opt[ok].max() / eps_opt[ok].min()) if ok.any() else None,
        genauigkeit=float((np.abs(h.reiseweite[ok] - c_obs[ok]) / c_obs[ok]).max()) if ok.any() else None,
        beta_ende=band(h.beta), beta_start=band(1.0 / c_obs),
        spur=dict(schritte=s0, beta=h.beta_verlauf[:s0, 0].tolist(), eps_opt=(1.0 / (h.beta_verlauf[:s0, 0] * opt[0])).tolist(), weite=h.weite_verlauf[:s0, 0].tolist(),
                  c_obs=float(c_obs[0]), opt=float(opt[0]), produkt=float(prod[0]), status=int(h.status[0])),
    )


def reihe_wahrheit(rs=(0.3, 1.0, 3.0), n: int = 12, anzahl: int = C.ANZAHL_EXP3, startregel: str = C.START_PROP, seed: int = C.SERIEN_SEED):
    """Probe mit bekannter Wahrheit: der Plan zu ε_w = r · OPT liefert die beobachtete Reiseweite; die Schleife rechnet ε zurück, kalt und warm gestartet."""
    zs, kosten, a, b, opt, prod = zonen_stapel(n, anzahl, C.ZWECK_EXP3, seed)
    zeilen = []
    for r in rs:
        K = np.exp(-kosten / (r * opt)[:, None, None])
        w = abgleich_stapel(K, a, b, tol=1e-12, max_durchlaeufe=400000)
        c_obs = (kosten * w.plan).sum(axis=(1, 2))
        theta = (c_obs - opt) / (prod - opt)
        z = dict(r=r, theta=band(theta), wahrheit_ok=int((w.status == 0).sum()))
        for warm in (False, True):
            h = hyman_stapel(kosten, a, b, c_obs, startregel, warm)
            ok = h.status == ST_OK
            z["warm" if warm else "kalt"] = dict(konvergiert=int(ok.sum()), schritte=band(h.schritte[ok]), durchlaeufe=band(h.durchlaeufe[ok]),
                                                 rueck=float((np.abs(1.0 / (h.beta[ok] * opt[ok]) / r - 1.0)).max()))
        k, wm = z["kalt"]["durchlaeufe"][0], z["warm"]["durchlaeufe"][0]
        z["warm_spart"] = float(1.0 - wm / k)
        zeilen.append(z)
    return dict(n=n, anzahl=anzahl, startregel=startregel, zeilen=zeilen)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: Randsummen stimmen, die Matrix nicht
# ----------------------------------------------------------------------------------------------------------------------------------------

def wahrheit(familie: str, kosten: np.ndarray, opt: float, g) -> np.ndarray:
    """Wahre Matrix (Gesamtmasse 1): T1 = exp(−C/OPT) mal lognormales Paar-Rauschen (σ = 0,5), T2 = (C + 0,05)⁻² ohne Rauschen."""
    if familie == C.FAMILIE_T1:
        z = np.array([g.normal() for _ in range(kosten.size)]).reshape(kosten.shape)
        T = np.exp(-kosten / opt) * np.exp(C.RAUSCH_T1 * z)
    else:
        T = (kosten + C.HEMM_VERSATZ) ** -2.0
    return T / T.sum()


def tv_abstand(P, T) -> np.ndarray:
    """Anteil der Gesamtmenge in der falschen Zelle: ½ Σ |P − T| (Gesamtmasse 1), je Instanz."""
    return 0.5 * np.abs(P - T).reshape(len(P), -1).sum(axis=1)


def _instanzen_randsummen(familie: str, n: int, anzahl: int):
    """Wahre Matrizen (Gesamtmasse 1) mit Kosten, Randsummen und den Standardnormalzahlen der Störung, je Instanz ein eigener Zufallsstrom."""
    Ts, Ks, As, Bs, Zs = [], [], [], [], []
    for i in range(anzahl):
        g = instanz_rng(C.SERIEN_SEED, C.ZWECK_EXP4, i)
        z = zonen(n, g)
        opt_z = opt_wert(z.kosten, z.a, z.b)
        T = wahrheit(familie, z.kosten, opt_z, g)
        Zn = np.array([g.normal() for _ in range(n * n)]).reshape(n, n)
        Ts.append(T)
        Ks.append(z.kosten)
        As.append(T.sum(1))
        Bs.append(T.sum(0))
        Zs.append(Zn)
    return np.array(Ts), np.array(Ks), np.array(As), np.array(Bs), np.array(Zs)


def reihe_gravitation(familie: str, n: int, anzahl: int):
    """Gravitationskern mit β aus der mittleren Reiseweite der Wahrheit (Hyman-Schleife, warm gestartet, innere Obergrenze 400 000): die teure Zeile von Experiment 4."""
    T, kosten, a, b, _ = _instanzen_randsummen(familie, n, anzahl)
    kleinste = np.minimum(a.min(axis=1), b.min(axis=1))
    h = hyman_stapel(kosten, a, b, (kosten * T).sum(axis=(1, 2)), warm=True, max_innen=C.MAX_INNEN_EXP4)
    ok = h.status == ST_OK
    grav = abgleich_stapel(np.exp(-h.beta[:, None, None] * kosten), a, b, tol=C.TOL_RAND, max_durchlaeufe=C.MAX_INNEN_EXP4)
    gut = ok & (grav.status == 0)
    return dict(familie=familie, n=n, anzahl=anzahl, kalibriert=int(gut.sum()), tv=band(tv_abstand(grav.plan, T)[gut]) if gut.any() else None,
                rand=float((np.abs(grav.plan.sum(2) - a).max(axis=1) / kleinste)[gut].max()) if gut.any() else None,
                durchlaeufe=band(h.durchlaeufe))


def reihe_randsummen(familie: str, n: int, anzahl: int, sigmas=C.SIGMA_STUFEN):
    """Die wahre Matrix liefert nur ihre Randsummen. Wiederhergestellt wird sie aus (a) dem Produktmaß und (b) einer gestörten alten Matrix T · exp(σ Z) (Z standardnormal je Zelle),
    abgeglichen; zum Vergleich (c) dieselbe alte Matrix ohne Abgleich. Der Gravitationskern mit Hyman-β steht in `reihe_gravitation`."""
    T, kosten, a, b, Z = _instanzen_randsummen(familie, n, anzahl)
    kleinste = np.minimum(a.min(axis=1), b.min(axis=1))
    produkt = a[:, :, None] * b[:, None, :]
    ergebnis = dict(familie=familie, n=n, anzahl=anzahl, produktmatrix=band(tv_abstand(produkt, T)), sigmas=[])
    rand_max = 0.0
    nicht = 0
    for sg in sigmas:
        alt = T * np.exp(sg * Z)
        r = abgleich_stapel(alt, a, b, tol=C.TOL_RAND, max_durchlaeufe=C.MAX_INNEN_HYMAN)
        rand = np.maximum(np.abs(r.plan.sum(2) - a).max(axis=1), np.abs(r.plan.sum(1) - b).max(axis=1)) / kleinste
        gut = r.status == 0
        rand_max = max(rand_max, float(rand[gut].max()))
        nicht += int((~gut).sum())
        roh = alt / alt.sum(axis=(1, 2))[:, None, None]
        ergebnis["sigmas"].append(dict(sigma=sg, abgeglichen=band(tv_abstand(r.plan, T)), ohne=band(tv_abstand(roh, T)), rand=float(rand[gut].max()),
                                       konvergiert=int((r.status == 0).sum()),
                                       bild=dict(wahrheit=T[0].tolist(), abgeglichen=r.plan[0].tolist(), ohne=roh[0].tolist())))
    ergebnis["rand_max"] = rand_max
    ergebnis["nicht_konvergiert"] = nicht
    ergebnis["max_durchlaeufe"] = C.MAX_INNEN_HYMAN
    ergebnis["rang"] = dict(festgelegt=2 * n - 1, frei=(n - 1) ** 2, zahlen=n * n)
    return ergebnis


# ----------------------------------------------------------------------------------------------------------------------------------------
# Vorrechnung für die Standardeinstellungen (siehe mab_vorgerechnet.py)
# ----------------------------------------------------------------------------------------------------------------------------------------

def gravitation_v(familie: str, n: int, anzahl: int):
    """Gravitationszeile von Experiment 4 (vorgerechnet für die Standardeinstellung, sonst live)."""
    from mab_vorgerechnet import hole

    treffer = hole("reihe_gravitation", familie=familie, n=int(n), anzahl=int(anzahl))
    return treffer if treffer is not None else jsonbar(reihe_gravitation(familie, int(n), int(anzahl)))


def vorzurechnen():
    """Alle Einträge von tools/vorgerechnet.json als Liste (Name, Parameter, Aufruf): beide Wahrheitsfamilien mit n = 12 und 50 Instanzen."""
    d = C.DEFAULTS
    return [("reihe_gravitation", dict(familie=f, n=d["n_mess"], anzahl=d["inst4"]), lambda f=f: reihe_gravitation(f, d["n_mess"], d["inst4"])) for f in C.FAMILIEN]
