"""Messreihen der README (alle Zahlen, die die README nennt): rechnet mit dem Code der Demo (Zufalls-Seed 2026 der Reihen, SplitMix64) und schreibt `tools/messreihen.json`.

Aufruf (im Ordner der Demo): <Test-venv>/Scripts/python.exe -X utf8 tools/messreihen.py      (rund 3 Minuten)
`tests/test_claims.py` bildet die README-Zahlen aus dieser Datei und rechnet kleine Reihen neu (als Band)."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import mab_constants as C  # noqa: E402
import mab_evaluation as E  # noqa: E402
import mab_scaling as S  # noqa: E402
from mab_feasibility import klasse  # noqa: E402
from orakel_hilfe import sinkhorn_log  # noqa: E402

GROSSE_LISTEN = ("verlauf", "bild")


def kurz(x):
    """Entfernt die langen Listen (Verläufe, Bilder), die die README nicht braucht."""
    if isinstance(x, dict):
        return {k: kurz(v) for k, v in x.items() if k not in GROSSE_LISTEN}
    if isinstance(x, list):
        return [kurz(v) for v in x]
    return x


def zeit(name, f):
    t = time.time()
    r = f()
    print(f"{name}: {time.time() - t:.1f} s", flush=True)
    return kurz(E.jsonbar(r))


def unzulaessig_grosse_saetze(art: str, p: float, anzahl: int, indizes):
    """Zahl der Instanzen ohne Lösung in Sätzen zu `anzahl` Instanzen (eigene Zufallsströme je Satz), Klasse nach der Hall-Bedingung."""
    out = []
    for pi in indizes:
        K, a, b = E.instanzen_nullen(p, art, anzahl, C.N_EXP2, pi)
        out.append(int(sum(klasse(K[i] > 0, a[i], b[i]) == 0 for i in range(anzahl))))
    return out


def gleichheit(anzahl=20, n=10, durchlaeufe=30):
    """Furness gegen Sinkhorn in der Log-Domäne nach jedem der ersten `durchlaeufe` Durchläufe: größte Abweichung der Matrizen (relativ zum größten Eintrag)."""
    from mab_scenario import basis_matrix, instanz_rng, randsummen_wachstum, zonen

    groesste = 0.0
    for i in range(anzahl):
        g = instanz_rng(C.SERIEN_SEED, C.ZWECK_RS, i)
        z = zonen(n, g)
        T0 = basis_matrix(z.kosten, g)
        a, b = randsummen_wachstum(T0, g, C.RS_BREITE[C.RS_V3])
        for k in range(1, durchlaeufe + 1):
            P1 = S.plan_nach(T0, a, b, k)
            P2 = sinkhorn_log(T0, a, b, k)
            groesste = max(groesste, float(np.abs(P1 - P2).max() / np.abs(P2).max()))
    return dict(anzahl=anzahl, n=n, durchlaeufe=durchlaeufe, groesste=groesste)


def handbeispiele():
    """Zahlen der Handbeispiele, die die README nennt (Durchlauf der NaN-Meldung, Randfehler 2/(2k + 1))."""
    K0 = np.array([[1.0, 1.0], [0.0, 1.0]])
    uv = S.abgleichen(K0, [4, 6], [5, 5], form=C.FORM_UV, max_durchlaeufe=20000)
    mat = S.abgleichen(K0, [4, 6], [5, 5], form=C.FORM_MATRIX, max_durchlaeufe=20000)
    gleich = S.abgleichen(K0, [1, 1], [1, 1], max_durchlaeufe=1000)
    return dict(nan_durchlauf=uv.durchlaeufe, matrix_fehler=float(mat.fehler_verlauf[-1]) * 10.0, matrix_status=mat.status,
                randfehler_k30=float(gleich.fehler_verlauf[29]), randfehler_k30_formel=2.0 / 61.0)


ANDERE_SEEDS = (2026, 2027, 2028, 2029)


def andere_seeds():
    """Dieselben Reihen mit anderen Zufalls-Seeds: Verhältnis größter zu kleinster Wert von ε*/OPT bei θ = 0,1, Ersparnis des Warmstarts in der Probe mit bekannter Wahrheit,
    in Klasse 2 fehlende Läufe je Nullen-Tabelle (u-v-Form, 8 Nullanteile zu je 100 Instanzen). Die README nennt nur Bänder über diese Seeds."""
    out = dict(seeds=list(ANDERE_SEEDS), verhaeltnis_theta01=[], warm_spart=[], nullen_fehlt={art: [] for art in (C.RS_BELIEBIG, C.RS_V3, C.RS_GLEICH)})
    for seed in ANDERE_SEEDS:
        r = E.reihe_kalibrierung(0.1, C.BEOB_THETA, C.START_PROP, False, 12, seed=seed)
        out["verhaeltnis_theta01"].append(float(r["eps_verhaeltnis"]))
        w = E.reihe_wahrheit(seed=seed)
        out["warm_spart"].append([float(z["warm_spart"]) for z in w["zeilen"]])
        for art in out["nullen_fehlt"]:
            res = E.reihe_nullen(art, 100, C.FORM_UV, seed=seed)
            out["nullen_fehlt"][art].append(sum(z["klasse2"] - z["konv2"] for z in res["zeilen"]))
        print("seed", seed, flush=True)
    return out


def main():
    d = {"meta": {"seed": C.SERIEN_SEED, "tol": C.TOL, "max_durchlaeufe": C.MAX_DURCHLAEUFE}}
    d["summen"] = zeit("summen", E.reihe_summen)
    d["strafe"] = zeit("strafe", E.reihe_strafe)
    d["nullen"] = {}
    for art, form in ((C.RS_BELIEBIG, C.FORM_UV), (C.RS_BELIEBIG, C.FORM_MATRIX), (C.RS_MILD, C.FORM_UV), (C.RS_V3, C.FORM_UV), (C.RS_STARK, C.FORM_UV), (C.RS_GLEICH, C.FORM_UV)):
        d["nullen"][f"{art}|{form}"] = zeit(f"nullen {art} {form}", lambda art=art, form=form: E.reihe_nullen(art, 100, form))
    d["nullen_gross"] = {art: zeit(f"nullen gross {art}", lambda art=art: dict(saetze=unzulaessig_grosse_saetze(art, 0.5, 1000, (50, 51)),
                                                                              saetze100=unzulaessig_grosse_saetze(art, 0.5, 100, (60, 61, 62, 63)))) for art in (C.RS_BELIEBIG, C.RS_V3)}
    d["kalibrierung"] = {}
    for th in C.THETA_STUFEN:
        d["kalibrierung"][f"{th}|kalt"] = zeit(f"kalibrierung {th} kalt", lambda th=th: E.reihe_kalibrierung(th, C.BEOB_THETA, C.START_PROP, False, 12))
    d["kalibrierung"]["0.1|warm"] = zeit("kalibrierung 0.1 warm", lambda: E.reihe_kalibrierung(0.1, C.BEOB_THETA, C.START_PROP, True, 12))
    d["kalibrierung"]["0.1|faktor"] = zeit("kalibrierung 0.1 faktor", lambda: E.reihe_kalibrierung(0.1, C.BEOB_THETA, C.START_FAKTOR, False, 12))
    d["kalibrierung"]["rand"] = zeit("kalibrierung randfall", lambda: E.reihe_kalibrierung(0.1, C.BEOB_UNTER, C.START_PROP, False, 12))
    d["wahrheit"] = {s: zeit(f"wahrheit {s}", lambda s=s: E.reihe_wahrheit(startregel=s)) for s in C.STARTREGELN}
    d["randsummen"] = {f: zeit(f"randsummen {f}", lambda f=f: E.reihe_randsummen(f, 12, 50)) for f in C.FAMILIEN}
    d["gravitation"] = {f: zeit(f"gravitation {f}", lambda f=f: E.reihe_gravitation(f, 12, 50)) for f in C.FAMILIEN}
    d["gleichheit"] = zeit("gleichheit", gleichheit)
    d["hand"] = zeit("handbeispiele", handbeispiele)
    d["andere_seeds"] = zeit("andere seeds", andere_seeds)
    (ROOT / "tools" / "messreihen.json").write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("geschrieben: tools/messreihen.json")


if __name__ == "__main__":
    main()
