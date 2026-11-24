"""Zeilen und Sätze der README, aus `tools/messreihen.json` mit denselben Formatfunktionen wie in der App gebildet (mab_evaluation.de, de_g, ganz, band_ganz; Verdikte aus mab_verdikte).

Die README muss jede dieser Zeilen wörtlich enthalten (`tests/test_claims.py`); `python tests/readme_zeilen.py` druckt sie zum Einfügen."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mab_constants as C  # noqa: E402
import mab_evaluation as E  # noqa: E402
import mab_verdikte as V  # noqa: E402

J = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))
T1, T2 = C.FAMILIEN
NULLEN_SCHLUESSEL = {
    "beliebig": f"{C.RS_BELIEBIG}|{C.FORM_UV}",
    "beliebig_matrix": f"{C.RS_BELIEBIG}|{C.FORM_MATRIX}",
    "mild": f"{C.RS_MILD}|{C.FORM_UV}",
    "breit": f"{C.RS_V3}|{C.FORM_UV}",
    "stark": f"{C.RS_STARK}|{C.FORM_UV}",
    "gleich": f"{C.RS_GLEICH}|{C.FORM_UV}",
}


def zeile_summen(z: dict) -> str:
    dl = E.ganz(z["durchlaeufe"][0]) if z["durchlaeufe"] else "–"
    return f"| {E.de(z['delta'], 3)} | {z['konvergiert']} von {z['anzahl']} | {E.de_g(z['fehler'][0], 3)} | {dl} |"


def zeile_strafe(z: dict, anzahl: int) -> str:
    return f"| {E.de_g(z['tau'], 4)} | {E.de_g(z['abweichung'][0], 3)} | {E.band_ganz(z['durchlaeufe'])} | {z['konvergiert']} von {anzahl} |"


def zeile_nullen(z: dict) -> str:
    dl = E.ganz(z["durchlaeufe2"][0]) if z["durchlaeufe2"] else "–"
    return f"| {E.de(z['nullanteil'], 1)} | {z['klasse0']} | {z['klasse1']} | {z['klasse2']} | {z['konv2']} von {z['klasse2']} | {z['nan0']} | {dl} |"


def zeile_kalibrierung(r: dict) -> str:
    return (f"| {E.de_g(r['theta'])} | {r['konvergiert']} von {r['anzahl']} | {E.band_ganz(r['schritte'])} | {E.band_ganz(r['durchlaeufe'])} | {E.band_ganz(r['produkte'])} | "
            f"{E.band_g(r['eps_opt'], 3)} | {E.de_g(r['eps_verhaeltnis'], 2)} |")


def zeile_wahrheit(z: dict) -> str:
    return (f"| {E.de_g(z['r'], 2)} | {E.de_g(z['theta'][0], 2)} | {E.band_ganz(z['kalt']['schritte'])} | {E.band_ganz(z['kalt']['durchlaeufe'])} | "
            f"{E.band_ganz(z['warm']['durchlaeufe'])} | {E.prozent(z['warm_spart'], 0)} | {E.de_g(z['kalt']['rueck'], 2)} |")


def zeile_randsummen(z: dict) -> str:
    return f"| {E.de(z['sigma'], 1)} | {E.de(z['abgeglichen'][0], 3)} | {E.de(z['ohne'][0], 3)} | {E.de_g(z['rand'], 2)} |"


def zeilen_alle() -> list:
    z = []
    z += [zeile_summen(r) for r in J["summen"]["zeilen"]]
    z += [zeile_strafe(r, J["strafe"]["anzahl"]) for r in J["strafe"]["zeilen"]]
    for k in ("beliebig", "breit", "gleich"):
        z += [zeile_nullen(r) for r in J["nullen"][NULLEN_SCHLUESSEL[k]]["zeilen"]]
    z += [zeile_kalibrierung(J["kalibrierung"][f"{th}|kalt"]) for th in C.THETA_STUFEN]
    z += [zeile_wahrheit(r) for r in J["wahrheit"][C.START_PROP]["zeilen"]]
    z += [zeile_randsummen(r) for r in J["randsummen"][T1]["sigmas"]]
    z += [zeile_randsummen(r) for r in J["randsummen"][T2]["sigmas"]]
    return z


def verdikte() -> dict:
    """Die Verdikt-Sätze der App für die Standardeinstellungen, wie sie die README wörtlich enthält."""
    return {
        "summen": V.verdikt_summen(J["summen"], 0.1)[1],
        "strafe": V.verdikt_strafe(J["strafe"], C.TAU_AUS)[1],
        "nullen": V.verdikt_nullen(J["nullen"][NULLEN_SCHLUESSEL["beliebig"]], 0.5)[1],
        "kalibrierung": V.verdikt_kalibrierung(J["kalibrierung"]["0.1|kalt"])[1],
        "randfall": V.verdikt_kalibrierung(J["kalibrierung"]["rand"])[1],
        "randsummen_t1": V.verdikt_randsummen(J["randsummen"][T1], J["gravitation"][T1], 0.6)[1],
        "randsummen_t2": V.verdikt_randsummen(J["randsummen"][T2], J["gravitation"][T2], 0.6)[1],
        "gleichheit": satz_gleichheit(),
        "gross": satz_gross(),
        "hand": satz_hand(),
        "start": satz_start(),
        "nullanteil_zaehlung": E.band_g(J["strafe"]["nullanteil"], 2),
        "nan": satz_nan(),
        "klasse1": satz_klasse1(),
        "seeds_fehlt": satz_seeds_fehlt(),
        "seeds_verhaeltnis": satz_seeds_verhaeltnis(),
        "seeds_warm": satz_seeds_warm(),
        "aufwand": satz_aufwand(),
    }


def satz_gleichheit() -> str:
    g = J["gleichheit"]
    return (f"Furness und Sinkhorn-Skalierung in der Log-Domäne stimmen nach jedem der ersten {g['durchlaeufe']} Durchläufe überein: größte relative Abweichung {E.de_g(g['groesste'], 2)} "
            f"({g['anzahl']} Instanzen, n = {g['n']}, dichte Basismatrix)")


def satz_gross() -> str:
    b, w = J["nullen_gross"][C.RS_BELIEBIG], J["nullen_gross"][C.RS_V3]
    return (f"Bei Nullanteil 0,5, n = 8 und beliebigen Randsummen haben in zwei Sätzen zu je 1 000 Instanzen {b['saetze'][0]} und {b['saetze'][1]} keine Lösung, "
            f"das sind {E.de(b['saetze'][0] / 10, 1)} % und {E.de(b['saetze'][1] / 10, 1)} % (in vier Sätzen zu je 100: {', '.join(str(x) for x in b['saetze100'])}); "
            f"bei Randsummen aus Wachstumsfaktoren der Breite U(0,7; 1,6) sind es {w['saetze'][0]} und {w['saetze'][1]} von je 1 000 "
            f"(in vier Sätzen zu je 100: {', '.join(str(x) for x in w['saetze100'])})")


def satz_start() -> str:
    a, b = J["kalibrierung"]["0.1|kalt"], J["kalibrierung"]["0.1|faktor"]
    return (f"Mit der anderen Startregel (erster Schritt 1,5 · β) braucht θ = 0,1 im Median {E.ganz(b['schritte'][0])} Schritte und {E.ganz(b['durchlaeufe'][0])} Durchläufe statt "
            f"{E.ganz(a['schritte'][0])} Schritte und {E.ganz(a['durchlaeufe'][0])} Durchläufe: bei θ = 0,1 hängt die Zahl der Schritte und Durchläufe an der Startregel (für andere θ nicht verglichen)")


def satz_hand() -> str:
    h = J["hand"]
    return (f"In der u-v-Form wird der Zeilenfehler des Handbeispiels 3(b) ab Durchlauf {E.ganz(h['nan_durchlauf'])} zu NaN (ein Unterlauf der Gleitkomma-Rechnung: die Zahl hängt von der Arithmetik ab, geprüft wird ein Band von 1 000 bis 20 000); in der Matrix-Form bleibt er nach 20 000 Durchläufen bei "
            f"{E.de_g(h['matrix_fehler'], 3)} Fahrten")


def _nan_anteil(schluessel: str) -> tuple:
    r = J["nullen"][NULLEN_SCHLUESSEL[schluessel]]
    return sum(z["nan0"] for z in r["zeilen"]), sum(z["klasse0"] for z in r["zeilen"])


def satz_nan() -> str:
    """NaN-Abbrüche unter den Läufen ohne Lösung (Klasse 0), über alle Nullanteile, je Art der Randsummen; Matrix-Form bei beliebigen Randsummen."""
    t = {k: _nan_anteil(k) for k in ("beliebig", "gleich", "mild", "breit", "stark", "beliebig_matrix")}
    z = lambda k: f"{t[k][0]} von {t[k][1]}"  # noqa: E731
    return (f"Unter den Läufen ohne Lösung (Klasse 0, u-v-Form, alle Nullanteile) enden in NaN: bei beliebigen Randsummen {z('beliebig')}, bei gleichen Randsummen {z('gleich')}, "
            f"bei Randsummen aus Wachstumsfaktoren der Breite U(0,9; 1,1) {z('mild')}, U(0,7; 1,6) {z('breit')} und U(0,4; 2,5) {z('stark')}; die übrigen erreichen die Obergrenze mit endlichem Zeilenfehler. "
            f"In der Matrix-Form endet bei beliebigen Randsummen keiner dieser Läufe in NaN ({z('beliebig_matrix')})")


def satz_klasse1() -> str:
    g = J["nullen"][NULLEN_SCHLUESSEL["gleich"]]["gesamt"]
    return (f"In den {g['klasse1']} Instanzen der Klasse 1 (Tabelle mit gleichen Randsummen, n = 8) konvergiert in keinem Lauf der Randfehler unter 1e-9, er fällt in diesen Instanzen etwa wie 1/k")


def satz_seeds_fehlt() -> str:
    f = [x for v in J["andere_seeds"]["nullen_fehlt"].values() for x in v]
    s = J["andere_seeds"]["seeds"]
    return (f"mit den Zufalls-Seeds {s[0]} bis {s[-1]} fehlen in jeder der drei Tabellen {min(f)} bis {max(f)} Läufe an der Obergrenze")


def satz_seeds_verhaeltnis() -> str:
    v = J["andere_seeds"]["verhaeltnis_theta01"]
    s = J["andere_seeds"]["seeds"]
    return f"Dieses Verhältnis ist ein Extremwert aus 20 Instanzen: bei θ = 0,1 liegt es in den Zufalls-Seeds {s[0]} bis {s[-1]} zwischen {E.de_g(min(v), 2)} und {E.de_g(max(v), 2)}"


def satz_seeds_warm() -> str:
    w = [x for v in J["andere_seeds"]["warm_spart"] for x in v]
    s = J["andere_seeds"]["seeds"]
    return f"in den Zufalls-Seeds {s[0]} bis {s[-1]} spart der Warmstart {E.prozent(min(w), 0)} bis {E.prozent(max(w), 0)}"


def satz_aufwand() -> str:
    z = next(r for r in J["wahrheit"][C.START_PROP]["zeilen"] if r["r"] == 1.0)["kalt"]["durchlaeufe"]
    return f"Bei gleichem ε/OPT = 1 braucht die Kalibrierung in der Probe mit bekannter Wahrheit im Median {E.ganz(z[0])} Durchläufe (kalt gestartet), einzelne Instanzen {E.ganz(z[1])} bis {E.ganz(z[2])}"


if __name__ == "__main__":
    for zeile in zeilen_alle():
        print(zeile)
    for k, v in verdikte().items():
        print("##", k, "\n", v)
