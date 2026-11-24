"""Verdikte der Demo: reine Funktionen (Art, Text), die die App mit `getattr(st, Art)(Text)` anzeigt. Jede Zahl eines Verdikts stammt aus der Reihe, die die Funktion bekommt.

Die Reihen selbst stehen in `mab_evaluation.py`; hier nur die Sätze. Aussagen gelten in diesem Modell, bei diesen Parametern (synthetische Zonen, Zählung in Durchläufen,
Matrix-Vektor-Produkten oder Instanzen)."""

import mab_constants as C
from mab_evaluation import AnsichtErgebnis, band_g, band_ganz, de, de_g, ganz, prozent
from mab_scaling import STATUS_NAN, STATUS_OK


def eps_text(e: float) -> str:
    """ε/OPT für die Anzeige: höchstens drei gültige Stellen."""
    return de_g(e, 3)


def verdikt_ansicht(erg: AnsichtErgebnis):
    """Verdikt zur Primäransicht: konvergiert, keine Lösung (Summen verschieden, Nullen) oder nur Randlösung."""
    ab = erg.ab
    A = erg.gesamt
    delta = erg.b.sum() - 1.0
    if delta > 1e-12:
        rest = float(ab.fehler_verlauf[-1]) * A if len(ab.fehler_verlauf) else float("nan")
        return "warning", (f"Keine Lösung: die Spaltensummen sind um {prozent(delta, 1)} größer als die Zeilensummen ({de(A * (1 + delta), 1)} gegen {de(A, 1)} Fahrten), "
                           f"keine Matrix hat beide Randsummen. Der Abgleich läuft bis zur Obergrenze von {ganz(C.MAX_DURCHLAEUFE)} Durchläufen und endet ohne Fehlermeldung des Verfahrens; "
                           f"der Zeilenfehler bleibt bei {de_g(rest, 3)} Fahrten, der Differenz der Summen. Diese Meldung stammt von der App.")
    if ab.status == STATUS_OK:
        return "success", (f"Der Abgleich konvergiert nach {ganz(ab.durchlaeufe)} Durchläufen, das sind {ganz(ab.produkte)} Matrix-Vektor-Produkte "
                           f"(Abbruch: Zeilenfehler unter 1e-9 bei Gesamtmasse 1, float64).")
    nullen = erg.nullanteil_real
    if erg.klasse == 0:
        if ab.status == STATUS_NAN:
            return "error", (f"Keine Lösung: bei Durchlauf {ganz(ab.durchlaeufe)} wird eine Summe null und der Abgleich teilt durch null (NaN). Die Randsummen verlangen von den Nullzellen "
                             f"der alten Matrix mehr, als der Träger hergibt (Klasse 0: Hall-Bedingung verletzt; Nullanteil {de(nullen, 2)}). Diese Meldung stammt von der App, das Verfahren selbst meldet nichts.")
        return "error", (f"Keine Lösung: der Abgleich erreicht die Obergrenze von {ganz(C.MAX_DURCHLAEUFE)} Durchläufen, der Zeilenfehler bleibt bei {de_g(float(ab.fehler_verlauf[-1]) * A, 3)} Fahrten "
                         f"(Klasse 0: Hall-Bedingung verletzt; Nullanteil {de(nullen, 2)}).")
    if erg.klasse == 1:
        return "warning", ("Die Aufgabe hat eine Lösung, aber keine, die auf dem ganzen Träger streng positiv ist (Klasse 1): der Abgleich konvergiert nur langsam (in den gemessenen Fällen fällt der Randfehler etwa wie 1/k), "
                           f"die Obergrenze von {ganz(C.MAX_DURCHLAEUFE)} Durchläufen ist erreicht.")
    return "warning", f"Obergrenze von {ganz(C.MAX_DURCHLAEUFE)} Durchläufen erreicht, obwohl eine streng positive Lösung existiert (Klasse 2): der Abgleich konvergiert hier langsam."


def antwort_loesung(erg: AnsichtErgebnis) -> str:
    """Antwort auf Kernfrage 1: Hat die Aufgabe eine Lösung?"""
    if erg.b.sum() - 1.0 > 1e-12:
        return "Nein: die Summen sind verschieden."
    return {0: "Nein: die Nullstruktur der alten Matrix lässt diese Randsummen nicht zu (Klasse 0).",
            1: "Ja, aber nur mit Nullen auf dem Träger (Klasse 1): der Abgleich konvergiert nur langsam (in den gemessenen Fällen fällt der Randfehler etwa wie 1/k).",
            2: "Ja, mit einer Matrix, die auf dem ganzen Träger streng positiv ist (Klasse 2): der Abgleich konvergiert linear."}[erg.klasse]


def verdikt_summen(res, delta: float):
    """Experiment 1, Teil 1: die Summen passen nicht. `res` = reihe_summen."""
    zeilen = res["zeilen"]
    z = min(zeilen, key=lambda r: abs(r["delta"] - delta))
    anz, n, mx = res["anzahl"], res["n"], res["max_durchlaeufe"]
    ungleich = [r for r in zeilen if r["delta"] > 0]
    summe_konv = sum(r["konvergiert"] for r in ungleich)
    summe_n = sum(r["anzahl"] for r in ungleich)
    stufen = ", ".join(de_g(r["delta"]) for r in ungleich)
    if z["delta"] == 0:
        return "success", (f"(B − A)/A = 0: {z['konvergiert']} von {anz} Läufen konvergieren, im Median nach {band_ganz(z['durchlaeufe'])} Durchläufen "
                           f"(n = {n}, {anz} Instanzen, Obergrenze {ganz(mx)}). Bei verschiedenen Summen konvergiert dagegen keiner: zusammen {summe_konv} von {summe_n} Läufen "
                           f"(Stufen {stufen}).")
    gleich = all(abs(r["fehler"][1] - r["delta"]) < 1e-6 * r["delta"] and abs(r["fehler"][2] - r["delta"]) < 1e-6 * r["delta"] for r in ungleich)
    zusatz = " Der Zeilenfehler ist in jedem Lauf genau diese Differenz (kleinster und größter Wert gleich; bei diesen dichten Matrizen, allgemein ist er mindestens |B − A|)." if gleich else ""
    return "warning", (f"(B − A)/A = {de_g(z['delta'])}: {z['konvergiert']} von {anz} Läufen konvergieren; der Zeilenfehler bleibt bei {de_g(z['fehler'][0], 3)} = |B − A| "
                       f"(Gesamtmasse 1; n = {n}, {anz} Instanzen, Obergrenze {ganz(mx)} Durchläufe). Über alle drei Stufen ({stufen}) zusammen konvergieren {summe_konv} von {summe_n} Läufen.{zusatz}")


def strafe_faktoren(sres):
    """Faktoren je Zehnerschritt von τ = 10 an: (Abweichung vorher/nachher, Durchläufe nachher/vorher)."""
    zs = sres["zeilen"]
    dev, its = [], []
    for k in range(1, len(zs) - 1):
        dev.append(zs[k]["abweichung"][0] / zs[k + 1]["abweichung"][0])
        its.append(zs[k + 1]["durchlaeufe"][0] / zs[k]["durchlaeufe"][0])
    return dev, its


def verdikt_strafe(sres, tau):
    """Experiment 1, Teil 2: Strafverfahren statt harter Randsummen. `tau`: Text aus C.TAU_STUFEN."""
    zs = sres["zeilen"]
    dev, its = strafe_faktoren(sres)
    s5 = sres["summen"]
    konv = sorted({z["konvergiert"] for z in zs})
    konv_text = f"{konv[0]} von {sres['anzahl']}" if len(konv) == 1 else f"{konv[0]} bis {konv[-1]} von {sres['anzahl']}"
    satz = (f"Mit KL-Strafe konvergieren {konv_text} Läufen (n = {sres['n']}, ε = 1, Kern = Zählung "
            f"mit natürlichen Nullen). Ab τ = 10 sinkt die Abweichung vom exakten Furness je Zehnerschritt (zwei Schritte: 10, 100, 1 000) um den Faktor {de_g(min(dev), 2)} bis {de_g(max(dev), 2)} (in diesem Bereich etwa 1/τ) und die Zahl der "
            f"Durchläufe steigt um den Faktor {de_g(min(its), 2)} bis {de_g(max(its), 2)} (in diesem Bereich etwa τ); bei τ = 1 gilt das nicht. Bei verschiedenen Summen (Zeilenziele +{de_g(100 * s5['plus'], 2)} %): Furness konvergiert in "
            f"{s5['furness_konvergiert']} von {sres['anzahl']} Läufen (Zeilenfehler {de_g(s5['furness_fehler'][0], 3)}), das Strafverfahren mit τ = 100 in {s5['strafe_konvergiert']} von {sres['anzahl']} "
            f"(Zeilenfehler {de_g(s5['strafe_fehler'][0], 3)}); beide Zeilenfehler sind relativ zur Summe der Zeilenziele angegeben.")
    if tau != C.TAU_AUS:
        z = next(r for r in zs if r["tau"] == float(tau))
        satz = f"τ = {tau}: größte relative Abweichung vom exakten Furness {de_g(z['abweichung'][0], 3)} (Median), {band_ganz(z['durchlaeufe'])} Durchläufe. " + satz
    return "info", satz


def konvergenz_text(z) -> str:
    """„konvergiert“ je Klasse, leere Klassen ausgelassen."""
    teile = [f"{z[f'konv{c}']} von {z[f'klasse{c}']} in Klasse {c}" for c in (0, 1, 2) if z[f"klasse{c}"] > 0]
    return ", ".join(teile)


def zeile_nullen(res, p):
    return min(res["zeilen"], key=lambda z: abs(z["nullanteil"] - p))


def verdikt_nullen(res, p: float):
    """Experiment 2: Nullen in der alten Matrix. `res` = reihe_nullen; p = gewählter Nullanteil."""
    z = zeile_nullen(res, p)
    g = res["gesamt"]
    anz, n = z["anzahl"], res["n"]
    kopf = (f"Nullanteil {de(z['nullanteil'], 1)}, Randsummen {res['art']}, n = {n}, {anz} Instanzen, {res['form']}: {z['klasse0']} Instanzen ohne Lösung (Klasse 0), {z['klasse1']} nur mit Nullen lösbar "
            f"(Klasse 1), {z['klasse2']} streng positiv lösbar (Klasse 2). Konvergiert: {konvergenz_text(z)}.")
    mehr = (f" Über alle Nullanteile von {de(res['zeilen'][0]['nullanteil'], 1)} bis {de(res['zeilen'][-1]['nullanteil'], 1)} zusammen ({g['anzahl']} Instanzen): {g['klasse0']} ohne Lösung, "
            f"{g['klasse1']} in Klasse 1, {g['klasse2']} in Klasse 2; konvergiert {konvergenz_text(g)} "
            f"(Obergrenze {ganz(res['max_durchlaeufe'])} Durchläufe). „Ohne Lösung“ ist mit der Hall-Bedingung geprüft.")
    z0, z7 = res["zeilen"][0], res["zeilen"][-1]
    if z0["durchlaeufe2"] and z7["durchlaeufe2"]:
        mehr += (f" Median der Durchläufe in Klasse 2: {ganz(z0['durchlaeufe2'][0])} bei Nullanteil {de(z0['nullanteil'], 1)} und {ganz(z7['durchlaeufe2'][0])} bei {de(z7['nullanteil'], 1)} "
                 f"(Matrix-Vektor-Produkte doppelt so viele; bei {de(z7['nullanteil'], 1)} sind es nur {z7['klasse2']} Instanzen in Klasse 2).")
    if z["fehler1"]:
        f = z["fehler1"]
        k5 = str(res["max_durchlaeufe"])
        mehr += (f" Klasse 1: Randfehler (Median) {de_g(f['100'][0], 2)} nach 100, {de_g(f['1000'][0], 2)} nach 1 000 und {de_g(f[k5][0], 2)} nach {ganz(res['max_durchlaeufe'])} Durchläufen, "
                 "also in diesen Instanzen etwa 1/k; unter 1e-9 erreicht keiner dieser Läufe.")
    art = "error" if z["klasse0"] > z["anzahl"] / 2 else "warning" if z["klasse0"] + z["klasse1"] > 0 else "success"
    return art, kopf + mehr


def verdikt_kalibrierung(res):
    """Experiment 3: ε aus der beobachteten mittleren Reiseweite. `res` = reihe_kalibrierung."""
    n, anz = res["n"], res["anzahl"]
    if res["randfall"]:
        return "error", (f"Beobachtete Reiseweite 10 % unter dem LP-Minimum: kein ε kann sie erzeugen, denn kein Plan mit diesen Randsummen kostet weniger als das LP-Minimum. "
                         f"{res['konvergiert']} von {anz} Läufen konvergieren; bei {res['innen']} scheitert der innere Abgleich (β wächst; bei {res['innen_nan']} unterläuft exp(−βC) und die Rechnung endet in NaN, bei {res['innen'] - res['innen_nan']} "
                         f"erreicht er die Obergrenze von {ganz(C.MAX_INNEN_HYMAN)} Durchläufen), β steht am Ende bei "
                         f"{de_g(res['beta_ende'][0], 2)} (Median; Start {de_g(res['beta_start'][0], 2)}). n = {n}, {anz} Instanzen, höchstens {C.MAX_SCHRITTE_RANDFALL} Schritte.")
    sch, dl, pr, eo = res["schritte"], res["durchlaeufe"], res["produkte"], res["eps_opt"]
    start = "warm" if res["warm"] else "kalt"
    text = (f"θ = {de_g(res['theta'])} (n = {n}, {anz} Instanzen, {start} gestartet, {res['startregel']}): ε*/OPT im Median {eps_text(eo[0])} "
            f"[{eps_text(eo[1])}; {eps_text(eo[2])}]; {res['konvergiert']} von {anz} Läufen konvergieren, {band_ganz(sch)} Schritte, {band_ganz(dl)} Durchläufe ({band_ganz(pr)} Matrix-Vektor-Produkte). "
            f"Dasselbe θ ergibt bei verschiedenen Instanzen verschiedene ε*/OPT: das Verhältnis des größten zum kleinsten Wert ist hier {de_g(res['eps_verhaeltnis'], 2)} "
            f"(ein Extremwert aus {anz} Instanzen, er hängt vom Zufalls-Seed ab). Dasselbe ε/OPT bedeutet bei verschiedenen Instanzen auch verschiedenen Aufwand (Probe mit bekannter Wahrheit unten). "
            "Startwert und erster Schritt der Schleife sind eigene Wahl; Wilson und Hyman sind nicht im Original gelesen.")
    return "info", text


def verdikt_randsummen(res, grav, sigma: float):
    """Experiment 4: Randsummen stimmen, die Matrix nicht. `res` = reihe_randsummen, `grav` = reihe_gravitation (oder None)."""
    z = min(res["sigmas"], key=lambda r: abs(r["sigma"] - sigma))
    n, anz = res["n"], res["anzahl"]
    f = res["rang"]
    erste = (f"σ = {de_g(z['sigma'])}: Anteil der Menge in der falschen Zelle {band_g(z['abgeglichen'], 2)} nach dem Abgleich, {band_g(z['ohne'], 2)} ohne Abgleich, "
             f"{band_g(res['produktmatrix'], 2)} mit der Produktmatrix")
    if grav and grav["tv"]:
        erste += f", {band_g(grav['tv'], 2)} mit dem Gravitationskern (β aus der mittleren Reiseweite, in {grav['kalibriert']} von {grav['anzahl']} Instanzen berechnet)"
    erste += f" (Median [Min; Max], n = {n}, {anz} Instanzen, Wahrheit: {res['familie']})."
    zweite = f"Der größte Randsummenfehler nach einem konvergierten Abgleich liegt bei {de_g(res['rand_max'], 2)} (relativ zur kleinsten Randsumme; Maximum über alle Stufen und Instanzen, es hängt vom Zufalls-Seed ab; Schranke ist die Abbruchtoleranz 1e-12 geteilt durch die kleinste Randsumme)."
    if res["nicht_konvergiert"]:
        zweite += (f" {res['nicht_konvergiert']} der Läufe mit σ > 0 erreichen die Obergrenze von {ganz(res['max_durchlaeufe'])} Durchläufen; ihre Ergebnisse stehen in den Medianen, nicht in dieser Randsummenzahl.")
    hilft = "Der Abgleich hilft" if z["abgeglichen"][0] < z["ohne"][0] else "Der Abgleich hilft hier nicht"
    dritte = f"{hilft}, aber seine Ränder zeigen den Rest nicht: die Randsummen legen 2n − 1 = {f['festgelegt']} von {f['zahlen']} Zahlen fest, {f['frei']} bleiben frei."
    vierte = ("Gilt für synthetische Wahrheiten mit bekannter Struktur. Li/Chen/Jin (Preprint 2026, nur Abstract und Einleitung gelesen) vergleichen Verfahren an Streckenzählungen "
              "statt an Randsummen, ihr Befund überträgt sich nicht ohne Weiteres.")
    return "info", " ".join([erste, zweite, dritte, vierte])
