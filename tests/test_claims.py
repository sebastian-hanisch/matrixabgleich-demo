"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4 und §11a).

Die Messreihen liegen in `tools/messreihen.json` (erzeugt von `tools/messreihen.py` mit dem Code der Demo), die vorgerechnete Gravitationszeile von Experiment 4 in
`tools/vorgerechnet.json`. Dieser Test macht drei Dinge:
(1) die README-Zahlen werden aus der Datei mit denselben Formatfunktionen wie in der App gebildet (`readme_zeilen.py`) und müssen wörtlich in der README stehen (so kann keine
    Zahl der README von der Messung abweichen);
(2) die kleinen Reihen werden neu gerechnet und als Band mit der Datei verglichen (Zählungen auf 2 % und 2, Gleitkommazahlen relativ 1e-6), weil Gleitkomma-Summen auf anderen
    Plattformen abweichen können; die teure Gravitationszeile steht in zwei Dateien desselben Codes, die gegeneinander geprüft werden;
(3) Aussagen mit Absolutwörtern und Verweise der Tabelle „Zahl → Test“ werden geprüft."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import mab_constants as C
import mab_evaluation as E
from readme_zeilen import J, NULLEN_SCHLUESSEL, T1, T2, verdikte, zeilen_alle

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
V = json.loads((ROOT / "tools" / "vorgerechnet.json").read_text(encoding="utf-8"))
ZAEHLFELDER = {"konvergiert", "klasse0", "klasse1", "klasse2", "konv0", "konv1", "konv2", "nan0", "anzahl", "innen", "grenze", "nan", "kalibriert", "nicht_konvergiert",
               "wahrheit_ok", "schritte", "durchlaeufe", "produkte", "durchlaeufe2", "furness_konvergiert", "strafe_konvergiert", "festgelegt", "frei", "zahlen", "n", "max_durchlaeufe"}


def naehe(a, b, schluessel="", rel=1e-6, abs_=1e-12):
    """Band-Vergleich zweier Ergebnisse (verschachtelte Listen und Dicts): Zählfelder auf 2 % und 2, sonst relativ `rel`; Texte und Wahrheitswerte exakt."""
    if isinstance(b, dict):
        assert set(a) == set(b), (schluessel, set(a) ^ set(b))
        for k in b:
            naehe(a[k], b[k], k, rel, abs_)
    elif isinstance(b, list):
        assert len(a) == len(b), schluessel
        for x, y in zip(a, b):
            naehe(x, y, schluessel, rel, abs_)
    elif isinstance(b, (bool, str)) or b is None:
        assert a == b, schluessel
    elif schluessel in ZAEHLFELDER:
        assert abs(a - b) <= max(2.0, 0.02 * abs(b)), (schluessel, a, b)
    else:
        assert a == pytest.approx(b, rel=rel, abs=abs_), (schluessel, a, b)


GROSSE = ("verlauf", "bild")


def _kurz(x):
    if isinstance(x, dict):
        return {k: _kurz(v) for k, v in x.items() if k not in GROSSE}
    if isinstance(x, list):
        return [_kurz(v) for v in x]
    return x


def _neu(f):
    return _kurz(E.jsonbar(f()))


# ----------------------------------------------------------------------------------------------------------------------------------
# (1) die README enthält die Zahlen der Messreihen
# ----------------------------------------------------------------------------------------------------------------------------------

def test_readme_enthaelt_jede_zeile_der_messreihen():
    fehlt = [z for z in zeilen_alle() if z not in README]
    assert not fehlt, fehlt[:3]


def test_readme_enthaelt_jeden_satz_der_verdikte():
    fehlt = [k for k, v in verdikte().items() if v not in README]
    assert not fehlt, fehlt


def test_testzahl_der_readme_stimmt_mit_der_suite_ueberein():
    out = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    gesammelt = int(re.search(r"(\d+) tests? collected", out).group(1))
    genannt = int(re.search(r"\*\*(\d+) Tests\*\*", README).group(1))
    assert gesammelt == genannt, (gesammelt, genannt)


def test_tabelle_zahl_test_verweist_auf_vorhandene_tests():
    verweise = re.findall(r"`tests/(\w+\.py)::(\w+)`", README)
    assert len(verweise) >= 20
    for datei, name in verweise:
        quelle = (ROOT / "tests" / datei).read_text(encoding="utf-8")
        assert re.search(rf"^def {name}\(", quelle, re.M), (datei, name)


def test_readme_nennt_jede_python_datei_des_wurzelordners_und_der_tools():
    for pfad in list(ROOT.glob("*.py")) + list((ROOT / "tools").glob("*.py")):
        assert pfad.name in README, pfad.name


def test_readme_gegenrede_nennt_nur_was_der_abstract_sagt_und_quelle_ist_nicht_als_gesamtsieger_zitiert():
    absatz = next(z for z in README.splitlines() if z.startswith("- **Die Gegenrede:**"))
    assert "65 von 66" in absatz and "45 von 66" in absatz and "2611.09678" in absatz and "Abstract" in absatz
    assert "am besten ab" not in absatz and "Entropiestrafe" not in absatz
    assert "5·10⁸" in README and "10⁹ Durchläufen" not in README
    assert "Idel nennt Raking nicht" in README and "Idel 2016, gelesen)" not in README


def test_readme_ohne_planbegriffe_und_ohne_verbotene_woerter():
    for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "Vormessung", "V1", "V3", "Messlücke", "ZUSCHNITT", "läuft lokal", "Saatwert", "Schwanz"):
        assert verboten not in README, verboten
    assert not re.search(r"\bträgt\b|\btragen\b|\btragende", README)
    assert not re.search(r"\bSeed\b", README.replace("Zufalls-Seed", ""))
    assert "Wilson zeigte" not in README and "Furness zeigte" not in README and "Deming" in README
    assert "Neuheit wird nicht behauptet" in README


def test_keine_scheinpruefungen_in_den_tests():
    muster = "or " + "True"
    for pfad in (ROOT / "tests").glob("*.py"):
        assert muster not in pfad.read_text(encoding="utf-8"), pfad.name


# ----------------------------------------------------------------------------------------------------------------------------------
# (3) Aussagen mit Absolutwörtern gegen die Zählung über alle Zellen
# ----------------------------------------------------------------------------------------------------------------------------------

def test_in_keiner_nullen_tabelle_fehlt_mehr_als_ein_konvergierter_lauf_in_klasse_2():
    for schluessel in ("beliebig", "breit", "gleich"):
        res = J["nullen"][NULLEN_SCHLUESSEL[schluessel]]
        assert sum(z["klasse2"] - z["konv2"] for z in res["zeilen"]) <= 1, schluessel
        assert res["gesamt"]["konv0"] == 0 and res["gesamt"]["konv1"] == 0, schluessel          # nie in Klasse 0 oder 1


def test_nan_anteil_in_klasse_0_haengt_an_der_art_der_randsummen_und_die_matrix_form_hat_kein_nan():
    """Die Bildunterschrift von Experiment 2 sagt „meist NaN“ nur für beliebige und gleiche Randsummen; bei Wachstumsfaktoren endet höchstens ein Teil in NaN."""
    anteil = {k: sum(z["nan0"] for z in J["nullen"][NULLEN_SCHLUESSEL[k]]["zeilen"]) / sum(z["klasse0"] for z in J["nullen"][NULLEN_SCHLUESSEL[k]]["zeilen"])
              for k in ("beliebig", "gleich", "mild", "breit", "stark")}
    assert anteil["beliebig"] > 0.9 and anteil["gleich"] == 1.0
    assert all(0.05 < anteil[k] < 0.7 for k in ("mild", "breit", "stark")), anteil
    assert sum(z["nan0"] for z in J["nullen"][NULLEN_SCHLUESSEL["beliebig_matrix"]]["zeilen"]) == 0


def test_klasse_1_konvergiert_in_keinem_lauf_und_der_randfehler_faellt_in_den_gemessenen_instanzen_wie_1_durch_k():
    z = J["nullen"][NULLEN_SCHLUESSEL["gleich"]]
    assert z["gesamt"]["klasse1"] == 190 and z["gesamt"]["konv1"] == 0
    for zeile in z["zeilen"]:
        if zeile["fehler1"]:
            f = zeile["fehler1"]
            assert 8 < f["100"][0] / f["1000"][0] < 12 and 4 < f["1000"][0] / f["5000"][0] < 6          # Faktor 10 und 5 = 1/k, Median


def test_andere_zufalls_seeds_stuetzen_die_baender_der_readme():
    s = J["andere_seeds"]
    assert s["seeds"] == [2026, 2027, 2028, 2029]
    assert 3.0 < min(s["verhaeltnis_theta01"]) and max(s["verhaeltnis_theta01"]) < 9.0
    assert s["verhaeltnis_theta01"][0] == pytest.approx(J["kalibrierung"]["0.1|kalt"]["eps_verhaeltnis"], rel=1e-6)          # Seed 2026 ist die Reihe der README
    assert 0.15 < min(x for v in s["warm_spart"] for x in v) and max(x for v in s["warm_spart"] for x in v) < 0.30
    assert all(0 <= x <= 3 for v in s["nullen_fehlt"].values() for x in v)
    assert sum(z["klasse2"] - z["konv2"] for z in J["nullen"][NULLEN_SCHLUESSEL["beliebig"]]["zeilen"]) == s["nullen_fehlt"][C.RS_BELIEBIG][0]


def test_anderer_zufalls_seed_neu_gerechnet_stimmt_mit_der_datei_ueberein():
    r = E.reihe_kalibrierung(0.1, C.BEOB_THETA, C.START_PROP, False, 12, seed=2027)
    assert r["eps_verhaeltnis"] == pytest.approx(J["andere_seeds"]["verhaeltnis_theta01"][1], rel=1e-3)
    assert r["eps_verhaeltnis"] != pytest.approx(J["kalibrierung"]["0.1|kalt"]["eps_verhaeltnis"], rel=1e-3)         # der Zufalls-Seed wirkt wirklich


def test_matrix_form_und_u_v_form_zaehlen_dieselben_klassen_und_konvergenzen():
    uv, mat = J["nullen"][NULLEN_SCHLUESSEL["beliebig"]], J["nullen"][NULLEN_SCHLUESSEL["beliebig_matrix"]]
    for a, b in zip(uv["zeilen"], mat["zeilen"]):
        assert [a[k] for k in ("klasse0", "klasse1", "klasse2", "konv0", "konv1", "konv2")] == [b[k] for k in ("klasse0", "klasse1", "klasse2", "konv0", "konv1", "konv2")]


def test_summen_probe_furness_und_strafe_zeilenfehler_haben_dieselbe_bezugsgroesse():
    s = J["strafe"]["summen"]
    assert s["furness_fehler"][0] == pytest.approx(0.05 / 1.05, rel=1e-6)          # |B − A| geteilt durch die Summe der Zeilenziele 1,05
    assert s["strafe_fehler"][0] < s["furness_fehler"][0]


def test_ungleiche_summen_konvergieren_in_keinem_lauf_und_der_fehler_ist_die_differenz():
    for z in J["summen"]["zeilen"]:
        if z["delta"] > 0:
            assert z["konvergiert"] == 0 and z["fehler"] == pytest.approx([z["delta"]] * 3, rel=1e-6)
        else:
            assert z["konvergiert"] == z["anzahl"]


def test_strafverfahren_konvergiert_in_jedem_lauf_und_wird_mit_tau_genauer():
    zs = J["strafe"]["zeilen"]
    assert all(z["konvergiert"] == J["strafe"]["anzahl"] for z in zs)
    assert [z["abweichung"][0] for z in zs] == sorted((z["abweichung"][0] for z in zs), reverse=True)
    assert [z["durchlaeufe"][0] for z in zs] == sorted(z["durchlaeufe"][0] for z in zs)
    assert J["strafe"]["summen"]["furness_konvergiert"] == 0 and J["strafe"]["summen"]["strafe_konvergiert"] == J["strafe"]["anzahl"]


def test_kalibrierung_konvergiert_in_jedem_lauf_der_erreichbaren_beobachtung_und_nie_im_randfall():
    for th in C.THETA_STUFEN:
        r = J["kalibrierung"][f"{th}|kalt"]
        assert r["konvergiert"] == r["anzahl"] and r["genauigkeit"] < 1e-6
    assert J["kalibrierung"]["rand"]["konvergiert"] == 0
    rand = J["kalibrierung"]["rand"]
    assert rand["innen"] == 20 and 0 < rand["innen_nan"] < rand["innen"]               # nicht alle Fälle sind Unterlauf: ein Teil erreicht die Obergrenze der inneren Durchläufe
    assert J["kalibrierung"]["0.1|warm"]["schritte"] == J["kalibrierung"]["0.1|kalt"]["schritte"]            # Warmstart ändert die Schritte nicht
    assert J["kalibrierung"]["0.1|warm"]["durchlaeufe"][0] < J["kalibrierung"]["0.1|kalt"]["durchlaeufe"][0]
    steps = [J["kalibrierung"][f"{th}|kalt"]["durchlaeufe"][0] for th in C.THETA_STUFEN]
    assert steps == sorted(steps, reverse=True)                                                          # je näher am Produktmaß, desto weniger Durchläufe


def test_experiment_4_abgleich_hilft_in_jeder_gestoerten_stufe_und_randfehler_ist_klein():
    for fam in (T1, T2):
        for z in J["randsummen"][fam]["sigmas"]:
            if z["sigma"] > 0:
                assert z["abgeglichen"][0] < z["ohne"][0], (fam, z["sigma"])
        assert J["randsummen"][fam]["rand_max"] < 1e-9
    assert [z["abgeglichen"][0] for z in J["randsummen"][T1]["sigmas"]] == sorted(z["abgeglichen"][0] for z in J["randsummen"][T1]["sigmas"])


def test_vorgerechnete_gravitationszeile_stimmt_mit_den_messreihen_ueberein():
    for fam in C.FAMILIEN:
        naehe(V[f"reihe_gravitation|anzahl=50,familie={fam!r},n=12"], J["gravitation"][fam], "gravitation")
        assert E.gravitation_v(fam, 12, 50) == V[f"reihe_gravitation|anzahl=50,familie={fam!r},n=12"]


# ----------------------------------------------------------------------------------------------------------------------------------
# (2) kleine Reihen neu gerechnet
# ----------------------------------------------------------------------------------------------------------------------------------

def test_kleine_reihen_neu_gerechnet_stimmen_als_band_mit_der_datei():
    naehe(_neu(E.reihe_summen), J["summen"], "summen")
    naehe(_neu(E.reihe_strafe), J["strafe"], "strafe")
    for key, (art, form) in {NULLEN_SCHLUESSEL["beliebig"]: (C.RS_BELIEBIG, C.FORM_UV), NULLEN_SCHLUESSEL["gleich"]: (C.RS_GLEICH, C.FORM_UV)}.items():
        naehe(_neu(lambda art=art, form=form: E.reihe_nullen(art, 100, form)), J["nullen"][key], key)
    for th in (0.01, 0.1, 0.9):
        naehe(_neu(lambda th=th: E.reihe_kalibrierung(th, C.BEOB_THETA, C.START_PROP, False, 12)), J["kalibrierung"][f"{th}|kalt"], f"kalibrierung {th}")
    naehe(_neu(lambda: E.reihe_kalibrierung(0.1, C.BEOB_THETA, C.START_FAKTOR, False, 12)), J["kalibrierung"]["0.1|faktor"], "faktor")
    naehe(_neu(lambda: E.reihe_wahrheit(startregel=C.START_PROP)), J["wahrheit"][C.START_PROP], "wahrheit")
    naehe(_neu(lambda: E.reihe_randsummen(T2, 12, 50)), J["randsummen"][T2], "randsummen T2")


def test_die_zahlen_der_handbeispiele_in_der_datei():
    h = J["hand"]
    assert 2500 < h["nan_durchlauf"] < 4000 and h["matrix_fehler"] == pytest.approx(2.0, abs=1e-3) and h["matrix_status"].startswith("nicht konvergiert")
    assert J["gleichheit"]["groesste"] < 1e-12 and J["gleichheit"]["anzahl"] == 20
