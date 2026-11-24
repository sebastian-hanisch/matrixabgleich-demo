"""Kennzahlen, Reihen und Verdikte: Verdikte an handgesetzten Zeilen (Schwellen und jede Zahl im Text), Reihen klein, Primäransicht je Preset."""

from types import SimpleNamespace

import numpy as np
import pytest

import mab_constants as C
import mab_evaluation as E
import mab_scaling as S
import mab_verdikte as V


# ----------------------------------------------------------------------------------------------------------------------------------
# Formatierung
# ----------------------------------------------------------------------------------------------------------------------------------

def test_formatfunktionen():
    assert E.de(0.1234, 2) == "0,12" and E.de_g(1234.5678, 3) == "1,23e3" and E.de_g(12.34, 3) == "12,3"
    assert E.de_g(0.0) == "0" and E.de_g(0.00047, 2) == "0,00047" and E.de_g(1.2e-11, 2) == "1,2e-11"
    assert E.ganz(1234567) == "1 234 567" and E.ganz(0.4) == "0"
    assert E.prozent(0.1234, 1) == "12,3 %" and E.band_ganz([6, 5, 8]) == "6 [5; 8]" and E.band_g([0.5, 0.25, 1.0], 2) == "0,5 [0,25; 1]"
    assert E.band([3, 1, 2]) == [2.0, 1.0, 3.0] and np.isnan(E.band([])[0])
    assert E.jsonbar({"a": np.array([1, 2]), "b": (np.float64(1.5), np.int64(3), np.bool_(True))}) == {"a": [1, 2], "b": [1.5, 3, True]}


def test_settings_aus_zustand_und_standard():
    assert E.settings_aus_zustand(C.DEFAULTS) == E.Settings()
    s = E.settings_aus_zustand({"n_slider": 9.0, "theta": 0.2})
    assert s.n == 9 and isinstance(s.n, int) and s.theta == 0.2 and s.quelle == C.QUELLE_ZONEN


# ----------------------------------------------------------------------------------------------------------------------------------
# Verdikte an handgesetzten Zeilen
# ----------------------------------------------------------------------------------------------------------------------------------

def _summen(fehler_gleich=True):
    zeilen = [dict(delta=0.0, konvergiert=20, anzahl=20, fehler=[1e-10, 1e-11, 9e-10], durchlaeufe=[6, 5, 6], nan=0)]
    for d in (0.001, 0.01, 0.1):
        zeilen.append(dict(delta=d, konvergiert=0, anzahl=20, fehler=[d, d, d * (1 if fehler_gleich else 1.5)], durchlaeufe=None, nan=0))
    return dict(n=8, anzahl=20, max_durchlaeufe=5000, zeilen=zeilen)


def test_verdikt_summen_nullstufe_und_ungleiche_stufen():
    art, text = V.verdikt_summen(_summen(), 0.0)
    assert art == "success" and "20 von 20" in text and "0 von 60" in text and "6 [5; 6]" in text
    art, text = V.verdikt_summen(_summen(), 0.01)
    assert art == "warning" and "(B − A)/A = 0,01" in text and "0 von 20" in text and "0 von 60" in text and "genau diese Differenz" in text and "5 000" in text
    assert "bei diesen dichten Matrizen, allgemein ist er mindestens |B − A|" in text         # die Aussage gilt nur für dichte Matrizen
    _, text = V.verdikt_summen(_summen(fehler_gleich=False), 0.01)
    assert "genau diese Differenz" not in text                          # die Aussage entfällt, wenn kleinster und größter Fehler verschieden sind


def _strafe(dev=(0.2, 0.04, 0.004, 0.0004), its=(16, 100, 800, 6400), konv=20):
    zeilen = [dict(tau=t, abweichung=[d, d * 0.9, d * 1.1], durchlaeufe=[i, i - 1, i + 1], konvergiert=konv, rand=[0, 0, 0]) for t, d, i in zip(C.STRAFE_TAUS, dev, its)]
    return dict(n=10, anzahl=20, zeilen=zeilen, summen=dict(plus=0.05, furness_konvergiert=0, furness_fehler=[0.05, 0.05, 0.05], strafe_konvergiert=20, strafe_fehler=[0.025, 0.02, 0.03], tau=100.0))


def test_verdikt_strafe_faktoren_und_gewaehltes_tau():
    dev, its = V.strafe_faktoren(_strafe())
    assert dev == pytest.approx([10.0, 10.0]) and its == pytest.approx([8.0, 8.0])
    art, text = V.verdikt_strafe(_strafe(), C.TAU_AUS)
    assert art == "info" and "konvergieren 20 von 20 Läufen" in text and "Faktor 10 bis 10" in text and "Faktor 8 bis 8" in text and "0 von 20 Läufen (Zeilenfehler 0,05)" in text
    assert "in diesem Bereich etwa 1/τ" in text and "in diesem Bereich etwa τ)" in text and "zwei Schritte" in text and "beide Zeilenfehler sind relativ zur Summe der Zeilenziele" in text
    _, text = V.verdikt_strafe(_strafe(), "100")
    assert text.startswith("τ = 100: größte relative Abweichung vom exakten Furness 0,004 (Median), 800 [799; 801] Durchläufe")
    _, text = V.verdikt_strafe(_strafe(konv=18), C.TAU_AUS)
    assert "18 von 20 Läufen" in text


def test_reihe_strafe_klein_beide_zeilenfehler_der_summen_probe_sind_relativ_zur_summe_der_zeilenziele():
    r = E.reihe_strafe(taus=(10.0,), n=6, anzahl=3)
    s = r["summen"]
    assert s["furness_konvergiert"] == 0 and s["furness_fehler"] == pytest.approx([0.05 / 1.05] * 3, rel=1e-6)          # Furness: |B − A| geteilt durch 1,05
    assert s["strafe_konvergiert"] == 3 and s["strafe_fehler"][0] < s["furness_fehler"][0]


def _nullen(k1=0):
    zeilen = []
    for i, p in enumerate(C.NULLANTEIL_STUFEN):
        k0 = [0, 0, 1, 3, 20, 70, 90, 97][i]
        zeilen.append(dict(nullanteil=p, anzahl=100, klasse0=k0, klasse1=k1, klasse2=100 - k0 - k1, konv0=0, konv1=0, konv2=100 - k0 - k1, nan0=k0,
                           durchlaeufe2=[7 + 10 * i, 5, 90] if 100 - k0 - k1 else None,
                           fehler1={"100": [0.0022, 0.002, 0.003], "1000": [0.00022, 0.0002, 0.0003], "5000": [4.4e-5, 4e-5, 5e-5]} if k1 else None))
    gesamt = {f"klasse{c}": sum(z[f"klasse{c}"] for z in zeilen) for c in (0, 1, 2)}
    gesamt.update({f"konv{c}": sum(z[f"konv{c}"] for z in zeilen) for c in (0, 1, 2)})
    gesamt["anzahl"] = 800
    return dict(art=C.RS_BELIEBIG, form=C.FORM_UV, n=8, anzahl=100, max_durchlaeufe=5000, zeilen=zeilen, gesamt=gesamt)


def test_verdikt_nullen_art_zaehlungen_und_klasse_eins():
    res = _nullen()
    art, text = V.verdikt_nullen(res, 0.5)
    assert art == "error" and "70 Instanzen ohne Lösung (Klasse 0)" in text and "0 von 70 in Klasse 0, 30 von 30 in Klasse 2" in text
    assert "Klasse 1" in text and "von 0 in Klasse 1" not in text               # leere Klasse steht nicht als „0 von 0“ da
    assert "(800 Instanzen)" in text and "bei 0,7 sind es nur 3 Instanzen in Klasse 2" in text and "Hall-Bedingung" in text
    art, text = V.verdikt_nullen(res, 0.1)
    assert art == "success" and "0 Instanzen ohne Lösung" in text
    art, text = V.verdikt_nullen(res, 0.3)
    assert art == "warning"                                                     # wenige unzulässige Instanzen: weder grün noch rot
    _, text = V.verdikt_nullen(_nullen(k1=5), 0.5)
    assert "1/k" in text and "0,0022 nach 100" in text and "4,4e-5 nach 5 000" in text


def _kalib(randfall=False, warm=False):
    if randfall:
        return dict(theta=0.1, randfall=True, n=12, anzahl=20, warm=False, startregel=C.START_PROP, konvergiert=0, innen=20, innen_nan=16, grenze=0, schritte=None, beta_ende=[4100.0, 200.0, 1e5],
                    beta_start=[12.0, 7.0, 24.0])
    return dict(theta=0.1, randfall=False, n=12, anzahl=20, warm=warm, startregel=C.START_PROP, konvergiert=20, innen=0, grenze=0, schritte=[6.0, 5.0, 8.0],
                durchlaeufe=[644.0, 355.0, 1088.0], produkte=[1288.0, 710.0, 2176.0], eps_opt=[1.08, 0.54, 1.99], eps_verhaeltnis=3.7)


def test_verdikt_kalibrierung_und_randfall():
    art, text = V.verdikt_kalibrierung(_kalib())
    assert art == "info" and "θ = 0,1" in text and "kalt gestartet" in text and "6 [5; 8] Schritte" in text and "644 [355; 1 088] Durchläufe" in text and "1 288 [710; 2 176]" in text
    assert "ε*/OPT im Median 1,08 [0,54; 1,99]" in text and "hier 3,7" in text and "nicht im Original gelesen" in text
    assert "Dasselbe θ ergibt bei verschiedenen Instanzen verschiedene ε*/OPT" in text and "hängt vom Zufalls-Seed ab" in text         # das Verhältnis belegt verschiedene ε/OPT, nicht verschiedenen Aufwand
    assert "warm gestartet" in V.verdikt_kalibrierung(_kalib(warm=True))[1]
    art, text = V.verdikt_kalibrierung(_kalib(randfall=True))
    assert art == "error" and "0 von 20 Läufen konvergieren" in text and "bei 20 scheitert der innere Abgleich" in text and "4,1e3" in text and "40 Schritte" in text
    assert "bei 16 unterläuft exp(−βC) und die Rechnung endet in NaN, bei 4 erreicht er die Obergrenze von 50 000 Durchläufen" in text         # Ursache aufgeteilt, nicht „bei 20 Unterlauf“


def _randsummen(abgeglichen=0.11, ohne=0.23, nicht=0):
    return dict(familie=C.FAMILIE_T1, n=12, anzahl=50, produktmatrix=[0.67, 0.44, 0.83], rand_max=1e-11, nicht_konvergiert=nicht, max_durchlaeufe=50000,
                rang=dict(festgelegt=23, frei=121, zahlen=144), sigmas=[dict(sigma=0.6, abgeglichen=[abgeglichen, 0.01, 0.2], ohne=[ohne, 0.15, 0.3], rand=1e-11)])


def test_verdikt_randsummen_hilft_oder_hilft_nicht_und_nennt_die_einschraenkung():
    grav = dict(familie=C.FAMILIE_T1, n=12, anzahl=50, kalibriert=49, tv=[0.095, 0.03, 0.17])
    art, text = V.verdikt_randsummen(_randsummen(), grav, 0.6)
    assert art == "info" and "σ = 0,6" in text and "0,11 [0,01; 0,2] nach dem Abgleich, 0,23 [0,15; 0,3] ohne Abgleich" in text and "0,67 [0,44; 0,83] mit der Produktmatrix" in text
    assert "in 49 von 50 Instanzen berechnet" in text and "2n − 1 = 23 von 144" in text and "121 bleiben frei" in text and "Preprint 2026" in text and "1e-11" in text
    assert "Maximum über alle Stufen und Instanzen" in text and "hängt vom Zufalls-Seed ab" in text and "1e-12 geteilt durch die kleinste Randsumme" in text
    assert "am besten" not in text and "Entropie" not in text                      # nur belegen, was der Abstract sagt
    assert "Der Abgleich hilft," in text
    assert "hilft hier nicht" in V.verdikt_randsummen(_randsummen(abgeglichen=0.3), None, 0.6)[1]
    assert "Obergrenze von 50 000" in V.verdikt_randsummen(_randsummen(nicht=2), None, 0.6)[1] and "Obergrenze" not in V.verdikt_randsummen(_randsummen(), None, 0.6)[1]


# ----------------------------------------------------------------------------------------------------------------------------------
# Reihen (klein)
# ----------------------------------------------------------------------------------------------------------------------------------

def test_reihe_summen_klein():
    res = E.reihe_summen(deltas=(0.0, 0.1), n=5, anzahl=4, max_durchlaeufe=200)
    z0, z1 = res["zeilen"]
    assert z0["konvergiert"] == 4 and z1["konvergiert"] == 0
    assert z1["fehler"][1] == pytest.approx(0.1, rel=1e-9) == z1["fehler"][2] and len(z1["verlauf_ks"]) == len(z1["verlauf"])


def test_reihe_nullen_klein_klassen_zaehlen_sich_auf_die_instanzen_auf():
    res = E.reihe_nullen(C.RS_BELIEBIG, 12, C.FORM_UV, stufen=(0.0, 0.5), n=6, max_durchlaeufe=500)
    for z in res["zeilen"]:
        assert z["klasse0"] + z["klasse1"] + z["klasse2"] == 12 and z["konv0"] == 0 and z["konv1"] == 0
    assert res["zeilen"][0]["klasse0"] == 0 and res["gesamt"]["anzahl"] == 24
    res = E.reihe_nullen(C.RS_GLEICH, 12, C.FORM_UV, stufen=(0.5,), n=6, max_durchlaeufe=500)
    assert res["zeilen"][0]["traeger_gleich_klasse"] is True


def test_reihe_kalibrierung_klein_und_randfall():
    r = E.reihe_kalibrierung(0.2, C.BEOB_THETA, C.START_PROP, False, 6, anzahl=4)
    assert r["konvergiert"] == 4 and r["genauigkeit"] < 1e-6 and r["spur"]["schritte"] == len(r["spur"]["beta"]) > 0
    assert r["eps_opt"][1] <= r["eps_opt"][0] <= r["eps_opt"][2]
    r = E.reihe_kalibrierung(0.2, C.BEOB_UNTER, C.START_PROP, False, 6, anzahl=3)
    assert r["konvergiert"] == 0 and r["innen"] + r["grenze"] == 3 and r["schritte"] is None
    assert 0 <= r["innen_nan"] <= r["innen"]                                  # NaN-Fälle sind eine Teilmenge der gescheiterten inneren Abgleiche


def test_reihe_wahrheit_findet_epsilon_wieder_und_warm_spart():
    w = E.reihe_wahrheit(rs=(1.0,), n=6, anzahl=4)
    z = w["zeilen"][0]
    assert z["kalt"]["konvergiert"] == 4 and z["kalt"]["rueck"] < 1e-4 and z["warm"]["durchlaeufe"][0] <= z["kalt"]["durchlaeufe"][0] and 0 <= z["warm_spart"] < 1


def test_reihe_randsummen_klein_sigma_null_ist_trivial_und_der_fehler_waechst_mit_sigma():
    r = E.reihe_randsummen(C.FAMILIE_T2, 6, 5)
    s = {z["sigma"]: z for z in r["sigmas"]}
    assert s[0.0]["abgeglichen"][0] < 1e-12 and s[0.0]["ohne"][0] < 1e-12
    assert [s[x]["abgeglichen"][0] for x in (0.1, 0.3, 0.6, 1.0)] == sorted(s[x]["abgeglichen"][0] for x in (0.1, 0.3, 0.6, 1.0))
    assert all(z["abgeglichen"][0] < z["ohne"][0] for z in r["sigmas"] if z["sigma"] > 0) and r["rang"] == dict(festgelegt=11, frei=25, zahlen=36)
    g = E.reihe_gravitation(C.FAMILIE_T1, 6, 3)
    r = E.reihe_randsummen(C.FAMILIE_T1, 6, 3)
    assert g["kalibriert"] >= 1 and g["tv"][0] < r["produktmatrix"][0]


# ----------------------------------------------------------------------------------------------------------------------------------
# Primäransicht und Presets
# ----------------------------------------------------------------------------------------------------------------------------------

def _preset(name):
    return E.settings_aus_zustand(C.PRESETS[name])


def test_klasse_1_wird_nur_als_gemessene_beobachtung_formuliert():
    """Klasse 1: „etwa 1/k“ gilt in den gemessenen Fällen, nicht für jede Aufgabe (Verdikt, Antwort der Kernfrage 1)."""
    erg = SimpleNamespace(b=np.full(3, 1 / 3), klasse=1, gesamt=1.0, nullanteil_real=0.5,
                          ab=SimpleNamespace(status=S.STATUS_GRENZE, fehler_verlauf=np.array([1e-4]), durchlaeufe=5000))
    art, text = V.verdikt_ansicht(erg)
    assert art == "warning" and "Klasse 1" in text and "in den gemessenen Fällen fällt der Randfehler etwa wie 1/k" in text and "Obergrenze von 5 000" in text
    assert "in den gemessenen Fällen fällt der Randfehler etwa wie 1/k" in V.antwort_loesung(erg)
    assert "konvergiert linear" in V.antwort_loesung(SimpleNamespace(b=np.full(3, 1 / 3), klasse=2))


def test_jedes_preset_hat_seine_erwartete_verdiktzeile_ohne_app():
    erwartet = {"Zonen 6, Furness": ("success", "Der Abgleich konvergiert nach"), "Von Hand 2 × 2": ("success", "Der Abgleich konvergiert nach 5 Durchläufen"),
                "Summen +5 %": ("warning", "Keine Lösung: die Spaltensummen sind um 5,0 %"), "Nullanteil 0,5": ("error", "Keine Lösung: bei Durchlauf"),
                "ε kalibrieren": ("success", "Der Abgleich konvergiert nach"), "Alte Matrix σ = 0,6": ("success", "Der Abgleich konvergiert nach")}
    assert set(erwartet) == set(C.PRESETS)
    for name, (art_soll, anfang) in erwartet.items():
        art, text = V.verdikt_ansicht(E.ansicht_ergebnis(_preset(name)))
        assert art == art_soll and text.startswith(anfang), (name, art, text[:80])


def test_preset_von_hand_5_durchlaeufe_und_die_zahlen_der_handrechnung():
    erg = E.ansicht_ergebnis(_preset("Von Hand 2 × 2"))
    assert erg.ab.durchlaeufe == 5 and erg.ab.produkte == 10 and erg.klasse == 2
    assert E.plan_fahrten(erg, 1) == pytest.approx(np.array([[175 / 89, 210 / 71], [270 / 89, 216 / 71]]), rel=1e-12)
    assert E.plan_fahrten(erg, 0) == pytest.approx(np.array([[1.0, 2.0], [3.0, 4.0]]) * 11 / 10)    # Ausgangsmatrix im Maßstab der Zeilenziele
    art, text = V.verdikt_ansicht(erg)
    assert art == "success" and "nach 5 Durchläufen" in text and "10 Matrix-Vektor-Produkte" in text


def test_preset_zonen_6_furness_konvergiert_und_hat_loesung():
    erg = E.ansicht_ergebnis(_preset("Zonen 6, Furness"))
    art, text = V.verdikt_ansicht(erg)
    assert art == "success" and erg.klasse == 2 and "Ja, mit einer Matrix" in V.antwort_loesung(erg)
    assert erg.eps_zurueck is not None and erg.eps_zurueck > 1
    P = E.plan_fahrten(erg, erg.ab.durchlaeufe)
    assert P.sum(1) == pytest.approx(erg.a * erg.gesamt, rel=1e-6) and P.sum(0) == pytest.approx(erg.b * erg.gesamt, rel=1e-6)


def test_preset_summen_plus_5_prozent_meldet_keine_loesung():
    erg = E.ansicht_ergebnis(_preset("Summen +5 %"))
    art, text = V.verdikt_ansicht(erg)
    assert art == "warning" and "5,0 % größer" in text and "Keine Lösung" in text and erg.klasse is None
    assert erg.ab.status == S.STATUS_GRENZE and erg.ab.fehler_verlauf[-1] == pytest.approx(0.05, rel=1e-6)
    assert V.antwort_loesung(erg).startswith("Nein: die Summen")


def test_preset_nullanteil_gleich_halb_hat_keine_loesung_und_meldet_nan():
    erg = E.ansicht_ergebnis(_preset("Nullanteil 0,5"))
    art, text = V.verdikt_ansicht(erg)
    assert art == "error" and erg.klasse == 0 and erg.ab.status == S.STATUS_NAN and "Hall-Bedingung" in text and "NaN" in text
    assert V.antwort_loesung(erg).startswith("Nein: die Nullstruktur")


def test_preset_epsilon_kalibrieren_rechnet_das_vorgegebene_epsilon_zurueck():
    s = _preset("ε kalibrieren")
    erg = E.ansicht_ergebnis(s)
    assert V.verdikt_ansicht(erg)[0] == "success"
    assert erg.eps_zurueck == pytest.approx(s.eps_rel, rel=1e-4) and erg.eps == pytest.approx(s.eps_rel * erg.opt)


def test_preset_alte_matrix_sigma_gleich_0_6_setzt_nur_experiment_4():
    s = _preset("Alte Matrix σ = 0,6")
    assert s.sigma == 0.6 and E.Settings(**{**E.Settings().__dict__, "sigma": 0.6}) == s
    ohne = [name for name in C.PRESETS if _preset(name).sigma == 0.6]
    assert ohne == ["Alte Matrix σ = 0,6"]


def test_kein_preset_ohne_effekt_meldet_einen_effekt():
    """Die Presets „Zonen 6, Furness“ und „Alte Matrix σ = 0,6“ ändern die Primäransicht nicht: ihr Verdikt ist das der Standardansicht (konvergiert), nie „keine Lösung“."""
    standard = V.verdikt_ansicht(E.ansicht_ergebnis(E.Settings()))
    for name in ("Zonen 6, Furness", "Alte Matrix σ = 0,6"):
        assert V.verdikt_ansicht(E.ansicht_ergebnis(_preset(name))) == standard and standard[0] == "success"


def test_nullanteil_aenderung_aendert_die_instanz_aber_nicht_die_zonen():
    a = E.ansicht_ergebnis(E.Settings(nullanteil=0.0))
    b = E.ansicht_ergebnis(E.Settings(nullanteil=0.3))
    assert np.array_equal(a.an.kosten, b.an.kosten) and (b.an.T0 == 0).any() and not (a.an.T0 == 0).any()


def test_obergrenze_wird_gemeldet_nicht_still_beendet():
    erg = E.ansicht_ergebnis(E.Settings(delta=0.001))
    art, text = V.verdikt_ansicht(erg)
    assert art == "warning" and "Obergrenze von 5 000 Durchläufen" in text
