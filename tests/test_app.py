"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, setzt jeden Regler auf seine
Grenzen beziehungsweise auf jede Option und prüft den Permalink; kein Python-Fehler darf auftreten (insbesondere `StreamlitDuplicateElementId` bei Diagrammen mit gleichem Inhalt)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import mab_constants as C
import mab_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox")
SCHWER = {"n_mess", "familie", "inst4"}      # lösen die Gravitationszeile von Experiment 4 live aus (mehrere Sekunden): in eigenen Tests


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def test_app_loads_without_exception_and_zeigt_alle_ueberschriften():
    at = _fresh_app()
    unter = [h.value for h in at.subheader]
    for teil in ("Vier Kernfragen", "Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4", "Wo die Annahmen enden"):
        assert any(teil in u for u in unter), teil
    assert any("Mathematische Formulierung" in e.label for e in at.expander)
    assert [e.value.startswith("Nullanteil 0,5, Randsummen beliebig") for e in at.error] == [True]       # nur das Verdikt von Experiment 2 ist bei den Standardwerten rot (68 von 100 ohne Lösung)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().sidebar.button]
    assert len(labels) == len(C.PRESETS) + 1
    for index in range(len(labels)):
        at = _fresh_app()
        at.sidebar.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (labels[index], [str(e) for e in at.exception])


def test_jedes_preset_zeigt_sein_erwartetes_verdikt():
    erwartet = {"Zonen 6, Furness": "konvergiert nach", "Von Hand 2 × 2": "konvergiert nach 5 Durchläufen", "Summen +5 %": "Keine Lösung: die Spaltensummen sind um 5,0 %",
                "Nullanteil 0,5": "Keine Lösung: bei Durchlauf", "ε kalibrieren": "konvergiert nach", "Alte Matrix σ = 0,6": "konvergiert nach"}
    for name, teil in erwartet.items():
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        meldungen = [m.value for m in list(at.success) + list(at.warning) + list(at.error)]
        assert any(teil in m for m in meldungen), (name, meldungen[:3])


def test_every_slider_and_number_input_at_its_min_and_max_does_not_raise():
    for key in ("n_slider", "seed_input"):
        spec = P.SETTING_SPECS[key]
        for wert in (spec.lo, spec.hi):
            at = _fresh_app()
            _regler(at, key).set_value(wert).run(timeout=TIMEOUT)
            assert not at.exception, (key, wert, [str(e) for e in at.exception])


@pytest.mark.parametrize("art", ["select_slider", "selectbox"])
def test_jede_option_jedes_auswahlreglers_laeuft_ohne_ausnahme(art):
    """Je Regler jede Option nacheinander in einer laufenden App (Zwischenspeicher gilt im Prozess und beschleunigt die späteren Läufe)."""
    at = _fresh_app()
    for key, spec in P.SETTING_SPECS.items():
        if spec.options is None or key in SCHWER:
            continue
        widgets = {"select_slider": at.sidebar.select_slider, "selectbox": at.sidebar.selectbox}[art]
        try:
            regler = widgets(key=key)
        except KeyError:
            continue
        if regler.proto.disabled:                           # ε/OPT gilt nur beim Gravitationskern: eigener Test unten
            continue
        for option in spec.options:
            regler.set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e) for e in at.exception])
            regler = widgets(key=key)
        regler.set_value(spec.default).run(timeout=TIMEOUT)


def test_epsilon_regler_beim_gravitationskern_jede_stufe():
    at = _fresh_app()
    assert _regler(at, "eps_rel").proto.disabled                                   # beim Furness-Modus ohne Wirkung und deshalb gesperrt
    _regler(at, "kern").set_value(C.KERN_GRAV).run(timeout=TIMEOUT)
    assert not _regler(at, "eps_rel").proto.disabled
    for stufe in C.EPS_REL_STUFEN:
        _regler(at, "eps_rel").set_value(stufe).run(timeout=TIMEOUT)
        assert not at.exception, (stufe, [str(e) for e in at.exception])


def test_schwere_regler_der_experimente_3_und_4_an_den_grenzen():
    """n = 6 und 20 Instanzen rechnen die Gravitationszeile live (die vorgerechnete Datei gilt nur für n = 12 und 50 Instanzen); die andere Wahrheit bei den Standardwerten ist vorgerechnet."""
    at = _fresh_app()
    _regler(at, "familie").set_value(C.FAMILIE_T2).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    _regler(at, "familie").set_value(C.FAMILIE_T1)
    _regler(at, "n_mess").set_value(C.N_MESS_MIN)
    _regler(at, "inst4").set_value(20).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_beobachtung_unter_dem_lp_minimum_zeigt_die_meldung():
    at = _fresh_app()
    _regler(at, "beobachtung").set_value(C.BEOB_UNTER).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("kein ε kann sie erzeugen" in e.value for e in at.error)


def test_worst_case_alle_maxima_zugleich():
    at = _fresh_app()
    _regler(at, "n_slider").set_value(C.N_MAX)
    _regler(at, "nullanteil").set_value(0.7)
    _regler(at, "delta").set_value(0.1)
    _regler(at, "k_durch").set_value(1000)
    _regler(at, "nullanteil2").set_value(0.7)
    _regler(at, "inst2").set_value(200)
    _regler(at, "form").set_value(C.FORM_MATRIX)
    _regler(at, "theta").set_value(0.01)
    _regler(at, "sigma").set_value(1.0)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_permalink_uebernimmt_werte_und_begrenzt_sie():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "999"
    at.query_params["q"] = C.QUELLE_HAND
    at.query_params["d"] = "0.04"
    at.query_params["tau"] = "unbekannt"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="n_slider").value == C.N_MAX
    assert at.sidebar.selectbox(key="quelle").value == C.QUELLE_HAND
    assert at.sidebar.select_slider(key="delta").value == 0.05
    assert at.sidebar.selectbox(key="tau").value == C.TAU_AUS                     # ungültiger Wert: Standard


def test_kein_planbegriff_und_kein_orakel_im_anzeigetext():
    at = _fresh_app()
    texte = [m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.subheader] + [e.label for e in at.expander]
    voll = "\n".join(texte)
    for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "Orakel", "V1", "V3", "Messlücke", "ZUSCHNITT", "Stück 2", "Stück 5", "trägt", "läuft lokal", "Saatwert"):
        assert verboten not in voll, verboten


def test_anzeigetexte_sagen_nan_klasse_1_und_grossen_eps_nur_mit_ihren_voraussetzungen():
    at = _fresh_app()
    voll = "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    assert "bei beliebigen und bei gleichen Randsummen meist in NaN" in voll and "bei Randsummen aus Wachstumsfaktoren erreicht er oft nur die Obergrenze" in voll
    assert "Ein sehr großes ε*/OPT heißt nur, dass die alte Matrix nahe am Produktmaß der Randsummen liegt" in voll
    assert "in den gemessenen Fällen fällt der Randfehler nur etwa wie 1/k" in voll and "bei diesen dichten Matrizen" in voll and "wächst mit der Strafstärke τ" in voll
    assert "10⁹ Durchläufen" not in voll and "kostet aber mehr Durchläufe" not in voll
