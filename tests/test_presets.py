"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzen und einrasten; jedes Preset lässt sich rechnen."""

import re

import pytest

import mab_constants as C
import mab_evaluation as E
import mab_presets as P

STANDARD = "Zonen 6, Furness"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.SETTING_SPECS), name          # jedes Preset legt ALLE Regler fest
        assert C.PRESET_HELP[name]
    assert list(C.PRESETS) == [STANDARD, "Von Hand 2 × 2", "Summen +5 %", "Nullanteil 0,5", "ε kalibrieren", "Alte Matrix σ = 0,6"]


def test_preset_values_are_inside_the_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert wert in spec.options, (name, key, wert)
            else:
                assert spec.lo <= wert <= spec.hi, (name, key, wert)


def test_no_two_presets_are_identical_and_the_default_is_exactly_one_preset():
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)
    assert [n for n, p in C.PRESETS.items() if p == C.DEFAULTS] == [STANDARD]
    assert E.Settings() == E.settings_aus_zustand(C.DEFAULTS)


def test_setting_specs_cover_every_preset_key_and_every_default():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) == set(C.DEFAULTS)
    urls = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(set(urls)) == len(urls)                             # keine doppelten Adressparameter


def test_preset_hilfetexte_nennen_keine_messzahlen():
    """Die Hilfetexte der Presets nennen keine gemessenen Zahlen (jede Zahl der App steht in der README oder in einem Test); erlaubt sind nur Reglerwerte und Handwerte."""
    erlaubt = ("0,9 und 1,1", "2 × 2", "[[1, 2], [3, 4]]", "(5, 6) und (5, 6)", "Randsummen (5, 6)", "5 %", "0,5", "Acht Zonen", "Sechs Zonen", "Experiment 3", "Experiment 4", "θ = 0,1", "σ = 0,6")
    for name, text in C.PRESET_HELP.items():
        rest = text
        for token in erlaubt:
            rest = rest.replace(token, "")
        assert not re.search(r"\d", rest), (name, rest)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laesst_sich_rechnen_und_zeigt_die_erwartete_einstellung(name):
    s = E.settings_aus_zustand(C.PRESETS[name])
    erg = E.ansicht_ergebnis(s)
    assert erg.K.shape == (erg.an.n, erg.an.n) and erg.ab.plan.shape == erg.K.shape
    if name == "Von Hand 2 × 2":
        assert s.quelle == C.QUELLE_HAND and erg.an.n == 2
    if name == "Summen +5 %":
        assert s.delta == 0.05 and erg.klasse is None
    if name == "Nullanteil 0,5":
        assert s.nullanteil == 0.5 and (erg.an.T0 == 0).any()
    if name == "ε kalibrieren":
        assert s.kern == C.KERN_GRAV


def test_einrasten_und_begrenzen_des_permalinks():
    sp = P.SETTING_SPECS
    assert P._eingerastet(sp["n_slider"], 500) == C.N_MAX and P._eingerastet(sp["n_slider"], -3) == C.N_MIN
    assert P._eingerastet(sp["n_mess"], 500) == C.N_MESS_MAX and P._eingerastet(sp["seed_input"], -1) == 0
    assert P._eingerastet(sp["eps_rel"], 0.45) == 0.5 and P._eingerastet(sp["eps_rel"], 1e9) == 10.0
    assert P._eingerastet(sp["k_durch"], 7) in (6, 8) and P._eingerastet(sp["k_durch"], 99999) == 1000
    assert P._eingerastet(sp["delta"], 0.04) == 0.05 and P._eingerastet(sp["theta"], 0.3) in (0.2, 0.4) and P._eingerastet(sp["sigma"], 0.5) in (0.3, 0.6)
    assert P._eingerastet(sp["quelle"], C.QUELLE_HAND) == C.QUELLE_HAND and P._eingerastet(sp["quelle"], "unbekannt") is None
    assert P._eingerastet(sp["tau"], "100") == "100" and P._eingerastet(sp["tau"], "7") is None
    assert P._eingerastet(sp["inst2"], 120) in (100, 200) and P._eingerastet(sp["inst4"], 40) == 50


def test_permalink_werte_der_adresszeile_werden_uebernommen(monkeypatch):
    from types import SimpleNamespace

    zustand = {}
    qp = {"n": "9", "q": C.QUELLE_HAND, "p": "0.5", "tau": "10", "sig": "kein_wert", "fam": "unbekannt"}
    monkeypatch.setattr(P, "st", SimpleNamespace(session_state=zustand, query_params=qp))
    P.load_permalink_settings()
    assert zustand["n_slider"] == 9 and zustand["quelle"] == C.QUELLE_HAND and zustand["nullanteil"] == 0.5 and zustand["tau"] == "10"
    assert "sigma" not in zustand and "familie" not in zustand       # ungültige Werte werden ignoriert
