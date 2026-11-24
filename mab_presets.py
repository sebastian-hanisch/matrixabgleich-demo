"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Jeder Regler hat einen Eintrag in SETTING_SPECS (Name im Session State = Schlüssel in `mab_constants.DEFAULTS`). Reglerwerte aus der Adresszeile werden auf die
Grenzen begrenzt (Zahlen) beziehungsweise auf den nächsten erlaubten Wert eingerastet (Stufen und Auswahllisten); ungültige Werte werden ignoriert."""

import random
from dataclasses import dataclass

import streamlit as st

import mab_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # erlaubte Werte (Auswahllisten und Stufen)


def _opt(param, caster, key, options):
    return SettingSpec(param, caster, C.DEFAULTS[key], options=options)


SETTING_SPECS = {
    "quelle": _opt("q", str, "quelle", C.QUELLEN),
    "n_slider": SettingSpec("n", int, C.DEFAULTS["n_slider"], C.N_MIN, C.N_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULTS["seed_input"], 0, C.SEED_MAX),
    "nullanteil": _opt("p", float, "nullanteil", C.NULLANTEIL_STUFEN),
    "rs_ansicht": _opt("rs", str, "rs_ansicht", (C.RS_MILD, C.RS_BELIEBIG)),
    "delta": _opt("d", float, "delta", C.DELTA_STUFEN),
    "kern": _opt("kern", str, "kern", C.KERNE),
    "eps_rel": _opt("eps", float, "eps_rel", C.EPS_REL_STUFEN),
    "k_durch": _opt("k", int, "k_durch", C.K_STUFEN),
    "tau": _opt("tau", str, "tau", C.TAU_STUFEN),
    "nullanteil2": _opt("p2", float, "nullanteil2", C.NULLANTEIL_STUFEN),
    "randsummen": _opt("rs2", str, "randsummen", C.RANDSUMMEN),
    "inst2": _opt("i2", int, "inst2", C.INSTANZEN2_STUFEN),
    "form": _opt("form", str, "form", C.FORMEN),
    "n_mess": SettingSpec("nm", int, C.DEFAULTS["n_mess"], C.N_MESS_MIN, C.N_MESS_MAX),
    "beobachtung": _opt("beob", str, "beobachtung", C.BEOBACHTUNGEN),
    "theta": _opt("th", float, "theta", C.THETA_STUFEN),
    "startregel": _opt("start", str, "startregel", C.STARTREGELN),
    "warmstart": _opt("warm", str, "warmstart", C.WARMSTARTS),
    "sigma": _opt("sig", float, "sigma", C.SIGMA_STUFEN),
    "familie": _opt("fam", str, "familie", C.FAMILIEN),
    "inst4": _opt("i4", int, "inst4", C.INSTANZEN4_STUFEN),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _eingerastet(spec: SettingSpec, wert):
    """Wert auf Grenzen begrenzt beziehungsweise auf die nächste erlaubte Option eingerastet; None, wenn nichts passt."""
    if spec.options is not None:
        if spec.caster is str:
            return wert if wert in spec.options else None
        return min(spec.options, key=lambda o: abs(o - wert))
    if spec.lo is not None:
        wert = max(spec.lo, wert)
    if spec.hi is not None:
        wert = min(spec.hi, wert)
    return wert


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen (begrenzt beziehungsweise eingerastet)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                wert = _eingerastet(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if wert is not None:
                st.session_state[state_key] = wert
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
