"""Vorgerechnete Reihen: `tools/vorgerechnet.json` enthält die Ergebnisse der schweren Reihen für die Standardeinstellungen (erzeugt von `tools/vorrechnen.py` mit dem Code der Demo).

Wozu: Die Gravitationszeile von Experiment 4 (Kalibrierung auf 50 Instanzen mit sehr ungleichen Randsummen, bis zu 400 000 innere Durchläufe) braucht auf dem Entwicklungsrechner
10 bis 35 Sekunden, je nach Last (bei freiem Rechner nahe 10 Sekunden). Mit den Standardeinstellungen erscheint die Seite sofort; bei jeder anderen Einstellung rechnet die App live (Spinner, Zwischenspeicher).
`tests/test_claims.py` rechnet die Einträge neu und vergleicht sie als Band."""

import json
from functools import lru_cache
from pathlib import Path

DATEI = Path(__file__).resolve().parent / "tools" / "vorgerechnet.json"


def schluessel(name: str, **params) -> str:
    """Eindeutiger Schlüssel aus Name und Parametern (Zahlen als int bzw. float, Texte als Texte)."""
    return name + "|" + ",".join(f"{k}={params[k]!r}" for k in sorted(params))


@lru_cache(maxsize=1)
def _lade() -> dict:
    try:
        return json.loads(DATEI.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def hole(name: str, **params):
    """Der vorgerechnete Eintrag oder None."""
    return _lade().get(schluessel(name, **params))
