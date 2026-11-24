"""Rechnet die Reihen für die Standardeinstellungen vor und schreibt `tools/vorgerechnet.json` (Aufruf aus dem Demo-Ordner: python tools/vorrechnen.py).

Die Datei wird nur gelesen, nie von der App geschrieben; `tests/test_claims.py` rechnet die Einträge neu und vergleicht sie als Band."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mab_evaluation as E  # noqa: E402
from mab_vorgerechnet import DATEI, schluessel  # noqa: E402


def main():
    out = {}
    for name, params, rechne in E.vorzurechnen():
        t = time.time()
        out[schluessel(name, **params)] = E.jsonbar(rechne())
        print(f"{name} {params}: {time.time() - t:.1f} s")
    DATEI.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"geschrieben: {DATEI} ({len(out)} Einträge)")


if __name__ == "__main__":
    main()
