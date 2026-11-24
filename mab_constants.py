"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und Hilfetexte.

Rechenaufwand am Regler-Maximum (numpy, ein Kern; Zeiten auf dem Entwicklungsrechner, auf der Cloud wahrscheinlich 2- bis 3-mal länger):
- Ein Durchlauf des Abgleichs kostet zwei Matrix-Vektor-Produkte mit der dichten n × m-Matrix K, also O(n · m); bei n = 12 sind bis zu 5 000 Durchläufe die Obergrenze.
- Die Reihen laufen als Stapel über alle Instanzen zugleich (ein Durchlauf je Instanz ist ein einziger Aufruf von einsum): Experiment 2 mit 200 Instanzen und
  8 Stufen etwa 10 Rechnungen mit bis zu 5 000 Durchläufen; Experiment 3 (Hyman-Schleife) bis zu 60 Schritte mit je bis zu 50 000 inneren Durchläufen, in der Regel
  einige Tausend insgesamt.
- Der OPT (kleinster Wert des Transportproblems) kommt vom eigenen Transportsimplex, O(n³) je Instanz, bei n = 12 etwa 1 ms."""

import math

# --- Auswahllisten -------------------------------------------------------------------------------------------------------------------
QUELLE_HAND = "Von Hand 2 × 2"
QUELLE_ZONEN = "Zonen im Einheitsquadrat"
QUELLEN = (QUELLE_HAND, QUELLE_ZONEN)

KERN_ALT = "Alte Matrix (Furness)"
KERN_GRAV = "Gravitation exp(−C/ε)"
KERNE = (KERN_ALT, KERN_GRAV)

RS_BELIEBIG = "beliebig (Dirichlet(1) je Seite)"
RS_MILD = "Wachstum mild, U(0,9; 1,1)"
RS_V3 = "Wachstum breit, U(0,7; 1,6)"
RS_STARK = "Wachstum stark, U(0,4; 2,5)"
RS_GLEICH = "gleich, je 1/n"
RANDSUMMEN = (RS_BELIEBIG, RS_MILD, RS_V3, RS_STARK, RS_GLEICH)
RS_BREITE = {RS_MILD: (0.9, 1.1), RS_V3: (0.7, 1.6), RS_STARK: (0.4, 2.5)}

FORM_UV = "u-v-Form"
FORM_MATRIX = "Matrix-Form"
FORMEN = (FORM_UV, FORM_MATRIX)

BEOB_THETA = "Anteil θ zwischen LP-Minimum und Produktmaß"
BEOB_UNTER = "Randfall: 10 % unter dem LP-Minimum"
BEOBACHTUNGEN = (BEOB_THETA, BEOB_UNTER)

START_PROP = "erster Schritt β · c/c_obs"
START_FAKTOR = "erster Schritt 1,5 · β"
STARTREGELN = (START_PROP, START_FAKTOR)

WARM_KALT = "kalt (jeder Abgleich beginnt bei v = 1)"
WARM_WARM = "warm (v vom Schritt davor)"
WARMSTARTS = (WARM_KALT, WARM_WARM)

FAMILIE_T1 = "Gravitation mit Rauschen"
FAMILIE_T2 = "andere Hemmfunktion (C + 0,05)⁻²"
FAMILIEN = (FAMILIE_T1, FAMILIE_T2)

TAU_AUS = "aus"
TAU_STUFEN = (TAU_AUS, "1", "10", "100", "1000")

DELTA_STUFEN = (0.0, 0.001, 0.01, 0.05, 0.1)
NULLANTEIL_STUFEN = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
EPS_REL_STUFEN = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
THETA_STUFEN = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9)
SIGMA_STUFEN = (0.0, 0.1, 0.3, 0.6, 1.0)
INSTANZEN2_STUFEN = (50, 100, 200)
INSTANZEN4_STUFEN = (20, 50)
K_STUFEN = (0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 300, 1000)

N_MIN, N_MAX, DEFAULT_N = 6, 12, 6          # Primäransicht (Zonen)
N_MESS_MIN, N_MESS_MAX, DEFAULT_N_MESS = 6, 12, 12   # Experimente 3 und 4
SEED_MAX, DEFAULT_SEED = 999999, 35

# --- feste Annahmen -------------------------------------------------------------------------------------------------------------------
TOL = 1e-9                         # Abbruch: Zeilenfehler (L1, Gesamtmasse 1) unter 1e-9, float64
TOL_RAND = 1e-12                   # Experiment 4: schärferer Abbruch, damit der Randsummenfehler als Zahl lesbar ist
MAX_DURCHLAEUFE = 5000             # Obergrenze der Durchläufe; bei Erreichen steht eine Meldung
MAX_INNEN_HYMAN = 50000            # Obergrenze der Durchläufe je innerem Abgleich der Hyman-Schleife
MAX_INNEN_EXP4 = 400000            # Experiment 4, Gravitationszeile: Obergrenze der inneren Durchläufe (die Wahrheit hat sehr ungleiche Randsummen)
MAX_SCHRITTE_HYMAN = 60            # Obergrenze der Schritte der Hyman-Schleife (Randfall: 40)
MAX_SCHRITTE_RANDFALL = 40
TOL_HYMAN_REL = 1e-6               # relative Abweichung der mittleren Reiseweite zur Beobachtung
TOL_HYMAN_INNEN = 1e-10            # Abbruch des inneren Abgleichs
GRAVITATION_VOR = 3.0              # Basismatrix exp(−C/3)
RAUSCH_LO, RAUSCH_HI = 0.5, 1.5    # Rauschfaktor je Zelle der Basismatrix, U(0,5; 1,5)
TRAEGER_LO, TRAEGER_HI = 0.5, 2.0  # Einträge der Matrizen mit Nullen
GESAMT_FAHRTEN = 1000.0            # Anzeige: die alte Matrix hat etwa so viele Fahrten (nur Anzeigegröße, gerechnet wird mit Gesamtmasse 1)
N_ZEILE_EXP1 = 8                   # Experiment 1: Zeilenfehler bei verschiedenen Summen, n = 8
N_STRAFE = 10                      # Experiment 1: Strafverfahren, n = 10
N_EXP2 = 8                         # Experiment 2: n = 8
ANZAHL_EXP1 = 20                   # Instanzen je Stufe in Experiment 1
ANZAHL_EXP3 = 20                   # Instanzen in Experiment 3
SERIEN_SEED = 2026                 # Zufalls-Seed der Reihen: nie der Regler „Zufalls-Seed“
STRAFE_TAUS = (1.0, 10.0, 100.0, 1000.0)
STRAFE_EPS = 1.0
STRAFE_TOL = 1e-10                 # Abbruch des Strafverfahrens: größte Änderung der Logarithmen der Skalierungen
STRAFE_MAX = 60000
STRAFE_SUMMEN_PLUS = 0.05          # „Summen +5 %“ in Experiment 1
POISSON_FAKTOR = 90.0              # λ_ij = 90 · n · exp(−C/3) · a_i · a_j (Summe a = 1): die Zählung der alten Matrix
KREUZ_GRENZE = 0.25                # Zellen mit mehr als 25 % Abweichung (Experiment 4, nur Hinweis)
HEMM_VERSATZ = 0.05                # Hemmfunktion (C + 0,05)⁻²
RAUSCH_T1 = 0.5                    # lognormales Paar-Rauschen der Wahrheit T1

ZWECK_ANSICHT, ZWECK_EXP1, ZWECK_STRAFE, ZWECK_EXP2, ZWECK_EXP3, ZWECK_EXP4, ZWECK_RS = 1, 2, 3, 4, 5, 6, 7   # getrennte Zufallsströme

COLOR_LINE = "#4c78a8"
COLOR_ZIEL = "#e08a1e"
COLOR_OK = "#2F6B65"
COLOR_BAD = "#c0504d"
COLOR_BAND = "rgba(76,120,168,0.18)"

EPS_H4_A = 2.0 / math.log(361.0 / 11.0)   # ε* des Handbeispiels 4 bei Kosten 0,24 (Anteil θ = 1/8)

DEFAULTS = {
    "quelle": QUELLE_ZONEN,
    "n_slider": DEFAULT_N,
    "seed_input": DEFAULT_SEED,
    "nullanteil": 0.0,
    "delta": 0.0,
    "rs_ansicht": RS_MILD,
    "kern": KERN_ALT,
    "eps_rel": 1.0,
    "k_durch": 3,
    "tau": TAU_AUS,
    "nullanteil2": 0.5,
    "randsummen": RS_BELIEBIG,
    "inst2": 100,
    "form": FORM_UV,
    "n_mess": DEFAULT_N_MESS,
    "beobachtung": BEOB_THETA,
    "theta": 0.1,
    "startregel": START_PROP,
    "warmstart": WARM_KALT,
    "sigma": 0.3,
    "familie": FAMILIE_T1,
    "inst4": 50,
}

# Presets: jede Zeile legt ALLE Regler fest (Schlüssel wie in DEFAULTS); `aktives_preset` vergleicht jeden Regler.
SEED_NULLEN_UNZULAESSIG = 35   # wird beim Bau gesetzt (Instanz mit Nullanteil 0,5, n = 8, die keine Lösung hat)
PRESETS = {
    "Zonen 6, Furness": dict(DEFAULTS),
    "Von Hand 2 × 2": dict(DEFAULTS, quelle=QUELLE_HAND, k_durch=1),
    "Summen +5 %": dict(DEFAULTS, delta=0.05),
    "Nullanteil 0,5": dict(DEFAULTS, n_slider=8, nullanteil=0.5, rs_ansicht=RS_BELIEBIG, seed_input=SEED_NULLEN_UNZULAESSIG),
    "ε kalibrieren": dict(DEFAULTS, kern=KERN_GRAV, eps_rel=1.0, theta=0.1),
    "Alte Matrix σ = 0,6": dict(DEFAULTS, sigma=0.6),
}
PRESET_HELP = {
    "Zonen 6, Furness": "Sechs Zonen, alte Matrix mit Rauschen, neue Randsummen aus Wachstumsfaktoren zwischen 0,9 und 1,1: der Abgleich konvergiert in wenigen Durchläufen.",
    "Von Hand 2 × 2": "Das Beispiel zum Nachrechnen: alte Matrix [[1, 2], [3, 4]], Randsummen (5, 6) und (5, 6). Nach einem Durchlauf stehen die Zahlen als Brüche im Expander „Mathematische Formulierung“.",
    "Summen +5 %": "Die Spaltensummen sind 5 % größer als die Zeilensummen: der Abgleich läuft bis zur Obergrenze und der Zeilenfehler bleibt bei der Differenz der Summen.",
    "Nullanteil 0,5": "Acht Zonen, die Hälfte der Zellen der alten Matrix ist null: bei diesen Randsummen gibt es keine Matrix mit dieser Nullstruktur, der Abgleich konvergiert nicht.",
    "ε kalibrieren": "Die Matrix ist ein Gravitationskern exp(−C/ε); Experiment 3 rechnet ε aus einer beobachteten mittleren Reiseweite zurück (θ = 0,1).",
    "Alte Matrix σ = 0,6": "Experiment 4 mit stark gestörter alter Matrix (σ = 0,6): die Randsummen stimmen nach dem Abgleich, ein Teil der Menge liegt trotzdem in der falschen Zelle.",
}
