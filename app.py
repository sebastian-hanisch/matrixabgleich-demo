"""Matrix abgleichen – dieselben vier Zeilen Code in vier Fächern - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Eine alte Matrix wird auf neue Randsummen abgeglichen (Furness, RAS, Raking, Sinkhorn-Knopp: eine Rechnung). Die App zählt, was darum herum geschieht: Lösbarkeit,
Matrix-Vektor-Produkte, ε aus einer Beobachtung, Randsummen gegen Zellen.

Lauffähig mit: streamlit run app.py
"""

import inspect

import streamlit as st

import mab_constants as C
import mab_evaluation as E
import mab_scaling as S
import mab_verdikte as V
import mab_visualization as VIS
from mab_evaluation import de, de_g, ganz, prozent
from mab_layout import dunkles_design, kopf_css
from mab_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Matrix abgleichen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _ansicht(s):
    return E.ansicht_ergebnis(s)


@st.cache_data(show_spinner=False)
def _summen():
    return E.jsonbar(E.reihe_summen())


@st.cache_data(show_spinner=False)
def _strafe():
    return E.jsonbar(E.reihe_strafe())


@st.cache_data(show_spinner=False)
def _nullen(art, anzahl, form):
    return E.jsonbar(E.reihe_nullen(art, anzahl, form))


@st.cache_data(show_spinner=False)
def _kalibrierung(theta, beobachtung, startregel, warm, n):
    return E.jsonbar(E.reihe_kalibrierung(theta, beobachtung, startregel, warm, n))


@st.cache_data(show_spinner=False)
def _wahrheit(n, startregel):
    return E.jsonbar(E.reihe_wahrheit(n=n, startregel=startregel))


@st.cache_data(show_spinner=False)
def _randsummen(familie, n, anzahl):
    return E.jsonbar(E.reihe_randsummen(familie, n, anzahl))


@st.cache_data(show_spinner=False)
def _gravitation(familie, n, anzahl):
    return E.gravitation_v(familie, n, anzahl)


def zeige(verdikt):
    art, text = verdikt
    getattr(st, art)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mab_layout.py).
st.html(kopf_css(dunkles_design()))
st.html('<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>')
with st.container(key="kopf"):
    st.title("🧮 Matrix abgleichen – dieselben vier Zeilen Code in vier Fächern")
st.markdown(
    """
Eine **alte Matrix** (zum Beispiel Fahrten zwischen Zonen aus einer früheren Zählung) soll auf neue **Randsummen** (marginals) angepasst werden: neue Zeilensummen (Herkunft) und neue Spaltensummen (Ziel).
Die Rechnung heißt in der Verkehrsplanung **Furness-Verfahren**, in der Volkswirtschaft **RAS**, in der Umfragestatistik **Raking** (iterative proportionale Anpassung, IPF) und in der Mathematik
**Sinkhorn-Knopp**: abwechselnd Zeilen und Spalten anpassen. Dass es dieselbe Rechnung ist, ist ein Name, keine Entdeckung. Die Demo zählt, was darum herum geschieht:
**Hat die Aufgabe eine Lösung? Wie viele Matrix-Vektor-Produkte braucht der Abgleich? Woher kommt ε im Gravitationsmodell? Was sagen die Randsummen über die Zellen nicht?**

Weiter unten: vier Experimente (**🔬**), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Abgleich", expanded=True):
    st.markdown(
        """
1. **Gegeben:** eine nichtnegative alte Matrix $T^0$ ($n \\times n$ Zonen) und neue Zeilensummen $a$ und Spaltensummen $b$.
2. **Ein Durchlauf (sweep):** jede Zeile mit $a_i / (\\text{Zeilensumme}_i)$ multiplizieren, danach jede Spalte mit $b_j / (\\text{Spaltensumme}_j)$. Das sind zwei Matrix-Vektor-Produkte.
3. **Wiederholen**, bis der Zeilenfehler (Summe der Abweichungen der Zeilensummen, Gesamtmasse 1) unter $10^{-9}$ liegt. Das Ergebnis hat die Form $P_{ij} = u_i\\, T^0_{ij}\\, v_j$.
4. **Warum das so heißt, was es ist:** unter allen Matrizen mit den neuen Randsummen ist $P$ diejenige, die der alten Matrix im Sinn der relativen Entropie (Kullback-Leibler, KL) am nächsten liegt.
   Der **Gravitationskern** $K = \\exp(-C/\\varepsilon)$ statt der alten Matrix gibt das doppelt beschränkte Gravitationsmodell der Verkehrsplanung: derselbe Abgleich mit einem anderen Kern.
5. **Wann „gelaufen“ nicht „gelöst“ heißt:** wenn die Summen verschieden sind, wenn Nullen in der alten Matrix die Randsummen unmöglich machen, und wenn die Randsummen stimmen, aber die Zellen nicht (Experimente 1, 2 und 4).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()


def ansicht_regler():
    st.header("⚙️ Primäransicht")
    quelle = st.selectbox("Aufgabe", C.QUELLEN, key="quelle", help="Das Handbeispiel 2 × 2 zum Nachrechnen oder Zonen im Einheitsquadrat mit einer zufälligen alten Matrix.")
    zonen = quelle == C.QUELLE_ZONEN
    st.slider("Zonen n", *bounds("n_slider"), key="n_slider", disabled=not zonen, help="Anzahl der Zonen (Herkunft = Ziel).")
    st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1, disabled=not zonen, help="Wirkt nur auf die Primäransicht; die Experimente rechnen mit festem Zufalls-Seed.")
    st.select_slider("Nullanteil der alten Matrix", C.NULLANTEIL_STUFEN, key="nullanteil", disabled=not zonen, format_func=lambda x: de(x, 1),
                     help="Anteil der Zellen, die null sind (jede Zeile und Spalte behält mindestens einen Eintrag). Null: dichte Matrix mit Rauschen.")
    st.selectbox("Neue Randsummen", (C.RS_MILD, C.RS_BELIEBIG), key="rs_ansicht", disabled=not zonen,
                 help="Wachstumsfaktoren: Zeilen- und Spaltensummen der alten Matrix mal je einen Faktor zwischen 0,9 und 1,1. Beliebig: je Seite zufällige Anteile (Dirichlet(1)).")
    st.select_slider("Summen verschieden: (B − A)/A", C.DELTA_STUFEN, key="delta", format_func=lambda x: de(x, 3),
                     help="Die Spaltensumme B ist um diesen Anteil größer als die Zeilensumme A. Ab 0,001 hat die Aufgabe keine Lösung (Experiment 1).")
    st.selectbox("Kern", C.KERNE, key="kern", disabled=not zonen, help="Alte Matrix: Furness-Modus ohne ε. Gravitation: K = exp(−C/ε) mit den Randsummen der Zonen.")
    gravitation = zonen and st.session_state["kern"] == C.KERN_GRAV
    st.select_slider("ε/OPT (nur Gravitation)", C.EPS_REL_STUFEN, key="eps_rel", disabled=not gravitation, format_func=lambda x: de_g(x, 2),
                     help="ε im Verhältnis zum LP-Minimum OPT derselben Randsummen bei Gesamtmasse 1 und größtem Abstand 1.")
    st.select_slider("Gezeigter Durchlauf k", C.K_STUFEN, key="k_durch", help="Die Matrix nach k Durchläufen; 0 ist die Ausgangsmatrix.")
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, disabled=not zonen)


def experiment_regler():
    st.header("🔬 Experiment 1: Summen")
    st.selectbox("KL-Strafe τ statt harter Randsummen", C.TAU_STUFEN, key="tau", help="Aus: harte Randsummen (Furness). Sonst das Strafverfahren mit ε = 1 und diesem τ.")
    st.header("🔬 Experiment 2: Nullen")
    st.select_slider("Nullanteil (Experiment 2)", C.NULLANTEIL_STUFEN, key="nullanteil2", format_func=lambda x: de(x, 1), help="Anteil der Nullzellen der alten Matrix, n = 8.")
    st.selectbox("Randsummen (Experiment 2)", C.RANDSUMMEN, key="randsummen", help="Beliebig, aus Wachstumsfaktoren der alten Matrix oder gleich (1/n).")
    st.select_slider("Instanzen je Stufe (Experiment 2)", C.INSTANZEN2_STUFEN, key="inst2", help="Zahl der Zufallsinstanzen je Nullanteil.")
    st.selectbox("Rechenform", C.FORMEN, key="form", help="u-v-Form (Teilung, zählt die Produkte) oder Matrix-Form (die Matrix wird skaliert). Bei unlösbaren Aufgaben enden sie verschieden.")
    st.header("🔬 Experiment 3: ε zurückrechnen")
    st.slider("Zonen n (Experimente 3 und 4)", *bounds("n_mess"), key="n_mess", help="Anzahl der Zonen der Reihen von Experiment 3 und 4.")
    st.selectbox("Beobachtung", C.BEOBACHTUNGEN, key="beobachtung", help="Eine erreichbare mittlere Reiseweite oder der Randfall unter dem LP-Minimum.")
    st.select_slider("Anteil θ", C.THETA_STUFEN, key="theta", format_func=lambda x: de(x, 2),
                     help="Wo zwischen dem LP-Minimum (θ = 0) und dem Produktmaß (θ = 1) die beobachtete mittlere Reiseweite liegt.")
    st.selectbox("Startregel", C.STARTREGELN, key="startregel", help="Wie der erste Schritt der Schleife aus dem Startwert β = 1/c_obs gewählt wird (eigene Wahl).")
    st.selectbox("Start des inneren Abgleichs", C.WARMSTARTS, key="warmstart", help="Kalt: jeder Abgleich beginnt bei v = 1. Warm: er beginnt mit dem v des Schritts davor.")
    st.header("🔬 Experiment 4: Zellen")
    st.select_slider("Störung σ der alten Matrix", C.SIGMA_STUFEN, key="sigma", format_func=lambda x: de(x, 1), help="Die alte Matrix ist die Wahrheit mal exp(σ·Z) mit standardnormalem Z je Zelle.")
    st.selectbox("Wahrheit", C.FAMILIEN, key="familie", help="Gravitation mit lognormalem Rauschen oder eine andere Hemmfunktion ohne Rauschen.")
    st.select_slider("Instanzen (Experiment 4)", C.INSTANZEN4_STUFEN, key="inst4", help="Zahl der Zufallsinstanzen; n steht beim Regler von Experiment 3.")


with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    ansicht_regler()
    experiment_regler()

s = E.settings_aus_zustand(st.session_state)
sync_query_params({key: st.session_state[key] for key in (
    "quelle", "n_slider", "seed_input", "nullanteil", "rs_ansicht", "delta", "kern", "eps_rel", "k_durch", "tau", "nullanteil2", "randsummen", "inst2", "form",
    "n_mess", "beobachtung", "theta", "startregel", "warmstart", "sigma", "familie", "inst4")})

# ------------------------------------------------------------------------------------------------------------------------------------
# Primäransicht
# ------------------------------------------------------------------------------------------------------------------------------------
erg = _ansicht(s)
ab = erg.ab
A = erg.gesamt
st.header("Ergebnis: der Abgleich Durchlauf für Durchlauf")
kopfzeile = st.columns(4)
ok = ab.status == S.STATUS_OK
kopfzeile[0].metric("Durchläufe", ganz(ab.durchlaeufe) if ok else "nicht konvergiert")
kopfzeile[1].metric("Matrix-Vektor-Produkte", ganz(ab.produkte) if ok else "–")
kopfzeile[2].metric("Lösung vorhanden?", "nein" if (erg.klasse == 0 or erg.klasse is None) else "ja")
kopfzeile[3].metric("ε*/OPT aus dem Ergebnis", de_g(erg.eps_zurueck, 3) if erg.eps_zurueck is not None else "–",
                    help="Aus der mittleren Reiseweite des Ergebnisses zurückgerechnet (Experiment 3); nur bei Zonen und konvergiertem Abgleich.")
zeige(V.verdikt_ansicht(erg))

links, rechts = st.columns([3, 2])
with links:
    k = s.k_durch
    P = E.plan_fahrten(erg, k)
    titel = "Ausgangsmatrix (im Maßstab der Zeilenziele)" if k == 0 else f"Nach {k} Durchläufen"
    st.plotly_chart(VIS.build_matrix(P, erg.a * A, erg.b * A, titel=titel + " – Fahrten: „ist von Ziel“ je Zeile und Spalte"), key="ansicht_matrix", width="stretch", config={"displayModeBar": False})
    if ok and k > ab.durchlaeufe:
        st.caption(f"Nach {ab.durchlaeufe} Durchläufen ist der Abgleich fertig; weitere Durchläufe ändern die Matrix nicht mehr sichtbar.")
with rechts:
    st.markdown(
        """
**Vier Namen, eine Rechnung** (die Namen stehen bei Idel 2016 und Fahli/Galichon 2026, gelesen; welches Fach welchen Namen bevorzugt, ist Sprachgebrauch und hier nicht belegt)

| Fach | Name |
|---|---|
| Verkehrsplanung | Furness-Verfahren |
| Volkswirtschaft | RAS, biproportionales Verfahren |
| Umfragestatistik | Raking, iterative proportionale Anpassung (IPF) |
| Mathematik | Sinkhorn-Knopp |
"""
    )
    try:
        quelltext = inspect.getsource(S.vier_zeilen)
    except OSError:
        quelltext = "(Quelltext nicht verfügbar)"
    st.code(quelltext, language="python")
    st.caption("Diese Funktion rechnet die Matrix links; der gezeigte Text ist der ausgeführte (`mab_scaling.vier_zeilen`). Der Zähler und der Fehlerverlauf kommen aus derselben Rechnung mit Buchführung.")
    if len(ab.fehler_verlauf):
        delta = erg.b.sum() - 1.0
        st.plotly_chart(VIS.build_fehlerkurve(ab.fehler_verlauf * A, k, schranke=delta * A if delta > 1e-12 else None), key="ansicht_fehler", width="stretch", config={"displayModeBar": False})

st.subheader("Vier Kernfragen")
n_z = erg.an.n
if s.quelle == C.QUELLE_ZONEN:
    if s.kern == C.KERN_GRAV:
        eps_satz = (f"Hier vorgegeben: ε/OPT = {de_g(s.eps_rel, 2)}. In der Verkehrsplanung wird β = 1/ε aus der beobachteten mittleren Reiseweite zurückgerechnet (Experiment 3); "
                    f"die Rückrechnung aus dem Ergebnis oben liefert ε*/OPT = {de_g(erg.eps_zurueck, 3) if erg.eps_zurueck else '–'}"
                    + (f" in {erg.schritte_zurueck} Schritten." if erg.schritte_zurueck else "."))
    else:
        eps_satz = ("Der Furness-Modus hat kein ε. Jede positive alte Matrix ist aber ein Gibbs-Kern ($T^0 = \\exp(-C/\\varepsilon)$ mit $C = -\\varepsilon \\ln T^0$, einer von ε abhängigen Matrix und keiner Abstandsmatrix); "
                    f"aus der mittleren Reiseweite des Ergebnisses rechnet Experiment 3 ein ε für den Gravitationskern zurück: ε*/OPT = {de_g(erg.eps_zurueck, 3) if erg.eps_zurueck else '–'}. "
                    "Ein sehr großes ε*/OPT heißt nur, dass die alte Matrix nahe am Produktmaß der Randsummen liegt (θ nahe 1); es sagt nichts über kalibrierte β der Verkehrsplanung.")
else:
    eps_satz = "Das Handbeispiel hat keine Kosten, also kein ε; die Brücke (jede positive Matrix ist ein Gibbs-Kern) steht im Expander „Mathematische Formulierung“."
st.markdown(
    f"""
1. **Hat die Aufgabe eine Lösung?** {V.antwort_loesung(erg)}
2. **Wie viele Matrix-Vektor-Produkte?** {('Der Abgleich braucht ' + ganz(ab.produkte) + ' (' + ganz(ab.durchlaeufe) + ' Durchläufe zu je zwei Produkten).') if ok else 'Er konvergiert nicht; gezählt wird bis zur Obergrenze von ' + ganz(C.MAX_DURCHLAEUFE) + ' Durchläufen.'}
3. **Woher kommt ε?** {eps_satz}
4. **Was sagen die Randsummen nicht?** Sie legen $2n - 1 = {2 * n_z - 1}$ von $n^2 = {n_z * n_z}$ Zahlen fest, $(n-1)^2 = {(n_z - 1) ** 2}$ bleiben frei (Experiment 4). Der Abgleich bewahrt nur die Kreuzverhältnisse der alten Matrix.
"""
)

# ------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1
# ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 1: Die Summen passen nicht")
st.markdown(
    "Die Spaltensumme $B$ ist um den Anteil $(B - A)/A$ größer als die Zeilensumme $A$. Das Verfahren läuft, aber keine Matrix hat beide Randsummen: es meldet nichts, "
    "der Zeilenfehler bleibt bei $|B - A|$ (bei diesen dichten Matrizen; allgemein ist er mindestens $|B - A|$). Mit einer KL-Strafe statt harter Randsummen läuft in den gemessenen Instanzen jeder Lauf zu Ende; "
    "die Zahl der Durchläufe wächst mit der Strafstärke τ (Tabelle unten; das Abbruchkriterium ist ein anderes als beim harten Furness)."
)
with st.spinner("Rechne Experiment 1 …"):
    res1 = _summen()
    sres = _strafe()
zeile1 = min(res1["zeilen"], key=lambda z: abs(z["delta"] - s.delta))
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(VIS.build_summen_verlauf(zeile1), key="exp1_verlauf", width="stretch", config={"displayModeBar": False})
    st.caption(f"Zeilenfehler (relativ zur Summe der Zeilenziele) je Durchlauf, {res1['anzahl']} Instanzen, n = {res1['n']}, (B − A)/A = {de(zeile1['delta'], 3)}; unten die Obergrenze {ganz(res1['max_durchlaeufe'])}.")
with c2:
    st.plotly_chart(VIS.build_strafe(sres), key="exp1_strafe", width="stretch", config={"displayModeBar": False})
    st.caption(f"Strafverfahren (ε = 1, τ = 1, 10, 100, 1 000; n = {sres['n']}, {sres['anzahl']} Instanzen): Median, Balken von kleinstem bis größtem Wert.")
st.dataframe([{"(B − A)/A": de(z["delta"], 3), "Läufe konvergiert": f"{z['konvergiert']} von {z['anzahl']}", "Zeilenfehler (Median)": de_g(z["fehler"][0], 3),
               "Durchläufe (Median)": ganz(z["durchlaeufe"][0]) if z["durchlaeufe"] else "–"} for z in res1["zeilen"]], hide_index=True, width="stretch")
st.dataframe([{"τ": de_g(z["tau"], 4), "Abweichung vom Furness (Median)": de_g(z["abweichung"][0], 3), "Durchläufe (Median [Min; Max])": E.band_ganz(z["durchlaeufe"]),
               "Läufe konvergiert": f"{z['konvergiert']} von {sres['anzahl']}"} for z in sres["zeilen"]], hide_index=True, width="stretch")
zeige(V.verdikt_summen(res1, s.delta))
zeige(V.verdikt_strafe(sres, s.tau))
st.caption("Die Gesamtmasse, die das Strafverfahren erzeugt, behandelt die Demo zu ungleichen Mengen; hier steht nur seine Konvergenz. "
           "Die Iteration des Strafverfahrens ist aus dem Gedächtnis nach Chizat u. a. (2018; Abstract und Abschnitt 2 gelesen) nachgebaut und nicht gegen deren Algorithmus geprüft.")

# ------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2
# ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 2: Nullen in der alten Matrix")
st.markdown(
    "Nullen bleiben Nullen. Ob es dann eine Matrix mit den neuen Randsummen gibt, hängt an der Nullstruktur und an den Randsummen, nicht nur an der Gesamtsumme: "
    "in [[1, 1], [0, 1]] hat Spalte 1 nur die Zeile 1 als Quelle, bei den Zeilenzielen (4, 6) und den Spaltenzielen (5, 5) fehlt ihr eine Einheit, obwohl beide Summen 10 sind. "
    "Die Klassen: **0** keine Lösung, **1** eine Lösung nur mit zusätzlichen Nullen (in den gemessenen Fällen fällt der Randfehler nur etwa wie 1/k), **2** eine auf dem ganzen Träger streng positive Lösung."
)
with st.spinner("Rechne Experiment 2 …"):
    res2 = _nullen(s.randsummen, s.inst2, s.form)
d1, d2 = st.columns(2)
with d1:
    st.plotly_chart(VIS.build_klassen(res2), key="exp2_klassen", width="stretch", config={"displayModeBar": False})
with d2:
    st.plotly_chart(VIS.build_durchlaeufe_klasse2(res2), key="exp2_durchlaeufe", width="stretch", config={"displayModeBar": False})
st.dataframe([{"Nullanteil": de(z["nullanteil"], 1), "ohne Lösung (Klasse 0)": z["klasse0"], "Klasse 1": z["klasse1"], "Klasse 2": z["klasse2"],
               "konvergiert in Klasse 2": f"{z['konv2']} von {z['klasse2']}", "NaN-Abbrüche in Klasse 0": z["nan0"],
               "Durchläufe in Klasse 2 (Median)": ganz(z["durchlaeufe2"][0]) if z["durchlaeufe2"] else "–"} for z in res2["zeilen"]], hide_index=True, width="stretch")
zeige(V.verdikt_nullen(res2, s.nullanteil2))
st.caption("In der u-v-Form endet ein Lauf ohne Lösung bei beliebigen und bei gleichen Randsummen meist in NaN (Teilung durch null); bei Randsummen aus Wachstumsfaktoren erreicht er oft nur die Obergrenze mit endlichem Zeilenfehler "
           "(die Spalte „NaN-Abbrüche“ zählt es je Reihe). In der Matrix-Form bleibt stets ein endlicher Zeilenfehler (gemessen bei beliebigen Randsummen: kein NaN); beide Formen konvergieren nicht, die Meldung stammt von der App.")

# ------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3
# ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 3: ε aus der Beobachtung zurückrechnen")
st.markdown(
    "In der Verkehrsplanung ist β = 1/ε kein freier Regler, sondern kalibriert: β wird so gesetzt, dass die mittlere Reiseweite des Modells der beobachteten gleicht. Das Furness-Verfahren liefert β nicht, "
    "die Kalibrierung ist eine **äußere Schleife** (hier ein Sekantenverfahren): jeder **Schritt** wertet den Abgleich einmal vollständig aus, die **Durchläufe** sind die Summe aller inneren Durchläufe."
)
with st.spinner("Rechne Experiment 3 …"):
    res3 = _kalibrierung(s.theta, s.beobachtung, s.startregel, s.warmstart == C.WARM_WARM, s.n_mess)
    wahr = _wahrheit(s.n_mess, s.startregel)
if res3["spur"]["schritte"] > 0:
    st.plotly_chart(VIS.build_kalibrierung_spur(res3["spur"]), key="exp3_spur", width="stretch", config={"displayModeBar": False})
    st.caption(f"Die Schleife an der ersten Instanz: {res3['spur']['schritte']} Schritte, Status {'konvergiert' if res3['spur']['status'] == 0 else 'abgebrochen'}.")
zeige(V.verdikt_kalibrierung(res3))
st.markdown("**Probe mit bekannter Wahrheit:** der Plan zu einem bekannten ε liefert die Beobachtung; die Schleife rechnet ε zurück.")
st.dataframe([{"Wahrheit ε/OPT": de_g(z["r"], 2), "Anteil θ (Median)": de_g(z["theta"][0], 2), "Schritte": E.band_ganz(z["kalt"]["schritte"]),
               "Durchläufe kalt": E.band_ganz(z["kalt"]["durchlaeufe"]), "Durchläufe warm": E.band_ganz(z["warm"]["durchlaeufe"]), "Warmstart spart": prozent(z["warm_spart"], 0),
               "größte Abweichung von ε": de_g(z["kalt"]["rueck"], 2)} for z in wahr["zeilen"]], hide_index=True, width="stretch")
st.caption(f"n = {wahr['n']}, {wahr['anzahl']} Instanzen, {wahr['startregel']}; Median [Min; Max]. Die Zahl der Schritte und Durchläufe kann an der Startregel hängen (eigene Wahl; ein Vergleich beider Regeln liegt nur für θ = 0,1 vor).")

# ------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4
# ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 4: Randsummen stimmen, die Matrix nicht")
st.markdown(
    "Auch wenn der Abgleich konvergiert, stimmen nur die Randsummen: sie legen $2n - 1$ von $n^2$ Zahlen fest. Hier ist die Wahrheit eine bekannte Matrix; "
    "man kennt nur ihre Randsummen (und für den Gravitationskern ihre mittlere Reiseweite) und eine gestörte alte Matrix. Gemessen wird der **Anteil der Gesamtmenge in der falschen Zelle**."
)
with st.spinner("Rechne Experiment 4 (bei anderen Einstellungen als den Standardwerten rechnet die Gravitationszeile live: auf dem Entwicklungsrechner 10 bis 35 Sekunden, je nach Last) …"):
    res4 = _randsummen(s.familie, s.n_mess, s.inst4)
    grav = _gravitation(s.familie, s.n_mess, s.inst4)
st.plotly_chart(VIS.build_vergleich(res4, grav, s.sigma), key="exp4_vergleich", width="stretch", config={"displayModeBar": False})
zeile4 = min(res4["sigmas"], key=lambda z: abs(z["sigma"] - s.sigma))
st.plotly_chart(VIS.build_drei_matrizen(zeile4["bild"]), key="exp4_matrizen", width="stretch", config={"displayModeBar": False})
st.caption("Erste Instanz, Fahrten bei Gesamtmenge 1 000; Farbe der Differenz: blau = der Abgleich hat zu viel, rot = zu wenig in der Zelle.")
st.dataframe([{"σ": de(z["sigma"], 1), "abgeglichen (Median)": de(z["abgeglichen"][0], 3), "ohne Abgleich (Median)": de(z["ohne"][0], 3),
               "größter Randsummenfehler": de_g(z["rand"], 2)} for z in res4["sigmas"]], hide_index=True, width="stretch")
zeige(V.verdikt_randsummen(res4, grav, s.sigma))

# ------------------------------------------------------------------------------------------------------------------------------------
# Annahmen
# ------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
**Eine Annahme steht im Mittelpunkt:** *Ein Abgleich, der durchläuft und dessen Randsummen stimmen, hat die gesuchte Matrix geliefert.* Sie scheitert in drei Stufen: keine Lösung (Summen, Nullen),
zu langsam (je nach Kern und ε) und richtige Ränder bei falschen Zellen.

| Annahme, die der Vorgänger stehen lässt | Was diese Demo aufhebt | Wo |
|---|---|---|
| Der Kern $K = \\exp(-C/\\varepsilon)$ ist überall positiv, beide Randverteilungen haben dieselbe Gesamtmasse, ε ist ein frei gewählter Regler, und „gelaufen“ heißt „gelöst“. | Nullen im Kern, verschiedene Gesamtmassen, ε aus einer Beobachtung zurückgerechnet. | Experimente 1 bis 3 |
| Minimum und Projektion sind dieselbe Aufgabe: *Minimum* ist der billigste Plan bei festen Randsummen, *Projektion* die der alten Matrix nächste Matrix mit neuen Randsummen (Kullback-Leibler). | Die Kosten sind hier nur die beobachtete mittlere Reiseweite, nicht das Ziel. | Primäransicht, Experiment 3 |

**Weitere Annahmen, die diese Demo stehen lässt:** die alte Matrix und die neuen Randsummen sind exakt bekannt; die Randsummen sind Zählungen ohne Messfehler; die Zonen sind fest (die Zonengröße ändert Kennzahlen derselben Daten, ungemessen).

**Grenzen:** Alle Instanzen sind synthetisch (Zonen im Einheitsquadrat, n = 6 bis 12, Experiment 2 mit n = 8); **es wurde keine Realdatenmatrix gemessen**. Die Zahlen gelten in diesem Modell, bei diesen Parametern,
mit der genannten Zähleinheit (Durchläufe, Matrix-Vektor-Produkte oder Instanzen) und dem Zufalls-Seed der Reihen. Ob kalibrierte β in der Verkehrsplanung bei ε/OPT zwischen 0,5 und 1 liegen, ist nicht gemessen.
Start und erster Schritt der Kalibrierschleife sind eigene Wahl. Das Strafverfahren ist aus dem Gedächtnis nachgebaut. Die Namen Wilson, Erlander/Stewart, Furness, Deming/Stephan, Fratar und Hyman
stehen nur, soweit andere Quellen sie nennen; ihre Originale sind nicht gelesen. Das Fratar-Verfahren rechnet diese Demo nicht. Das Wort „Sinkhorn-Distanz“ benutzt diese Demo nicht.

**Bekannte Probleme und Unsicherheiten:** Die Reihen rechnen mit einem festen Zufalls-Seed und wenigen Dutzend bis einigen Hundert Instanzen; einzelne Instanzen brauchen ein Vielfaches des Medians. Die Gravitationszeile von Experiment 4
ist für die Standardeinstellungen vorgerechnet; bei anderen Einstellungen rechnet sie live und braucht auf dem Entwicklungsrechner 10 bis 35 Sekunden (je nach Last), auf der Cloud wahrscheinlich länger. Die beiden Rechenformen (u-v und Matrix) zeigen bei unlösbaren Aufgaben
verschiedene Fehlerbilder. Keine der Aussagen gilt für Realdaten.

| | Wo es weitergeht |
|---|---|
| Rechengenauigkeit | Unterlauf und Rundung des Sinkhorn-Verfahrens bei kleinem ε (die Reihe zu Sinkhorn) |
| Ungleiche Mengen | Strafe und Teilbedienung statt Normalisieren, mit der Gesamtmasse bei der Strafe |
"""
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        """
**Lesehilfe.** Gegeben sind eine alte Matrix $T^0 \\ge 0$ und neue Randsummen; gesucht ist eine Matrix $P$ mit diesen Randsummen, die $T^0$ am nächsten liegt. Die Symbole:

| Symbol | Bedeutung |
|---|---|
| $T^0$ | alte Matrix (nichtnegativ, $n \\times m$) |
| $a$, $b$ | neue Zeilensummen (Herkunft) und Spaltensummen (Ziel); gleiche Summe, hier 1 |
| $K$ | Kern: $K = T^0$ (Furness) oder $K_{ij} = \\exp(-C_{ij}/\\varepsilon)$ (Gravitation) |
| $u$, $v$ | Zeilen- und Spaltenfaktoren, $P_{ij} = u_i K_{ij} v_j$ |
| $C$, $\\varepsilon$, OPT | Kosten (Abstand durch größten Abstand, also $\\max C = 1$); Temperatur; LP-Minimum $\\langle C, P \\rangle$ bei denselben Randsummen. Skala: ε/OPT |
| $\\theta$ | Anteil: $\\theta = (c_{obs} - \\text{OPT}) / (\\langle C, a b^\\top \\rangle - \\text{OPT})$ |
| Durchlauf, Schritt | ein Zeilen- plus Spaltenschritt (zwei Matrix-Vektor-Produkte); eine Auswertung des Abgleichs in der äußeren Schleife |

**Aufgabe.** $\\min_P \\sum_{ij} P_{ij} \\ln (P_{ij} / T^0_{ij})$ unter $\\sum_j P_{ij} = a_i$, $\\sum_i P_{ij} = b_j$, $P \\ge 0$. Wenn die Lösung dieser Aufgabe (die KL-nächste Matrix) auf dem ganzen Träger von $T^0$ streng positiv ist, hat sie die Form
$P_{ij} = u_i T^0_{ij} v_j$ (Benamou u. a. 2015, Nutz: gelesen).

**Rechenvorschrift.** $u \\leftarrow a / (K v)$, $v \\leftarrow b / (K^\\top u)$, elementweise geteilt. Bei der Gestalt $P = \\mathrm{diag}(u)\\, K\\, \\mathrm{diag}(v)$ hängt das **Kreuzverhältnis** (odds ratio)
$P_{11}P_{22}/(P_{12}P_{21})$ nicht von $u$ und $v$ ab: der Abgleich bewahrt es.

**Brücke.** Jede positive Matrix ist ein Gibbs-Kern: $T^0 = \\exp(-C/\\varepsilon)$ mit $C = -\\varepsilon \\ln T^0$ (einer von $\\varepsilon$ abhängigen Matrix, keiner Abstandsmatrix; sie kann negative Einträge haben).

**Handbeispiel 1 (Brüche).** $T^0 = \\begin{pmatrix}1 & 2\\\\ 3 & 4\\end{pmatrix}$, $a = (5, 6)$, $b = (5, 6)$. Zeilenschritt: Zeile 1 mit $5/3$, Zeile 2 mit $6/7$, Spaltensummen $89/21$ und $142/21$.
Spaltenschritt: $\\begin{pmatrix}175/89 & 210/71\\\\ 270/89 & 216/71\\end{pmatrix}$, Zeilenfehler (L1) $960/6319 \\approx 0{,}152$. Grenzwert $Q = \\begin{pmatrix}2 & 3\\\\ 3 & 3\\end{pmatrix}$ mit
$u = (2, 1)$, $v = (1, 3/4)$, Kreuzverhältnis $2/3$ in jedem Schritt.

**Handbeispiel 2 (Summen verschieden).** Zeilenziele $(5, 6)$, Spaltenziele $(10, 12)$: der Abgleich pendelt zwischen $2Q$ und $Q$, der Zeilenfehler ist $|B - A| = 11$ in jedem Durchlauf.

**Handbeispiel 3 (Nullen), $T^0 = \\begin{pmatrix}1 & 1\\\\ 0 & 1\\end{pmatrix}$.** Zeilenziele $(4, 6)$ und Spaltenziele $(3, 7)$: Lösung $\\begin{pmatrix}3 & 1\\\\ 0 & 6\\end{pmatrix}$ (Klasse 2). Bei $(5, 5)$: gleiche Summe 10, keine Lösung (Klasse 0),
weil Spalte 1 nur aus Zeile 1 gespeist wird und $5 > 4$. Bei Zielen $(1, 1)$: Lösung ist die Einheitsmatrix (Klasse 1), der Zeilenfehler nach $k$ Durchläufen ist (in Fahrten, Gesamtmasse 2) genau $2/(2k + 1)$; bezogen auf die Gesamtmasse 1 wie beim Abbruch ist er $1/(2k + 1)$, unter $10^{-9}$ erst nach etwa $5 \\cdot 10^8$ Durchläufen.

**Zulässigkeit.** Für Randsummen mit gleicher Summe gibt es eine Matrix auf dem Träger genau dann, wenn für jede Zeilenmenge $I$ gilt $a(I) \\le b(N(I))$ ($N(I)$: Spalten, die $I$ berührt; Hall-Bedingung, Menon, Brualdi, nach Idel 2016 gelesen).
Gilt für eine Menge Gleichheit und hat der Träger eine Zelle außerhalb von $I$ in $N(I)$, ist keine streng positive Lösung möglich (Klasse 1).

**Kalibrierung.** Gesucht ist $\\beta$ mit $\\langle C, P(\\beta) \\rangle = c_{obs}$, $P(\\beta)$ = Abgleich mit $K = \\exp(-\\beta C)$ (monoton fallend in $\\beta$). Handbeispiel 4 ($C = \\begin{pmatrix}0 & 1\\\\ 1 & 0\\end{pmatrix}$,
$a = (2/5, 3/5)$, $b = (3/5, 2/5)$, OPT $= 1/5$): bei beobachteten Kosten $0{,}24$ ($\\theta = 1/8$) ist $\\varepsilon^* = 2/\\ln(361/11) = 0{,}573$, also $\\varepsilon^*/\\text{OPT} = 2{,}86$; bei $0{,}40$ ($\\theta = 5/8$) ist $\\varepsilon^* = 2/\\ln 3 = 1{,}82$, also $9{,}10$.
Kosten unter OPT sind für kein $\\varepsilon$ erreichbar.

**Randsummen und Zellen.** Die $2n$ Bedingungen haben Rang $2n - 1$; es bleiben $(n-1)^2$ freie Zahlen. Handbeispiel 5: Wahrheit $\\begin{pmatrix}4 & 2\\\\ 2 & 4\\end{pmatrix}$ und Abgleich der Einsmatrix $\\begin{pmatrix}3 & 3\\\\ 3 & 3\\end{pmatrix}$
haben dieselben Randsummen $(6, 6)$ und $(6, 6)$; ein Sechstel der Menge liegt in der falschen Zelle.

**Quellen und was gelesen ist.** Gelesen: Idel 2016 (arXiv 1609.06349; Furness, RAS, Raking und Sinkhorn-Knopp als dasselbe Verfahren, Existenz und Konvergenz bei Nullen), Fahli/Galichon 2026 (arXiv, **Preprint 2026**),
Benamou u. a. 2015 und Nutz (KL-Projektion, Schrödinger-Brücke), van der Zijpp 1996 und Yan u. a. 2014 (doppelt beschränktes Gravitationsmodell, β über die mittlere Reiseweite), Chizat u. a. 2018 (nur Abstract und Abschnitt 2; die Iteration der Demo ist nachgebaut),
Li/Chen/Jin 2026 (**Preprint 2026**, nur Abstract und Einleitung). **Nicht im Original gelesen** (nur über die genannten Quellen bekannt): Wilson 1967 bis 1970, Erlander/Stewart 1990, Deming/Stephan 1940, Furness 1962, Hyman 1969, Fratar.
Die Zahlen dieser Seite sind eigene Messungen an synthetischen Instanzen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Optimaler Transport: vom Plan zum Abstand](https://sebastianhanisch.net/konzepte-optimaler-transport.html)."
)
