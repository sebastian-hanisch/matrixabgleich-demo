"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Beschriftungen von Zellen sind Annotationen (kein Text-Trace, keine texttemplate); alle Annotationen einer Figur werden auf einmal gesetzt."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mab_constants as C

MAX_ZELLEN_MIT_ZAHL = 144      # bis 12 x 12 steht die Zahl in der Zelle
LOG_BODEN = 1e-16


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def de1(x: float, stellen: int = 1) -> str:
    return f"{x:.{stellen}f}".replace(".", ",")


def build_matrix(P, a, b, titel="", hoehe=380, stellen=1):
    """Eine Matrix in Fahrten als Heatmap mit Zahl in jeder Zelle; Achsenbeschriftungen zeigen „ist von Ziel“ der Zeilen- und Spaltensummen."""
    P = np.asarray(P, float)
    n, m = P.shape
    zeilen = np.nan_to_num(P.sum(axis=1), nan=float("nan"))
    spalten = P.sum(axis=0)
    xs = [f"Ziel {j + 1}" for j in range(m)]
    ys = [f"Herkunft {i + 1}" for i in range(n)]
    z = np.where(np.isfinite(P), P, np.nan)
    zmax = float(np.nanmax(z)) if np.isfinite(z).any() and np.nanmax(z) > 0 else 1.0
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ys, colorscale=[[0, "#eef3f2"], [1, C.COLOR_OK]], showscale=False, xgap=1, ygap=1, zmin=0, zmax=zmax,
                               hovertemplate="%{y} → %{x}: %{z:.4f}<extra></extra>"))
    if n * m <= MAX_ZELLEN_MIT_ZAHL:
        beschriftungen = []
        for i in range(n):
            for j in range(m):
                wert = P[i, j]
                text = "–" if not np.isfinite(wert) else ("0" if wert == 0 else de1(wert, stellen))
                hell = np.isfinite(wert) and wert > 0.55 * zmax
                beschriftungen.append(dict(x=xs[j], y=ys[i], text=text, showarrow=False, font=dict(size=12 if n <= 4 else 9, color="white" if hell else "#1b3b37")))
        fig.update_layout(annotations=beschriftungen)
    fig.update_xaxes(side="top", tickmode="array", tickvals=xs, ticktext=[f"Z{j + 1}<br>{de1(spalten[j])} von {de1(b[j])}" for j in range(m)], tickfont=dict(size=9))
    fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=ys, ticktext=[f"H{i + 1}: {de1(zeilen[i])} von {de1(a[i])}" for i in range(n)], tickfont=dict(size=9))
    fig = _base(fig, height=hoehe, legend=False)
    if titel:
        fig.update_layout(title=dict(text=titel, font=dict(size=13)), margin=dict(l=10, r=10, t=95, b=10))
    return fig


def build_fehlerkurve(fehler, k_marker, schranke=None, einheit="Fahrten"):
    """Zeilenfehler (L1) nach jedem Durchlauf, logarithmische Achse; Marke beim gezeigten Durchlauf; optional die untere Schranke |B − A|."""
    fehler = np.asarray(fehler, float)
    xs = np.arange(1, len(fehler) + 1)
    y = np.where(np.isfinite(fehler) & (fehler > 0), fehler, np.nan)
    fig = go.Figure(go.Scatter(x=xs, y=y, mode="lines", name="Zeilenfehler", line=dict(color=C.COLOR_LINE, width=2), hovertemplate="Durchlauf %{x}: %{y:.3g}<extra></extra>"))
    if schranke is not None and schranke > 0:
        fig.add_hline(y=schranke, line=dict(color=C.COLOR_ZIEL, dash="dash"), annotation_text="|B − A|", annotation_position="top right")
    if 1 <= k_marker <= len(fehler) and np.isfinite(y[k_marker - 1]):
        fig.add_trace(go.Scatter(x=[k_marker], y=[y[k_marker - 1]], mode="markers", name="gezeigter Durchlauf", marker=dict(size=11, color=C.COLOR_BAD)))
    fig.update_xaxes(title_text="Durchlauf", type="log" if len(fehler) > 60 else "linear")
    fig.update_yaxes(title_text=f"Zeilenfehler ({einheit})", type="log")
    return _base(fig, height=300)


def build_summen_verlauf(zeile):
    """Experiment 1: Zeilenfehler gegen Durchlauf für alle Instanzen einer Stufe (dünne Linien) und die Linie |B − A| (relativ zur Summe der Zeilenziele)."""
    ks = zeile["verlauf_ks"]
    mat = np.array(zeile["verlauf"], float)          # Stützstellen × Instanzen
    fig = go.Figure()
    for j in range(mat.shape[1]):
        gut = np.isfinite(mat[:, j]) & (mat[:, j] > 0)             # nach der Konvergenz aller Läufe gibt es keine weiteren Durchläufe: keine NaN-Punkte an Plotly schicken
        fig.add_trace(go.Scatter(x=[k for k, g in zip(ks, gut) if g], y=mat[gut, j], mode="lines", line=dict(color="rgba(76,120,168,0.45)", width=1), showlegend=False, hovertemplate="Durchlauf %{x}: %{y:.3g}<extra></extra>"))
    if zeile["delta"] > 0:
        fig.add_hline(y=zeile["delta"], line=dict(color=C.COLOR_ZIEL, dash="dash"), annotation_text="|B − A|", annotation_position="top right")
    fig.update_xaxes(title_text="Durchlauf", type="log")
    fig.update_yaxes(title_text="Zeilenfehler (relativ)", type="log")
    return _base(fig, height=320, legend=False)


def build_strafe(sres, tau_gewaehlt=None):
    """Experiment 1, Strafverfahren: Abweichung vom exakten Furness und Zahl der Durchläufe je τ (doppelt logarithmisch)."""
    taus = [z["tau"] for z in sres["zeilen"]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Abweichung vom exakten Furness", "Durchläufe"))
    for col, (name, farbe) in enumerate((("abweichung", C.COLOR_LINE), ("durchlaeufe", C.COLOR_ZIEL)), start=1):
        med = [z[name][0] for z in sres["zeilen"]]
        lo = [z[name][1] for z in sres["zeilen"]]
        hi = [z[name][2] for z in sres["zeilen"]]
        fig.add_trace(go.Scatter(x=taus, y=med, mode="lines+markers", line=dict(color=farbe), marker=dict(size=9), error_y=dict(type="data", symmetric=False, array=[h - m for h, m in zip(hi, med)],
                                 arrayminus=[m - l for m, l in zip(med, lo)]), showlegend=False, hovertemplate="τ = %{x}: %{y:.3g}<extra></extra>"), row=1, col=col)
    fig.update_xaxes(title_text="τ (bei ε = 1)", type="log")
    fig.update_yaxes(type="log")
    fig = _base(fig, height=300, legend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_klassen(res):
    """Experiment 2: Zahl der Instanzen je Klasse über dem Nullanteil (gestapelt) und darüber der Anteil konvergierter Läufe."""
    xs = [z["nullanteil"] for z in res["zeilen"]]
    fig = go.Figure()
    for c, name, farbe in ((2, "Klasse 2: streng positive Lösung", C.COLOR_OK), (1, "Klasse 1: nur mit Nullen lösbar", C.COLOR_ZIEL), (0, "Klasse 0: keine Lösung", C.COLOR_BAD)):
        fig.add_trace(go.Bar(x=xs, y=[z[f"klasse{c}"] for z in res["zeilen"]], name=name, marker_color=farbe, hovertemplate=name + ": %{y}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Nullanteil der alten Matrix", tickmode="array", tickvals=xs, ticktext=[de1(x) for x in xs])
    fig.update_yaxes(title_text="Instanzen")
    return _base(fig, height=300)


def build_durchlaeufe_klasse2(res):
    """Experiment 2: Median der Durchläufe in Klasse 2 über dem Nullanteil, mit kleinstem und größtem Wert."""
    zs = [z for z in res["zeilen"] if z["durchlaeufe2"]]
    xs = [z["nullanteil"] for z in zs]
    med = [z["durchlaeufe2"][0] for z in zs]
    fig = go.Figure(go.Scatter(x=xs, y=med, mode="lines+markers", line=dict(color=C.COLOR_LINE), marker=dict(size=8), name="Median",
                               error_y=dict(type="data", symmetric=False, array=[z["durchlaeufe2"][2] - z["durchlaeufe2"][0] for z in zs],
                                            arrayminus=[z["durchlaeufe2"][0] - z["durchlaeufe2"][1] for z in zs]), hovertemplate="Nullanteil %{x}: %{y} Durchläufe<extra></extra>"))
    fig.update_xaxes(title_text="Nullanteil der alten Matrix")
    fig.update_yaxes(title_text="Durchläufe in Klasse 2", type="log")
    return _base(fig, height=300, legend=False)


def build_kalibrierung_spur(spur):
    """Experiment 3: die Schleife an der ersten Instanz: ε/OPT je Schritt (links) und mittlere Reiseweite je Schritt gegen die Beobachtung (rechts)."""
    s = spur["schritte"]
    xs = list(range(1, s + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ε/OPT je Schritt", "mittlere Reiseweite je Schritt"))
    fig.add_trace(go.Scatter(x=xs, y=spur["eps_opt"][:s], mode="lines+markers", line=dict(color=C.COLOR_LINE), showlegend=False, hovertemplate="Schritt %{x}: ε/OPT = %{y:.3g}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=spur["weite"][:s], mode="lines+markers", line=dict(color=C.COLOR_LINE), showlegend=False, hovertemplate="Schritt %{x}: %{y:.4g}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=spur["c_obs"], line=dict(color=C.COLOR_ZIEL, dash="dash"), annotation_text="beobachtet", annotation_position="top right", row=1, col=2)
    fig.update_xaxes(title_text="Schritt", dtick=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig = _base(fig, height=300, legend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_vergleich(res, grav, sigma):
    """Experiment 4: Anteil der Menge in der falschen Zelle je Verfahren (Median, Balken von kleinstem bis größtem Wert)."""
    z = min(res["sigmas"], key=lambda r: abs(r["sigma"] - sigma))
    zeilen = [("Produktmatrix a·bᵀ", res["produktmatrix"], "#9aa5b1"), ("alte Matrix ohne Abgleich", z["ohne"], C.COLOR_ZIEL)]
    if grav and grav.get("tv"):
        zeilen.insert(1, ("Gravitationskern, β aus der Reiseweite", grav["tv"], C.COLOR_LINE))
    zeilen.append((f"alte Matrix abgeglichen (σ = {de1(z['sigma'])})", z["abgeglichen"], C.COLOR_OK))
    fig = go.Figure(go.Bar(y=[n for n, _, _ in zeilen], x=[b[0] for _, b, _ in zeilen], orientation="h", marker_color=[f for _, _, f in zeilen],
                           error_x=dict(type="data", symmetric=False, array=[b[2] - b[0] for _, b, _ in zeilen], arrayminus=[b[0] - b[1] for _, b, _ in zeilen]),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Anteil der Menge in der falschen Zelle", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height=300, legend=False)


def build_drei_matrizen(bild, titel_links="Wahrheit", titel_mitte="abgeglichen", gesamt=C.GESAMT_FAHRTEN):
    """Experiment 4: Wahrheit, Abgleichsergebnis und Differenz der ersten Instanz (Fahrten bei Gesamtmenge 1 000), ohne Zahlen in den Zellen."""
    T = np.array(bild["wahrheit"]) * gesamt
    P = np.array(bild["abgeglichen"]) * gesamt
    D = P - T
    n = T.shape[0]
    namen = [f"{i + 1}" for i in range(n)]
    zmax = float(max(T.max(), P.max()))
    dmax = float(np.abs(D).max()) or 1.0
    fig = make_subplots(rows=1, cols=3, subplot_titles=(titel_links, titel_mitte, "Differenz (abgeglichen − Wahrheit)"), horizontal_spacing=0.06)
    for col, (z, farben, lo, hi) in enumerate(((T, [[0, "#eef3f2"], [1, C.COLOR_OK]], 0, zmax), (P, [[0, "#eef3f2"], [1, C.COLOR_OK]], 0, zmax),
                                               (D, [[0, C.COLOR_BAD], [0.5, "#f7f7f7"], [1, C.COLOR_LINE]], -dmax, dmax)), start=1):
        fig.add_trace(go.Heatmap(z=z, x=namen, y=namen, colorscale=farben, zmin=lo, zmax=hi, showscale=False, xgap=1, ygap=1,
                                 hovertemplate="Herkunft %{y} → Ziel %{x}: %{z:.2f}<extra></extra>"), row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    fig = _base(fig, height=300, legend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig
