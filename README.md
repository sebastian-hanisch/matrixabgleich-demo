# Matrix abgleichen – dieselben vier Zeilen Code in vier Fächern – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-matrixabgleich-demo.streamlit.app/)**

Demo der Reihe **Optimaler Transport: Verteilungen ineinander verschieben** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie baut auf der Demo zum Transportproblem (der billigste Plan bei festen Randsummen) und auf der Demo zu Sinkhorn (dieselbe Skalierungsrechnung mit dem Kern exp(−C/ε)) auf. Vehikel sind **Zonen im Einheitsquadrat**: eine **alte Matrix** (zum Beispiel Fahrten von Herkunfts- zu Zielzonen aus einer früheren Zählung) soll auf neue **Randsummen** (Zeilensummen = Herkunft, Spaltensummen = Ziel) abgeglichen werden.

Die Rechnung heißt in der Verkehrsplanung **Furness-Verfahren**, in der Volkswirtschaft **RAS** (biproportionales Verfahren), in der Umfragestatistik **Raking** (iterative proportionale Anpassung, IPF) und in der Mathematik **Sinkhorn-Knopp**: abwechselnd Zeilen und Spalten anpassen, vier Zeilen Code. Dass es dieselbe Rechnung ist, ist ein Name, keine Entdeckung (Idel 2016 nennt Furness, RAS, iterative proportionale Anpassung und Sinkhorn-Knopp als dasselbe Verfahren, Fahli/Galichon 2026 nennt auch Raking; beide gelesen); diese Demo belegt es nur durch eine Zählung. Was sie zeigt, sind die Voraussetzungen und die Zählungen darum herum.

## Warum dieses Problem

Ein Abgleich, der durchläuft und dessen Randsummen stimmen, hat die gesuchte Matrix geliefert: diese Annahme scheitert in drei Stufen. (1) **Keine Lösung:** bei verschiedenen Summen konvergiert der Abgleich nie, und bei Nullen in der alten Matrix kann es trotz gleicher Summen keine Matrix mit der Nullstruktur geben; das Verfahren meldet nichts, es läuft weiter oder endet in NaN. (2) **Zu langsam und ε nicht frei:** der Abgleich braucht je nach Kern wenige bis viele Tausend Matrix-Vektor-Produkte, und im Gravitationsmodell der Verkehrsplanung ist ε aus einer beobachteten mittleren Reiseweite zurückzurechnen, mit einer äußeren Schleife über ganze Abgleiche. (3) **Richtige Ränder, falsche Zellen:** die Randsummen legen nur 2n − 1 von n² Zahlen fest.

Die Kanten zu den Vorgängern, im Wortlaut: *Der Kern K = exp(−C/ε) ist überall positiv, beide Randverteilungen haben dieselbe Gesamtmasse, ε ist ein frei gewählter Regler, und „gelaufen“ heißt „gelöst“.* Diese Demo hebt auf: Nullen im Kern, verschiedene Gesamtmassen, ε aus einer Beobachtung zurückgerechnet. Und: *Minimum* (der billigste Plan bei festen Randsummen) gegen *Projektion* (die der alten Matrix nächste Matrix mit neuen Randsummen, Kullback-Leibler); die Kosten sind hier nur die beobachtete mittlere Reiseweite.

## Modell

Gegeben eine nichtnegative alte Matrix T⁰ und Randsummen a (Zeilen), b (Spalten) mit gleicher Summe. Gesucht ist die Matrix P mit diesen Randsummen, die T⁰ im Sinn der relativen Entropie (Kullback-Leibler, KL) am nächsten liegt: min Σ P_ij ln(P_ij / T⁰_ij). Wenn die Lösung dieser Aufgabe (die KL-nächste Matrix) auf dem ganzen Träger von T⁰ streng positiv ist, hat sie die Form P_ij = u_i T⁰_ij v_j. Der **Gravitationskern** K = exp(−C/ε) mit C = Abstand der Zonenmittelpunkte durch den größten Abstand (max C = 1) statt T⁰ gibt das doppelt beschränkte Gravitationsmodell. Jede positive Matrix ist ein Gibbs-Kern: T⁰ = exp(−C/ε) mit C = −ε ln T⁰ für jedes ε; diese Matrix C hängt von ε ab, ist keine Abstandsmatrix und kann negative Einträge haben (max C = 1 gilt nicht). Die Formeln, die Handrechnungen in Brüchen und die Quellen mit dem, was gelesen ist, stehen im Expander „Mathematische Formulierung“ der App.

**Zählweise.** Ein **Durchlauf** (sweep) = ein Zeilen- plus ein Spaltenschritt = zwei Matrix-Vektor-Produkte (K v und Kᵀ u); die Prüfung des Randfehlers nach einem Durchlauf verwendet das K v des nächsten Durchlaufs, die Prüfung nach dem letzten Durchlauf braucht ein weiteres K v, das nicht mitgezählt ist. Abbruch: Zeilenfehler (L1, Summe der Zeilenziele 1) unter 1e-9 in float64, Obergrenze 5 000 Durchläufe; bei Erreichen steht eine Meldung. In der Hyman-Schleife (Experiment 3) ist ein **Schritt** ein vollständiger Abgleich, **Durchläufe** sind die Summe aller inneren Durchläufe. Der Name „Matrix-Vektor-Produkt“ kommt auch in der Demo zu PDLP vor und zählt dort Produkte mit der Restriktionsmatrix eines linearen Programms; hier ist es die dichte n × n-Matrix K: gleicher Name der Einheit, nicht gleiche Kosten je Produkt.

**Skala.** Im Gravitationsmodus nie rohes ε, immer ε/OPT mit OPT = LP-Minimum ⟨C, P⟩ bei denselben Randsummen (Gesamtmasse 1, max C = 1). Zahlen aus verschiedenen Skalen (ε/OPT hier, ε/D in der Demo zu Abständen von Verteilungen) stehen nie ohne Umrechnung nebeneinander.

## Methodik

- **Instanzen:** Zonen gleichverteilt im Einheitsquadrat; Herkunftssummen ganzzahlig 10 bis 99, Zielsummen 1 + Multinomial(S − n; Dirichlet(3)); alte Matrix (i) dicht, exp(−C/3) mal Rauschfaktor U(0,5; 1,5) je Zelle, (ii) Zählung (Poisson) mit natürlichen Nullen, (iii) zufälliger Träger mit gegebenem Nullanteil (jede Zeile und Spalte behält einen Eintrag), Einträge U(0,5; 2). Randsummen beliebig (je Seite Dirichlet(1)), aus Wachstumsfaktoren der alten Matrix (je Zeile und Spalte ein unabhängiger Faktor, Spaltenziele auf die Summe der Zeilenziele skaliert) oder gleich (1/n).
- **Zufall:** SplitMix64 aus dem Haus-Generator (`mab_rng.py`, bytegleich zur Vorlage), je Zweck und Instanz ein eigener Zufallsstrom, Zufalls-Seed der Reihen 2026 (nie der Regler „Zufalls-Seed“). Die Reihen laufen als Stapel über alle Instanzen zugleich.
- **Zulässigkeit** (Experiment 2): Hall-Bedingung über alle Zeilenmengen (exakt, n ≤ 12), geprüft gegen ein lineares Programm, gegen die Aufzählung aller 512 Träger für n = 3 und gegen die Aufzählung der Permutationen. Klassen: 0 keine Lösung, 1 nur mit zusätzlichen Nullen, 2 streng positive Lösung auf dem ganzen Träger.
- **Kalibrierung** (Experiment 3): Sekantenverfahren auf β = 1/ε; Startwert β = 1/c_obs und erster Schritt (β · c/c_obs oder 1,5 · β) sind eigene Wahl, nicht aus Hyman 1969 gelesen. Warm gestartet heißt: der innere Abgleich beginnt mit dem v des Schritts davor.
- **Strafverfahren** (Experiment 1): Skalierung für eine KL-Strafe statt harter Randsummen, aus dem Gedächtnis nach Chizat u. a. 2018 nachgebaut (Abstract und Abschnitt 2 gelesen), nicht gegen deren Algorithmus geprüft; gegen die direkte Minimierung der Zielfunktion geprüft.
- **OPT:** eigener Transportsimplex (Kopie aus der Demo zum Transportproblem, `mab_simplex.py`); zur Laufzeit kein scipy.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind eigene Messungen an synthetischen Instanzen (Zonen im Einheitsquadrat, n = 8 bis 12), bei diesen Parametern, mit der genannten Zähleinheit (Durchläufe, Matrix-Vektor-Produkte oder Instanzen). Es wurde keine Realdatenmatrix gemessen.

### Die Gleichheit ist ein Name (Einheit: größte relative Abweichung)

Furness und Sinkhorn-Skalierung in der Log-Domäne stimmen nach jedem der ersten 30 Durchläufe überein: größte relative Abweichung 1,5e-15 (20 Instanzen, n = 10, dichte Basismatrix).

### Experiment 1: die Summen passen nicht (n = 8, 20 Instanzen je Stufe, Obergrenze 5 000 Durchläufe)

| (B − A)/A | Läufe konvergiert | Zeilenfehler (Median, Gesamtmasse 1) | Durchläufe (Median) |
|---|---|---|---|
| 0,000 | 20 von 20 | 1,75e-10 | 6 |
| 0,001 | 0 von 20 | 0,001 | – |
| 0,010 | 0 von 20 | 0,01 | – |
| 0,100 | 0 von 20 | 0,1 | – |

(B − A)/A = 0,1: 0 von 20 Läufen konvergieren; der Zeilenfehler bleibt bei 0,1 = |B − A| (Gesamtmasse 1; n = 8, 20 Instanzen, Obergrenze 5 000 Durchläufe). Über alle drei Stufen (0,001, 0,01, 0,1) zusammen konvergieren 0 von 60 Läufen. Der Zeilenfehler ist in jedem Lauf genau diese Differenz (kleinster und größter Wert gleich; bei diesen dichten Matrizen, allgemein ist er mindestens |B − A|).

Strafverfahren statt harter Randsummen (ε = 1, Kern = Zählung mit natürlichen Nullen; Nullanteil der Zählungen im Median 0,025 [0; 0,07]):

| τ | Abweichung vom exakten Furness (Median) | Durchläufe (Median [Min; Max]) | Läufe konvergiert |
|---|---|---|---|
| 1 | 0,215 | 16 [14; 17] | 20 von 20 |
| 10 | 0,0429 | 100 [87; 103] | 20 von 20 |
| 100 | 0,00477 | 836 [714; 866] | 20 von 20 |
| 1000 | 0,000482 | 7 166 [5 951; 7 463] | 20 von 20 |

Mit KL-Strafe konvergieren 20 von 20 Läufen (n = 10, ε = 1, Kern = Zählung mit natürlichen Nullen). Ab τ = 10 sinkt die Abweichung vom exakten Furness je Zehnerschritt (zwei Schritte: 10, 100, 1 000) um den Faktor 9 bis 9,9 (in diesem Bereich etwa 1/τ) und die Zahl der Durchläufe steigt um den Faktor 8,4 bis 8,6 (in diesem Bereich etwa τ); bei τ = 1 gilt das nicht. Bei verschiedenen Summen (Zeilenziele +5 %): Furness konvergiert in 0 von 20 Läufen (Zeilenfehler 0,0476), das Strafverfahren mit τ = 100 in 20 von 20 (Zeilenfehler 0,0252); beide Zeilenfehler sind relativ zur Summe der Zeilenziele angegeben.

Die Gesamtmasse, die das Strafverfahren erzeugt, behandelt die Demo zu ungleichen Mengen; hier steht nur die Konvergenz.

### Experiment 2: Nullen in der alten Matrix (n = 8, je Nullanteil 100 Instanzen, Obergrenze 5 000 Durchläufe, u-v-Form)

Beliebige Randsummen (je Seite Dirichlet(1)):

| Nullanteil | Klasse 0 | Klasse 1 | Klasse 2 | konvergiert in Klasse 2 | NaN-Abbrüche in Klasse 0 | Durchläufe in Klasse 2 (Median) |
|---|---|---|---|---|---|---|
| 0,0 | 0 | 0 | 100 | 100 von 100 | 0 | 7 |
| 0,1 | 0 | 0 | 100 | 100 von 100 | 0 | 11 |
| 0,2 | 3 | 0 | 97 | 97 von 97 | 3 | 17 |
| 0,3 | 19 | 0 | 81 | 81 von 81 | 16 | 24 |
| 0,4 | 44 | 0 | 56 | 56 von 56 | 38 | 36 |
| 0,5 | 68 | 0 | 32 | 32 von 32 | 60 | 50 |
| 0,6 | 92 | 0 | 8 | 7 von 8 | 87 | 52 |
| 0,7 | 97 | 0 | 3 | 3 von 3 | 95 | 1 304 |

Nullanteil 0,5, Randsummen beliebig (Dirichlet(1) je Seite), n = 8, 100 Instanzen, u-v-Form: 68 Instanzen ohne Lösung (Klasse 0), 0 nur mit Nullen lösbar (Klasse 1), 32 streng positiv lösbar (Klasse 2). Konvergiert: 0 von 68 in Klasse 0, 32 von 32 in Klasse 2. Über alle Nullanteile von 0,0 bis 0,7 zusammen (800 Instanzen): 323 ohne Lösung, 0 in Klasse 1, 477 in Klasse 2; konvergiert 0 von 323 in Klasse 0, 476 von 477 in Klasse 2 (Obergrenze 5 000 Durchläufe). „Ohne Lösung“ ist mit der Hall-Bedingung geprüft. Median der Durchläufe in Klasse 2: 7 bei Nullanteil 0,0 und 1 304 bei 0,7 (Matrix-Vektor-Produkte doppelt so viele; bei 0,7 sind es nur 3 Instanzen in Klasse 2).

Bei Nullanteil 0,5, n = 8 und beliebigen Randsummen haben in zwei Sätzen zu je 1 000 Instanzen 662 und 708 keine Lösung, das sind 66,2 % und 70,8 % (in vier Sätzen zu je 100: 75, 60, 73, 67); bei Randsummen aus Wachstumsfaktoren der Breite U(0,7; 1,6) sind es 4 und 9 von je 1 000 (in vier Sätzen zu je 100: 0, 1, 2, 1). Die Zählungen zu je 100 Instanzen streuen stark, belastbarer sind die Sätze zu 1 000. Die Zahl für alle Nullanteile zusammen (800 Instanzen) und die für Nullanteil 0,5 allein stehen immer getrennt.

Randsummen aus Wachstumsfaktoren der alten Matrix, Breite U(0,7; 1,6): wer die Randsummen aus der alten Matrix ableitet, begegnet dem Problem später.

| Nullanteil | Klasse 0 | Klasse 1 | Klasse 2 | konvergiert in Klasse 2 | NaN-Abbrüche in Klasse 0 | Durchläufe in Klasse 2 (Median) |
|---|---|---|---|---|---|---|
| 0,0 | 0 | 0 | 100 | 100 von 100 | 0 | 6 |
| 0,1 | 0 | 0 | 100 | 100 von 100 | 0 | 8 |
| 0,2 | 0 | 0 | 100 | 100 von 100 | 0 | 10 |
| 0,3 | 0 | 0 | 100 | 100 von 100 | 0 | 13 |
| 0,4 | 0 | 0 | 100 | 100 von 100 | 0 | 16 |
| 0,5 | 1 | 0 | 99 | 99 von 99 | 1 | 23 |
| 0,6 | 6 | 0 | 94 | 94 von 94 | 1 | 38 |
| 0,7 | 31 | 0 | 69 | 68 von 69 | 16 | 66 |

Gleiche Randsummen 1/n (hier kommt Klasse 1 vor; Klasse 0 = keine positive Diagonale, Klasse 2 = totaler Träger, beide Zählungen sind für gleiche Randsummen dieselben Klassen):

| Nullanteil | Klasse 0 | Klasse 1 | Klasse 2 | konvergiert in Klasse 2 | NaN-Abbrüche in Klasse 0 | Durchläufe in Klasse 2 (Median) |
|---|---|---|---|---|---|---|
| 0,0 | 0 | 0 | 100 | 100 von 100 | 0 | 6 |
| 0,1 | 0 | 0 | 100 | 100 von 100 | 0 | 9 |
| 0,2 | 0 | 0 | 100 | 100 von 100 | 0 | 12 |
| 0,3 | 0 | 1 | 99 | 99 von 99 | 0 | 17 |
| 0,4 | 0 | 16 | 84 | 84 von 84 | 0 | 26 |
| 0,5 | 0 | 47 | 53 | 53 von 53 | 0 | 36 |
| 0,6 | 8 | 62 | 30 | 30 von 30 | 8 | 56 |
| 0,7 | 33 | 64 | 3 | 3 von 3 | 33 | 65 |

In den 190 Instanzen der Klasse 1 (Tabelle mit gleichen Randsummen, n = 8) konvergiert in keinem Lauf der Randfehler unter 1e-9, er fällt in diesen Instanzen etwa wie 1/k; für andere Klasse-1-Aufgaben ist das nicht gemessen. Von Hand: für die Matrix [[1, 1], [0, 1]] mit den Zielen (1, 1) ist der Zeilenfehler nach k Durchläufen in Fahrten (Gesamtmasse 2) genau 2/(2k + 1); bezogen auf die Gesamtmasse 1 wie beim Abbruch ist er 1/(2k + 1), unter 1e-9 erst nach etwa 5·10⁸ Durchläufen. In der u-v-Form wird der Zeilenfehler des Handbeispiels 3(b) ab Durchlauf 3 176 zu NaN (ein Unterlauf der Gleitkomma-Rechnung: die Zahl hängt von der Arithmetik ab, geprüft wird ein Band von 1 000 bis 20 000); in der Matrix-Form bleibt er nach 20 000 Durchläufen bei 2 Fahrten; beide Formen konvergieren nicht, die Meldung stammt von der App. Hat die Aufgabe eine streng positive Lösung (Klasse 2), konvergiert der Abgleich in den Reihen oben fast immer (bei Zufalls-Seed 2026 fehlt in jeder Tabelle höchstens ein Lauf an der Obergrenze von 5 000 Durchläufen; mit den Zufalls-Seeds 2026 bis 2029 fehlen in jeder der drei Tabellen 0 bis 2 Läufe an der Obergrenze); das gilt nur für diese Klasse.

Unter den Läufen ohne Lösung (Klasse 0, u-v-Form, alle Nullanteile) enden in NaN: bei beliebigen Randsummen 299 von 323, bei gleichen Randsummen 41 von 41, bei Randsummen aus Wachstumsfaktoren der Breite U(0,9; 1,1) 3 von 28, U(0,7; 1,6) 18 von 38 und U(0,4; 2,5) 54 von 84; die übrigen erreichen die Obergrenze mit endlichem Zeilenfehler. In der Matrix-Form endet bei beliebigen Randsummen keiner dieser Läufe in NaN (0 von 323). Die Reihen zu den Breiten U(0,9; 1,1) und U(0,4; 2,5) stehen in `tools/messreihen.json`.

### Experiment 3: ε aus der Beobachtung zurückrechnen (n = 12, 20 Instanzen, kalt gestartet, Startregel β · c/c_obs)

Beobachtet wird die mittlere Reiseweite c_obs = OPT + θ · (Produktmaß − OPT):

| θ | konvergiert | Schritte | Durchläufe | Matrix-Vektor-Produkte | ε*/OPT (Median [Min; Max]) | Verhältnis größter zu kleinster Wert |
|---|---|---|---|---|---|---|
| 0,01 | 20 von 20 | 10 [8; 12] | 3 944 [1 251; 11 088] | 7 887 [2 502; 22 176] | 0,354 [0,0858; 0,732] | 8,5 |
| 0,02 | 20 von 20 | 8 [6; 12] | 2 526 [977; 4 734] | 5 052 [1 954; 9 468] | 0,493 [0,183; 1] | 5,5 |
| 0,05 | 20 von 20 | 7 [5; 9] | 1 260 [529; 1 920] | 2 521 [1 058; 3 840] | 0,755 [0,379; 1,48] | 3,9 |
| 0,1 | 20 von 20 | 6 [5; 8] | 644 [355; 1 088] | 1 289 [710; 2 176] | 1,08 [0,537; 1,99] | 3,7 |
| 0,2 | 20 von 20 | 5 [4; 7] | 292 [204; 574] | 584 [408; 1 148] | 1,58 [0,781; 2,8] | 3,6 |
| 0,4 | 20 von 20 | 5 [3; 6] | 118 [87; 208] | 235 [174; 416] | 2,64 [1,28; 4,57] | 3,6 |
| 0,7 | 20 von 20 | 5 [5; 6] | 66 [54; 92] | 132 [108; 184] | 5,69 [2,75; 10,2] | 3,7 |
| 0,9 | 20 von 20 | 5 [4; 5] | 42 [37; 49] | 84 [74; 98] | 16,4 [7,86; 29,9] | 3,8 |

θ = 0,1 (n = 12, 20 Instanzen, kalt gestartet, erster Schritt β · c/c_obs): ε*/OPT im Median 1,08 [0,537; 1,99]; 20 von 20 Läufen konvergieren, 6 [5; 8] Schritte, 644 [355; 1 088] Durchläufe (1 289 [710; 2 176] Matrix-Vektor-Produkte). Dasselbe θ ergibt bei verschiedenen Instanzen verschiedene ε*/OPT: das Verhältnis des größten zum kleinsten Wert ist hier 3,7 (ein Extremwert aus 20 Instanzen, er hängt vom Zufalls-Seed ab). Dasselbe ε/OPT bedeutet bei verschiedenen Instanzen auch verschiedenen Aufwand (Probe mit bekannter Wahrheit unten). Startwert und erster Schritt der Schleife sind eigene Wahl; Wilson und Hyman sind nicht im Original gelesen. Dieses Verhältnis ist ein Extremwert aus 20 Instanzen: bei θ = 0,1 liegt es in den Zufalls-Seeds 2026 bis 2029 zwischen 3,2 und 7,9. Bei gleichem ε/OPT = 1 braucht die Kalibrierung in der Probe mit bekannter Wahrheit im Median 690 Durchläufe (kalt gestartet), einzelne Instanzen 147 bis 4 748.

Beobachtete Reiseweite 10 % unter dem LP-Minimum: kein ε kann sie erzeugen, denn kein Plan mit diesen Randsummen kostet weniger als das LP-Minimum. 0 von 20 Läufen konvergieren; bei 20 scheitert der innere Abgleich (β wächst; bei 16 unterläuft exp(−βC) und die Rechnung endet in NaN, bei 4 erreicht er die Obergrenze von 50 000 Durchläufen), β steht am Ende bei 4,1e3 (Median; Start 12). n = 12, 20 Instanzen, höchstens 40 Schritte.

Probe mit bekannter Wahrheit (der Plan zu ε = r · OPT liefert die Beobachtung, die Schleife rechnet ε zurück; Startregel β · c/c_obs):

| Wahrheit ε/OPT | θ (Median) | Schritte | Durchläufe kalt | Durchläufe warm | Warmstart spart | größte Abweichung von ε |
|---|---|---|---|---|---|---|
| 0,3 | 0,0073 | 10 [10; 11] | 5 450 [677; 29 232] | 4 268 [558; 19 707] | 22 % | 1,6e-5 |
| 1 | 0,087 | 6 [6; 7] | 690 [147; 4 748] | 542 [119; 3 705] | 21 % | 1,9e-6 |
| 3 | 0,46 | 5 [4; 5] | 106 [55; 429] | 82 [45; 350] | 23 % | 1,6e-6 |

Gemessen wird die Ersparnis als 1 − Median warm / Median kalt; in den Zufalls-Seeds 2026 bis 2029 spart der Warmstart 19 % bis 23 %. Die größte Abweichung von ε hängt vom Zufalls-Seed ab; die Abbruchtoleranz 1e-6 gilt für die Reiseweite, nicht für ε.

Mit der anderen Startregel (erster Schritt 1,5 · β) braucht θ = 0,1 im Median 7 Schritte und 736 Durchläufe statt 6 Schritte und 644 Durchläufe: bei θ = 0,1 hängt die Zahl der Schritte und Durchläufe an der Startregel (für andere θ nicht verglichen).

### Experiment 4: Randsummen stimmen, die Matrix nicht (n = 12, 50 Instanzen; Anteil der Gesamtmenge in der falschen Zelle)

Wahrheit **Gravitation mit Rauschen** (exp(−C/OPT) mal lognormales Paar-Rauschen mit σ = 0,5):

| σ der alten Matrix | abgeglichen (Median) | ohne Abgleich (Median) | größter Randsummenfehler |
|---|---|---|---|
| 0,0 | 0,000 | 0,000 | 2,1e-15 |
| 0,1 | 0,018 | 0,040 | 3,8e-15 |
| 0,3 | 0,053 | 0,119 | 3,8e-15 |
| 0,6 | 0,108 | 0,230 | 2e-15 |
| 1,0 | 0,184 | 0,366 | 3,8e-15 |

σ = 0,6: Anteil der Menge in der falschen Zelle 0,11 [0,013; 0,19] nach dem Abgleich, 0,23 [0,15; 0,29] ohne Abgleich, 0,67 [0,44; 0,83] mit der Produktmatrix, 0,095 [0,027; 0,17] mit dem Gravitationskern (β aus der mittleren Reiseweite, in 49 von 50 Instanzen berechnet) (Median [Min; Max], n = 12, 50 Instanzen, Wahrheit: Gravitation mit Rauschen). Der größte Randsummenfehler nach einem konvergierten Abgleich liegt bei 3,8e-15 (relativ zur kleinsten Randsumme; Maximum über alle Stufen und Instanzen, es hängt vom Zufalls-Seed ab; Schranke ist die Abbruchtoleranz 1e-12 geteilt durch die kleinste Randsumme). 4 der Läufe mit σ > 0 erreichen die Obergrenze von 50 000 Durchläufen; ihre Ergebnisse stehen in den Medianen, nicht in dieser Randsummenzahl. Der Abgleich hilft, aber seine Ränder zeigen den Rest nicht: die Randsummen legen 2n − 1 = 23 von 144 Zahlen fest, 121 bleiben frei. Gilt für synthetische Wahrheiten mit bekannter Struktur. Li/Chen/Jin (Preprint 2026, nur Abstract und Einleitung gelesen) vergleichen Verfahren an Streckenzählungen statt an Randsummen, ihr Befund überträgt sich nicht ohne Weiteres.

Wahrheit **andere Hemmfunktion** ((C + 0,05)⁻², ohne Rauschen):

| σ der alten Matrix | abgeglichen (Median) | ohne Abgleich (Median) | größter Randsummenfehler |
|---|---|---|---|
| 0,0 | 0,000 | 0,000 | 4,3e-16 |
| 0,1 | 0,011 | 0,039 | 5,4e-12 |
| 0,3 | 0,035 | 0,116 | 2,5e-12 |
| 0,6 | 0,075 | 0,232 | 2,8e-12 |
| 1,0 | 0,137 | 0,367 | 6,3e-12 |

σ = 0,6: Anteil der Menge in der falschen Zelle 0,075 [0,04; 0,12] nach dem Abgleich, 0,23 [0,12; 0,35] ohne Abgleich, 0,76 [0,68; 0,8] mit der Produktmatrix, 0,12 [0,085; 0,16] mit dem Gravitationskern (β aus der mittleren Reiseweite, in 48 von 50 Instanzen berechnet) (Median [Min; Max], n = 12, 50 Instanzen, Wahrheit: andere Hemmfunktion (C + 0,05)⁻²). Der größte Randsummenfehler nach einem konvergierten Abgleich liegt bei 6,3e-12 (relativ zur kleinsten Randsumme; Maximum über alle Stufen und Instanzen, es hängt vom Zufalls-Seed ab; Schranke ist die Abbruchtoleranz 1e-12 geteilt durch die kleinste Randsumme). Der Abgleich hilft, aber seine Ränder zeigen den Rest nicht: die Randsummen legen 2n − 1 = 23 von 144 Zahlen fest, 121 bleiben frei. Gilt für synthetische Wahrheiten mit bekannter Struktur. Li/Chen/Jin (Preprint 2026, nur Abstract und Einleitung gelesen) vergleichen Verfahren an Streckenzählungen statt an Randsummen, ihr Befund überträgt sich nicht ohne Weiteres.

Bei dieser Wahrheit ist die Matrix symmetrisch, also sind Zeilen- und Spaltensummen gleich und OPT ist null: ε/OPT ist dort nicht definiert, die Kalibrierung braucht sehr viele innere Durchläufe (Obergrenze 400 000).

## Ehrliche Grenzen

- **Keine Realdaten.** Alle Instanzen sind synthetisch; ob Fahrtenmatrizen der Verkehrsplanung eine ähnliche Spreizung haben wie die dichte Basismatrix hier, ist nicht gemessen. Die Aussagen gelten für Zonen im Einheitsquadrat, n = 6 bis 12, die genannten Randsummen und den Abbruch 1e-9 in float64.
- **Die Gleichheit der Verfahren ist trivial.** Die Demo kann nichts „entdecken“, nur zählen. Neuheit wird nicht behauptet.
- **Die Gegenrede:** In einer Preprint-Arbeit von 2026 (Li/Chen/Jin, arXiv 2611.09678, nur Abstract und Einleitung gelesen) passte die zellenweise Anpassung die Streckenzählungen laut Abstract in 65 von 66 Bedingungen am genauesten an, stellte die wahre Nachfrage aber in 45 von 66 am schlechtesten wieder her; das sind Streckenzählungen, ein anderer Datentyp als Randsummen, und die Zählungen zu treffen heißt nicht, die Nachfrage zu treffen. Experiment 4 zeigt nur, dass Randsummen die Zellen nicht festlegen, nicht dass der Abgleich auf Realdaten schlecht ist.
- **ε/OPT bei kalibrierten β:** ob kalibrierte β der Verkehrsplanung bei ε/OPT zwischen 0,5 und 1 liegen, ist nicht gemessen; nur der Zusammenhang zwischen θ und ε*/OPT auf synthetischen Zonen steht oben. Dasselbe ε/OPT bedeutet bei verschiedenen Instanzen verschiedenen Aufwand (gemessen in der Probe mit bekannter Wahrheit); dasselbe θ liefert verschiedene ε/OPT (Verhältnis größter zu kleinster Wert bei θ = 0,1 in vier Zufalls-Seeds 3,2 bis 7,9).
- **Namen und Quellen.** Gelesen sind Idel 2016, Fahli/Galichon 2026 (**Preprint 2026**), Benamou u. a. 2015, Nutz, van der Zijpp 1996, Yan u. a. 2014. Wilson, Erlander/Stewart, Deming/Stephan, Furness, Hyman und Fratar sind nicht im Original gelesen; ihre Namen stehen nur, soweit gelesene Quellen sie nennen, nie als „X zeigte“. Das Fratar-Verfahren rechnet die Demo nicht. Idel nennt Raking nicht; den Namen Raking nennt Fahli/Galichon, und dass die Umfragestatistik ihn verwendet, ist hier nicht aus einer gelesenen Quelle belegt. Ob eine bestimmte Behörde das Verfahren einsetzt, belegt diese Demo nicht.
- **Strafverfahren:** aus dem Gedächtnis nachgebaut; seine Gesamtmasse gehört zur Demo zu ungleichen Mengen.
- **Zonengröße:** dieselben Daten, in kleinere oder größere Zonen aggregiert, ändern Kennzahlen wie die mittlere Reiseweite (Small/Song 1992, nur zitiert); die Demo misst das nicht.
- **Rechenzeit auf der Cloud:** die Gravitationszeile von Experiment 4 ist für die Standardeinstellungen vorgerechnet (`tools/vorgerechnet.json`); bei anderen Einstellungen rechnet sie live und braucht auf dem Entwicklungsrechner 10 bis 35 Sekunden, je nach Last (gemessen 27 bis 34 Sekunden bei belastetem Rechner), auf der Cloud wahrscheinlich länger.
- **Die Hall-Bedingung** ist für n ≤ 12 exakt (höchstens 4 095 Zeilenmengen); größere Zonenzahlen sind nicht vorgesehen.

## Zahl → Test

Jede Zahl der Tabellen und Sätze oben wird aus `tools/messreihen.json` mit den Formatfunktionen der App gebildet und muss wörtlich in dieser README stehen (`tests/test_claims.py`); kleine Reihen werden neu gerechnet und als Band mit der Datei verglichen (Zählungen ±2 und 2 %, Gleitkomma relativ 1e-6). Handrechnungen stehen als Rechnung in Brüchen im Test.

| Zahl oder Aussage | Test |
|---|---|
| Handbeispiel 1: [[175/89, 210/71], [270/89, 216/71]], Zeilenfehler 960/6319, Grenzwert [[2, 3], [3, 3]], 5 Durchläufe, Faktor 0,01 je Durchlauf, Kreuzverhältnis 2/3 | `tests/test_scaling.py::test_handbeispiel_1_erster_durchlauf_in_bruechen_und_im_code`, `tests/test_scaling.py::test_handbeispiel_1_grenzwert_und_faktoren`, `tests/test_scaling.py::test_kreuzverhaeltnis_bleibt_in_jedem_schritt_exakt_erhalten` |
| 10 Durchläufe = 20 Matrix-Vektor-Produkte (+ 1 für die letzte Prüfung); der gezeigte Quelltext ist der ausgeführte | `tests/test_scaling.py::test_zaehlweise_zehn_durchlaeufe_kosten_zwanzig_produkte_plus_eine_pruefung`, `tests/test_scaling.py::test_die_gezeigte_schleife_ist_die_ausgefuehrte_und_gleich_der_buchfuehrung` |
| Handbeispiel 2: Zweier-Zyklus, Zeilenfehler 11 = \|B − A\| | `tests/test_scaling.py::test_handbeispiel_2_zweier_zyklus_und_zeilenfehler_gleich_differenz_der_summen` |
| Handbeispiel 3: Klassen 2, 0, 1; Fehler 2/(2k + 1) in Brüchen und 1/(2k + 1) < 1e-9 ab k = 5·10⁸; NaN der u-v-Form im Band 1 000 bis 20 000 | `tests/test_scaling.py::test_handbeispiel_3a_konvergiert_gegen_die_loesung_mit_nullen`, `tests/test_scaling.py::test_handbeispiel_3b_gleiche_summe_aber_keine_loesung_beide_formen_konvergieren_nicht`, `tests/test_scaling.py::test_handbeispiel_3c_randfehler_ist_genau_zwei_durch_zwei_k_plus_eins`, `tests/test_feasibility.py::test_handbeispiel_3_klassen_gegen_das_lp_und_die_hall_bedingung` |
| Handbeispiel 4: θ = 1/8 und 5/8, ε*/OPT = 2,86 und 9,10, q² = 11/361 und 1/3 | `tests/test_oracle_matrixabgleich.py::test_handbeispiel_4_anteil_theta_und_epsilon_ueber_opt_in_bruechen_und_im_code`, `tests/test_oracle_matrixabgleich.py::test_kalibrierung_bei_zwei_zonen_gleich_der_geschlossenen_formel` |
| Handbeispiel 5: ein Sechstel in der falschen Zelle, Rang 2n − 1 | `tests/test_scaling.py::test_handbeispiel_5_ein_sechstel_in_der_falschen_zelle_und_rang_2n_minus_1` |
| Furness gleich Sinkhorn in der Log-Domäne, gleich KL-Projektion, gleich SLSQP | `tests/test_oracle_matrixabgleich.py::test_furness_gleich_sinkhorn_in_der_log_domaene_nach_jedem_durchlauf`, `tests/test_oracle_matrixabgleich.py::test_abgleich_gleich_kl_projektion_ueber_die_lagrange_gleichungen`, `tests/test_oracle_matrixabgleich.py::test_abgleich_gleich_slsqp_auf_der_kl_zielfunktion` |
| „ohne Lösung“ (Hall-Bedingung) gegen ein lineares Programm, gegen alle 512 Träger, gegen die Permutationen | `tests/test_feasibility.py::test_hall_klassen_gleich_lp_klassen_auf_vierhundert_zufallsaufgaben_alle_drei_klassen_kommen_vor`, `tests/test_feasibility.py::test_alle_512_traeger_fuer_n_gleich_3_gegen_das_lp`, `tests/test_feasibility.py::test_traeger_klasse_gleich_aufzaehlung_der_permutationen_und_gleich_hall_bei_gleichen_randsummen` |
| OPT gegen HiGHS und `networkx`; Hyman-Schleife gegen `brentq`; Strafverfahren gegen die direkte Minimierung | `tests/test_oracle_matrixabgleich.py::test_opt_eigener_simplex_gleich_highs_gleich_networkx`, `tests/test_oracle_matrixabgleich.py::test_hyman_schleife_gleich_brentq_und_findet_das_wahre_epsilon_wieder`, `tests/test_oracle_matrixabgleich.py::test_strafverfahren_gleich_direkter_minimierung_der_zielfunktion` |
| Tabellen und Sätze der Experimente 1 bis 4 | `tests/test_claims.py::test_readme_enthaelt_jede_zeile_der_messreihen`, `tests/test_claims.py::test_readme_enthaelt_jeden_satz_der_verdikte`, `tests/test_claims.py::test_kleine_reihen_neu_gerechnet_stimmen_als_band_mit_der_datei` |
| NaN-Anteil je Art der Randsummen, Klasse 1 in 190 Instanzen, Bänder über andere Zufalls-Seeds, gleiche Bezugsgröße der Zeilenfehler (Furness gegen Strafe), Ursache des Randfalls aufgeteilt | `tests/test_claims.py::test_nan_anteil_in_klasse_0_haengt_an_der_art_der_randsummen_und_die_matrix_form_hat_kein_nan`, `tests/test_claims.py::test_klasse_1_konvergiert_in_keinem_lauf_und_der_randfehler_faellt_in_den_gemessenen_instanzen_wie_1_durch_k`, `tests/test_claims.py::test_andere_zufalls_seeds_stuetzen_die_baender_der_readme`, `tests/test_claims.py::test_anderer_zufalls_seed_neu_gerechnet_stimmt_mit_der_datei_ueberein`, `tests/test_claims.py::test_summen_probe_furness_und_strafe_zeilenfehler_haben_dieselbe_bezugsgroesse` |
| Presets: „Von Hand“ 5 Durchläufe, „Summen +5 %“ ohne Lösung, „Nullanteil 0,5“ NaN, „ε kalibrieren“ gibt ε zurück | `tests/test_evaluation.py::test_preset_von_hand_5_durchlaeufe_und_die_zahlen_der_handrechnung`, `tests/test_evaluation.py::test_preset_summen_plus_5_prozent_meldet_keine_loesung`, `tests/test_evaluation.py::test_preset_nullanteil_gleich_halb_hat_keine_loesung_und_meldet_nan`, `tests/test_evaluation.py::test_preset_epsilon_kalibrieren_rechnet_das_vorgegebene_epsilon_zurueck` |

## Tests

**168 Tests** (`pytest --collect-only -q tests`), seriell in rund 2 Minuten (119 und 134 Sekunden gemessen): Handrechnungen in Brüchen, Orakel (unabhängige Rechenwege, s. o.), Eigenschaftstests (Skalierung, Umnummerieren, Transponieren, Kreuzverhältnisse in Brüchen, Nullen bleiben Nullen, Strafverfahren gegen den Abgleich), Zulässigkeit gegen ein lineares Programm, Verdikte an handgesetzten Zeilen, Claims (README gegen `tools/messreihen.json`, kleine Reihen neu gerechnet), Presets, Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Reglerwirkung), AppTest-Rauchtests (Standard, jedes Preset, jede Option, Permalink, Worst-Case am Regler-Maximum), Footer und Laufzeit-Importe. scipy und networkx gehören nur zu den Gegenproben in `tests/` (`requirements-dev.txt`); die App braucht nur Streamlit, NumPy und Plotly.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Primäransicht, Kernfragen, vier Experimente, Annahmen, Formeln |
| `mab_scaling.py` | der Abgleich: die vier Zeilen (`vier_zeilen`), der Stapelrechner mit Zählung und Abbruch, u-v- und Matrix-Form, Strafverfahren |
| `mab_feasibility.py` | Zulässigkeitsklassen (Hall-Bedingung), positive Diagonale und totaler Träger |
| `mab_hyman.py` | die Kalibrierschleife (Sekante auf β = 1/ε) als Stapel |
| `mab_scenario.py` | Zonen, alte Matrizen, Randsummen, Primäransicht (SplitMix64) |
| `mab_evaluation.py` | Kennzahlen und Reihen der vier Experimente, Primäransicht, Vorrechnung |
| `mab_verdikte.py` | die Verdikte als reine Funktionen |
| `mab_simplex.py` | Transportsimplex für OPT (Kopie aus der Demo zum Transportproblem, mit eigenem Präfix; Kopier-Test) |
| `mab_rng.py` | SplitMix64 (bytegleiche Kopie der Vorlage; Kopier-Test) |
| `mab_constants.py`, `mab_presets.py`, `mab_layout.py`, `mab_visualization.py`, `mab_vorgerechnet.py` | Konstanten und Presets, Permalink, fester Kopf, Plotly, vorgerechnete Reihen |
| `tools/messreihen.py`, `tools/messreihen.json` | alle Zahlen dieser README (Zufalls-Seed 2026, dazu die Bänder über die Zufalls-Seeds 2026 bis 2029), rund 3 Minuten |
| `tools/vorrechnen.py`, `tools/vorgerechnet.json` | die Gravitationszeile von Experiment 4 für die Standardeinstellungen |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy, networkx (Orakel) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handrechnungen, Orakel, Eigenschaften, Zulässigkeit, Szenario und Kopien, Verdikte, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe; Hilfsmodule `orakel_hilfe.py`, `readme_zeilen.py` |

## Bewusst nicht umgesetzt

- Keine Realdatenmatrix und keine Zonengrößen-Messung.
- Das Fratar-Verfahren und die Formel „aus dem Gedächtnis“ dazu sind nicht gerechnet (Originale nicht gelesen).
- Die Gesamtmasse beim Strafverfahren (Demo zu ungleichen Mengen) und der Unterlauf bei sehr kleinem ε (Demo zur Rechengenauigkeit des Sinkhorn-Verfahrens) sind nicht Teil dieser Demo; der Abgleich der App rechnet im K-Bereich und meldet NaN, wo eine Summe null wird.
- Keine Zahl der Durchläufe für dichte Basismatrizen als Experiment (die Zählung steht in den Experimenten 2 und 3).
- Kein PDF-Export, keine Zonenzahl über 12.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
