# SSM MATLAB vs Python in `Attractor/`

Stand: 7. April 2026  
Verglichene lokale Repos:
- `/home/hz/Data/Attractor/SSMtool/`
- `/home/hz/Data/Attractor/SSMLearn/`
- `/home/hz/Data/Attractor/SSMLearnPy/`
- `/home/hz/Data/Attractor/globalized-SSM/`

Diese Datei vergleicht nicht jede einzelne Funktion, sondern den **praktischen Workflow**:
Wie kommt man von Modell oder Daten zu einer SSM-basierten reduzierten Beschreibung, wie weit reicht die jeweilige Pipeline, und wo passen die Python-Repos in die MATLAB-Landschaft hinein.

---

## Kurzfazit

- `SSMtool` ist das **modellgetriebene MATLAB-Tool**: man startet mit bekannten Gleichungen bzw. FE-Matrizen und berechnet daraus lokale SSM-Taylorreihen analytisch.
- `SSMLearn` ist das **datengetriebene MATLAB-Tool**: man startet mit Trajektorien und identifiziert daraus lokale SSM-Geometrie und reduzierte Dynamik.
- `SSMLearnPy` ist die **Python-Portierung des datengetriebenen Gedankens**, aber der Workflow ist stärker als ML-Pipeline gebaut und derzeit schlanker als das MATLAB-Original.
- `globalized-SSM` ist **kein vierter Grundworkflow**, sondern ein **Aufsatz** auf lokale SSM-Modelle: es macht aus lokalen Taylor-/Polynomdarstellungen globalere rationale Approximationen.

---

## Die vier Rollen

| Repo | Sprache | Startpunkt | Primäre Rolle |
|---|---|---|---|
| `SSMtool` | MATLAB | bekannte Gleichungen / FE-Modell | lokale analytische SSM-Berechnung |
| `SSMLearn` | MATLAB | Trajektoriendaten | lokale datengetriebene SSM-Identifikation |
| `SSMLearnPy` | Python | Trajektoriendaten | Python-Workflow für datengetriebene ROMs auf SSMs |
| `globalized-SSM` | Python | bereits vorhandene lokale SSM-Koeffizienten oder Daten | Taylor/Polynom → rational/global |

---

## Workflow-Matrix

| Schritt | `SSMtool` (MATLAB) | `SSMLearn` (MATLAB) | `SSMLearnPy` (Python) | `globalized-SSM` (Python) |
|---|---|---|---|---|
| 1. Eingang | mechanisches Modell, Matrizen `M,C,K`, Nichtlinearität | Trajektorien `xData` / Beobachtungen | Listen von `t` und `x` oder Dateipfad | Taylor-Koeffizienten, `.mat`-Exports aus SSMTool oder direkte Daten |
| 2. Vorverarbeitung | modaler Spektral-Unterraum, Resonanzchecks | Spektralanalyse, Trunkierung, Delay-Embedding, evtl. Oblique Projection | optionales Delay-Embedding, dann direkte Datenpipeline | keine klassische Vorverarbeitung; arbeitet auf bereits vorhandener lokaler Darstellung oder auf Trainingsdaten |
| 3. Reduktion / Chart | parametrization method um Fixpunkt | `IMGeometry` konstruiert lokale SSM-Geometrie | `reduce_dimensions()` per `linearchart`/SVD oder simpler Projektion | keine SSM-Berechnung; übernimmt lokale Koordinaten/Polynome |
| 4. Parametrisierung | Taylor-Koeffizienten `W(z)` analytisch | lokale Parametrisierung aus Daten | Decoder via Ridge-Regression | Taylor → Padé oder rationale Funktion |
| 5. Reduzierte Dynamik | `R(z)` analytisch aus Kohomologie | `IMDynamicsFlow` / `Map` / `Mech` | Ridge-Fit auf Shift oder Ableitung; optional Normalform-Optimierung | rationalisierte reduzierte Dynamik oder rationaler Datenfit |
| 6. Validierung | Backbone, FRC, Resonanzanalyse | Rekonstruktion, Trajektorie-Fehler, Backbone, FRC | `predict_geometry`, `predict_reduced_dynamics`, `predict`, Fehlermaße | Vergleich Taylor vs Padé bzw. rationales Modell auf größerem Bereich |
| 7. Reichweite | lokal um Fixpunkt | lokal im Trainingsbereich | lokal bis mäßig extrapolativ | explizit zur **Globalisierung** lokaler Modelle |

---

## Die wichtigsten Unterschiede

### 1. `SSMtool` und `SSMLearn` lösen verschiedene Probleme

`SSMtool` beantwortet:  
"Ich kenne mein Systemmodell. Welche lokale SSM und welche reduzierte Dynamik folgen daraus?"

`SSMLearn` beantwortet:  
"Ich kenne das Systemmodell nicht oder will es nicht benutzen. Welche lokale SSM und reduzierte Dynamik kann ich direkt aus Mess- oder Simulationsdaten lernen?"

Das ist der größte Schnitt in der gesamten Landschaft. Alles andere ist zweitrangig.

### 2. MATLAB-`SSMLearn` ist geometrischer, `SSMLearnPy` ist pipeline-artiger

Im MATLAB-Repo ist der Workflow klar in getrennte mathematische Blöcke aufgeteilt:
- Embedding
- Geometrie-Fit der Mannigfaltigkeit
- Projektion auf reduzierte Koordinaten
- Fit der reduzierten Dynamik
- Postprocessing mit Backbone/FRC/Advect

Im Python-Repo ist derselbe Grundgedanke vorhanden, aber die Umsetzung ist pragmatischer:
- optionales Delay-Embedding
- lineare Reduktion per SVD oder Basisauswahl
- Decoder per Ridge-Regression
- Dynamik-Fit per Ridge-Regression
- optional nachgeschaltete Normalform-Optimierung

Das heißt praktisch:
- MATLAB-`SSMLearn` wirkt stärker wie ein spezialisierter SSM-Forschungsworkflow.
- `SSMLearnPy` wirkt stärker wie ein Python-ML-/Regression-Framework mit SSM-Zielsetzung.

### 3. `SSMLearnPy` deckt den Kern ab, aber nicht die volle MATLAB-Breite

Das MATLAB-Repo ist deutlich breiter:
- mehr Beispielsysteme
- mechaniknahe Spezialpfade
- eigene Varianten für `Flow`, `Map`, `Mech`
- stärkere Verzahnung mit `SSMTool` und COCO/FRC-Postprocessing
- nichtglatte und parametrierte Spezialfälle in Unterordnern

`SSMLearnPy` deckt den Kern ab:
- Delay-Embedding
- reduzierte Koordinaten
- Parametrisierung
- reduzierte Dynamik
- Normalform-Transformation
- Vorhersage und Fehlerauswertung

Aber es ist noch nicht dieselbe Arbeitsbank wie das MATLAB-Original.

### 4. `globalized-SSM` gehört hinter die lokale Identifikation

`globalized-SSM` macht nicht:
- keine Delay-Rekonstruktion,
- keine SSM-Auswahl aus Rohdaten,
- keine vollständige lokale SSM-Berechnung von Grund auf.

Es macht:
- lokale Taylor-/Polynomdarstellungen in **rationale** Darstellungen überführen,
- mit Padé-Approximationen oder rationaler Regression den nutzbaren Bereich vergrößern,
- SSMTool-MATLAB-Ausgaben per `matlab_integration.py` einlesen,
- alternativ datengetriebene rationale Fits bauen.

Methodisch ist das daher:

1. erst lokale SSM berechnen oder lernen,  
2. dann optional globalisieren.

---

## Praktische Zuordnung der Repos

### A. Wenn du ein mechanisches Modell hast

Nimm zuerst `SSMtool`.

Typischer Ablauf:
1. FE-/ODE-Modell formulieren.
2. Spektral-Unterraum wählen.
3. lokale SSM-Taylorreihe berechnen.
4. Backbone/FRC lokal analysieren.
5. Falls die lokale Taylor-Darstellung zu früh unbrauchbar wird: `globalized-SSM` als zweite Stufe benutzen.

### B. Wenn du nur Zeitreihen oder Messungen hast

Nimm zuerst `SSMLearn` oder `SSMLearnPy`.

Typischer Ablauf:
1. Trajektorien sammeln und zeitlich säubern.
2. optional delay-embedden.
3. reduzierte Koordinaten bestimmen.
4. Parametrisierung und reduzierte Dynamik fitten.
5. Rekonstruktion und Vorhersagefehler prüfen.
6. Bei Bedarf lokale Polynomdarstellung nachträglich mit `globalized-SSM` rationalisieren.

### C. Wenn du in Python bleiben willst

Dann ist die natürliche Kette:

1. `SSMLearnPy` für den lokalen datengetriebenen Fit.  
2. `globalized-SSM` für größere Reichweite der bereits gefundenen lokalen Karten/Dynamiken.

Das ist die sauberste Python-Entsprechung zur MATLAB-Welt, auch wenn sie noch nicht feature-identisch ist.

---

## Direkte Workflow-Äquivalenzen

| MATLAB-Idee | Python-Entsprechung |
|---|---|
| `coordinatesEmbedding` in `SSMLearn` | `ssmlearnpy.geometry.coordinates_embedding.coordinates_embedding` |
| reduzierte lineare Karte / Tangentialraum | `reduce_dimensions(method='linearchart')` |
| lokale Parametrisierung der Geometrie | `get_parametrization()` mit Ridge-Decoder |
| reduzierte Dynamik fitten | `get_reduced_dynamics()` |
| Normalform | `ssmlearnpy.reduced_dynamics.normalform` |
| Vorwärtsintegration / Rekonstruktion | `predict_*()` in `SSMLearnPy` |
| Taylor-Koeffizienten aus SSMTool weiterverwenden | `globalized-SSM/taylor_to_pade/matlab_integration.py` |
| Backbone/Frequenz/Dämpfung aus lokaler Darstellung ableiten | `globalized-SSM/taylor_to_pade/utils.py` |

Wichtig:
Die Python-Entsprechung ist nicht überall mathematisch identisch, sondern oft funktional ähnlich.

---

## Wo Python aktuell noch schwächer ist

- Kein vollständiges Python-Pendant zu `SSMtool` als analytischem Gleichungs-Solver für mechanische Systeme.
- `SSMLearnPy` ist weniger breit dokumentiert als die MATLAB-Readmes mit Schritt-für-Schritt-Beispielen.
- Viele Python-Beispiele liegen als Notebooks vor; das ist explorativ gut, aber reproduktionstechnisch weniger streng als reine Skriptpipelines.
- Die MATLAB-Seite hat mehr ausgebautes Postprocessing rund um FRC, Backbone, Spezialfälle und mechanische Sonderstrukturen.

---

## Wo Python bereits klar im Vorteil ist

- einfache Installation mit `pip install -e .`
- leicht kombinierbar mit `numpy`, `scipy`, `scikit-learn`, `sympy`
- sauberer Übergang von lokalem Datenfit (`SSMLearnPy`) zu rationaler Globalisierung (`globalized-SSM`)
- besser integrierbar in moderne Analyse- und Experimentpipelines

---

## Sinnvolle Gesamtarchitektur für `Attractor/`

Für deine Sammlung in `Attractor/` ergibt sich eine klare Viererlogik:

1. `SSMtool/`  
   analytischer MATLAB-Ausgangspunkt, wenn das Modell bekannt ist.

2. `SSMLearn/`  
   datengetriebener MATLAB-Referenzworkflow mit der größten methodischen Breite.

3. `SSMLearnPy/`  
   Python-Implementierung des lokalen datengetriebenen Kerns.

4. `globalized-SSM/`  
   Python-Erweiterung, um lokale SSM-Approximationen rational zu globalisieren.

Kurz gesagt:
- `SSMtool` = lokal, modellgetrieben
- `SSMLearn` = lokal, datengetrieben
- `SSMLearnPy` = Python-Version des lokalen datengetriebenen Wegs
- `globalized-SSM` = Globalisierungsschicht für lokale Modelle

---

## Urteil in einem Satz

Die Python-Seite ersetzt die MATLAB-Seite derzeit **nicht 1:1**, aber sie bildet bereits eine sinnvolle Kombination:
`SSMLearnPy` für lokale datengetriebene SSM-Modelle und `globalized-SSM` für deren rationale Erweiterung über den rein lokalen Taylor-/Polynom-Bereich hinaus.
