> Zusammengeführt aus SSMLearnHaller.md + SSMLearnHaller_new.md (Stand 11.07.2026)

# SSMLearn — Datengetriebene SSM-Identifikation (Haller / Cenedese / Axås): Workflow, Mathematik, lokale Einordnung

Quellen:
- MATLAB-Repo: `/home/hz/Data/Attractor/SSMLearn/` (geklont von `haller-group/SSMLearn`)
- Python-Implementierung: `/home/hz/Data/Attractor/SSMLearnPy/` (Cenedese, `SSMLearnPy`)
- Modellbasierte Gegenfamilie: `/home/hz/Data/Attractor/SSMtool/` (V1.0) und `/home/hz/Data/Attractor/SSMTool_jain/` (SSMTool 2.6) — Doku: `/home/hz/Data/Attractor/SSMToolHaller_merged.md`
- Globalisierung lokaler SSM-Darstellungen: `/home/hz/Data/Attractor/globalized-SSM/`

Zentrale Referenzen (PDFs im Verzeichnis `/home/hz/Data/Attractor/`):
- `Haller_Ponsioen2016_NNM_SSM_arxiv.pdf` — Existenzsatz für SSMs.
- Cenedese, Axås, Bäuerlein, Avila, Haller, *Nat. Commun.* 13 (2022) 872 — SSMLearn-Paper.
- Axås, Cenedese, Haller, *Nonlinear Dyn.* 111 (2023) 7941 — fastSSM.
- Axås, Haller, *Nonlinear Dyn.* 111 (2023) 22079 — Delay-Embedded SSMs.
- Liu, Axås, Haller, *Chaos* 34 (2024) 033140 — Inertial Manifolds als SSMs (chaotische Dynamik).
- Kaszás, Haller, *Globalizing manifold-based reduced models for equations and data* (submitted, 2025) — Rational/Padé-Fortsetzung lokaler SSM-Darstellungen über die Taylor-Umgebung hinaus.

---

## Inhaltsverzeichnis

**Teil I — Grundlagen und Einordnung**
1. Was ist eine Spectral Submanifold (SSM)
2. Repo-Landschaft und Abgrenzung (SSMLearn / SSMLearnPy / SSMtool / SSMTool 2.6 / globalized-SSM)

**Teil II — SSMLearn (MATLAB-Referenz)**
3. Datenformat und Annahmen
4. Pipeline-Übersicht
5. Datei-Inventur unter `src/`
6. End-to-End-Workflow als Schrittliste
7. Zentrale Mathematik
8. Outputs
9. Beispiele
10. Kritische Limitationen

**Teil III — Python und Anwendung**
11. Python: SSMLearnPy und Portierungs-Referenz
12. Anwendbarkeit auf BTC-log-Residuen
13. Quellen-Index (Datei → Zweck)

**Anhang**
- A. Aufgelöste Widersprüche

---

# Teil I — Grundlagen und Einordnung

## 1. Was ist eine Spectral Submanifold (SSM)?

### 1.1 Definition

Sei
$$\dot x = A x + f(x), \qquad x \in \mathbb{R}^N, \quad f(x)=\mathcal{O}(|x|^2),$$
ein autonomes glattes dynamisches System mit Fixpunkt $x=0$. Die Linearisierung $A$ habe Eigenwerte $\{\lambda_j\}_{j=1}^{N}$. Wähle eine Modale Untermenge $E \subset \{1,\dots,N\}$ mit Dimension $d = |E|$ (in `getSSM.m`: `S.choose_E(E)`). $E$ entspricht einem $A$-invarianten Spektral-Unterraum mit Eigenwerten $\{\lambda_e\}_{e\in E}$. Die übrigen Eigenwerte heißen "outer" $\{\lambda_o\}_{o\notin E}$.

Eine **Spectral Submanifold (SSM)** $\mathcal{W}(E)$ ist eine $d$-dimensionale, $C^\infty$-glatte, $f$-invariante Untermannigfaltigkeit von $\mathbb{R}^N$, die im Ursprung zu dem Eigenraum $\mathrm{span}\{v_e\}_{e\in E}$ tangential liegt und die gleiche Glattheitsklasse wie das Vektorfeld besitzt.

### 1.2 Existenz- und Eindeutigkeits-Theorem (Cabré, Fontich, de la Llave 2003; Haller, Ponsioen 2016)

Eine SSM existiert eindeutig glatt (bis auf endliche Differenzierbarkeitsordnung), falls:

1. **Hyperbolische Stabilität + strikter Spektralgap:**
   $$\max_{e \in E} \mathrm{Re}\,\lambda_e < 0, \qquad \max_{o\notin E} \mathrm{Re}\,\lambda_o \;<\; \min_{e\in E}\mathrm{Re}\,\lambda_e$$
   d.h. alle Eigenwerte sind stabil und **alle** Slaved-Moden sind strikt schneller (Realteil weiter links) als **alle** Master-Moden. Das ist die starke Form des Spektralgaps, die das Cabré-Fontich-de la Llave-Theorem für die Existenz und Eindeutigkeit der slow stable SSM verlangt.

2. **Spektral-Quotient (innerer Faktor):**
   Setze
   $$\sigma(E) := \mathrm{Int}\!\left(\frac{\min_{o\notin E}\mathrm{Re}\,\lambda_o}{\max_{e\in E}\mathrm{Re}\,\lambda_e}\right).$$
   Die SSM ist als $C^k$-Mannigfaltigkeit mit $k\le \sigma(E)$ eindeutig.

3. **Non-Resonanz-Bedingung (algebraisch, Ordnung für Ordnung):**
   $$\lambda_o \;\neq\; \sum_{e\in E} m_e\,\lambda_e \qquad \forall\,o\notin E,\;\forall\,m\in \mathbb{N}^d \text{ mit } 2 \le \sum_e m_e \le \sigma(E).$$
   Verletzungen dieser Bedingung erzeugen "small denominators" in der Cohomological Equation (siehe §7) und damit Singularitäten in der Parametrisierung. SSMLearn detektiert diese in `IMDynamicsFlow.m` über `tol_nf` (siehe §5.4.1 und §10.9).

---

## 2. Repo-Landschaft und Abgrenzung

### 2.1 Übersicht (Pfade auf Platte verifiziert)

| Familie | Lokal |
|---|---|
| klassische modellbasierte SSM-Toolbox (V1.0) | `/home/hz/Data/Attractor/SSMtool/` |
| neue modellbasierte SSM-Toolbox (SSMTool 2.6) | `/home/hz/Data/Attractor/SSMTool_jain/` |
| datengetriebene SSM-Familie (MATLAB-Referenz) | `/home/hz/Data/Attractor/SSMLearn/` |
| datengetriebene SSM-Familie (Python) | `/home/hz/Data/Attractor/SSMLearnPy/` |
| Globalisierung lokaler SSM-Darstellungen (Padé) | `/home/hz/Data/Attractor/globalized-SSM/` |
| BTC-spezifischer Datenworkflow | `/home/hz/Data/Attractor/analyze_residuals/`, später `analyze_n_ens/` |

Es gibt **kein "neues SSMLearn-Repo"**: `SSMTool_jain` ist trotz neuer Fähigkeiten kein SSMLearn-Nachfolger (Begründung in §2.3). Die datengetriebene Referenz bleibt SSMLearn (MATLAB) bzw. SSMLearnPy (Python); diese Datei bleibt die Grundreferenz für datengetriebene Workflows, `SSMToolHaller_merged.md` die Modellseite.

### 2.2 SSMLearn vs. SSMtool (datengetrieben vs. modellbasiert)

- **SSMtool / SSMTool 2.6** (Jain, Haller et al.) löst die Invarianz-Gleichung **analytisch** aus den **bekannten** Tensoren $M, C, K, f_{nl}$ eines mechanischen Systems. Es ist ein Forward-Modell.
- **SSMLearn** rekonstruiert die SSM-Parametrisierung und die reduzierte Dynamik **rein aus Trajektorien-Daten** — die zugrundeliegenden Gleichungen müssen weder bekannt sein noch als analytisches Modell existieren. Es ist ein Inverse-Problem-Solver. Für die Identifikation reicht: Trajektorien (cell array), SSM-Dimension $d$, Polynom-Ordnungen $M$ (für die Parametrisierung) und $M_R$ (für die reduzierte Dynamik).
- Für Vergleichszwecke ruft SSMLearn aber SSMtool auf, falls verfügbar — siehe `src/utils/getSSM.m:24-30`, `getSSMIC.m`, `SSMToolFRC.m`.

### 2.3 Warum SSMTool 2.6 (`SSMTool_jain`) kein SSMLearn-Nachfolger ist

`SSMTool_jain` bleibt architektonisch fundamental anders als SSMLearn:
- es baut ein `DynamicalSystem`-Objekt auf einem **bekannten Modell** auf,
- es startet von der Spektralanalyse der Linearisierung,
- es löst cohomologische Gleichungen auf Modellniveau,
- es arbeitet mit der Parametrization Method, nicht mit Manifold Learning auf Daten.

Zentrale Klassen:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@DynamicalSystem/DynamicalSystem.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/Manifold.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/SSM.m`

**"non-intrusive" ≠ "data-driven":** Das 2.6-README spricht von "data-free non-intrusive model reduction". Gemeint ist: man gibt keine expliziten Polynomtensoren vor, hat aber weiterhin ein **modellbasiertes** System bzw. eine auswertbare Nichtlinearität/FE-Blackbox, aus der die SSM-Koeffizienten modellbasiert berechnet werden. Belege:
- `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/private/fnl_nonIntrusive.m`
- `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/private/dfnl_nonIntrusive.m`

Das ist **nicht** der SSMLearn-Weg (Daten → PCA/Embedding → Decoder/Encoder → Regression).

### 2.4 Aus `SSMTool_jain` übernehmbare Ideen für SSMLearn-artige Workflows

Keine direkte Pipeline-Übernahme, aber methodisch nützlich:

1. **Vorprüfungen:** Spektrum prüfen, interne Resonanzen prüfen, Invarianzfehler / Gültigkeitsbereich mitdenken. Hilfreiche Funktionen:
   - `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/compute_auto_invariance_error.m`
   - `/home/hz/Data/Attractor/SSMTool_jain/src/@Manifold/compute_analyticity_domain.m`
   - `/home/hz/Data/Attractor/SSMTool_jain/src/@SSM/private/detect_resonant_modes.m`
2. **Resonanzorientierter Workflow** (für die `harmonic*`-Arbeit relevant): Resonanz nicht nur visuell behaupten, Mastermoden explizit auswählen, interne Resonanzbeziehungen systematisch prüfen.
3. **Saubere Trennung autonom / nichtautonom:** autonomer Teil `compute_whisker`, nichtautonomer Teil `compute_perturbed_whisker`. Für BTC-/Residuenarbeit heißt das nicht "nutze SSMTool direkt", sondern: Workflow sauberer strukturieren — freie Dynamik, erzwungene Beiträge und Beobachtungsmodell klar trennen.

### 2.5 `globalized-SSM` als nachgelagerter Layer

- `globalized-SSM` ist **kein Ersatz** für SSMLearn oder SSMtool, sondern ein **nachgelagerter Approximation-Layer**.
- Input-seitig setzt es typischerweise eine schon vorhandene **lokale** SSM-Darstellung voraus: entweder Taylor-Koeffizienten aus `SSMtool` oder datengetriebene lokale Fits aus `SSMLearn`.
- Kernidee: die nur lokal gültige polynomiale Darstellung durch **rationale Approximationen** erweitern, konkret über **Padé-Approximationen** und **rationale Regression**.
- Praktisch: SSMLearn liefert die lokale Geometrie und reduzierte Dynamik aus Daten; `globalized-SSM` kann diese Darstellung danach oft auf einen größeren Zustandsbereich extrapolierbar machen, insbesondere wenn reine Taylor-Polynome außerhalb der Fixpunkt-Umgebung schnell instabil oder ungenau werden.
- Für eine Python-Implementierung heißt das: Pipeline sauber in zwei Schichten aufteilen — (1) lokale SSM-Identifikation, (2) optionale Globalisierung der bereits identifizierten lokalen Karten und reduzierten Dynamiken.

---

# Teil II — SSMLearn (MATLAB-Referenz)

Diese Referenz folgt der echten Code-Struktur, nicht der Marketing-Sprache des README. Originalkommentare aus dem Code werden wörtlich als Blockzitate übernommen.

## 3. Datenformat und Annahmen (Inputs)

### 3.1 Pflicht-Format `xData` / `yData`

Cell-Array der Dimension `{nTraj, 2}`:
- Spalte 1: Zeitvektor `1 x mi` pro Trajektorie.
- Spalte 2: Zustand-/Beobachter-Matrix `n x mi` pro Trajektorie.

Sample-Rate wird als konstant angenommen:

> "Sampling time is assumed to be constant"
> — `src/reduceddynamics/IMDynamicsMap.m:41-42`

Das gilt auch für `IMDynamicsFlow.m:43-44`. Es existiert kein Code, der ungleichmäßiges Sampling rebinned — der User muss vor SSMLearn auf ein Gitter resamplen.

### 3.2 Annahmen

- **Autonomie**: das zugrundeliegende System ist ungezwungen oder der zwingende Anteil wird separat behandelt (`forcedSSMROM.m`).
- **Fixpunkt im Ursprung**: hardgecodet in `IMGeometryGraphT0.m`:

  > "Construct the parametrization for an invariant manifold related to a fixed point (assumed to be the origin) as a graph the coordinates y = V_e'*x"
  > — `src/geometry/IMGeometryGraphT0.m:2-5`

  Daten müssen vorher zentriert werden falls der Fixpunkt $\neq 0$ ist. Bei nicht-trivialen Fixpunkten siehe `IMGeometryParaCon.m:11-19` (parameterabhängige Variante).

- **Beobachterraum** kann der volle Phasenraum oder ein Vektor generischer Observable sein.

### 3.3 Mehrere Trajektorien

`nTraj > 1` ist normaler Fall. Keine Anforderung an Anzahl. Empirische Beispiele: 1–8 Trajektorien (Couette: 1 Trainings + 7 Test; Sloshing: 3; Brake-Reuss: 1).

---

## 4. Pipeline-Übersicht (drei Hauptstufen)

Aus dem README zitiert (`README.md:12-16`):

> "The computational steps for achieving a reduced-order model are:
>  1. Embedding of the measurements in a suitable observable space;
>  2. Computation of the invariant manifold parametrization and its reduced order coordinates;
>  3. Identification of the reduced dynamics and its normal form."

In Code-Funktionen ausgedrückt:

| Stufe | Hauptfunktion | Datei |
|-------|---------------|-------|
| 1. Observable-Embedding | `coordinatesEmbedding` / `embedCoordinates` | `src/utils/coordinatesEmbedding.m`, `embedCoordinates.m` |
| 2. SSM-Parametrisierung | `IMGeometry` → `IMGeometryGraphT0` | `src/geometry/IMGeometry.m`, `IMGeometryGraphT0.m` |
| 3. Reduzierte Dynamik + Normal Form | `IMDynamicsFlow` (kontinuierlich) / `IMDynamicsMap` (diskret) / `IMDynamicsMech` (mechanisch) → `dynamicsCoordChangeNF` | `src/reduceddynamics/*.m` |
| 4. Post-Processing (BBC, FRC) | `backboneCurves`, `computeFRC`, `analyticalFRC`, `forcedSSMROM` | `src/postprocessing/*.m`, `src/timedependentmanifold/*.m` |

Außerdem ist `fastSSM/fastSSM.m` eine eigenständige, minimale Re-Implementation des kompletten Stacks in 122 Zeilen, ausschließlich für 2D-SSMs — ausführlich in §11.4 (beste Vorlage für eine Python-Portierung des kohomologischen Kerns).

---

## 5. Vollständige Datei-Inventur unter `src/`

Jede relevante `.m`-Datei mit Signatur, Zweck und (wenn vorhanden) Original-Header-Zitat; Reihenfolge nach Pipeline-Logik. (`SSMLearn/src/` enthält insgesamt 104 `.m`-Dateien mit ~136 `function`-Definitionen, viele davon triviale Helfer.)

### 5.1 `src/preprocessing/`

#### 5.1.1 `showSpectrogram.m` (41 Zeilen)
**Signatur:** `[powerdensity, frequencies, times] = showSpectrogram(xData, varargin)`

> "Plot a spectrogram for a scalar signal xData{1,2} at times xData{1,1}"
> — `src/preprocessing/showSpectrogram.m:2-3`

Berechnet das Short-Time-Fourier-Spektrogramm pro Trajektorie und summiert die Power über alle Trajektorien (Zeile 16-22). Fenstergröße: `Nwin = round(length(t)/10)`, Überlapp 50%. Sample-Rate aus `2*pi/(t(2)-t(1))`. Output dient nachfolgend zur Bestimmung des "linear regime" und der SSM-Dimension.

#### 5.1.2 `analyzeSpectr.m` (28 Zeilen)
**Signatur:** `dominantFreqs_tot = analyzeSpectr(times, frequencies, powerdensity, epsilon)`

Identifiziert dominante Frequenzpeaks per `findpeaks` mit `'MinPeakDistance',10` (Zeile 9). Schwelle wird beim ersten Zeitschritt gesetzt: `minimum_pks = epsilon * max(pks)` (Zeile 12). Nur Peaks $\ge$ Schwelle gelten als dominant. Wird in `SSM_startTime` benutzt um den Beginn der "single-mode"-Phase zu detektieren.

#### 5.1.3 `SSM_startTime.m` (34 Zeilen)
**Signatur:** `[startTime, indStartTime, SSMDim] = SSM_startTime(data, indplot)`

Heuristik zur automatischen Bestimmung von Start-Zeit (ab der die Daten "auf der SSM" liegen) und empirischer SSM-Dimension. Logik (Zeile 8-20):
```
für jeden Zeitschritt:
    n = Anzahl dominanter Frequenzen
    SSMDim(t) = 2  falls n in {0,1}    (1 Mode → 2D komplex-konjugiert)
    SSMDim(t) = 2*n  sonst
startTime = erster Zeitpunkt mit n == 1
SSMDim_endgültig = SSMDim(end)
```

Das ist die einzige automatische SSM-Dimension-Schätzung im Repo. Praktisch gibt der User die Dimension meist selbst vor.

#### 5.1.4 `DFT.m` (27 Zeilen)
**Signatur:** `[amp,ph,freq] = DFT(X,dt)`

Standard-DFT einer Time-Series-Matrix `n.series x n.time_inst`. Behandelt gerade/ungerade $N$ getrennt (Zeilen 13-24). Output ist die einseitige Amplitude / Phase / Frequenzachse. Helfer für Spektralanalysen.

#### 5.1.5 `DMD.m` (49 Zeilen)
**Signatur:** `[Phi, omega, lambda, b, Xdmd, t] = DMD(X1, X2, r, dt)`

> "Computes the Dynamic Mode Decomposition of X1, X2 — INPUTS: X1 = X data matrix, X2 = X' shifted data matrix"
> — `src/preprocessing/DMD.m:1-8`

Standard-DMD (Tu, Rowley, Kutz). SVD von $X_1$ auf Rang $r$, dann
$$\tilde A = U_r^\top X_2 V_r S_r^{-1}, \qquad \mathrm{eig}(\tilde A) \to \text{DMD-Modes}.$$
DMD wird in SSMLearn als optionale Linearisierungs-Kontrolle genutzt, ist aber **nicht** Teil der SSM-Parametrisierung.

#### 5.1.6 `obliqueProjection.m` (116 Zeilen)
**Signatur:** `[data_projected, P_min, V_trunc_slow, data_non_projected, data_non_projected_trunc] = obliqueProjection(data, index_proj, SSMDim, overEmbed, ShiftStep, varargin)`

Implementiert die "oblique projection" aus Cenedese et al. 2022:

> "computation of the linear oblique projection via minimization of the oscillations of the backbone curve"
> — `src/preprocessing/obliqueProjection.m:27-28`

Schritte:
1. Daten werden auf den Index-Bereich `[index_proj, index_end]` getrunkiert (Zeile 9-25).
2. Delay-Embedding über `coordinatesEmbedding` (Zeile 30).
3. SVD von `X = [snapshots]` auf Rang `2*SSMDim` (Zeile 42-44).
4. DMD-ähnliche $\tilde S = U^\top Y V S^{-1}$ und Eigen-Sortierung (Zeile 46-48).
5. Falls `oblique_projection == true`: nichtlineare Optimierung von $P$ um die Backbone-Oszillationen zu minimieren (Zeile 58 ff.).

#### 5.1.7 `regimeLinear.m` (48 Zeilen)
**Signatur:** `index_time_linear = regimeLinear(data, lim)`

Findet pro Trajektorie den ersten Index ab dem die Frequenz aus PFF (`PFFk`) "linear" wird, also $|\Delta f|<$ `lim`. Gibt den Mittelwert über alle Trajektorien zurück. Wird benutzt um die Trainingsphase auf den nichtlinearen Regime-Bereich zu begrenzen.

#### 5.1.8 `PFF.m` und `PFFk.m` (59 / 85 Zeilen)
**Signatur:** `[amp,freq,damp,time] = PFF(t, x, varargin)`

> "Implementation of the PFF algorithm for extraction of instantaneous damping and frequency of the time signal x known at times t"
> — `src/preprocessing/PFF.m:1-8`

> "See the paper below for more: M. Jin, W. Chen, M. R. W. Brake, and H. Song. Identification of instantaneous frequency and damping from transient decay data. Journal of Vibration and Acoustics, 142(5):051111."
> — `src/preprocessing/PFF.m:10-13`

Algorithmus:
1. Optionales Bandpassfiltern (Zeile 16-19).
2. Null-Crossings finden via Vorzeichenwechsel von $x_k x_{k+1}$ (Zeile 25-29). Lineare Interpolation auf Null.
3. Frequenz: $f = 1/(2\,\Delta t_{zero})$ (Zeile 30).
4. Lokale Amplituden-Maxima per `findpeaks` und parabolische Anpassung an je 3 Punkte (Zeile 39-48).
5. Damping: $\zeta = (\Delta A/\Delta t)/A$ (Zeile 49).
6. Glättung über `movmean(.,4)` (Zeile 53-55).

`PFFk.m` ist die Variante mit benutzerdefinierter Glättungsfensterbreite `kmean`.

### 5.2 `src/utils/coordinatesEmbedding.m` (110 Zeilen) — Phasenraum-Rekonstruktion

**Signatur:** `[yData, optsEmbd] = coordinatesEmbedding(xData, SSMDim, varargin)`

> "Returns the n-dim. time series x into a time series of properly embedded coordinate system y of dimension p."
> — `src/utils/coordinatesEmbedding.m:4-6`

#### Pflicht-Inputs
- `xData`: cell `{nTraj, 2}` (Zeit + Zustand)
- `SSMDim`: SSM-Dimension $d$.

#### Optionen
| Name | Default | Beschreibung (Code-Zitat) |
|------|---------|-----|
| `OverEmbedding` | 0 | "augment the minimal embedding dimension with a number of time delayed measurements" — Z. 15-16 |
| `ForceEmbedding` | false | "force the embedding in the states of x" — Z. 17 |
| `TimeStepping` | 1 | "time stepping in the time series" — Z. 18 |
| `ShiftSteps` | 1 | "number of timesteps passed between components (but subsequent measurements are kept intact)" — Z. 19-20 |

#### Algorithmus (Zeilen 47-95)

1. Bestimme Beobachter-Anzahl `N` aus `xData{1,2}` (Zeile 49-50).
2. Berechne minimale Embedding-Dimension nach **Takens' Theorem** (Zeile 53):
   ```matlab
   n_N = ceil((2*SSMDim+1)/N) + OverEmbedding
   ```
   Das ist Whitneys/Takens Mindest-Einbettung: $p = (2d+1)$ Observable reichen generisch.
3. Falls `n_N > 1` und nicht `ForceEmbedding`: stacke `n_N` zeit-verschobene Kopien:
   ```
   p = n_N * N
   Y_j = [x(t); x(t+ShiftSteps*dt); x(t+2*ShiftSteps*dt); ...]
   ```
   Code Zeile 76-79.
4. Sonst (Zeile 84-95): identische Übernahme.
5. Print: "The p embedding coordinates consist of the measured states and their (n_N-1) time-delayed measurements" (Zeile 64-67).

#### Wahl von $\tau$
SSMLearn wählt $\tau = $ `ShiftSteps` $\cdot \mathrm{dt}$ standardmäßig $= \mathrm{dt}$. **Es gibt keine automatische AMI/Cao-basierte $\tau$-Schätzung im Repo.** Der User bestimmt das in der Praxis durch Cross-Validation oder Reconstruction-Error empirisch:

> "When using delay embedding, the number of time delays and the delay time might need to be adjusted. Increasing the number of time delays practically acts as a filter and may lead to more accurate models. These parameters should be chosen to minimize the reconstruction error, or based on the explicit formulas for the delay-embedded tangent space, as in Axås, Haller, *Nonlinear Dyn* 111, 22079 (2023)."
> — `docs/Practical_considerations.pdf` Seite 1, Punkt 2

#### `embedCoordinates.m` (38 Zeilen)
Einfachere alternative Funktion mit expliziten `embeddingDimension`, `delaySteps`. Macht dasselbe ohne den Takens-Default.

#### `delayTangentSpace.m` (16 Zeilen)
**Signatur:** `V = delayTangentSpace(tau, p, lambda)`

> "Compute the eigenvectors of the linear part of a delay embedded system given the eigenvalues lambda_j."
> — `src/geometry/utilsGraphT0/delayTangentSpace.m:3-4`

Implementiert die analytische Formel für Eigenvektoren im Delay-embedded-Raum:
$$V_{kj} = e^{k\tau\lambda_j}, \qquad k=0,\dots,p-1.$$
Real- und Imaginärteil werden bei komplex-konjugierten $\lambda$ getrennt (Z. 12-16). Wird genutzt um die Tangentialraum-Initialisierung in `IMGeometryGraphT0` zu beschleunigen, wenn die Eigenwerte vorab bekannt sind.

### 5.3 `src/geometry/` — SSM-Parametrisierung

#### 5.3.1 `IMGeometry.m` (127 Zeilen) — Wrapper

**Signatur:** `[IMInfo, IMChart, IMParam] = IMGeometry(yData, SSMDim, M, varargin)`

> "Description of the k-dim. invariant manifold geometry, in terms of coordinate chart and the parametrization. The default or natural method assumes that the manifold can be seen as a graph of a function for coordinates being the projection to the tangent space at the origin."
> — `src/geometry/IMGeometry.m:3-9`

Routet auf zwei Methoden via `optsGeomtery.style`:
- `'natural'` → ruft `IMGeometryGraphT0` (siehe §5.3.2). Chart $= V^\top$, Parametrisierung als Graph.
- `'custom'` → User-definierte Karte und reduzierte Koordinaten; Polynom-Regression über `multivariatePolynomial(SSMDim, 1, M)` (Zeile 100-112).

Optionen `IMGeometry`:

| Name | Default | Bedeutung |
|------|---------|-----------|
| `style` | 'natural' | 'natural' oder 'custom' |
| `chart` | [] | benutzerdef. Chart-Funktion (forciert custom) |
| `reducedCoordinates` | [] | benutzerdef. reduzierte Koords (forciert custom) |
| `l` | 0 | "coefficients regularization" — Z. 31 |
| `c1`, `c2` | 0, 0 | "error coefficient for slow manifolds weighting (1+c1*exp(-c2*t)).^(-1)" — Z. 32-35 |
| `t` | 1 | Zeit-Vektor; auto wenn cell |
| `Ve` | [] | bekannter Tangentialraum |
| `outdof` | [] | DoFs für Output-Erhaltung |

Die Gewichtung $L(t) = (1+c_1 e^{-c_2 t})^{-1}$ (Zeile 87) ist eine **zeitabhängige Reweighting** für "slow-manifold detection": frühe (transiente) Datenpunkte bekommen geringeres Gewicht weil sie noch nicht auf der SSM liegen.

#### 5.3.2 `IMGeometryGraphT0.m` (208 Zeilen) — Kern

**Signatur:** `[V_e, IMParam, IM_param_info] = IMGeometryGraphT0(X, k, M, varargin)`

> "Construct the parametrization for an invariant manifold related to a fixed point (assumed to be the origin) as a graph the coordinates y = V_e'*x:
>
>                x = IM_para(y) = V_e*y + H phi(y)
>
> V_e is an orthonormal representation of the eigenspace to which the manifold is tangent at the origin. Moreover, phi(y) is a k-variate polynomial mapping from degree 2 to degree M and its coefficients H are orthogonal to V_e, i.e., V_e'*H = 0."
> — `src/geometry/IMGeometryGraphT0.m:2-12`

##### Mathematisches Modell

Die SSM wird als **Graph über dem Tangentialraum** parametrisiert:
$$\boxed{\;x = W(y) := V_e\,y + H\,\phi(y), \qquad y = V_e^\top x \in \mathbb{R}^d\;}$$
wobei
- $V_e \in \mathbb{R}^{n\times d}$ orthonormal: $V_e^\top V_e = I_d$.
- $\phi(y)$ Vektor aller multivariaten Monome in $d$ Variablen vom Grad $2$ bis $M$. Anzahl: $\sum_{r=2}^{M}\binom{d+r-1}{r}$. Berechnet von `multivariatePolynomial(d, 2, M)` (Zeile 107, 134).
- $H \in \mathbb{R}^{n\times \dim\phi}$ unbekannte nichtlineare Koeffizienten.
- **Constraint $V_e^\top H = 0$**: stellt sicher dass die nichtlinearen Korrekturen orthogonal zur tangentialen Richtung liegen, sodass $V_e^\top W(y) = y$ gilt (Graph-Form ist konsistent).

##### Kostenfunktion

Frobenius-Rekonstruktionsfehler mit Zeitgewichtung $L_{ti} = (1+c_1 e^{-c_2 t_i})^{-1}$ und Ridge-Term:
$$\mathcal{J}(V_e, H) = \frac{1}{nN}\sum_{i=1}^{N}\big\| (x_i - V_e y_i - H\phi(y_i))\,L_{ti}\big\|_2^2 + \frac{1}{nN}\sum_{j} \lambda_j H_{:,j}^\top H_{:,j}$$
wobei $y_i = V_e^\top x_i$ (Datenprojektion).

Die Implementation in `fMinimize.m`:
```matlab
Y = transpose(V)*X;
Phi = phi(Y);
X_rec = V*Y + H*Phi;
Err = X - X_rec;
f = (sum(sum((Err.*L).^2)) + sum((H(:).^2).*LI(:))) / size(X,1)/size(X,2);
```
— `src/geometry/utilsGraphT0/fMinimize.m:11-16`

##### Constraints

Aus `defineNonlinConstraints.m`:
```matlab
Cv = transpose(V)*V - eye(k,k);   % V'V - I  (orthonormal)
Ch = transpose(V)*H;              % V'H = 0  (Graph-Bedingung)
```
— `src/geometry/utilsGraphT0/defineNonlinConstraints.m:12`

Die Constraints werden vektorisiert und an `fmincon` als Equality-Constraint gegeben (`extrasNonlinearConstraints.m` baut die Indizes für die Sparse-Jacobian auf).

##### Drei Lösungs-Modi (Zeilen 96-186)

1. **`V_e` ist a priori bekannt** (z.B. aus Linearisierung oder DMD), `M < 2`: nur lineare Approximation, $H = 0$. Zeile 103-104.

2. **`V_e` ist bekannt**, $M \ge 2$: Constrained Ridge Regression in geschlossener Form. Wenn `V_e` aus Identitäts-Indizes besteht (Z. 110): einfache Regression. Ansonsten:
   ```matlab
   H = constrainedRidgeRegression(transpose(Phi), transpose(X-V_e*Q), transpose(Phi.*L), l, V_e)
   ```
   Z. 117-119. Siehe `constrainedRidgeRegression.m`:

   > "Solves the least square fit for the model Y = X*H + e. Each error is penalized according the vector L and the weights are penalized according with a coefficient lambda. Additionally, the weights satisfy the relation H*V = 0 (equality constraint)."
   > — `src/geometry/utilsGraphT0/constrainedRidgeRegression.m:1-10`

   Lagrange-Methode: linearisiert das Constraint und löst das blocked-KKT-System
   $$\begin{pmatrix} X^\top L X + \lambda I & A_w \\ A_w^\top & 0 \end{pmatrix}\begin{pmatrix} \mathrm{vec}(H) \\ \mu \end{pmatrix} = \begin{pmatrix} X^\top L Y \\ 0 \end{pmatrix}$$
   wobei $A_w$ die Constraint-Matrix für $H V_e = 0$ ist. Implementiert sparse (Z. 25-30).

3. **`V_e` ist unbekannt**, $M \ge 2$: nichtlineare constrained Optimierung über `fmincon`. Initialisierung:
   - $V_{e,0}$ = gewichtete PCA von $X$ (`svds(transpose(XL), k)`, Z. 140) — falls nicht user-supplied.
   - $H_0$ aus `constrainedRidgeRegression(.., V_{e,0})` (Z. 151-153).
   - Optimierungsvektor: $z = [\mathrm{vec}(V_e); \mathrm{vec}(H)]$ (Z. 157).

   Lineare Constraints aus `alignmentLinearConstraint.m`:

   > "Linear equality and inequality constraints that ensure an alignment that makes the optimization routine to have a unique solution: given a reference representation V_ref, we aim to find V such that is 'aligned' to V_ref"
   > — `src/geometry/utilsGraphT0/alignmentLinearConstraint.m:3-6`

   Konkret: $\mathrm{diag}(V_{ref}^\top V) > 0$, $V_{ref}^\top V$ unter-diagonal $= 0$. Das entfernt die SO($d$)-Eichfreiheit der orthogonalen Basis.

   Dann `fmincon` mit user-supplied Gradient (`fMinimize.m` liefert $\nabla \mathcal{J}$, Z. 17-32) und Constraint-Gradient (`defineNonlinConstraints.m`, Z. 14-22). Default fmincon-Optionen (Z. 197-208):
   ```
   Display = 'iter'
   OptimalityTolerance = 1e-5
   MaxIter = 100
   MaxFunctionEvaluations = 300
   SpecifyObjectiveGradient = true
   SpecifyConstraintGradient = true
   ```

##### Output `IM_param_info` (Z. 188-192)
```
struct mit Feldern:
  map                  -> @(q) V_e*q + H*phi(q)
  polynomialOrder      -> M
  dimension            -> k
  tangentSpaceAtOrigin -> V_e
  nonlinearCoefficients-> H
  phi                  -> @(y) [monomials]
  exponents            -> exp_mat (Reihen sind Multi-Indizes)
  l, c1, c2            -> Hyperparameter
```

#### 5.3.3 `IMGeometryParaCon.m` (159 Zeilen) — Parameter-abhängige SSM

**Signatur:** `IMInfo = IMGeometryParaCon(yData, etaData, varargin)`

> "Identification of the parametrization of a manifold in k dynamical reduced coordinates and l parameters, i.e. the map y = V(x,p) via a weighted ridge regression. V(x,p) = W_v * phi(x,p) where phi is a (k+l)-variate polynomial featuring orders from 1 to order Mk in x and orders from 0 to order Ml in mu."
> — `src/geometry/IMGeometryParaCon.m:3-12`

Erweitert die Parametrisierung auf Familien von SSMs mit parametrischer Abhängigkeit (z.B. Bifurkations-Parameter $\mu$). Constraints können fixed points $y_0 = V(x_0,p_0)$, Origin-Fixierung $0 = V(0,p)$, und Linearpart $A_0 = D V(x_0,p_0)$ enthalten (Z. 13-22).

#### 5.3.4 Helper-Files in `src/geometry/utilsGraphT0/`

| Datei | Zeilen | Zweck |
|-------|--------|-------|
| `alignmentLinearConstraint.m` | 40 | Lineare Equality+Inequality für Alignment $V$ ↔ $V_{ref}$. |
| `constrainedRidgeRegression.m` | 32 | KKT-Lösung der Ridge mit $HV=0$ Constraint, sparse. |
| `defineNonlinConstraints.m` | 22 | Vektorisierte Form von $V^\top V=I$ und $V^\top H=0$ + Jacobian. |
| `delayTangentSpace.m` | 16 | Analytische Formel $V_{kj} = \exp(k\tau\lambda_j)$. |
| `extrasNonlinearConstraints.m` | 33 | Index-Helper für sparse Constraint-Jacobian. |
| `fMinimize.m` | 32 | Kostenfunktion und analytischer Gradient für `fmincon`. |

### 5.4 `src/reduceddynamics/` — Reduzierte Dynamik & Normal Form

#### 5.4.1 `IMDynamicsFlow.m` (452 Zeilen) — Generischer Flow-Identifier

**Signatur:** `[RDInfo, R, iT, N, T] = IMDynamicsFlow(etaData, varargin)`

> "Identification of the reduced dynamics in k coordinates, i.e. the vector field
>
>                       \dot{x} = R(x)
>
> via a weighted ridge regression. R(x) = W_r * phi(x) where phi is a k-variate polynomial from order 1 to order M. Cross-validation can be performed on random folds or on the trajectories for the map R."
> — `src/reduceddynamics/IMDynamicsFlow.m:3-9`

Optional wird die Dynamik via Koordinatenwechsel auf eine Normalform reduziert:

> "Upon request, the dynamics is returned via a coordinate change, i.e.
>
>                        R = D_T o N o iT
>
> where iT, T and N depend on the selected style.
> If the style is selected as modal, then the coordinate change is a linear map that transforms the linear part of R into the diagonal matrix of its eigenvalues.
> If the style is selected as normalform, then the functions seeks from data the maps iT, N and T such that the dynamics N is in normal form. This option is only available for purely oscillatory dynamics (i.e., the eigenvalues of the linear part of R are complex conjugated only)."
> — `src/reduceddynamics/IMDynamicsFlow.m:11-24`

##### Detaillierter Ablauf (Zeilen 99-264)

**Schritt 1: Zeitableitung berechnen** (Z. 99-109).
Pro Trajektorie wird `finiteTimeDifference(X_in, t_in, 3)` aufgerufen:

> "Central finite time difference with uniform monodimensional grid spacing of customized accuracy, equal to 2*halfw."
> — `src/utils/finiteTimeDifference.m:2-5`

Mit `halfw=3` ist das eine $O(\Delta t^6)$ zentrale Differenz. Koeffizienten-Matrix steht in `finiteTimeDifference.m:12-15`. Trajektorien werden durch Konkatenation in Matrizen `X` und `dXdt` gespeichert.

**Schritt 2: Polynom-Basis bauen** (Z. 116):
```matlab
[phi, Expmat] = multivariatePolynomial(k, 1, options.R_PolyOrd);
```
$\phi: \mathbb{R}^k \to \mathbb{R}^{\dim\phi}$ stapelt alle Monome vom Grad 1 bis $M_R$. Berechnet sich aus `multivariateExponents.m` rekursiv. Anzahl Monome: $\dim\phi = \sum_{r=1}^{M_R}\binom{k+r-1}{r}$.

**Schritt 3: Ridge Regression** (Z. 121-122):
```matlab
[W_r, l_opt, Err] = ridgeRegression(phi(X), dXdt, options.L2, options.idx_folds, options.l_vals);
```
Löst:
$$W_r^\star = \arg\min_{W} \big\| (\dot X - W\,\phi(X))\,L\big\|_F^2 + \lambda \|W\|_F^2$$
mit gewichteter geschlossener Form:
$$W_r = \left( (\phi L)^\top \phi + \lambda I\right)^{-1}(\phi L)^\top \dot X.$$
Implementation in `ridgeRegression.m`:
```matlab
H = diag(X_normal.^(-1)) * ((XL'*X + diag(l_opt*ones(1,Nfeatures))) \ (XL'*Y));
```
— `src/utils/ridgeRegression.m:27-28`

Die Spalten von $\phi(X)$ werden vor dem Solve auf max-Absolut-Wert normiert (Z. 14: `X = X./X_normal`), und nachher wieder de-normiert (Z. 27: `diag(X_normal.^(-1)) * ...`).

Cross-Validation (`ridgeregression_CVloop`, Z. 34-47): Für jeden Wert in `l_vals` werden alle Folds in `idx_folds` durchlaufen, der mittlere Test-Fehler genommen, und das beste $\lambda$ ausgewählt. Fold-Modi (Z. 433-449): `'default'` = randomisierte Folds, `'traj'` = Leave-One-Trajectory-Out.

Output: Vektorfeld-Funktion $R(y) = W_r\,\phi(y)$ (Z. 137).

**Schritt 4: Eigenwertanalyse der Linearisierung** (Z. 143):
```matlab
[V, D, d] = eigSorted(W_r(:,1:k));
```
$W_r(:,1:k)$ ist die lineare Submatrix (die ersten $k$ Spalten entsprechen den Monomen $y_1,\dots,y_k$). `eigSorted` (`src/reduceddynamics/utils/eigSorted.m`):

> "Sorting eigenvalues as cc with positive imaginary parts, reals and cc with negative imaginary parts. Frequencies are sorted from slowest to fastest."
> — `src/reduceddynamics/utils/eigSorted.m:2-5`

**Schritt 5: Koordinatenwechsel** (Z. 144-259), je nach `options.style`:

- **`'none'`** (Default): keine Transformation. $R$, $T$, $iT$, $N$ identisch.
- **`'modal'`** (Z. 147-156): Lineare Transformation $T = V$, $iT = V^{-1}$. Die nichtlinearen Koeffizienten werden über `multivariatePolynomialLinTransf` umgerechnet:

  > "Given phi, a k-variate polynomial of order M, and a transformation x = V*y, this code gets the matrix V_M such that phi(x) = phi(V*y) = V_M * phi(y)"
  > — `src/utils/multivariatePolynomialLinTransf.m:2-3`

  Damit: $W_n = V^{-1} W_r V_M$ (Z. 154), wobei $V_M$ block-diagonal pro Polynom-Grad ist.

- **`'normalform'`** (Z. 157-253): Suche nichtlineare Koordinatenwechsel $z = T^{-1}(y)$ und $\dot z = N(z)$ mit $N$ in Normalform. **Nur** verfügbar wenn alle Eigenwerte komplex-konjugiert sind (oszillatorisch). Falls reale Eigenwerte vorhanden (Z. 162-163): fallback auf `modal`.

##### Normalform-Optimierung (zentral)

Die Normalform-Initialisierung (`initialize_nf_flow`, Z. 282-392) detektiert die zu behaltenden Monome ("resonante Terme") über Small-Denominators:

```matlab
lidx_n = find( abs( repmat(d_nf,1,size(Expmat_n,1)) - ...
                    repmat(transpose(Expmat_n*d_nf),k,1) ) < tol_nf );
```
— `src/reduceddynamics/IMDynamicsFlow.m:332-333`

Das ist die diskrete Variante der **Cohomological Equation**:
$$\lambda_e - \langle m, \lambda\rangle \approx 0 \quad\Rightarrow\quad \text{Term resonant, wird in Normalform behalten}.$$
Die Toleranz `tol_nf` wird in zwei Modi gesetzt (Z. 320-330):

- `nf_style='center_mfld'` (Default, Z. 320-327): $\lambda_{nf} = i\,\mathrm{Im}(\lambda)$ (rein imaginär, Center-Manifold-Reduktion). Toleranz hardcoded `1e-8`. Bei vorgegebenen Frequenz-Verhältnissen `frequencies_norm = [m_1, m_2, ...]`: $d_{nf} = i\cdot[m_1,m_2,\dots,-m_1,-m_2,\dots]\cdot|\mathrm{Im}(\lambda_1)|$. Damit kann der User explizit eine 1:2- oder 1:3-Resonanz erzwingen.
- `nf_style='actual eigs'`: $d_{nf} = d$ (echte Eigenwerte), Toleranz `tol_nf*max(|Re(d)|)`.

##### Normalform-Theorie (Poincaré-Dulac, Murdock 2003)

Gesucht: Koordinatenwechsel $z = T^{-1}(y) = y + h_2(y) + h_3(y) + \dots$ und Normalform $\dot z = N(z) = D z + n_2(z) + n_3(z) + \dots$, sodass die ursprüngliche Dynamik $\dot y = D y + f_2(y) + \dots$ in $N$ konjugiert wird. Bei jedem Polynomgrad $k$ ergibt sich die **Cohomological Equation**:

$$\mathcal{L}_k\,h_k - n_k = R_k$$

mit dem **homologischen Operator**
$$(\mathcal{L}_k h_k)(y) := D h_k(y) - D y \cdot \nabla h_k(y),$$

und der rechten Seite $R_k$, die rekursiv aus $f_2,\dots,f_k$ und $h_2,\dots,h_{k-1}$ konstruiert wird. In Eigenraum-Basis ($D = \mathrm{diag}(\lambda_e)$) wirkt $\mathcal{L}_k$ diagonal auf jedem Monom $y^m e_e$:
$$\mathcal{L}_k(y^m e_e) = (\lambda_e - \langle m,\lambda\rangle)\,y^m e_e.$$

Die Lösung lautet daher:
- Wenn $\lambda_e - \langle m,\lambda\rangle \neq 0$ (nicht-resonant): $h_k$-Koeffizient $= R_k\text{-Koeffizient}/(\lambda_e - \langle m,\lambda\rangle)$, $n_k\text{-Koeffizient} = 0$.
- Wenn $\lambda_e - \langle m,\lambda\rangle = 0$ (resonant): $n_k\text{-Koeffizient} = R_k\text{-Koeffizient}$, $h_k\text{-Koeffizient}$ bleibt frei (typisch 0).

In SSMLearn wird diese analytische Auflösung **nicht** ordnung-für-ordnung gemacht; stattdessen werden alle Koeffizienten gleichzeitig per nicht-linearer Optimierung der Invarianz-Gleichung gefunden. Die Resonanz-Entscheidung wirkt nur darauf, welche Koeffizienten **als unbekannt** angesetzt werden.

##### Invarianz-Gleichung in der Implementation

Aus `dynamicsCoordChangeNF.m`:

> "Minimization of the invariance equation for a normal form coordinate change in a dynamical systems. For a flow, the invariance equation reads
>
>                DT^{-1}(y_k)\dot{y}_k = N(T^{-1}(y_k))
>
> while for a map
>
>                T^{-1}(y_{k+1}) = N(T^{-1}(y_k)).
>
> in which
>
>                z = T^{-1}(y) = y + W_it_nl \phi_it(y)
>                N(z) = Dz +  W_n_nl \phi_n(z)
>
> where the unknowns are the coefficients W_it_nl, W_n_nl, determined via an unconstrained optimization process. D is a known diagonal matrix."
> — `src/reduceddynamics/dynamicsCoordChangeNF.m:1-19`

Die Kosten-Funktion (`f_minimize`, Z. 88-153):
```matlab
% Err_k = dYdt - D*Y - D*W_it_nl*phi_it(Y) + W_it_nl*Dphi_it(Y)*dYdt 
%         - W_n*phi_n(Y + W_it_nl*phi_it(Y))
```
— `src/reduceddynamics/dynamicsCoordChangeNF.m:90`

Mathematisch: setze $z = T^{-1}(y) = y + W_{iT}\,\phi_{iT}(y)$ und $\dot z = D z + W_n\,\phi_n(z)$. Differenziere die erste Gleichung:
$$\dot z = \dot y + W_{iT}\,\nabla\phi_{iT}(y)\,\dot y.$$
Setze beide Ausdrücke gleich:
$$\dot y + W_{iT}\,\nabla\phi_{iT}(y)\,\dot y \;=\; D(y + W_{iT}\phi_{iT}(y)) + W_n\,\phi_n(y + W_{iT}\phi_{iT}(y)).$$
Umstellen ergibt die Residuum-Definition:
$$\mathrm{Err}(y,\dot y) := \dot y - Dy - D W_{iT}\phi_{iT}(y) + W_{iT}\,\nabla\phi_{iT}(y)\,\dot y - W_n\,\phi_n(y + W_{iT}\phi_{iT}(y)).$$

Das ist exakt der Code-Ausdruck (Z. 109-110):
```matlab
Err = Maps_info.Yk_1_DYk_r + W_it_nl*Maps_info.Phi_iT_Yk_1 - ...
      (Maps_info.d_r .* iTk_nl + W_n_nl*Phi_N);
```
wobei `Yk_1_DYk_r = dYdt - D*Y` (vorberechnet, Z. 386-388 in `IMDynamicsFlow.m`).

Kostenfunktional (Z. 111):
$$f(W_{iT}, W_n) = \frac{1}{N\cdot k_{red}} \sum_i \mathrm{Err}_i^* \mathrm{Err}_i\, L_{2,i}$$

Optimiert mit `fminunc` (`dynamicsCoordChangeNF.m:46`), ohne Constraints, mit user-supplied Gradient (Z. 112-152). Initialwert wird in `initialize_nf_flow` (Z. 371-377) konstruiert: für nicht-resonante Terme der iT-Koeffizienten wird die analytische Cohomological-Lösung als Startpunkt verwendet:
```matlab
W_it_0 = W_it_0 ./ (repmat(d,1,size(Expmat_it,1)) - ...
                    repmat(transpose(Expmat_it*d),k,1));
```
— Z. 344-345 (das ist exakt die analytische Cohomological-Equation-Lösung pro Monom).

Dann wird per `fminunc` die Invarianz-Gleichung wirklich minimiert (falls die Daten von der analytischen Initialisierung abweichen, weil die SSM aus Daten kommt, nicht aus dem exakten Vektorfeld).

##### Optionen `IMDynamicsFlow` (Auszug der wichtigsten, vollständig in Z. 44-93)

| Option | Default | Bedeutung |
|--------|---------|-----------|
| `R_PolyOrd` | 1 | Polynom-Ordnung von $R$ (lineare Default). Bei `nargin==2` von `varargin` übernommen. |
| `N_PolyOrd` | =`R_PolyOrd` | Ordnung der Normalform $N$. |
| `iT_PolyOrd` | =`R_PolyOrd` | Ordnung von $T^{-1}$. |
| `T_PolyOrd` | =`R_PolyOrd` | Ordnung von $T$. |
| `style` | `'none'` | `'none'` / `'modal'` / `'normalform'`. |
| `nf_style` | `'center_mfld'` | `'center_mfld'` oder `'actual eigs'`. |
| `frequencies_norm` | [] | Erwartete Frequenz-Verhältnisse (z.B. `[1 2]` für 1:2-Resonanz). |
| `tol_nf` | 10 | Multiplikator für Resonanz-Toleranz. |
| `IC_nf` | 1 | Initialwert für NF-Optimierung: 0=zero, 1=cohomological estimate, 2=randomized. |
| `l_vals` | 0 | Ridge-$\lambda$-Kandidaten. |
| `n_folds` | 0 | Anzahl CV-Folds (0 = keine CV). |
| `fold_style` | [] | `'default'` (random) oder `'traj'` (LOO-Trajectory). |
| `c1`, `c2` | 0, 0 | Slow-Manifold-Time-Weighting wie in §5.3. |
| `rescale` | 1 | Modal-Koord-Rescaling. |
| `R_coeff` | [] | Vorgegebene lineare Anteile (für Mech mit bekannter Massen-Matrix). |
| `MaxIter` | 1e3 | fminunc Iterationen. |
| `MaxFunctionEvaluations` | 1e4 | fminunc fevals. |

Die letzten fünf "are for Matlab function fminunc. For more information, check out its documentation." (Zeile 92-93).

##### Output `RDInfo` (Z. 260-263)
```
struct mit Feldern:
  reducedDynamics      -> {map: R, coefficients: W_r, polynomialOrder, phi, exponents, l_opt, CV_error}
  inverseTransformation-> {map: iT, coeff, phi, exponents, lintransf}
  conjugateDynamics    -> {map: N, coeff, phi, exponents, damping (von polarNormalForm), frequency, ...}
  transformation       -> {map: T, coeff, phi, exponents, lintransf}
  conjugacyStyle       -> 'none'/'modal'/'normalform'
  dynamicsType         -> 'flow'
  eigenvaluesLinPartFlow -> d (Vector der Eigenwerte)
  eigenvectorsLinPart  -> V
```

#### 5.4.2 `IMDynamicsMap.m` (393 Zeilen) — Diskrete Dynamik

**Signatur:** `[RDInfo, R, iT, N, T] = IMDynamicsMap(etaData, varargin)`

> "Identification of the reduced dynamics in k coordinates, i.e. the map
>
>                       x_{k+1} = R(x_k)
>
> via a weighted ridge regression."
> — `src/reduceddynamics/IMDynamicsMap.m:3-8`

Strukturell identisch zu `IMDynamicsFlow`, aber:
- Anstatt $\dot X$ zu berechnen, werden Snapshot-Paare $(X_k, X_{k+1})$ gebildet (Z. 100-103):
  ```matlab
  X = X_i(:, 1:end-1);
  X_1 = X_i(:, 2:end);
  ```
- Ridge-Regression löst $X_{k+1} = W_r\,\phi(X_k)$ (Z. 114).
- Eigenwerte werden über `log(λ)/Dt` in Continuous-Time umgerechnet (`eigSorted(W_r(:,1:k), Dt)`, Z. 123).
- Toleranz für Resonanzen ist diskret formuliert: `max(|eig|)*(1-exp(-tol_nf*max(|real(log(eig))|)))` (Doku Z. 60-61).

#### 5.4.3 `IMDynamicsMech.m` (431 Zeilen) — Mechanik-spezifisch

**Signatur:** `[RDInfo, R, iT, N, T] = IMDynamicsMech(X_traj, varargin)`

> "Identification of the reduced dynamics in k coordinates, i.e. the vector field for mechanical systems
>
>        \dot{x} = R(x) where x = (u,v) and R(x) = (v, f(u,v))"
> — `src/reduceddynamics/IMDynamicsMech.m:3-8`

Spezialfall: zerlegt den Zustand in $(u, v) = (\text{Verschiebung}, \text{Geschwindigkeit})$, fittet **nur** $f(u,v) = \dot v$ (die untere Hälfte), und setzt $\dot u = v$ analytisch (Z. 117-119):
```matlab
[W_r,l_opt,Err] = ridgeRegression(phi(X), dXdt(ndof+1:end,:), ...);
W_r = [zeros(ndof) eye(ndof) zeros(ndof, size(W_r,2)-2*ndof); W_r];
```
Das halbiert die zu schätzenden Parameter und garantiert die mechanische Struktur.

#### 5.4.4 `IMDynamicsFlowFractional.m` (245 Zeilen)

Variante mit fraktionellen Polynom-Exponenten, basiert auf `multivariateFractionalPolynomial.m`. Wird für Reibungs-Modelle und nicht-glatte Mechanik verwendet (siehe Bettini, Cenedese, Haller 2024 — `International Journal of Non-Linear Mechanics`).

#### 5.4.5 `IMDynamicsFlowParaCon.m` (163 Zeilen) und `IMDynamicsMapParaCon.m` (164 Zeilen)

Parameter-abhängige Varianten ($\dot x = R(x, p)$). Constraints können fixed points $0=R(x_0,p_0)$, Origin-Fixierung $0=R(0,p)$ oder Linear-Anteile $A_0 = D_x R(x_0,p_0)$ enthalten. Diese sind nötig für Bifurkations-Analyse.

#### 5.4.6 `dynamicsCoordChangeNF.m` (155 Zeilen) — Normalform-Solver
Bereits in §5.4.1 ausführlich dokumentiert. Wird sowohl von `IMDynamicsFlow` als auch `IMDynamicsMap` und `IMDynamicsMech` aufgerufen.

#### 5.4.7 `conjugacyErrorTrend.m` (44 Zeilen)

**Signatur:** `errorInfo = conjugacyErrorTrend(etaData, indTrain, indTest, polyOrders)`

> "Evaluate the conjugacy error on training and test trajectory for a set of polynomial orders. The routines fits the normal form on the training data for each polynomial order and computes the squared conjugacy error, showing a picture with its trend varying the polynomial order"
> — `src/reduceddynamics/conjugacyErrorTrend.m:1-7`

Diagnose-Tool: für jeden Polynom-Grad in `polyOrders`:
1. Fitte Normalform auf Train-Set.
2. Berechne $\Delta := \dot z - N(z)$ für alle Trajektorien.
3. Plotte $\sum |\Delta|^2$ über Train/Test als Funktion der Ordnung.

Wird benutzt um optimalen ROM-Order zu wählen (kein Auto-Mechanismus, User-Hilfe).

#### 5.4.8 `src/reduceddynamics/utils/`

| Datei | Zeilen | Zweck |
|-------|--------|-------|
| `dispNormalForm.m` | 54 | Tabellen-Anzeige der Normalform-Koeffizienten. |
| `dispNormalFormFigure.m` | 144 | LaTeX-Figure-Anzeige der NF-Gleichungen. |
| `eigSorted.m` | 25 | Sortiert Eigenwerte: cc+, real, cc-; Frequenz langsam→schnell. |
| `iterateMap.m` | 6 | Trivialer Iterator für diskrete Maps. |
| `polarNormalForm.m` | 300 | Konvertiert komplexe NF nach polar $(\rho,\theta)$, definiert `damps(\rho)` und `freqs(\rho)`. |
| `RidgeRegressionConstrainedParametric.m` | 155 | Ridge mit (parametrischen + fixed-point + linearpart) Constraints. |

### 5.5 `src/postprocessing/` — Forced Response & Backbone

#### 5.5.1 `backboneCurves.m` (132 Zeilen)

**Signatur:** `BBCInfo = backboneCurves(IMInfo, RDInfo, amplitudeFunction, maxRho, varargin)`

> "Plot instantaneous amplitude and frequency/damping curves." […INPUT/OUTPUT-Block ausgelassen…] "The amplitude is defined as the maximum absolute value reached by the the scalar input function amplitudeFunction along a full rotation in the angle theta for each value of the parametrizing amplitude from 0 to maxRho. If an additional argument is given, the values of frequency and damping are normalized with respect to the linear limit (input 'norm') or in terms of damping ration and Hertz for the frequency. For a SSM with more than 1 mode, the backbone curves are computed for each uncoupled limit."
> — `src/postprocessing/backboneCurves.m:6, 23-30`

Algorithmus pro DoF (Z. 47-61):
1. $\rho \in [0, \rho_{max}]$ in `nRhoEvals = 1001` Schritten gleichmäßig (Z. 48).
2. Für jeden $\rho$: berechne $\omega(\rho), \alpha(\rho)$ aus `RDInfo.conjugateDynamics.frequency` und `damping` (Z. 50-52). Diese sind aus `polarNormalForm.m` als Funktion-Handles.
3. Für die Amplitude: erzeuge $\theta \in [0, 2\pi]$ in 50 Schritten, $z = \rho e^{i\theta}$, transformiere komplex-konjugiert, dann $y = T(z)$, dann $x = W(y)$ (parametrization), dann `amplitudeFunction(x)`, nimm Maximum über $\theta$ (Z. 54-60).

Output: `damping`, `frequency`, `amplitude`, `amplitudeNormalForm` als $(d/2) \times 1001$-Matrizen. Je nach `varargin`:
- `'Hz'`: Frequenz in Hz, Damping als Damping-Ratio in %.
- `'norm'`: relativ zu linearem Limit normiert.
- (default): rad/s und 1/s.

#### 5.5.2 `backboneSurfaces.m` (150 Zeilen)

**Signatur:** `BBSInfo = backboneSurfaces(RDInfo, max_rho, varargin)`

Erweitert Backbone Curves auf 2D-Plots für 4D-SSMs mit zwei Modi (Internal Resonance). Erzeugt $40\times 40$-Meshgrid in $(\rho_1, \rho_2)$ und plottet Frequenz-/Damping-Surfaces.

#### 5.5.3 `computeFRC.m` (109 Zeilen) — Forced Response Curve via Continuation

**Signatur:** `FRC_data = computeFRC(IMInfo, RDInfo, reducedForcing, frequencySpan, amplitudeFunction)`

> "The non-autonomous dynamics has the form \dot{z} = N(z) + f(z,t) where f(z,t) = [ [f_1 f_2 ... f_{ndof}] * e^{\Omega t}, compl. conj.] where ndof = k/2 and k is dimension of reduced order model. The values f_1, f_2 can be complex, whose magnitude is the forcing amplitude on a specific mode. The matrix reducedForcing has dimension k/2 x nSweeps, whose rows store the forcing for each mode for each required sweep."
> — `src/postprocessing/computeFRC.m:5-10` (Hinweis: das `...` in `[f_1 f_2 ... f_{ndof}]` ist Teil der originalen mathematischen Notation, kein Auslassungsmarker)

Algorithmus pro Forcing-Vector (Z. 31-103):
1. Definiere zeitabhängiges System $\dot z = N(z) + i f e^{i\Omega t} + \text{c.c.}$ in Real/Imag-Splitting (Z. 33-35).
2. Initiale Schätzung der periodischen Bahn: 500 Perioden Transient via `ode45` (Z. 42-44).
3. Numerische Continuation periodischer Bahnen mit **coco** (`coco_prob`, `ode_isol2po`, Z. 47-69):
   - `'NTST', 120, 'NCOL', 4` — Diskretisierung der periodischen Bahn.
   - `'PtMX', 20000` — max Continuation-Steps.
   - `'NAdapt', 1` — adaptive Mesh-Anpassung.
4. Extrahiere $T_{vec}$, $X_0$, Stabilität via `coco_bd_col`.
5. Berechne Amplituden in physischen Koordinaten durch Lifting $z\to T(z)\to W(T(z))\to$ `amplitudeFunction`.

Output: struct mit Feldern `Freq`, `Amp`, `Nf_Amp`, `Nf_Phs`, `Stab` pro Sweep.

#### 5.5.4 `analyticalFRC.m` (124 Zeilen) — analytische FRC für 2D SSMs

**Signatur:** `FRC = analyticalFRC(IMInfoF, RDInfoF, fRed, amplitudeFunction, varargin)`

> "Compute the forced response curves on a 2D SSM analytically for each normal form forcing amplitude in fRed." […INPUT/OUTPUT-Block ausgelassen…] "The conventions used are
> \dot{z} = (\alpha(\rho) + i*(omega(\rho) - \Omega) ) z + i*f*e^{i\Omega t}
> z = \rho * e^{i\theta t}
> f = fscale * fRed * exp(1i fphase)
> psi = \theta - \Omega t - fphase [negative normal form phase psi is indeed a lag for positive damping]."
> — `src/postprocessing/analyticalFRC.m:3-4, 21-25`

Löst die Ein-Mode-Gleichgewichtsbedingung in polaren Koordinaten:
$$\alpha(\rho) = 0, \qquad \omega(\rho) = \Omega \pm \frac{f}{\rho}.$$
Implementiert über symbolisches Solven für jedes $\Omega$. Vermeidet numerische Continuation komplett, ist aber nur für 2D-SSMs (1 DoF in NF) gültig.

#### 5.5.5 `continuationFRCpo.m` (126 Zeilen)
Variante von `computeFRC` für Continuation mit `forcedSSMROM`-Output (zeit-periodische SSM-Parametrisierung).

#### 5.5.6 `continuationFRCep.m` (369 Zeilen)
Continuation der **fixed-points** der polaren Normalform anstatt periodischer Bahnen. Konvertiert das Time-Periodic Problem in ein Equilibrium-Problem in der polaren Form (`coordinates = 'polar'`, Z. 33).

> "This function converts the normal form style reduced dynamics on the SSM to fixed point problem to compute the periodic response (FRC) for internally resonant systems"
> — `src/postprocessing/continuationFRCep.m:27-29`

Nötig für **internal-resonant** Systeme (1:1, 1:2, 1:3, …) wo die analytische FRC nicht reicht.

#### 5.5.7 `calibrateFRC.m` (24 Zeilen)

**Signatur:** `fRed = calibrateFRC(IMInfo, RDInfo, yCal, Omega)`

> "Compute the forcing amplitudes in the normal form such that the FRC gives the responses yCal at forcing frequencies Omega"
> — `src/postprocessing/calibrateFRC.m:1-5`

Inversion: gegeben gemessene Forced-Response-Punkte, finde die korrespondierenden Normal-Form-Forcing-Konstanten via:
$$f = \rho\sqrt{(\omega(\rho)-\Omega)^2 + \alpha(\rho)^2}.$$
(Z. 23). Wird beim Sloshing-Beispiel benutzt um aus 3 experimentellen Forced-Punkten zu kalibrieren.

#### 5.5.8 `plotFRC.m` (110 Zeilen)
Nur Visualisierung. Plottet alle Sweeps (`F1`, `F2`, …) in einer Figur.

#### 5.5.9 `cocoutils/`
Helper für coco-Continuation: `add_slot_IP` (COCO-Slot-Helper für Internal Phase Continuation), `add_slot_Traj` (Trajektorien-Sampling während Continuation), `coco_bd_labs_and_period` (extrahiert Labels + Perioden aus bd-struct). Reine Glue-Funktionen.

### 5.6 `src/timedependentmanifold/forcedSSMROM.m` (170 Zeilen)

**Signatur:** `[IMInfoF, RDInfoF] = forcedSSMROM(IMInfo, RDInfo, varargin)`

> "Derive a forced m-dim. SSM model. The default is a time-periodic SSM model, but any finite number of (positive) frequencies can be inserted, standing hence for quasi-periodic forcing."
> — `src/timedependentmanifold/forcedSSMROM.m:2-5`

> "The output model does not take of resonances between the linearized frequencies and forcing ones: the reduced dynamics features forcing in every mode and the normal form retains only the +Omega t rotation, while the normal form coordinate change has the other rotation."
> — `src/timedependentmanifold/forcedSSMROM.m:6-9`

Wandelt eine autonome SSM aus `IMGeometry` + Normalform aus `IMDynamicsFlow` in einen forced ROM um, indem die Forcing-Vektoren in die normal-form-Koordinaten projiziert werden:
- `modalForcing = inverseTransformation.lintransf * (W_e * forcingVectors)` (Z. 65-67).
- Optional mit "outer" linearisierten Modi falls der volle Phasenraum bekannt ist (Z. 60-94).

Ist die einzige Datei in `src/timedependentmanifold/`.

### 5.7 `src/utils/` — Helper, Plot, I/O

Kompakt eine Zeile pro Datei; kritische Helper sind oben ausführlich behandelt.

| Datei | Zeilen | Zweck |
|-------|--------|-------|
| `advect.m` | 33 | Integriert RDInfo aus initialer Bedingung in `yData` und liftet zurück nach $y$. Zentraler Predict-Helper. |
| `advectRD.m` | (kurz) | Integriert nur in reduzierten Koordinaten (kein Lift). |
| `cocoSet.m` | (kurz) | Setzt coco-Optionen einheitlich (NTST, NCOL, PtMX, etc.). |
| `computeAmpPhaseErrors.m` | (kurz) | Berechnet Amp/Phasen-Fehler einer FRC vs Daten. |
| `computeParametrizationErrors.m` | 11 | $\|\text{lift}\circ\text{project}(y) - y\|/\|y\|$ pro Trajektorie. |
| `computeTrajectoryErrors.m` | 37 | $\|y_1-y_2\|/\max\|y_2\|$ pro Trajektorie + Amplituden-Fehler. |
| `contFRC.m` | (kurz) | Wrapper für FRC-Continuation, dispatcht zwischen `continuationFRCep`/`po`. |
| `convertLivescript2Markdown.m` | (kurz) | Konvertiert .mlx zu .md (Doku-Build). |
| `coordinatesEmbedding.m` | 110 | Delay-Embedding (siehe §5.2). |
| `embedCoordinates.m` | 38 | Vereinfachtes Delay-Embedding (siehe §5.2). |
| `ep_reduced_results.m` | (kurz) | Equilibrium-Point-Continuation-Output-Reader. |
| `errorAmplitudePhase.m` | (kurz) | Amp/Phase-Vergleich. |
| `finiteTimeDifference.m` | 26 | Zentrale FD-Ableitung Ordnung $2h$, $h\le 4$. |
| `fitRD2Data.m` | 32 | Wrapper: fittet `IMDynamicsMech` mit gegebener Linear-Part, advectet, vergleicht. |
| `fitSSM2Data.m` | 39 | Wrapper: fittet `IMGeometry` über bekannten Modal-Subspace $(V_e, W_e)$. |
| `functionFromTensors.m` | 40+ | Erzeugt anonyme Funktion $F(t,q)=\dot q$ aus $M,C,K, f_{nl}$ Tensoren (für Tests). |
| `funToCell.m` | (kurz) | Vektorfeld-Cell-Wrapper. |
| `getManifoldPoint.m` | 34 | Evaluiert Polynom-Parametrisierung aus mfd-struct. |
| `getSSM.m` | 30 | Ruft SSMtool auf um die analytische SSM zu berechnen (für Vergleich). |
| `getSSMIC.m` | 47 | Generiert nTraj initiale Bedingungen auf der analytischen SSM (für Tests). |
| `integrateFlows.m` | 26 | `ode15s`-Integration einer Vektorfeld-Funktion entlang aller Trajektorien-Zeiten. |
| `integrateTrajectories.m` | (kurz) | Voll-System-Integration. |
| `iterateMaps.m` | 19 | Diskrete Map-Iteration. |
| `liftTrajectories.m` | 17 | Wendet `IMInfo.parametrization.map` trajektorienweise an. |
| `linearpart.m` | 40 | Berechnet $A$, $V$, $\lambda$ aus $M, C, K$ (mit/ohne Damping). |
| `modal_analysis.m` | (kurz) | FE-Modal-Analyse für vorberechnete Strukturmodelle. |
| `monitor_states.m` | (kurz) | coco-Monitoring-Funktion. |
| `multivariateExponents.m` | 199 | Generiert die Multi-Index-Matrix für $k$-variate Polynome vom Grad $N$ (rekursiv via `expomatr`). Cf. §5.4.1. |
| `multivariateFractionalPolynomial.m` | (kurz) | Polynom mit fraktionellen Exponenten (Reibung). |
| `multivariatePolynomialLinTransf.m` | 25 | Berechnet $V_M$ s.d. $\phi(Vy) = V_M \phi(y)$ (für Modal-Transform der NL Koeffs). |
| `multivariatePolynomial.m` | 35 | Erzeugt symbolisch das Monom-Vektorfeld $\phi(u)$ und seine Ableitungs-Tabelle. |
| `multivariatePolynomialSelection.m` | 24 | Wie oben, aber nur für eine Index-Auswahl (für resonante Terme der NF). |
| `ode_2mDSSM_cartesian.m` | 40+ | Vektorfeld-Funktion für 2m-D forced SSM in Cartesian (z=Re+iIm, gestapelt). |
| `ode_2mDSSM_polar.m` | (kurz) | Selbe Sache in polaren Koordinaten $(\rho,\theta)$. |
| `orthogonalizeGramSchmidt.m` | 13 | Standard-Gram-Schmidt für $V$. |
| `paperFigure.m` | (kurz) | Plot-Style-Helper. |
| `PCA.m` | 26 | SVD-basierte PCA (gibt $V$ und Singular-Werte zurück). |
| `pickPointsOnHypersphere.m` | (kurz) | Quasi-zufällige Punkte auf $S^{d-1}$. |
| `pinkgreen.m` | (kurz) | Color-Map. |
| `plot2DSSM.m` | (kurz) | 2D-SSM-Plot. |
| `plotModalTrajectories.m` | (kurz) | Plot in Modal-Koordinaten. |
| `plotReducedCoordinates.m` | (kurz) | Plot in $\eta$-Koordinaten. |
| `plotSSMandTrajectories.m`, `plotSSMWithTrajectories.m` | (kurz) | 3D-SSM + Trajektorien. |
| `plotTrajectories.m` | (kurz) | Trajektorien-Time-Series-Plot. |
| `projectTrajectories.m` | 17 | Wendet `IMInfo.chart.map` trajektorienweise an. |
| `purpleorange.m` | (kurz) | Color-Map. |
| `RBF_interpolator.m` | (kurz) | Radial-Basis-Function-Interpolation. Alternative zur Polynom-Regression bei Extrapolations-Problemen. |
| `rcoordinatesStandardization.m` | 8 | SVD-basierte Rotation und Skalierung der reduzierten Koords (max-Norm = 1). |
| `ridgeRegression.m` | 47 | Gewichtete Ridge mit CV. Kern der Regressions-Pipeline. |
| `sliceTrajectories.m` | 24 | Schneidet Trajektorien auf Zeit-Intervalle. |
| `SSMToolFRCFE.m`, `SSMToolFRC.m` | (lang) | Bridge zu SSMtool für analytische FRC-Berechnung als Vergleich. |
| `static_analysis.m` | (kurz) | Statische Antwort einer Struktur. |
| `timeWeighting.m` | (kurz) | Time-Weighting-Vektor-Builder (Parameter $c_1, c_2$). |
| `transformationComplexConj.m` | 3 | $x \mapsto [x; \bar x]$. |
| `transformationReIm.m` | 4 | $x \mapsto [\mathrm{Re}\,x; \mathrm{Im}\,x]$. |
| `transformTrajectories.m` | 31 | Wendet eine Map auf alle Trajektorien-States an. |
| `unravelField.m` | (kurz) | Entrollt 2D-Felder zu Vektoren. |

---

## 6. End-to-End Workflow als nummerierte Schrittliste (für Codex-Checkliste)

Exakte Schritt-für-Schritt-Pipeline wie in den Beispielen; jeder Schritt zitiert die implementierende(n) Funktion(en).

### 6.1 Schritt: Daten laden und Cell-Format herstellen
- **Was:** Daten in `xData{i,1}=t_i`, `xData{i,2}=x_i` packen. Konstantes $\Delta t$ pro Trajektorie.
- **Code-Anker:** Standard-MATLAB. Beispiel: `examples/sloshing/sloshing.mlx`.

### 6.2 Schritt: SSM-Dimension bestimmen
- **Manuell** (häufigster Fall): User wählt $d \in \{2, 4, 6, ...\}$ basierend auf erwarteter Anzahl Eigenwert-Paare.
- **Automatisch:**
  - `[startTime, indStartTime, SSMDim] = SSM_startTime(data, indplot)` (`src/preprocessing/SSM_startTime.m`)
  - intern: `showSpectrogram` → `analyzeSpectr` (welche Frequenzen sind dominant über die Zeit?).

### 6.3 Schritt: Trunkierung der Daten auf den SSM-Bereich
- **Was:** Frühe Transienten verwerfen, weil sie noch nicht auf der SSM liegen.
- **Code-Anker:**
  - `regimeLinear(data, lim)` für PFF-basierte Detektion eines linearen Regimes.
  - `sliceTrajectories(data, [t_start, t_end])` für die Schnitt-Operation.
- **Originalkommentar:**
  > "We must make sure that the data approximates the SSM, i.e., sufficient truncation has been applied. This can be decided based on the spectrogram."
  > — `docs/Practical_considerations.pdf` Seite 1, Punkt 1

### 6.4 Schritt: Phasenraum-Rekonstruktion (Delay Embedding)
- **Was:** Wenn der Beobachterraum nicht reichhaltig genug ist (z.B. nur skalare Messung), stacke Zeit-verschobene Kopien.
- **Code-Anker:** `[yData, optsEmbd] = coordinatesEmbedding(xData, SSMDim, 'OverEmbedding', 0, 'ShiftSteps', 1)` (siehe §5.2).
- **Default $p$:** $\lceil (2d+1)/n\rceil$ Kopien (Takens-Mindest-Embedding), augmentiert mit `OverEmbedding` extra Kopien.
- **$\tau$-Wahl:** Default `ShiftSteps=1`, also $\tau=\Delta t$. User-tunable, kein Auto-AMI/Cao.

### 6.5 Schritt: SSM-Parametrisierung fitten
- **Was:** Konstruiere die Polynom-Parametrisierung $W: \mathbb{R}^d \to \mathbb{R}^p$.
- **Code-Anker:**
  ```matlab
  [IMInfo, IMChart, IMParam] = IMGeometry(yData, SSMDim, M, ...
                                          'l', 0, 'c1', 0, 'c2', 0)
  ```
- **Was passiert intern:**
  1. `style='natural'` (Default) → `IMGeometryGraphT0`.
  2. Initialisiere $V_{e,0}$ via SVD von $X$ (PCA, Zeile 140).
  3. Initialisiere $H_0$ via `constrainedRidgeRegression` (Zeile 151).
  4. Optimiere $\min_{V_e,H} \mathcal J$ unter $V_e^\top V_e=I$, $V_e^\top H=0$ via `fmincon`.
  5. Output: `IMInfo.chart.map = @(x) V_e'*x` (linear), `IMInfo.parametrization.map = @(q) V_e*q + H*phi(q)`.

### 6.6 Schritt: Reduzierte Koordinaten projizieren
- **Was:** $\eta_i = V_e^\top y_i$ pro Trajektorien-Snapshot.
- **Code-Anker:** `etaData = projectTrajectories(IMInfo, yData)` → wendet `IMInfo.chart.map` auf jeden Snapshot an.

### 6.7 Schritt: Fitting-Fehler der Parametrisierung
- **Code-Anker:** `errors = computeParametrizationErrors(IMInfo, yData)`.
- Berechnet $\|\text{lift}(\text{project}(y)) - y\| / \max\|y\|$.

### 6.8 Schritt: Reduzierte Dynamik fitten
- **Was:** $\dot \eta = R(\eta)$ als Polynom Grad $M_R$.
- **Code-Anker (Flow):**
  ```matlab
  [RDInfo, R, iT, N, T] = IMDynamicsFlow(etaData, ...
                          'R_PolyOrd', M_R, ...
                          'style', 'normalform', ...
                          'l_vals', [0, 1e-4, 1e-2], ...
                          'n_folds', 5)
  ```
- **Code-Anker (Map):** `IMDynamicsMap(etaData, 'R_PolyOrd', M_R, ...)`.
- **Code-Anker (Mechanik):** `IMDynamicsMech(etaData, ...)`.
- **Was passiert intern (siehe §5.4.1):**
  1. Numerische Ableitung via `finiteTimeDifference`.
  2. Polynom-Basis $\phi$ Ordnung 1 bis $M_R$.
  3. Ridge-Regression $W_r = (\Phi^\top L \Phi + \lambda I)^{-1}\Phi^\top L \dot X$ mit CV.
  4. Eigenwert-Sortierung.
  5. Optional Modal- oder Normalform-Wechsel (siehe §6.9).

### 6.9 Schritt: Normalform-Reduktion (optional, nur für oszillatorische SSMs)
- **Code-Anker:** automatisch wenn `'style','normalform'` in `IMDynamicsFlow`.
- **Was passiert (siehe §5.4.1):**
  1. `eigSorted` → komplex-konjugierte Eigenwerte sortiert.
  2. `initialize_nf_flow`: detektiere resonante Indizes via `|d_e - <m, d>| < tol_nf`.
  3. Initialisiere $W_{iT,0}$ aus Cohomological-Equation-Solution.
  4. `dynamicsCoordChangeNF`: minimiere $\|\dot z - Dz - DW_{iT}\phi_{iT} + W_{iT}\nabla\phi_{iT}\dot y - W_n \phi_n(z)\|^2$ via `fminunc`.
  5. Berechne $T$ aus $\eta = T(z)$ via Ridge-Regression auf $z = \eta + W_{iT}\phi_{iT}(\eta)$.
- **Output:** $iT, N, T$ als Funktions-Handles + Koeffizienten-Tabellen.

### 6.10 Schritt: Polare Normalform (Damping/Frequency-Funktionen)
- **Code-Anker:** `polarNormalForm(NInfo, optPlot)`.
- **Was:** Konvertiert komplexe NF in $(\rho, \theta)$ und definiert
  $$\alpha(\rho) = \mathrm{Re}\,(N\bar z/|z|^2), \qquad \omega(\rho) = \mathrm{Im}\,(N\bar z/|z|^2).$$
- Speichert `damping`, `frequency` als Function-Handles in `RDInfo.conjugateDynamics`.

### 6.11 Schritt: Backbone Curves
- **Code-Anker:** `BBCInfo = backboneCurves(IMInfo, RDInfo, amplitudeFunction, maxRho, 'Hz')`.
- **Output:** $\omega(\rho)$, $\alpha(\rho)$, plus Amplitude $A(\rho) = \max_\theta |\text{amplitudeFunction}(W(T(\rho e^{i\theta})))|$.

### 6.12 Schritt: Forced Response (für autonome SSM mit hinzu-modelliertem Forcing)
- **Code-Anker für 2D-SSM:** `analyticalFRC(IMInfoF, RDInfoF, fRed, ampFun)`.
- **Code-Anker für höherdim. oder resonante SSM:**
  ```matlab
  [IMInfoF, RDInfoF] = forcedSSMROM(IMInfo, RDInfo, ...);
  FRC = computeFRC(IMInfoF, RDInfoF, reducedForcing, frequencySpan, ampFun);
  ```
  oder `continuationFRCpo` / `continuationFRCep` für interne Resonanz.

### 6.13 Schritt: Validierung
- **Code-Anker:**
  - `[yRec, etaRec, zRec] = advect(IMInfo, RDInfo, yData)` — Predict aus initialer Bedingung.
  - `normedTrajDist = computeTrajectoryErrors(yRec, yData)` — NMTE pro Trajektorie.
  - `conjugacyErrorTrend` — Polynom-Ordnung-Sweep.
  - `calibrateFRC` — Forced-Calibration aus Messdaten.

---

## 7. Zentrale Mathematik (vollständig ausgeschrieben)

### 7.1 Invarianz-Gleichung der Parametrisierung

Eine SSM ist eine **invariante Mannigfaltigkeit**: wenn $x(0) \in \mathcal{W}$, dann $x(t) \in \mathcal{W}$ für alle $t$. Mit der Parametrisierung $x = W(y)$ und der reduzierten Dynamik $\dot y = R(y)$ folgt

$$\boxed{\;\dot W(y)\cdot R(y) \;=\; f(W(y))\;}$$

oder als Differentialgleichung in $y$:
$$\nabla W(y)\,R(y) = f(W(y)).$$

In der Parametrisierungs-Methode (Cabré, Fontich, de la Llave 2003) wird diese Gleichung **ordnungsweise** in einer Taylor-Entwicklung gelöst. Setze
$$W(y) = V_e y + W_2(y) + W_3(y) + \dots, \qquad R(y) = D_E y + R_2(y) + R_3(y) + \dots$$
mit $V_e \in \mathbb{R}^{N\times d}$ als Tangentialraum-Basis (Spalten = $d$ Eigenvektoren von $E$). Da $V_e$ rechteckig ist ($N \ge d$), bezeichnet $D_E \in \mathbb{R}^{d\times d}$ die **Restriktion/Repräsentation** von $A$ auf $E$ in dieser Basis: konkret gilt $A V_e = V_e D_E$, und für diagonalisierbares $A$ mit ausgewählten Master-Eigenwerten ist $D_E = \mathrm{diag}(\lambda^{(1)}_E,\dots,\lambda^{(d)}_E)$. $W_k, R_k$ sind homogene Polynome vom Grad $k$.

Die **lineare Ordnung** ($k=1$) ist trivial erfüllt durch $A V_e = V_e D_E$ (Eigenvektor-Definition).

Für **Ordnung $k \ge 2$** liefert das Einsetzen, das Anwenden der Produktregel auf $DW(y)\,R(y)$ und das Sortieren nach Polynomgrad:
$$\boxed{\;\mathcal{L}_k W_k(y) \;+\; V_e\,R_k(y) \;=\; G_k(y)\;}$$
mit dem **Cohomological Operator**
$$\mathcal{L}_k W_k(y) \;:=\; DW_k(y)\cdot D_E\,y \;-\; A\,W_k(y),$$
und der bekannten rechten Seite $G_k(y)$, die ausschließlich von $W_2,\dots,W_{k-1}$ und $R_2,\dots,R_{k-1}$ sowie den Taylor-Koeffizienten $f_2,\dots,f_k$ abhängt. Die rechte Seite sammelt alle Beiträge der Ordnung $k$, die NICHT $W_k$ oder $R_k$ enthalten.

**In SSMLearn wird diese ordnungsweise Auflösung NICHT durchgeführt.** SSMLearn löst die Invarianz-Gleichung nicht. Stattdessen werden zwei voneinander getrennte Regressionsprobleme gelöst:

1. **Geometrie $W$:** Minimierung des Rekonstruktionsfehlers
   $$\min_{V_e,\,H} \;\sum_i\;\big\|\,x_i \;-\; \big(V_e\,\eta_i + H\,\phi(\eta_i)\big)\,\big\|^2,\qquad \eta_i := V_e^{\top} x_i,$$
   unter Tangentialitäts-Constraint $V_e^{\top} H = 0$ (Graph-Form). Implementiert in `src/geometry/IMGeometryGraphT0.m:94-187` via `constrainedRidgeRegression` (bekanntes $V_e$) bzw. `fmincon` mit `defineNonlinConstraints` und `alignmentLinearConstraint` (unbekanntes $V_e$).

2. **Reduzierte Dynamik $R$:** Minimierung des Vektorfeld-Fits
   $$\min_{R}\;\sum_i\;\big\|\,\dot\eta_i \;-\; R(\eta_i)\,\big\|^2$$
   per Polynom-Regression auf $\phi(\eta)$, anschließend Normalform via `dynamicsCoordChangeNF`. Implementiert in `src/reduceddynamics/IMDynamicsFlow.m`. Die $\dot\eta_i$ werden aus den projizierten Trajektoriendaten geschätzt (numerische Ableitung oder finite differences); sie sind **nicht** automatisch das exakte reduzierte Vektorfeld der echten SSM, solange die Geometrie-Parametrisierung $W$ aus Schritt 1 nur näherungsweise gilt. Der Fit $R$ identifiziert also die "auf $W$-Bild beobachtete" Dynamik.

Beide Fits sind **algorithmisch unabhängig**. Die SSM-Invarianz $DW(y)\,R(y) = f(W(y))$ wird **nicht erzwungen**. Sie ist **näherungsweise** erfüllt unter den Voraussetzungen, dass (a) die Trainingsdaten tatsächlich nahe einer echten SSM liegen, (b) die Polynom-Ordnung in beiden Fits hinreichend hoch ist und (c) die Trajektoriendaten den interessanten Bereich der Mannigfaltigkeit hinreichend abdecken.

Diese Strategie ist **rigoros verschieden** von der Parametrisierungs-Methode in SSMtool, wo $W_k$ und $R_k$ ordnungsweise so konstruiert werden, dass die Invarianz-Gleichung exakt bis zur jeweiligen Polynomordnung erfüllt ist.

### 7.2 Cohomological Equation in Eigenraum-Basis

Sei $A = V \Lambda V^{-1}$ die Eigenzerlegung der Jacobi-Matrix mit $\Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_N)$, $D_E = \mathrm{diag}(\lambda^{(1)}_E,\dots,\lambda^{(d)}_E)$ die Master-Eigenwerte, und betrachte einen einzelnen Polynom-Term $y^m e_j := y_1^{m_1}\cdots y_d^{m_d} e_j$ in der $j$-ten Komponente einer geeigneten Basis. Wende den Cohomological Operator $\mathcal{L}_k W_k = DW_k\cdot D_E\,y - A\,W_k$ an: die Wirkung auf dieses Monom ist
$$\mathcal{L}_k\big(y^m e_j\big) \;=\; \big(\langle m, \lambda_E\rangle - \lambda_j\big)\,y^m\,e_j,$$
mit $\langle m,\lambda_E\rangle := m_1\lambda^{(1)}_E + \dots + m_d\lambda^{(d)}_E$, $|m|=k$, und $\lambda_j$ als Eigenwert in Richtung $e_j$.

**Resonanz-Bedingung:** $\langle m,\lambda_E\rangle - \lambda_j = 0$. In diesem Fall ist das Monom **resonant** und das Bild des Cohomological Operators in dieser Richtung ist null — der Term kann nicht durch Koordinatenwechsel auf die slaved Seite verschoben werden und bleibt in der reduzierten Dynamik $R$ (für $j\in E$) bzw. erfordert eine Modifikation der SSM-Parametrisierung (für $j\notin E$).

**Non-Resonanz:** $\big|\langle m,\lambda_E\rangle - \lambda_j\big| > 0$. Für slaved Richtungen $j\notin E$ folgt aus $\mathcal{L}_k W_k = G_k$ in dieser Komponente direkt
$$h_{k,m,j} \;=\; \frac{G_{k,m,j}}{\langle m,\lambda_E\rangle - \lambda_j}.$$

**Small-Denominators-Problem:** Wenn $|\lambda_e - \langle m,\lambda\rangle|$ klein ist (aber nicht null), wird $h_{k,m,e}$ groß und numerisch instabil. SSMLearn detektiert solche Terme über die Toleranz `tol_nf` (Code in §5.4.1, `src/reduceddynamics/IMDynamicsFlow.m:332-333`) und behandelt sie als "fast-resonant", d.h. behält sie in $N$ und entfernt sie aus $T^{-1}$.

### 7.3 Center-Manifold-Stil

Im Default `nf_style = 'center_mfld'` wird $\lambda_{nf} = i\,\mathrm{Im}(\lambda)$ gesetzt (Z. 320-323). Damit wird die Resonanz-Bedingung
$$|\mathrm{Im}(\lambda_e) - \langle m, \mathrm{Im}(\lambda)\rangle| < 10^{-8}$$
geprüft. Das ist eine **Frequenz-Resonanz** im Center-Manifold-Sinn: der nichttriviale Realteil (Damping) wird ignoriert, weil er für die Existenz der zentralen invarianten Mannigfaltigkeit irrelevant ist. Dies ist robuster als die "actual eigs"-Variante bei numerisch fast-resonanten Systemen.

### 7.4 Spektralgap (formell)

Mit $\lambda_e$ den Eigenwerten in $E$ und $\lambda_o$ denen außerhalb, gilt für eine **slow stable** SSM die **strikte** Spektralgap-Bedingung:
$$\boxed{\;\max_{o\notin E}\mathrm{Re}\,\lambda_o \;<\; \min_{e\in E}\mathrm{Re}\,\lambda_e \;<\; 0.\;}$$
Anschaulich: alle Eigenwerte sind stabil ($\mathrm{Re}\,\lambda < 0$), und **jede** Master-Mode in $E$ ist **näher** an der imaginären Achse (weniger stark gedämpft, "langsamer") als **jede** Slaved-Mode außerhalb $E$. Die strikte Form $\max_o < \min_e$ — anstatt der schwächeren $\min_o < \max_e$ — ist nötig, weil sonst nicht garantiert ist dass alle Slaved-Moden schneller sind als alle Master-Moden; einzelne Überlappungen würden den Spektralgap brechen.

Diese Form ist konsistent mit der Doppel-Bedingung in §1.2: $\max_{e\in E}\mathrm{Re}\,\lambda_e < 0$ (Stabilität) und $\max_{o\notin E}\mathrm{Re}\,\lambda_o < \min_{e\in E}\mathrm{Re}\,\lambda_e$ (strikter Spektralgap).

SSMLearn überprüft das **nicht** automatisch — der User muss sicherstellen, dass die gewählte Modal-Untermenge tatsächlich die strikt langsamste ist (durch Schauen auf die Eigenwerte aus `eigSorted`).

### 7.5 Multivariate-Polynom-Regression (Frobenius-Norm Kostenfunktion)

Für die SSM-Geometrie (`IMGeometryGraphT0`):
$$\min_{V_e,H} \;\frac{1}{nN}\sum_{i=1}^N \big\|(x_i - V_e y_i - H\phi(y_i))\,L_{ti}\big\|_2^2 + \frac{1}{nN}\,\lambda\,\|H\|_F^2$$
unter $V_e^\top V_e = I_d$ und $V_e^\top H = 0$.

Für die reduzierte Dynamik (`IMDynamicsFlow`):
$$\min_{W_r}\; \big\|(\dot X - W_r\,\Phi(X))\,L\big\|_F^2 + \lambda\|W_r\|_F^2.$$

Beide werden in geschlossener Form gelöst (Ridge), bzw. die Geometrie-Variante über `fmincon` falls $V_e$ unbekannt ist.

### 7.6 Iterative Orbit-Rekonstruktion

SSMLearn verwendet **keine** iterative Orbit-Rekonstruktion (im Sinne von Picard-Iterationen). Die Predict-Pipeline ist einfach:
```
y_0 → η_0 = V_e^T y_0 → z_0 = T^{-1}(η_0) → integrate ż = N(z) → z(t) → η(t) = T(z(t)) → y(t) = W(η(t))
```
Die Integration in `integrateFlows.m` benutzt MATLAB's `ode15s` mit `RelTol=1e-4`:
```matlab
opts = odeset('RelTol',1e-4);
[t, x] = ode15s(@(t,y) flow(y), linspace(tStart, tEnd, nSamp), etaData{iTraj,2}(:,1), opts);
```
— `src/utils/integrateFlows.m:3, 12`

---

## 8. Outputs

### 8.1 `IMInfo` (struct)
- `chart.map`: Funktion $\mathbb{R}^p \to \mathbb{R}^d$ (Projektion in reduzierte Koordinaten).
- `chart.polynomialOrder`: 1 (linear, $V_e^\top$).
- `parametrization.map`: Funktion $\mathbb{R}^d \to \mathbb{R}^p$ (Polynom $W$).
- `parametrization.polynomialOrder`: $M$.
- `parametrization.dimension`: $d$.
- `parametrization.tangentSpaceAtOrigin`: $V_e$ ($p\times d$ orthonormal).
- `parametrization.nonlinearCoefficients`: $H$ ($p\times \dim\phi$).
- `parametrization.phi`: $\phi$ als Funktions-Handle.
- `parametrization.exponents`: Multi-Index-Matrix.

### 8.2 `RDInfo` (struct)
- `reducedDynamics.{map, coefficients, polynomialOrder, phi, exponents, l_opt, CV_error}`: $R(y)$.
- `inverseTransformation.{map, coeff, phi, exponents, lintransf}`: $iT$.
- `transformation.{map, coeff, phi, exponents, lintransf}`: $T$.
- `conjugateDynamics.{map, coeff, phi, exponents, damping, frequency, LaTeXComplex}`: $N(z)$ + polare Funktionen.
- `conjugacyStyle`: 'none'/'modal'/'normalform'.
- `dynamicsType`: 'flow' oder 'map'.
- `eigenvaluesLinPartFlow`: Vektor der Eigenwerte.
- `eigenvectorsLinPart`: $V$.

### 8.3 `BBCInfo` (Backbone)
- `damping(d/2 × 1001)`: $\alpha(\rho)$ pro Mode.
- `frequency(d/2 × 1001)`: $\omega(\rho)$ pro Mode.
- `amplitude(d/2 × 1001)`: physikalische Amplitude.
- `amplitudeNormalForm(d/2 × 1001)`: $\rho$-Achse.

### 8.4 `FRC_data` (Forced Response)
- Pro Sweep `F1, F2, ...`: `Freq, Amp, Nf_Amp, Nf_Phs, Stab` als Vektoren.

---

## 9. Beispiele unter `examples/` (vollständige Liste)

### 9.1 Übersicht

`SSMLearn/examples/` enthält **20 Top-Level-Verzeichnisse**, jeweils mit 1-Zeilen-Beschreibung, primärem Hauptscript und Domäne. Mechanisch/FEM ist die überwältigende Mehrheit; nicht-mechanische Domänen sind explizit markiert.

| # | Verzeichnis | Hauptscript(s) | Domäne | Beschreibung |
|---|-------------|----------------|--------|--------------|
| 1 | `brakereussbeam/` | `brb.mlx` | **Mechanik (Experiment)** | Brake-Reuß-Beam mit gefügter Bolzenverbindung; DIC-Verschiebungs- und Beschleunigungs-Messungen aus Shaker-Ringdown. Slowest 2D SSM. |
| 2 | `bucklingbeam/` | `vonKarmanBuckling4D.mlx`, `vonKarmanBuckling2D.mlx` | **Mechanik (FE-Simulation)** | Pinned-pinned von-Kármán-Beam unter Längskompression; 4D primary mixed-mode SSM für dynamisches Buckling. |
| 3 | `couetteflow/` | `romRe134.mlx`, `romRe135.mlx`, `romRe146.mlx`, `romRe134Fractional.mlx` | **Strömungs-Mechanik (CFD)** | Plane Couette Flow zwischen Re=134 und 146; SSM um den Upper-Branch-Fixpunkt. CFD-Daten via Channelflow, PCA-komprimiert auf 20 Modi. **NICHT-mechanisch aber strukturell**. |
| 4 | `experiment_nonlinear_beam_oblique/` | `nonlinear_beam_oblique.mlx` | **Mechanik (Experiment)** | Nichtlinearer Beam mit oblique-projection-Methode. |
| 5 | `invertedflag/` | `regular_flapping.mlx`, `chaotic_case.mlx` | **Fluid-Struktur (Experiment)** | Inverted Flag im Wassertunnel; reguläre und chaotische Flapping-Regime. 2D und 4D SSMs. **Chaotisches Beispiel hier!** |
| 6 | `oscillatorchain/` | `oscillator2D.mlx`, `oscillator4D.mlx`, `oscillator2DTakens.mlx`, `oscillator_tutorial.mlx` | **Mechanik (synthetisch)** | $N$-DoF Oszillatorkette mit zusätzlicher nichtlinearer Feder; slow 4D SSM. `oscillator2DTakens.mlx` demonstriert den Delay-Embedding-Workflow für skalare Observable. |
| 7 | `parametricSSM_TimeDelay/` | `Furuta/`, `Various/` (Subordner) | **Mechanik (Experiment + Simulation)** | Parametrische SSM mit Zeit-Delays. Subordner: `Furuta/` mit Furuta-Pendel und chaotischem Attraktor (Trajectory Generator → ParametricSSM → ChaoticAttractor); `Various/` als Sammelkasten. **Ein einziges Top-Level-Verzeichnis.** |
| 8 | `planarexample/` | `planarexample.mlx`, `planar.m` | **Synthetisch 2D ODE** | Planares System $\dot x = x(y-b), \dot y = cy(x-a)$ mit zwei Fixpunkten und heteroklinem Orbit. **Reine ODE-Pädagogik**, kein mechanischer Hintergrund. Closest example zu generic phase-space dynamics. |
| 9 | `prismaticbeam/` | `prismaticbeam.mlx` | **Mechanik (FE)** | 4D SSM für 1:3 internal resonant prismatic beam. |
| 10 | `resonantdoublebeam/` | `resonantdoublebeam.mlx` | **Mechanik (Experiment)** | Doppel-Beam mit 1:2 internal resonance; Laser-Vibrometrie. 4D resonante SSM. |
| 11 | `shaw_pierre_cart_oblique/` | `shaw_pierre_cart_oblique.mlx` | **Mechanik (synthetisch)** | Shaw-Pierre 2-DoF Oszillator in kartesischen Koordinaten mit oblique projection. |
| 12 | `shaw_pierre_oblique/` | `shaw_pierre_oblique.mlx` | **Mechanik (synthetisch)** | Shaw-Pierre mit oblique projection. |
| 13 | `shawpierreforced/` | `shawpierreforced.mlx`, `poincareMap.m`, `shawpierre.m` | **Mechanik (synthetisch + forced)** | Shaw-Pierre forced; Übergänge zwischen periodischen Orbits via Poincaré-Map. |
| 14 | `sloshing/` | `sloshing.mlx` | **Fluid-Mechanik (Experiment)** | Schwappendes Wasser in horizontal angeregtem Tank; Center-of-Mass-Trajektorien (Decay-Daten). 2D SSM. Forced-Response-Vorhersage aus 3 kalibrierten Punkten. **Quasi-skalare Zeitreihe!** |
| 15 | `tribomechadynamics_benchmark/` | `SSMBasedROM.mlx`, `comsolPOs.mat`, `RDinfoO5NF.mat` | **Mechanik (FE-Benchmark)** | Tribomechadynamics Benchmark mit Reibung; Reduktion auf 3-DoF SSM (siehe Morsy et al. 2025). |
| 16 | `vonkarmanbeam/` | `vonkarmanbeam.mlx`, `vonkarmanbeamPS.mlx` | **Mechanik (FE)** | Clamped-clamped von-Kármán-Beam; 2D SSM mit skalarer Mittelpunkt-Verschiebung. |
| 17 | `vonkarmanplateIR/` | `vonkarmanplateIR.mlx` | **Mechanik (FE)** | Quadratische von-Kármán-Platte mit 1:1 internal resonance; intermediate 4D SSM. |
| 18 | `vonkarmanshell/` | `vonkarmanshell.mlx` | **Mechanik (FE)** | Shallow-curved Shell mit geometrischer Nichtlinearität. |
| 19 | `vonkarmanshellIR/` | `vonkarmanshellIR.mlx` | **Mechanik (FE)** | Shell mit internal resonance zwischen den zwei langsamsten Modi. |
| 20 | `vortexshedding/` | `vortexshedding.mlx` | **Strömung (CFD)** | POD-Trajektorien einer 2D-Strömung um einen Zylinder bei Re=70; Konvergenz zum Limit-Cycle des Vortex-Shedding-Regimes. **NICHT-mechanisch.** |

### 9.2 Weitere Beispiel-Bäume außerhalb `examples/`

Zwei separate Beispiel-Sammlungen liegen NICHT unter `examples/`, gehören aber zum Repo:

**`fastSSM/examples/`** (4 Beispiele) — Demos für die minimale `fastSSM.m`-Implementation:
- `oscillator/` — synthetischer 2D-Oszillator
- `resonantbeam/` — resonant beam
- `sloshing/` — Sloshing-Variante
- `vonkarmanbeam/` — vonKarman-Beam-Variante

**`miscellaneous/nonsmoothSSM/examples/`** (2 Beispiele) — nicht-glatte SSM-Varianten:
- `shaw_pierre/` — Shaw-Pierre mit nicht-glatten Komponenten
- `von_karman_beam/` — vonKarman-Beam mit nicht-glatten Komponenten

### 9.3 Zusammenfassung der Domänen
- **Mechanik (FE/Experimente):** 16 Beispiele.
- **Strömungs-Mechanik (CFD):** 2 (Couette, Vortex Shedding) — strukturell ähnlich aber kein Federsystem.
- **Fluid-Struktur (Experimente):** 1 (Inverted Flag) — chaotisches Regime!
- **Reine 2D ODE / Pädagogik:** 1 (planarexample).
- **Sloshing / Quasi-skalar:** 1 (sloshing).

### 9.4 Wo gibt es **keine** Beispiele
- **Finanz-Zeitreihen.** Kein einziges Beispiel mit BTC, Aktien, FX oder Asset Returns.
- **Klima/Wetter-Daten.** Keine Beispiele mit ENSO, Temperatur-Reihen, etc.
- **Biologie/EEG.** Keine Beispiele.
- **Generische Zeitreihen ohne ODE-Hintergrund:** Das nächste an "rein observational" ist `sloshing` (Center-of-Mass-Trajektorien aus Experiment), gefolgt von `brakereussbeam` (DIC-Daten), `invertedflag` (chaotic flapping experiments) und `vortexshedding` (CFD-Snapshots). In allen Fällen weiß man aber dass ein autonomes ODE/PDE-System dahintersteht.

### 9.5 Beispiele die strukturell zu BTC log-Residuen passen könnten
- **`sloshing/sloshing.mlx`** — quasi-skalare Time-Series, Decay zu einem Limit Cycle, Forced Response per Calibration. Strukturell am nächsten zu "Decay-zu-Attraktor".
- **`vortexshedding/vortexshedding.mlx`** — POD-komprimierte Time-Series, Convergence zu Limit Cycle aus instabilem Origin.
- **`invertedflag/chaotic_case.mlx`** — chaotic regime, demonstriert SSMLearn auf nicht-periodischen Trajektorien (Liu, Axås, Haller 2024 — Inertial Manifold als SSM).
- **`planarexample/planarexample.mlx`** — minimaler Test-Fall mit nur ODE-Daten ohne mechanischen Bezug.

---

## 10. Kritische Limitationen

Aus dem Code, dem `Practical_considerations.pdf` und dem Cenedese-2022-Paper extrahiert:

### 10.1 Autonomie ist Pflicht
Die SSM-Theorie ist für **autonome** Systeme formuliert (Cabré-Fontich-de la Llave 2003 verlangt $\dot x = f(x)$ ohne explizite $t$-Abhängigkeit). SSMLearn bietet eine forced-Erweiterung (`forcedSSMROM.m`), die aber nur **schwache** Forcing-Terme behandelt: das Forcing wird **linear** in den normal-form-Koordinaten an die autonome SSM angeklebt. Bei großen Forcing-Amplituden ist die SSM keine Mannigfaltigkeit mehr, sondern ein bewegtes Objekt.

### 10.2 Fixpunkt im Ursprung
`IMGeometryGraphT0.m:5` setzt das hartcodiert. Bei nicht-trivialen Fixpunkten muss man die Daten zentrieren. Bei mehreren Fixpunkten (wie in `couetteflow`): explizit auf einen Fixpunkt zentrieren und den Trajectory-Branch wählen.

### 10.3 Stationarität / keine Drift
SSMLearn nimmt implizit an, dass der Datensatz aus **Decay**- oder **steady-attractor**-Trajektorien besteht. Drift, Heteroskedastizität oder regimewechselnde Zeitreihen werden nicht modelliert. Es gibt **keine** Behandlung von Saisonalität, Trend oder strukturellen Brüchen.

### 10.4 Beobachtbarkeit (Observable Choice)
Die Wahl des Beobachterraums beeinflusst die Geometrie der SSM massiv:

> "If the observables are directly the phase space coordinates, the leading principal components obtained by SVD might not align with the tangent space of the slow SSM. In these cases, we need to use the subsequent principal components that do align with the slow directions. Alternatively, switching to delay-embedded scalar observables also fixes this issue, as delay embedded slow SSMs tend to be flat."
> — `docs/Practical_considerations.pdf` Seite 1, Punkt 5

### 10.5 Datenmenge
Es gibt keinen formellen Sample-Complexity-Satz. Empirisch: pro DoF-Mode braucht man eine Trajektorie die durch mindestens 3-5 Schwingungs-Perioden geht, mit ausreichender Auflösung um $\dot y$ via FD zu schätzen (i.e. min. 20-30 Sample/Periode).

### 10.6 Polynom-Extrapolation
> "Extrapolating the polynomial vector field of the reduced dynamics to unseen regions of the phase space might produce badly behaved solutions. In these cases, alternative representations such as k-nearest neighbors (kNN) or radial basis functions (RBFs) might improve the extrapolation properties, as in Liu et al. *Chaos* 34 033140 (2024) and Xu et al. *J. Fluid Mech.* 987 R7 (2024)."
> — `docs/Practical_considerations.pdf` Seite 1, Punkt 4

### 10.7 Finite-Time-Blowup bei zu hoher Polynom-Ordnung
> "If the reduced dynamics exhibits finite-time blowup, the polynomial order might need to be decreased."
> — `docs/Practical_considerations.pdf` Seite 1, Punkt 3

### 10.8 Normalform-Konvergenz
> "If the optimization finding the normal form reduced dynamics fails to converge, changing the initial guess might be required. This can be achieved by changing the 'IC_nf' argument of IMDynamicsFlow() from the default value of '0' to '1' or '2'."
> — `docs/Practical_considerations.pdf` Seite 1, Punkt 6

**Achtung — Manual nicht synchron mit dem Code:** Das PDF-Manual sagt der Default sei `0`, aber drei der vier `IMDynamics*`-Varianten setzen den Default auf **`1`** (gegen den Code verifiziert):
- `IMDynamicsFlow.m:403` → `'IC_nf', 1`
- `IMDynamicsMech.m:382` → `'IC_nf', 1`
- `IMDynamicsFlowFractional.m:195` → `'IC_nf', 1`
- `IMDynamicsMap.m:344` → `'IC_nf', 0`  ← einzige Variante, in der das Manual stimmt

Das heißt: in `Flow`, `Mech` und `FlowFractional` startet die NF-Optimierung per Default schon mit dem cohomologischen Schätzer (`IC_nf=1`), in `Map` mit dem Zero-Start (`IC_nf=0`). Wenn die Konvergenz versagt, sind die verbleibenden Optionen je nach Variante `0`, `1` oder `2`. Diese Diskrepanz ist vermutlich auf einen Code-Patch nach Erstellung des Manuals zurückzuführen; das Manual wurde nicht synchron nachgezogen.

### 10.9 Resonance-Detection
Die Toleranz `tol_nf` ist heuristisch. Bei numerisch fast-resonanten Systemen kann eine zu kleine `tol_nf` zu instabilen Koeffizienten führen, eine zu große zu einem unter-bestimmten System. Default ist `1e-8` (`center_mfld`) bzw. `10*max(|Re(eig)|)` (`actual eigs`).

### 10.10 Real-Eigenwerte = keine Normalform
Wenn die Linearisierung von $R$ reale Eigenwerte hat, fällt SSMLearn auf den Modal-Style zurück (`IMDynamicsFlow.m:162-163`). Der Normalform-Code unterstützt nur **rein oszillatorische** SSMs.

---

# Teil III — Python und Anwendung

## 11. Python: SSMLearnPy und Portierungs-Referenz

### 11.1 SSMLearnPy (vorhandene offizielle Python-Implementierung)

`/home/hz/Data/Attractor/SSMLearnPy/` ist die datengetriebene Python-Implementierung (Cenedese). Auf Platte verifiziert:
- Paket `ssmlearnpy/` mit Modulen `geometry/`, `reduced_dynamics/`, `utils/`, `main/`, `base/`; dazu `tests/`, `examples/`, `docs/`.
- README nennt dieselben drei Rechenschritte wie das MATLAB-README (Embedding → Parametrisierung/latente Koordinaten → reduzierte Dynamik) und: "with initial implementation of normal form transformation" — der Normalform-Teil ist in Python also weniger ausgebaut als in MATLAB.
- Mitgelieferte Beispiele: oscillator chain, Brake-Reuss beam, vortex shedding, Couette flow, sloshing, 1-DoF-Oszillator.
- Installation:
  ```sh
  git clone https://github.com/mattiacenedese/SSMLearnPy.git
  cd SSMLearnPy/
  conda create -n ssmlearn python=3.9   # optional
  ```

Die folgende Portierungs-Tabelle (ursprünglich als Spezifikation für eine Neuimplementierung geschrieben) dient damit als Referenz, um MATLAB-Features gegen SSMLearnPy zu prüfen bzw. fehlende Teile (v.a. Normalform, FRC-Continuation) nachzurüsten.

### 11.2 Minimaler Kern (zu portierende Dateien, mit Zeilen-Schätzung)

| Priorität | Datei | Zeilen | Python-Schwierigkeit | Notizen |
|-----------|-------|--------|----------------------|---------|
| 1 | `src/utils/coordinatesEmbedding.m` | 110 | trivial | numpy reshape + slicing |
| 1 | `src/utils/multivariateExponents.m` | 199 | mittel | Multi-Index-Generator; `itertools.combinations_with_replacement` |
| 1 | `src/utils/multivariatePolynomial.m` | 35 | trivial | Matrix-Form $\Phi(X)$ |
| 1 | `src/utils/ridgeRegression.m` | 47 | trivial | `numpy.linalg.solve` + CV-Loop |
| 1 | `src/utils/finiteTimeDifference.m` | 26 | trivial | numpy slicing |
| 1 | `src/geometry/IMGeometryGraphT0.m` | 208 | **schwer** | Constrained Optimization (`scipy.optimize.minimize` mit `SLSQP` oder `trust-constr`); Constraint-Jacobian muss von Hand wie in `defineNonlinConstraints.m` gebaut werden, sonst katastrophal langsam |
| 1 | `src/geometry/IMGeometry.m` | 127 | trivial | Wrapper |
| 1 | `src/geometry/utilsGraphT0/constrainedRidgeRegression.m` | 32 | trivial | Sparse-KKT-Lösung; `scipy.sparse.linalg.spsolve` |
| 1 | `src/geometry/utilsGraphT0/fMinimize.m` | 32 | mittel | Kostenfunktion + analytischer Gradient |
| 1 | `src/geometry/utilsGraphT0/defineNonlinConstraints.m` | 22 | mittel | Nichtlineare Constraint + Gradient |
| 1 | `src/geometry/utilsGraphT0/extrasNonlinearConstraints.m` | 33 | mittel | Index-Helpers für Sparse-Jacobian |
| 1 | `src/geometry/utilsGraphT0/alignmentLinearConstraint.m` | 40 | mittel | Lineare Inequality+Equality |
| 1 | `src/reduceddynamics/IMDynamicsFlow.m` | 452 | **schwer** | enthält Linearisierung + Normal-Form-Init + viele Optionen |
| 1 | `src/reduceddynamics/dynamicsCoordChangeNF.m` | 155 | **schwer** | Unconstrained Optimization mit komplex-wertigem Gradient (in Real/Imag-Splitting) |
| 1 | `src/reduceddynamics/utils/eigSorted.m` | 25 | trivial | numpy `eig` + sortieren |
| 1 | `src/reduceddynamics/utils/polarNormalForm.m` | 300 | mittel | Konvertiert komplexe NF zu polar; symbolische Auswertung kann mit `sympy` oder direkt numerisch gemacht werden |
| 1 | `src/utils/advect.m` | 33 | trivial | scipy `solve_ivp` |
| 1 | `src/utils/integrateFlows.m` | 26 | trivial | scipy `solve_ivp` |
| 1 | `src/utils/projectTrajectories.m` | 17 | trivial | List comprehension |
| 1 | `src/utils/liftTrajectories.m` | 17 | trivial | Identisch |
| 1 | `src/utils/transformTrajectories.m` | 31 | trivial | Wrapper |
| 1 | `src/utils/computeTrajectoryErrors.m` | 37 | trivial | numpy norm |
| 1 | `src/utils/orthogonalizeGramSchmidt.m` | 13 | trivial | numpy QR |
| 2 | `src/postprocessing/backboneCurves.m` | 132 | trivial | nur Plot + arithmetische Auswertung |
| 2 | `src/postprocessing/analyticalFRC.m` | 124 | mittel | symbolische FRC-Lösung |
| 2 | `src/timedependentmanifold/forcedSSMROM.m` | 170 | mittel | Forcing-Projektion |
| 3 | `src/postprocessing/computeFRC.m` | 109 | **schwer** | benötigt coco-Ersatz (PyCoBi/AUTO/manuelle Newton-Continuation) |
| 3 | `src/postprocessing/continuationFRCpo.m` | 126 | **schwer** | dito |
| 3 | `src/postprocessing/continuationFRCep.m` | 369 | **schwer** | dito |

**Total Kern (Priorität 1):** ca. 2000 Zeilen MATLAB → in Python deutlich kompakter (geschätzt 800-1200 Zeilen mit numpy/scipy).

### 11.3 Empfohlene Python-Bibliotheken
- `numpy`, `scipy.linalg`, `scipy.optimize` (für `minimize` mit `SLSQP`/`trust-constr`).
- `scipy.integrate.solve_ivp` (für `ode15s`-Ersatz, mit `BDF` oder `LSODA`).
- `scipy.sparse` (für KKT-System der constrained ridge).
- `sympy` (optional für symbolische Polynom-Generierung; alternativ direkt numpy).
- Kein PyTorch nötig — die Optimierung ist klein-skalig und numerisch viel besser direkt mit `scipy` zu machen.

### 11.4 `fastSSM/fastSSM.m` als Portierungs-Vorlage

Die Datei `fastSSM/fastSSM.m` (**122 Zeilen**, verfasst von Joar Axås) ist eine eigenständige minimale Re-Implementation des kompletten Stacks, ausschließlich für **2D-SSMs** ($\text{mfddim} = 2$ hardcoded). Die SSM-Polynom-Ordnung ist über den Eingabeparameter `mfdorder` frei wählbar; **fix hardcoded** sind hingegen ROM-Ordnung und Normalform-Ordnung (`romorder = 3`, `nforder = 3` in `fastSSM.m:20`). Sie macht alles in einer Datei:
- SVD für Tangentialraum (Zeile 26-28).
- Polynom-Regression für $W$ (Zeile 30-31).
- FD-Differentiation (Zeile 35-38).
- Polynom-Regression für $R$ (Zeile 39).
- Diagonalisierung (Zeile 40-41).
- **Analytische** Cohomological-Equation-Lösung Ordnung 2 und 3 (Zeile 45-67) — explizit ausgeschrieben mit den Resonanz-Nennern $2\lambda_\ell - \lambda_j$, $\lambda_\ell + \lambda_c - \lambda_j$, $3\lambda_\ell - \lambda_j$!
- Backbone Curves (Zeile 78-100).

Achtung: keine automatische Resonanzprüfung, keine Small-Denominator-Kontrolle, fest gewählte Gauge.

Das ist die **beste Vorlage** für eine minimale Python-Implementation, weil sie keine externen Constraint-Solver braucht. Varianten im selben Ordner: `fastSSMMap.m` (diskrete Maps), `fastSSMplus.m` (höhere Ordnungen), `miniComputeFRC.m`.

### 11.5 Fallstricke
- **MATLAB column-major vs numpy row-major.** Cell-Arrays werden zu list-of-tuples oder list-of-dicts.
- **`fmincon` vs `scipy`:** `SLSQP` ist ein guter Ersatz für `fmincon` mit nichtlinearen Constraints, aber langsamer bei großen Problemen. Für hochdimensionalen Fall (`p >> 100`) eher `trust-constr`.
- **Komplex-wertige Optimierung:** SSMLearn löst die Normalform-Optimierung in der Real/Imag-Aufspaltung des komplexen Koeffizienten-Vektors (Z. 377: `IC_opt = [real(IC_opt_complex); imag(IC_opt_complex)]`). Das sollte direkt portiert werden.
- **Toleranzen:** `tol_nf=1e-8` ist sehr streng; bei verrauschten Daten muss das hochgesetzt werden.

---

## 12. Anwendbarkeit auf BTC-log-Residuen

(Basierend auf den Limitationen und Beispielen.)

1. **Stationaritäts-Annahme prüfen.** SSMLearn nimmt an, dass der Attraktor stabil und stationär ist. BTC-log-Residuen müssen vorher trend-bereinigt sein (was sie als Residuen ja vermutlich sind).
2. **Fixpunkt im Ursprung herstellen.** Mittelwert subtrahieren.
3. **Delay-Embedding ist Pflicht** wenn nur eine 1D-Zeitreihe vorliegt. Default $p \ge 2d+1$ — bei vermuteter $d=2$ also mindestens 5 Delay-Komponenten. $\tau$ aus Mutual-Information oder False-Nearest-Neighbours wählen (außerhalb von SSMLearn).
4. **SSM-Dimension zuerst raten** ($d=2$ für Limit-Cycle-ähnlichen Attraktor; $d=4$ für Tori; $d>4$ vermutlich problematisch).
5. **Strikte Trennung Train/Test.** SSMLearn hat das eingebaut (`indTrain`, `indTest`), aber für Finanz-Daten sollte die Out-of-Sample-Validation auf einer separaten Zeitperiode geschehen.
6. **Polynom-Ordnung niedrig halten.** Empfehlung: $M=3-5$ für Geometrie, $M_R=3$ für Dynamik. Höher → finite-time-blowup-Risiko.
7. **Validate via `conjugacyErrorTrend`** über Polynom-Ordnungen, um den optimalen Trade-off zu finden.
8. **Wenn die Polynom-Regression schlecht extrapoliert:** RBF-Variante (siehe §10.6, Liu et al. 2024) oder Globalisierung via `globalized-SSM` (§2.5).
9. **Inverted-flag-Beispiel** (`examples/invertedflag/chaotic_case.mlx`) ist der einzige Test-Case auf chaotischer Dynamik im Repo und sollte als Vorbild genommen werden.

**Für den lokalen Workflow (`analyze_residuals/`, später `analyze_n_ens/`):** gezielt nur diese Dinge aus `SSMTool_jain` übernehmen (siehe §2.4) — Vorprüfungen, Resonanzlogik, Qualitätsdiagnostik, klare Workflow-Trennung (freie Dynamik / erzwungene Beiträge / Beobachtungsmodell). Alles andere würde nur Begriffe vermischen; ein Repo-Wechsel weg von SSMLearn/SSMLearnPy ist es nicht.

---

## 13. Quellen-Index (Datei → Zweck Quick-Lookup)

```
PIPELINE STAGE                    KEY FILE                                              LINES
─────────────────────────────────────────────────────────────────────────────────────────────
Spectrogram + linear regime       src/preprocessing/showSpectrogram.m                    41
                                  src/preprocessing/analyzeSpectr.m                      28
                                  src/preprocessing/SSM_startTime.m                      34
                                  src/preprocessing/regimeLinear.m                       48
                                  src/preprocessing/PFF.m / PFFk.m                       59 / 85
DMD/PCA preliminaries             src/preprocessing/DMD.m                                49
                                  src/utils/PCA.m                                        26
                                  src/preprocessing/DFT.m                                27
Oblique projection                src/preprocessing/obliqueProjection.m                 116
Phase space reconstruction        src/utils/coordinatesEmbedding.m                      110
                                  src/utils/embedCoordinates.m                           38
                                  src/geometry/utilsGraphT0/delayTangentSpace.m          16
SSM parametrization (geometry)    src/geometry/IMGeometry.m                             127
                                  src/geometry/IMGeometryGraphT0.m                      208
                                  src/geometry/IMGeometryParaCon.m                      159
   helpers                        src/geometry/utilsGraphT0/fMinimize.m                  32
                                  src/geometry/utilsGraphT0/constrainedRidgeRegression   32
                                  src/geometry/utilsGraphT0/defineNonlinConstraints.m    22
                                  src/geometry/utilsGraphT0/extrasNonlinearConstraints   33
                                  src/geometry/utilsGraphT0/alignmentLinearConstraint    40
   polynomial helpers             src/utils/multivariatePolynomial.m                     35
                                  src/utils/multivariateExponents.m                     199
                                  src/utils/multivariatePolynomialLinTransf.m            25
                                  src/utils/multivariatePolynomialSelection.m            24
                                  src/utils/multivariateFractionalPolynomial.m           ~30
   regression helper              src/utils/ridgeRegression.m                            47
   time derivative                src/utils/finiteTimeDifference.m                       26
   high-level wrappers            src/utils/fitSSM2Data.m                                39
                                  src/utils/fitRD2Data.m                                 32
Reduced dynamics + Normal Form    src/reduceddynamics/IMDynamicsFlow.m                  452
                                  src/reduceddynamics/IMDynamicsMap.m                   393
                                  src/reduceddynamics/IMDynamicsMech.m                  431
                                  src/reduceddynamics/IMDynamicsFlowFractional.m        245
                                  src/reduceddynamics/IMDynamicsFlowParaCon.m           163
                                  src/reduceddynamics/IMDynamicsMapParaCon.m            164
                                  src/reduceddynamics/dynamicsCoordChangeNF.m           155
                                  src/reduceddynamics/conjugacyErrorTrend.m              44
   helpers                        src/reduceddynamics/utils/eigSorted.m                  25
                                  src/reduceddynamics/utils/polarNormalForm.m           300
                                  src/reduceddynamics/utils/dispNormalForm.m             54
                                  src/reduceddynamics/utils/dispNormalFormFigure.m      144
                                  src/reduceddynamics/utils/iterateMap.m                  6
                                  src/reduceddynamics/utils/RidgeRegressionConstrained  155
Time-periodic / forced            src/timedependentmanifold/forcedSSMROM.m              170
Backbone curves                   src/postprocessing/backboneCurves.m                   132
                                  src/postprocessing/backboneSurfaces.m                 150
Forced response curves            src/postprocessing/computeFRC.m                       109
                                  src/postprocessing/analyticalFRC.m                    124
                                  src/postprocessing/continuationFRCpo.m               126
                                  src/postprocessing/continuationFRCep.m                369
                                  src/postprocessing/calibrateFRC.m                      24
                                  src/postprocessing/plotFRC.m                          110
   coco glue                      src/postprocessing/cocoutils/*.m                       (klein)
Predict / Validation              src/utils/advect.m                                     33
                                  src/utils/advectRD.m                                  (klein)
                                  src/utils/integrateFlows.m                             26
                                  src/utils/iterateMaps.m                                19
                                  src/utils/projectTrajectories.m                        17
                                  src/utils/liftTrajectories.m                           17
                                  src/utils/transformTrajectories.m                      31
                                  src/utils/computeTrajectoryErrors.m                    37
                                  src/utils/computeParametrizationErrors.m               11
                                  src/utils/sliceTrajectories.m                          24
                                  src/utils/computeAmpPhaseErrors.m                     (klein)
                                  src/utils/errorAmplitudePhase.m                        (klein)
SSMtool bridge (vergleich)        src/utils/getSSM.m                                     30
                                  src/utils/getSSMIC.m                                   47
                                  src/utils/getManifoldPoint.m                           34
                                  src/utils/SSMToolFRC.m / SSMToolFRCFE.m               (lang)
                                  src/utils/functionFromTensors.m                        ~80
                                  src/utils/linearpart.m                                 40
                                  src/utils/modal_analysis.m                            (klein)
                                  src/utils/static_analysis.m                           (klein)
Coordinate utils                  src/utils/transformationComplexConj.m                   3
                                  src/utils/transformationReIm.m                          4
                                  src/utils/orthogonalizeGramSchmidt.m                   13
                                  src/utils/rcoordinatesStandardization.m                 8
                                  src/utils/pickPointsOnHypersphere.m                   (klein)
ODE rhs (forced ROMs)             src/utils/ode_2mDSSM_cartesian.m                       ~80
                                  src/utils/ode_2mDSSM_polar.m                          (~50)
Plotting                          src/customFigure.m                                    (lang)
                                  src/utils/plot2DSSM.m                                 (klein)
                                  src/utils/plotSSMandTrajectories.m                    (klein)
                                  src/utils/plotSSMWithTrajectories.m                   (klein)
                                  src/utils/plotReducedCoordinates.m                    (klein)
                                  src/utils/plotModalTrajectories.m                     (klein)
                                  src/utils/plotTrajectories.m                          (klein)
                                  src/utils/paperFigure.m                               (klein)
                                  src/utils/pinkgreen.m, purpleorange.m                 (klein)
                                  src/utils/RBF_interpolator.m                          (klein)
Misc                              src/utils/funToCell.m                                 (klein)
                                  src/utils/cocoSet.m                                   (klein)
                                  src/utils/contFRC.m                                   (klein)
                                  src/utils/convertLivescript2Markdown.m                (klein)
                                  src/utils/timeWeighting.m                             (klein)
                                  src/utils/unravelField.m                              (klein)
                                  src/utils/monitor_states.m                            (klein)
                                  src/utils/integrateTrajectories.m                     (klein)
                                  src/utils/ep_reduced_results.m                        (klein)
fastSSM (alternative)             fastSSM/fastSSM.m                                     122
                                  fastSSM/fastSSMMap.m                                  (~80)
                                  fastSSM/fastSSMplus.m                                 (~80)
                                  fastSSM/miniComputeFRC.m                              (~80)
```

---

## Anhang A: Aufgelöste Widersprüche

Beim Zusammenführen gegen die Repos auf Platte verifiziert:

1. **Python-Implementierung:** Die ältere Doku war als Spezifikation für eine *noch zu schreibende* Python-Reimplementation formuliert. Auf Platte existiert `/home/hz/Data/Attractor/SSMLearnPy/` bereits (verifiziert: Paketmodule `geometry/`, `reduced_dynamics/`, `utils/`). Behalten: SSMLearnPy als vorhandene Implementierung (§11.1); die Portierungstabelle bleibt als Feature-Referenz.
2. **Modellseiten-Doku:** Die neuere Datei verwies auf `SSMToolHaller_new.md` als "auszubauende Modellseite". Auf Platte existiert inzwischen `SSMToolHaller_merged.md` (zusammengeführte Fassung); Verweis entsprechend aktualisiert.
3. **`PFF.m`:** Alt-interner Widerspruch — Funktions-Index nannte es "Power-Flow-Filter", die Detail-Sektion den Peak-Finding-and-Fitting-Algorithmus nach Jin/Chen/Brake/Song. Gegen den Code-Header von `src/preprocessing/PFF.m:1-13` verifiziert: die Jin-et-al.-Beschreibung ist korrekt (behalten, §5.1.8); die "Power-Flow-Filter"-Zeile war falsch (verworfen).
4. **`getSSM.m`:** Alt-interner Widerspruch — Funktions-Index sagte "Extrahiert SSM-Parametrisierung aus IMInfo". Gegen den Code-Header verifiziert: `getSSM.m` "Generates a DynamicalSystem and an SSM for given tensors" (SSMtool-Bridge für $M,C,K,f_{nl}$). Behalten: Bridge-Beschreibung (§5.7, §2.2); Index-Zeile verworfen.
5. **`IC_nf`-Default:** Manual (`Practical_considerations.pdf`) sagt Default `0`; Code verifiziert: `IMDynamicsFlow.m:403`, `IMDynamicsMech.m:382`, `IMDynamicsFlowFractional.m:195` haben `'IC_nf',1`, nur `IMDynamicsMap.m:344` hat `'IC_nf',0`. Behalten: Code-Stand, dokumentiert in §10.8.
6. **Kein "neues SSMLearn":** Die neuere Datei ist autoritativ — `SSMTool_jain` ist kein SSMLearn-Nachfolger (gegen Repo-Struktur verifiziert: `DynamicalSystem`/`Manifold`/`SSM`-Klassen, `fnl_nonIntrusive.m`, `detect_resonant_modes.m` existieren wie beschrieben). Eingearbeitet in §2.
