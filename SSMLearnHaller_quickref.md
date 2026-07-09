# SSMLearn — Quickref

**Vollständige Doku:** `SSMLearnHaller.md` (1500+ Zeilen, mit allen Code-Verweisen, Original-Kommentaren und vollständiger Mathematik).
**Diese Datei:** TL;DR — wann SSMLearn zu benutzen ist, wie die Pipeline aussieht, welche Mathematik dahintersteht, was Achtung verdient.

---

## TL;DR

SSMLearn ist die **data-driven** Variante der SSM-Berechnung von Hallers Gruppe. Eingabe: Trajektorien-Daten (Zeitreihe oder Ensemble). Ausgabe: eine 2D- oder 4D-glatte Submanifold-Approximation des Phasenraum-Attraktors plus eine reduzierte Polynom-Dynamik darauf, optional mit Forced-Response-Vorhersage.

**Im Gegensatz zu SSMtool**: SSMLearn löst die Invarianz-Gleichung $DW(y)\,R(y) = f(W(y))$ **nicht**. Es fittet die Geometrie $W$ und die reduzierte Dynamik $R$ in zwei algorithmisch unabhängigen Regressionen an die Daten. Die SSM-Eigenschaft ist **näherungsweise** erfüllt, sofern die Daten tatsächlich in der Nähe einer echten SSM liegen.

---

## Wann SSMLearn benutzen?

| Situation | SSMLearn passend? |
|---|---|
| Du hast Zeitreihen-Daten, aber kein ODE-Modell | **Ja** — SSMLearn ist genau dafür gebaut |
| Du hast ein ODE-Modell mit explizitem Vektorfeld | Eher SSMtool (parametrization method, exakt) |
| Du willst die Sattel-Geometrie eines Attraktors vermessen | Ja, wenn 2D oder 4D Master-Subraum reicht |
| Du brauchst exakte Normal-Form-Koeffizienten zur Bifurkationsanalyse | Eher SSMtool |
| Forced-Response aus reinen Messdaten kalibrieren | Ja — `calibrateFRC.m`, `analyticalFRC.m`, `computeFRC.m` (COCO-Continuation) |
| Stark non-autonom (nicht-periodisch geforct) | Schwierig — SSMLearn unterstützt autonome SSMs sowie zeit-periodisch / quasi-periodisch geforcte (`forcedSSMROM.m`); rein nicht-periodisches Forcing nicht direkt |
| Power-Law-Nichtlinearitäten mit nicht-ganzzahligem Exponenten | Theoretisch fragwürdig — SSM-Theorie verlangt Glattheit am Fixpunkt |

---

## Pipeline (Kurz)

1. **Daten laden**: Trajektorien als Cell-Array `{t_i, y_i}`, jede Zeile eine Trajektorie.
2. **(Optional) Delay-Embedding**: `coordinatesEmbedding(yData, p)` mit $p \ge \lceil(2d+1)/n\rceil$ Delay-Kopien (Takens-Bound). $\tau$ ist nicht automatisch — User wählt selbst.
3. **Spektral-Auto-Detektion**: `SSM_startTime(...)` schätzt empirisch die SSM-Dimension aus dem Spektrogramm als $2 \times$ Anzahl dominanter Peaks.
4. **Tangentialraum + Geometrie**: `IMGeometry(...)` → dispatcht zu `IMGeometryGraphT0.m`. Constrained Polynom-Regression $W(y) = V_e \eta + H \phi(\eta)$ unter $V_e^\top H = 0$. Bei unbekanntem $V_e$: gemeinsame Optimierung von $V_e$ und $H$ via `fmincon` (`defineNonlinConstraints`, `alignmentLinearConstraint`).
5. **Reduzierte Dynamik**: `IMDynamicsFlow.m` (oder `Map`/`Mech`/`FlowFractional`/`Para*`). Polynom-Regression $\dot\eta = R(\eta)$ aus finite-difference-Ableitungen. Anschließend Normalform via `dynamicsCoordChangeNF.m` (Optimierung in Real/Imag-Aufspaltung des komplexen Koeff-Vektors).
6. **Backbone Curves**: `backboneCurves.m` plottet $\rho \mapsto (\omega(\rho), \alpha(\rho))$ für 2D-SSMs, oder `backboneSurfaces.m` für 4D.
7. **Forced Response**: `analyticalFRC.m` (analytisch für 2D), `computeFRC.m` (COCO-Continuation), `calibrateFRC.m` (aus Messdaten).
8. **Validierung**: `advect.m` für Forward-Predict, `computeTrajectoryErrors.m` für NMTE, `conjugacyErrorTrend.m` für Polynom-Order-Sweep.

---

## Mathematik (knapp)

**SSM-Definition** (Cabré-Fontich-de la Llave 2003, Haller-Ponsioen 2016): Die Spectral Submanifold $\mathcal{W}(E)$ ist die smoothest invariant manifold tangent to the spectral subspace $E$ am Fixpunkt $x_0$ von $\dot x = f(x)$. Existiert eindeutig glatt unter:

1. **Strikter Spektralgap**: $\max_{j\notin E}\mathrm{Re}\,\lambda_j < \min_{e\in E}\mathrm{Re}\,\lambda_e < 0$ (alle slaved schneller als alle master, alle stabil)
2. **Non-Resonanz**: $\langle m, \lambda_E\rangle \neq \lambda_j$ für $|m| \ge 2$, $j\notin E$
3. **Glattheit**: $f \in C^r$ mit $r$ groß genug (durch Spektral-Quotient bestimmt)

**Invarianz-Gleichung**: $DW(y)\,R(y) = f(W(y))$. Ordnungsweise gelöst durch
$$\mathcal{L}_k W_k(y) + V_e\,R_k(y) = G_k(y),\qquad \mathcal{L}_k W_k := DW_k(y)\cdot D_E\,y - A\,W_k(y)$$
wobei $G_k$ aus niedrigeren Ordnungen bekannt ist.

**Cohomological Operator** auf Monomial $y^m e_j$: Multiplikator $\langle m,\lambda_E\rangle - \lambda_j$. Resonanz iff dieser null wird.

**Aber: SSMLearn löst diese Gleichung NICHT.** Es fittet $W$ und $R$ separat per Regression an die Daten. Die SSM-Eigenschaft ist nur approximativ.

---

## Achtung / Limitationen

| Aspekt | Was |
|---|---|
| Fixpunkt | Muss am Ursprung liegen → Mittelwert vorher subtrahieren |
| SSM-Dimension | $d=2$ Standard, $d=4$ funktioniert, $d>4$ problematisch |
| Polynom-Ordnung | $M=3-5$ für Geometrie, $M_R=3$ für Dynamik. Höher → finite-time-blowup-Risiko |
| `fmincon`-Konvergenz | Sehr niedriges Default `MaxIter=100, MaxFunctionEvaluations=300`. Bei großen Problemen hochsetzen |
| `IC_nf`-Default | Variiert je IMDynamics-Variante: Flow/Mech/FlowFractional → 1, Map → 0. Manual sagt überall 0 (veraltet) |
| Smoothness | Daten sollten glatt sein. Power-Laws mit nicht-ganzzahligem Exponenten verletzen die SSM-Theorie-Voraussetzung |
| Cross-Validation | Über `indTrain`/`indTest`-Splits eingebaut. Für Zeitreihen besser separates Out-of-Sample-Window |
| Beispiele | 20 Beispiele in `examples/`, alle mechanisch/strukturmechanisch/CFD. KEIN Finanz-, Klima- oder Bio-Beispiel im Repo |

---

## Beste Vorlagen für Python-Reimplementation

- **`fastSSM/fastSSM.m`** (122 Zeilen, 2D-SSM, kubisch) — minimaler Stack mit explizit ausgeschriebenen Cohomological-Equation-Lösungen für die kubische 2D-Single-Mode-Form (Z. 45-67). Beste Vorlage. Warnung: keine Resonanz-Prüfung, keine Small-Denominator-Kontrolle, fest gewählte Gauge.
- **`src/geometry/IMGeometryGraphT0.m`** für die Geometrie-Constraint-Optimierung
- **`src/reduceddynamics/IMDynamicsFlow.m`** für die reduzierte Dynamik + Normalform
- **`src/reduceddynamics/utils/polarNormalForm.m`** für Polar-Konvertierung und Backbone-Funktion-Handles

---

## Decision-Tree (knapp)

```
Hast du eine ODE oder nur Daten?
├── ODE explizit, autonom, mech (M,C,K,f), 2D-Master → SSMtool
└── Daten (Trajektorien) oder ODE als Black-Box-Simulator → SSMLearn

Mit SSMLearn:
├── 1 Trajektorie, 1 Observable → erst delay-embedden via coordinatesEmbedding
├── 1 Trajektorie, n Observables → direkt benutzbar
└── Ensemble von Trajektorien → noch besser (Daten überdecken Subraum besser)

SSM-Dimension wählen:
├── Limit-Cycle-artiger Attraktor → d=2
├── Quasi-periodisch (2 Frequenzen) → d=4
└── Stark mehrfach-periodisch → d>4 problematisch

Polynom-Ordnung:
├── Mech-System mit klarer Linearität → M=3
├── Stark nichtlinear → M=5, dann conjugacyErrorTrend für optimale Wahl
└── Bei Blowup → M reduzieren

Validation:
├── advect.m + computeTrajectoryErrors.m → NMTE
├── conjugacyErrorTrend → Polynom-Order-Sweep
└── Train/Test split → indTrain/indTest
```

---

## Referenzen

- Cabré, P., Fontich, E., de la Llave, R., 2003 — Existenzsatz für SSMs (parametrisierungs-methodisch)
- Haller, G., Ponsioen, S., 2016 — *Nonlinear Dyn.* 86: 1493–1534 — SSM-Theorem für mechanische Systeme
- Cenedese, M., Axås, J., Bäuerlein, B., Avila, K., Haller, G., 2022 — *Nat. Commun.* 13: 872 — SSMLearn-Hauptpaper
- Axås, J., Cenedese, M., Haller, G., 2023 — *Nonlinear Dyn.* 111: 7941 — fastSSM
- Axås, J., Haller, G., 2023 — *Nonlinear Dyn.* 111: 22079 — Delay-Embedded SSMs
- Liu, B., Axås, J., Haller, G., 2024 — *Chaos* 34: 033140 — Inertial Manifolds als SSMs (chaotische Dynamik)

---

**Detaillierte Doku in `SSMLearnHaller.md` (Sektionen 1–13 + Anhang A).**
