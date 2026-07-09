# SSMtool — Quickref

**Vollständige Doku:** `SSMToolHaller.md` (1080+ Zeilen, mit allen Code-Verweisen, Manual-Zitaten und vollständiger Mathematik der Parametrization Method).
**Diese Datei:** TL;DR — wann SSMtool zu benutzen ist, wie die Pipeline aussieht, welche Mathematik dahintersteht, was Achtung verdient.

---

## TL;DR

SSMtool ist die **ODE-basierte** Variante der SSM-Berechnung von Hallers Gruppe. Eingabe: ein autonomes mechanisches System $M\ddot u + C\dot u + K u + f_{nl}(u) = 0$ als symbolisches/numerisches Modell. Ausgabe: eine 2D-SSM als formale Taylor-Reihe um den Fixpunkt, die reduzierte Polar-Dynamik darauf, und Backbone-Kurven für die nichtlinearen Eigenmoden.

**Im Gegensatz zu SSMLearn**: SSMtool **löst die Invarianz-Gleichung exakt** ordnungsweise per Parametrization Method (Cabré-Fontich-de la Llave). Die SSM-Eigenschaft ist algorithmisch garantiert bis zur jeweiligen Polynomordnung.

---

## Wann SSMtool benutzen?

| Situation | SSMtool passend? |
|---|---|
| Du hast ein ODE-Modell als $M, C, K, f$ | **Ja** — exakt der Eingabeformat |
| Du hast eine generelle First-Order-ODE $\dot x = f(x)$ | **Nein** — SSMtool V1.0 verlangt mechanische Form |
| Du brauchst exakte Normalform-Koeffizienten | Ja — sehr stark hier |
| Du hast nur Zeitreihen-Daten | **Nein** — SSMLearn benutzen |
| Master-Subraum > 2D | **Nein** — SSMtool V1.0 ist 2D hardcoded |
| Forced Response (zeit-periodisch / harmonisch) | Ja, aber nur über Addendum_Isolas — separates Skript-Set ohne GUI-Integration; kein V1.0-Pfad |
| Forced Response (nicht-periodisch) | **Nein** — keine Unterstützung in V1.0 oder Addendum |
| Power-Law mit nicht-ganzzahligem Exponent | **Nein theoretisch** (Cabré-Fontich-de la Llave verlangt Analytizität); SSMtool ruft trotzdem `taylor()` auf, ohne Garantie |
| Fixpunkt nicht am Ursprung | Vorher koordinatentransformieren |
| Fixpunkt instabil (Re λ ≥ 0) | **Nein im nicht-konservativen Branch** (`compute_subspace.m:40-43` bricht ab); im konservativen Branch werden imaginäre Eigenwerte zugelassen |

---

## Pipeline (Kurz)

1. **GUI starten**: `SSM` → `SSM.m` öffnet das Hauptfenster
2. **System einladen**: drei Optionen — Drop-Down-Predefined, GUI-Input-Felder, oder `.mat`-Datei mit den Variablen `M, C, K, f, sys.spv` (Phasenraum-Vektor)
3. **`Analyze`-Button**: Linearisierung + Eigendecomposition (`compute_subspace.m`); zeigt Eigenwerte und Spektral-Quotient $\sigma(E)$
4. **Master-Subraum-Selektion**: User klickt zwei komplex-konjugierte Eigenwerte an, drückt `Select`. SSMtool prüft Non-Resonanz (`check_res.m`, Toleranzen $10^{-4}$ extern, $5\cdot 10^{-2}$ intern)
5. **`Compute SSM`-Button**: `compute_SSM.m` läuft. Symbolische Taylor-Entwicklung der Nichtlinearität (`taylor()`), Aufbau der $G_m$-Matrizen via `matGV2.m`, ordnungsweise Lösung der cohomologischen Gleichungen via `kronGK.m`, `kronKR.m`, `kronGK1n.m`. Output: $K_m$ (SSM-Polynom-Koeffizienten) und $R_m$ (reduzierte Dynamik) bis zur gewählten Ordnung. Schreibt nach `Data/run_<timestamp>/`
6. **Backbone-Curve**: `plot_SSM.m` rendert die 3D-SSM, abgeleitete Polar-Dynamik liefert $\rho \mapsto \omega(\rho), \alpha(\rho)$
7. **Trajektorien-Vergleich**: `int_dyn.m` (volle Dynamik) vs `int_red_dyn.m` (reduzierte Polar-Dynamik); Invarianz-Fehler via `measure_inv_autonomous.m`
8. **Forced Response (separat über Addendum_Isolas)**: `mech_sys_isola.m` baut die Order-1-Forced-Invariance-Equation, Polynom-Helfer aus `Addendum_Isolas/core/` (`poly_product_DWR.m`, `poly_power.m`, `man2cor.m`, `genlexd.m`, `nch.m`)

---

## Mathematik (knapp)

**SSM-Definition** (Cabré-Fontich-de la Llave 2003, Haller-Ponsioen 2016): wie in SSMLearn. Existenz/Eindeutigkeit unter striktem Spektralgap, Non-Resonanz und Glattheit.

**Invarianz-Gleichung**:
$$DW(y)\,R(y) = f(W(y))$$

**Parametrization Method (Cabré-Fontich-de la Llave)**: ordnungsweise Lösung mit
$$\mathcal{L}_k W_k(y) + V_e\,R_k(y) = G_k(y),\qquad \mathcal{L}_k W_k := DW_k(y)\cdot D_E\,y - A\,W_k(y)$$
wobei $G_k$ aus $W_2,\dots,W_{k-1}$, $R_2,\dots,R_{k-1}$ und den Taylor-Koeffizienten $f_2,\dots,f_k$ bekannt ist.

**Cohomological Operator** auf Monomial $y^m e_j$: Multiplikator $\langle m,\lambda_E\rangle - \lambda_j$. Für slaved $j$: $h_{k,m,j} = G_{k,m,j} / (\langle m,\lambda_E\rangle - \lambda_j)$.

**SSMtool implementiert das Kronecker-basiert**: $W$ wird als Liste von Kronecker-Tensoren $K_m$ repräsentiert, $G_m$ analog. Die Cohomological-Iteration läuft über `kronKR.m` (LHS) und `kronGK.m` (RHS) ordnungsweise bis zur User-spezifizierten max Ordnung (≤ 50).

**Normal-Form**: Resonante Terme bleiben in $R$, nicht-resonante werden in $W$ verschoben. Nach Konvergenz wird $R$ in Polar-Form übersetzt: $\dot\rho = \alpha(\rho)\rho$, $\dot\theta = \omega(\rho)$, woraus die Backbone-Curve ablesbar ist.

---

## Achtung / Limitationen

| Aspekt | Was |
|---|---|
| Eingabeform | NUR mechanisches System $M, C, K, f$. Kein generelles $\dot x = f(x)$. |
| SSM-Dimension | NUR 2D hardcoded (`compute_SSM.m:47`, `K1 = [eye(2); zeros(...)]`) |
| Master-Subraum | NUR komplex-konjugiertes Paar oder zwei reelle Eigenwerte (`orderT.m:81-87`) |
| Stabilität | Im `~conservative`-Branch ist asymptotische Stabilität Pflicht (`compute_subspace.m:40-43` bricht sonst ab) |
| Konservativer Branch | Lässt imaginär-achsen-Eigenwerte zu (`compute_subspace.m:12-26`), Resonanzbehandlung separat in `check_res.m:65-72` |
| Power-Law-Glattheit | NIRGENDS im Code geprüft. CFdlL-Voraussetzung — User in der Pflicht |
| Forcing | Im V1.0 NICHT unterstützt; nur über separates Addendum_Isolas |
| Parallel-Toolbox | Multiple Worker NICHT nötig (Manual S. 2.5), aber Code ruft `gcp/parcluster/parpool/spmd` direkt auf — Toolbox MUSS installiert sein |
| Fixpunkt | Muss am Ursprung liegen, sonst vorher transformieren |
| Resonanz-Toleranzen | Hardcoded: extern $10^{-4}$ (`check_res.m:24`), intern $5\cdot 10^{-2}$ (`check_res.m:45`, `check_higher_res.m:19`) |
| Maximale Ordnung | 50 (`SSM.m:575`) |
| Lokalität | SSM ist eine **formale Reihe** um den Fixpunkt. Konvergenzradius unbekannt; in der Praxis einige Prozent der Maximalamplitude für lightly-damped Mech-Systeme |
| Side-effects | Schreibt `cs.mat`, `cluster_info.mat`, `Data/run_<ts>/`, `R_sub_function.m`, `system_function.m` ins Working-Directory; jeder Lauf überschreibt |

---

## Decision-Tree (knapp)

```
Hast du eine ODE oder Daten?
├── Daten (Trajektorien) → SSMLearn
└── ODE → SSMtool oder Eigenbau

Mit ODE:
├── Mech-Form M,C,K,f explizit + Fixpunkt = 0 + asymptotisch stabil + 2D Master → SSMtool V1.0 GUI direkt
├── First-order ODE oder 3D+ Phasenraum oder instabiler Fixpunkt → SSMtool V1.0 funktioniert nicht
│   ├── Vereinfachen: Halvings raus, Power-Law runden, Damping konstant
│   └── Eigenbau mit Polynom-Helfern aus Addendum_Isolas/core/
└── Konservativ (Hamiltonian, ohne Damping) → konservativer Branch von compute_subspace.m

Polynom-Ordnung:
├── Lightly-damped Mech → Order 5-10
├── Stark gedämpft → Order 3-5
└── Über 20 → wahrscheinlich Konvergenz-Probleme

Forced Response:
├── Periodisch harmonisch → Addendum_Isolas Skripte
└── Nicht-periodisch → weder V1.0 noch Addendum
```

---

## Beste Vorlagen für Eigenbau / Python-Reimplementation

- **`Addendum_Isolas/core/poly_product_DWR.m`** — generisches Polynom-Produkt für die Invarianz-Gleichung
- **`Addendum_Isolas/core/poly_power.m`** — symbolische Polynom-Potenzen
- **`Addendum_Isolas/core/man2cor.m`** — Konvertierung Mannigfaltigkeit ↔ physikalische Koordinaten
- **`Addendum_Isolas/core/genlexd.m`** — lexikographische Multi-Index-Generierung
- **`Addendum_Isolas/core/nch.m`** — n-choose-k-Helfer
- **`Addendum_Isolas/example_shawpierre/ex_SP_W0_3_W1_0.m`** — Top-Level-Wrapper, am besten als Vorlage für ein eigenes Skript mit `ndof_spv = 3` (statt 4) und `nvar = 2`

---

## Anwendbarkeit auf das LPPL-System (`lpplattr02_ode.py`)

**Problem 1**: Nicht-mechanische Struktur. Das System ist 3D in $(y_1, y_2, z)$ ohne kanonische $(u, \dot u)$-Aufspaltung. SSMtool V1.0 nicht direkt anwendbar.

**Problem 2**: Power-Law $|y_2|^{M-1}$ mit $M=1.071$. Nicht analytisch am Ursprung. CFdlL-Voraussetzung verletzt. Theoretisch kein SSM-Existenzbeweis. SSMtool würde es trotzdem versuchen, aber ohne Garantie. **Workaround**: $M$ auf ganze Zahl runden, oder durch glattes Ersatzmodell ersetzen wie $y_2(\epsilon^2 + y_2^2)^{(M-1)/2}$.

**Problem 3**: Linearisierung am Ursprung hat Eigenwerte $\lambda_1 = 0$, $\lambda_2 = Z_{MIX} = 2\cdot 10^{-4} > 0$, $\lambda_3 = -Z_C = -3.9\cdot 10^{-3}$. Zwei Eigenwerte mit nicht-negativem Realteil → SSMtool im nicht-konservativen Branch bricht sofort ab.

**Problem 4**: Halvings, $1/t$-Damping, Sign-OU brechen die Autonomie. SSMtool V1.0 hat keine non-autonomous Unterstützung; selbst Addendum_Isolas behandelt nur periodisches Forcing.

**Verdikt**: SSMtool V1.0 für das LPPL-System **nicht direkt benutzbar**. Optionen: (a) Eigenbau mit Polynom-Helfern aus `Addendum_Isolas/core/` für ein massiv vereinfachtes autonomes Polynom-Modell; (b) SSMLearn als data-driven Alternative; (c) klassische Slow-Manifold-/Center-Manifold-Methoden, die zur LPPL-Slow-Fast-Struktur passen.

---

## Referenzen

- Cabré, P., Fontich, E., de la Llave, R., 2003 — Parametrisierungs-Methode für invariante Mannigfaltigkeiten
- Haller, G., Ponsioen, S., 2016 — *Nonlinear Dyn.* 86: 1493–1534 — SSM-Existenzsatz
- Ponsioen, S., Pedergnana, T., Haller, G., 2018 — *J. Sound Vib.* 420: 269–295 — SSMtool-Hauptpaper
- Manual: `SSMtool/SSMtool_manual.pdf` (15 Seiten)
- Repo: `haller-group/SSMtool` (V1.0 + Addendum_Isolas)

---

**Detaillierte Doku in `SSMToolHaller.md` (Sektionen 1–9 + Anhang A).**
