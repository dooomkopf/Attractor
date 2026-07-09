# WORKFLOW_WANG_SSM.md — SSM-Analyse des Wang 2-Scroll-Attraktors

## Ziel

Den Wang 2-Scroll-Attraktor mit dem SSM-Framework (Haller et al.) analysieren und klaeren, ob die beobachtete 2w-Komponente eine echte Harmonische (Slave auf der SSM) oder eine unabhaengige Mode ist.

Spaeteres Ziel: Methodik auf das 3-Variablen-BTC-Modell (LPPL-Attraktor, y1/y2/z) uebertragen.

---

## Wang-System

ODE (autonom, 3D, polynomial Grad 2):

    dx/dt =  a*(x - y) - y*z
    dy/dt = -b*y + x*z
    dz/dt = -c*z + d*x + x*y

Defaultparameter: a=3.5, b=6.4, c=5.0, d=0.6

Quelldateien: `system.py` (wang_rhs, wang_jacobian, equilibria, shifted_quadratic_terms)

---

## Ergebnis (Kurzfassung)

**2w ist ein geometrischer Slave auf der 2D SSM — kein unabhaengiger Freiheitsgrad.**

Belegt durch zwei unabhaengige Wege:

| Methode | Ergebnis |
|---|---|
| Analytisch (Parametrisierungsmethode, Ord. 2) | W2 != 0 in allen 3 Observablen, R2 = 0 → 2w ist rein geometrisch |
| Datengetrieben (SSMLearnPy, poly=2) | PC3 R^2 = 0.978 → PC3 ist Polynom von (PC1, PC2), fit_err = 3.8% |

Die reduzierte Dynamik auf der SSM kennt nur EINE Frequenz (w). Der 2w-Anteil entsteht durch die Kruemmung der Mannigfaltigkeit (W2), nicht durch die Dynamik (R2=0). Die Backbone-Kurve ist flach — keine amplitudenabhaengige Frequenzverschiebung bei Ordnung 2.

---

## Bedeutung fuer BTC

Wenn BTC-Residuen dieselbe Struktur haben (~1.97y als geometrischer Slave von ~3.86y):
- Die Dynamik hat EINE Frequenz (Halving-Zyklus ~3.86y)
- Die Halb-Halving-Schwingung (~1.97y) ist eine geometrische Konsequenz, kein eigener Freiheitsgrad
- Ein Mehr-Frequenz-Fit ist nur in GEBUNDENER Form korrekt (Amplitude/Phase von 2w determiniert durch SSM-Geometrie)
- Bitcoin waere linear-periodisch, nicht log-periodisch
- Der Test: harmonic_test_slave.py auf BTC-Residuen (R^2 von PC3 aus Polynom(PC1,PC2))

---

## Signalverarbeitung (vor SSM-Analyse)

| CLI-Tool | Was | Kernergebnis |
|---|---|---|
| `cli_precheck.py` | Gleichgewichte, Eigenwerte, Stabilitaet, V2.6-Gates | 5 Eq., Scroll-Center S4/S5, sigma_out=-6, V2.6 verfuegbar |
| `cli_harmonics.py` | Spektrale Analyse (Welch PSD) pro Kanal | w in x/y/pc1/pc2, 2w in z/pc3. f0=4.70, T=0.213 |
| `cli_phase.py` | Phasenkopplung (Bandpass + Hilbert) | R=0.999, median |dphi|=147.4 deg |
| `cli_scaling.py` | Amplitudenskalierung A_2w ~ c*A_w^2 | R^2=0.9998, aber CV nur 5% (kaum Modulation) |
| `cli_scan.py` | b-Parameter-Sweep (5.6..7.4) | Harmonische stabil ueber breiten b-Bereich |

---

## SSM-Analyse: Analytischer Pfad (SSMTool V2.6 Stil)

### Gleichgewichte und Scroll-Zentren

Attraktor besucht nur S4 und S5 (z < 0 Region, je 50% der Zeit).
S2, S3 (z > 0) werden nie erreicht.

| Eq | Typ | lambda_real | lambda_complex | T_osc | sigma_out | Scroll |
|---|---|---|---|---|---|---|
| S1 | real_saddle | -6.4, -5.0, +3.5 | - | - | - | - |
| S2 | saddle_focus | -10.78 | +1.44 +/- 5.09i | 1.234 | -7 | nein |
| S3 | saddle_focus | -11.89 | +2.00 +/- 5.21i | 1.205 | -5 | nein |
| S4 | saddle_focus | -11.19 | +1.64 +/- 7.46i | 0.842 | -6 | **ja** |
| S5 | saddle_focus | -11.80 | +1.95 +/- 7.58i | 0.829 | -6 | **ja** |

### SSM-Pfad-Entscheidung

| Pfad | Status | Grund |
|---|---|---|
| SSMTool V1.0 | blocked | braucht stabilen Fokus + mechanische Form |
| SSMTool V2.6 | **verfuegbar** | First-Order nativ, instabile SSM moeglich, |sigma_out| >= 5 |
| SSMLearn | **verfuegbar** | voller Zustand (x,y,z), kein Embedding noetig |

### Step 1: Precheck (`cli_precheck.py`)

- Alle Foki instabil (Re > 0) — SSM ist instabile Mannigfaltigkeit (Ausstroem-Richtung)
- |sigma_out| = 5..7 → exzellenter Spektralgap (Slave zerfaellt viel schneller als Master waechst)
- Keine Nah-Resonanzen bis Ordnung 4

### Step 2: Verschobene DGL (`cli_ssm_system.py --eq S4`)

- A-Matrix (Jacobian) + drei Hesse-Matrizen H0, H1, H2
- Quadratische Terme: du=-v*w, dv=+u*w, dw=+u*v
- Verschiebungs-Verifikation: Fehler = 1e-14 (exakt, Wang ist rein quadratisch)

### Step 3: Spektralanalyse (`cli_ssm_spectral.py`)

- Master: lambda = +1.64 +/- 7.46i (komplexes Paar, T=0.84)
- Slave: lambda = -11.19 (reell, schnell zerfallend)
- Nur EIN Schwingungspaar → 2w KANN keine eigene Mode sein
- sigma_out = -6, negativ weil Master instabil — Trennung ist dadurch sogar staerker
- Orthonormalitaet W^H V = I auf 2e-16

### Step 4: Cohomologische Gleichung Ordnung 2 (`cli_ssm_whisker.py`) — DER BEWEIS

Multi-Indizes bei Ordnung 2:
- (2,0) → Frequenz 2w (zweite Harmonische)
- (1,1) → Frequenz 0 (DC, Mittelpunktsverschiebung)
- (0,2) → Frequenz -2w (konjugiert)

Ergebnis:
- **R2 = 0** bei ALLEN Multi-Indizes → keine nichtlineare Korrektur der reduzierten Dynamik
- **W2 != 0** bei allen → die SSM-Mannigfaltigkeit ist gekruemmt

2w-Anteil pro Observable (|W2[(2,0)]|):
- u (→x): 2.77e-02 (staerkster 2w-Kanal)
- v (→y): 1.31e-02
- w (→z): 1.42e-02

Interpretation: Die Dynamik auf der SSM ist EINE einfache Spirale (Frequenz w). Die 2w-Oszillation in den Observablen entsteht durch die Kruemmung der 2D-Flaeche im 3D-Raum. Analogie: Wendeltreppe mit gewellten Stufen — die Gehgeschwindigkeit ist konstant (w), aber die Hoehe schwankt bei 2w wegen der Stufenform (W2).

### Step 5: Backbone (`cli_ssm_backbone.py`)

- alpha_0 = +1.64 (instabil, Spirale nach aussen)
- omega_0 = 7.46 rad/t (T=0.84)
- gamma = 0 (keine amplitudenabhaengige Frequenzkorrektur bei Ord. 2)
- Backbone flach: omega(rho) = const
- Korrekturen erst ab Ordnung 3 (Komposition von F2 mit W2)

---

## SSM-Analyse: Datengetriebener Pfad (SSMLearn)

### Step 6: SSMLearnPy-Fit (`cli_ssm_learn.py`)

- Observable: voller 3D-Zustand (x,y,z), kein Delay-Embedding
- PCA-Varianz: PC1=59.7%, PC2=33.6%, PC3=6.6%
- ssm_dim=2, poly_degree=2

Ergebnis:
- **fit_err = 3.85e-02** → Decoder rekonstruiert 3D-Trajektorie mit 96% Genauigkeit
- **Gefittete Eigenwerte**: lambda = -0.00005 +/- 30.93i (T=0.203, marginal stabil)
- **Slave-Test PC3: R^2 = 0.978 → SLAVED**
- PC3 ist zu 97.8% durch Polynom(PC1, PC2) bestimmbar → kein eigener Freiheitsgrad

Konsistenz mit analytischem Pfad:
- Analytisch: W2 != 0, R2 = 0 → 2w geometrisch, nicht dynamisch ✓
- Datengetrieben: PC3 ~ polynom(PC1,PC2) mit R^2=0.978 → dritte Koordinate ist Slave ✓
- Beide Wege bestaetigen: der Attraktor lebt auf einer 2D-Flaeche

Hinweis ssm_dim:
- ssm_dim=2 ist die einzige sinnvolle Wahl (3 Observable, 1 Slave-PC zum Testen)
- ssm_dim >= 3 → kein Slave-Test moeglich (alle PCs sind Master)
- poly_degree=2 matcht die wahre NL (Wang ist quadratisch); hoehere Grade overfitten

---

## Dateien in analyze_wang/

| Datei | Typ | Inhalt |
|---|---|---|
| `system.py` | Core | Wang-ODE, Jacobian, Gleichgewichte |
| `simulate.py` | Core | Trajektorie integrieren |
| `constants.py` | Core | Default-Parameter |
| `harmonics.py` | Signalverarbeitung | Welch PSD, Harmonik-Detektion |
| `phase.py` | Signalverarbeitung | Bandpass, Hilbert, Phasenkopplung |
| `scaling.py` | Signalverarbeitung | Amplitudenskalierung |
| `scan.py` | Signalverarbeitung | b-Parameter-Sweep |
| `precheck.py` | SSM analytisch | Gleichgewichte, Gates, V2.6-Check, Scroll-Zuordnung |
| `ssm_system.py` | SSM analytisch | Verschobene First-Order-Form, Hessians |
| `ssm_spectral.py` | SSM analytisch | Eigendec, Master/Slave-Wahl, Resonanz |
| `ssm_whisker.py` | SSM analytisch | Cohomologische Gl. Ord. 2, W2/R2 |
| `ssm_backbone.py` | SSM analytisch | Polarform, Backbone |
| `ssm_learn.py` | SSM datengetrieben | SSMLearnPy-Fit, Slave-Test |
| `cli_precheck.py` | CLI | → precheck |
| `cli_harmonics.py` | CLI | → harmonics + Plot |
| `cli_phase.py` | CLI | → phase + Plot |
| `cli_scaling.py` | CLI | → scaling + Plot |
| `cli_scan.py` | CLI | → scan + Plot |
| `cli_ssm_system.py` | CLI | → ssm_system |
| `cli_ssm_spectral.py` | CLI | → ssm_spectral |
| `cli_ssm_whisker.py` | CLI | → ssm_whisker |
| `cli_ssm_backbone.py` | CLI | → ssm_backbone |
| `cli_ssm_learn.py` | CLI | → ssm_learn + Plot |
| `WORKFLOW_WANG_SSM.md` | Doku | Diese Datei |
| `todo.txt` | Doku | CLI-Reihenfolge + Erkenntniskette |

---

## Referenzdateien

| Datei | Zweck |
|---|---|
| `/home/hz/Data/Attractor/SSMToolHaller_new.md` | SSMTool V2.6 Workflow-Doku |
| `/home/hz/Data/Attractor/SSMLearnHaller_new.md` | Einordnung SSMLearn vs SSMTool_jain |
| `/home/hz/Data/Attractor/SSMToolHaller.md` | SSMTool V1.0 Langdoku |
| `/home/hz/Data/Attractor/SSMLearnHaller.md` | SSMLearn Langdoku |
| `/home/hz/Data/Attractor/SSMToolHaller_quickref.md` | V1.0 Kurzreferenz |
| `/home/hz/Data/Attractor/SSMLearnHaller_quickref.md` | SSMLearn Kurzreferenz |
| `/home/hz/Data/Attractor/SSMTool_jain/` | Referenz-Repo (MATLAB) |
| `/home/hz/Data/Attractor/analyze_residuals/` | BTC-Residuen-Pipeline |

---

## Status

- [x] Step 0: WORKFLOW_WANG_SSM.md
- [x] Step 1: Precheck V2.6 Update (sigma_out, Resonanz-Scan, 3-Pfad-Summary, Scroll-Markierung)
- [x] Step 2: ssm_system.py + cli_ssm_system.py (shifted first-order form, Hessians, verification=0)
- [x] Step 3: ssm_spectral.py + cli_ssm_spectral.py (Eigendec, V/W normiert, Master/Slave, sigma=-6, keine Resonanzen)
- [x] Step 4: ssm_whisker.py + cli_ssm_whisker.py (cohomologische Gl. Ord.2, W2 non-zero, R2=0, 2w BEWIESEN)
- [x] Step 5: ssm_backbone.py + cli_ssm_backbone.py (Polarform, Backbone flach bei Ord.2, R2=0, gamma=0)
- [x] Step 6: ssm_learn.py + cli_ssm_learn.py (SSMLearnPy, fit_err=3.8%, PC3 R^2=0.978 SLAVED)
- [x] Step 7: Zusammenfassung (diese Datei)

## Naechste Schritte (optional)

- [ ] Ordnung 3 Whisker (gamma1 Koeffizient fuer amplitudenabhaengige Backbone)
- [ ] BTC-Residuen: Slave-Test mit SSM-Interpretation
- [ ] BTC 3-Variablen-Modell (LPPL): analoge SSM-Analyse wie Wang
