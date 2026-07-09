# WORKFLOW_BTC_SSM.md — SSM-Analyse der BTC-Residuen + LPPL-Modell

## Ziel

Klaeren, ob die beobachtete ~1.97y-Komponente in BTC-Residuen eine unabhaengige Schwingung oder ein geometrischer Slave der ~3.86y-Halving-Mode ist. Methodik: identischer SSM-Workflow wie bei Wang 2-scroll (analyze_wang/), angewandt auf das LPPL-Modell (analytisch) und BTC-Daten (datengetrieben).

---

## Ergebnis (Kurzfassung)

**LPPL-Modell (analytisch):** 2w ist ein geometrischer Slave auf der 2D-SSM — identisches Ergebnis wie Wang.

| | Wang S4 | LPPL E2 | LPPL E3 |
|---|---|---|---|
| R2 | **0** | **0** | **0** |
| W2(2w) != 0? | ja | ja | ja |
| Staerkster 2w | u (->x) | u (->y1) | u (->y1) |
| Backbone | flach | flach | flach |
| sigma_out | -6 | +22 | +24 |

**BTC-Daten (SSMLearn):** PC3/PC4 R^2 = 0.50-0.64 (PARTIAL). Nicht so sauber wie Wang (0.978), aber konsistent mit einer verrauschten Version derselben Struktur.

**Schlussfolgerung:** Wenn BTC-Residuen die LPPL-Struktur haben, ist die ~1.97y-Komponente kein eigener Freiheitsgrad. Die Dynamik hat EINE Frequenz (Halving-Zyklus ~3.86y). Die Halb-Halving-Schwingung entsteht durch Mannigfaltigkeits-Kruemmung, nicht durch eine zweite Mode. Bitcoin waere linear-periodisch, nicht log-periodisch.

---

## LPPL-Modell

ODE (autonom, 3D, M=1 Polynom-Approximation, Grad 3):

    dy1/dt = Z_MIX*y1 + (1-Z_MIX)*y2 - Z_A*y2*z
    dy2/dt = alpha*y2 - gamma*y1^3 + Z_B*y1*z
    dz/dt  = -Z_C*z + Z_D*y1 + Z_E*y1*y2

Parameter (in lppl_system.py DEFAULT_PARAMS):
    alpha=-0.00074, gamma=0.003
    Z_A=0.008, Z_B=0.008, Z_C=0.0039, Z_D=1e-6, Z_E=2.0, Z_MIX=0.0002

M=1 Approximation: Das originale LPPL hat M=1.071 (fraktionale Potenz |y2|^0.071). Simulation mit M=1 vs M=1.071 zeigt kaum Unterschied. Die fraktionale Potenz ist dynamisch irrelevant — die ECHTE Nichtlinearitaet sind die bilinearen Kopplungsterme (identisch zu Wang).

---

## Strukturvergleich Wang vs LPPL

| | Wang | LPPL |
|---|---|---|
| z-Antrieb | x*y (bilinear) | Z_E*y1*y2 (bilinear, Z_E=2.0) |
| Kreuz-Kopplung | -y*z, +x*z | -Z_A*y2*z, +Z_B*y1*z |
| Extra NL | keine | -gamma*y1^3 (kubisch) |
| Fixpunkte | alle instabil (saddle-focus) | E2, E3 STABIL (stable_focus) |
| sigma_out | -6 (Master instabil) | +22/+24 (Master stabil, klassisch) |
| Perioden (linearisiert) | T=0.84 | T=35y (klein wegen schwacher Rueckstellkraft am Eq) |
| 2w Mechanismus | identisch | identisch |

---

## BTC-Daten

Quelle: ziel.csv (BTC-Preise), Log-Residuen, Delay-Embedding M=35, years=3.77, TAU=40d.

Beobachtete Perioden (SSMLearn-Fit):
- T_main = 3.84-3.85y (Halving-Zyklus)
- T_sub = 2.02y (Kandidat fuer 2:1 Harmonische)
- Detuning: 4.9% bei poly=1 (innerhalb Toleranz, Rauschen + Chaos erklaeren die Abweichung)

---

## CLI-Reihenfolge und Erkenntniskette

### Signalverarbeitung (01-05)

01_cli_precheck.py
  -> Embedding-Budget, PCA, fit_err, ODE_err, Eigenwerte, Resonanz
  -> ssm_dim=2: eine Mode (3.84y), ssm_dim=4: zwei Moden (3.85y + 2.02y)
  -> fit_err 0.37-0.54, ODE_err ~0.94 (Rauschen-dominiert, Smoothing offen)
  -> Spektralgap im Fit nicht verfuegbar (Slave langsamer als Master)

02_cli_harmonics.py
  -> PSD pro PC, dominante Frequenz + 2w Marker
  -> PC1-4 dominiert bei f~1/2.8y, PC6 zeigt 1/1.4y

03_cli_phase.py --ssm_dim 4 --poly_degree 1
  -> Phasenkopplung psi = 2*phi_main - phi_sub
  -> R = 0.673 (maessig gelockt, deutlich schwaecher als Wang 0.999)
  -> mean delta phi = 55 deg, median |delta phi| = 57 deg
  -> Drift 0.21 rad/y vs erwartet 0.15 rad/y (ratio 1.38)

04_cli_scaling.py --ssm_dim 4 --poly_degree 1
  -> |z_sub| vs |z_main|^2 Skalierung
  -> corr = +0.136, R^2 = 0.773 durch Ursprung
  -> CV main = 25.8% (identifizierbar)

05_cli_scan.py --dims 2,3,4 --polys 1,2,3
  -> Sweep-Tabelle ueber ssm_dim x poly_degree
  -> dim=2 poly=2: PC3 R^2=0.50 (PARTIAL)
  -> dim=2 poly=3: PC3 R^2=0.60 (PARTIAL)
  -> dim=4 poly=3: T kollabiert auf 1.44y (Overfitting)
  -> Ehrlicher Bereich: dim=2, poly=2-3

### SSM analytisch — LPPL-Modell (06-09)

06_cli_lppl_system.py
  -> LPPL Gleichgewichte, V2.6-Gates, Strukturvergleich zu Wang
  -> E1 Sattel, E2/E3 stabile Foki
  -> V2.6 offen (sigma_out +22/+24), V1.0 blockiert (kein M,C,K)

07_cli_lppl_spectral.py --eq E2
  -> Eigendec: Master lambda=-1.81e-4 +/- 4.88e-4i (T=35.2y)
  -> Slave lambda=-4.08e-3 (reell, 22x schneller)
  -> Keine Resonanzen, Ortho 9e-16

08_cli_lppl_whisker.py --eq E2
  -> Cohomologische Gleichung Ordnung 2: R2=0, W2!=0
  -> 2w ist REIN GEOMETRISCH (SSM-Kruemmung), nicht in der reduzierten Dynamik
  -> Staerkster 2w-Kanal: u (->y1, Preisabweichung), |W2|=30.9
  -> DC-Verschiebung dominiert in u: |W2|=178.5

09_cli_lppl_backbone.py --eq E2
  -> Backbone flach: omega(rho) = const, alpha = -1.81e-4 (stabil)
  -> Kein gamma (R2=0), Korrekturen erst bei Ordnung 3 (kubischer Term)
  -> Linearisierte Periode T=35.2y (nicht ~4y, weil kubische Rueckstellkraft
     am Gleichgewicht winzig ist; beobachtete ~4y kommt von grossen Amplituden)

### SSM datengetrieben — BTC-Residuen (10)

10_cli_ssm_learn.py --ssm_dim 2 --poly_degree 2
  -> Slave-Test: PC3 R^2=0.50, PC4 R^2=0.48 (PARTIAL)
  -> fit_err = 0.477
  -> T_main = 3.84y (robust ueber dim/poly Variationen)
  -> Kein klares SLAVED wie bei Wang (0.978), aber auch kein INDEPENDENT
  -> Interpretation: verrauschte Version derselben Struktur

---

## Vergleich der drei Systeme

| Metrik | Wang (analytisch) | LPPL (analytisch) | BTC (datengetrieben) |
|---|---|---|---|
| R2 (red. Dynamik Ord.2) | 0 | 0 | n/a (kein ODE) |
| W2 (2w Geometrie) | != 0 | != 0 | n/a |
| PC3 Slave R^2 | 0.978 | n/a | 0.50-0.64 |
| Phase-Lock R | 0.999 | n/a | 0.673 |
| sigma_out | -6 | +22 | n/a im Fit |
| fit_err | 0.038 | n/a | 0.37-0.48 |
| Backbone | flach | flach | n/a |
| Fazit | 2w = Slave (bewiesen) | 2w = Slave (bewiesen) | 2w = teilweise Slave (konsistent, verrauscht) |

---

## Bedeutung fuer BTC-Modellierung

Wenn die ~1.97y-Komponente ein geometrischer Slave ist:
- Die Dynamik hat EINE Frequenz (Halving-Zyklus ~3.86y)
- Ein Mehr-Frequenz-Fit ist nur in GEBUNDENER Form korrekt
  (Amplitude/Phase von 2w determiniert durch SSM-Geometrie, nicht frei)
- ssm_dim=2 + polynomialer Decoder ist das richtige Modell
- ssm_dim=4 zaehlt den 2w-Freiheitsgrad doppelt
- Bitcoin waere linear-periodisch, nicht log-periodisch

---

## Dateien in analyze_residuals/

| Datei | Typ | Inhalt |
|---|---|---|
| `common.py` | Core | identify_modes, smoothing |
| `constants.py` | Core | Defaults, Halvings |
| `data.py` | Core | BTC-Daten laden, Embedding |
| `amplitude.py` | Core | Moden-Amplituden, Support |
| `cycles.py` | Core | Halving-Segmente |
| `precheck.py` | Core | SSMLearn Precheck (Daten) |
| `lppl_system.py` | LPPL | ODE, Jacobian, Gleichgewichte, Hessians |
| `ssm_learn.py` | SSMLearn | Slave-Test |
| `01_cli_precheck.py` | CLI | BTC Daten-Precheck |
| `02_cli_harmonics.py` | CLI | PSD pro PC + Plot |
| `03_cli_phase.py` | CLI | Phasenkopplung + 4-Panel Plot |
| `04_cli_scaling.py` | CLI | Amplitudenskalierung + Plot |
| `05_cli_scan.py` | CLI | dim x poly Sweep + 4-Panel Plot |
| `06_cli_lppl_system.py` | CLI | LPPL Modell-Precheck |
| `07_cli_lppl_spectral.py` | CLI | LPPL Spektralanalyse |
| `08_cli_lppl_whisker.py` | CLI | LPPL Cohomologische Gl. (2w-Beweis) |
| `09_cli_lppl_backbone.py` | CLI | LPPL Backbone |
| `10_cli_ssm_learn.py` | CLI | BTC Slave-Test + Plots |
| `WORKFLOW_BTC_SSM.md` | Doku | Diese Datei |
| `WORKFLOW_OLD_VS_NEW.md` | Doku | Alt vs Neu Vergleich |

---

## Naechste Schritte (optional)

- [ ] ODE-Error Smoothing (dp/dt verrauscht, braucht Glaettung vor Vergleich mit R(p))
- [ ] LPPL Ordnung 3 Whisker (gamma1 aus kubischem Term, amplitudenabhaengige Backbone)
- [ ] Segment-lokale Slave-Tests (H2-H3, H3-H4, H4+ getrennt)
- [ ] Forward-Integration CLI mit Plot (reduzierte ODE integrieren, Trajektorie vergleichen)
- [ ] BTC 3-Variablen-Modell: direkte SSM-Analyse wenn LPPL-Simulation vorliegt

---

## Status

- [x] 01: Precheck (BTC Daten)
- [x] 02: Harmonics (PSD pro PC + Plot)
- [x] 03: Phase (Phasenkopplung + Plot)
- [x] 04: Scaling (Amplitudenskalierung + Plot)
- [x] 05: Scan (dim x poly Sweep + Plot)
- [x] 06: LPPL System (Gleichgewichte, V2.6-Gates)
- [x] 07: LPPL Spektral (Eigendec, Master/Slave, sigma=+22)
- [x] 08: LPPL Whisker (R2=0, W2!=0, 2w BEWIESEN)
- [x] 09: LPPL Backbone (flach, alpha=-1.81e-4 stabil)
- [x] 10: SSMLearn Slave-Test (PC3 R^2=0.50 PARTIAL)
- [x] Workflow-Zusammenfassung (diese Datei)
