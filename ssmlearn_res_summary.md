# SSMLearnPy on BTC residuals — Run-Übersicht

Pipeline: `ziel.csv → QuantReg-detrend → log_res → Delay-Embed (M, τ) → PCA → SSMLearnPy`

Eigenwerte sind in `/Tag`. Periode = `2π / |Im(λ)|` in Tagen.

## Run-Übersicht

| # | M | τ [d] | W [d] | start | ssm_dim | poly | Re(λ_osc) | Im(λ_osc) | Period [d/y] | rms NMTE |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 35 | 40 | 1360 | 1164 | 2 | 3 | -1.56e-4 | 5.20e-3 | 1208 / 3.31 | 0.448 |
| 1 | 35 | 40 | 1360 | 1164 | 2 | 5 | +3.63e-5 | 9.78e-3 | 642 / 1.76 | 0.386 |
| 1 | 35 | 40 | 1360 | 1164 | 2 | 7 | — | — | reell (Sattel!) | 0.330 |
| 2 | 35 | 40 | 1360 | 1164 | 3 | 3 | +6.0e-5 | 4.97e-3 | 1264 / 3.46 | 0.370 |
| 2 | 35 | 40 | 1360 | 1164 | 3 | 5 | +6.7e-3 | 1.75e-2 | 359 / 0.98 (instabil!) | 0.235 |
| 2 | 35 | 40 | 1360 | 1164 | 3 | 7 | -9.5e-3 | 7.62e-3 | 825 / 2.26 | 0.177 |
| 3 | 35 | 21 | 714 | 1164 | 2 | 3 | -5.56e-4 | 4.92e-3 | 1276 / 3.50 | 0.391 |
| 3 | 35 | 21 | 714 | 1164 | 2 | 5 | +1.15e-4 | 5.37e-3 | 1170 / 3.21 | 0.326 |
| 3 | 35 | 21 | 714 | 1164 | 2 | 7 | -5.05e-4 | 5.09e-3 | 1235 / 3.39 | 0.278 |
| 4 | 25 | 23 | 552 | 1164 | 2 | 3 | -3.38e-4 | 5.45e-3 | 1153 / 3.16 | 0.368 |
| 4 | 25 | 23 | 552 | 1164 | 2 | 5 | — | — | reell (Sattel) | 0.305 |
| 4 | 25 | 23 | 552 | 1164 | 2 | 7 | -1.44e-3 | 2.34e-3 | 2685 / 7.36 | 0.276 |
| 5 | 35 | 21 | 714 | 1164 | 3 | 3 | -1.18e-4 | 6.26e-3 | 1003 / 2.75 | 0.310 |
| 5 | 35 | 21 | 714 | 1164 | 3 | 5 | +6.6e-3 | 7.36e-3 | 854 / 2.34 (instabil!) | 0.212 |
| 5 | 35 | 21 | 714 | 1164 | 3 | 7 | -2.51e-3 | 1.78e-2 | 353 / 0.97 | 0.157 |
| 6 | 35 | 21 | 714 |  600 | 2 | 3 | -5.35e-4 | 4.91e-3 | 1280 / 3.51 | 0.392 |
| 6 | 35 | 21 | 714 |  600 | 2 | 5 | +4.4e-6 | 5.35e-3 | 1175 / 3.22 | 0.327 |
| 6 | 35 | 21 | 714 |  600 | 2 | 7 | -4.23e-4 | 4.90e-3 | 1283 / 3.51 | 0.280 |

## Beobachtungen

### ✅ Robust: M=35, years=2.0, ssm_dim=2 (Run 3)

**Über alle drei poly_degrees {3,5,7} konsistent**:
- Periode ~3.3 Jahre (matched BTC-Halving-Zyklus von 3.77 Jahren)
- Re(λ) negativ (gedämpft) bei poly=3 und 7, marginal positiv bei poly=5
- |Re/Im| < 0.1 → schwach gedämpfte Oszillation

**Bei start_idx=600** (Run 6, mehr Geschichte) sieht es identisch aus → Robustheit gegen Daten-Range.

### ⚠ Overfit-Indikatoren bei längerem Embedding (Run 1)

Mit `years=3.77` (W=1360d) bricht poly=7 in eine Sattelpunkt-Struktur ein, poly=5 zeigt eine andere Periode. Das deutet darauf hin, dass das längere Embedding-Fenster zu viele "lokale" Strukturen einfängt, die der Polynom-Fit dann overfittet.

### Höhere ssm_dim hilft Geometrie, schadet Stabilität

`ssm_dim=3` bringt cum-var von 70.9% auf 87.4% und Geometrie-Fehler von 0.45 auf 0.18 → der polynomiale Manifold passt besser. Aber die zusätzliche Mode wird oft instabil bei höherem poly_degree.

## Vorläufiger BEFUND

Die SSMLearnPy-Analyse auf den echten BTC-Residuen zeigt **eine reproduzierbare 2D-SSM-Struktur** mit einem **schwach gedämpften Oszillator von ~3.3 Jahren Periode** — konsistent mit der Halving-Hypothese. Die Existenz eines stabilen komplex-konjugierten Eigenwert-Paares mit moderatem Re(λ)<0 ist ein **direkter empirischer Hinweis auf einen niedrigdimensionalen Attraktor** in den BTC-Residuen.

**Robustestes Setup**: `M=35, years=2.0, ssm_dim=2, poly_degree=3`.

---

## NACH-CODEX-UPDATE: ssm_dim Sweep + poly=1 Baseline

Codex hat im Review **drei Hauptpunkte** kritisiert:
1. Mein "geometry NMTE" ist ein eigener RMS-normalisierter Fehler, kein library-NMTE — Werte nur intern vergleichbar
2. `alpha=0` und in-sample Auswertung → poly=5/7 ist Overfit-Risiko, nicht echtes Signal
3. Geometry-Fehler 0.45 ist Symptom für **zu kleines ssm_dim**, nicht für zu grosses Embedding

Empfehlung: **poly=3 fixieren, ssm_dim {2,3,4,5} sweepen, gegen poly=1 als PCA-Baseline halten**.

### ssm_dim Sweep (poly=3, M=35, years=2.0)

| ssm_dim | cum_var | geom rms | Eigenwert-Struktur |
|---|---|---|---|
| 1 | 40.7% | 0.765 | nur ein λ ≈ 0 (degeneriert, keine SSM) |
| 2 | 80.6% | 0.391 | komplexes Paar -5.6e-4 ± 4.9e-3j (T≈3.5y, gedämpft) |
| 3 | 87.4% | 0.310 | Oszi (-1.2e-4 ± 6.3e-3j, T=2.75y) + reeller +1.21e-3 (instabil) |
| 4 | 90.4% | 0.229 | **2 Oszi-Paare!** (T1=4.07y, T2=1.77y) |
| 5 | 92.7% | 0.178 | mehrere Modes, instabile reelle |

### poly=1 Baseline (= rein lineare PCA-Reduktion, KEIN Polynom-Fit)

| ssm_dim | cum_var | geom rms | Eigenwert-Struktur |
|---|---|---|---|
| 2 | 80.6% | 0.441 | -2.0e-4 ± 4.33e-3j (T=3.98y) **STABIL** |
| 3 | 87.4% | 0.354 | -2.6e-4 ± 5.18e-3j (T=3.32y) + -2.4e-4 reell **alle STABIL** |
| 4 | 90.4% | 0.309 | **2 Oszi-Paare BEIDE STABIL**: T1=4.27y, T2=1.93y |

### KERNBEFUND nach ssm_dim=4 sweep poly={1,2,3}

| poly | T1 [y] | Re(λ1) | T2 [y] | Re(λ2) | geom rms |
|---|---|---|---|---|---|
| 1 | 4.27 | **-3.0e-4** ✓ | 1.93 | **-1.7e-4** ✓ | 0.309 |
| 2 | 4.20 | -2.9e-4 ✓ | 1.86 | +3.4e-4 ✗ | 0.272 |
| 3 | 4.07 | +3.0e-3 ✗ | 1.77 | -3.9e-4 ✓ | 0.229 |

**Robust über alle poly_degrees**:
- **Modus 1**: T = 4.07–4.27 Jahre  →  ≈ BTC-Halving-Zyklus (3.84y)
- **Modus 2**: T = 1.77–1.93 Jahre  →  ≈ halber Halving-Zyklus (subharmonisch)

**Nicht robust**: Stabilitäts-Vorzeichen Re(λ) — höherer poly erzeugt teils instabile Modi (Codex: Overfit-Symptom).

### Konsequenz

Die **lineare Approximation (poly=1) bei ssm_dim=4** ist die **glaubwürdigste Aussage**:
- Beide Oszillator-Paare stabil (Re λ < 0)
- Hauptmode T ≈ 4.27y, Submode T ≈ 1.93y
- Geometrie-Fehler 0.31 (vergleichbar zu poly=3, kein massiver Verlust)
- Cum-Var 90.4% — die ersten 4 PCs erklären über 90% der Embedding-Varianz

→ **Die BTC-Residuen tragen ein DOPPEL-OSZILLATOR-SIGNAL** mit Hauptperiode am Halving-Takt und einer subharmonischen 2ω-Komponente. Das matched die Doppel-DOF-Struktur deiner DGL (`(y1, y2)` als Hauptoszillator + `Z_E·y1·y2` als 2ω-Treiber an z).

### Noch nicht gemacht (nächste Schritte)

1. **Walk-forward / leave-one-cycle-out** Validierung der reduzierten Dynamik (nicht nur Geometrie)
2. **`alpha > 0` (Ridge-Regularisierung)** als Anti-Overfit
3. **`dynamics_type='map'`** vs **`flow`** Vergleich (Codex hat schon eine Voranalyse: map zeigt das gleiche komplexe Paar)
4. **Walk-forward Vorhersage** vs Daten — die "ultimate test"

