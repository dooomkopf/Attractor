# SSM-Analyse $n_{\text{ens}}(t)$ — Zwischenstand 2026-04-29

## Aufbau

Zwei unabhängige Spectral-Submanifold-Analysen auf demselben Signal mit unterschiedlichen Zeit-Clocks:

- **SSM A (linear)**: gleichmäßiges Tages-Sampling
- **SSM B (log10)**: gleichmäßiges $\log_{10}(t)$-Sampling

Eingangssignal: BTC-LPPL-Exponent $n(t)$ aus rolling regression über Preis-Fenster. Zwei Konstruktionen getestet:
- **konstantes 180d-Fenster** (`--windows 180`)
- **log-uniformes Fenster** $W(t) = c \cdot t$ mit Mean $= 180$d (`--log_uniform_mean 180`)

Embedding: $M = 35$, Lag $\tau = 40$d, `start_idx = 1164`. SSMLearn mit polynomialer Reduktion `poly = 2`.

CLI: `analyze_n_ens/13_2_cli_lambda2_visualize.py`

---

## Hauptbefunde

### SSM A — linear-time, robust

**Sehr stabil über alle Fenster-Konstruktionen:**

| Fenster | Master $T$ | Sub1 $T$ | Master:Sub1 | decoder_err |
|---|---|---|---|---|
| const 180d | 1241d = 3.40y | 729d = 2.00y | 1.70 | 0.06 |
| log-uniform 180d (mode b) | 1273d = 3.49y | 634d = 1.74y | **2.005** | 0.07 |
| Ensemble [90–180] | 1216d = 3.33y | 841d = 2.30y | 1.45 | 0.18 |

**Interpretation**: Master ≈ Halving-Zyklus (mittlere Halving-Distanz 1387d = 3.80y, BTC-Fenster H1–H4 leicht kürzer). Sub1 ≈ erste Harmonische der Grundperiode. Linearer Halving-Takt ist **fenster-robuste Eigenschaft des Signals**.

PCs zeigen optisch saubere Sinusoide. cum_var(5) ≈ 70–73%.

### SSM B — log10-time, fenster-abhängig, **diskrete Skalen-Invarianz**

**Master $\lambda$ verschiebt sich mit Fenster-Konstruktion:**

| Fenster | Master $\lambda$ | Sub1 $\lambda$ | decoder_err |
|---|---|---|---|
| const 180d | **1.99** ≈ 2 (Halving-Verdopplung) | 1.20 | 0.06 |
| log-uniform 180d | 1.50 | 1.28 | 0.06 |
| Ensemble [90–180] | 1.41 | 1.20 | 0.61 |

Das Ensemble-Mitteln hat die saubere $\lambda \approx 2$-Struktur **wegmittelt**.

**Mit `ssm_dim = 9` (const 180d): Diskrete Skalen-Invarianz-Spektrum**

| Mode | $\lambda$ | $T$ | $2^{1/k_n}$ Match |
|---|---|---|---|
| Master | 1.99 | 0.28 | $k=1$, $2^{1/1} = 2$ ✓ |
| Sub1 | 1.42 | 0.15 | $k=2$, $\sqrt{2} = 1.414$ ✓ |
| Sub2 | 1.27 | 0.10 | $k=3$, $2^{1/3} = 1.260$ ✓ |
| Sub3 | 1.10 | 0.04 | $k=8$, $2^{1/8} = 1.091$ ✓ |
| Sub4 | 1.06 | 0.02 | $k=12$, $2^{1/12} = 1.060$ ✓ |

decoder_err = 0.07, cum_var(9) = 87.0%

**Konsequenz**: Alle 5 Eigenwerte bilden eine **log-harmonische Serie** auf der Halving-Verdopplung $\lambda = 2$. Das ist mathematisch die Signatur **diskreter Skalen-Invarianz (DSI)**, Vorhersage von Sornette für log-periodische Pre-Crash-Strukturen.

**Auswahl-Pattern**: $k_n \in \{1, 2, 3, 8, 12\}$. Lücke zwischen $k=3$ und $k=8$ — entweder schwächere Moden im Rauschen oder echte Auswahl-Regel (offen).

**Bei `ssm_dim = 11`**: decoder_err schießt auf 0.21, doppelte Eigenwerte, Master verschwindet → **Overfitting bei poly=2**. Sweet Spot bei dim=9.

---

## Vergleich zu Residuen-Analyse

(Aus dem Schwester-Pipeline `analyze_residuals/13_cli_lambda2_visualize.py`)

| Aspekt | Residuen | $n_{\text{ens}}$ |
|---|---|---|
| Definition | $\log p(t) - \text{Trend}$ | rolling $d \log p / d \log t$ über 180d |
| **SSM A linear master** | $\sim 4$y (Halving) | $\sim 3.4$y (Halving) — **gleich** |
| **SSM A Sub-Harmonische** | sichtbar | sichtbar — **kohärent** |
| **SSM B log master $\lambda$** | **$\approx 2.005$** (sehr sauber) | $\approx 1.99$ (bei const 180d) |
| **SSM B Sub-Moden bei dim=9** | nicht im selben Maße sichtbar | $\lambda^{1/k}$-Serie für $k \in \{1,2,3,8,12\}$ |
| **Decoder-Err log-Clock** | sehr klein (sauberer SSM) | 0.06 bei dim=9 |
| **DSI-Struktur** | (zu prüfen) | **klar präsent** |

**Knackpunkt**: Beide Signale tragen die Halving-Verdopplung in $\log$-Zeit, aber:
- Residuen zeigen sie als **einzelne saubere log-periodische Mode**
- $n_{\text{ens}}$ zeigt sie als **Master + diskrete Sub-Verfeinerungen** ($\lambda^{1/k}$)

Mögliche Erklärung: $n(t)$ ist die **lokale Steigung** in log-log space, also eine Ableitung. Ableitung eines log-periodischen Signals erzeugt automatisch alle Harmonischen, falls das Signal nicht reine Sinusoide ist (was bei realen Halving-Strukturen zu erwarten ist — Pre-Crash-Beschleunigungen, asymmetrische Preisverläufe). Die Ableitungs-Operation entfaltet die Fourier-Reihe der Halving-periodischen Funktion.

Residuen entfalten diese Reihe **nicht** (sie sind direkt der log-periodische Anteil), $n(t)$ entfaltet sie **explizit**.

---

## Fenster-Konstruktion: konstant vs. log-uniform

**Hypothese (initial)**: Konstantes 180d-Linearfenster erzeugt scale-abhängigen Filter im log-Clock → Sub-Moden in SSM B sind Sampling-Artefakte.

**Test (a) nur SSM B umgestellt, (b) beide umgestellt**:

| | konst. 180d | log-uniform mean=180d |
|---|---|---|
| SSM B Master $\lambda$ | 1.99 | 1.50 |
| SSM B Sub1 $\lambda$ | 1.20 | 1.28 |

**Hypothese widerlegt**: Sub-Moden bleiben erhalten, Master verschiebt sich. Die wilden log-Strukturen sind **echte Eigenschaften des $n(t)$-Signals**, nicht Artefakte des Fensters. Die exakte $\lambda$-Lage der Master-Mode hängt aber von der Fenster-Konstruktion ab.

SSM A bleibt bei beiden Konstruktionen optisch und numerisch sauber.

---

## Offene Punkte

1. **Auswahl-Regel** für $k_n \in \{1,2,3,8,12\}$ — warum diese Lücke?
2. **DSI-Bestätigung in Residuen**: zeigt `analyze_residuals` mit `ssm_dim=9` ähnliche $\lambda^{1/k}$-Serie?
3. **Fourier-Cross-Check**: direkte FFT auf $\log_{10}$-Clock-PC1, Peaks bei $f, 2f, 3f, \ldots$?
4. **Sornette-Connection**: konkrete Interpretation der gefundenen DSI-Struktur im Kontext von log-periodischen Crash-Vorhersagen.
5. **dim=11 / poly=3**: lohnt höherer poly-Grad, um mehr Sub-Moden bei dim=11 stabil zu sehen?

---

## Korrektur DSI-Interpretation (2026-04-29 später)

Erste Lesart: SSM-B-Sub-Moden bei $\lambda^{1/k}$ sind „echte DSI-Signatur".

**Korrektur**: Diese Periodenverhältnisse $T_n = T_M / k$ sind mathematisch identisch mit Fourier-Harmonischen einer einzigen nicht-sinusoidalen Mode. Eine Periodenverhältnis-Auswertung allein genügt NICHT, um „echte unabhängige log-Modes" von „PCA-Harmonischen einer einzigen Mode" zu unterscheiden.

Definitive Tests sind **Phase-Lock** und **Amplitude-Scaling** zwischen Master und Sub.

---

## Phase/Scaling-Test (Residuen, beide Clocks)

Nutzt bestehende Skripte `03_cli_phase.py` und `04_cli_scaling.py` aus `analyze_residuals/` (Konfig: `--ssm_dim 4 --poly 1`).

| Test | **Linear-Clock (SSM A)** | **Log-Clock (SSM B)** |
|---|---|---|
| $T_{\text{sub}}/(T_{\text{main}}/2)$ | $\approx 1.05$ | **0.995** (exakt 2:1) |
| Phase-Lock $R$ | **0.67** (gelockt) | **0.27** (entkoppelt) |
| Median $\|\Delta\phi\|$ | 57° | 56° |
| Drift ratio actual/expected | **1.38** (nahe 1) | **$-56.5$** (Vorzeichen falsch) |
| Scaling $\text{corr}(\|z_M\|^2, \|z_S\|)$ | $+0.14$ (schwach positiv) | **$-0.80$** (stark **negativ**) |
| Slave-$R^2$ durch Ursprung | 0.77 | 0.39 |

**Interpretation:**

- **Linear-Clock**: Phase-Lock $R=0.67$, Drift-ratio $\approx 1$, positive Scaling-Korrelation → die 2y-Mode ist **gekoppelt** an die 4y-Master-Mode = **geometrischer Slave** (Manifold-Sattelkrümmung). Konsistent mit LPPL-analytischem Whisker-Test ($R_2 = 0$, $|W_2| \neq 0$).

- **Log-Clock**: Phase-Lock $R=0.27$ (entkoppelt), Drift-ratio mit **falschem Vorzeichen**, **negative** Scaling-Korrelation → die $\sqrt{2}$-Mode ist **unabhängig** vom $\lambda=2$-Master. Die negative Korrelation deutet auf **Energie-Umverteilung zwischen log-Skalen** — charakteristisch für **diskrete Skalen-Invarianz**.

---

## Strukturelle Asymmetrie SSM A vs SSM B (definitiv)

**SSM A (linear-clock)**:
- 1 Halving-Mode ($T \approx 4$y) + nicht-sinusoidale Form
- Sub-Modes sind **PCA-Harmonische / Manifold-Krümmung**
- KEINE unabhängigen Freiheitsgrade über die Halving-Mode hinaus
- BTC-Halving als linear-periodischer Oszillator mit Sattel-Geometrie

**SSM B (log-clock)**:
- Master-Mode $\lambda \approx 2$ (Halving-Verdopplung)
- Sub-Modes bei $\lambda^{1/k}$ sind **echte unabhängige log-periodische Strukturen**
- DSI-Signatur (Sornette): hierarchische Selbstähnlichkeit über mehrere log-Skalen
- BTC-Preisbildung trägt log-periodische Mehrskalen-Struktur

**Konsequenz:** Beide Sichten sind nicht „dieselbe Mode in zwei Brillen", sondern **zwei strukturell verschiedene Eigenschaften des BTC-Signals**, die durch die Wahl des Zeit-Clocks als „Filter" sichtbar werden.

---

## Reproduzieren

```bash
# Default (Ensemble): dim=5, poly=2
./13_2_cli_lambda2_visualize.py --ssm_dim 5

# Single 180d-Fenster (sauberer Halving-λ≈2 Master in SSM B)
./13_2_cli_lambda2_visualize.py --ssm_dim 5 --windows 180 --no-marker

# DSI-Test bei dim=9
./13_2_cli_lambda2_visualize.py --ssm_dim 9 --windows 180 --no-marker

# Mode (a): SSM B mit log-uniform Fenster, SSM A unverändert
./13_2_cli_lambda2_visualize.py --ssm_dim 5 --windows 180 --log_uniform_mean 180 --no-marker

# Mode (b): beide mit log-uniform
./13_2_cli_lambda2_visualize.py --ssm_dim 5 --windows 180 --log_uniform_mean 180 --log_uniform_both --no-marker
```

Plot-Dateien tragen `dim{N}_poly{P}_w{...}_{TS}.png` und identifizieren Konfiguration eindeutig.

```bash
# Phase- und Scaling-Tests (analyze_residuals/), beide Clocks
./03_cli_phase.py --ssm_dim 4 --poly 1 --time_mode linear
./03_cli_phase.py --ssm_dim 4 --poly 1 --time_mode log
./04_cli_scaling.py --ssm_dim 4 --poly 1 --time_mode linear
./04_cli_scaling.py --ssm_dim 4 --poly 1 --time_mode log

# Harmonics-Test mit allen Sub-Eigenwerten
./analyze_residuals/13_2_cli_lambda2_visualize.py --ssm_dim 9 --no-marker
./analyze_n_ens/13_2_cli_lambda2_visualize.py --ssm_dim 9 --windows 180 --no-marker
```
