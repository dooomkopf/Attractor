# Attractor Pipeline

Dokumentiert den Workflow ab fertigen Log-Residuen. Identisch anwendbar auf synthetische Zeitreihen.

---

## 0. Eingabe

| Variable | Beschreibung |
|----------|-------------|
| `log_res` | 1D-Array, Log-Residuen im Log-Raum (bereits berechnet) |
| `t_index` | 1D-Array, Zeitindex (Tage / Blockhöhe / beliebige monotone Einheit) |
| `dates`   | Optional: Datums-Array für Achsenbeschriftungen |

**Cutoff (START_IDX)**:
```python
mask   = t_index >= START_IDX   # default: START_IDX = 0 → gesamte Serie
log_res = log_res[mask]
t_index = t_index[mask]
N       = len(log_res)
```

---

## 1. Delay-Embedding (Takens)

### Parameter

| Parameter | Bedeutung | Empfehlung |
|-----------|-----------|------------|
| `M`       | Embedding-Dimension (Anzahl Verzögerungen) | 20–50 |
| `TAU`     | Zeitverzögerung in Index-Einheiten | `round(T_cycle / (M-1))` oder frei |
| `W`       | Gesamtfensterbreite = `(M-1) * TAU` | ergibt sich aus M, TAU |

### Konstruktion der Delay-Matrix

Zeile `i` enthält das Fenster der letzten `W` Punkte mit Abstand `TAU`:

```
D[i, j] = log_res[i + W - j*TAU]   für j = 0 … M-1
```

Implementierung:
```python
W = (M - 1) * TAU
D = np.empty((N - W, M))
for j in range(M):
    D[:, j] = log_res[W - j*TAU : N - j*TAU]
```

Ergebnis: `D` hat Shape `(N-W, M)`. Jede Zeile ist ein Zustandsvektor im M-dimensionalen Einbettungsraum.

**Optional** — Fenster-Normierung (default: off):
```python
D -= D.mean(axis=1, keepdims=True)
```

Zeitindex der Zustandsvektoren (aktuellster Punkt pro Zeile):
```python
t_vec = t_index[W:]   # length N-W
```

---

## 2. PCA

### Schritt 1 — Spalten-Zentrierung
```python
D_c = D - D.mean(axis=0)
```

### Schritt 2 — SVD
```python
_, s, Vt = np.linalg.svd(D_c, full_matrices=False)
```

### Schritt 3 — Projektion
```python
pc  = D_c @ Vt.T          # Shape: (N-W, M)
var = s**2 / (s**2).sum() # Varianzanteile je PC
```

`pc[:, 0]` = PC1, `pc[:, 1]` = PC2, `pc[:, 2]` = PC3 usw.  
Die Achsen werden automatisch durch den SVD bestimmt (kein manueller Eingriff).

---

## 3. Glättung (optional, nur für Visualisierung)

Gaussian-Glättung der Trajektorie für den 3D-Plot:
```python
from scipy.ndimage import gaussian_filter1d
pc_s = pc.copy()
for j in range(pc_s.shape[1]):
    pc_s[:, j] = gaussian_filter1d(pc[:, j], sigma=SMOOTH_SIGMA)
```

**Wichtig**: Glättung `pc_s` nur für Darstellung. Phase wird auf **ungeglättetem** `pc` berechnet.

---

## 4. Phase-Berechnung

```python
theta    = np.arctan2(pc[:, 1], pc[:, 0])   # gewrappter Winkel, ∈ (-π, π]
r        = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)  # Amplitude (Radius in PC1-PC2-Ebene)
theta_uw = np.unwrap(theta)                  # kumulierter Winkel (kein Reset bei ±π)

# CCW-Korrektur: sicherstellen dass θ_uw monoton steigt
if theta_uw[-1] < theta_uw[0]:
    theta    = -theta
    theta_uw = -theta_uw
```

Kein externer Phasen-Anker. Phase startet bei ihrem natürlichen Anfangswert.

---

## 5. Zyklus-Detektion aus θ_unwrapped

Jede vollständige Umdrehung (Δθ = 2π) gilt als ein Zyklus.

```python
cross_idx = []
for k in range(1, 6):                        # maximal 5 Zyklen
    level = k * 2 * np.pi
    idx = np.where(
        (theta_uw[:-1] < level) & (theta_uw[1:] >= level)
    )[0]
    if len(idx) > 0:
        cross_idx.append(idx[0])
```

Zykluslängen und extrapolierte nächste Grenze:
```python
bounds     = [0] + cross_idx + [len(theta_uw) - 1]
cycle_durs = []
for ci in range(1, len(bounds) - 2):         # nur vollständige Zyklen
    i1, i2 = bounds[ci], bounds[ci + 1]
    cycle_durs.append(t_vec[i2] - t_vec[i1])

if cycle_durs:
    med_dur        = np.median(cycle_durs)
    next_boundary  = t_vec[cross_idx[-1]] + med_dur
```

---

## 6. Color-Coding

Normierter Zeitparameter für alle Plots (alt → neu = blau → rot):
```python
t_min, t_max = t_vec[0], t_vec[-1]
t_norm = (t_vec - t_min) / (t_max - t_min)   # ∈ [0, 1]
CMAP   = plt.cm.coolwarm
```

Für Rohdaten-Plot (volle Länge):
```python
t_norm_all = (t_index - t_index[0]) / (t_index[-1] - t_index[0])
```

---

## 7. Visualisierung — 4 Fenster

### Fenster 1 — Hauptfenster (22×10 inches)

**Links — Rohdaten-Plot (ax1)**:
- X: Zeitindex (`t_index`), Y: Rohdaten (Log-Skala empfohlen)
- `LineCollection` mit `t_norm_all` → color-coded
- Zyklus-Grenzen als vertikale gestrichelte Linien (`#FFDD44`)
- Zykluslängen-Labels über den Grenzen
- Nächste extrapolierte Grenze einzeichnen
- Aktuellster Punkt: weißer Marker

**Rechts — 3D Attraktor (ax2, projection='3d')**:
- X=PC1, Y=PC2, Z=PC3 (geglättet `pc_s`)
- `Line3DCollection` mit `t_norm_vec` → color-coded
- Aktuellster Punkt: weißer Marker
- Achsenbeschriftung mit Varianzanteil: `f'PC1 ({var[0]*100:.1f}%)'`

**Unten — Slider**:
- Bereich: `[t_vec[0], t_vec[-1]]`
- Bewegt neon-gelbes Highlight (±`LABEL_WINDOW` Einheiten) synchron in ax1 und ax2

Layout:
```python
ax1   = fig.add_axes([0.04, 0.30, 0.42, 0.43])
ax2   = fig.add_axes([0.50, 0.06, 0.50, 0.90], projection='3d')
ax_sl = fig.add_axes([0.10, 0.02, 0.80, 0.025])
```

---

### Fenster 2 — Phasenraum (14×7 inches)

**Links — Polarplot (ax3a, projection='polar')**:
- θ (angle) vs r (radius)
- `LineCollection` mit `t_norm_vec` → color-coded
- Keine radialen Tick-Labels (`set_yticklabels([])`, `set_rticks([])`)

**Rechts — Phase-Plot (ax3b)**:
- X: θ ∈ [0, 2π], Y: PC3
- Scatter mit `t_norm_vec` → color-coded
- X-Ticks: `[0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4, 2π]`

---

### Fenster 3 — θ_unwrapped (14×6 inches, optional)

**Links — linear (ax4a)**:
- X: `t_vec` (linear), Y: `theta_uw`
- `LineCollection` mit `t_norm_vec` → color-coded
- Zyklus-Grenzen eingezeichnet
- Gerade = konstante Periode

**Rechts — log (ax4b)**:
- X: `log(t_vec)`, Y: `theta_uw`
- `LineCollection` mit `t_norm_vec` → color-coded
- Gerade = log-periodisch

---

## 8. Parameter-Übersicht

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| `START_IDX` | 0 | Cutoff: Embedding startet ab diesem Index |
| `M` | 35 | Embedding-Dimension |
| `TAU` | `round(T_cycle / (M-1))` | Zeitverzögerung |
| `NORMALIZE_WINDOWS` | False | Zeilen-Normierung der Delay-Matrix |
| `SMOOTH_SIGMA` | 60 | Gaussian-Sigma für PC-Glättung (nur Plot) |
| `LABEL_WINDOW` | 30 | Slider-Highlight-Breite (±Einheiten) |
| `CMAP` | `coolwarm` | Colormap alt→neu |
| `MAX_CYCLES` | 5 | Maximale Zyklus-Crossings die gesucht werden |
| `SHOW_FIG3` | True | Fenster 3 (θ_unwrapped) anzeigen |

---

## 9. Abhängigkeiten

```
numpy, matplotlib, scipy.ndimage.gaussian_filter1d
```

Optionale Abhängigkeiten für Residuen-Berechnung (nicht Teil dieser Pipeline):
```
statsmodels (QuantReg)
```
