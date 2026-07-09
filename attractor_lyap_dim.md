# Erweiterungen fuer `attractor.py`: Lyapunov-Exponent und fraktale Dimension

Ausgangslage in [`attractor.py`](/home/hz/Data/Attractor/attractor.py): Das Delay-Embedding ist

- `tau = 30` Tage
- `m = 50`
- `W = (m - 1) * tau = 1470` Tage
- `D.shape = (N - W, 50)`
- `pc = D_c @ Vt.T`

Fuer die Schaetzung sollte die rohe Trajektorie verwendet werden:

```python
pc3 = pc[:, :3]          # N x 3
# Nicht verwenden: pc_s (ist nur fuer die Visualisierung geglaettet)
```

## 1) Lyapunov-Exponent

### Was ist aus dem Delay-Embedding sinnvoll schaetzbar?

Ja, man kann aus dem Delay-Embedding mehr als nur einen globalen Wert ableiten, aber man muss sauber unterscheiden:

- Ein **globaler groesster Lyapunov-Exponent** `lambda_1` ist aus der rekonstruierten Trajektorie am robustesten schaetzbar.
- **Lokale Werte** sind nur als **finite-time / local divergence rates** sinnvoll. Das sind diagnostische Groessen entlang der Bahn, aber keine stabilen Invarianten.
- Ein **voller lokaler Exponentensatz** ist aus dem auf 3D per PCA reduzierten `pc3` nicht zuverlaessig. Die PCA wirft 47 Richtungen weg; damit geht Tangenteninformation verloren.

Fuer unsere BTC-Zeitreihe mit `tau = 30d`, `m = 50` ist deshalb die beste Praxis:

1. Den **groessten globalen Exponenten** mit der **Rosenstein-Methode** schaetzen.
2. Die Nachbarsuche mit einem **Theiler-Window mindestens `W = 1470` Tagen** machen, weil sich benachbarte Embedding-Vektoren sonst fast dieselben Samples teilen.
3. Nur die **fruehe lineare Divergenzphase** fitten, bevor die Kurve in die Attraktor-Groesse saettigt.
4. Wenn moeglich die Methode auf dem **vollen Delay-Embedding `D`** anwenden; fuer den geplotteten 3D-Attraktor ist `pc3` aber als praktische Naeherung verwendbar.

Kurz gesagt: **Global `lambda_1` ja, lokale FTLEs nur vorsichtig als Zusatzdiagnostik.**

### Konkreter Python-Code

```python
import numpy as np
from scipy.spatial import cKDTree


def _zscore_columns(x):
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (x - mu) / sd


def _nearest_valid_neighbor(tree, x, i, theiler, last_start, k0=8):
    n = len(x)
    k = min(max(k0, 2), n)

    while True:
        dists, idxs = tree.query(x[i], k=k)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)

        # idx 0 ist der Punkt selbst
        for d, j in zip(dists[1:], idxs[1:]):
            j = int(j)
            if not np.isfinite(d) or d <= 0.0:
                continue
            if j >= last_start:
                continue
            if abs(j - i) <= theiler:
                continue
            return j, float(d)

        if k >= n:
            return None, None
        k = min(n, 2 * k)


def estimate_max_lyapunov_rosenstein(
    pc,
    dt=1.0,
    theiler=1470,
    max_horizon=120,
    fit_range=(10, 50),
    standardize=True,
):
    """
    Rosenstein-Schaetzer fuer den groessten Lyapunov-Exponent.

    Parameter
    ---------
    pc : array, shape (N, 3)
        3D-PCA-Trajektorie.
    dt : float
        Zeitschritt pro Sample. Bei euren Daten: 1.0 Tag.
    theiler : int
        Ausschlussfenster in Samples. Fuer euer Embedding sinnvoll: W = 1470.
    max_horizon : int
        Wie viele Tage die Nachbarpaare gemeinsam vorwaerts verfolgt werden.
    fit_range : tuple(int, int)
        Bereich der linearen Fruehphase, auf den die Steigung gefittet wird.
    standardize : bool
        Z-Score je PC-Achse; bei PCA-Projektionen meist stabiler.
    """
    x = np.asarray(pc, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("pc muss die Form (N, 3) haben.")

    if standardize:
        x = _zscore_columns(x)

    n = len(x)
    last_start = n - max_horizon
    if last_start <= 10:
        raise ValueError("Zeitreihe ist fuer das gewaehlte max_horizon zu kurz.")

    tree = cKDTree(x)
    pairs = []

    for i in range(last_start):
        j, d0 = _nearest_valid_neighbor(tree, x, i, theiler, last_start)
        if j is not None:
            pairs.append((i, j, d0))

    if len(pairs) < 30:
        raise ValueError(
            "Zu wenige gueltige Nachbarpaare. "
            "theiler oder max_horizon reduzieren oder mehr Daten verwenden."
        )

    eps = np.finfo(float).eps
    curves = []
    for i, j, _ in pairs:
        d = np.linalg.norm(x[i:i + max_horizon] - x[j:j + max_horizon], axis=1)
        curves.append(np.log(np.maximum(d, eps)))
    curves = np.asarray(curves)

    mean_log_div = np.nanmean(curves, axis=0)
    t = np.arange(max_horizon, dtype=float) * dt

    k0, k1 = fit_range
    if not (0 <= k0 < k1 <= max_horizon):
        raise ValueError("fit_range muss innerhalb von [0, max_horizon] liegen.")

    slope, intercept = np.polyfit(t[k0:k1], mean_log_div[k0:k1], 1)
    fit = slope * t[k0:k1] + intercept

    ss_res = np.sum((mean_log_div[k0:k1] - fit) ** 2)
    ss_tot = np.sum((mean_log_div[k0:k1] - mean_log_div[k0:k1].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    local_ftle = []
    for row in curves:
        s, _ = np.polyfit(t[k0:k1], row[k0:k1], 1)
        local_ftle.append(s)
    local_ftle = np.asarray(local_ftle)

    return {
        "lambda_global_per_day": slope,
        "lambda_global_per_year": slope * 365.25,
        "r2": r2,
        "pairs_used": len(pairs),
        "t": t,
        "mean_log_divergence": mean_log_div,
        "local_ftle_per_day": local_ftle,
        "fit_range": fit_range,
    }
```

### Beispiel fuer `attractor.py`

```python
pc3 = pc[:, :3]

lyap = estimate_max_lyapunov_rosenstein(
    pc3,
    dt=1.0,          # 1 Tag pro Zeitschritt
    theiler=W,       # W = 1470
    max_horizon=120,
    fit_range=(10, 50),
    standardize=True,
)

print(f"lambda_1 = {lyap['lambda_global_per_day']:.5f} / Tag")
print(f"lambda_1 = {lyap['lambda_global_per_year']:.3f} / Jahr")
print(f"R^2 der linearen Fruehphase = {lyap['r2']:.3f}")
print(f"Median lokaler FTLE = {np.median(lyap['local_ftle_per_day']):.5f} / Tag")
```

Interpretation:

- `lambda_global_per_day > 0` bedeutet hier eine **effektive mittlere Divergenzrate** im rekonstruierten Raum.
- Das ist bei BTC **kein Beweis fuer einen autonomen chaotischen Determinismus**, weil die Reihe verrauscht und nichtstationaer ist.
- `local_ftle_per_day` sind **nur lokale finite-time Divergenzraten**. Sie sind nuetzlich zum Markieren von Regimewechseln, aber nicht als "wahre lokale Exponenten" zu lesen.

## 2) Fraktale Dimension

### Welche Methode passt fuer den 3D-PCA-Attraktor?

Fuer euren geplotteten `pc3`-Attraktor ist die **Korrelationsdimension nach Grassberger-Procaccia (GP)** die beste Wahl.

Warum GP hier besser ist als Box-Counting:

- GP arbeitet direkt auf der **Punktwolke / Trajektorie** und ist fuer rekonstruierten Attraktoren Standard.
- **Box-Counting** ist in 3D bei endlichen, anisotropen und ungleich besetzten Trajektorien stark **grid-abhaengig**.
- GP laesst sich mit einem **Theiler-Window** kombinieren; das ist bei eurem Delay-Embedding wichtig.
- Fuer `pc3` erhaltet ihr damit eine sinnvolle **Dimension des projizierten 3D-Attraktors**.

Wichtig:

- Auf `pc3` bekommt ihr **nicht** die volle Dimension des urspruenglichen 50D-Embeddings, sondern nur die Dimension der 3D-Projektion.
- Wenn ihr die invarianten Eigenschaften der rekonstruierten Dynamik wollt, ist GP auf `D` methodisch besser. Fuer den sichtbaren 3D-Attraktor ist GP auf `pc3` genau die passende Zusatzgroesse.

### Konkreter Python-Code

```python
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


def _zscore_columns(x):
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (x - mu) / sd


def _valid_pair_count(n, theiler):
    total = n * (n - 1) // 2
    w = int(np.clip(theiler, 0, n - 1))
    excluded = w * n - w * (w + 1) // 2
    return total - excluded


def _auto_radii(x, n_radii=24, q_low=0.02, q_high=0.30, sample_size=1500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    m = min(n, sample_size)
    idx = rng.choice(n, size=m, replace=False)
    d = pdist(x[idx])
    d = d[d > 0.0]
    if len(d) == 0:
        raise ValueError("Keine positiven Distanzen gefunden.")

    r_min, r_max = np.quantile(d, [q_low, q_high])
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_min <= 0.0 or r_max <= r_min:
        raise ValueError("Automatische Radiuswahl ist fehlgeschlagen.")

    return np.geomspace(r_min, r_max, n_radii)


def _find_plateau(log_r, log_c, counts, valid_pairs, min_len=4, min_pairs=100):
    local_slope = np.gradient(log_c, log_r)
    ok = (
        np.isfinite(local_slope)
        & (counts >= min_pairs)
        & (counts <= 0.10 * valid_pairs)
    )

    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        raise ValueError("Kein brauchbarer Skalierungsbereich gefunden.")

    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    groups = [g for g in groups if len(g) >= min_len]
    if not groups:
        raise ValueError("Kein ausreichend langer Skalierungsbereich gefunden.")

    best = min(groups, key=lambda g: (np.std(local_slope[g]), -len(g)))
    return int(best[0]), int(best[-1]) + 1, local_slope


def estimate_correlation_dimension_gp(
    pc,
    theiler=1470,
    n_radii=24,
    fit_idx=None,
    standardize=True,
    seed=0,
):
    """
    Grassberger-Procaccia auf der 3D-PCA-Trajektorie.

    Rueckgabe: geschaetzte Korrelationsdimension D2 des 3D-Attraktors.
    """
    x = np.asarray(pc, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("pc muss die Form (N, 3) haben.")

    if standardize:
        x = _zscore_columns(x)

    radii = _auto_radii(x, n_radii=n_radii, seed=seed)
    tree = cKDTree(x)

    coo = tree.sparse_distance_matrix(tree, radii[-1], output_type="coo_matrix")
    keep = (coo.row < coo.col) & ((coo.col - coo.row) > theiler)
    d = np.sort(np.asarray(coo.data[keep], dtype=float))

    valid_pairs = _valid_pair_count(len(x), theiler)
    counts = np.searchsorted(d, radii, side="right")
    c_r = counts / valid_pairs

    good = (counts > 0) & np.isfinite(c_r) & (c_r > 0.0)
    if good.sum() < 6:
        raise ValueError("Zu wenige Radien mit gueltiger Korrelationssumme.")

    log_r = np.log(radii[good])
    log_c = np.log(c_r[good])
    counts_good = counts[good]

    if fit_idx is None:
        i0, i1, local_slope = _find_plateau(log_r, log_c, counts_good, valid_pairs)
    else:
        i0, i1 = fit_idx
        local_slope = np.gradient(log_c, log_r)

    slope, intercept = np.polyfit(log_r[i0:i1], log_c[i0:i1], 1)
    fit = slope * log_r[i0:i1] + intercept

    ss_res = np.sum((log_c[i0:i1] - fit) ** 2)
    ss_tot = np.sum((log_c[i0:i1] - log_c[i0:i1].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "D2": slope,
        "r2": r2,
        "radii": radii[good],
        "correlation_sum": c_r[good],
        "log_r": log_r,
        "log_c": log_c,
        "local_slope": local_slope,
        "fit_idx": (i0, i1),
        "pairs_within_rmax": len(d),
        "valid_pairs": valid_pairs,
    }
```

### Beispiel fuer `attractor.py`

```python
pc3 = pc[:, :3]

dim_gp = estimate_correlation_dimension_gp(
    pc3,
    theiler=W,       # W = 1470
    n_radii=24,
    fit_idx=None,    # automatische Plateau-Suche
    standardize=True,
)

print(f"Korrelationsdimension D2 = {dim_gp['D2']:.3f}")
print(f"R^2 des log-log-Fits = {dim_gp['r2']:.3f}")
print(f"benutzter Fitbereich = {dim_gp['fit_idx']}")
```

Interpretation:

- `D2 < 1`: fast kurvenartige Struktur
- `1 < D2 < 2`: duenne flaechige / bandartige Struktur
- `2 < D2 < 3`: volumiger 3D-Attraktor
- `D2 > 3` sollte auf `pc3` nicht stabil herauskommen; dann ist meist der Fitbereich schlecht gewaehlt

## Praktische Empfehlung fuer `attractor.py`

Wenn ihr die zwei Erweiterungen in `attractor.py` spaeter direkt einbaut, dann:

1. **Lyapunov auf `pc[:, :3]` oder besser auf `D` rechnen, aber nie auf `pc_s`.**
2. **Immer `theiler=W` als Startwert nehmen.**
3. **Den Fitbereich visuell kontrollieren**:
   - bei Lyapunov die lineare Fruehphase von `mean_log_divergence`
   - bei GP das Plateau von `local_slope = d log C(r) / d log r`
4. Fuer BTC eher von **"effektiver Divergenzrate"** und **"Dimension der rekonstruierten / projizierten Struktur"** sprechen als von streng invarianten Chaos-Kennzahlen.
