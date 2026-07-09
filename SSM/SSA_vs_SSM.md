# SSA vs. SSM — tabellarischer Überblick der mathematischen Schritte

Stand: 07.06.2026 · Kontext: BTC-μ/Ensemble-n-Modenanalyse (`/home/hz/Data/mu_t/`)
und Attractor/TDE-Projekt (`/home/hz/Data/Attractor/`).

## Die mathematischen Schritte

| Schritt | **SSA** (Singular Spectrum Analysis) | **SSM** (State-Space-Modell / Attractor-Ansatz) |
|---|---|---|
| **1. Embedding** | Hankel-/Trajektorien-Matrix aus verschobenen Fenstern:<br>`X = [x⃗₁ … x⃗_K]`, `x⃗_k = (x_k, …, x_{k+L−1})ᵀ`<br>= TDE mit **τ=1, Dimension m=L** (μ-Analyse: L=1470) | Takens-Delay-Vektor:<br>`z⃗_t = (x_t, x_{t−τ}, …, x_{t−(m−1)τ})`<br>**τ aus 1. Minimum der Mutual Information**, **m aus FNN/Scan**<br>(attractor_n.py: τ=30, m=50 → W=1470d) |
| **2. Kern-Operation** | **SVD** (lineare Algebra, statisch):<br>`X = Σᵢ σᵢ·u⃗ᵢ·v⃗ᵢᵀ`<br>u⃗ᵢ = Schwingungsform im Fenster, v⃗ᵢ = deren Zeitverlauf, σᵢ² ∝ Varianzanteil | **Dynamik-Identifikation** (Modell, evolutiv):<br>Zustandsgleichung `z⃗_{t+1} = f(z⃗_t) + w⃗`<br>Beobachtung `x_t = h(z⃗_t) + v`<br>f wird gefittet (DGL-System / Analoga / Kalman bei linearem f) |
| **3. Selektion** | **Gruppierung der Eigentripel:** Oszillation = Paar mit σᵢ ≈ σᵢ₊₁ und ~90°-versetzten Eigenvektoren (ET1+ET2 …); Trend = unpaarige langsame Komponente; Rest = Rauschen | **Modellordnung/Struktur wählen:** Zustandsdimension, Form von f (linear → Kalman-Filter; nichtlinear → DGL-Terme, ssmtoolbox), Rausch-Kovarianzen Q, R |
| **4. Rekonstruktion / Nutzung** | **Diagonal-Mittelung** (Hankelisierung): Eigentripel-Gruppe → additive Zeitreihen-Komponente:<br>`x_t = Σ Moden + Rest` | **Filterung & Prädiktion:** Zustand vorwärts iterieren `ẑ_{t+1} = f(ẑ_t)` (Prognose); bei Kalman Predict/Update-Zyklus; Attraktor-Geometrie (Lyapunov, Dimension, FTLE) |

## Gegenüberstellung

| | SSA | SSM |
|---|---|---|
| **Leitfrage** | "Aus welchen additiven Komponenten besteht das Signal?" | "Welche **Dynamik** erzeugt das Signal?" |
| **Modellannahme** | keine (nichtparametrisch) | f muss postuliert/gefittet werden (parametrisch) |
| **Output** | Moden als Zeitreihen, Varianzanteile | generatives Modell, Vorhersage, Attraktor |
| **Zeitrichtung** | beschreibt Vergangenheit, extrapoliert nicht selbst | iteriert nach vorn (Prognose eingebaut) |
| **Stärke** | robust, ehrlich, keine Strukturannahme | Physik/Kausalität, out-of-sample-fähig |
| **Schwäche** | keine Prognose, Ränder degradiert (Diagonal-Mittelung) | Modellfehler wachsen exponentiell (Chaos) |

## Verbindung im Projekt

- **Schritt 1 ist identisch** — deshalb passte das TDE-Optimum aus dem Attractor-Projekt
  (Mutual-Info-Scan, W = 1470 d) direkt als SSA-Fensterlänge L.
- SSA = "TDE + lineare Statik": Hauptachsen der Trajektorienwolke.
- SSM = "TDE + Dynamik": wie bewegt sich ein Punkt **in** dieser Wolke.
- Arbeitsteilung in der μ/n-Analyse: **SSA identifiziert** die Moden (3.6a / 1.9a / 0.8a),
  **phasefit_halving** fittet sie parametrisch (Mini-SSM mit fixen Frequenzen),
  ein volles SSM (cDGL/Attractor-Zweig) würde sie **dynamisch erklären und vorhersagen**.
