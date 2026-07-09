# 2-Scroll Attraktor & Nambu-Mechanik — Projektgedächtnis

**Quelle:** Patra & Ganguli, "Emergence of Multi-Scroll Attractors", arXiv:2511.13332v1 (Nov 2025)
**Ziel:** Verstehen, wie man aus empirischen PCA-Trajektorien (BTC) die Nambu-Hamiltonians H₁, H₂ rekonstruiert.

---

## 1. Das Wang-System (die Basis)

Das 3D-autonome quadratische System:

```
ẋ = a(x−y) − yz
ẏ = −by + xz
ż = −cz + dx + xy
```

Parameter: a, b, c, d ∈ ℝ; Zustandsvariablen: x, y, z.

**Divergenz:**
```
∇·v = ∂ẋ/∂x + ∂ẏ/∂y + ∂ż/∂z = a − b − c
```
Für a < b+c gilt ∇·v < 0 → **dissipativ** → Phasenraumvolumen schrumpft → Attraktor existiert.

**5 Gleichgewichtspunkte** (∇ẋ=ẏ=ż=0):
- S1 = (0, 0, 0)  ← Index-1-Sattelfokus (Grenze zwischen Scrolls)
- S2, S3, S4, S5 symmetrische Paare  ← Index-2-Sattelfoki (Zentren der Scrolls)

Index-2-Sattelfokus: 1 negativer reeller EW + 2 komplexe konjugierte EW mit positivem Realteil → spiralförmige Anziehung.

**Jacobian:**
```
J = | a    −(a+z)  −y |
    | z    −b       x |
    | d+y   x      −c |
```

**Parameter für verschiedene Scroll-Zahlen** (aus Paper, Fig. 1 + Fig. 6):

| Scrolls | a     | b    | c | d    | Quelle |
|---------|-------|------|---|------|--------|
| 1-Scroll | 1.0  | 1.67 | 5 | 0.06 | Fig 1a, Tabelle I |
| **2-Scroll** | **3.5** | **9.0** | **5** | **0.06** | **Fig 6b(ii) ← unser Simulationsfall** |
| 2-Scroll (alt) | 3.1 | 9.0 | 1 | 0.06 | Tabelle I |
| 2-Scroll (Bifurkation) | 1.0 | 3.2 | 5 | 0.06 | Fig 2 (Bifurkationsdiagramm) |
| 3-Scroll | 1.46 | 8.44 | 2 | 0.48 | Tabelle I |
| 4-Scroll | 0.977 | 10 | 4 | 0.06 | Tabelle I |

**Hinweis:** b und c erscheinen NUR im dissipativen Teil v_D = (ax, −by, −cz).
In den Nambu-Hamiltonians H₁, H₂ und im nicht-dissipativen Fluss kommen nur **a** und **d** vor.
Daher bestimmen a und d die Scroll-Geometrie; b und c nur den Chaosgrad / die Dissipation.

Tabelle I (Paper): vollständige Parametersätze mit Eigenwerten aller 5 Gleichgewichtspunkte.

---

## 2. Nambu-Mechanik: Grundformalismus

### 2.1 Verallgemeinerung von Hamilton

Klassische Hamilton-Mechanik:
```
ẋᵢ = {xᵢ, H}_PB = εᵢⱼ ∂H/∂xⱼ
```
Ein Hamiltonian, eine Erhaltungsgröße.

Nambu-Mechanik (Nambu 1973, Eq. 3.2):
```
ẋᵢ = {xᵢ, H₁, H₂, ..., Hₙ₋₁}_NB
```
Für **3 Variablen** (n=3, unser Fall, Eq. 3.3):
```
ẋᵢ = {xᵢ, H₁, H₂}_NB = εᵢⱼₖ ∂ⱼH₁ ∂ₖH₂ = ∇H₁ × ∇H₂
```

Kompakt: **ẋ = ∇H₁ × ∇H₂**

### 2.2 Geometrische Bedeutung

∇H₁ steht senkrecht auf der Fläche H₁=const.
∇H₂ steht senkrecht auf der Fläche H₂=const.
∇H₁ × ∇H₂ zeigt **entlang der Schnittlinie** beider Flächen.

→ **Die Trajektorie liegt auf dem Schnitt H₁=k₁ ∩ H₂=k₂.**

Das ist die fundamentale geometrische Aussage: Die Bahn ist durch die Geometrie zweier Flächen vollständig bestimmt — nicht durch numerische Integration.

### 2.3 Liouville-Theorem in Nambu

Nambu-Dynamik ist **volumenerhaltend** (∇·(∇H₁×∇H₂) = 0 immer).
Daher gilt: reine Nambu-Mechanik → kein Attraktor möglich (kein Volumenverlust).
Für Attraktor **muss Dissipation** hinzugefügt werden.

### 2.4 Kanonische Transformationsinvarianz (Eq. 3.6-3.7)

Transformation H₁,H₂ → H₁',H₂' erlaubt, wenn Jacobi-Determinante = 1:
```
|∂(H₁',H₂')/∂(H₁,H₂)| = 1
```
Die vier möglichen Nambu-Flächentypen (unter kanonischen Transformationen):
**Hyperboloid, Paraboloid, Ellipsoid, Zylinder**

---

## 3. Nambu-Zerlegung des Wang-Systems

### 3.1 Helmholtz-Hodge-Zerlegung

Der Fluss wird in zwei orthogonale Teile zerlegt:
```
v = v_ND + v_D
```
mit Bedingungen:
```
∇·v_ND = 0  (divergenzfrei, rotational = konservativ)
∇×v_D = 0   (rotationsfrei, irrotational = dissipativ)
```

### 3.2 Explizite Zerlegung für Wang-System

**Nicht-dissipativer Teil** (Nambu-Teil):
```
v_ND = (−ay − yz,  xz,  dx + xy)
```

**Dissipativer Teil:**
```
v_D = (ax, −by, −cz)
```

Probe: v_ND + v_D = (a(x−y)−yz, −by+xz, −cz+dx+xy) ✓ = Wang-System

### 3.3 Die Nambu-Hamiltonians (Eq. 4.7) — KERNSTÜCK

```
H₁ = z² − (y+d)²

H₂ = ¼(x²+y²) + ½az − ½ad·log|z+y+d|
```

**Verifikation:** ∇H₁ × ∇H₂ = v_ND ✓ (im Paper gezeigt)

### 3.4 Geometrie der Nambu-Flächen

**H₁ = z² − (y+d)² = k₁:**
- Bei k₁=0: z = ±(y+d) → zwei Ebenen, Schnitt entlang (y+d=0, z=0)
- Für k₁≠0: **Hyperboloid** in der (y,z)-Ebene
- Hat Nullsteigung parallel zur x-Achse → "zwei Blätter" = X-Form in FIG.7 (Cyan-Fläche)
- Diese Fläche ist **immer hyperbolisch** — unabhängig von Parameterwerten!

**H₂ = k₂:**
- Stellt ein **deformiertes Paraboloid/Ellipsoid** dar (Gold-Fläche in FIG.7)
- Form hängt von Parametern a, d ab
- Kann 1, 2 oder 4 Äste der H₁-Fläche schneiden

### 3.5 Mechanismus der Scroll-Entstehung

**Die Scroll-Anzahl = Anzahl der Schnittkomponenten H₁=k₁ ∩ H₂=k₂**

- H₁-Fläche: immer X-förmig (4 Teilflächen, paarweise)
- H₂-Ellipsoid berührt 1 Ast von H₁ → 1 geschlossene Schnittlinie → 1-Scroll
- H₂-Ellipsoid berührt 2 Äste von H₁ → 2 Schnittlinien → **2-Scroll**
- H₂-Ellipsoid berührt alle 4 Äste von H₁ → 4-Scroll

Parameter a,d bestimmen Form von H₂ → bestimmen Scroll-Anzahl.
Parameter b,c nur im dissipativen Teil → ändern Scroll-Zahl nicht, nur Chaosgrad.

---

## 4. Vollständige Dynamik mit Dissipation

### 4.1 Dissipative Nambu-Form (Eq. 4.9)

Mit η₁=−a, η₂=b, η₃=c:
```
ẋ = (∇H₁ × ∇H₂)₁ − η₁·x
ẏ = (∇H₁ × ∇H₂)₂ − η₂·y
ż = (∇H₁ × ∇H₂)₃ − η₃·z
```

### 4.2 Neue Variablen (Eq. 4.10)
```
u = e^{η₁t}·x,  v = e^{η₂t}·y,  w = e^{η₃t}·z
τ = (1/(η₁+η₂+η₃)) · e^{(η₁+η₂+η₃)t}
```
In (u,v,w)-Koordinaten ist das System wieder divergenzfrei:
```
d/dτ (u,v,w) = ∇_q̃ H₁(e^{−η₁t}u, e^{−η₂t}v, e^{−η₃t}w) × ∇_q̃ H₂(...)
```
Die Hamiltonians werden **zeitabhängig** — kontinuierlich deformierende Flächen.

### 4.3 Dissipationsfunktion (Eq. 4.12)
```
D = ½(ax² − by² − cz²)
```

### 4.4 Zeitentwicklung von H₁, H₂, D (Eq. 4.13)
```
Ḣ₁ = ∇H₁·v = ∇H₁·v_D  =  −2az·z + 2(y+d)b·y  ... (nicht Null!)
Ḣ₂ = ∇H₂·v = ∇H₂·v_D  (nicht Null)
Ḋ  = ∇D·v   = {D, H₁, H₂} + (∇D)²
```
Mit Dissipation sind H₁, H₂ **keine Erhaltungsgrößen mehr** → Flächen deformieren sich → System springt zwischen Orbits → Chaos.

---

## 5. Verbindung zum BTC-PCA-Attraktor

### 5.1 Was wir gefunden haben

Aus `attractor_analysis.py` / `canonical_cycle.py`:
- TAU=30, M=50, SMOOTH_SIGMA=60, QuantReg q=0.01 (PL-Bottom)
- PCA-Varianzen: PC1≈37%, PC2≈33%, PC3≈6.5%
- PCA-Zyklen (θ = arctan2(PC2,PC1) unwrapped, Kreuzungen bei k·2π):
  - Zyklus 1: Start → 2018-05
  - Zyklus 2: 2018-05 → 2021-12  ← kanonischer Zyklus (single.py, canonical_cycle.py)
  - Zyklus 3: 2021-12 → 2025-07

**Kanonische Fourier-Parametrisierung (Ordnung 2, aus canonical_cycle.py):**
```
PC1(t) = +0.277 + 3.103·cos(t) + 1.315·sin(t) + 0.170·cos(2t) − 0.042·sin(2t)
PC2(t) = +1.295 − 1.293·cos(t) + 2.940·sin(t) − 0.013·cos(2t) + 0.086·sin(2t)
PC3(t) = −0.105 + 0.052·cos(t) − 0.174·sin(t) + 1.483·cos(2t) − 0.429·sin(2t)
```
**Schlüsselbeobachtung:** PC3 wird von cos(2t) dominiert (Amplitude 1.48 vs. 0.05 für cos(t)).

### 5.2 Analytische Sattelform (aus derive_saddle_analytically() in canonical_cycle.py)

Mit X = PC1−0.277, Y = PC2−1.295:
```
PC3 ≈ −0.105
    − 0.00682·X − 0.05619·Y
    + 0.06040·X² − 0.25374·XY − 0.07015·Y²
```
Quadratische Matrix M = [[+0.0604, −0.1269], [−0.1269, −0.0702]]:
- det(M) = −0.0203 < 0 → **Hyperbolisches Paraboloid (Sattel)** ✓
- Eigenwerte: λ₁ = +0.1378, λ₂ = −0.1476
- Hauptachsenform: PC3 ≈ z₀ + 0.1378·u² − 0.1476·v²
- Kanonische Parameter: **d ≈ 2.69, e ≈ 2.60, f = 1** (fast symmetrisch)

### 5.3 Die Brücke: H₁ im Wang-System ↔ Sattelform im BTC

Wang-System H₁:
```
H₁ = z² − (y+d)²  =  z² − y² − 2dy − d²
```
Das ist eine quadratische Form in (y,z) — genau wie unsere:
```
PC3 ~ +α·PC1² − β·PC2²  mit α,β > 0
```
**Strukturelle Übereinstimmung:** H₁(z,y) ↔ PC3(PC1,PC2) — beide sind indefinite quadratische Formen mit negativer Determinante.

### 5.4 Interpretation: BTC als 2-Scroll-System

| Wang-System | BTC-PCA-Attraktor |
|-------------|-------------------|
| x, y, z | PC1, PC2, PC3 |
| H₁ = z²−(y+d)² | PC3 ≈ αPC1²−βPC2² (Sattelform) |
| H₂ = deform. Ellipsoid | unbekannt — zu rekonstruieren |
| 2 Scrolls (a=3.5, b=9, c=5, d=0.06) | 2 vollständige PCA-Zyklen sichtbar |
| Dissipation η₁,η₂,η₃ | Rosenstein-Lyapunov λ₁≈0.71/yr |
| Index-2-Sattelfoki S2-S5 | BTC-Gleichgewichtspunkte ≈ PL-Bottom-Niveaus? |

---

## 6. Offene Forschungsfrage: Nambu-Rekonstruktion aus Messdaten

### 6.1 Das Problem

Gegeben: empirische Trajektorie (PC1(t), PC2(t), PC3(t)) aus BTC-Daten.
Gesucht: H₁, H₂ so dass ∇H₁ × ∇H₂ ≈ v_ND (nicht-dissipativer Flussanteil).

### 6.2 Lösungsansatz

**Schritt 1: Fluss schätzen**
```python
v(t) = d/dt [PC1, PC2, PC3]  # numerische Ableitung der geglätteten Trajektorie
```

**Schritt 2: Helmholtz-Hodge-Zerlegung**
```
v = v_ND + v_D
∇·v_ND = 0,  ∇×v_D = 0
```
In diskreter Form: v_D = ∇φ (Potential-Anteil), v_ND = ∇×A (Wirbel-Anteil).
Implementierbar via: v_D ≈ Σᵢ ηᵢ·xᵢ·eᵢ mit ηᵢ geschätzt aus Lyapunov-Analyse.

**Schritt 3: H₁ aus Invarianzbedingung finden**

Notwendige Bedingung: v_ND·∇H₁ = 0 (H₁ ist Erhaltungsgröße unter v_ND).
```
(∇H₁ × ∇H₂) · ∇H₁ = 0  identisch erfüllt
→ v_ND · ∇H₁ = 0
```
Ansatz: H₁ als Polynom in (PC1, PC2, PC3) — z.B. quadratisch:
```
H₁ = a₁·PC3² + a₂·(PC2+d)² + a₃·PC3·PC2 + ...
```
Dann: v_ND·∇H₁ = 0 entlang der Trajektorie → **lineares Gleichungssystem für aᵢ**.

**Schritt 4: H₂ analog**

v_ND·∇H₂ = 0 → zweites lineares System.
Dann prüfen: ∇H₁ × ∇H₂ =? v_ND.

**Schritt 5: Dissipation schätzen**
```
D = ½(η₁·PC1² − η₂·PC2² − η₃·PC3²)
```
mit ηᵢ aus der Divergenz: ∇·v = −η₁−η₂+η₃ = Lyapunov-Summe.

### 6.3 Erwartete Form von H₁ für BTC

Aus der Sattelstruktur (Abschnitt 5.2):
```
H₁_BTC = PC3² − (f(PC1,PC2))²
```
wobei f eine Linearkombination von PC1, PC2 ist — analog zu (y+d) im Wang-System.
Der Parameter d im Wang-System entspricht dem DC-Offset der Gleichgewichtspunkte.

Konkret: H₁_BTC ≈ PC3² − (c₁·PC1 + c₂·PC2 + c₃)²  mit c₁,c₂,c₃ zu bestimmen.

### 6.4 Machbarkeitsgrenze

Mit nur ~1320 Datenpunkten (Zyklus 2) und 3D-Einbettung:
- Polynom-Ansatz bis Grad 2: 10 freie Parameter für H₁
- 1320 Gleichungen (v_ND·∇H₁ = 0) → überbestimmtes System → LS-Lösung
- Regularisierung nötig (Ridge-Regression oder physikalisch motivierte Constraints)

**Hauptrisiko:** v_ND ≈ v − v_D erfordert, dass die Dissipationsraten η₁,η₂,η₃ bekannt sind. Diese sind nicht direkt aus PCA-Daten messbar — müssen aus Lyapunov-Analyse oder Bifurkationsdiagramm geschätzt werden.

---

## 7. Nächste Schritte (nach Rücksprache)

1. **Simulation Wang-System numerisch** mit a=3.5, b=9, c=5, d=0.06 (Fig 6b(ii))
   → `attractor_2scroll_eq.py` implementiert: Nambu-Flächen H₁, H₂ + Schnittkurve + S1–S5
   → Gleichgewichte: S1=(0,0,0), S2=(−6.774,−3.108,+4.129), S3=(−6.674,+5.658,−7.629),
     S4=(+6.643,+3.048,+4.129), S5=(+6.744,−5.718,−7.629)

2. **Helmholtz-Zerlegung der BTC-Trajektorie** implementieren
   → v_ND und v_D aus PC1/PC2/PC3-Zeitreihe schätzen

3. **H₁ aus Invarianzbedingung** v_ND·∇H₁ = 0 per LS lösen
   → Quadratischer Ansatz: 10 Parameter

4. **H₂ analog** lösen, dann ∇H₁×∇H₂ auf Übereinstimmung prüfen

5. **Geometrische Visualisierung:** H₁- und H₂-Flächen im (PC1,PC2,PC3)-Raum plotten
   → Vergleich mit FIG.7 aus Paper (sollte X-Form für H₁, Ellipsoid für H₂ ergeben)

---

## 8. Technische Dateien

| Datei | Inhalt |
|-------|--------|
| `attractor_2scroll_eq.py` | Wang 2-Scroll (a=3.5,b=9,c=5,d=0.06): Nambu-Flächen H₁∩H₂, S1–S5, → attractor-paper.png |
| `attractor_analysis.py` | Haupt-Analyse: Embedding, PCA, FTLE, Rosenstein-Lyapunov, Plots |
| `single.py` | Einzelner PCA-Zyklus (CYCLE_NR=1,2,3), Sattel-Fit |
| `canonical_cycle.py` | Fourier-kanonischer Zyklus, analytische Sattelform, `derive_saddle_analytically()` |
| `attractor_analysis_mean.py` | Wie attractor_analysis.py aber mit PL-Mean statt Bottom |
| `attractor_raw.py` | Roher BTC-Preis statt Residuen (offene Spirale) |
| `ziel.csv` | BTC-Tagesdaten: day_index, price, date |

**Wichtige Parameter:**
```python
TAU = 30          # Delay-Embedding Zeitverzögerung [Tage]
M = 50            # Embedding-Dimension
SMOOTH_SIGMA = 60 # Gauss-Glättung [Tage]
PERCENTILE = 0.01 # QuantReg für Power-Law-Bottom
START_IDX = 1164  # Ab diesem Day-Index wird eingebettet
SEGMENT_SLICE = slice(785, 2105)  # Zyklus 2 im PCA-Raum
```

**Lyapunov-Ergebnis:** λ₁ ≈ 0.71/yr (Rosenstein), FTLE-Schwelle 2/yr für Expansions-/Kontraktionsregimes.

---

## 9. Maßstab: Strukturerhaltung statt Parametererhaltung

**Der Maßstab ist nicht Parametererhaltung, sondern Formerhaltung von H₁ als Sattelstruktur.**
Alles andere ist nachgeordnet.

### 9.1 Der konservative Kern

H₁ = z² − (y+d)² ist **immer hyperbolisch** (X-Form, Sattelstruktur) — strukturell fest, unabhängig von Parameterwerten. H₂ liefert die zweite Fläche, deren Schnitt mit H₁ die Scroll-Komponenten erzeugt. Die orange Linie ist geometrisch an die H₁/H₂-Struktur gebunden, nicht an einzelne Parameterwerte.

### 9.2 Der eigentliche Invarianzbegriff

Nicht: gleiche Parameter, gleiche Achsen, gleiche PCA-Orientierung.

Sondern:
- gleiche **Sattelklasse** von H₁
- gleiche qualitative **Schnittstruktur** H₁=k₁ ∩ H₂=k₂
- bei Dissipation: reale Trajektorie bleibt nahe dieser Struktur

### 9.3 Was sich ändern darf

**Alle Parameter dürfen geändert werden, sofern die H-Struktur mit Sattelcharakter erhalten bleibt und die Schnittgeometrie weiterhin den gewünschten Orbit trägt.**

- a, d → bestimmen Form von H₂ → bestimmen Scroll-Anzahl → dürfen variieren solange Sattelklasse erhalten
- b, c → nur dissipativer Teil → dürfen frei variieren (ändern Scroll-Zahl nicht)

### 9.4 Präzisierung H₁

H₁ = z²−(y+d)² ist streng genommen kein hyperbolisches Paraboloid, sondern eine **hyperbolische Quadrik**, die entlang x entartet ist (X-Form). Entscheidend ist:
- indefinite quadratische Form
- Sattelcharakter
- zwei gegenüberliegende Äste
- robuste Schnittgeometrie mit H₂

### 9.5 Leitfrage für weitere Arbeit

**Welche Transformationen erlaubst du, solange H₁ in eine Sattelform überführt werden kann?**

Gesuchtes System:
```
Gesucht ist ein dissipatives System, dessen konservatives Skelett ein H₁
mit Sattelklasse besitzt und dessen Attraktor nahe den zugehörigen
Schnittorbits liegt.
```

Offene Frage: Soll H₁ funktional nahe bei z²−(y+d)² bleiben, oder reicht jede äquivalente Sattelform nach Koordinatentransformation?
