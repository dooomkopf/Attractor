# Aperiodic-Problem — Stand 2026-04-29

## Worum es geht

BTC-Log-Residuen zeigen eine oszillatorische Struktur, die **nicht in eine einzige
Mode-Familie** passt. Datengetriebene SSM-Diagnostik (`analyze_residuals/10_cli_ssm_learn.py`)
liefert in zwei verschiedenen Zeit-Uhren je einen sauber erkennbaren Master, aber
nie beide gleichzeitig:

| Uhr | Master-Periode | Slave R² (PC3 / PC4) | Beobachtung |
|---|---|---|---|
| linear (Tage) | T = 3.846 Jahre | 0.50 / 0.48 | Halving-Zyklus, klassisch linear-periodisch |
| log₁₀(Tage) | T = 0.302 = log₁₀(2.005) | 0.16 / 0.08 | Master mit Skalierungsfaktor 10^0.302 ≈ 2 |

Linear- und log₁₀-Uhr finden je einen Master an unterschiedlichen Frequenzen; die
Slave-R² liegen in beiden Uhren deutlich unter der Wang-Referenz (0.978). Das ist
der **strukturelle Hinweis auf Mehr-Skalen-Verhalten** — die Identifikation der
zweiten Skala (log-periodische LPPL-Struktur o.ä.) erfolgt durch externe Analysen,
nicht hier.

## Die drei Analyse-Pipelines

| Pipeline | Material | Methode | Aktuelles Ergebnis |
|---|---|---|---|
| `analyze_wang/` | analytische Wang-2-scroll-DGL | DGL-driven SSM (Theorem 1–4) + SSMLearn | Slave R² = 0.978, sauber linear-periodisch |
| `analyze_LPPL/` | analytische LPPL+Wang-DGL | wie oben | strukturell identisch zu Wang (R²=0, W2≠0) |
| `analyze_residuals/` | echte BTC-Daten | NUR datengetriebenes SSMLearn (kein DGL-Zugang) | R²=0.50 (linear) bzw. 0.16 (log₁₀) |

Der R²-Verlust von Wang (0.978) zu BTC (0.50) ist der quantitative Hinweis auf
**zusätzliche Struktur außerhalb der lokalen 2D-SSM**.

## Der Theorierahmen — und seine Grenze

**Aperiodic-SSM-Paper** (Haller & Kaundinya, *Chaos* 2024, arXiv:2404.05355) behandelt
Systeme der Form

    ẋ = A x + f₀(x) + f₁(x,t),    f₀(x) = o(|x|)

mit explizit bekanntem Vektorfeld. Die Theorie ist **strikt equation-driven** —
nicht direkt auf BTC-Beobachtungen anwendbar. Brücke zur Realität geht nur über
ein DGL-Modell, das anschließend gegen Daten validiert wird.

## Die User-DGL als Brücke

Verwendete Equation: `lpplattr01_ode.py` (LPPL+Wang+Sign-OU, M=1-Vereinfachung).

Aufspaltung am Fixpunkt (Wang-Konvention, Anker an **S₄ oder S₅** — Spiegelbild-Paar
mit z=−6.80, NICHT die mathematischen Listen-Indices E₂/E₃):

    A    = Jacobian an S₄ bzw. S₅
    f₀(x) = LPPL-Polynom-NL + Wang-Bilinearterme + kub. Rückstellterm
    f₁(x,t) = Halving-getriebene Drifts (mu_offset_by_cycle, kappa-Stiffness)

Eine etwaige **log-periodische Struktur** der Residuen ist hier weder gezeigt
noch vorausgesetzt — externe FFT-Analysen (außerhalb dieser Pipeline) deuten
darauf hin. Falls real, würde sie strukturell als **autonomer Transient aus den
Anfangsbedingungen** der ersten 0–300 Tage interpretiert — getragen von diskreter
Skaleninvarianz in f₀(x) plus x(0). Damit bräuchte der Paper-Apparat **keine
unbekannte log-periodische Forcing-Komponente**.

## Festgelegte konzeptionelle Vorgaben

| Setting | Wert | Begründung |
|---|---|---|
| t_c | 0 (Genesis = ~3.1.2009) | Big-Bang-Variante, invertiert zur Standard-LPPL — Perioden wachsen, schrumpfen nicht |
| M | 1 (statt 1.071) | empirisch identisch, mathematisch C^∞ |
| SSM-Anker | S₄ oder S₅ (Wang) bzw. E₂ (LPPL) | wo die Trajektorien tatsächlich kreisen |
| Polynom-Grad | poly = 2 | enthält 2ω automatisch; höhere Grade overfitten |
| Slave-Schwelle | 0.07 | empirisch sinnvoll oberhalb des Rausch-R² ~0.005 |
| Zeit-Achse log-Uhr | log₁₀ | konsistent zu anderen Pipelines des Projekts |

Externe Erwartungswerte (NICHT in Pipeline eingespeist, nur als Sanity-Vergleich):
Λ ≈ 2 und 2–3 sichtbare Harmonische aus FFT-Analysen außerhalb dieser Pipeline.

## Was das Paper liefert, sobald die Aufspaltung steht

| Theorem | Liefert | Brauche ich dafür |
|---|---|---|
| Theorem 1 | Existenz Anchor-Trajektorie x*(t) | nur f₁ am Fixpunkt + Schranke |
| Theorem 2 | x*(t) ≈ ∫ e^{A(t−τ)} f₁(0,τ) dτ + höhere Ordnungen | nur Halvings als f₁ |
| Theorem 3+4 | reduzierte Dynamik auf 2D-SSM mit Forcing-Term | A, f₀, f₁ |
| Eq. (41) | ξ̇ = A^u ξ + Q_u f₀(P[ξ; h₀(ξ)]) + ε Q_u f̃₁(t) | dasselbe Inventar |

Alles davon ist mit der DGL berechenbar **ohne unbekannte Größen**.

## Offene Hypothese (zentrale Validierungsfrage)

Annahme zur Validierung: **falls** die externen FFT-Hinweise auf log-periodische
Struktur reale Eigenschaft der BTC-Residuen sind, **dann** sei diese Struktur
nur ein autonomer Transient aus Anfangsbedingungen, kein externes log-Forcing.

Test:

- Simuliere `lpplattr01_ode.py` mit realistischen x(0) + nur Halving-f₁
- Erwartung bei Annahme-richtig: log-periodisch-ähnliche Phase in den ersten
  0–~3000 Tagen, danach Übergang zu reiner Halving-Mode
- Wenn Test bestanden → Paper-Apparat passt, keine f₁-LPPL-Komponente nötig
- Wenn nicht → es gibt doch einen unbekannten externen log-Antrieb,
  Plan B mit explizitem f₁^{LPPL}(t)

## Vorgeschlagene nächste Schritte (klein, sequentiell)

1. **Theorie**: A, f₀, f₁ aus `lpplattr01_ode.py` + `lpplattr01_params.py`
   am Fixpunkt S₄ (oder S₅) explizit hinschreiben — kein Daten-Touch
2. **Simulation**: DGL mit dieser Aufspaltung simulieren, echte ICs
3. **Vergleich**: Simulation durch `analyze_residuals/`-Pipeline + parallel echte BTC-Daten;
   Spektren und R² vergleichen
4. **Falls nötig**: log-periodischen Term in f₁ einbauen und schauen, ob es besser
   passt als der reine IC-Transient

## Beobachtungs-Asymmetrien, die nicht aus den Augen verloren werden dürfen

- Wang R²=0.978 ohne Forcing, BTC R²=0.50 mit (vermutlich) Halving-f₁:
  der R²-Drop bezeichnet Reichweite der Transient-Region außerhalb der lokalen SSM
- log₁₀-Master 0.302 = log₁₀(2.005) liegt erstaunlich nahe am Halving-Faktor 2 —
  ein nicht-trivialer numerischer Befund. Die Interpretation als LPPL-Λ gehört
  zu externen Analysen, nicht zu dieser Pipeline.
- Neither-clock-captures-everything-Argument ist der **strukturelle Mehr-Skalen-Beleg**
  unabhängig von Schwellen-Konventionen

---

# Stand-Update 2026-04-29 (späterer Tag) — Stufe-A-Implementierung + Bilanz

## Was implementiert wurde

`analyze_LPPL/11_cli_ssm_whisker_adiabatic.py` — adiabatischer Whisker-Sweep
über die 18 (Cycle, Phase)-Tupel von `SIGN_OU['mu_offset_by_cycle']`. Ruft die
existierende `compute_whisker_order2` an einem Operating-Point auf, der
parametrisch von $\alpha$ = mu_offset abhängt.

**Wichtig zum Anker (Klärung der Frage "wo war unser Anker"):**

Der naive Stufe-A-Anker WAR NICHT der Schwerpunkt der Trajektorie,
und auch NICHT der Origin (0,0,0). Der Anker war:

```python
operating_point = (y1_E2, alpha / t_ref, z_E2)
```

— also der **autonome Fixpunkt $E_2 \approx (-0.006, 0, 0)$** mit nur
einer y₂-Verschiebung um den Sign-OU-Sollwert $\alpha/t_{\text{ref}}$.
$y_1$ und $z$ blieben am autonomen $E_2$-Wert.

## Centroid-Validierung — der Anker war methodisch nicht haltbar

Numerische Centroid-Bestimmung durch Simulation der vollen DGL
(smooth ODE + stochastisches Sign-OU mit $\sigma > 0$) bei eingefrorenem $\alpha$:

| $\alpha$ | naiver Anker (E₂-basiert) | echter Schwerpunkt $\langle x\rangle_t$ |
|---|---|---|
| 0 | $(-6\!\cdot\!10^{-3},\ 0,\ -5\!\cdot\!10^{-6})$ | $(-0.105,\ -7\!\cdot\!10^{-5},\ -2\!\cdot\!10^{-3})$ |
| +5 | $(-6\!\cdot\!10^{-3},\ +1.4\!\cdot\!10^{-3},\ -5\!\cdot\!10^{-6})$ | $(+1.14,\ +1.2\!\cdot\!10^{-3},\ -0.137)$ |
| -5 | $(-6\!\cdot\!10^{-3},\ -1.4\!\cdot\!10^{-3},\ -5\!\cdot\!10^{-6})$ | $(-1.17,\ -1.3\!\cdot\!10^{-3},\ -0.138)$ |
| ±10..15 | $(-6\!\cdot\!10^{-3},\ \pm 4\!\cdot\!10^{-3},\ -5\!\cdot\!10^{-6})$ | $\langle z\rangle \approx +102$ — komplett anderes Regime |

Die echte Trajektorie lebt **weit weg** von $E_2$. Bei großen $|\alpha|$ läuft das System
in einen Bereich mit $\langle z\rangle \sim 100$ — möglicherweise ein anderer
Wang-2-scroll-Lobe, jedenfalls keine Perturbation von $E_2$.

## Whisker am echten Schwerpunkt — bricht zusammen

Centroid-Modus (`--anchor centroid`) in `11_cli_ssm_whisker_adiabatic.py`:

| Spektrum-Klasse am Centroid | naive (E₂) | centroid (echte Sim) |
|---|---|---|
| stable Fokus (komplexes Paar, Re < 0) | 10/18 | **0/18** |
| UNSTABLE Fokus (komplexes Paar, Re > 0) | 8/18 | 4/18 |
| **REAL (kein komplexes Paar)** | 0/18 | **14/18** |
| Continuous Sweep valid | 120/121 | **0/21** |

Bedeutung: in 14 von 18 Cycle/Phase-Slices ist der Schwerpunkt in einer
**rein hyperbolischen Region** (saddle-saddle-Regime, alle Eigenwerte reell).
Die Standard-Whisker-Maschinerie (verlangt komplexes Master-Paar) **greift dort
strukturell nicht**.

## Vergleich Wang-Baseline vs LPPL+Wang autonomous

Standalone-Whisker an autonomen Fixpunkten (kein Sign-OU):

| | Wang an $S_4$ | LPPL+Wang an $E_2$ |
|---|---|---|
| Master-Eigenwerte | $+1.64 \pm 7.46\,i$ | $-3.6\!\cdot\!10^{-4} \pm 9.8\!\cdot\!10^{-4}\,i$ |
| Master-Periode | $0.84$ | $\approx 6430$ d |
| $|W_2(2\omega)|$ max | $0.028$ | **$30.9$** (1000-fach größer!) |
| $|W_2(\text{DC})|$ max | $0.060$ | **$178.5$** |
| cond$(C_k)$ | $3.10$ | $1.24\!\cdot\!10^6$ |
| $R_2$ | $0$ | $0$ |

LPPL+Wang an $E_2$ hat **nahe-singuläre Jacobian** (Eigenwerte $\sim 10^{-4}$,
Konditionszahl $\sim 10^6$). Die rohen $|W_2|$-Werte sind wegen der schlecht
konditionierten Cohomologie-Gleichung künstlich aufgeblasen — das ist Numerik,
nicht Physik.

Wang ist sauber konditioniert, weil seine Eigenwerte $\mathcal{O}(1)$ sind.

## Bilanz: das Paper passt nicht 1:1 auf unsere DGL

Fünf konkrete Hürden, die zusammen das Paper-Apparat ausschließen:

1. **Spektrale Pathologie an $E_2$:** Eigenwerte $\sim 10^{-4}$, cond$(A) \sim 10^6$
   → Linearisierungs-Mathematik instabil, Cohomologie-Gleichung nahe-singulär
2. **Forcing nicht klein:** $|f_1| \sim 5\!\cdot\!10^{-3}$ vs. lokale Spektral-Skala
   $\sim 10^{-4}$ → Theorem-1-Smallness-Bedingung verletzt
3. **Anker existiert nicht im Paper-Sinn:** Theorem 1 verlangt eindeutige
   beschränkte Trajektorie nahe $E_2$. Der echte Schwerpunkt liegt weit weg
4. **Section 3 (adiabatic) passt auch nicht:** Sign-OU ist bang-bang
   (diskontinuierlich in $x$), stochastisch ($\sigma > 0$), springt an
   Phase-Übergängen. Section 3 verlangt glatte langsame Drifts
5. **Stochastik außerhalb Paper:** Theoreme 1-6 sind alle deterministisch.
   Stochastische Forcings sind nicht abgedeckt

## Was nutzbar bleibt

| Werkzeug | Was es liefert |
|---|---|
| **Wang-Baseline-Whisker** an $S_4$ | sauber konditioniert, $R_2 = 0$, $|W_2| \sim 0.03$. Beweist 2$\omega$-geometrisch im chaotischen Wang-Regime |
| **`analyze_LPPL/08_cli_ssm_whisker.py`** an $E_2$ | gibt Whisker formal, aber numerisch fragil. Als algebraischer Proxy verstehbar |
| **datengetriebenes SSMLearn** auf simulierter oder echter Trajektorie | findet effektive 2D-SSM ohne Anker-Annahme. Liefert PC3-Slave-R² direkt |
| **`analyze_LPPL/12_cli_centroid_check.py`** | dokumentiert wo die Trajektorie tatsächlich lebt — wichtige Validations-Größe für jedes DGL-Modell |

## Pragmatischer Pfad nach vorne

Statt Paper-konforme aperiodic-SSM-Theorie auf BTC-DGL anzuwenden
(geht nicht), die folgenden Bausteine als **Validations-Brücke**:

1. **DGL simulieren** (`lpplattr02_ode` mit $\sigma > 0$, vollem Sign-OU + Halvings)
2. **Trajektorie durch SSMLearn-Pipeline** schicken (existiert bereits in `analyze_residuals/`)
3. **Vergleich**: SSMLearn-Diagnostik der Simulation vs. der echten BTC-Daten
4. Wenn Diagnostik übereinstimmt → DGL ist gutes Modell. Wenn nicht → DGL anpassen
5. **Wang-Baseline** als sauber-konditionierte Kalibrierungsgröße im Hintergrund

Dieser Pfad umgeht den Paper-Anker-Albtraum und nutzt SSMLearn als
**modell-agnostisches** Vergleichswerkzeug. Stufe A in der jetzigen Form
(11_cli_ssm_whisker_adiabatic.py) bleibt als illustratives Diagnose-Werkzeug
liegen, ist aber methodisch nicht das Hauptinstrument.

## Zusatzbefund zur log-periodischen Frage

Externe FFT-Analysen postulieren $\Lambda \approx 2$ und einige Harmonische.
Datenseitig findet `analyze_residuals/10_cli_ssm_learn.py --time_mode log`
einen Master mit Periode $\log_{10}(2.005)$ — also $\Lambda \approx 2$ direkt
aus den Daten ohne Vorgabe. Das bestätigt die externen Analysen, ohne sie zu
brauchen, und es bestätigt den **Mehr-Skalen-Charakter** der Residuen.

Kombiniert mit $T = 3.846$ Jahre in linearer Uhr (Halving-Zyklus) ergibt sich:
linear- und log-Uhr finden je einen Master, aber an unterschiedlichen Frequenzen.
Keine einzelne Uhr fängt beide Strukturen gleichzeitig ein. Das ist der
strukturelle Mehr-Skalen-Beleg, unabhängig vom Anker-Problem.
