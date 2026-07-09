# Wang Harmonic Scaling Analysis — Review

## 1. Ist Plot 3 mathematisch sinnvoll oder irreführend? Warum genau?

**Plot 3 ist irreführend.** Das Problem liegt in der Zentrierung mit dem Median statt mit dem durch-Ursprung-gefitteten Modell:

```python
# Zeile 90-91 in cli_scaling.py
pred_masked = scaling['slope_zero'] * scaling['x_masked']
pred_anom = pred_masked - float(np.median(pred_masked))
obs_anom = scaling['y_masked'] - float(np.median(scaling['y_masked']))
```

Die Zentrierung subtrahiert unabhängige Mediane von x und y, was die lineare Beziehung `y = c*x` zerstört. Bei nahezu konstanten Envelopes (CV ≈ 5%) sind beide Mediane fast gleich dem Mittelwert. Nach Subtraktion bleibt nur Rauschen um Null übrig — keine Diagonale sichtbar.

## 2. Wie kann Plot 2 gut aussehen, während Plot 3 scheinbar keine Übereinstimmung zeigt?

Plot 2 zeigt die **absoluten** Werte: `y_harm` vs `c*x_main²` über Zeit. Beide oszillieren um ähnliche Mittelwerte (≈18), daher sehen sie ähnlich aus.

Plot 3 zeigt **Anomalien** nach Median-Subtraktion. Da die Signale fast konstant sind (CV < 5%), entfernt die Median-Subtraktion fast das gesamte Signal. Was bleibt: minimale Fluktuationen (±0.5 bei Mittelwert 18) — das Signal-Rausch-Verhältnis kollabiert.

Mathematisch: Bei `y = c*x + noise` mit kleinem noise und fast konstantem x,y:
- Plot 2: zeigt y vs c*x direkt → sichtbare Übereinstimmung
- Plot 3: zeigt (y - median(y)) vs (c*x - median(c*x)) → nur noise bleibt

## 3. Ist das R² through-origin hier eine schlechte Kennzahl? Wenn ja, wodurch wird es künstlich groß?

**Ja, R² through-origin ist hier eine schlechte Kennzahl.** Es wird künstlich groß durch:

```python
# scaling.py, Zeile 14-15
sst0 = float(np.dot(y, y))  # Summe der Quadrate von y
r2_zero = 1.0 - sse / max(sst0, 1e-30)
```

Bei fast konstanten Signalen mit großem Offset (y ≈ 18): 
- SST₀ = Σy² ≈ n * 18² ist riesig
- SSE = Σ(y - c*x)² ist klein relativ zu SST₀
- R² = 1 - SSE/SST₀ ≈ 0.9998

Das R² misst hier hauptsächlich, dass beide Signale einen ähnlichen DC-Offset haben, nicht die Qualität der quadratischen Skalierung.

## 4. Welche 1-2 besseren Diagnostiken/Plots würdest du stattdessen empfehlen?

### Diagnostik 1: Relative Amplitude-Variation
```python
rel_variation = (y_harm / np.mean(y_harm)) / (x_main_sq / np.mean(x_main_sq))
plot(t, rel_variation)  # sollte ≈ 1 sein wenn Skalierung stimmt
```

### Diagnostik 2: Residual-zu-Signal-Verhältnis
```python
residual = y_harm - slope_zero * x_main_sq
rsr = np.std(residual) / np.std(y_harm)  # < 0.1 wäre gut
```

### Plot-Verbesserung für Plot 3:
```python
# Statt unabhängige Mediane zu subtrahieren:
y_detrended = y_masked - fit['slope'] * np.median(x_masked)
x_detrended = x_masked - np.median(x_masked)
plot(fit['slope'] * x_detrended, y_detrended)  # behält lineare Struktur
```

## 5. Ist die "quadratisch erzeugte Harmonische"-Aussage durch die Wang-Ergebnisse gestützt?

**Die Phase-Lock-Aussage ist sehr gut gestützt** (R = 0.999, median |Δφ| = 147°).

**Die quadratische Amplitudenskalierung ist NICHT überzeugend nachgewiesen:**
- CV der Hauptamplitude nur 5.2% → zu wenig Variation für robuste Skalierungsanalyse
- Korrelation zwischen A² und A₂ω ist -0.075 (praktisch Null!)
- Das hohe R² = 0.9998 ist irreführend (DC-Offset-Artefakt)
- Die beobachteten Harmonischen könnten auch durch konstante Kopplung oder intrinsische 2ω-Dynamik entstehen

**Fazit:** Die Wang-Analyse bestätigt starkes Phase-Locking, aber die quadratische Amplitudenskalierung ist mit den vorliegenden Daten nicht belastbar nachweisbar. Das System zeigt zu wenig Amplitudenvariation für eine aussagekräftige Skalierungsanalyse.

## Empfehlungen

1. **Andere ICs testen** mit stärkerer Amplitudenvariation (CV > 20%)
2. **Parametersweep** über a,b,c,d für größere Amplitudendynamik
3. **Direkte numerische Tests**: Skaliere x(t) künstlich → prüfe ob Harmonische quadratisch mitskalieren
4. **Alternative Hypothese testen**: Konstante Harmonische vs quadratische Skalierung mit Likelihood-Ratio-Test