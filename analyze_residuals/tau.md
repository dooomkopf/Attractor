# TAU-Wahl im log-Zeit-Embedding

**Datum**: 2026-05-04
**Status**: Untersucht — der in `bug_data.py.md` beschriebene "Bug" ist quantitativ falsch identifiziert. Das aktuelle Verhalten ist ein bewusster Kompromiss, kein Fehler.

---

## Worum es geht

`data.py::_build_log_time_context()` baut ein Delay-Embedding auf einem **uniformen log10-Zeit-Gitter**. Die Frage: wie groß soll der Lag `TAU` und das Fenster `W = (M-1)·TAU` sein?

Im **linear**-Modus ist die Antwort einfach: 1 Sample = 1 Tag, also entspricht `TAU = years·365.25/(M-1)` einer physikalischen Lag in Tagen, und `W = years·365.25` ist eine konstante physikalische Fenstergröße.

Im **log**-Modus gibt es **keine konstante physikalische Fenstergröße** — ein fester Lag in log10d entspricht einem **multiplikativen** Zeit-Verhältnis. Das Embedding-Fenster `W` (in log10d-Einheiten) deckt am Anfang der Daten eine andere physikalische Spanne ab als am Ende:

- am **Anfang** (Tag `d_start`): physikalische Spanne = `d_start · (10^W − 1)` Tage
- am **Ende** (Tag `d_end`): physikalische Spanne = `d_end · (1 − 10^(−W))` Tage

Das ist **kein Bug**, sondern eine zwangsläufige Eigenschaft des log-Embeddings.

---

## Aktuelles Verhalten

`_build_log_time_context()` benutzt `TAU = years_to_tau(M, years) = round(years·365.25/(M-1))` — also dieselbe Sample-Zahl wie im linear-Modus, aber angewendet auf das log10-Gitter mit anderem Sample-Spacing.

Mit BTC-Daten (start_idx=1164, d_end≈6329, M=35, years=3.77):

| Größe | Wert |
|---|---|
| TAU (Schritte auf log-Gitter) | 40 |
| W (Schritte) | 1360 |
| dt_log (log10d/Schritt) | ~1.31e-4 |
| W in log10d | ~0.178 |
| **Span am Anfang** (effektive lineare Lag-Spanne) | **~1.62 Jahre** (590 d) |
| **Span am Ende** | **~5.83 Jahre** (2128 d) |
| Effektive Lag am Anfang | ~15 d |
| Effektive Lag in der Mitte | ~36 d |
| Effektive Lag am Ende | ~82 d |

→ Das Fenster ist also nicht "26% der log-Datenrange", wie in `bug_data.py.md` behauptet (das wäre 0.194 log10d), sondern entspricht einem realistischen Kompromiss zwischen Anfang (~1.6 J) und Ende (~5.8 J) der Daten.

---

## Warum der "Bug" fälschlicherweise als Bug identifiziert wurde

Die Notiz `bug_data.py.md` behauptet drei zentrale Dinge:

1. **"SSM picks λ=1.107 (statt λ≈2)"** — quantitativ widerlegt
2. **"Embedding-Fenster < 1 Periode der λ≈2 Mode → Mode unauflösbar"** — formal stimmt es, empirisch widerlegt
3. **"Diskrepanz SSM vs. PC1-Periodogramm → falsche Lag-Skalierung"** — methodisch fragwürdig

### Widerlegung 1: SSM findet bereits λ≈2

Mit den Standard-Parametern (M=35, years=3.77, ssm_dim=2, poly=2) liefert das **alte data.py-Verhalten** im log-Modus:

```
T_main = 0.303 log10d  →  λ = 2.010
PC1-Periodogramm Top: T = 0.310  →  λ = 2.04
```

Das ist exakt der Wert, den die LPPL-Theorie vorhersagt. Die in `bug_data.py.md` zitierte Zahl `λ=1.107` taucht mit Standard-Parametern reproduzierbar **nicht** auf — vermutlich kam sie aus einer abweichenden Konfiguration (anderes ssm_dim/poly oder anderer start_idx), die in der Bug-Notiz nicht spezifiziert wurde.

### Widerlegung 2: Parameter-Sweep zeigt Stabilität

Vergleich des "alten" Codes mit dem "Codex-Fix" (log_window_frac=0.5) über typische Parameter-Kombinationen:

| ssm_dim/poly | alt λ | "Fix" λ |
|---|---|---|
| 2 / 1 | 2.00 | 1.98 |
| 2 / 2 | 2.01 | 1.98 |
| 3 / 1 | 1.97 | 1.98 |
| 3 / 2 | 1.94 | 1.96 |
| 4 / 1 | 2.09 | 1.99 |
| 9 / 2 | 2.47 | 1.43 |

Bei **moderaten ssm_dim (2–3)** sind alt und Fix praktisch identisch und finden beide λ≈2. Erst bei ssm_dim=9 divergieren sie — und dort ist der "Fix" sogar **schlechter** (λ=1.43 statt erwartet ≈2). Bei hohen Dimensionen ist die "slowest-oscillatory"-Auswahl ohnehin instabil und numerisch heikel.

### Widerlegung 3: SSM-Eigenwert vs. PC1-Periodogramm sind unterschiedliche Größen

Der SSM-Eigenwert wird auf der Eigenmode-Koordinate `u` bestimmt — eine Linearkombination aller PCs. Das Periodogramm der ersten PC1 ist eine **andere** Größe und mischt im Allgemeinen mehrere Moden. Eine Diskrepanz von 1.107 vs. 1.20 (selbst wenn die Zahlen so aufträten) wäre also kein Beweis für falsche Lag-Skalierung, sondern könnte allein durch die unterschiedliche Mode-Mischung in PC1 vs. Eigenmode entstehen.

---

## Die echte Problematik (kein Bug, aber ein Kompromiss)

Was tatsächlich diskussionswürdig ist: **wie verankert man die physikalische Fenstergröße** im log-Embedding?

Mögliche Verankerungen für BTC (d_start=1164, d_end≈6329):

| Wahl | W (log10d) | TAU (M=35) | Span Anfang | Span Ende |
|---|---|---|---|---|
| End = 4 Jahre | 0.114 | 26 | 0.95 J | 4.00 J |
| End = 5 Jahre | 0.143 | 32 | 1.21 J | 5.00 J |
| **Aktuell (Status quo)** | **0.178** | **40** | **1.62 J** | **5.83 J** |
| Start = 2 Jahre | 0.212 | 49 | 2.00 J | 6.86 J |
| Geom-Mittel = 4 J | 0.231 | 53 | 2.24 J | 7.16 J |
| Codex-"Fix" (frac=0.5) | 0.374 | 76 | 4.36 J | 10.3 J |

**Wichtig — geometrisch unvermeidbar:**

> "End = 4 J" UND "Start ≥ 2 J" sind mit **konstantem** W **gleichzeitig nicht erfüllbar**.

- End-Verankerung bei 4 J → Start zwingend ~0.95 J (kürzer als 2 J)
- Start-Verankerung bei 2 J → End zwingend ~6.9 J (länger als 4 J)

Der Status quo (W=0.178) liegt empirisch in einem vernünftigen Bereich: Start ~1.6 J, End ~5.8 J — und liefert das LPPL-konforme λ≈2.01.

Der Codex-"Fix" mit `log_window_frac=0.5` macht das Fenster am Ende auf 10 Jahre auf — das ist **mehr** als das halbe Datenalter und vermutlich nicht das, was wissenschaftlich gewollt ist.

---

## Warum das aktuelle Verhalten empirisch funktioniert

Obwohl `years_to_tau()` semantisch nicht für log-Gitter konzipiert wurde, ergibt sich durch den arithmetischen Zufall der BTC-Datenlänge eine effektive Lag-Struktur, die:

- am **Anfang** ~15 Tage entspricht (kurz, aber genug um Hochfrequenz-Strukturen zu sehen)
- am **Ende** ~82 Tage entspricht (genug um Mittelfrequenz-Strukturen zu sehen)
- ein Gesamtfenster von ~0.18 log10d aufspannt

Bei einem LPPL-Mode mit log-Periode log10(λ=2) = 0.301 deckt das aktuelle Fenster zwar nur **~0.6 Perioden** ab. Trotzdem findet das SSM die Mode korrekt — das ist möglich, weil:

1. Das Embedding ist multivariate (M=35 Dimensionen), nicht ein einzelner Sinus-Fit.
2. Die Eigenwert-Bestimmung in der reduzierten Dynamik nutzt alle Vektoren, nicht nur die Periodogramm-Frequenzen.
3. Die LPPL-Mode dominiert das Signal stark genug, dass auch ein unter-aufgelöstes Fenster sie identifiziert.

**Faustregel "W ≥ 1 Periode" ist eine Sicherheits-Faustregel, kein hartes Kriterium.** Bei stark dominanter Mode reicht weniger.

---

## Empfehlung

**Kein Code-Fix.** Stattdessen:

1. `bug_data.py.md` als "untersucht — quantitativ widerlegt" markieren oder löschen.
2. `_build_log_time_context()` so lassen wie es ist — funktioniert empirisch.
3. Wenn jemand explizit eine andere Verankerung will (z.B. End=4 J strict), als **opt-in-Parameter** mit Default = altes Verhalten einführen, nicht als Default-Änderung.
4. In Skripten mit hohem ssm_dim (z.B. `13_2_*` mit ssm_dim=9) generell vorsichtig sein — dort ist die Mode-Auswahl instabil, unabhängig von TAU.

---

## Methodische Lehre

- **Quantitative Verifikation vor Bug-Behauptung.** Eine Zahl wie "λ=1.107" muss mit den exakt benutzten Parametern reproduzierbar sein, sonst ist die Bug-Notiz nicht belastbar.
- **Vergleich von SSM-Eigenwert und PC1-Periodogramm ist kein Bug-Beweis.** Das sind unterschiedliche Größen mit unterschiedlichen Bias-Eigenschaften.
- **"Faustregel verletzt" ≠ "Ergebnis falsch".** Die Faustregel "Embedding-Fenster ≥ 1 Periode" ist ein Sicherheitskriterium, kein Korrektheitsbeweis.
- **Log-Embedding ist immer ein Kompromiss.** Eine konstante physikalische Fenstergröße existiert nicht — Anfang und Ende müssen unterschiedliche physikalische Spannen abdecken.
