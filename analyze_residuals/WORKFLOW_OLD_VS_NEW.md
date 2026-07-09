| Aspekt | Alt: `harmonic_test_*` | Neu: `analyze_residuals/*` |
|---|---|---|
| Startpunkt | Direkt `phase` oder `slave` laufen lassen | Erst `precheck`, dann gezielter Test |
| Fokus | Ergebnisplots schnell sehen | Vorher klären, ob die gewaehlten Hyperparameter ueberhaupt tragfaehig sind |
| Datenquelle | Direkt im Skript verdrahtet | Zentrale Residuen-Datenquelle in `data.py` |
| Was ist fix / was ist gelernt | Im Output kaum getrennt | Explizit getrennt in `FIXED` vs `LEARNED` |
| Vorpruefung | Verstreut oder implizit | Eigener erster Schritt: `cli_precheck.py` |
| Phase/Slave | Jede Datei mischt Daten, Fit, Statistik, Plot, CLI | Geplant als getrennte Core-/Plot-/CLI-Schichten |
| Wiederverwendung | Gering, viel Copy/Paste | Hoch, gemeinsame Utilities und Konstanten |
| Erweiterbarkeit | `--loc`, neue Segmente, neue Observables teuer | Kernfunktionen so geplant, dass Segmentmasken spaeter sauber passen |
| Vergleichbarkeit | Gefahr stiller Drift zwischen Skripten | Einheitliche Defaults und gemeinsame Hilfsfunktionen |
| Debugging | Schnell fuer Ad-hoc-Arbeit, aber schwer zu isolieren | Fehler lassen sich pro Modul einkreisen |
| Nachteil | Wenig Struktur, grosse Dateien | Etwas mehr Initialaufwand, Alt- und Neu-Welt leben eine Zeit lang parallel |
| Neuer Hauptvorteil | - | Erst Vorpruefung, dann gezielte Analyse statt direkt Plot/Fit |

## Funktionsmatrix: `harmonic_test*` vs `cli_precheck.py`

| Funktion / Aussage | Alt: `harmonic_test_phase.py` | Alt: `harmonic_test_slave.py` | Neu: `cli_precheck.py` | Status im neuen Code |
|---|---|---|---|---|
| BTC-Daten laden und Residuen bauen | Ja | Ja | Ja | Bereits da |
| Delay-Embedding bauen | Ja | Ja | Ja | Bereits da |
| PCA und Varianzspektrum | Ja | Ja | Ja | Bereits da |
| `M`, `years`, `TAU`, `W` explizit als vorgegeben ausweisen | Teilweise | Teilweise | Ja | Verbessert |
| `ssm_dim` und `poly_degree` explizit als nicht gelernt ausweisen | Nein | Nein | Ja | Neu |
| SSM fitten | Ja | Ja | Ja | Bereits da |
| Zahl oszillatorischer Paare berichten | Indirekt | Indirekt | Ja | Neu als Vorpruefung |
| Gelernte Perioden direkt tabellarisch ausgeben | Teilweise | Teilweise | Ja | Neu als Vorpruefung |
| Harmonik-Kandidat `T_sub / (T_main/2)` | Nein | Nein | Ja | Neu |
| Segmentgroessen `H2-H3`, `H3-H4`, `H4+` unter aktuellem Embedding | Nein | Nein | Ja | Neu |
| Phase-Kopplung `Delta phi` testen | Ja | Nein | Nein | Noch nur alt |
| Moden-Phasen plotten | Ja | Nein | Nein | Noch nur alt |
| Amplitudenverhaeltnis `main/sub` plotten | Ja | Nein | Nein | Noch nur alt |
| Halvings in Phase-Plot markieren | Ja | Nein | Nein | Noch nur alt |
| Polarplot der Relativphase | Ja | Nein | Nein | Noch nur alt |
| Slave-Test `PCk aus Master-PCs rekonstruierbar?` | Nein | Ja | Nein | Noch nur alt |
| `R^2` fuer `PC3/PC4/PC5...` | Nein | Ja | Nein | Noch nur alt |
| Compare ueber mehrere `ssm_dim` | Alt vorhanden | Alt vorhanden | Noch nicht | Noch offen |
| Compare ueber mehrere `poly_degree` | Alt vorhanden | Alt vorhanden | Noch nicht | Noch offen |
| Saubere Trennung Daten / Fit / Analyse / Plot / CLI | Nein | Nein | Teilweise | Im Umbau |

## Was der neue Code aktuell **noch nicht** macht

| Fehlt im neuen Code bisher | Wo es aktuell noch lebt |
|---|---|
| eigentliche Phase-Analyse | `harmonic_test_phase.py` |
| eigentliche Slave-Analyse | `harmonic_test_slave.py` |
| Harmonik-Plot mit `Delta phi`, Polarplot, BTC-Achse | `harmonic_test_phase.py` |
| `R^2`-basierter Slave-Plot ueber `PC3..PCk` | `harmonic_test_slave.py` |
| Compare-Modus ueber `ssm_dim`/`poly_degree` | `harmonic_test_phase.py`, `harmonic_test_slave.py` |

## Alt vs aktueller Code

| Altdatei / Workflow | Zweck alt | Aktueller Code in `analyze_residuals` | Status jetzt |
|---|---|---|---|
| `harmonic_test_phase.py` | Globaler Residuen-Phase-Test inkl. Plot und CLI in einer Datei | Noch kein modulares Pendant fertig | Noch legacy |
| `harmonic_test_slave.py` | Globaler Residuen-Slave-Test inkl. Compare-Logik, Plot und CLI in einer Datei | Noch kein modulares Pendant fertig | Noch legacy |
| Implizite Vorpruefung in `harmonic_test_phase.py` | Erst im Fit/Plot sichtbar, ob zwei Moden da sind | `cli_precheck.py` + `precheck.py` | Bereits migriert |
| Verstreute Konstanten (`START_IDX`, `HALVINGS`, Defaults) | In mehreren Skripten kopiert | `constants.py` | Bereits migriert |
| Verstreute Hilfsfunktionen (`identify_modes`, Smoothing) | In jeder Analysedatei erneut definiert | `common.py` | Bereits migriert |
| Direkte Residuen-Pipeline im Skript | Datenaufbau + Embedding direkt im Testskript | `data.py` | Bereits migriert |

## Was der aktuelle Code heute schon kann

- `./analyze_residuals/cli_precheck.py --ssm_dim 4 --poly_degree 1`
- trennt klar:
  - `FIXED / VORGEGEBEN`
  - `LEARNED / AUS DATEN + FIT`
- liefert sofort:
  - Embedding-Budget
  - kumulative PCA-Varianz
  - Zahl oszillatorischer Paare
  - gelernte Perioden
  - Harmonik-Kandidatur
  - Segmentgroessen `H2-H3`, `H3-H4`, `H4+`

## Was noch alt ist

- Die eigentliche Phase-Analyse laeuft noch ueber `harmonic_test_phase.py`.
- Die eigentliche Slave-Analyse laeuft noch ueber `harmonic_test_slave.py`.
- Das ist bewusst so, bis `phase_core/plot/cli` und `slave_core/plot/cli` sauber migriert sind.

## Neuer Fokus

1. Vorprüfung vor jeder tieferen Harmonik-Analyse.
2. Globale Residuen-SSM sauber charakterisieren, bevor Spezialtests folgen.
3. Phase- und Slave-Analyse auf derselben gemeinsamen Daten-/Modenbasis aufbauen.

## Was jetzt anders ist

Frueher:

1. Hyperparameter waehlen.
2. Direkt `harmonic_test_phase.py` oder `harmonic_test_slave.py` starten.
3. Aus dem Plot rueckwaerts schliessen, ob die Wahl gut war.

Neu:

1. Hyperparameter waehlen.
2. `./analyze_residuals/cli_precheck.py` starten.
3. Erst sehen:
   - was fix vorgegeben ist
   - was der Fit wirklich gelernt hat
   - ob `ssm_dim=4` ueberhaupt zwei oszillatorische Paare ergibt
   - ob `T_sub ~ T_main/2` plausibel ist
   - wie viel Datenbudget pro Halving-Segment unter dem aktuellen Embedding bleibt
4. Danach erst `phase` oder `slave`.

## Konkreter Vorteil des ersten Slices

- Ein neuer Run wie `python3 analyze_residuals/cli_precheck.py --ssm_dim 4 --poly_degree 1`
  sagt sofort:
  - Embedding-Budget
  - kumulative PCA-Varianz
  - Zahl oszillatorischer Paare
  - Perioden
  - Harmonik-Kandidatur
  - Segmentgrößen `H2-H3`, `H3-H4`, `H4+`

## Verbleibender Nachteil

- Bis `phase_core/plot` und `slave_core/plot` migriert sind, leben Alt- und Neu-Workflow parallel.
